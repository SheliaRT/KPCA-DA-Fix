"""Shared classification metrics and safe checkpoint capture."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Hashable, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score)


def clone_state_dict(model: Any) -> dict[str, Any]:
    """Copy tensor storage so later optimizer steps cannot alter a checkpoint."""
    return deepcopy(model.state_dict())


def classification_metrics(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    *,
    labels: Optional[Sequence[Hashable]] = None,
    positive_label: Optional[Hashable] = None,
) -> dict[str, Any]:
    """Return consistently defined accuracy and class-wise metrics.

    Binary sensitivity and specificity are reported only when callers name the
    positive class. Macro metrics include every label in `labels`, including a
    class with no predicted examples.
    """
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.ndim != 1 or pred.ndim != 1 or truth.size != pred.size or not truth.size:
        raise ValueError("y_true and y_pred must be non-empty 1-D arrays of equal length")
    class_labels = np.asarray(labels if labels is not None else np.unique(truth))
    if class_labels.ndim != 1 or not class_labels.size:
        raise ValueError("labels must be a non-empty 1-D sequence")
    if not set(np.unique(truth)).issubset(set(class_labels)):
        raise ValueError("labels must include every observed true class")
    if not set(np.unique(pred)).issubset(set(class_labels)):
        raise ValueError("labels must include every predicted class")

    cm = confusion_matrix(truth, pred, labels=class_labels)
    per_class_recall = recall_score(truth, pred, labels=class_labels,
                                    average=None, zero_division=0)
    true_negatives = cm.sum() - cm.sum(axis=1) - cm.sum(axis=0) + np.diag(cm)
    false_positives = cm.sum(axis=0) - np.diag(cm)
    per_class_specificity = np.divide(
        true_negatives,
        true_negatives + false_positives,
        out=np.zeros_like(true_negatives, dtype=float),
        where=(true_negatives + false_positives) != 0,
    )
    result: dict[str, Any] = {
        "labels": class_labels.tolist(),
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=class_labels,
                                   average="macro", zero_division=0)),
        "sensitivity_by_class": {
            str(label): float(value)
            for label, value in zip(class_labels.tolist(), per_class_recall)
        },
        "specificity_by_class": {
            str(label): float(value)
            for label, value in zip(class_labels.tolist(), per_class_specificity)
        },
    }
    if positive_label is not None:
        if class_labels.size != 2 or positive_label not in class_labels:
            raise ValueError("positive_label requires a binary label set containing that class")
        negative_label = next(label for label in class_labels.tolist()
                              if label != positive_label)
        result["positive_label"] = positive_label
        result["sensitivity"] = float(recall_score(
            truth, pred, pos_label=positive_label, average="binary", zero_division=0))
        result["specificity"] = float(recall_score(
            truth, pred, pos_label=negative_label, average="binary", zero_division=0))
    return result


def balanced_class_weights(y_train: Sequence[Hashable],
                           labels: Sequence[Hashable]) -> np.ndarray:
    """Return inverse-frequency weights using only the supplied training labels.

    The weights follow sklearn's ``balanced`` convention: n_samples /
    (n_classes * n_samples_for_class). All requested classes must be present.
    """
    truth = np.asarray(y_train)
    classes = np.asarray(labels)
    if truth.ndim != 1 or truth.size == 0 or classes.ndim != 1 or classes.size == 0:
        raise ValueError("Training labels and classes must be non-empty vectors")
    if np.unique(classes).size != classes.size:
        raise ValueError("Class labels must be unique")
    counts = np.asarray([np.count_nonzero(truth == label) for label in classes],
                        dtype=float)
    if np.any(counts == 0) or not set(np.unique(truth)).issubset(set(classes)):
        raise ValueError("Every training label class must be listed and represented")
    return truth.size / (classes.size * counts)


def select_binary_threshold(
    y_true: Sequence[Hashable],
    positive_probabilities: Sequence[float],
    *,
    positive_label: Hashable,
    negative_label: Hashable,
) -> dict[str, float]:
    """Select a binary threshold by balanced accuracy on source validation.

    This function must only receive development/validation labels. It is
    intended to reduce majority-class collapse while keeping target labels out
    of classifier and threshold selection. Ties choose the threshold closest
    to 0.5, then the smaller threshold for deterministic behavior.
    """
    truth = np.asarray(y_true)
    probabilities = np.asarray(positive_probabilities, dtype=float)
    if (truth.ndim != 1 or probabilities.ndim != 1 or truth.size == 0
            or truth.size != probabilities.size):
        raise ValueError("Validation labels and probabilities must be non-empty equal-length vectors")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Positive-class probabilities must be finite and within [0, 1]")
    if set(np.unique(truth).tolist()) != {negative_label, positive_label}:
        raise ValueError("Threshold selection requires both binary classes in validation data")

    unique = np.unique(probabilities)
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    candidates = np.unique(np.concatenate(([0.0], unique, midpoints, [1.0])))
    scored = []
    for threshold in candidates:
        predicted = np.where(probabilities >= threshold, positive_label, negative_label)
        score = balanced_accuracy_score(truth, predicted)
        scored.append((float(score), float(threshold)))
    best_score = max(score for score, _ in scored)
    best_threshold = min(
        (threshold for score, threshold in scored if np.isclose(score, best_score)),
        key=lambda threshold: (abs(threshold - 0.5), threshold),
    )
    return {"threshold": float(best_threshold),
            "source_validation_balanced_accuracy": float(best_score)}


def fit_with_validation(
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    criterion: Any,
    optimizer: Any,
    *,
    scheduler: Optional[Any] = None,
    max_epochs: int = 100,
    patience: int = 10,
    monitor: str = "balanced_accuracy",
) -> dict[str, Any]:
    """Train with source validation only; this API accepts no target-test data.

    `train_loader` should contain source training patients. The validation
    loader should contain source validation patients and must use deterministic
    preprocessing. The best model weights are copied and restored before
    returning.
    """
    import torch

    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    if monitor not in {"accuracy", "balanced_accuracy", "macro_f1"}:
        raise ValueError("monitor must be accuracy, balanced_accuracy, or macro_f1")

    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")

    best_score = -np.inf
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().item())
            train_batches += 1
        if not train_batches:
            raise ValueError("train_loader produced no batches")

        model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)
                y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
                y_true.extend(labels.detach().cpu().tolist())
        if not y_true:
            raise ValueError("validation_loader produced no samples")

        metrics = classification_metrics(
            y_true, y_pred,
            labels=np.unique(np.asarray(y_true + y_pred)),
        )
        val_score = metrics[monitor]
        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_score)
            else:
                scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_batches,
            **metrics,
        }
        history.append(record)
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = clone_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("No source-validation checkpoint was produced")
    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "best_epoch": best_epoch,
            "best_validation_score": float(best_score), "history": history}
