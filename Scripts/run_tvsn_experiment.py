"""Run a reproducible matched baseline or KPCA-DA classification experiment.

Example after activating .venv:
    python Scripts/run_tvsn_experiment.py --arm baseline --seed 42
    python Scripts/run_tvsn_experiment.py --arm kpcada --seed 42

The ``baseline`` arm uses independently fitted domain KPCA maps but performs
zero target-coordinate optimization; it is a controlled KPCA-map ablation.
The ``raw_deepinsight`` arm fits an ordinary t-SNE DeepInsight map on source
training shared genes only and reuses that frozen map for target-test images,
matching the paper's no-alignment baseline more closely. ``kpcada`` adapts
target coordinates. All arms use patient splits, source-validation classifier
selection, and the same classifier protocol. Binary evaluation is the default;
``--allow-multiclass`` enables tasks such as PAM50.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import timm
import torchvision.transforms as transforms
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAMBDA_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
import sys

sys.path.insert(0, str(ROOT / "Scripts"))

from evaluation_protocol import (balanced_class_weights, classification_metrics,
                                 fit_with_validation, select_binary_threshold)
from experiment_protocol import (GeneAwareMinMax, make_patient_splits,
                                 ordered_common_gene_indices,
                                 patient_group_ids, patient_groups_from_mapping,
                                 transform_target_to_images)
from kpcada import load_expression_dataset
from pyDeepInsightv3.image_transformer import ImageTransformer
from pyDeepInsightv3.image_transformer_new import DomainAdaptiveImageTransformer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def python_source_tree_sha256() -> str:
    """Fingerprint the Python source actually available to this run.

    The Git commit alone is insufficient when the research tree has
    uncommitted edits. Hash relative paths and contents of all Scripts Python
    files, including untracked modules used by the canonical runner.
    """
    digest = hashlib.sha256()
    for path in sorted((ROOT / "Scripts").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def resolve_patient_groups(sample_ids: np.ndarray, mapping_value: str | None,
                           label: str) -> tuple[np.ndarray, dict[str, Any]]:
    if mapping_value is None:
        return patient_group_ids(sample_ids), {
            "source": "built-in sample-ID parser",
            "mapping_file": None,
            "mapping_file_sha256": None,
        }
    mapping_path = Path(mapping_value)
    if not mapping_path.is_absolute():
        mapping_path = ROOT / mapping_path
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or not value.strip()
        for key, value in payload.items()
    ):
        raise ValueError(f"{label} patient map must be a JSON object of sample IDs to patient IDs")
    groups = patient_groups_from_mapping(sample_ids, payload)
    return groups, {
        "source": "explicit sample-to-patient map",
        "mapping_file": str(mapping_path.relative_to(ROOT)
                             if mapping_path.is_relative_to(ROOT) else mapping_path),
        "mapping_file_sha256": sha256(mapping_path),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(images: np.ndarray, labels: np.ndarray | None, preprocess: Any,
                batch_size: int, shuffle: bool = False) -> DataLoader:
    # Float32 is required by the float32 ResNet weights; image inputs are
    # continuous expression values, not integer-valued display pixels.
    x = torch.stack([preprocess(image) for image in images]).float()
    if labels is None:
        dataset = TensorDataset(x)
    else:
        dataset = TensorDataset(x, torch.as_tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def geometry_diagnostics(transformer: DomainAdaptiveImageTransformer) -> dict[str, Any]:
    """Summarize target-coordinate deformation and gene-to-pixel crowding.

    These are descriptive diagnostics, not guarantees of biological or
    predictive preservation. They make the geometry effects of the objective
    and common pixel frame visible beside classifier metrics.
    """
    model = transformer._kpca_da
    initial = np.asarray(model.Z_t_initial_, dtype=float)
    adapted = np.asarray(model.Z_t_, dtype=float)
    if initial.shape != adapted.shape or initial.ndim != 2:
        raise ValueError("KPCA-DA initial and adapted target coordinates are inconsistent")

    initial_centered = initial - initial.mean(axis=0, keepdims=True)
    adapted_centered = adapted - adapted.mean(axis=0, keepdims=True)
    initial_energy = float(np.square(initial_centered).sum())
    if initial_energy <= 1e-12:
        raise ValueError("Initial target embedding is degenerate")
    movement = float(np.square(adapted - initial).sum() / initial_energy)
    variance_before = np.var(initial, axis=0)
    variance_after = np.var(adapted, axis=0)
    variance_ratio = np.divide(
        variance_after, variance_before,
        out=np.full_like(variance_after, np.nan), where=variance_before > 1e-12,
    )

    def pairwise_upper(points: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        return distances[np.triu_indices(points.shape[0], k=1)]

    before_distances = pairwise_upper(initial)
    after_distances = pairwise_upper(adapted)
    nonzero = (before_distances > 1e-12) | (after_distances > 1e-12)
    if nonzero.sum() > 1:
        distance_rank_correlation = float(
            spearmanr(before_distances[nonzero], after_distances[nonzero]).statistic
        )
    else:
        distance_rank_correlation = None

    def covariance_eigenvalues(points: np.ndarray) -> list[float]:
        covariance = np.atleast_2d(np.cov(points, rowvar=False))
        return np.linalg.eigvalsh(covariance).clip(min=0).tolist()

    def collision_summary(coords: np.ndarray) -> dict[str, int | float]:
        unique = np.unique(coords, axis=0).shape[0]
        count = int(coords.shape[0])
        return {
            "genes": count,
            "occupied_pixels": int(unique),
            "genes_in_collisions": int(count - unique),
            "collision_fraction": float((count - unique) / count) if count else 0.0,
        }

    source_coords = np.asarray(transformer._coords_source)
    target_coords = np.asarray(transformer._coords_target)
    common_source = np.asarray(model.common_source_idx_)
    common_target = np.asarray(model.common_target_idx_)
    shared_pixel_displacements = np.linalg.norm(
        source_coords[common_source] - target_coords[common_target], axis=1
    )
    shared_same_pixel = float(np.mean(shared_pixel_displacements == 0))
    return {
        "target_normalized_squared_movement": movement,
        "target_variance_ratio_by_component": variance_ratio.tolist(),
        "target_covariance_eigenvalues_initial": covariance_eigenvalues(initial),
        "target_covariance_eigenvalues_adapted": covariance_eigenvalues(adapted),
        "target_pairwise_distance_spearman": distance_rank_correlation,
        "source_gene_pixel_collisions": collision_summary(source_coords),
        "target_gene_pixel_collisions": collision_summary(target_coords),
        "shared_gene_same_pixel_fraction": shared_same_pixel,
        "shared_gene_pixel_displacement": {
            "mean": float(shared_pixel_displacements.mean()),
            "median": float(np.median(shared_pixel_displacements)),
            "p95": float(np.percentile(shared_pixel_displacements, 95)),
            "maximum": float(shared_pixel_displacements.max()),
        },
    }


def scaling_range_summary(values: np.ndarray) -> dict[str, float]:
    """Describe unit-range excursions without using labels or fitting data."""
    matrix = np.asarray(values, dtype=float)
    if matrix.size == 0 or not np.isfinite(matrix).all():
        raise ValueError("Scaling diagnostics require a finite, non-empty matrix")
    below_zero = matrix < 0.0
    above_one = matrix > 1.0
    return {
        "minimum": float(matrix.min()),
        "maximum": float(matrix.max()),
        "below_zero_fraction": float(below_zero.mean()),
        "above_one_fraction": float(above_one.mean()),
        "outside_unit_interval_fraction": float((below_zero | above_one).mean()),
    }


def classifier_input_diagnostics(images: np.ndarray) -> dict[str, Any]:
    """Summarize float RGB values before and after ResNet normalization.

    ``ToTensor`` does not divide floating-point NumPy arrays by 255, so the
    RGB values emitted by the expression mapper pass through unchanged before
    ImageNet normalization. Record the actual scale seen by the classifier.
    """
    array = np.asarray(images, dtype=float)
    if array.ndim != 4 or array.shape[-1] != 3 or array.size == 0:
        raise ValueError("images must be a non-empty batch of HxWx3 float arrays")
    mean = np.asarray([0.485, 0.456, 0.406], dtype=float).reshape(1, 1, 1, 3)
    std = np.asarray([0.229, 0.224, 0.225], dtype=float).reshape(1, 1, 1, 3)
    low, high = np.inf, -np.inf
    beyond_ten = 0
    count = 0
    quantile_samples = []
    # Bound temporary memory: full 227x227 RGB cohorts already occupy several
    # hundred MB, so normalize a few patients at a time for diagnostics.
    for start in range(0, len(array), 16):
        normalized = (array[start:start + 16] - mean) / std
        low = min(low, float(normalized.min()))
        high = max(high, float(normalized.max()))
        beyond_ten += int(np.count_nonzero(np.abs(normalized) > 10.0))
        count += int(normalized.size)
        quantile_samples.append(normalized.reshape(-1)[::53].copy())
    sample = np.concatenate(quantile_samples)
    return {
        "float_rgb_pixels": scaling_range_summary(array),
        "imagenet_normalized_pixels": {
            "minimum": low,
            "maximum": high,
            "absolute_p99_sampled": float(np.quantile(np.abs(sample), 0.99)),
            "absolute_p999_sampled": float(np.quantile(np.abs(sample), 0.999)),
            "absolute_gt_10_fraction": float(beyond_ten / count),
        },
        "normalized_quantile_sampling_stride": 53,
        "to_tensor_float_scaling": "unchanged; no division by 255",
    }


def run(args: argparse.Namespace) -> Path | None:
    set_seed(args.seed)
    source_tree_hash = python_source_tree_sha256()
    environment_lock_path = ROOT / "audit" / "environment.lock.txt"
    environment_lock_hash = (
        sha256(environment_lock_path) if environment_lock_path.is_file() else None
    )
    source_path = Path(args.source_path)
    target_path = Path(args.target_path)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    if not target_path.is_absolute():
        target_path = ROOT / target_path
    Xs_gxp, genes_s, ids_s, y_s = load_expression_dataset(str(source_path))
    target_labels_needed = not (
        args.withhold_target_test_labels or args.preflight_only or args.split_only
    )
    Xt_gxp, genes_t, ids_t, y_t = load_expression_dataset(
        str(target_path), include_labels=target_labels_needed
    )
    ids_s = np.asarray(ids_s, dtype=str)
    ids_t = np.asarray(ids_t, dtype=str)
    original_source_sample_count = int(len(y_s))
    excluded_source_mask = np.isin(y_s, np.asarray(args.exclude_source_labels))
    excluded_source_sample_ids = ids_s[excluded_source_mask].tolist()
    if excluded_source_mask.any():
        keep_source = ~excluded_source_mask
        Xs_gxp = Xs_gxp[:, keep_source]
        ids_s = ids_s[keep_source]
        y_s = y_s[keep_source]
    source_classes = np.unique(y_s)
    if source_classes.size < 2 or (source_classes.size != 2 and not args.allow_multiclass):
        raise ValueError(
            "This runner requires two source classes unless --allow-multiclass is set; "
            f"found {source_classes.tolist()}. Use --exclude-source-labels for "
            "non-binary source rows and record their IDs."
        )
    source_patient_groups, source_patient_mapping = resolve_patient_groups(
        ids_s, getattr(args, "source_patient_map", None), "source"
    )
    target_patient_groups, target_patient_mapping = resolve_patient_groups(
        ids_t, getattr(args, "target_patient_map", None), "target"
    )
    Xs, Xt = Xs_gxp.T, Xt_gxp.T

    splits = make_patient_splits(
        y_s,
        source_ids=source_patient_groups,
        target_ids=target_patient_groups,
        source_validation_size=args.source_validation_size,
        target_test_size=args.target_test_size,
        random_state=args.seed,
    )
    if (set(source_patient_groups[splits.source_train].tolist())
            & set(source_patient_groups[splits.source_validation].tolist())):
        raise RuntimeError("Source patient groups crossed train/validation")
    if (set(target_patient_groups[splits.target_adapt].tolist())
            & set(target_patient_groups[splits.target_test].tolist())):
        raise RuntimeError("Target patient groups crossed adaptation/test")
    if args.split_only:
        split_report = {
            "result": "patient split validation passed",
            "task_name": args.task_name,
            "python_source_tree_sha256": source_tree_hash,
            "environment_lock_sha256": environment_lock_hash,
            "source_sample_count": int(len(y_s)),
            "target_sample_count": int(len(ids_t)),
            "source_patient_count": int(np.unique(source_patient_groups).size),
            "target_patient_count": int(np.unique(target_patient_groups).size),
            "source_patient_mapping": source_patient_mapping,
            "target_patient_mapping": target_patient_mapping,
            "source_classes": {
                str(label): int(count)
                for label, count in zip(*np.unique(y_s, return_counts=True))
            },
            "source_train_n": int(len(splits.source_train)),
            "source_validation_n": int(len(splits.source_validation)),
            "target_adaptation_n": int(len(splits.target_adapt)),
            "target_test_n": int(len(splits.target_test)),
            "source_patient_overlap": 0,
            "target_patient_overlap": 0,
            "target_labels_used_for_split": False,
            "model_fitting": False,
        }
        print(json.dumps(split_report, indent=2))
        return None
    Xs_train, ys_train = Xs[splits.source_train], y_s[splits.source_train]
    Xs_val, ys_val = Xs[splits.source_validation], y_s[splits.source_validation]
    Xt_adapt = Xt[splits.target_adapt]
    Xt_test_raw = Xt[splits.target_test]

    raw_deepinsight = args.arm == "raw_deepinsight"
    representation_fit_source_ids = ids_s[splits.source_train]
    representation_fit_target_ids = (
        np.array([], dtype=str) if raw_deepinsight
        else ids_t[splits.target_adapt]
    )
    if raw_deepinsight and args.feature_scope != "shared":
        raise ValueError("The raw DeepInsight baseline requires --feature-scope shared")
    preprocessing = GeneAwareMinMax(
        args.variance_percentile,
        clip_scaled_values=args.clip_scaled_values,
    )
    if raw_deepinsight:
        preprocessing.fit_source_shared(Xs_train, genes_s, genes_t)
    else:
        preprocessing.fit(Xs_train, genes_s, Xt_adapt, genes_t)
    Xs_train = preprocessing.transform_source(Xs_train, genes_s)
    Xs_val = preprocessing.transform_source(Xs_val, genes_s)
    genes_source_fit = preprocessing.source_genes_
    genes_target_fit = preprocessing.target_genes_
    if raw_deepinsight:
        source_idx, target_idx, shared_genes = ordered_common_gene_indices(
            genes_source_fit, genes_target_fit
        )
        Xs_train = Xs_train[:, source_idx]
        Xs_val = Xs_val[:, source_idx]
        genes_source_fit = shared_genes
        genes_target_fit = shared_genes.copy()
        # This baseline is strictly inductive: it does not fit on the target
        # adaptation values, even for unsupervised preprocessing.
        Xt_adapt = np.empty((0, len(shared_genes)), dtype=float)
    else:
        Xt_adapt = preprocessing.transform_target(Xt_adapt, genes_t)
        if args.feature_scope == "shared":
            source_idx, target_idx, shared_genes = ordered_common_gene_indices(
                genes_source_fit, genes_target_fit
            )
            Xs_train = Xs_train[:, source_idx]
            Xs_val = Xs_val[:, source_idx]
            Xt_adapt = Xt_adapt[:, target_idx]
            genes_source_fit = shared_genes
            genes_target_fit = shared_genes.copy()

    encoder = LabelEncoder().fit(ys_train)
    ys_train_enc = encoder.transform(ys_train)
    ys_val_enc = encoder.transform(ys_val)
    epochs = 0 if args.arm == "baseline" or raw_deepinsight else args.kpca_epochs
    lambda_grid = tuple(args.lambda_grid or DEFAULT_LAMBDA_GRID)
    if not lambda_grid or any(not np.isfinite(value) or value < 0 for value in lambda_grid):
        raise ValueError("Lambda candidates must be finite and nonnegative")
    configured_lambda = (
        args.lambda_reg if args.lambda_reg is not None
        else lambda_grid[min(1, len(lambda_grid) - 1)]
    )
    lambda_selection = None
    if raw_deepinsight:
        transformer = ImageTransformer(
            feature_extractor=TSNE(n_components=2, metric="cosine",
                                   random_state=args.seed),
            pixels=(args.image_size, args.image_size),
        )
        transformer.fit(Xs_train, plot=False)
        source_images = transformer.transform(Xs_train)
        validation_images = transformer.transform(Xs_val)
        target_shared = preprocessing.transform_target(Xt_test_raw, genes_t)
        target_shared = target_shared[:, target_idx]
        test_images = transformer.transform(target_shared)
        coords = transformer.coords()
        occupied = int(np.unique(coords, axis=0).shape[0])
        geometry = {
            "mapping_method": "source-training-fitted raw DeepInsight with t-SNE",
            "map_fit_samples": int(len(Xs_train)),
            "mapped_shared_genes": int(len(coords)),
            "occupied_pixels": occupied,
            "gene_pixel_collision_fraction": float((len(coords) - occupied) / len(coords)),
            "shared_gene_source_target_same_pixel_fraction": 1.0,
            "target_expression_used_to_fit_map": False,
            "target_adaptation_expression_used": False,
        }
        scaling_diagnostics = {
            "source_training": scaling_range_summary(Xs_train),
            "source_validation": scaling_range_summary(Xs_val),
            "target_adaptation": None,
            "target_test_shared_genes": scaling_range_summary(target_shared),
        }
    else:
        transformer = DomainAdaptiveImageTransformer(
            kpca_da_params={
                "n_components": 2,
                "kernel": "rbf",
                "kpca_gamma": args.kpca_gamma,
                "auto_gamma": True,
                "lambda_reg": configured_lambda,
                "alignment_weight": args.alignment_weight,
                "lr": args.kpca_lr,
                "epochs": epochs,
                "regularization_mode": args.regularization_mode,
                "kpca_random_state": args.seed,
                "align_shared_genes": args.align_shared_genes,
                "verbose": args.verbose_kpca,
            },
            pixels=(args.image_size, args.image_size),
        )
        if args.arm == "kpcada" and args.lambda_reg is None and not args.preflight_only:
            lambda_selection = transformer._kpca_da.tune_lambda(
                Xs_train.T, Xt_adapt.T,
                genes_source_fit, genes_target_fit,
                lambda_values=lambda_grid,
            )
        transformer.fit_transform(
            Xs_train, Xt_adapt,
            genes_source_fit, genes_target_fit,
            plot=False,
        )
        if args.anchor_shared_pixels:
            transformer.anchor_shared_gene_pixels()
        source_images = transformer.transform_source(Xs_train)
        validation_images = transformer.transform_source(Xs_val)
        test_images = transform_target_to_images(
            preprocessing, transformer, Xt_test_raw, genes_t,
            feature_scope=args.feature_scope,
        )
        geometry = geometry_diagnostics(transformer)
        shared_target_columns = np.isin(genes_target_fit, preprocessing.shared_genes_)
        scaling_diagnostics = {
            "source_training": scaling_range_summary(Xs_train),
            "source_validation": scaling_range_summary(Xs_val),
            "target_adaptation_all_retained": scaling_range_summary(Xt_adapt),
            "target_adaptation_shared_genes": scaling_range_summary(
                Xt_adapt[:, shared_target_columns]
            ),
        }

    if args.preflight_only:
        print(json.dumps({
            "result": "preflight passed",
            "arm": args.arm,
            "python_source_tree_sha256": source_tree_hash,
            "environment_lock_sha256": environment_lock_hash,
            "kpca_epochs_run": epochs,
            "source_train_images": list(source_images.shape),
            "source_validation_images": list(validation_images.shape),
            "target_test_images": list(test_images.shape),
            "source_genes_retained": int(len(preprocessing.source_genes_)),
            "target_genes_retained": int(len(preprocessing.target_genes_)),
            "source_features_used": int(len(genes_source_fit)),
            "target_features_used": int(len(genes_target_fit)),
            "feature_scope": args.feature_scope,
            "kpca_alignment": (
                transformer._kpca_da.kpca_alignment_ if not raw_deepinsight else None
            ),
            "shared_gene_pixel_anchoring": (
                transformer._shared_gene_pixel_anchor_ if not raw_deepinsight else None
            ),
            "geometry_diagnostics": geometry,
            "scaling_diagnostics": scaling_diagnostics,
            "classifier_input_diagnostics": {
                "source_training": classifier_input_diagnostics(source_images),
                "source_validation": classifier_input_diagnostics(validation_images),
                "target_test": classifier_input_diagnostics(test_images),
            },
            "clip_scaled_values": bool(args.clip_scaled_values),
            "note": "No classifier was constructed or trained; target-test labels were not accessed.",
        }, indent=2))
        return None

    device = choose_device()
    model = timm.create_model(
        args.model, pretrained=not args.no_pretrained,
        num_classes=len(encoder.classes_),
    ).to(device)
    preprocess_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_loader = make_loader(
        source_images, ys_train_enc, preprocess_image, args.batch_size, shuffle=True
    )
    validation_loader = make_loader(
        validation_images, ys_val_enc, preprocess_image, args.batch_size
    )
    test_loader = make_loader(test_images, None, preprocess_image, args.batch_size)

    loss_weights = None
    if args.class_weighted_loss:
        loss_weights = balanced_class_weights(
            ys_train_enc, np.arange(len(encoder.classes_))
        )
    criterion = torch.nn.CrossEntropyLoss(
        weight=(torch.as_tensor(loss_weights, dtype=torch.float32, device=device)
                if loss_weights is not None else None)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.scheduler_patience
    )
    training = fit_with_validation(
        model, train_loader, validation_loader, criterion, optimizer,
        scheduler=scheduler, max_epochs=args.classifier_epochs,
        patience=args.early_stopping_patience,
        monitor="balanced_accuracy",
    )

    positive_label = (
        int(encoder.transform(["Tumor"])[0])
        if len(encoder.classes_) == 2 and "Tumor" in encoder.classes_
        else None
    )
    negative_label = (
        next(index for index in range(len(encoder.classes_)) if index != positive_label)
        if positive_label is not None else None
    )
    decision_threshold = None
    if args.source_validation_threshold and positive_label is not None:
        model.eval()
        validation_truth: list[int] = []
        validation_positive_probability: list[float] = []
        with torch.no_grad():
            for inputs, labels in validation_loader:
                logits = model(inputs.to(device))
                validation_truth.extend(labels.tolist())
                validation_positive_probability.extend(
                    logits.softmax(dim=1)[:, positive_label].cpu().tolist()
                )
        decision_threshold = select_binary_threshold(
            validation_truth, validation_positive_probability,
            positive_label=positive_label, negative_label=negative_label,
        )

    # The test loader carries images only. In ordinary final-evaluation mode,
    # labels enter only after checkpoint selection; development runs can
    # withhold them entirely with --withhold-target-test-labels.
    model.eval()
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for (inputs,) in test_loader:
            logits = model(inputs.to(device))
            probabilities.extend(logits.softmax(dim=1).cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if decision_threshold is not None:
        positive_probability = np.asarray(probabilities)[:, positive_label]
        predictions = np.where(
            positive_probability >= decision_threshold["threshold"],
            positive_label, negative_label,
        ).astype(int).tolist()
    target_test_labels = None
    metrics = None
    if not args.withhold_target_test_labels:
        target_test_labels = encoder.transform(y_t[splits.target_test])
        metrics = classification_metrics(
            target_test_labels,
            predictions,
            labels=np.arange(len(encoder.classes_)),
            positive_label=positive_label,
        )

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    task_name = args.task_name.strip()
    if not task_name or any(char in task_name for char in "/\\") or task_name in {".", ".."}:
        raise ValueError("--task-name must be one non-empty path component")
    run_dir = results_root / task_name / f"{args.arm}_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "arm": args.arm,
        "task_name": task_name,
        "seed": args.seed,
        "git_commit": commit,
        "python_source_tree_sha256": source_tree_hash,
        "environment_lock_sha256": environment_lock_hash,
        "timestamp_utc": timestamp,
        "device": str(device),
        "source_file_sha256": sha256(source_path),
        "target_file_sha256": sha256(target_path),
        "source_file": str(source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT) else source_path),
        "target_file": str(target_path.relative_to(ROOT) if target_path.is_relative_to(ROOT) else target_path),
        "source_sample_count_before_exclusions": original_source_sample_count,
        "source_excluded_sample_count": len(excluded_source_sample_ids),
        "source_excluded_sample_ids": excluded_source_sample_ids,
        "source_sample_count": int(len(y_s)),
        "target_sample_count": int(len(ids_t)),
        "source_patient_count": int(len(np.unique(source_patient_groups))),
        "target_patient_count": int(len(np.unique(target_patient_groups))),
        "patient_mapping": {
            "source": source_patient_mapping,
            "target": target_patient_mapping,
        },
        "source_train_n": int(len(splits.source_train)),
        "source_validation_n": int(len(splits.source_validation)),
        "target_adaptation_n": int(len(splits.target_adapt)),
        "target_test_n": int(len(splits.target_test)),
        "representation_fit_policy": {
            "mapping": (
                "source-training-only raw DeepInsight map"
                if raw_deepinsight else "source-training plus target-adaptation KPCA map"
            ),
            "target_adaptation_expression_used": not raw_deepinsight,
            "target_test_expression_used_for_fit": False,
        },
        "preprocessing_policy": {
            "variance_filter": "stable rank, discard floor(percentile * feature_count); remove constants",
            "variance_percentile": float(args.variance_percentile),
            "variance_tie_break": "stable input gene order",
            "source_filter_fit_scope": "source training samples only",
            "target_private_filter_fit_scope": (
                "target adaptation samples only" if not raw_deepinsight else "not fitted"
            ),
            "shared_gene_scaler_fit_scope": "source training samples only",
            "target_private_scaler_fit_scope": (
                "target adaptation samples only" if not raw_deepinsight else "not fitted"
            ),
            "clip_scaled_values": bool(args.clip_scaled_values),
        },
        "source_genes_retained": int(len(preprocessing.source_genes_)),
        "target_genes_retained": int(len(preprocessing.target_genes_)),
        "scaling_diagnostics": scaling_diagnostics,
        "classifier_input_diagnostics": {
            "source_training": classifier_input_diagnostics(source_images),
            "source_validation": classifier_input_diagnostics(validation_images),
            "target_test": classifier_input_diagnostics(test_images),
        },
        "source_features_used": int(len(genes_source_fit)),
        "target_features_used": int(len(genes_target_fit)),
        "feature_scope": args.feature_scope,
        "shared_genes_retained": int(
            len(transformer._kpca_da.common_source_idx_)
            if not raw_deepinsight else len(genes_source_fit)
        ),
        "effective_mmd_gamma": (
            float(transformer._kpca_da.mmd_gamma_) if not raw_deepinsight else None
        ),
        "final_mmd": (
            float(transformer._kpca_da.final_mmd_) if not raw_deepinsight else None
        ),
        "final_regularization": (
            float(transformer._kpca_da.final_regularization_)
            if not raw_deepinsight else None
        ),
        "final_objective": (
            float(transformer._kpca_da.final_loss_) if not raw_deepinsight else None
        ),
        "regularization_mode": (
            transformer._kpca_da.regularization_mode_ if not raw_deepinsight else None
        ),
        "kpca_alignment": (
            transformer._kpca_da.kpca_alignment_ if not raw_deepinsight else None
        ),
        "shared_gene_pixel_anchoring": (
            transformer._shared_gene_pixel_anchor_ if not raw_deepinsight else None
        ),
        "selected_lambda_reg": (
            float(transformer._kpca_da.lambda_reg_) if not raw_deepinsight else None
        ),
        "alignment_weight": (
            float(transformer._kpca_da.alignment_weight_)
            if not raw_deepinsight else None
        ),
        "lambda_selection": (
            {
                "rule": "lowest final full empirical MMD squared; target labels unused",
                "candidates": lambda_selection["results"],
                "selected_lambda": float(lambda_selection["best_lambda_mmd"]),
            }
            if lambda_selection is not None else None
        ),
        "label_classes": encoder.classes_.tolist(),
        "kpca_epochs": epochs,
        "best_source_validation_epoch": training["best_epoch"],
        "best_source_validation_balanced_accuracy": training["best_validation_score"],
        "decision_threshold": (
            {"selection": "source-validation balanced accuracy",
             "positive_class": str(encoder.classes_[positive_label]),
             **decision_threshold}
            if decision_threshold is not None else {"selection": "argmax"}
        ),
        "class_loss_weights": (
            {str(label): float(weight)
             for label, weight in zip(encoder.classes_, loss_weights)}
            if loss_weights is not None else None
        ),
        "source_validation_history": training["history"],
        "geometry_diagnostics": geometry,
        "target_test_labels_scored": target_test_labels is not None,
        "config": vars(args),
        "test_metrics": metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    prediction_fields = {
        "sample_ids": ids_t[splits.target_test],
        "patient_ids": target_patient_groups[splits.target_test].astype(str),
        "y_pred": np.asarray(predictions),
        "probabilities": np.asarray(probabilities),
    }
    if target_test_labels is not None:
        prediction_fields["y_true"] = target_test_labels
    np.savez_compressed(run_dir / "predictions.npz", **prediction_fields)
    np.savez_compressed(
        run_dir / "split_ids.npz",
        source_train=ids_s[splits.source_train],
        source_validation=ids_s[splits.source_validation],
        target_adaptation=ids_t[splits.target_adapt],
        target_test=ids_t[splits.target_test],
        source_train_patient=source_patient_groups[splits.source_train].astype(str),
        source_validation_patient=source_patient_groups[splits.source_validation].astype(str),
        target_adaptation_patient=target_patient_groups[splits.target_adapt].astype(str),
        target_test_patient=target_patient_groups[splits.target_test].astype(str),
        feature_map_source_train=representation_fit_source_ids.astype(str),
        feature_map_target_adaptation=representation_fit_target_ids.astype(str),
    )
    torch.save(training["model"].state_dict(), run_dir / "best_model.pt")
    with (run_dir / "fitted_preprocessing_and_map.pkl").open("wb") as handle:
        pickle.dump({
            "preprocessor": preprocessing,
            "transformer": transformer,
            "label_encoder": encoder,
            "feature_scope": args.feature_scope,
            "source_gene_ids_for_map": np.asarray(genes_source_fit, dtype=object),
            "target_gene_ids_for_map": np.asarray(genes_target_fit, dtype=object),
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(metadata, indent=2))
    print(f"Saved run artifacts to {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "kpcada", "raw_deepinsight"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-name", default="tvsn",
                        help="artifact subdirectory label (for example: pam50_smc_to_cptac)")
    parser.add_argument("--results-root", default="results",
                        help="root directory for run artifacts, relative to repository root by default")
    parser.add_argument("--allow-multiclass", action="store_true",
                        help="allow two or more source classes; default is binary-only")
    parser.add_argument("--split-only", action="store_true",
                        help="validate target patient splitting and source class coverage without fitting preprocessing or models")
    parser.add_argument("--source-path", default=str(ROOT / "Data" / "TvsN" / "TCGA_BRCA_NvT_new.csv"),
                        help="source expression CSV; relative paths resolve from repository root")
    parser.add_argument("--target-path", default=str(ROOT / "Data" / "TvsN" / "GSE45498_TvsN_official_200.csv"),
                        help="target expression CSV; relative paths resolve from repository root")
    parser.add_argument("--source-patient-map", default=None,
                        help="optional JSON object mapping source sample IDs to patient IDs")
    parser.add_argument("--target-patient-map", default=None,
                        help="optional JSON object mapping target sample IDs to patient IDs")
    parser.add_argument("--exclude-source-labels", nargs="*", default=["Unknown"],
                        help="source labels to omit before splitting (for example: Unknown); target labels are never used here")
    parser.add_argument("--source-validation-size", type=float, default=0.2)
    parser.add_argument("--target-test-size", type=float, default=0.3)
    parser.add_argument("--variance-percentile", type=float, default=30.0,
                        help="discard this source-training variance percentile (supplement specifies 30)")
    parser.add_argument("--clip-scaled-values", action="store_true",
                        help="clip transformed values to [0, 1] after fitting scalers on permitted training/adaptation data")
    parser.add_argument("--feature-scope", choices=("all", "shared"), default="all",
                        help="all retained features (paper-like) or shared genes only (cohort-coverage ablation)")
    parser.add_argument("--kpca-epochs", type=int, default=500)
    parser.add_argument("--kpca-gamma", type=float, default=0.2)
    parser.add_argument("--lambda-reg", type=float, default=None,
                        help="fixed lambda; omit for label-free candidate selection")
    parser.add_argument("--alignment-weight", type=float, default=1.0,
                        help="weight on empirical MMD squared (1 for full KPCA-DA; 0 for VML-only ablation)")
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=None,
                        help="optional candidate grid for label-free tuning; defaults to the six paper values")
    parser.add_argument("--kpca-lr", type=float, default=0.001)
    parser.add_argument("--regularization-mode", choices=("trace", "reference", "none"),
                        default="trace",
                        help="trace reproduces the manuscript objective; reference is the audited method revision")
    parser.add_argument("--align-shared-genes", action="store_true",
                        help="apply label-free orthogonal Procrustes to initial KPCA coordinates using shared gene IDs (method ablation)")
    parser.add_argument("--anchor-shared-pixels", action="store_true",
                        help="map shared target genes to their source pixels while retaining private-feature coordinates")
    parser.add_argument("--image-size", type=int, default=227)
    parser.add_argument("--model", default="resnet50d")
    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--source-validation-threshold", action="store_true",
                        help="optimize the binary decision threshold on source validation only (development ablation)")
    parser.add_argument("--withhold-target-test-labels", action="store_true",
                        help="save predictions without materializing or scoring target-test labels")
    parser.add_argument("--class-weighted-loss", action="store_true",
                        help="weight source-training classes inversely by count; method ablation selected with the same source-validation rule")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--verbose-kpca", action="store_true")
    parser.add_argument("--preflight-only", action="store_true",
                        help="fit the coordinate mapper and build images, but do not train a classifier")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
