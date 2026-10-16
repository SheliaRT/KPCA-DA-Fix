"""Generate post-hoc target Grad-CAM gene rankings from one saved model run.

This utility only analyzes a finished checkpoint. It uses the exact saved target
map and retained gene order, computes raw Grad-CAM maps, averages within each
held-out target class, then normalizes the mean map before the main-text top-10
ranking. Target labels are used for this post-hoc grouping only; they do not
change training, checkpoint, or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from experiment_protocol import ordered_common_gene_indices, patient_group_ids
from kpcada import load_expression_dataset
from pyDeepInsightv3.gene_attribution import rank_gene_attributions


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    metadata = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    target_path = resolve(metadata["target_file"])
    if digest(target_path) != metadata["target_file_sha256"]:
        raise ValueError("Target file hash differs from saved run metadata")
    with np.load(run_dir / "split_ids.npz", allow_pickle=False) as split:
        test_ids = split["target_test"].astype(str)
        patient_ids = split["target_test_patient"].astype(str)
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as predictions:
        if not np.array_equal(predictions["sample_ids"].astype(str), test_ids):
            raise ValueError("Prediction sample IDs differ from the held-out target split")
        if not np.array_equal(predictions["patient_ids"].astype(str), patient_ids):
            raise ValueError("Prediction patient IDs differ from the held-out target split")
        if not np.array_equal(patient_group_ids(test_ids).astype(str), patient_ids):
            raise ValueError("Target patient IDs do not match official GEO sample IDs")
        true_labels = predictions["y_true"].astype(int)

    with (run_dir / "fitted_preprocessing_and_map.pkl").open("rb") as stream:
        fitted = pickle.load(stream)
    preprocessor = fitted["preprocessor"]
    transformer = fitted["transformer"]
    encoder = fitted["label_encoder"]
    X_gxp, target_genes, all_ids, all_labels = load_expression_dataset(str(target_path))
    all_ids = np.asarray(all_ids, dtype=str)
    row_by_id = {sample_id: row for row, sample_id in enumerate(all_ids)}
    if any(sample_id not in row_by_id for sample_id in test_ids):
        raise ValueError("Held-out target sample ID is absent from the hashed target matrix")
    rows = np.asarray([row_by_id[sample_id] for sample_id in test_ids], dtype=int)
    target_truth = encoder.transform(all_labels[rows])
    if not np.array_equal(target_truth, true_labels):
        raise ValueError("Saved target labels do not match the hashed target matrix")
    X_test = preprocessor.transform_target(X_gxp[:, rows].T, target_genes)
    retained_target_genes = preprocessor.target_genes_.copy()
    if metadata["config"].get("feature_scope", "all") == "shared":
        _, target_shared_idx, shared = ordered_common_gene_indices(
            preprocessor.source_genes_, preprocessor.target_genes_)
        X_test = X_test[:, target_shared_idx]
        retained_target_genes = shared
    coords = transformer.coords_target()
    if len(retained_target_genes) != len(coords) or X_test.shape[1] != len(coords):
        raise ValueError("Saved target gene order, intensities, and coordinates do not align")
    images = transformer.transform_target(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cpu")
    model = timm.create_model(metadata["config"]["model"], pretrained=False,
                              num_classes=len(encoder.classes_)).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device,
                                     weights_only=True))
    model.eval()
    if not hasattr(model, "layer4"):
        raise ValueError("Expected the saved ResNet model to expose layer4 for Grad-CAM")
    target_layer = model.layer4[-1]
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        captured["activations"] = output
        output.register_hook(lambda grad: captured.__setitem__("gradients", grad))

    hook = target_layer.register_forward_hook(forward_hook)
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    inputs = torch.stack([preprocessing(image) for image in images]).float()
    cams_by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    try:
        for start in range(0, len(inputs), args.batch_size):
            stop = min(start + args.batch_size, len(inputs))
            batch = inputs[start:stop].to(device)
            class_ids = torch.as_tensor(true_labels[start:stop], dtype=torch.long, device=device)
            model.zero_grad(set_to_none=True)
            logits = model(batch)
            chosen_scores = logits.gather(1, class_ids[:, None]).sum()
            chosen_scores.backward()
            activations = captured["activations"]
            gradients = captured["gradients"]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            raw_cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
            raw_cam = F.interpolate(raw_cam, size=images.shape[1:3], mode="bilinear",
                                    align_corners=False)[:, 0]
            raw_np = raw_cam.detach().cpu().numpy()
            for offset, class_id in enumerate(class_ids.detach().cpu().tolist()):
                cams_by_class[int(class_id)].append(raw_np[offset])
    finally:
        hook.remove()

    class_results: dict[str, Any] = {}
    rows_out: list[dict[str, Any]] = []
    for class_id in sorted(cams_by_class):
        cam_batch = np.stack(cams_by_class[class_id])
        ranked = rank_gene_attributions(
            cam_batch, retained_target_genes, coords, top_k=args.top_k,
            normalize_mean_map=True,
        )
        class_name = str(encoder.classes_[class_id])
        class_results[class_name] = {
            "heldout_samples": int(len(cam_batch)),
            "cam_normalization": "mean raw Grad-CAM, then min-max normalize to [0,1]",
            "ranking_policy": f"top {args.top_k} genes by normalized mean CAM score",
            "genes": ranked,
        }
        rows_out.extend({"class": class_name, **row} for row in ranked)

    output = args.output or (run_dir / "target_gradcam_top_genes.json")
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "run_dir": str(run_dir),
        "arm": metadata["arm"],
        "seed": metadata["seed"],
        "target_file_sha256": metadata["target_file_sha256"],
        "target_test_ids": test_ids.tolist(),
        "feature_scope": metadata["config"].get("feature_scope", "all"),
        "target_gene_count": int(len(retained_target_genes)),
        "class_grouping": "saved held-out target labels; post-hoc attribution only",
        "target_labels_used_for_model_selection": False,
        "class_results": class_results,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "class", "rank", "gene_id", "feature_index", "score",
            "pixel_row", "pixel_column", "genes_at_pixel",
        ])
        writer.writeheader()
        writer.writerows(rows_out)
    print(json.dumps({"result": "passed", "json": str(output),
                      "csv": str(csv_path), "class_counts": {
                          name: values["heldout_samples"]
                          for name, values in class_results.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
