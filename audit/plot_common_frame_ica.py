"""Plot the saved source-fitted, paired ICA diagnostic without re-fitting it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


def render(metrics_path: Path, output_prefix: Path) -> dict[str, object]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("valid_paired_diagnostic") is not True:
        raise ValueError("Input is not a verified paired diagnostic")
    if metrics.get("target_labels_used") is not False:
        raise ValueError("Target labels must not be used for this diagnostic")
    if metrics.get("target_test_used_to_fit_frame_or_ica") is not False:
        raise ValueError("Held-out target data entered the fitted frame or ICA")

    archive_path = metrics_path.with_suffix(".npz")
    with np.load(archive_path, allow_pickle=False) as archive:
        source = archive["source_ica"]
        baseline = archive["baseline_target_ica"]
        adapted = archive["kpcada_target_ica"]
        for key in ("source_train_ids", "target_adaptation_ids", "target_test_ids"):
            if not np.array_equal(archive[key].astype(str),
                                  np.asarray(metrics[key], dtype=str)):
                raise ValueError(f"Saved NPZ sample IDs do not match metrics field {key}")
    if source.ndim != 2 or baseline.shape != adapted.shape or baseline.ndim != 2:
        raise ValueError("Saved source and target ICA coordinates have invalid shapes")
    if source.shape[1] != 2 or baseline.shape[1] != 2:
        raise ValueError("This plot requires exactly two saved ICA components")

    recomputed = np.asarray([
        [wasserstein_distance(source[:, j], baseline[:, j]),
         wasserstein_distance(source[:, j], adapted[:, j])]
        for j in range(2)
    ])
    stored = np.asarray([
        metrics["baseline_diagnostics"]["marginal_1d_wasserstein_by_component"],
        metrics["kpcada_diagnostics"]["marginal_1d_wasserstein_by_component"],
    ]).T
    if not np.allclose(recomputed, stored, rtol=1e-10, atol=1e-12):
        raise ValueError("Recomputed marginal W1 values do not match the saved metrics")

    colors = {"source": "#667085", "baseline": "#2F6B9A", "KPCA-DA": "#C65D32"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    ax = axes[0]
    ax.scatter(source[:, 0], source[:, 1], s=13, c=colors["source"], alpha=.32,
               edgecolors="none", label=f"Source train (n={len(source)})")
    ax.scatter(baseline[:, 0], baseline[:, 1], s=32, c=colors["baseline"], alpha=.78,
               marker="o", edgecolors="white", linewidths=.25,
               label=f"Baseline target test (n={len(baseline)})")
    ax.scatter(adapted[:, 0], adapted[:, 1], s=40, c=colors["KPCA-DA"], alpha=.78,
               marker="^", edgecolors="white", linewidths=.25,
               label="KPCA-DA target test")
    ax.set(title="Shared source-fitted ICA frame", xlabel="ICA component 1",
           ylabel="ICA component 2")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.grid(alpha=.18)

    for component, ax in enumerate(axes[1:]):
        ax.hist(source[:, component], bins=24, density=True, color=colors["source"],
                alpha=.25, label="Source train")
        ax.hist(baseline[:, component], bins=24, density=True, histtype="step",
                linewidth=1.8, color=colors["baseline"], label="Baseline target test")
        ax.hist(adapted[:, component], bins=24, density=True, histtype="step",
                linewidth=1.8, color=colors["KPCA-DA"], label="KPCA-DA target test")
        w1_base, w1_adapted = recomputed[component]
        ax.set(title=(f"Component {component + 1} marginal W1\n"
                      f"baseline {w1_base:.3f}; KPCA-DA {w1_adapted:.3f}"),
               xlabel=f"ICA component {component + 1}", ylabel="Density")
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        f"Seed {metrics['seed']}: unshifted source-fitted ICA diagnostic",
        fontsize=14, fontweight="bold")
    fig.text(.5, -.015,
             "ICA fit on source-training images only; target test labels and points were not used to fit the frame or projection. "
             "Marginal W1 is reported per component, not as multivariate Wasserstein distance.",
             ha="center", va="top", fontsize=8)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    report = {
        "source_fit_count": int(source.shape[0]),
        "target_test_count": int(baseline.shape[0]),
        "baseline_marginal_w1_by_component": recomputed[:, 0].tolist(),
        "kpcada_marginal_w1_by_component": recomputed[:, 1].tolist(),
        "baseline_mean_marginal_w1": float(recomputed[:, 0].mean()),
        "kpcada_mean_marginal_w1": float(recomputed[:, 1].mean()),
        "target_labels_used": False,
        "target_test_used_to_fit_frame_or_ica": False,
        "figure_png": str(output_prefix.with_suffix(".png")),
        "figure_pdf": str(output_prefix.with_suffix(".pdf")),
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path,
                        nargs="?", default=Path("audit/seed47_common_frame_ica_comparison.json"))
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("audit/seed47_common_frame_ica_figure"))
    args = parser.parse_args()
    print(json.dumps(render(args.metrics, args.output_prefix), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
