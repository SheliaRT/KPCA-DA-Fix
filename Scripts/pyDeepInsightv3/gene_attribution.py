"""Map image attribution maps back to the exact ordered gene feature space."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def rank_gene_attributions(
    cams: np.ndarray,
    gene_ids: np.ndarray,
    feature_coords: np.ndarray,
    *,
    top_k: int | None = None,
    threshold: float | None = None,
    normalize_mean_map: bool = True,
) -> list[dict[str, Any]]:
    """Average CAMs over a declared sample/class group and rank mapped genes.

    ``gene_ids[i]`` must be the gene at ``feature_coords[i]`` in the exact
    feature order used to construct the images. Multiple genes may occupy one
    pixel; each receives that pixel's score and the returned collision count
    makes this ambiguity explicit. By default, the mean map is min-max
    normalized to [0, 1], as specified in Supplement Note 4. Use ``top_k`` for
    the main-paper ranking or ``threshold`` for the separate supplement policy.
    """
    maps = np.asarray(cams, dtype=float)
    genes = np.asarray(gene_ids).astype(str)
    coords = np.asarray(feature_coords)
    if maps.ndim == 2:
        mean_cam = maps
    elif maps.ndim == 3 and maps.shape[0] > 0:
        mean_cam = maps.mean(axis=0)
    else:
        raise ValueError("cams must have shape (height, width) or (samples, height, width)")
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(genes):
        raise ValueError("feature_coords must have one (row, column) per ordered gene")
    if not np.issubdtype(coords.dtype, np.integer):
        if not np.isfinite(coords).all() or not np.equal(coords, np.floor(coords)).all():
            raise ValueError("feature coordinates must be finite integer pixel indices")
        coords = coords.astype(int)
    if len(coords) and (
        np.any(coords < 0)
        or np.any(coords[:, 0] >= mean_cam.shape[0])
        or np.any(coords[:, 1] >= mean_cam.shape[1])
    ):
        raise ValueError("feature coordinates fall outside the CAM dimensions")
    if not np.isfinite(mean_cam).all():
        raise ValueError("CAM values must be finite")
    if top_k is not None and threshold is not None:
        raise ValueError("top_k and threshold are separate attribution policies")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be nonnegative or None")
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be finite and in [0, 1]")
    if normalize_mean_map:
        low = float(mean_cam.min())
        high = float(mean_cam.max())
        if high > low:
            mean_cam = (mean_cam - low) / (high - low)
        else:
            mean_cam = np.zeros_like(mean_cam)
    collision_counts = Counter(map(tuple, coords.tolist()))
    scores = mean_cam[coords[:, 0], coords[:, 1]]
    order = np.argsort(-scores, kind="stable")
    if top_k is not None:
        order = order[:top_k]
    elif threshold is not None:
        order = order[scores[order] > threshold]
    return [
        {
            "rank": rank,
            "gene_id": genes[index],
            "feature_index": int(index),
            "score": float(scores[index]),
            "pixel_row": int(coords[index, 0]),
            "pixel_column": int(coords[index, 1]),
            "genes_at_pixel": int(collision_counts[tuple(coords[index].tolist())]),
        }
        for rank, index in enumerate(order, start=1)
    ]
