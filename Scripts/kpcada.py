import hashlib
import csv
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from typing import Any, Dict, List, Optional, Tuple


_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_PATH = str(_REPOSITORY_ROOT / "Data" / "TvsN" / "TCGA_BRCA_NvT_new.csv")
DEFAULT_TARGET_PATH = str(_REPOSITORY_ROOT / "Data" / "TvsN" / "GSE45498_TvsN_official_200.csv")


def _fit_signature(X_source: np.ndarray, X_target: np.ndarray,
                   genes_source: np.ndarray, genes_target: np.ndarray,
                   settings: tuple[Any, ...] = ()) -> str:
    digest = hashlib.sha256()
    for value in (X_source, X_target, genes_source, genes_target):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        if array.dtype.kind in "OUS":
            digest.update(repr(array.tolist()).encode("utf-8"))
        else:
            digest.update(array.view(np.uint8))
    digest.update(repr(settings).encode("utf-8"))
    return digest.hexdigest()


def load_expression_matrix(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load gene-by-patient expression data and ordered gene identifiers."""
    X, genes, _, _ = load_expression_dataset(csv_path)
    return X, genes


def load_expression_dataset(
    csv_path: str, *, include_labels: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load expression, gene IDs, patient IDs, and labels from the CSV layout.

    Returns expression as genes x patients, matching the KPCA-DA API. Patient
    IDs are preserved so callers can split original people before processing.
    Set ``include_labels=False`` for unlabeled target inference/preflight; the
    label column is omitted from the feature matrix and is not materialized as
    a label array.
    """
    read_kwargs = {"encoding": "utf-8", "engine": "python"}
    if not include_labels:
        # The repository matrix stores target labels in its final CSV row.
        # Skip that row in the parser rather than reading it and discarding it
        # later, preserving a genuinely label-free preflight path.
        read_kwargs["skipfooter"] = 1
    expr = pd.read_csv(csv_path, **read_kwargs)
    expr = expr.transpose()
    expr.columns = expr.iloc[0]
    samples = expr.index[1:].to_numpy()
    if include_labels:
        matrix = expr.iloc[1:, 0:-1].astype(float).values.T
        genes = expr.iloc[:, :-1].columns.to_numpy()
        labels = expr.iloc[1:, -1].to_numpy()
    else:
        matrix = expr.iloc[1:, :].astype(float).values.T
        genes = expr.columns.to_numpy()
        labels = None
    return matrix, genes, samples, labels


def load_labels(csv_path: str) -> np.ndarray:
    """Read only the final label row of the expression CSV.

    Runners call this after model fitting and held-out prediction. Streaming
    the last row avoids loading the large expression matrix a second time at
    scoring time.
    """
    last_row = None
    expected_columns = None
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError("Expression CSV must include a header and sample columns")
        expected_columns = len(header)
        for row in reader:
            if row:
                last_row = row
    if last_row is None or len(last_row) != expected_columns:
        raise ValueError("Expression CSV final label row is missing or malformed")
    return np.asarray(last_row[1:], dtype=str)


def standardize_datasets(X_source: np.ndarray, X_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit feature scaling on source rows and apply it to target rows.

    The arrays must have the same ordered features. Use gene-aware preparation
    when the domains have different feature sets.
    """
    if X_source.ndim != 2 or X_target.ndim != 2:
        raise ValueError("Source and target arrays must be two-dimensional")
    if X_source.shape[1] != X_target.shape[1]:
        raise ValueError("Source-fitted scaling requires matching feature counts")
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    return X_source_scaled, X_target_scaled


def find_common_gene_indices(genes_source: np.ndarray, genes_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return one-to-one shared-gene indices, ordered by source gene ID.

    When an input contains duplicate gene identifiers, only the first row for
    each ID participates in the paired MMD term. The full embeddings remain in
    the preservation penalty.
    """
    target_first = {}
    for idx, gene in enumerate(np.asarray(genes_target, dtype=object).tolist()):
        target_first.setdefault(gene, idx)
    source_seen = set()
    source_idx, target_idx = [], []
    for idx, gene in enumerate(np.asarray(genes_source, dtype=object).tolist()):
        if gene in source_seen:
            continue
        source_seen.add(gene)
        if gene in target_first:
            source_idx.append(idx)
            target_idx.append(target_first[gene])
    if not source_idx:
        raise ValueError("No shared gene identifiers found between the domains")
    return np.asarray(source_idx, dtype=int), np.asarray(target_idx, dtype=int)


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    pairwise_distances = torch.cdist(x, y) ** 2
    return torch.exp(-gamma * pairwise_distances)


def resolve_optimization_device(requested: str = "cpu") -> torch.device:
    """Resolve the tensor-optimization device without changing the KPCA fits."""
    value = str(requested).lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA optimization was requested but CUDA is unavailable")
    if value == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS optimization was requested but MPS is unavailable")
    if value not in {"cpu", "cuda", "mps"}:
        raise ValueError("optimization_device must be auto, cpu, cuda, or mps")
    return torch.device(value)


def compute_median_gamma(Zs: torch.Tensor, Zt: torch.Tensor) -> float:
    """Set RBF gamma via the median pairwise-distance heuristic.

    Pools source and target points, computes all squared distances, and returns
    gamma = 1 / median(d²) so the kernel is neither saturated nor near-zero.
    """
    Z_all = torch.cat([Zs, Zt], dim=0)
    dists_sq = torch.cdist(Z_all, Z_all) ** 2
    median_sq = torch.median(dists_sq[dists_sq > 0])
    return (1.0 / median_sq.clamp(min=1e-8)).item()


def mmd_loss(
    Zs_common: torch.Tensor,
    Zt_common: torch.Tensor,
    gamma: float = 1.0,
    *,
    source_kernel_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Biased empirical squared MMD for an RBF kernel (main paper Eq. 15)."""
    if source_kernel_mean is None:
        source_kernel_mean = rbf_kernel(Zs_common, Zs_common, gamma).mean()
    elif source_kernel_mean.numel() != 1:
        raise ValueError("source_kernel_mean must be a scalar kernel average")
    K_tt = rbf_kernel(Zt_common, Zt_common, gamma)
    K_st = rbf_kernel(Zs_common, Zt_common, gamma)
    return source_kernel_mean + K_tt.mean() - 2.0 * K_st.mean()


def regularization_term_from_kernel(
    Z_t: torch.Tensor, K_t: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the manuscript trace penalty with a precomputed fixed kernel."""
    if K_t.ndim != 2 or K_t.shape[0] != K_t.shape[1] or K_t.shape[0] != Z_t.shape[0]:
        raise ValueError("K_t must be square and match the target gene coordinate rows")
    return torch.trace(Z_t.T @ K_t @ Z_t)


def regularization_term(Z_t: torch.Tensor, X_t: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    K_t = rbf_kernel(X_t.T, X_t.T, gamma)
    return regularization_term_from_kernel(Z_t, K_t)


def reference_preservation_term(Z_t: torch.Tensor, Z_t_initial: torch.Tensor,
                                eps: float = 1e-12) -> torch.Tensor:
    """Normalized squared movement from the initial target gene coordinates."""
    centered = Z_t_initial - Z_t_initial.mean(dim=0, keepdim=True)
    reference_energy = centered.square().sum()
    if reference_energy.detach().item() <= eps:
        raise ValueError("Initial target embedding has no variance to preserve")
    return (Z_t - Z_t_initial).square().sum() / reference_energy.clamp_min(eps)


def target_coordinate_integrity(
    Z_initial: np.ndarray, Z_adapted: np.ndarray,
) -> Dict[str, Any]:
    """Measure coordinate validity and deformation without target labels.

    The collapse cutoff is a numerical guard, not a performance-derived
    threshold. Other measures are reported but do not affect selection.
    """
    initial = np.asarray(Z_initial, dtype=float)
    adapted = np.asarray(Z_adapted, dtype=float)
    if initial.shape != adapted.shape or initial.ndim != 2 or initial.size == 0:
        raise ValueError("Initial and adapted target coordinates must be matching non-empty matrices")
    finite = bool(np.isfinite(initial).all() and np.isfinite(adapted).all())
    if not finite:
        return {"finite": False, "variance_retention_ratio": None,
                "normalized_displacement": None, "collapsed": True}
    initial_centered = initial - initial.mean(axis=0, keepdims=True)
    adapted_centered = adapted - adapted.mean(axis=0, keepdims=True)
    initial_energy = float(np.mean(np.sum(initial_centered ** 2, axis=1)))
    adapted_energy = float(np.mean(np.sum(adapted_centered ** 2, axis=1)))
    if initial_energy <= np.finfo(float).tiny:
        return {
            "finite": True,
            "initial_centered_energy": initial_energy,
            "adapted_centered_energy": adapted_energy,
            "variance_retention_ratio": None,
            "normalized_displacement": None,
            "collapsed": True,
        }
    variance_ratio = adapted_energy / initial_energy
    scale = float(np.sqrt(initial_energy))
    displacement = float(np.sqrt(np.mean(np.sum((adapted - initial) ** 2, axis=1))))
    normalized_displacement = displacement / scale
    collapsed = adapted_energy <= 1e-12 * initial_energy
    return {
        "finite": True,
        "initial_centered_energy": initial_energy,
        "adapted_centered_energy": adapted_energy,
        "variance_retention_ratio": float(variance_ratio),
        "normalized_displacement": float(normalized_displacement),
        "collapsed": bool(collapsed),
    }


def align_target_embedding_to_source(
    Z_source: np.ndarray,
    Z_target: np.ndarray,
    common_source_idx: np.ndarray,
    common_target_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a label-free orthogonal Procrustes orientation using shared genes.

    The mapping uses only paired gene identities and their initial KPCA
    coordinates. It rotates/refects and translates the entire target embedding
    so shared-gene positions are least-squares aligned before MMD optimization.
    It does not scale distances. This is an optional method revision because
    the manuscript does not specify this orientation-alignment step.
    """
    source = np.asarray(Z_source, dtype=float)
    target = np.asarray(Z_target, dtype=float)
    source_idx = np.asarray(common_source_idx, dtype=int)
    target_idx = np.asarray(common_target_idx, dtype=int)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Source and target embeddings must share a component dimension")
    if source_idx.shape != target_idx.shape or source_idx.ndim != 1 or source_idx.size < 2:
        raise ValueError("At least two paired shared genes are required for alignment")
    if (np.any(source_idx < 0) or np.any(source_idx >= len(source))
            or np.any(target_idx < 0) or np.any(target_idx >= len(target))):
        raise ValueError("Shared-gene indices fall outside the embedding")
    src_anchor = source[source_idx]
    tgt_anchor = target[target_idx]
    src_center = src_anchor.mean(axis=0)
    tgt_center = tgt_anchor.mean(axis=0)
    src_centered = src_anchor - src_center
    tgt_centered = tgt_anchor - tgt_center
    source_rank = int(np.linalg.matrix_rank(src_centered))
    target_rank = int(np.linalg.matrix_rank(tgt_centered))
    if source_rank == 0 or target_rank == 0:
        raise ValueError("Shared-gene anchors must have nonzero coordinate variance")
    u, singular_values, vt = np.linalg.svd(tgt_centered.T @ src_centered,
                                           full_matrices=False)
    rotation = u @ vt
    aligned = (target - tgt_center) @ rotation + src_center
    before_rmse = float(np.sqrt(np.mean(np.square(tgt_anchor - src_anchor))))
    after_rmse = float(np.sqrt(np.mean(np.square(aligned[target_idx] - src_anchor))))
    return aligned, {
        "rotation": rotation.tolist(),
        "source_anchor_center": src_center.tolist(),
        "target_anchor_center": tgt_center.tolist(),
        "singular_values": singular_values.tolist(),
        "source_anchor_rank": source_rank,
        "target_anchor_rank": target_rank,
        "shared_gene_count": int(source_idx.size),
        "paired_coordinate_rmse_before": before_rmse,
        "paired_coordinate_rmse_after": after_rmse,
    }


def fit_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    genes_source: Optional[np.ndarray] = None,
    genes_target: Optional[np.ndarray] = None,
    *,
    n_components: int = 2,
    kernel: str = "rbf",
    kpca_gamma: Optional[float] = 0.2,
    mmd_gamma: float = 0.5,
    auto_gamma: bool = True,
    lambda_reg: float = 0.001,
    alignment_weight: float = 1.0,
    lr: float = 0.001,
    epochs: int = 500,
    regularization_mode: str = "trace",
    kpca_random_state: int = 42,
    align_shared_genes: bool = False,
    optimization_device: str = "cpu",
    initial_embeddings: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Align source and target domains using kernel PCA and MMD-based adaptation.

    Uses gene-based matching to find common features between datasets.
    When auto_gamma=True (default), the MMD kernel bandwidth is set automatically
    via the median pairwise-distance heuristic on the initial KPCA embeddings,
    replacing the fixed mmd_gamma value.

    optimization_device selects the CPU/CUDA/MPS device for the differentiable
    objective only; the two sklearn KernelPCA fits remain on CPU.
    """
    # Find common gene indices
    if genes_source is None or genes_target is None:
        raise ValueError("genes_source and genes_target must be provided for gene-based matching")
    if not np.isfinite(alignment_weight) or alignment_weight < 0:
        raise ValueError("alignment_weight must be finite and nonnegative")
    objective_device = resolve_optimization_device(optimization_device)
    
    common_source_idx, common_target_idx = find_common_gene_indices(genes_source, genes_target)

    if common_source_idx.size == 0 or common_target_idx.size == 0:
        raise ValueError("No common genes found between the source and target datasets.")

    # Note: Standardization is optional - commented out in modified version
    # X_source_scaled, X_target_scaled = standardize_datasets(X_source, X_target)
    # Using data as-is (already scaled/normalized externally if needed)
    X_source_scaled = X_source
    X_target_scaled = X_target

    # Fit KPCA once per domain. A tuning sweep may pass these fixed initial
    # coordinates so candidate lambdas differ only in their regularization.
    kpca_alignment = None
    if initial_embeddings is None:
        kpca_source = KernelPCA(n_components=n_components, kernel=kernel,
                                gamma=kpca_gamma, random_state=kpca_random_state)
        kpca_target = KernelPCA(n_components=n_components, kernel=kernel,
                                gamma=kpca_gamma, random_state=kpca_random_state)
        Z_s_np = kpca_source.fit_transform(X_source_scaled)
        Z_t_np = kpca_target.fit_transform(X_target_scaled)
        if align_shared_genes:
            Z_t_np, kpca_alignment = align_target_embedding_to_source(
                Z_s_np, Z_t_np, common_source_idx, common_target_idx
            )
    else:
        kpca_source = initial_embeddings.get("kpca_source_model")
        kpca_target = initial_embeddings.get("kpca_target_model")
        if kpca_target is None:
            # Backward compatibility for cached results created before the
            # source/target estimator split.
            kpca_target = initial_embeddings["kpca_model"]
        Z_s_np = np.asarray(initial_embeddings["Z_s"])
        Z_t_np = np.asarray(initial_embeddings["Z_t_initial"])
        kpca_alignment = initial_embeddings.get("kpca_alignment")
        if align_shared_genes and kpca_alignment is None:
            raise ValueError("Cached initial embeddings do not include shared-gene alignment")

    # KPCA fitting remains in sklearn/NumPy. Only the differentiable objective
    # and its fixed kernels move to the requested accelerator.
    Z_s = torch.tensor(Z_s_np, dtype=torch.float32, device=objective_device)
    Z_t = torch.tensor(Z_t_np, dtype=torch.float32, device=objective_device,
                       requires_grad=True)
    Z_t_initial = torch.tensor(Z_t_np, dtype=torch.float32, device=objective_device)
    X_t_tensor = torch.tensor(X_target_scaled.T, dtype=torch.float32,
                              device=objective_device)

    # Auto-select MMD gamma from the initial embedding distances
    if auto_gamma:
        mmd_gamma = compute_median_gamma(
            Z_s[common_source_idx], Z_t[common_target_idx].detach()
        )
        if verbose:
            print(f"Auto-computed MMD gamma (median heuristic): {mmd_gamma:.6f}")

    # Both kernels depend only on fixed fit inputs and gamma. Recompute neither
    # during the 500-step coordinate optimization nor during loss-history logging.
    Zs_common = Z_s[common_source_idx]
    source_kernel_mean = rbf_kernel(
        Zs_common, Zs_common, mmd_gamma
    ).mean().detach()
    target_regularization_kernel = None
    if regularization_mode == "trace":
        with torch.no_grad():
            target_regularization_kernel = rbf_kernel(
                X_t_tensor.T, X_t_tensor.T, mmd_gamma
            ).detach()

    optimizer = optim.SGD([Z_t], lr=lr)
    loss_history_tensors: List[torch.Tensor] = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Extract common points
        Zt_common = Z_t[common_target_idx]

        # Compute losses
        mmd = mmd_loss(
            Zs_common, Zt_common, gamma=mmd_gamma,
            source_kernel_mean=source_kernel_mean,
        )
        if regularization_mode == "reference":
            reg = reference_preservation_term(Z_t, Z_t_initial)
        elif regularization_mode == "trace":
            reg = regularization_term_from_kernel(Z_t, target_regularization_kernel)
        elif regularization_mode == "none":
            reg = Z_t.new_zeros(())
        else:
            raise ValueError("regularization_mode must be 'reference', 'trace', or 'none'")
        loss = alignment_weight * mmd + lambda_reg * reg

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Log all terms at the same post-update endpoint returned to callers.
        with torch.no_grad():
            mmd_after = mmd_loss(
                Zs_common, Z_t[common_target_idx], gamma=mmd_gamma,
                source_kernel_mean=source_kernel_mean,
            )
            if regularization_mode == "reference":
                reg_after = reference_preservation_term(Z_t, Z_t_initial)
            elif regularization_mode == "trace":
                reg_after = regularization_term_from_kernel(
                    Z_t, target_regularization_kernel
                )
            else:
                reg_after = Z_t.new_zeros(())
            loss_after = alignment_weight * mmd_after + lambda_reg * reg_after
        loss_history_tensors.append(loss_after.detach())

        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}, Loss: {loss_after.item():.4f}, MMD: {mmd_after.item():.4f}, Reg: {reg_after.item():.4f}")

    # Return all loss components at the exact coordinates serialized by the
    # caller. This also covers the zero-epoch no-adaptation control.
    with torch.no_grad():
        final_mmd = mmd_loss(
            Zs_common, Z_t[common_target_idx], gamma=mmd_gamma,
            source_kernel_mean=source_kernel_mean,
        )
        if regularization_mode == "reference":
            final_reg = reference_preservation_term(Z_t, Z_t_initial)
        elif regularization_mode == "trace":
            final_reg = regularization_term_from_kernel(
                Z_t, target_regularization_kernel
            )
        else:
            final_reg = Z_t.new_zeros(())
        final_loss = alignment_weight * final_mmd + lambda_reg * final_reg

    loss_history = (
        torch.stack(loss_history_tensors).cpu().tolist()
        if loss_history_tensors else []
    )

    return {
        "Z_s": Z_s.detach().cpu().numpy(),
        "Z_t": Z_t.detach().cpu().numpy(),
        "Z_t_initial": Z_t_initial.detach().cpu().numpy(),
        "common_source_idx": common_source_idx,
        "common_target_idx": common_target_idx,
        "loss_history": loss_history,
        # Preserve both independent fits. `kpca_model` remains the historical
        # target-model alias used by existing callers.
        "kpca_model": kpca_target,
        "kpca_source_model": kpca_source,
        "kpca_target_model": kpca_target,
        "kpca_alignment": kpca_alignment,
        "common_genes_count": len(common_source_idx),
        "mmd_gamma": mmd_gamma,
        "regularization_mode": regularization_mode,
        "optimization_device": str(objective_device),
        "lambda_reg": lambda_reg,
        "alignment_weight": alignment_weight,
        "learning_rate": lr,
        "epochs": epochs,
        "final_mmd": final_mmd.item(),
        "final_regularization": final_reg.item(),
        "final_loss": final_loss.item(),
    }


class KPCA_DA:
    """Kernel PCA-based Domain Adaptation using MMD and regularization.
    
    This class implements domain adaptation by aligning source and target domains
    in a kernel PCA space using Maximum Mean Discrepancy (MMD) loss with regularization.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "rbf",
        kpca_gamma: Optional[float] = 0.2,
        mmd_gamma: float = 0.5,
        auto_gamma: bool = True,
        lambda_reg: float = 0.001,
        alignment_weight: float = 1.0,
        lr: float = 0.001,
        epochs: int = 500,
        regularization_mode: str = "trace",
        kpca_random_state: int = 42,
        align_shared_genes: bool = False,
        optimization_device: str = "cpu",
        verbose: bool = False,
    ):
        """Initialize KPCA_DA model.

        Parameters
        ----------
        n_components : int, default=2
            Number of components for KernelPCA
        kernel : str, default='rbf'
            Kernel type for KernelPCA
        kpca_gamma : float or None, default=0.2
            Gamma parameter for KernelPCA kernel
        mmd_gamma : float, default=0.5
            Gamma parameter for MMD kernel (ignored when auto_gamma=True)
        auto_gamma : bool, default=True
            Automatically set MMD gamma via median pairwise-distance heuristic
        lambda_reg : float, default=0.001
            Regularization strength
        lr : float, default=0.001
            Learning rate for optimization
        epochs : int, default=500
            Number of optimization epochs
        regularization_mode : str, default='trace'
            Target-coordinate penalty: historical manuscript trace penalty
            ('trace'), normalized movement from initial coordinates
            ('reference'), or no penalty ('none').
        optimization_device : str, default='cpu'
            Device used for the differentiable objective: 'cpu', 'cuda', 'mps',
            or 'auto'. KernelPCA fitting remains on CPU.
        verbose : bool, default=False
            Whether to print progress
        """
        self.n_components = n_components
        self.kernel = kernel
        self.kpca_gamma = kpca_gamma
        self.mmd_gamma = mmd_gamma
        self.auto_gamma = auto_gamma
        self.lambda_reg = lambda_reg
        self.alignment_weight = alignment_weight
        self.lr = lr
        self.epochs = epochs
        self.regularization_mode = regularization_mode
        self.kpca_random_state = kpca_random_state
        self.align_shared_genes = align_shared_genes
        self.optimization_device = optimization_device
        self.verbose = verbose
        
        self.kpca_model = None
        self.kpca_source_model_ = None
        self.kpca_target_model_ = None
        self.Z_s_ = None
        self.Z_t_ = None
        self.Z_t_initial_ = None
        self.common_source_idx_ = None
        self.common_target_idx_ = None
        self.loss_history_ = None
        self._tuned_fit_result = None
        self._tuned_fit_signature = None

    def _fit_cache_settings(self) -> tuple[Any, ...]:
        """Settings that determine cached KPCA-DA coordinates and losses."""
        return (
            self.n_components, self.kernel, self.kpca_gamma, self.mmd_gamma,
            self.auto_gamma, self.lambda_reg, self.alignment_weight, self.lr,
            self.epochs, self.regularization_mode, self.kpca_random_state,
            self.align_shared_genes, self.optimization_device,
        )
    
    def fit_transform(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        genes_source: Optional[np.ndarray] = None,
        genes_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the model and transform both source and target data.
        
        Parameters
        ----------
        X_source : np.ndarray
            Source domain data (genes x samples)
        X_target : np.ndarray
            Target domain data (genes x samples)
        genes_source : np.ndarray or None
            Gene names for source data
        genes_target : np.ndarray or None
            Gene names for target data
            
        Returns
        -------
        Z_s : np.ndarray
            Transformed source data
        Z_t : np.ndarray
            Adapted target data
        """
        if genes_source is None or genes_target is None:
            raise ValueError("genes_source and genes_target must be provided for gene-based matching")
        signature = _fit_signature(
            X_source, X_target, genes_source, genes_target,
            settings=self._fit_cache_settings(),
        )
        if (self._tuned_fit_result is not None
                and signature == self._tuned_fit_signature):
            results = self._tuned_fit_result
        else:
            results = fit_transform(
                X_source=X_source,
                X_target=X_target,
                genes_source=genes_source,
                genes_target=genes_target,
                n_components=self.n_components,
                kernel=self.kernel,
                kpca_gamma=self.kpca_gamma,
                mmd_gamma=self.mmd_gamma,
                auto_gamma=self.auto_gamma,
                lambda_reg=self.lambda_reg,
                alignment_weight=self.alignment_weight,
                lr=self.lr,
                epochs=self.epochs,
                regularization_mode=self.regularization_mode,
                kpca_random_state=self.kpca_random_state,
                align_shared_genes=self.align_shared_genes,
                optimization_device=self.optimization_device,
                verbose=self.verbose,
            )
        self._tuned_fit_result = None
        self._tuned_fit_signature = None
        
        # Store results
        self.Z_s_ = results["Z_s"]
        self.Z_t_ = results["Z_t"]
        self.Z_t_initial_ = results["Z_t_initial"]
        self.common_source_idx_ = results["common_source_idx"]
        self.common_target_idx_ = results["common_target_idx"]
        self.loss_history_ = results["loss_history"]
        self.mmd_gamma_ = results["mmd_gamma"]
        self.final_mmd_ = results["final_mmd"]
        self.final_regularization_ = results["final_regularization"]
        self.final_loss_ = results["final_loss"]
        self.lambda_reg_ = results["lambda_reg"]
        self.alignment_weight_ = results["alignment_weight"]
        self.regularization_mode_ = results["regularization_mode"]
        self.optimization_epochs_ = results["epochs"]
        self.optimization_lr_ = results["learning_rate"]
        self.optimization_device_ = results["optimization_device"]
        self.kpca_model = results["kpca_model"]
        self.kpca_source_model_ = results.get("kpca_source_model")
        self.kpca_target_model_ = results.get("kpca_target_model", self.kpca_model)
        self.kpca_alignment_ = results.get("kpca_alignment")
        
        return self.Z_s_, self.Z_t_
    
    def tune_lambda(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        genes_source: np.ndarray,
        genes_target: np.ndarray,
        lambda_values: List[float] = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
        y_source: Optional[np.ndarray] = None,
        y_target: Optional[np.ndarray] = None,
        classifier=None,
        allow_target_label_selection: bool = False,
    ) -> Dict[str, Any]:
        """Grid-search lambda_reg; target-label accuracy selection is opt-in.

        See module-level tune_lambda() for full parameter documentation.
        Returns the summary dict and sets self.lambda_reg to the best value
        (best_lambda_accuracy if labels provided, else best_lambda_mmd).

        Also pins self.mmd_gamma to the gamma value used during the search and
        sets self.auto_gamma = False, so a later fit_transform() on this same
        instance reuses the already-computed gamma instead of recomputing it.
        """
        summary = tune_lambda(
            X_source=X_source,
            X_target=X_target,
            genes_source=genes_source,
            genes_target=genes_target,
            lambda_values=lambda_values,
            y_source=y_source,
            y_target=y_target,
            classifier=classifier,
            allow_target_label_selection=allow_target_label_selection,
            n_components=self.n_components,
            kernel=self.kernel,
            kpca_gamma=self.kpca_gamma,
            mmd_gamma=self.mmd_gamma,
            auto_gamma=self.auto_gamma,
            alignment_weight=self.alignment_weight,
            lr=self.lr,
            epochs=self.epochs,
            regularization_mode=self.regularization_mode,
            kpca_random_state=self.kpca_random_state,
            align_shared_genes=self.align_shared_genes,
            optimization_device=self.optimization_device,
            verbose=self.verbose,
        )
        if y_source is not None and y_target is not None:
            selected_lambda = float(summary["best_lambda_accuracy"])
        else:
            selected_lambda = float(summary["best_lambda_mmd"])
        self.lambda_reg = selected_lambda
        self.mmd_gamma = summary["mmd_gamma"]
        self.auto_gamma = False
        self._tuned_fit_result = summary["_fit_results_by_lambda"][selected_lambda]
        self._tuned_fit_signature = _fit_signature(
            X_source, X_target, genes_source, genes_target,
            settings=self._fit_cache_settings(),
        )
        summary.pop("_fit_results_by_lambda")
        return summary

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compatibility alias for :meth:`transform_target_genes_unadapted`.

        This method keeps the historical estimator-style API. It does not
        perform patient inference or return the optimized target coordinates.
        """
        return self.transform_target_genes_unadapted(X)

    def transform_target_genes_unadapted(self, X: np.ndarray) -> np.ndarray:
        """Project target gene rows with the frozen, unadapted target KPCA fit.

        This does not return the optimized target coordinates from ``Z_t_``
        and is not patient-image inference. For patient inference, use the
        fitted image transformer's frozen target gene-to-pixel map.
        
        Parameters
        ----------
        X : np.ndarray
            Data to transform
            
        Returns
        -------
        Z : np.ndarray
            Transformed data
        """
        if self.kpca_target_model_ is None:
            raise ValueError("Model has not been fitted yet. Call fit_transform first.")
        
        return self.kpca_target_model_.transform(X)

    def transform_source(self, X: np.ndarray) -> np.ndarray:
        """Project gene profiles with the independently fitted source KPCA.

        The input must have the source KPCA training feature dimension. This
        projects gene profiles; it is not inference for a new patient sample.
        """
        if self.kpca_source_model_ is None:
            raise ValueError("Model has not been fitted with a source KPCA estimator")
        return self.kpca_source_model_.transform(X)


def tune_lambda(
    X_source: np.ndarray,
    X_target: np.ndarray,
    genes_source: np.ndarray,
    genes_target: np.ndarray,
    lambda_values: List[float] = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
    y_source: Optional[np.ndarray] = None,
    y_target: Optional[np.ndarray] = None,
    classifier=None,
    *,
    n_components: int = 2,
    kernel: str = "rbf",
    kpca_gamma: Optional[float] = 0.2,
    mmd_gamma: float = 0.5,
    auto_gamma: bool = True,
    alignment_weight: float = 1.0,
    lr: float = 0.001,
    epochs: int = 500,
    regularization_mode: str = "trace",
    allow_target_label_selection: bool = False,
    kpca_random_state: int = 42,
    align_shared_genes: bool = False,
    optimization_device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Search over lambda_values and report final MMD and (optionally) accuracy for each.

    Label-free MMD selection excludes candidates with nonfinite objectives or
    numerically collapsed target coordinates. Every candidate reports its
    variance-retention ratio and normalized displacement; those measurements
    do not otherwise impose an empirical distortion threshold.

    Parameters
    ----------
    lambda_values : sequence of float
        Candidate regularization strengths to evaluate.
    y_source, y_target : array-like or None
        Optional labels for a separately reported supervised diagnostic. They
        cannot select lambda unless ``allow_target_label_selection=True``.
    classifier : sklearn-compatible estimator or None
        Trained/fitted on source KPCA embeddings and evaluated on adapted target.
        Defaults to KNeighborsClassifier(n_neighbors=5) when labels are supplied.

    Returns
    -------
    dict with keys:
        "results"     — list of dicts, one per lambda, with keys:
                          lambda, final_mmd, final_loss, accuracy (if labels given)
        "best_lambda_mmd"      — eligible lambda with lowest final MMD
        "rejected_candidates"  — lambda values excluded by numerical validity checks
        "best_lambda_accuracy" — lambda with highest accuracy (if labels given)
        "results_df"  — pandas DataFrame summary
    """
    use_accuracy = y_source is not None and y_target is not None
    if use_accuracy and not allow_target_label_selection:
        raise ValueError(
            "Target labels cannot select adaptation strength in the unsupervised "
            "protocol. Set allow_target_label_selection=True only for a separately "
            "reported supervised development experiment."
        )
    if (y_source is None) != (y_target is None):
        raise ValueError("Supply both label arrays for supervised diagnostics, or neither")
    if not np.isfinite(alignment_weight) or alignment_weight < 0:
        raise ValueError("alignment_weight must be finite and nonnegative")
    if use_accuracy and classifier is None:
        classifier = KNeighborsClassifier(n_neighbors=5)

    lambda_values = tuple(float(value) for value in lambda_values)
    if not lambda_values or not np.isfinite(lambda_values).all() or min(lambda_values) < 0:
        raise ValueError("lambda_values must be a non-empty sequence of finite nonnegative values")

    # Fit KPCA only once. Every candidate then starts from identical embeddings
    # and uses the same MMD bandwidth, eliminating candidate-specific solver
    # sign/order differences.
    kpca_source_init = KernelPCA(n_components=n_components, kernel=kernel,
                                 gamma=kpca_gamma, random_state=kpca_random_state)
    kpca_target_init = KernelPCA(n_components=n_components, kernel=kernel,
                                 gamma=kpca_gamma, random_state=kpca_random_state)
    Z_s_init = kpca_source_init.fit_transform(X_source)
    Z_t_init = kpca_target_init.fit_transform(X_target)
    common_src_init, common_tgt_init = find_common_gene_indices(genes_source, genes_target)
    kpca_alignment = None
    if align_shared_genes:
        Z_t_init, kpca_alignment = align_target_embedding_to_source(
            Z_s_init, Z_t_init, common_src_init, common_tgt_init
        )
    initial_embeddings = {
        "Z_s": Z_s_init,
        "Z_t_initial": Z_t_init,
        "kpca_model": kpca_target_init,
        "kpca_source_model": kpca_source_init,
        "kpca_target_model": kpca_target_init,
        "kpca_alignment": kpca_alignment,
    }
    if auto_gamma:
        mmd_gamma = compute_median_gamma(
            torch.tensor(Z_s_init[common_src_init], dtype=torch.float32),
            torch.tensor(Z_t_init[common_tgt_init], dtype=torch.float32),
        )
        if verbose:
            print(f"Pre-computed MMD gamma (median heuristic, shared across all lambda): {mmd_gamma:.6f}")

    records = []
    fit_results: dict[float, Dict[str, Any]] = {}
    for lam in lambda_values:
        if verbose:
            print(f"\n--- lambda = {lam} ---")

        res = fit_transform(
            X_source=X_source,
            X_target=X_target,
            genes_source=genes_source,
            genes_target=genes_target,
            n_components=n_components,
            kernel=kernel,
            kpca_gamma=kpca_gamma,
            mmd_gamma=mmd_gamma,
            auto_gamma=False,   # already computed above; don't recompute per run
            lambda_reg=lam,
            alignment_weight=alignment_weight,
            lr=lr,
            epochs=epochs,
            regularization_mode=regularization_mode,
            kpca_random_state=kpca_random_state,
            align_shared_genes=align_shared_genes,
            optimization_device=optimization_device,
            initial_embeddings=initial_embeddings,
            verbose=verbose,
        )
        fit_results[lam] = res

        Z_s = res["Z_s"]
        Z_t = res["Z_t"]
        common_src = res["common_source_idx"]
        common_tgt = res["common_target_idx"]

        final_mmd = res["final_mmd"]
        final_reg = res["final_regularization"]
        final_loss = res["final_loss"]

        record: Dict[str, Any] = {
            "lambda": lam,
            "final_mmd": final_mmd,
            "final_loss": final_loss,
            "final_regularization": final_reg,
            "coordinate_integrity": target_coordinate_integrity(
                res["Z_t_initial"], Z_t
            ),
            "final_loss_history_endpoint": (
                res["loss_history"][-1] if res["loss_history"] else final_loss
            ),
        }
        objective_finite = bool(np.isfinite([
            final_mmd, final_loss, final_reg
        ]).all())
        record["eligible_for_label_free_selection"] = bool(
            objective_finite
            and record["coordinate_integrity"]["finite"]
            and not record["coordinate_integrity"]["collapsed"]
        )

        if use_accuracy:
            # Project samples into gene-embedding space:
            # X (genes × samples) transposed gives (samples × genes),
            # multiplied by Z (genes × components) → (samples × components)
            X_s_proj = X_source[common_src, :].T @ Z_s[common_src]
            X_t_proj = X_target[common_tgt, :].T @ Z_t[common_tgt]
            clf = type(classifier)(**classifier.get_params()) if hasattr(classifier, "get_params") else classifier
            clf.fit(X_s_proj, y_source)
            preds = clf.predict(X_t_proj)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_target, preds)
            record["accuracy"] = acc
            if verbose:
                print(f"  MMD={final_mmd:.4f}  Loss={final_loss:.4f}  Acc={acc:.4f}")
        else:
            if verbose:
                print(f"  MMD={final_mmd:.4f}  Loss={final_loss:.4f}")

        records.append(record)

    eligible_records = [record for record in records
                        if record["eligible_for_label_free_selection"]]
    if not eligible_records:
        raise ValueError(
            "No lambda candidate has finite losses and non-collapsed target coordinates; "
            "run the explicit no-adaptation baseline instead"
        )
    best_mmd_rec = min(eligible_records, key=lambda r: r["final_mmd"])
    summary: Dict[str, Any] = {
        "results": records,
        "best_lambda_mmd": best_mmd_rec["lambda"],
        "selection_policy": (
            "lowest final full empirical MMD squared among candidates with finite "
            "losses and non-collapsed target coordinates"
        ),
        "rejected_candidates": [
            {"lambda": record["lambda"],
             "reason": ("nonfinite objective or coordinates"
                       if not record["coordinate_integrity"]["finite"]
                       or not np.isfinite([record["final_mmd"], record["final_loss"],
                                           record["final_regularization"]]).all()
                       else "near-zero target-coordinate variance")}
            for record in records if not record["eligible_for_label_free_selection"]
        ],
        "results_df": pd.DataFrame(records),
        "mmd_gamma": mmd_gamma,
        "_fit_results_by_lambda": fit_results,
    }
    if use_accuracy:
        best_acc_rec = max(eligible_records, key=lambda r: r["accuracy"])
        summary["best_lambda_accuracy"] = best_acc_rec["lambda"]

    return summary


def main() -> None:
    """Delegate direct execution to the canonical, configurable experiment CLI.

    Keeping one runner prevents this API module from silently using legacy
    cohort files and a second, incompatible set of training defaults.
    """
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from run_tvsn_experiment import parse_args, run

    run(parse_args())


if __name__ == "__main__":
    main()
