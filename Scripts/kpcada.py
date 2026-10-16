import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SOURCE_PATH = r"C:\Zmyfiles\codes\research\KPCA-DA\KPCA-DA\examples\data\TvsN\TCGA_BRCA_NvT.csv"
DEFAULT_TARGET_PATH = r"C:\Zmyfiles\codes\research\KPCA-DA\KPCA-DA\examples\data\TvsN\GSE45498_NvT.csv"


def load_expression_matrix(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and reshape the expression matrix from disk, returning data and gene names."""
    expr = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    expr = expr.transpose()
    expr.columns = expr.iloc[0]
    matrix = expr.iloc[1:, 0:-1].astype(float).values
    genes = expr.iloc[:, :-1].columns.to_numpy()
    return matrix.T, genes


def load_labels(csv_path: str) -> np.ndarray:
    """Extract sample labels from the last column of the expression CSV."""
    expr = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    expr = expr.transpose()
    expr.columns = expr.iloc[0]
    return expr.iloc[1:, -1].values


def standardize_datasets(X_source: np.ndarray, X_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize source and target datasets independently."""
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.fit_transform(X_target)
    return X_source_scaled, X_target_scaled


def find_common_gene_indices(genes_source: np.ndarray, genes_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find common genes between source and target datasets and return their indices."""
    common_genes = []
    for gene_s in genes_source:
        for gene_t in genes_target:
            if gene_s == gene_t:
                common_genes.append(gene_s)
    
    condition_source = np.isin(genes_source, common_genes)
    index_source = np.where(condition_source)[0]
    
    condition_target = np.isin(genes_target, common_genes)
    index_target = np.where(condition_target)[0]
    
    return index_source, index_target


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    pairwise_distances = torch.cdist(x, y) ** 2
    return torch.exp(-gamma * pairwise_distances)


def compute_median_gamma(Zs: torch.Tensor, Zt: torch.Tensor) -> float:
    """Set RBF gamma via the median pairwise-distance heuristic.

    Pools source and target points, computes all squared distances, and returns
    gamma = 1 / median(d²) so the kernel is neither saturated nor near-zero.
    """
    Z_all = torch.cat([Zs, Zt], dim=0)
    dists_sq = torch.cdist(Z_all, Z_all) ** 2
    median_sq = torch.median(dists_sq[dists_sq > 0])
    return (1.0 / median_sq.clamp(min=1e-8)).item()


def mmd_loss(Zs_common: torch.Tensor, Zt_common: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Modified MMD loss using only K_tt and K_st terms."""
    K_tt = rbf_kernel(Zt_common, Zt_common, gamma)
    K_st = rbf_kernel(Zs_common, Zt_common, gamma)
    
    # Modified MMD: abs(K_tt.mean() - 2 * K_st.mean())
    mmd = torch.abs(K_tt.mean() - 2 * K_st.mean())
    return mmd


def regularization_term(Z_t: torch.Tensor, X_t: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    K_t = rbf_kernel(X_t.T, X_t.T, gamma)
    A = Z_t.T @ K_t @ Z_t
    return torch.trace(A)


def fit_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    genes_source: Optional[np.ndarray] = None,
    genes_target: Optional[np.ndarray] = None,
    *,
    n_components: int = 2,
    kernel: str = "rbf",
    kpca_gamma: Optional[float] = None,
    mmd_gamma: float = 0.5,
    auto_gamma: bool = True,
    lambda_reg: float = 0.01,
    lr: float = 0.0001,
    epochs: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Align source and target domains using kernel PCA and MMD-based adaptation.

    Uses gene-based matching to find common features between datasets.
    When auto_gamma=True (default), the MMD kernel bandwidth is set automatically
    via the median pairwise-distance heuristic on the initial KPCA embeddings,
    replacing the fixed mmd_gamma value.
    """
    # Find common gene indices
    if genes_source is None or genes_target is None:
        raise ValueError("genes_source and genes_target must be provided for gene-based matching")
    
    common_source_idx, common_target_idx = find_common_gene_indices(genes_source, genes_target)

    if common_source_idx.size == 0 or common_target_idx.size == 0:
        raise ValueError("No common genes found between the source and target datasets.")

    # Note: Standardization is optional - commented out in modified version
    # X_source_scaled, X_target_scaled = standardize_datasets(X_source, X_target)
    # Using data as-is (already scaled/normalized externally if needed)
    X_source_scaled = X_source
    X_target_scaled = X_target

    # Apply KPCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=kpca_gamma)
    Z_s_np = kpca.fit_transform(X_source_scaled)
    Z_t_np = kpca.fit_transform(X_target_scaled)

    # Convert to tensors
    Z_s = torch.tensor(Z_s_np, dtype=torch.float32)
    Z_t = torch.tensor(Z_t_np, dtype=torch.float32, requires_grad=True)
    X_t_tensor = torch.tensor(X_target_scaled.T, dtype=torch.float32)

    # Auto-select MMD gamma from the initial embedding distances
    if auto_gamma:
        mmd_gamma = compute_median_gamma(
            Z_s[common_source_idx], Z_t[common_target_idx].detach()
        )
        if verbose:
            print(f"Auto-computed MMD gamma (median heuristic): {mmd_gamma:.6f}")

    optimizer = optim.SGD([Z_t], lr=lr)
    loss_history: List[float] = []

    for epoch in range(epochs):
        # Detach and re-enable gradients (as in modified version)
        Z_t = Z_t.detach().requires_grad_(True)
        optimizer = optim.SGD([Z_t], lr=lr)  # Reinitialize optimizer with new Z_t
        optimizer.zero_grad()

        # Extract common points
        Zs_common = Z_s[common_source_idx]
        Zt_common = Z_t[common_target_idx]

        # Compute losses
        mmd = mmd_loss(Zs_common, Zt_common, gamma=mmd_gamma)
        reg = regularization_term(Z_t, X_t_tensor, gamma=mmd_gamma)
        loss = mmd + lambda_reg * reg

        # Backpropagate
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, MMD: {mmd.item():.4f}, Reg: {reg.item():.4f}")

    return {
        "Z_s": Z_s.detach().numpy(),
        "Z_t": Z_t.detach().numpy(),
        "common_source_idx": common_source_idx,
        "common_target_idx": common_target_idx,
        "loss_history": loss_history,
        "kpca_model": kpca,
        "common_genes_count": len(common_source_idx),
        "mmd_gamma": mmd_gamma,
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
        kpca_gamma: Optional[float] = None,
        mmd_gamma: float = 0.5,
        auto_gamma: bool = True,
        lambda_reg: float = 0.01,
        lr: float = 0.0001,
        epochs: int = 100,
        verbose: bool = False,
    ):
        """Initialize KPCA_DA model.

        Parameters
        ----------
        n_components : int, default=2
            Number of components for KernelPCA
        kernel : str, default='rbf'
            Kernel type for KernelPCA
        kpca_gamma : float or None, default=None
            Gamma parameter for KernelPCA kernel
        mmd_gamma : float, default=0.5
            Gamma parameter for MMD kernel (ignored when auto_gamma=True)
        auto_gamma : bool, default=True
            Automatically set MMD gamma via median pairwise-distance heuristic
        lambda_reg : float, default=0.01
            Regularization strength
        lr : float, default=0.0001
            Learning rate for optimization
        epochs : int, default=100
            Number of optimization epochs
        verbose : bool, default=False
            Whether to print progress
        """
        self.n_components = n_components
        self.kernel = kernel
        self.kpca_gamma = kpca_gamma
        self.mmd_gamma = mmd_gamma
        self.auto_gamma = auto_gamma
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        
        self.kpca_model = None
        self.Z_s_ = None
        self.Z_t_ = None
        self.common_source_idx_ = None
        self.common_target_idx_ = None
        self.loss_history_ = None
    
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
            lr=self.lr,
            epochs=self.epochs,
            verbose=self.verbose,
        )
        
        # Store results
        self.Z_s_ = results["Z_s"]
        self.Z_t_ = results["Z_t"]
        self.common_source_idx_ = results["common_source_idx"]
        self.common_target_idx_ = results["common_target_idx"]
        self.loss_history_ = results["loss_history"]
        self.kpca_model = results["kpca_model"]
        
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
    ) -> Dict[str, Any]:
        """Grid-search lambda_reg and report MMD and accuracy for each value.

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
            n_components=self.n_components,
            kernel=self.kernel,
            kpca_gamma=self.kpca_gamma,
            mmd_gamma=self.mmd_gamma,
            auto_gamma=self.auto_gamma,
            lr=self.lr,
            epochs=self.epochs,
            verbose=self.verbose,
        )
        if y_source is not None and y_target is not None:
            self.lambda_reg = summary["best_lambda_accuracy"]
        else:
            self.lambda_reg = summary["best_lambda_mmd"]
        self.mmd_gamma = summary["mmd_gamma"]
        self.auto_gamma = False
        return summary

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted KPCA model.
        
        Parameters
        ----------
        X : np.ndarray
            Data to transform
            
        Returns
        -------
        Z : np.ndarray
            Transformed data
        """
        if self.kpca_model is None:
            raise ValueError("Model has not been fitted yet. Call fit_transform first.")
        
        return self.kpca_model.transform(X)


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
    kpca_gamma: Optional[float] = None,
    mmd_gamma: float = 0.5,
    auto_gamma: bool = True,
    lr: float = 0.0001,
    epochs: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Search over lambda_values and report final MMD and (optionally) accuracy for each.

    Parameters
    ----------
    lambda_values : sequence of float
        Candidate regularization strengths to evaluate.
    y_source, y_target : array-like or None
        Class labels for source and target. Required for accuracy reporting.
    classifier : sklearn-compatible estimator or None
        Trained/fitted on source KPCA embeddings and evaluated on adapted target.
        Defaults to KNeighborsClassifier(n_neighbors=5) when labels are supplied.

    Returns
    -------
    dict with keys:
        "results"     — list of dicts, one per lambda, with keys:
                          lambda, final_mmd, final_loss, accuracy (if labels given)
        "best_lambda_mmd"      — lambda with lowest final MMD
        "best_lambda_accuracy" — lambda with highest accuracy (if labels given)
        "results_df"  — pandas DataFrame summary
    """
    use_accuracy = y_source is not None and y_target is not None
    if use_accuracy and classifier is None:
        classifier = KNeighborsClassifier(n_neighbors=5)

    # Compute gamma once from the initial embeddings so it is identical across
    # all lambda runs. Re-fitting KPCA inside each fit_transform call can
    # produce small numerical differences (eigenvector sign flips, reordering)
    # that shift the embedding coordinates and change the median distance.
    if auto_gamma:
        kpca_init = KernelPCA(n_components=n_components, kernel=kernel, gamma=kpca_gamma)
        Z_s_init = kpca_init.fit_transform(X_source)
        Z_t_init = kpca_init.fit_transform(X_target)
        common_src_init, common_tgt_init = find_common_gene_indices(genes_source, genes_target)
        mmd_gamma = compute_median_gamma(
            torch.tensor(Z_s_init[common_src_init], dtype=torch.float32),
            torch.tensor(Z_t_init[common_tgt_init], dtype=torch.float32),
        )
        if verbose:
            print(f"Pre-computed MMD gamma (median heuristic, shared across all lambda): {mmd_gamma:.6f}")

    records = []
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
            lr=lr,
            epochs=epochs,
            verbose=verbose,
        )

        Z_s = res["Z_s"]
        Z_t = res["Z_t"]
        common_src = res["common_source_idx"]
        common_tgt = res["common_target_idx"]

        # Final MMD (without regularization weight)
        Zs_t = torch.tensor(Z_s[common_src], dtype=torch.float32)
        Zt_t = torch.tensor(Z_t[common_tgt], dtype=torch.float32)
        used_gamma = res["mmd_gamma"]   # actual gamma used during training
        final_mmd = mmd_loss(Zs_t, Zt_t, gamma=used_gamma).item()
        final_loss = res["loss_history"][-1]

        record: Dict[str, Any] = {
            "lambda": lam,
            "final_mmd": final_mmd,
            "final_loss": final_loss,
        }

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

    best_mmd_rec = min(records, key=lambda r: r["final_mmd"])
    summary: Dict[str, Any] = {
        "results": records,
        "best_lambda_mmd": best_mmd_rec["lambda"],
        "results_df": pd.DataFrame(records),
        "mmd_gamma": mmd_gamma,
    }
    if use_accuracy:
        best_acc_rec = max(records, key=lambda r: r["accuracy"])
        summary["best_lambda_accuracy"] = best_acc_rec["lambda"]

    return summary


def main() -> None:
    # Load data with gene names
    X_source, genes_source = load_expression_matrix(DEFAULT_SOURCE_PATH)
    X_target, genes_target = load_expression_matrix(DEFAULT_TARGET_PATH)

    # Using functional API
    results = fit_transform(
        X_source,
        X_target,
        genes_source=genes_source,
        genes_target=genes_target,
        n_components=2,
        kernel="rbf",
        kpca_gamma=0.5,
        mmd_gamma=0.5,
        lambda_reg=0.01,
        lr=0.0001,
        epochs=100,
        verbose=True,
    )

    print(f"\nCommon genes found: {results['common_genes_count']}")
    print("Optimized Z_t shape:", results["Z_t"].shape)
    
    # Using class-based API
    print("\n" + "="*50)
    print("Using KPCA_DA class API:")
    print("="*50)
    
    model = KPCA_DA(
        n_components=2,
        kernel="rbf",
        kpca_gamma=0.5,
        mmd_gamma=0.5,
        lambda_reg=0.01,
        lr=0.0001,
        epochs=100,
        verbose=True,
    )
    
    Z_s, Z_t = model.fit_transform(X_source, X_target, genes_source, genes_target)
    print(f"\nTransformed source shape: {Z_s.shape}")
    print(f"Transformed target shape: {Z_t.shape}")


if __name__ == "__main__":
    main()