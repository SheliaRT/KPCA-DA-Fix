"""Small audit demonstrations; does not train classifiers or change research code.

Run from any directory: python audit/reproduce_checks.py
Dependencies: numpy, pandas, scipy, scikit-learn, torch, matplotlib,
typing_extensions. The optional dataset checks require the local Data directory.
"""
from pathlib import Path
import ast
import csv
import importlib
import json
import math
import platform
import sys
import types
from collections import Counter

import numpy as np
import sklearn
import torch
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
sys.dont_write_bytecode = True
import kpcada

results = {"environment": {"python": platform.python_version(),
                          "numpy": np.__version__, "sklearn": sklearn.__version__,
                          "torch": torch.__version__}}

# D1: A lower implementation loss can mean a HIGHER conventional MMD.
s = torch.zeros((1, 2))
t0 = s.clone()
t1 = torch.tensor([[math.sqrt(math.log(2)), 0.]])
def full_mmd(s, t):
    return (kpcada.rbf_kernel(s, s).mean()
            + kpcada.rbf_kernel(t, t).mean()
            - 2 * kpcada.rbf_kernel(s, t).mean()).item()
results["mmd_counterexample"] = {
    "identical": {"code_loss": kpcada.mmd_loss(s, t0).item(), "full_mmd2": full_mmd(s, t0)},
    "separated": {"code_loss": kpcada.mmd_loss(s, t1).item(), "full_mmd2": full_mmd(s, t1)}}
assert full_mmd(s, t0) < full_mmd(s, t1)
assert kpcada.mmd_loss(s, t0) > kpcada.mmd_loss(s, t1)

# D2: Minimize the actual regularizer alone for a fixed target kernel.
z = torch.tensor([[-1., 0.], [1., 0.]], requires_grad=True)
x = torch.tensor([[0., 10.]])  # regularization_term transposes this to gene rows
r0 = kpcada.regularization_term(z, x)
grad, = torch.autograd.grad(r0, z)
z1 = (z - .1 * grad).detach()
results["regularizer_step"] = {"trace_before": r0.item(),
    "trace_after": kpcada.regularization_term(z1, x).item(),
    "variance_before": z.detach().numpy().var(axis=0).sum().item(),
    "variance_after": z1.numpy().var(axis=0).sum().item()}
assert results["regularizer_step"]["variance_after"] < results["regularizer_step"]["variance_before"]

# D3: Load the actual image module without its package __init__, which eagerly
# imports optional Grad-CAM dependencies. No image/KPCA methods are replaced.
pkg = types.ModuleType("pyDeepInsightv3")
pkg.__path__ = [str(ROOT / "Scripts" / "pyDeepInsightv3")]
sys.modules["pyDeepInsightv3"] = pkg
DA = importlib.import_module("pyDeepInsightv3.image_transformer_new").DomainAdaptiveImageTransformer
p = np.array([[0., 0.], [1., 0.], [0., 1.], [.3, .6]])
mapper = DA(kpca_da_params={"epochs": 1}, pixels=(31, 31))
mapper._xrot_source = p
mapper._xrot_target = p * 2 + np.array([10., 20.])
mapper._calculate_coords()
results["coordinate_rescaling"] = {
    "embedding_centroid_distance": float(np.linalg.norm(p.mean(0) - mapper._xrot_target.mean(0))),
    "pixel_maps_identical": bool(np.array_equal(mapper.coords_source(), mapper.coords_target()))}
assert results["coordinate_rescaling"]["pixel_maps_identical"]

# D4: Reproduce the notebook's post-ICA translation algebra.
source = np.array([[0., 0.], [1., 1.]])
target = source + [3., 4.]
d0 = np.linalg.norm(source.mean(0) - target.mean(0))
shifted = target + .8 * (source.mean(0) - target.mean(0))
d1 = np.linalg.norm(source.mean(0) - shifted.mean(0))
results["posthoc_centroid_shift"] = {"before": float(d0), "after": float(d1),
                                      "reduction_percent": float(100 * (1 - d1/d0))}
assert np.isclose(d1, .2 * d0)

# D5: The transform method returns the unadapted target KPCA projection.
rng = np.random.default_rng(4)
xs, xt = rng.normal(size=(7, 5)), rng.normal(size=(7, 4))
genes = np.array([f"g{i}" for i in range(7)])
model = kpcada.KPCA_DA(kpca_gamma=.2, auto_gamma=False, mmd_gamma=.5,
                       lambda_reg=.1, lr=.01, epochs=5)
zs, zt = model.fit_transform(xs, xt, genes, genes)
plain = model.kpca_model.transform(xt)
via_api = model.transform(xt)
results["transform_api"] = {"matches_unadapted_kpca": bool(np.allclose(via_api, plain)),
    "max_difference_from_optimized_target": float(np.max(np.abs(via_api - zt)))}
assert results["transform_api"]["max_difference_from_optimized_target"] > 1e-6

# D6: KPCA score convention differs from the displayed paper equations 7-8.
kp = KernelPCA(n_components=2, kernel="rbf", gamma=.2)
z = kp.fit_transform(xs)
paper_z = kp.eigenvectors_ / np.sqrt(kp.eigenvalues_)
results["kpca_equation"] = {
    "code_matches_V_times_sqrt_eigenvalues": bool(np.allclose(z, kp.eigenvectors_ * np.sqrt(kp.eigenvalues_))),
    "code_matches_paper_V_divided_by_sqrt_eigenvalues": bool(np.allclose(z, paper_z))}

# D7: Single-endpoint accounting versus tune_lambda's mixed endpoints.
summary = kpcada.tune_lambda(xs, xt, genes, genes, lambda_values=[.1],
    kpca_gamma=.2, auto_gamma=False, mmd_gamma=.5, lr=.01, epochs=5)
record = summary["results"][0]
end_mmd = kpcada.mmd_loss(torch.tensor(zs), torch.tensor(zt), gamma=.5).item()
end_reg = kpcada.regularization_term(torch.tensor(zt), torch.tensor(xt.T, dtype=torch.float32), gamma=.5).item()
results["reported_final_loss"] = {"reported": record["final_loss"],
    "recomputed_at_returned_embeddings": end_mmd + .1 * end_reg}

# Read CSV metadata using exactly the notebooks' gene-row/last-label-row layout.
files = ["TvsN/TCGA_BRCA_NvT.csv", "TvsN/GSE45498_NvT.csv",
    "TCGAvsDesmd/modified_TCGA_BRCA_log2.csv", "TCGAvsDesmd/filtered_desmd_average.csv",
    "cptacvssmc/smc_labels.csv", "cptacvssmc/cptac_labels.csv"]
metadata = {}
gene_names = {}
for rel in files:
    path = ROOT / "Data" / rel
    if not path.exists():
        continue
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = []
        for row in reader:
            names.append(row[0])
            last = row
    metadata[rel] = {"samples": len(header)-1, "gene_rows": len(names)-1,
        "unique_gene_names": len(set(names[:-1])), "label_row_name": names[-1],
        "labels": dict(Counter(last[1:]))}
    gene_names[rel] = names[:-1]
    if rel == "TCGAvsDesmd/filtered_desmd_average.csv":
        # Track original sample IDs through exactly the oversample-then-split
        # procedure used in claude_code cells 3-4.
        y = np.array(last[1:]); ids = np.arange(len(y))
        classes, counts = np.unique(y, return_counts=True)
        all_ids, all_labels = [], []
        for cls in classes:
            chosen = ids[y == cls]
            if len(chosen) < counts.max():
                chosen = resample(chosen, replace=True, n_samples=int(counts.max()), random_state=42)
            all_ids.extend(chosen.tolist()); all_labels.extend([cls] * len(chosen))
        a, b = train_test_split(np.array(all_ids), test_size=.3, random_state=42, stratify=all_labels)
        overlap = set(a) & set(b)
        results["oversample_before_split"] = {"original_samples": len(y), "after_resampling": len(all_ids),
            "adapt_rows": len(a), "test_rows": len(b),
            "original_patient_ids_in_both": len(overlap),
            "test_rows_with_patient_in_adapt": sum(i in overlap for i in b)}
results["dataset_metadata"] = metadata
pair_checks = []
for a, b in [(files[0], files[1]), (files[2], files[3]), (files[4], files[5])]:
    if a not in gene_names or b not in gene_names:
        continue
    common = set(gene_names[a]) & set(gene_names[b])
    ga = [g for g in gene_names[a] if g in common]
    gb = [g for g in gene_names[b] if g in common]
    pair_checks.append({"source": a, "target": b, "shared_names": len(common),
        "source_rows": len(ga), "target_rows": len(gb),
        "position_matches": sum(x == y for x, y in zip(ga, gb)),
        "first_mismatched_pairs": [[x, y] for x, y in zip(ga, gb) if x != y][:3]})
results["combat_gene_order"] = pair_checks

# D8: Parse all source without importing/training the notebook dependencies.
syntax = []
for path in sorted((ROOT / "Scripts").rglob("*.py")):
    ast.parse(path.read_text(), filename=str(path))
    syntax.append(str(path.relative_to(ROOT)))
nb_count = 0
for path in sorted((ROOT / "Scripts").glob("*.ipynb")):
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb["cells"], 1):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{path.name}:cell{i}")
    nb_count += 1
results["syntax"] = {"python_files": len(syntax), "notebooks": nb_count, "all_code_cells_parse": True}
print(json.dumps(results, indent=2, sort_keys=True))
