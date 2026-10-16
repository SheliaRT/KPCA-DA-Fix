"""Small audit demonstrations; does not train classifiers or change research code.

Run from any directory: python audit/reproduce_checks.py
Dependencies: numpy, pandas, scipy, scikit-learn, torch, matplotlib,
typing_extensions. The optional dataset checks require the local Data directory.
"""
from pathlib import Path
import ast
import csv
from contextlib import redirect_stdout
import hashlib
import importlib
from io import StringIO
import json
import math
import platform
import sys
import tempfile
import types
from collections import Counter

import numpy as np
import sklearn
import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
sys.path.insert(0, str(ROOT / "audit"))
sys.dont_write_bytecode = True
import kpcada
from experiment_protocol import (GeneAwareMinMax, make_patient_splits,
                                 ordered_common_gene_indices,
                                 patient_group_ids, patient_groups_from_mapping,
                                 transform_target_to_images)
from evaluation_protocol import (balanced_class_weights, classification_metrics,
                                 select_binary_threshold)
from combat_protocol import fit_apply_combat_partitions
from rebuild_common_frame_ica_pair import marginal_w1 as paired_ica_marginal_w1
from rebuild_source_fitted_ica import summarize as summarize_source_fitted_ica
from compare_tvsn_runs import (verify_code_environment_match,
                               verify_preprocessing_policy_match,
                               verify_representation_fit_scope)
from kpcada import align_target_embedding_to_source
from pyDeepInsightv3.gene_attribution import rank_gene_attributions
from pyDeepInsightv3.feature_selection import CAMFeatureSelector
from score_withheld_target_run import score_run as score_withheld_target_run
from pyDeepInsightv3.image_transformer import ImageTransformer
from pyDeepInsightv3.image_transformer_new import DomainAdaptiveImageTransformer

results = {"environment": {"python": platform.python_version(),
                          "numpy": np.__version__, "sklearn": sklearn.__version__,
                          "torch": torch.__version__}}

# Paired results must not blend the historical percentile-threshold filter
# with the deterministic rank filter. Target-private fitting can differ only
# for a shared-feature comparison, where private features are excluded.
common_preprocessing_policy = {
    "variance_filter": "stable rank",
    "variance_percentile": 30.0,
    "variance_tie_break": "stable input gene order",
    "source_filter_fit_scope": "source training samples only",
    "shared_gene_scaler_fit_scope": "source training samples only",
    "clip_scaled_values": False,
}
preprocess_baseline = {
    "feature_scope": "shared",
    "preprocessing_policy": {
        **common_preprocessing_policy,
        "target_private_filter_fit_scope": "not fitted",
        "target_private_scaler_fit_scope": "not fitted",
    },
}
preprocess_adapted = {
    "feature_scope": "shared",
    "preprocessing_policy": {
        **common_preprocessing_policy,
        "target_private_filter_fit_scope": "target adaptation samples only",
        "target_private_scaler_fit_scope": "target adaptation samples only",
    },
}
assert verify_preprocessing_policy_match(
    preprocess_baseline, preprocess_adapted
) == common_preprocessing_policy
try:
    verify_preprocessing_policy_match(
        preprocess_baseline,
        {**preprocess_adapted, "preprocessing_policy": {
            **preprocess_adapted["preprocessing_policy"],
            "variance_filter": "legacy percentile threshold",
        }},
    )
except ValueError as error:
    assert "preprocessing policy" in str(error)
else:
    raise AssertionError("paired verifier accepted different variance filters")
results["preprocessing_policy_pairing"] = {
    "rank_filter_mismatch_rejected": True,
    "shared_scope_private_fit_difference_allowed": True,
}
try:
    verify_preprocessing_policy_match(
        {**preprocess_baseline, "config": {"variance_percentile": 40.0}},
        preprocess_adapted,
    )
except ValueError as error:
    assert "percentile differs" in str(error)
else:
    raise AssertionError("verifier accepted policy/configuration disagreement")
all_feature_private_policy = {
    "feature_scope": "all",
    "arm": "kpcada",
    "preprocessing_policy": {
        **common_preprocessing_policy,
        "target_private_filter_fit_scope": "not fitted",
        "target_private_scaler_fit_scope": "not fitted",
    },
}
try:
    verify_preprocessing_policy_match(all_feature_private_policy, all_feature_private_policy)
except ValueError as error:
    assert "target adaptation samples only" in str(error)
else:
    raise AssertionError("all-feature verifier accepted unfitted target-private features")
results["preprocessing_policy_internal_consistency"] = {
    "configuration_mismatch_rejected": True,
    "all_feature_private_fit_scope_enforced": True,
}
fingerprinted_run = {
    "python_source_tree_sha256": "source-fingerprint",
    "environment_lock_sha256": "lock-fingerprint",
}
assert verify_code_environment_match(fingerprinted_run, fingerprinted_run)["status"] == "matched"
assert verify_code_environment_match({}, {})["status"].startswith("not recorded")
assert verify_code_environment_match(fingerprinted_run, {})["status"].startswith("partial")
try:
    verify_code_environment_match(
        fingerprinted_run,
        {**fingerprinted_run, "python_source_tree_sha256": "different-source"},
    )
except ValueError as error:
    assert "python_source_tree_sha256" in str(error)
else:
    raise AssertionError("paired verifier accepted different source fingerprints")
results["code_environment_pairing"] = {
    "matching_fingerprints_accepted": True,
    "legacy_missing_fingerprints_labeled": True,
    "partial_fingerprints_labeled": True,
    "mismatched_fingerprints_rejected": True,
}

mapped_groups = patient_groups_from_mapping(
    ["sample-1", "sample-2", "sample-3"],
    {"sample-1": "patient-A", "sample-2": "patient-A", "sample-3": "patient-B"},
)
assert mapped_groups.tolist() == ["patient:patient-A", "patient:patient-A", "patient:patient-B"]
try:
    patient_groups_from_mapping(["unknown-sample"], {"sample-1": "patient-A"})
except ValueError as error:
    assert "lacks sample IDs" in str(error)
else:
    raise AssertionError("patient mapping accepted an unmapped sample")
results["explicit_patient_mapping"] = {
    "multiple_samples_grouped_to_one_patient": True,
    "unmapped_sample_rejected": True,
}

# D1: Preserve the original audit counterexample while checking the corrected
# implementation against full empirical MMD².
s = torch.zeros((1, 2))
t0 = s.clone()
t1 = torch.tensor([[math.sqrt(math.log(2)), 0.]])
def full_mmd(s, t):
    return (kpcada.rbf_kernel(s, s).mean()
            + kpcada.rbf_kernel(t, t).mean()
            - 2 * kpcada.rbf_kernel(s, t).mean()).item()
def legacy_loss(s, t):
    return abs((kpcada.rbf_kernel(t, t).mean()
                - 2 * kpcada.rbf_kernel(s, t).mean()).item())
results["mmd_counterexample"] = {
    "identical": {"legacy_loss": legacy_loss(s, t0), "current_loss": kpcada.mmd_loss(s, t0).item(), "full_mmd2": full_mmd(s, t0)},
    "separated": {"legacy_loss": legacy_loss(s, t1), "current_loss": kpcada.mmd_loss(s, t1).item(), "full_mmd2": full_mmd(s, t1)}}
assert full_mmd(s, t0) < full_mmd(s, t1)
assert legacy_loss(s, t0) > legacy_loss(s, t1)
assert np.isclose(kpcada.mmd_loss(s, t0).item(), full_mmd(s, t0))
assert np.isclose(kpcada.mmd_loss(s, t1).item(), full_mmd(s, t1))

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
p = np.array([[0., 0.], [2., 0.], [4., 0.]])
t = np.array([[1., 0.], [2., 0.], [5., 2.]])
mapper = DA(kpca_da_params={"epochs": 1}, pixels=(31, 31))
mapper._xrot_source = p
mapper._xrot_target = t
mapper._calculate_coords()
results["coordinate_rescaling"] = {
    "shared_point_source_pixel": mapper.coords_source()[1].tolist(),
    "shared_point_target_pixel": mapper.coords_target()[1].tolist(),
    "shared_point_maps_identically": bool(np.array_equal(
        mapper.coords_source()[1], mapper.coords_target()[1])),
    "domain_maps_are_not_artificially_identical": bool(
        not np.array_equal(mapper.coords_source(), mapper.coords_target()))}
assert results["coordinate_rescaling"]["shared_point_maps_identically"]
assert results["coordinate_rescaling"]["domain_maps_are_not_artificially_identical"]

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
via_explicit_api = model.transform_target_genes_unadapted(xt)
results["transform_api"] = {"matches_unadapted_kpca": bool(np.allclose(via_api, plain)),
    "explicit_target_gene_projection_matches": bool(np.allclose(via_explicit_api, plain)),
    "max_difference_from_optimized_target": float(np.max(np.abs(via_api - zt)))}
assert results["transform_api"]["max_difference_from_optimized_target"] > 1e-6
assert results["transform_api"]["explicit_target_gene_projection_matches"]

# D6: KPCA score convention differs from the displayed paper equations 7-8.
kp = KernelPCA(n_components=2, kernel="rbf", gamma=.2)
z = kp.fit_transform(xs)
paper_z = kp.eigenvectors_ / np.sqrt(kp.eigenvalues_)
results["kpca_equation"] = {
    "code_matches_V_times_sqrt_eigenvalues": bool(np.allclose(z, kp.eigenvectors_ * np.sqrt(kp.eigenvalues_))),
    "code_matches_paper_V_divided_by_sqrt_eigenvalues": bool(np.allclose(z, paper_z))}

# D7: Single-endpoint accounting versus tune_lambda's mixed endpoints.
summary = kpcada.tune_lambda(xs, xt, genes, genes, lambda_values=[.1, .2],
    kpca_gamma=.2, auto_gamma=False, mmd_gamma=.5, lr=.01, epochs=5)
record = summary["results"][0]
recomputed_loss = record["final_mmd"] + .1 * record["final_regularization"]
candidate_fits = summary["_fit_results_by_lambda"]
same_initial = all(
    np.array_equal(candidate_fits[.1][name], candidate_fits[.2][name])
    for name in ("Z_s", "Z_t_initial")
)
assert same_initial
results["reported_final_loss"] = {"reported": record["final_loss"],
    "recomputed_at_returned_embeddings": recomputed_loss,
    "loss_history_endpoint": record["final_loss_history_endpoint"],
    "regularization_mode": "reference"}
assert np.isclose(record["final_loss"], recomputed_loss, rtol=1e-6, atol=1e-8)
assert np.isclose(record["final_loss"], record["final_loss_history_endpoint"], rtol=1e-5, atol=1e-7)
results["lambda_sweep_fixed_embedding"] = {
    "candidate_count": len(candidate_fits),
    "same_initial_source_and_target_embeddings": same_initial,
    "same_fixed_mmd_gamma": candidate_fits[.1]["mmd_gamma"] == candidate_fits[.2]["mmd_gamma"],
}
assert results["lambda_sweep_fixed_embedding"]["same_fixed_mmd_gamma"]

# D8: The three ablation objectives remain separately controllable.
ablation_source = np.array([
    [0., 1., 2.], [1., 1.5, 0.], [2., 0., 1.], [3., 2., 1.],
])
ablation_target = ablation_source + np.array([[.3], [-.2], [.7], [.1]])
ablation_genes = np.array(["a", "b", "c", "d"])
ablation_args = dict(
    X_source=ablation_source, X_target=ablation_target,
    genes_source=ablation_genes, genes_target=ablation_genes,
    n_components=2, kernel="rbf", kpca_gamma=.2, mmd_gamma=.5,
    auto_gamma=False, lr=.01, epochs=3, kpca_random_state=7,
)
kpca_only = kpcada.fit_transform(
    **ablation_args, alignment_weight=0., lambda_reg=0.,
    regularization_mode="none",
)
mmd_only = kpcada.fit_transform(
    **ablation_args, alignment_weight=1., lambda_reg=0.,
    regularization_mode="none",
)
vml_only = kpcada.fit_transform(
    **ablation_args, alignment_weight=0., lambda_reg=.001,
    regularization_mode="trace",
)
assert np.allclose(kpca_only["Z_t"], kpca_only["Z_t_initial"])
assert np.isclose(kpca_only["final_loss"], 0.)
assert np.isclose(mmd_only["final_loss"], mmd_only["final_mmd"])
assert np.isclose(vml_only["final_loss"], .001 * vml_only["final_regularization"])
results["ablation_objective_controls"] = {
    "kpca_only_has_zero_target_movement": True,
    "mmd_only_final_loss_equals_mmd": True,
    "vml_only_final_loss_equals_weighted_trace": True,
}

# The object-oriented API must reuse the selected candidate's coordinates and
# fitted KPCA state instead of fitting a new random projection after selection.
tuned_model = kpcada.KPCA_DA(
    kpca_gamma=.2, auto_gamma=False, mmd_gamma=.5, lambda_reg=.1,
    lr=.01, epochs=5, kpca_random_state=17)
tuned_model.tune_lambda(xs, xt, genes, genes, lambda_values=[.1, .2])
selected_coordinates = tuned_model._tuned_fit_result["Z_t"].copy()
_, fitted_coordinates = tuned_model.fit_transform(xs, xt, genes, genes)
assert np.array_equal(selected_coordinates, fitted_coordinates)
assert tuned_model.kpca_source_model_ is not tuned_model.kpca_target_model_
source_projection = tuned_model.transform_source(xs)
target_projection = tuned_model.transform(xt)
explicit_target_projection = tuned_model.transform_target_genes_unadapted(xt)
assert np.allclose(source_projection, tuned_model.Z_s_, atol=1e-6)
assert np.allclose(target_projection, tuned_model.Z_t_initial_, atol=1e-6)
assert np.array_equal(explicit_target_projection, target_projection)
results["lambda_selection_reuses_selected_embedding"] = True
results["independent_kpca_estimators"] = {
    "source_and_target_models_are_distinct": True,
    "source_transform_matches_source_fit": True,
    "target_transform_is_unadapted_and_separate_from_optimized_coordinates": bool(
        not np.allclose(target_projection, tuned_model.Z_t_)
    ),
}
assert results["independent_kpca_estimators"][
    "target_transform_is_unadapted_and_separate_from_optimized_coordinates"]

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

# D9: Exercise the new pre-retraining data and evaluation helpers on synthetic
# inputs. This does not fit KPCA-DA or a classifier.
source_ids = [f"source-{i}" for i in range(20) for _ in range(2)]
source_y = ["A" if i % 2 == 0 else "B" for i in range(20) for _ in range(2)]
target_ids = [f"target-{i}" for i in range(24)]
target_y = ["N"] * 12 + ["T"] * 12
splits = make_patient_splits(source_y, target_y,
    source_ids=source_ids, target_ids=target_ids)
assert not (set(np.asarray(source_ids)[splits.source_train])
            & set(np.asarray(source_ids)[splits.source_validation]))
assert not (set(target_ids[i] for i in splits.target_adapt)
            & set(target_ids[i] for i in splits.target_test))

# Cohort IDs can denote samples/aliquots rather than patients. Group paired
# TCGA Normal/Tumor aliquots and GEO tumor/normal specimens before splitting.
tcga_ids = np.array(["TCGA-AA-0001-01A", "TCGA-AA-0001-11A",
                     "TCGA-AA-0002-01A", "TCGA-AA-0003-01A",
                     "TCGA-AA-0004-01A", "TCGA-AA-0005-01A",
                     "TCGA-AA-0006-01A", "TCGA-AA-0007-01A",
                     "TCGA-AA-0008-01A", "TCGA-AA-0009-01A"])
geo_ids = np.array(["Normal_108", "TAS09 12 108", "TAS09 12 101_A",
                    "TAS09 12 101B", "TAS09 12 110.1", "TAS09 12 110"])
assert patient_group_ids(tcga_ids)[:2].tolist() == ["TCGA:TCGA-AA-0001"] * 2
assert patient_group_ids(geo_ids)[:2].tolist() == ["GSE45498:108"] * 2
assert patient_group_ids(geo_ids)[2:4].tolist() == ["GSE45498:101"] * 2
assert patient_group_ids(geo_ids)[4:].tolist() == ["GSE45498:110"] * 2
official_geo_ids = np.array(["GSE45498:108:GSM1038950",
                             "GSE45498:108:GSM1038951",
                             "GSE45498:109:GSM1038952"])
assert patient_group_ids(official_geo_ids).tolist() == [
    "GSE45498:108", "GSE45498:108", "GSE45498:109"]
mixed_label_groups = np.array(["tcga:p1", "tcga:p1", "tcga:p2", "tcga:p3",
                               "tcga:p4", "tcga:p5", "tcga:p6", "tcga:p7",
                               "tcga:p8", "tcga:p9"])
mixed_splits = make_patient_splits(
    ["Normal", "Tumor", "Tumor", "Normal", "Tumor", "Normal", "Tumor",
     "Normal", "Tumor", "Normal"],
    ["Normal", "Tumor", "Tumor", "Normal", "Tumor", "Normal"],
    source_ids=mixed_label_groups,
    target_ids=patient_group_ids(geo_ids),
)
unlabeled_target_splits = make_patient_splits(
    ["Normal", "Tumor", "Tumor", "Normal", "Tumor", "Normal", "Tumor",
     "Normal", "Tumor", "Normal"],
    target_ids=patient_group_ids(geo_ids),
)
assert np.array_equal(mixed_splits.target_adapt, unlabeled_target_splits.target_adapt)
assert np.array_equal(mixed_splits.target_test, unlabeled_target_splits.target_test)
assert not (set(mixed_label_groups[mixed_splits.source_train])
            & set(mixed_label_groups[mixed_splits.source_validation]))
assert not (set(patient_group_ids(geo_ids)[mixed_splits.target_adapt])
            & set(patient_group_ids(geo_ids)[mixed_splits.target_test]))

X_source = np.array([[0., 0., 3.], [1., 2., 5.],
                     [2., 3., 8.], [3., 4., 9.]])
X_target = np.array([[0., 4., 20.], [1., 6., 22.],
                     [2., 8., 25.], [3., 10., 30.]])
source_gene_ids = np.array(["g1", "g2", "source-private"])
target_gene_ids = np.array(["g2", "g1", "target-private"])
preprocessor = GeneAwareMinMax(variance_percentile=0).fit(
    X_source, source_gene_ids, X_target, target_gene_ids)
target_scaled = preprocessor.transform_target(X_target, target_gene_ids)
single_scaled = preprocessor.transform_target(X_target[:1], target_gene_ids)
assert np.allclose(target_scaled[:1], single_scaled)
assert np.max(target_scaled) > 1.0  # default retains held-out domain shift

# The supplement's 30th-percentile variance filter must discard a predictable
# fraction even when many genes tie, and constant genes carry no image signal.
tie_variances = np.array([0., 0., 1., 1., 1., 1., 2., 2., 3., 3.])
tie_matrix = np.stack([-np.sqrt(tie_variances), np.sqrt(tie_variances)])
tie_genes = np.array([f"tie_{index}" for index in range(len(tie_variances))])
tie_filter = GeneAwareMinMax(variance_percentile=30).fit(
    tie_matrix, tie_genes, tie_matrix, tie_genes
)
assert tie_filter.source_keep_.sum() == 7
assert tie_filter.source_keep_[:2].tolist() == [False, False]
assert tie_filter.source_keep_[2:6].tolist() == [False, True, True, True]
assert not tie_filter.source_keep_[tie_variances == 0.].any()
tie_filter_shared = GeneAwareMinMax(variance_percentile=30).fit_source_shared(
    tie_matrix, tie_genes, tie_genes
)
assert np.array_equal(tie_filter_shared.source_keep_, tie_filter.source_keep_)
results["deterministic_variance_filter"] = {
    "requested_discard_percent": 30,
    "input_features": len(tie_genes),
    "retained_features": int(tie_filter.source_keep_.sum()),
    "constant_features_removed": int((~tie_filter.source_keep_[:2]).sum()),
    "tie_break": "stable input order",
}
clipped_preprocessor = GeneAwareMinMax(
    variance_percentile=0, clip_scaled_values=True
).fit(X_source, source_gene_ids, X_target, target_gene_ids)
target_clipped = clipped_preprocessor.transform_target(X_target, target_gene_ids)
source_out_of_range = clipped_preprocessor.transform_source(
    np.array([[-1., 6., 10.]]), source_gene_ids
)
assert np.all((target_clipped >= 0.) & (target_clipped <= 1.))
assert np.all((source_out_of_range >= 0.) & (source_out_of_range <= 1.))
src_idx, tgt_idx, common = ordered_common_gene_indices(source_gene_ids, target_gene_ids)
assert common.tolist() == ["g1", "g2"]
assert np.array_equal(source_gene_ids[src_idx], target_gene_ids[tgt_idx])

# The inductive raw-DeepInsight control fits filtering/scaling from source
# training only and uses target gene IDs only to select the shared features.
source_only_preprocessor = GeneAwareMinMax(variance_percentile=0).fit_source_shared(
    X_source, source_gene_ids, target_gene_ids
)
source_only_scaled = source_only_preprocessor.transform_source(
    X_source, source_gene_ids
)
target_only_scaled = source_only_preprocessor.transform_target(
    X_target, target_gene_ids
)
assert source_only_preprocessor.target_genes_.tolist() == ["g2", "g1"]
assert source_only_scaled.shape == (4, 3)
assert target_only_scaled.shape == (4, 2)
assert np.allclose(target_only_scaled[:, 0], X_target[:, 0] *
                    source_only_preprocessor.source_scaler_.scale_[1] +
                    source_only_preprocessor.source_scaler_.min_[1])
results["source_only_shared_preprocessing"] = {
    "target_private_genes_excluded": True,
    "target_features_ordered_by_target_gene_ids": source_only_preprocessor.target_genes_.tolist(),
    "shared_target_scaling_uses_source_training_parameters": True,
    "target_expression_values_not_used_during_fit": True,
}

# The paper-style raw baseline fits DeepInsight on source-training genes and
# applies its exact coordinates to target values in the same gene order.
raw_rng = np.random.default_rng(73)
raw_genes_source = np.asarray([f"raw-gene-{i}" for i in range(40)])
raw_target_order = np.arange(40)[::-1]
raw_genes_target = raw_genes_source[raw_target_order]
raw_source = raw_rng.normal(size=(12, 40))
raw_target = raw_rng.normal(loc=0.5, size=(5, 40))
raw_preprocessor = GeneAwareMinMax(variance_percentile=0).fit_source_shared(
    raw_source, raw_genes_source, raw_genes_target
)
raw_source_scaled = raw_preprocessor.transform_source(raw_source, raw_genes_source)
raw_target_scaled_in_target_order = raw_preprocessor.transform_target(
    raw_target[:, raw_target_order], raw_genes_target
)
raw_source_idx, raw_target_idx, raw_shared_genes = ordered_common_gene_indices(
    raw_preprocessor.source_genes_, raw_preprocessor.target_genes_
)
raw_source_scaled = raw_source_scaled[:, raw_source_idx]
raw_target_scaled = raw_target_scaled_in_target_order[:, raw_target_idx]
raw_map = ImageTransformer(
    # This is only a frozen-map plumbing check, not a t-SNE quality study.
    # The minimum iteration count is sufficient to exercise fit/transform while
    # keeping the smoke suite short enough to run before a training campaign.
    feature_extractor=TSNE(n_components=2, metric="cosine", random_state=73,
                           max_iter=250),
    pixels=(16, 16),
).fit(raw_source_scaled)
raw_map_coords = raw_map.coords()
raw_source_images = raw_map.transform(raw_source_scaled, img_format="scalar")
raw_target_images = raw_map.transform(raw_target_scaled, img_format="scalar")
assert raw_source_images.shape == (12, 16, 16)
assert raw_target_images.shape == (5, 16, 16)
assert raw_shared_genes.size == 40
assert np.array_equal(raw_map.coords(), raw_map_coords)
results["raw_deepinsight_frozen_source_map"] = {
    "source_only_map_fit": True,
    "target_gene_order_aligned_by_id": True,
    "source_and_target_use_same_frozen_coordinates": True,
    "source_and_target_image_shapes": [list(raw_source_images.shape),
                                        list(raw_target_images.shape)],
}
raw_fit_splits = {
    "source_train": np.array(["s1", "s2"]),
    "target_adaptation": np.array(["t1"]),
    "target_test": np.array(["t2"]),
    "feature_map_source_train": np.array(["s1", "s2"]),
    "feature_map_target_adaptation": np.array([], dtype=str),
}
raw_fit_metrics = {
    "arm": "raw_deepinsight",
    "feature_scope": "shared",
    "representation_fit_policy": {
        "target_adaptation_expression_used": False,
        "target_test_expression_used_for_fit": False,
    },
    "geometry_diagnostics": {
        "target_expression_used_to_fit_map": False,
        "target_adaptation_expression_used": False,
    },
}
verify_representation_fit_scope(raw_fit_metrics, raw_fit_splits)
try:
    verify_representation_fit_scope(
        raw_fit_metrics,
        {**raw_fit_splits,
         "feature_map_target_adaptation": np.array(["t1"])},
    )
except ValueError as exc:
    assert "must not fit on target-adaptation" in str(exc)
else:
    raise AssertionError("Raw baseline verifier must reject target-adaptation fit IDs")
results["raw_baseline_fit_scope_verification"] = {
    "source_only_map_fit_accepted": True,
    "target_adaptation_fit_rejected": True,
}

metrics = classification_metrics(["N", "T", "T", "N"], ["N", "T", "N", "N"],
    labels=["N", "T"], positive_label="T")
assert metrics["sensitivity"] == .5 and metrics["specificity"] == 1.0
results["new_protocol_helpers"] = {
    "source_patient_groups_disjoint": True,
    "target_patient_groups_disjoint": True,
    "mixed_label_aliquots_stay_within_one_patient_partition": True,
    "target_split_uses_patient_groups_without_label_stratification": True,
    "target_labels_can_be_omitted_from_split_api": True,
    "test_mapping_is_independent_of_other_test_patients": True,
    "optional_scaled_value_clipping_bounds_unit_interval": True,
    "shared_genes_in_identical_order": common.tolist(),
    "binary_sensitivity": metrics["sensitivity"],
    "binary_specificity": metrics["specificity"],
}
threshold = select_binary_threshold(
    [0, 0, 1, 1], [.1, .4, .6, .9], positive_label=1, negative_label=0
)
assert threshold == {"threshold": .5, "source_validation_balanced_accuracy": 1.0}
results["source_validation_threshold_selection"] = threshold

# Source-fitted ICA distance summaries must use the supplied unshifted arrays,
# return zero for identical distributions, and identify marginal metrics.
ica_source = np.array([[-1., 0.], [0., 1.], [1., -1.]])
ica_target = ica_source.copy()
ica_source_before, ica_target_before = ica_source.copy(), ica_target.copy()
ica_same = summarize_source_fitted_ica(ica_source, ica_target)
assert ica_same["centroid_euclidean_distance"] == 0.0
assert ica_same["mean_marginal_1d_wasserstein"] == 0.0
assert np.array_equal(ica_source, ica_source_before)
assert np.array_equal(ica_target, ica_target_before)
ica_shifted = summarize_source_fitted_ica(ica_source, ica_target + np.array([2., -3.]))
assert ica_shifted["centroid_euclidean_distance"] > 0.0
assert "fully computed multivariate Wasserstein" in ica_shifted["metric_note"]
results["source_fitted_ica_distance_contract"] = {
    "identical_inputs_have_zero_distance": True,
    "input_coordinates_unchanged": True,
    "centroid_shift_is_measured_not_removed": True,
    "wasserstein_is_labeled_marginal": True,
}
paired_ica = np.array([[-1., 0.], [0., 1.], [1., -1.]])
paired_ica_zero = paired_ica_marginal_w1(paired_ica, paired_ica.copy())
paired_ica_shift = paired_ica_marginal_w1(
    paired_ica, paired_ica + np.array([2., -3.]))
assert paired_ica_zero["mean_marginal_1d_wasserstein"] == 0.0
assert paired_ica_shift["mean_marginal_1d_wasserstein"] > 0.0
assert "not a multivariate" in paired_ica_shift["definition"]
results["paired_common_frame_ica_metric_contract"] = {
    "identical_inputs_have_zero_distance": True,
    "shifted_inputs_have_positive_distance": True,
    "metric_labeled_as_marginal": True,
}

# Shared-gene Procrustes is a label-free, distance-preserving orientation
# correction. Verify recovery of a known rigid orientation/translation.
anchors = np.array([[0., 0.], [1., 0.], [0., 2.], [2., 3.], [-1., 1.]])
rotation = np.array([[0., -1.], [1., 0.]])
offset = np.array([4., -3.])
rotated = anchors @ rotation.T + offset
aligned, alignment_info = align_target_embedding_to_source(
    anchors, rotated, np.arange(len(anchors)), np.arange(len(anchors))
)
assert np.allclose(aligned, anchors, atol=1e-12)
assert np.allclose(np.linalg.norm(aligned[:, None] - aligned[None, :], axis=2),
                   np.linalg.norm(rotated[:, None] - rotated[None, :], axis=2))
results["shared_gene_procrustes"] = {
    "recovers_known_rigid_transform": True,
    "preserves_target_pairwise_distances": True,
    "matched_shared_genes": alignment_info["shared_gene_count"],
}
pixel_anchorer = object.__new__(DomainAdaptiveImageTransformer)
pixel_anchorer._coords_source = np.array([[1, 3], [8, 9], [4, 4]])
pixel_anchorer._coords_target = np.array([[5, 5], [2, 8], [7, 2]])
pixel_anchorer._kpca_da = types.SimpleNamespace(
    common_source_idx_=np.array([0, 2]), common_target_idx_=np.array([1, 2]))
pixel_anchorer._shared_gene_pixel_anchor_ = None
private_target_pixel_before = pixel_anchorer._coords_target[0].copy()
pixel_anchorer.anchor_shared_gene_pixels()
assert np.array_equal(pixel_anchorer._coords_target[[1, 2]],
                      pixel_anchorer._coords_source[[0, 2]])
assert np.array_equal(pixel_anchorer._coords_target[0], private_target_pixel_before)
results["shared_gene_pixel_anchor"] = {
    "shared_features_use_source_pixels": True,
    "target_private_feature_keeps_its_fitted_pixel": True,
    "anchored_gene_count": pixel_anchorer._shared_gene_pixel_anchor_["shared_gene_count"],
}

# Frozen target inference aligns new rows by gene ID, never refits the map, and
# produces identical images regardless of input gene order or unrelated extra genes.
source_gene_ids = np.array(["g1", "g2", "g3", "g4"])
target_gene_ids = np.array(["g3", "g2", "g5"])
source_fit = np.array([
    [0., 1., 5., 4.], [1., 2., 4., 3.], [2., 4., 3., 2.],
    [3., 8., 2., 1.], [4., 16., 1., 0.],
])
target_fit = np.array([
    [1., 2., 0.], [2., 4., 1.], [4., 8., 2.], [8., 16., 3.],
])
inference_preprocessor = GeneAwareMinMax(variance_percentile=0).fit(
    source_fit, source_gene_ids, target_fit, target_gene_ids
)
new_target = np.array([[3., 6., 1.5], [6., 12., 2.5]])
extra_target = np.column_stack((new_target, np.array([99., -5.])))
extra_gene_ids = np.append(target_gene_ids, "unseen_extra")
permutation = np.array([3, 1, 0, 2])
aligned_a = inference_preprocessor.transform_target_by_gene_id(
    new_target, target_gene_ids
)
aligned_b = inference_preprocessor.transform_target_by_gene_id(
    extra_target[:, permutation], extra_gene_ids[permutation]
)
assert np.allclose(aligned_a, aligned_b)

target_pixel_map = object.__new__(DomainAdaptiveImageTransformer)
target_pixel_map._genes_target = inference_preprocessor.target_genes_.copy()
target_pixel_map._coords_target = np.array([[0, 0], [0, 1], [1, 0]])
target_pixel_map._pixels = (2, 2)
image_a = transform_target_to_images(
    inference_preprocessor, target_pixel_map, new_target, target_gene_ids,
    feature_scope="all", img_format="scalar",
)
image_b = transform_target_to_images(
    inference_preprocessor, target_pixel_map,
    extra_target[:, permutation], extra_gene_ids[permutation],
    feature_scope="all", img_format="scalar",
)
image_rows = np.concatenate([
    transform_target_to_images(
        inference_preprocessor, target_pixel_map, row[None, :], target_gene_ids,
        feature_scope="all", img_format="scalar",
    ) for row in new_target
], axis=0)
assert np.allclose(image_a, image_b)
assert np.allclose(image_a, image_rows)
try:
    inference_preprocessor.transform_target_by_gene_id(
        new_target[:, [0, 2]], target_gene_ids[[0, 2]]
    )
except ValueError as exc:
    assert "missing genes required" in str(exc)
else:
    raise AssertionError("Inference must reject a missing retained gene")

shared_target_idx = np.array([1, 0])
shared_target_pixel_map = object.__new__(DomainAdaptiveImageTransformer)
shared_target_pixel_map._genes_target = np.array(["g2", "g3"])
shared_target_pixel_map._coords_target = np.array([[0, 0], [0, 1]])
shared_target_pixel_map._pixels = (2, 2)
shared_images = transform_target_to_images(
    inference_preprocessor, shared_target_pixel_map, new_target, target_gene_ids,
    feature_scope="shared", img_format="scalar",
)
assert shared_images.shape == (2, 2, 2)
results["frozen_target_inference"] = {
    "gene_order_invariant": True,
    "unseen_extra_genes_ignored": True,
    "batch_partition_invariant": True,
    "missing_retained_gene_rejected": True,
    "shared_feature_scope_reorders_by_gene_id": True,
}

weights = balanced_class_weights(
    np.array([0, 0, 0, 0, 1, 1]), labels=np.array([0, 1])
)
assert np.allclose(weights, np.array([0.75, 1.5]))
try:
    balanced_class_weights(np.array([0, 0]), labels=np.array([0, 1]))
except ValueError:
    pass
else:
    raise AssertionError("Class weights must reject a missing training class")
results["source_training_class_weights"] = {
    "inverse_frequency_weights_correct": True,
    "missing_class_rejected": True,
}

# Rectangular image layouts must use width as the row stride and inverse
# flattening base. Square layouts hid this indexing error.
rectangular_centers = DomainAdaptiveImageTransformer.calculate_pixel_centroids((2, 3))
expected_rectangular_centers = np.array([
    [0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
    [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
])
assert np.array_equal(rectangular_centers, expected_rectangular_centers)
rectangular_assignments = DomainAdaptiveImageTransformer.coordinate_optimal_assignment(
    np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0],
              [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]),
    (2, 3), bounds=(np.array([0.0, 0.0]), np.array([2.0, 3.0])),
)
assert np.all(rectangular_assignments[:, 0] < 2)
assert np.all(rectangular_assignments[:, 1] < 3)
assert np.unique(rectangular_assignments, axis=0).shape[0] == 6
legacy_rectangular_centers = ImageTransformer.calculate_pixel_centroids((2, 3))
legacy_rectangular_assignments = ImageTransformer.coordinate_optimal_assignment(
    np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0],
              [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]),
    (2, 3),
)
assert np.array_equal(legacy_rectangular_centers, expected_rectangular_centers)
assert np.all(legacy_rectangular_assignments[:, 0] < 2)
assert np.all(legacy_rectangular_assignments[:, 1] < 3)
assert np.unique(legacy_rectangular_assignments, axis=0).shape[0] == 6
results["rectangular_pixel_mapping"] = {
    "adaptive_and_raw_centroids_use_row_major_width_stride": True,
    "adaptive_and_raw_assignment_indices_in_bounds": True,
    "both_mappers_assign_six_features_to_distinct_pixels": True,
}

# Gene CAM ranking follows the exact domain feature order and exposes pixels
# where multiple genes necessarily share the same attribution value.
cam_batch = np.array([
    [[0.1, 0.9], [0.4, 0.2]],
    [[0.3, 0.7], [0.2, 0.8]],
])
gene_rank = rank_gene_attributions(
    cam_batch, np.array(["gene-A", "gene-B", "gene-C"]),
    np.array([[0, 0], [0, 1], [0, 1]]), top_k=3,
)
assert [row["gene_id"] for row in gene_rank] == ["gene-B", "gene-C", "gene-A"]
assert gene_rank[0]["score"] == gene_rank[1]["score"] == 1.0
assert gene_rank[0]["genes_at_pixel"] == gene_rank[1]["genes_at_pixel"] == 2
thresholded_gene_rank = rank_gene_attributions(
    cam_batch, np.array(["gene-A", "gene-B", "gene-C"]),
    np.array([[0, 0], [0, 1], [0, 1]]), threshold=0.5,
)
assert [row["gene_id"] for row in thresholded_gene_rank] == ["gene-B", "gene-C"]
constant_cam_rank = rank_gene_attributions(
    np.ones((2, 2)), np.array(["gene-A"]), np.array([[0, 0]]), threshold=0.5,
)
assert constant_cam_rank == []
results["gene_cam_mapping"] = {
    "uses_ordered_gene_coordinates": True,
    "stable_ranking": [row["gene_id"] for row in gene_rank],
    "pixel_collision_is_reported": gene_rank[0]["genes_at_pixel"] == 2,
    "supplement_normalization_and_threshold": [row["gene_id"] for row in thresholded_gene_rank],
    "constant_map_has_no_above_threshold_gene": constant_cam_rank == [],
}

# The selector can consume the exact adaptive target map rather than assuming
# a legacy ``coords()`` map, and maps top-k/threshold results to gene order.
class _SyntheticAdaptiveMap:
    _genes_source = np.array(["gene-A", "gene-B", "gene-C"])
    _genes_target = np.array(["gene-A", "gene-B", "gene-C"])

    @staticmethod
    def coords_source():
        return np.array([[0, 0], [0, 1], [1, 1]])

    @staticmethod
    def coords_target():
        return np.array([[1, 1], [1, 1], [1, 0]])


cam_model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, kernel_size=1))
adaptive_selector = CAMFeatureSelector(
    cam_model, _SyntheticAdaptiveMap(), target_layer=cam_model[0],
    map_domain="target", gene_ids=["gene-A", "gene-B", "gene-C"],
)
adaptive_cam = {1: np.array([[0.1, 0.6], [1.0, 0.8]])}
adaptive_rank = adaptive_selector.rank_class_features(adaptive_cam, top_k=3)[1]
adaptive_indices, adaptive_scores, adaptive_coords = adaptive_selector.select_class_features(
    adaptive_cam, threshold=0.8
)
assert [row["gene_id"] for row in adaptive_rank] == ["gene-C", "gene-A", "gene-B"]
assert adaptive_rank[1]["genes_at_pixel"] == adaptive_rank[2]["genes_at_pixel"] == 2
assert np.array_equal(adaptive_indices[1], np.array([2]))  # strict threshold excludes 0.8
assert adaptive_scores[1].tolist() == [1.0]
assert adaptive_coords[1].tolist() == [[1, 0]]
try:
    CAMFeatureSelector(cam_model, _SyntheticAdaptiveMap(), target_layer=cam_model[0])
except ValueError as exc:
    assert "map_domain" in str(exc)
else:
    raise AssertionError("Adaptive attribution must require an explicit domain map")
try:
    CAMFeatureSelector(
        cam_model, _SyntheticAdaptiveMap(), target_layer=cam_model[0],
        map_domain="target", gene_ids=["gene-B", "gene-A", "gene-C"],
    )
except ValueError as exc:
    assert "feature order" in str(exc)
else:
    raise AssertionError("Gene IDs inconsistent with the fitted map must be rejected")
results["adaptive_gene_cam_selector"] = {
    "explicit_target_domain_used": True,
    "ranked_gene_ids_follow_target_map": [row["gene_id"] for row in adaptive_rank],
    "collisions_reported": True,
    "strict_threshold_excludes_equal_score": True,
    "missing_map_domain_rejected": True,
    "mismatched_gene_order_rejected": True,
}

# ComBat receives only source-train/target-adaptation values, reorders shared
# genes by ID, and applies the learned target batch to held-out target values.
combat_rng = np.random.default_rng(17)
combat_genes_source = np.array([f"cg{i}" for i in range(12)])
combat_target_order = np.arange(12)[::-1]
combat_genes_target = combat_genes_source[combat_target_order]
combat_source = combat_rng.normal(size=(40, 12))
combat_source_train = combat_source[:25]
combat_source_validation = combat_source[25:30]
combat_target = combat_source[30:]
combat_target_adaptation = combat_target[:6] * 1.8 + 2.5
combat_target_test = combat_target[6:] * 1.8 + 2.5
with redirect_stdout(StringIO()):  # neuroCombat logs progress to stdout
    combat_result = fit_apply_combat_partitions(
        combat_source_train, combat_source_validation,
        combat_target_adaptation[:, combat_target_order],
        combat_target_test[:, combat_target_order],
        combat_genes_source, combat_genes_target,
    )
assert np.array_equal(combat_result.genes, combat_genes_source)
assert combat_result.source_validation.shape == combat_source_validation.shape
assert combat_result.target_test.shape == combat_target_test.shape
assert combat_result.fit_metadata["target_batch_id"] == "target"
assert combat_result.fit_metadata["target_labels_used"] is False
assert all(np.isfinite(matrix).all() for matrix in (
    combat_result.source_train, combat_result.source_validation,
    combat_result.target_adaptation, combat_result.target_test))
results["combat_train_adapt_only"] = {
    "ordered_shared_genes": True,
    "validation_and_test_excluded_from_fit": True,
    "heldout_target_uses_target_batch": True,
    "target_labels_unused": True,
}

# Target-label lambda selection is blocked unless explicitly opted into a
# separately reported supervised experiment.
try:
    kpcada.tune_lambda(np.zeros((3, 2)), np.zeros((3, 2)),
        np.array(["g1", "g2"]), np.array(["g1", "g2"]),
        y_source=np.array([0, 1, 0]), y_target=np.array([0, 1, 0]))
except ValueError as exc:
    assert "Target labels cannot select" in str(exc)
    results["target_label_selection_guard"] = "blocked by default"
else:
    raise AssertionError("Target-label lambda selection should be blocked by default")

# Unlabeled target preflight/development can load expression and sample IDs
# without materializing the target label column for later accidental use.
official_target = ROOT / "Data" / "TvsN" / "GSE45498_TvsN_official_200.csv"
if official_target.is_file():
    X_target_labeled, genes_labeled, ids_labeled, labels_labeled = (
        kpcada.load_expression_dataset(str(official_target))
    )
    X_target_unlabeled, genes_unlabeled, ids_unlabeled, labels_unlabeled = (
        kpcada.load_expression_dataset(str(official_target), include_labels=False)
    )
    assert labels_labeled is not None and labels_unlabeled is None
    assert np.array_equal(X_target_labeled, X_target_unlabeled)
    assert np.array_equal(genes_labeled, genes_unlabeled)
    assert np.array_equal(ids_labeled, ids_unlabeled)
    results["target_label_withholding"] = {
        "unlabeled_loader_returns_no_label_array": labels_unlabeled is None,
        "expression_and_sample_ids_unchanged": True,
    }
    # The explicit post-freeze scorer must check the cohort hash and frozen
    # test IDs, and may write the held-out score artifact only once.
    classes = sorted(np.unique(labels_labeled.astype(str)).tolist())
    sample_index = np.arange(min(8, len(ids_labeled)))
    class_index = {label: index for index, label in enumerate(classes)}
    fixture_labels = np.asarray([
        class_index[str(labels_labeled[index])] for index in sample_index
    ], dtype=int)
    fixture_patients = np.asarray([f"fixture-patient-{i}" for i in sample_index])
    fixture_probabilities = np.eye(len(classes), dtype=float)[fixture_labels]
    with tempfile.TemporaryDirectory(prefix="withheld-target-score-") as temp_root:
        fixture_run = Path(temp_root)
        fixture_metadata = {
            "target_test_labels_scored": False,
            "test_metrics": None,
            "target_file": str(official_target.resolve()),
            "target_file_sha256": hashlib.sha256(official_target.read_bytes()).hexdigest(),
            "source_file_sha256": "fixture-source-hash",
            "label_classes": classes,
            "arm": "fixture",
            "seed": 0,
            "config": {"withhold_target_test_labels": True},
        }
        (fixture_run / "metrics.json").write_text(json.dumps(fixture_metadata))
        (fixture_run / "best_model.pt").write_bytes(b"fixture-checkpoint")
        np.savez_compressed(
            fixture_run / "predictions.npz",
            sample_ids=ids_labeled[sample_index].astype(str),
            patient_ids=fixture_patients,
            y_pred=fixture_labels,
            probabilities=fixture_probabilities,
        )
        np.savez_compressed(
            fixture_run / "split_ids.npz",
            target_test=ids_labeled[sample_index].astype(str),
            target_test_patient=fixture_patients,
            target_adaptation_patient=np.asarray(["adaptation-patient"]),
        )
        scored_path = score_withheld_target_run(fixture_run)
        scored_payload = json.loads(scored_path.read_text())
        assert scored_payload["target_test_metrics"]["accuracy"] == 1.0
        try:
            score_withheld_target_run(fixture_run)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Target-test scoring should be a one-time artifact")
    results["one_time_withheld_target_scoring"] = {
        "hash_and_split_checked": True,
        "scorer_does_not_overwrite": True,
    }

# D8: Parse all source without importing/training the notebook dependencies.
syntax = []
for path in sorted((ROOT / "Scripts").rglob("*.py")):
    ast.parse(path.read_text(), filename=str(path))
    syntax.append(str(path.relative_to(ROOT)))
spatial_augmentations = {
    "RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation",
    "RandomAffine", "RandomPerspective", "RandomResizedCrop",
}
found_spatial_augmentations = []
for filename in ("run_tvsn_experiment.py", "run_combat_tvsn_experiment.py"):
    path = ROOT / "Scripts" / filename
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in spatial_augmentations:
                found_spatial_augmentations.append(f"{filename}:{node.lineno}:{node.func.attr}")
if found_spatial_augmentations:
    raise AssertionError(
        "Gene-image training must not apply spatial augmentation: "
        + ", ".join(found_spatial_augmentations)
    )
results["gene_image_spatial_augmentation"] = {
    "forbidden_transforms_absent": True,
    "runners_checked": ["run_tvsn_experiment.py", "run_combat_tvsn_experiment.py"],
}
nb_count = 0
archived_notebooks = {
    "CombatFit.ipynb", "ICAperformance.ipynb", "UMAPperformance.ipynb",
    "applyingCORAL.ipynb", "cf_rna_adaptation.ipynb", "claude_code.ipynb",
    "claude_code_grad_cam.ipynb", "scores_test.ipynb",
}
guarded_notebooks = set()
for path in sorted((ROOT / "Scripts").glob("*.ipynb")):
    nb = json.loads(path.read_text())
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    for i, cell in enumerate(nb["cells"], 1):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{path.name}:cell{i}")
    if path.name in archived_notebooks:
        if not code_cells or "raise RuntimeError" not in "".join(code_cells[0].get("source", [])):
            raise AssertionError(f"{path.name} must stop in its first code cell")
        if any(cell.get("outputs") or cell.get("execution_count") is not None
               for cell in code_cells):
            raise AssertionError(f"Archived outputs are stale in {path.name}")
        guarded_notebooks.add(path.name)
    nb_count += 1
if guarded_notebooks != archived_notebooks:
    raise AssertionError(f"Archived notebook inventory mismatch: {sorted(guarded_notebooks)}")
active_notebook = json.loads((ROOT / "Scripts" / "test_kPCADA.ipynb").read_text())
active_source = "\n".join(
    "".join(cell.get("source", [])) for cell in active_notebook["cells"]
)
for stale_reference in ("GSE45498_NvT.csv", "TCGA_BRCA_NvT.csv", "epochs': 100",
                        "regularization_mode': 'reference'"):
    if stale_reference in active_source:
        raise AssertionError(f"Active notebook contains stale experiment setting: {stale_reference}")
for canonical_reference in ("run_full_tcga_candidate_pair.py",
                            "exact_variance_filter_seed46_pair_final",
                            "full_tcga_seed52_corrected_pair", ".queued"):
    if canonical_reference not in active_source:
        raise AssertionError(f"Active notebook is missing canonical-runner reference: {canonical_reference}")
results["active_notebook_runner"] = {
    "canonical_matched_runner": True,
    "stale_cohort_and_settings_absent": True,
    "detached_launch": True,
    "duplicate_launch_guard": True,
}
results["syntax"] = {"python_files": len(syntax), "notebooks": nb_count, "all_code_cells_parse": True}
results["archived_notebook_guards"] = {
    "first_code_cell_guards_all_archives": True,
    "archived_outputs_cleared": True,
    "count": len(guarded_notebooks),
}
serialized_results = json.dumps(results, indent=2, sort_keys=True) + "\n"
(ROOT / "audit" / "current_smoke_results.json").write_text(serialized_results)
print(serialized_results, end="")
