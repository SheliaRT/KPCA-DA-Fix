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
import inspect
from io import StringIO
import json
import math
import platform
import subprocess
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
                                 rank_variance_keep_mask,
                                 transform_target_to_images)
from evaluation_protocol import (balanced_class_weights, classification_metrics,
                                 configure_trainable_parameters, fit_with_validation,
                                 select_binary_threshold)
from combat_protocol import fit_apply_combat_partitions
from rebuild_common_frame_ica_pair import marginal_w1 as paired_ica_marginal_w1
from rebuild_source_fitted_ica import summarize as summarize_source_fitted_ica
from aggregate_tvsn_comparisons import adaptation_method_signature
from compare_tvsn_runs import (MATCHED_CONFIG_KEYS, config_value,
                               verify_code_environment_match,
                               verify_preprocessing_policy_match,
                               verify_representation_fit_scope,
                               verify_optimization_device_match)
from run_withheld_experiment_grid import (verify_matched_grid_configs,
                                          declared_config_differences,
                                          load_frozen_spec,
                                          verify_scaler_fit_ids)
from kpcada import align_target_embedding_to_source, target_coordinate_integrity
from run_tvsn_experiment import (parse_args as parse_tvsn_args, resolve_mmd_gamma,
                                 resolve_effective_regularization_mode,
                                 verify_input_fingerprint)
from analyze_saved_target_gradcam import verify_patient_ids as verify_attribution_patient_ids
from pyDeepInsightv3.gene_attribution import rank_gene_attributions
from pyDeepInsightv3.feature_selection import CAMFeatureSelector
from score_withheld_target_run import score_run as score_withheld_target_run
from pyDeepInsightv3.image_transformer import ImageTransformer
from pyDeepInsightv3.image_transformer_new import DomainAdaptiveImageTransformer

import_check = subprocess.run(
    [sys.executable, "-c", (
        "import sys; "
        "from Scripts.pyDeepInsightv3.image_transformer_new import "
        "DomainAdaptiveImageTransformer; "
        "assert any(p.endswith('/Scripts') for p in sys.path if p)"
    )],
    cwd=ROOT, capture_output=True, text=True, check=True,
)
results = {"environment": {"python": platform.python_version(),
                          "numpy": np.__version__, "sklearn": sklearn.__version__,
                          "torch": torch.__version__}}
results["nested_image_transformer_import_fallback"] = {
    "repository_root_package_import_succeeds": True,
    "scripts_parent_added_to_path": True,
}

for callable_ in (kpcada.fit_transform, kpcada.KPCA_DA, kpcada.tune_lambda):
    parameters = inspect.signature(callable_).parameters
    assert parameters["regularization_mode"].default == "trace"
    assert parameters["lr"].default == 0.001
    assert parameters["epochs"].default == 500
    assert parameters["kpca_gamma"].default == 0.2
    if callable_ in (kpcada.fit_transform, kpcada.KPCA_DA):
        assert parameters["lambda_reg"].default == 0.001
results["paper_faithful_kpcada_api_defaults"] = {
    "fit_transform": "trace",
    "KPCA_DA": "trace",
    "tune_lambda": "trace",
    "default_learning_rate": 0.001,
    "default_adaptation_steps": 500,
    "default_lambda_reg": 0.001,
    "default_kpca_gamma": 0.2,
}

gamma_args_auto = parse_tvsn_args(["--arm", "kpcada"])
gamma_args_fixed = parse_tvsn_args(["--arm", "kpcada", "--mmd-gamma", "965.55"])
device_args_cpu = parse_tvsn_args(
    ["--arm", "kpcada", "--kpca-optimization-device", "cpu"]
)
assert gamma_args_auto.mmd_gamma is None
assert gamma_args_fixed.mmd_gamma == 965.55
assert gamma_args_auto.kpca_optimization_device == "auto"
assert device_args_cpu.kpca_optimization_device == "cpu"
assert str(kpcada.resolve_optimization_device("auto")) in {"cpu", "cuda", "mps"}
assert resolve_effective_regularization_mode("baseline", "trace") == "none"
assert resolve_effective_regularization_mode("kpcada", "trace") == "trace"
assert resolve_mmd_gamma(None) == (0.5, True)
assert resolve_mmd_gamma(965.55) == (965.55, False)
try:
    resolve_mmd_gamma(965.55, arm="baseline")
except ValueError:
    pass
else:
    raise AssertionError("fixed MMD gamma was accepted for the baseline arm")
for invalid_gamma in (0.0, -1.0, float("inf"), float("nan")):
    try:
        resolve_mmd_gamma(invalid_gamma)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid MMD gamma was accepted: {invalid_gamma}")
initial_coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
integrity = target_coordinate_integrity(initial_coordinates, initial_coordinates + 0.1)
assert integrity["finite"] and not integrity["collapsed"]
assert not target_coordinate_integrity(initial_coordinates * 1e-8, initial_coordinates * 1e-8)["collapsed"]
assert np.isclose(integrity["variance_retention_ratio"], 1.0)
collapsed_integrity = target_coordinate_integrity(initial_coordinates, np.zeros_like(initial_coordinates))
assert collapsed_integrity["collapsed"]
nonfinite_integrity = target_coordinate_integrity(initial_coordinates, np.full_like(initial_coordinates, np.nan))
assert not nonfinite_integrity["finite"] and nonfinite_integrity["collapsed"]
results["lambda_candidate_coordinate_integrity"] = {
    "variance_retention_reported": True,
    "normalized_displacement_reported": True,
    "near_zero_variance_rejected": True,
    "nonfinite_coordinates_rejected": True,
}
original_tune_fit = kpcada.fit_transform
def fake_candidate_fit(*, initial_embeddings, lambda_reg, **kwargs):
    start = initial_embeddings["Z_t_initial"]
    adapted = np.zeros_like(start) if lambda_reg == 0.0001 else start.copy()
    mmd = 0.0 if lambda_reg == 0.0001 else 0.5
    return {
        "Z_s": initial_embeddings["Z_s"], "Z_t": adapted,
        "Z_t_initial": start, "common_source_idx": np.arange(len(start)),
        "common_target_idx": np.arange(len(start)),
        "final_mmd": mmd, "final_regularization": 0.0,
        "final_loss": mmd, "loss_history": [mmd],
    }
kpcada.fit_transform = fake_candidate_fit
try:
    selector_rng = np.random.default_rng(321)
    selector_source = selector_rng.normal(size=(8, 10))
    selector_target = selector_rng.normal(size=(8, 9))
    selector_genes = np.array([f"g{i}" for i in range(8)])
    selector_result = kpcada.tune_lambda(
        selector_source, selector_target, selector_genes, selector_genes,
        lambda_values=(0.0001, 0.1), kpca_gamma=0.2, auto_gamma=False,
        epochs=0,
    )
finally:
    kpcada.fit_transform = original_tune_fit
assert selector_result["best_lambda_mmd"] == 0.1
assert selector_result["rejected_candidates"] == [{
    "lambda": 0.0001, "reason": "near-zero target-coordinate variance"
}]
results["lambda_selection_rejects_coordinate_collapse"] = True
base_aggregate_method = {
    "regularization_mode": "trace", "alignment_weight": 1.0,
    "kpca_epochs": 500, "kpca_lr": 0.001, "kpca_gamma": 0.2,
    "mmd_gamma_setting": None, "effective_mmd_gamma": 7.8, "lambda_reg_setting": None,
    "lambda_grid_setting": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    "align_shared_genes": False, "anchor_shared_pixels": False,
}
auto_gamma_signature = adaptation_method_signature({"method_configurations": {"comparison": base_aggregate_method}})
auto_other_seed = {**base_aggregate_method, "effective_mmd_gamma": 8.2}
assert adaptation_method_signature({"method_configurations": {"comparison": auto_other_seed}}) == auto_gamma_signature
fixed_gamma_method = {**base_aggregate_method, "mmd_gamma_setting": 965.55, "effective_mmd_gamma": 965.55}
fixed_gamma_signature = adaptation_method_signature({"method_configurations": {"comparison": fixed_gamma_method}})
assert auto_gamma_signature != fixed_gamma_signature
cpu_device_signature = adaptation_method_signature({
    "method_configurations": {"comparison": base_aggregate_method}
})
mps_device_signature = adaptation_method_signature({
    "method_configurations": {"comparison": {
        **base_aggregate_method, "kpca_optimization_device": "mps"
    }}
})
assert cpu_device_signature != mps_device_signature
results["aggregation_separates_mmd_bandwidths"] = True
results["aggregation_separates_objective_devices"] = True
results["runner_mmd_gamma_protocol"] = {
    "automatic_default": "supplement median-distance heuristic",
    "fixed_paper_bandwidth_accepted": 965.55,
    "nonpositive_or_nonfinite_rejected": True,
    "baseline_fixed_gamma_rejected": True,
}

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
adaptation_scaled_policy = {
    "feature_scope": "shared",
    "config": {"shared_scaling_policy": "target_adaptation"},
    "preprocessing_policy": {
        **common_preprocessing_policy,
        "shared_gene_scaler_fit_scope": "target adaptation samples only",
        "target_private_filter_fit_scope": "not fitted",
        "target_private_scaler_fit_scope": "not fitted",
    },
}
adaptation_scaled_pair_signature = verify_preprocessing_policy_match(
    adaptation_scaled_policy,
    {**adaptation_scaled_policy, "arm": "kpcada"},
)
assert adaptation_scaled_pair_signature["shared_gene_scaler_fit_scope"] == (
    "target adaptation samples only"
)
try:
    verify_preprocessing_policy_match(
        {**adaptation_scaled_policy, "config": {"shared_scaling_policy": "source"}},
        adaptation_scaled_policy,
    )
except ValueError as error:
    assert "disagrees with saved shared_scaling_policy" in str(error)
else:
    raise AssertionError("verifier accepted a scaling fit scope inconsistent with config")
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
    "adaptation_only_shared_scaling_supported": True,
    "scaling_policy_fit_scope_mismatch_rejected": True,
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
scaler_split_manifest = {
    "source_train": np.array(["source-1", "source-2"]),
    "target_adaptation": np.array(["target-adapt-1", "target-adapt-2"]),
    "target_test": np.array(["target-test-1"]),
}
verify_representation_fit_scope(
    {"config": {"shared_scaling_policy": "target_adaptation"}},
    {**scaler_split_manifest,
     "shared_scaler_fit_sample_ids": scaler_split_manifest["target_adaptation"]},
)
for invalid_scaler_manifest in (
    scaler_split_manifest,
    {**scaler_split_manifest,
     "shared_scaler_fit_sample_ids": scaler_split_manifest["source_train"]},
):
    try:
        verify_representation_fit_scope(
            {"config": {"shared_scaling_policy": "target_adaptation"}},
            invalid_scaler_manifest,
        )
    except ValueError as error:
        assert "scaler" in str(error).lower()
    else:
        raise AssertionError("verifier accepted missing or incorrect adaptation scaler IDs")
results["shared_scaler_fit_manifest"] = {
    "adaptation_only_fit_ids_accepted": True,
    "missing_or_source_fit_ids_rejected_for_adaptation_policy": True,
}
grid_split_manifest = {
    **scaler_split_manifest,
    "shared_scaler_fit_sample_ids": scaler_split_manifest["target_adaptation"],
}
assert np.array_equal(
    verify_scaler_fit_ids(grid_split_manifest, "target_adaptation", "fixture"),
    scaler_split_manifest["target_adaptation"],
)
for invalid_grid_manifest, invalid_policy in (
    ({key: value for key, value in grid_split_manifest.items()
      if key != "shared_scaler_fit_sample_ids"}, "target_adaptation"),
    ({**grid_split_manifest,
      "shared_scaler_fit_sample_ids": scaler_split_manifest["source_train"]},
     "target_adaptation"),
):
    try:
        verify_scaler_fit_ids(invalid_grid_manifest, invalid_policy, "fixture")
    except ValueError:
        pass
    else:
        raise AssertionError("withheld grid accepted invalid scaler-fit provenance")
results["withheld_grid_scaler_manifest"] = {
    "adaptation_ids_accepted": True,
    "missing_or_source_ids_rejected": True,
}
grid_common_config = {
    "arm": "baseline",
    "shared_scaling_policy": "source",
    "seed": 55,
    "model": "resnet50d",
    "classifier_lr": 1e-4,
    "results_root": "results/one",
}
verify_matched_grid_configs(
    grid_common_config,
    {**grid_common_config, "arm": "kpcada",
     "shared_scaling_policy": "target_adaptation",
     "results_root": "results/two"},
    allowed_differences={"arm", "shared_scaling_policy"},
    reference_name="baseline", candidate_name="kpcada",
)
assert declared_config_differences({
    "runs": [
        {"options": {"--shared-scaling-policy": "source"}},
        {"options": {"--shared-scaling-policy": "target_adaptation"}},
    ],
}) == {"arm", "shared_scaling_policy"}
try:
    verify_matched_grid_configs(
        grid_common_config,
        {**grid_common_config, "model": "resnet50"},
        allowed_differences={"arm", "shared_scaling_policy"},
        reference_name="baseline", candidate_name="mismatched",
    )
except ValueError as error:
    assert "model" in str(error)
else:
    raise AssertionError("withheld grid accepted an undeclared classifier mismatch")
results["withheld_grid_matched_configuration"] = {
    "declared_variants_accepted": True,
    "undeclared_model_difference_rejected": True,
}
with tempfile.TemporaryDirectory() as temp_dir:
    spec_file = Path(temp_dir) / "grid.json"
    original_spec_bytes = b'{"name":"frozen","seed":55}\n'
    spec_file.write_bytes(original_spec_bytes)
    frozen_spec, frozen_digest = load_frozen_spec(spec_file)
    spec_file.write_text('{"name":"edited-after-launch","seed":56}\n')
    assert frozen_spec["name"] == "frozen" and frozen_spec["seed"] == 55
    assert frozen_digest == hashlib.sha256(original_spec_bytes).hexdigest()
results["withheld_grid_spec_snapshot"] = {
    "loaded_spec_and_digest_remain_bound_to_same_bytes": True,
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

# The article's reported 158 mRNA tumors match the count of unique GEO tumor
# patient IDs, not the number of public tumor profiles. Preserve that distinction
# and ensure all profiles from either repeated patient stay in one split group.
geo_manifest = json.loads(
    (ROOT / "audit/gse45498_tvsn_rebuild_manifest.json").read_text(encoding="utf-8")
)
tumor_reconciliation = geo_manifest["tumor_profile_patient_reconciliation"]
assert tumor_reconciliation["tumor_profile_count"] == 160
assert tumor_reconciliation["unique_tumor_patient_count"] == 158
assert tumor_reconciliation["additional_profiles_beyond_one_per_patient"] == 2
repeated = tumor_reconciliation["repeated_tumor_profiles"]
assert [group["patient_id"] for group in repeated] == ["101", "91"]
for group in repeated:
    sample_ids = [f"GSE45498:{group['patient_id']}:{sample['gsm']}"
                  for sample in group["samples"]]
    mapped_groups = patient_group_ids(sample_ids)
    assert len(set(mapped_groups.tolist())) == 1
all_samples = geo_manifest["samples"]
all_ids = [
    f"GSE45498:{sample['patient_id']}:{sample['gsm']}" for sample in all_samples
]
all_groups = patient_group_ids(all_ids)
labels_by_group = {}
for group_id, sample in zip(all_groups.tolist(), all_samples):
    labels_by_group.setdefault(group_id, set()).add(sample["label"])
paired_patient_groups = [
    group_id for group_id, labels in labels_by_group.items()
    if labels == {"Normal", "Tumor"}
]
assert len(all_samples) == 200
assert len(labels_by_group) == 158
assert len(paired_patient_groups) == 40
results["geo_profile_patient_reconciliation"] = {
    "160_tumor_profiles_from_158_patients": True,
    "repeated_patient_ids": [group["patient_id"] for group in repeated],
    "all_repeated_profiles_share_patient_split_group": True,
    "40_matched_normal_tumor_patient_groups_stay_together": True,
    "paper_counting_unit_remains_unconfirmed": True,
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

# Caching fixed kernel terms in the optimization loop must preserve both the
# objective value and its coordinate gradients.
cache_source = torch.tensor([[-0.4, 0.2], [0.8, -0.1], [0.3, 1.1]])
cache_target = torch.tensor([[0.2, 0.6], [-0.2, 0.5], [1.0, -0.7]])
cache_gamma = 0.7
cache_source_mean = kpcada.rbf_kernel(
    cache_source, cache_source, cache_gamma
).mean().detach()
target_direct = cache_target.clone().requires_grad_(True)
target_cached = cache_target.clone().requires_grad_(True)
loss_direct = kpcada.mmd_loss(cache_source, target_direct, gamma=cache_gamma)
loss_cached = kpcada.mmd_loss(
    cache_source, target_cached, gamma=cache_gamma,
    source_kernel_mean=cache_source_mean,
)
grad_direct, = torch.autograd.grad(loss_direct, target_direct)
grad_cached, = torch.autograd.grad(loss_cached, target_cached)
assert torch.allclose(loss_direct, loss_cached, rtol=1e-7, atol=1e-8)
assert torch.allclose(grad_direct, grad_cached, rtol=1e-6, atol=1e-7)

cache_x_target = torch.tensor([[0., 1., 3.], [2., 0., 1.]])
cache_z_direct = torch.tensor([[0.4, -0.2], [-0.7, 0.6], [0.3, 1.2]],
                              requires_grad=True)
cache_z_cached = cache_z_direct.detach().clone().requires_grad_(True)
cache_kernel = kpcada.rbf_kernel(
    cache_x_target.T, cache_x_target.T, cache_gamma
).detach()
reg_direct = kpcada.regularization_term(
    cache_z_direct, cache_x_target, gamma=cache_gamma
)
reg_cached = kpcada.regularization_term_from_kernel(cache_z_cached, cache_kernel)
reg_grad_direct, = torch.autograd.grad(reg_direct, cache_z_direct)
reg_grad_cached, = torch.autograd.grad(reg_cached, cache_z_cached)
assert torch.allclose(reg_direct, reg_cached, rtol=1e-7, atol=1e-8)
assert torch.allclose(reg_grad_direct, reg_grad_cached, rtol=1e-6, atol=1e-7)
results["fixed_kernel_cache_equivalence"] = {
    "mmd_value_and_gradient_match_uncached": True,
    "trace_value_and_gradient_match_uncached": True,
}

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
default_mapper = DA()
assert default_mapper.pixels == (227, 227)
assert default_mapper._kpca_da.kpca_gamma == 0.2
assert default_mapper._kpca_da.lambda_reg == 0.001
assert default_mapper._kpca_da.lr == 0.001
assert default_mapper._kpca_da.epochs == 500
assert default_mapper._kpca_da.regularization_mode == "trace"
results["image_transformer_defaults_match_core_api"] = True
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

if torch.backends.mps.is_available():
    cpu_fit = kpcada.fit_transform(
        xs, xt, genes, genes, kpca_gamma=.2, mmd_gamma=.5,
        auto_gamma=False, lambda_reg=.1, lr=.01, epochs=5,
        optimization_device="cpu",
    )
    mps_fit = kpcada.fit_transform(
        xs, xt, genes, genes, kpca_gamma=.2, mmd_gamma=.5,
        auto_gamma=False, lambda_reg=.1, lr=.01, epochs=5,
        optimization_device="mps",
    )
    max_coordinate_difference = float(np.max(np.abs(cpu_fit["Z_t"] - mps_fit["Z_t"])))
    assert np.allclose(cpu_fit["Z_t"], mps_fit["Z_t"], rtol=1e-5, atol=1e-6)
    assert np.isclose(cpu_fit["final_loss"], mps_fit["final_loss"], rtol=1e-5, atol=1e-7)
    assert np.allclose(cpu_fit["loss_history"], mps_fit["loss_history"],
                       rtol=1e-5, atol=1e-7)
    assert mps_fit["optimization_device"] == "mps"
    results["accelerator_objective_equivalence"] = {
        "cpu_and_mps_final_coordinates_match": True,
        "cpu_and_mps_final_objective_match": True,
        "cpu_and_mps_loss_history_match": True,
        "maximum_coordinate_absolute_difference": max_coordinate_difference,
    }
else:
    results["accelerator_objective_equivalence"] = {
        "accelerator_available": False,
        "cpu_fallback_used": True,
    }

# D6: KPCA score convention differs from the displayed paper equations 7-8.
kp = KernelPCA(n_components=2, kernel="rbf", gamma=.2)
z = kp.fit_transform(xs)
paper_z = kp.eigenvectors_ / np.sqrt(kp.eigenvalues_)
results["kpca_equation"] = {
    "code_matches_V_times_sqrt_eigenvalues": bool(np.allclose(z, kp.eigenvectors_ * np.sqrt(kp.eigenvalues_))),
    "code_matches_paper_V_divided_by_sqrt_eigenvalues": bool(np.allclose(z, paper_z))}

# D7: Single-endpoint accounting versus tune_lambda's mixed endpoints.
summary = kpcada.tune_lambda(xs, xt, genes, genes, lambda_values=[.1, .2],
    kpca_gamma=.2, auto_gamma=False, mmd_gamma=.5, lr=.01, epochs=5,
    regularization_mode="reference")
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

if torch.backends.mps.is_available():
    cache_invalidation_model = kpcada.KPCA_DA(
        kpca_gamma=.2, auto_gamma=False, mmd_gamma=.5, lambda_reg=.1,
        lr=.01, epochs=2, kpca_random_state=17, optimization_device="cpu",
    )
    cache_invalidation_model.tune_lambda(
        xs, xt, genes, genes, lambda_values=[.1]
    )
    cache_invalidation_model.optimization_device = "mps"
    cache_invalidation_model.fit_transform(xs, xt, genes, genes)
    assert cache_invalidation_model.optimization_device_ == "mps"
    results["tuned_cache_invalidated_on_device_change"] = True

assert verify_optimization_device_match(
    {"kpca_optimization_device": "mps"},
    {"kpca_optimization_device": "mps"},
) == "mps"
try:
    verify_optimization_device_match(
        {"kpca_optimization_device": "cpu"},
        {"kpca_optimization_device": "mps"},
    )
except ValueError as error:
    assert "different effective" in str(error)
else:
    raise AssertionError("paired-run verifier accepted mixed objective devices")
results["paired_objective_device_contract"] = {
    "same_effective_device_accepted": True,
    "different_effective_devices_rejected": True,
}

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
combat_rank_filter = rank_variance_keep_mask(tie_matrix, 30)
assert np.array_equal(combat_rank_filter, tie_filter.source_keep_)
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
    "combat_uses_same_filter": True,
}
adapt_source = np.array([
    [0., 0., 0., 0.],
    [.001, 10., 20., 1.],
    [.0005, 5., 10., 2.],
])
adapt_target = np.array([[200., 5., 100.], [400., 7., 200.],
                         [300., 6., 150.]])
adapt_source_genes = np.array([
    "source-low", "shared-a", "shared-b", "source-only"
])
adapt_target_genes = np.array(["shared-b", "target-only", "shared-a"])
adaptation_scaler = GeneAwareMinMax(
    variance_percentile=25, shared_scaling_policy="target_adaptation"
).fit(adapt_source, adapt_source_genes, adapt_target, adapt_target_genes)
adapt_target_scaled = adaptation_scaler.transform_target(
    adapt_target, adapt_target_genes
)
adapt_source_scaled = adaptation_scaler.transform_source(
    adapt_source, adapt_source_genes
)
assert np.allclose(adapt_target_scaled[:, [0, 2]], [[0., 0.], [1., 1.], [.5, .5]])
assert np.allclose(
    adapt_source_scaled[:, [0, 1]],
    [[-1., -1.], [-.9, -.9], [-.95, -.95]],
)
assert not adaptation_scaler.source_keep_[0]
assert np.all(adapt_source_scaled[:, 2] >= 0.)
assert np.allclose(
    adaptation_scaler.transform_target_by_gene_id(
        adapt_target[:, ::-1], adapt_target_genes[::-1]
    ),
    adapt_target_scaled,
)
assert adaptation_scaler.shared_scaler_.data_min_.tolist() == [200., 100.]
legacy_source_policy = GeneAwareMinMax(
    variance_percentile=0, shared_scaling_policy="source"
).fit(X_source, source_gene_ids, X_target, target_gene_ids)
legacy_expected_source = legacy_source_policy.transform_source(X_source, source_gene_ids)
legacy_expected_target = legacy_source_policy.transform_target(X_target, target_gene_ids)
# Older saved fitted objects predate the shared-scaler state and its index arrays.
for legacy_attribute in (
    "shared_scaler_", "_source_shared_scaler_indices",
    "_target_shared_scaler_indices", "_source_shared_output_indices",
    "shared_scaling_policy",
):
    delattr(legacy_source_policy, legacy_attribute)
assert np.allclose(
    legacy_source_policy.transform_source(X_source, source_gene_ids),
    legacy_expected_source,
)
assert np.allclose(
    legacy_source_policy.transform_target(X_target, target_gene_ids),
    legacy_expected_target,
)
results["legacy_preprocessor_pickle_compatibility"] = {
    "pre_shared_scaling_source_transform_matches": True,
    "pre_shared_scaling_target_transform_matches": True,
}
try:
    GeneAwareMinMax(
        variance_percentile=0, shared_scaling_policy="target_adaptation"
    ).fit_source_shared(adapt_source, adapt_source_genes, adapt_target_genes)
except ValueError as error:
    assert "cannot use target-adaptation scaling" in str(error)
else:
    raise AssertionError("inductive raw DeepInsight path accepted adaptation-fitted scaling")
results["adaptation_fitted_shared_scaling"] = {
    "adaptation_values_map_to_unit_interval": True,
    "same_scaler_applied_to_source_and_target": True,
    "gene_order_invariant": True,
    "inductive_raw_deepinsight_rejects_transductive_policy": True,
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
with tempfile.TemporaryDirectory(prefix="input-fingerprint-") as fingerprint_tmp:
    fingerprint_path = Path(fingerprint_tmp) / "target.csv"
    fingerprint_path.write_text("sample data\n", encoding="utf-8")
    expected_fingerprint = hashlib.sha256(fingerprint_path.read_bytes()).hexdigest()
    verify_input_fingerprint(fingerprint_path, expected_fingerprint, "Target")
    fingerprint_path.write_text("changed sample data\n", encoding="utf-8")
    try:
        verify_input_fingerprint(fingerprint_path, expected_fingerprint, "Target")
    except RuntimeError as error:
        assert "changed during the run" in str(error)
    else:
        raise AssertionError("Input fingerprint guard accepted a changed cohort file")
results["input_fingerprint_guard"] = {
    "unchanged_input_accepted": True,
    "changed_input_rejected": True,
}
threshold = select_binary_threshold(
    [0, 0, 1, 1], [.1, .4, .6, .9], positive_label=1, negative_label=0
)
assert threshold == {"threshold": .5, "source_validation_balanced_accuracy": 1.0}
results["source_validation_threshold_selection"] = threshold

# Frozen-feature training must update only the classifier head and must not
# update pretrained BatchNorm running statistics.
class _TinyTimmLikeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3), torch.nn.ReLU()
        )
        self.head = torch.nn.Linear(3, 2)

    def get_classifier(self):
        return self.head

    def forward(self, inputs):
        return self.head(self.backbone(inputs))

frozen_classifier = _TinyTimmLikeClassifier()
freeze_policy = configure_trainable_parameters(frozen_classifier, freeze_backbone=True)
assert freeze_policy["freeze_backbone"] is True
assert freeze_policy["frozen_parameter_count"] > 0
assert all(parameter.requires_grad for parameter in frozen_classifier.head.parameters())
assert all(not parameter.requires_grad for parameter in frozen_classifier.backbone.parameters())
batchnorm = frozen_classifier.backbone[1]
running_mean_before = batchnorm.running_mean.clone()
train_inputs = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 10.0
train_labels = torch.tensor([0, 1] * 8, dtype=torch.long)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_inputs, train_labels), batch_size=4
)
optimizer = torch.optim.SGD(
    (parameter for parameter in frozen_classifier.parameters() if parameter.requires_grad),
    lr=0.01,
)
fit_with_validation(
    frozen_classifier, loader, loader, torch.nn.CrossEntropyLoss(), optimizer,
    max_epochs=2, patience=2, freeze_batchnorm_stats=True,
)
assert torch.equal(batchnorm.running_mean, running_mean_before)
results["frozen_backbone_training"] = {
    "only_classifier_head_trainable": True,
    "batchnorm_running_statistics_unchanged": True,
    "exposed_as_explicit_ablation": True,
}
assert "freeze_backbone" in MATCHED_CONFIG_KEYS
assert config_value({}, "freeze_backbone") is False
assert config_value({"freeze_backbone": True}, "freeze_backbone") is True
results["frozen_backbone_pairing"] = {
    "default_policy_is_full_finetuning": True,
    "freeze_setting_in_matched_pair_contract": True,
}

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
with tempfile.TemporaryDirectory() as mapping_tmp:
    attribution_map_path = Path(mapping_tmp) / "patient_map.json"
    attribution_map_path.write_text(json.dumps({"s1": "p1", "s2": "p2"}))
    attribution_map_hash = hashlib.sha256(attribution_map_path.read_bytes()).hexdigest()
    attribution_metadata = {"patient_mapping": {"source": {
        "source": "explicit sample-to-patient map",
        "mapping_file": str(attribution_map_path),
        "mapping_file_sha256": attribution_map_hash,
    }}}
    verify_attribution_patient_ids(
        attribution_metadata, "source", np.array(["s1", "s2"]),
        np.array(["patient:p1", "patient:p2"]),
    )
    try:
        verify_attribution_patient_ids(
            attribution_metadata, "source", np.array(["s1", "s2"]),
            np.array(["patient:p2", "patient:p1"]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Attribution accepted mismatched patient IDs")
results["attribution_patient_map_provenance"] = {
    "explicit_map_hash_checked": True,
    "saved_sample_patient_alignment_checked": True,
}
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
combat_estimates = combat_result.estimates
target_batch_index = int(np.flatnonzero(
    np.asarray(combat_estimates["batches"], dtype=str) == "target"
)[0])
pooled_variance = np.asarray(combat_estimates["var.pooled"], dtype=float).reshape(-1)
standard_mean = (
    np.asarray(combat_estimates["stand.mean"], dtype=float)[:, 0]
    + np.asarray(combat_estimates["mod.mean"], dtype=float).mean(axis=1)
)
gamma_star = np.asarray(
    combat_estimates["gamma.star"], dtype=float
)[target_batch_index].reshape(-1)
delta_star = np.asarray(
    combat_estimates["delta.star"], dtype=float
)[target_batch_index].reshape(-1)
target_test_in_common_order = combat_target_test
expected_target_test = (
    ((target_test_in_common_order - standard_mean) / np.sqrt(pooled_variance)
     - gamma_star)
    / np.sqrt(delta_star)
    * np.sqrt(pooled_variance)
    + standard_mean
)
assert np.allclose(combat_result.target_test, expected_target_test, rtol=1e-10, atol=1e-10)
with redirect_stdout(StringIO()):
    changed_heldout = fit_apply_combat_partitions(
        combat_source_train, combat_source_validation + 1000.0,
        combat_target_adaptation[:, combat_target_order],
        combat_target_test[:, combat_target_order] - 1000.0,
        combat_genes_source, combat_genes_target,
    )
assert np.allclose(combat_result.estimates["gamma.star"],
                    changed_heldout.estimates["gamma.star"])
assert np.allclose(combat_result.estimates["delta.star"],
                    changed_heldout.estimates["delta.star"])
assert np.allclose(combat_result.source_train, changed_heldout.source_train)
assert np.allclose(combat_result.target_adaptation, changed_heldout.target_adaptation)
results["combat_train_adapt_only"] = {
    "ordered_shared_genes": True,
    "validation_and_test_excluded_from_fit": True,
    "heldout_target_uses_target_batch": True,
    "heldout_target_matches_neurocombat_from_training_equation": True,
    "extreme_heldout_values_do_not_change_fitted_estimates": True,
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
    original_read_csv = kpcada.pd.read_csv
    unlabeled_read_options = {}
    def record_unlabeled_csv_options(*args, **kwargs):
        unlabeled_read_options.update(kwargs)
        return original_read_csv(*args, **kwargs)
    kpcada.pd.read_csv = record_unlabeled_csv_options
    try:
        X_target_unlabeled, genes_unlabeled, ids_unlabeled, labels_unlabeled = (
            kpcada.load_expression_dataset(
                str(official_target), include_labels=False
            )
        )
    finally:
        kpcada.pd.read_csv = original_read_csv
    assert labels_labeled is not None and labels_unlabeled is None
    assert unlabeled_read_options.get("skipfooter") == 1
    assert np.array_equal(X_target_labeled, X_target_unlabeled)
    assert np.array_equal(genes_labeled, genes_unlabeled)
    assert np.array_equal(ids_labeled, ids_unlabeled)
    original_read_csv_for_labels = kpcada.pd.read_csv
    def reject_full_matrix_read(*args, **kwargs):
        raise AssertionError("load_labels must not reload the expression matrix")
    kpcada.pd.read_csv = reject_full_matrix_read
    try:
        streamed_labels = kpcada.load_labels(str(official_target))
    finally:
        kpcada.pd.read_csv = original_read_csv_for_labels
    assert np.array_equal(streamed_labels.astype(str), labels_labeled.astype(str))
    results["target_label_withholding"] = {
        "unlabeled_loader_returns_no_label_array": labels_unlabeled is None,
        "label_row_skipped_by_csv_parser": True,
        "expression_and_sample_ids_unchanged": True,
        "final_scoring_reads_labels_without_reloading_expression_matrix": True,
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
for runner_filename in ("run_tvsn_experiment.py", "run_combat_tvsn_experiment.py"):
    runner_path = ROOT / "Scripts" / runner_filename
    runner_tree = ast.parse(runner_path.read_text(), filename=str(runner_path))
    unlabeled_target_load_lines = [
        node.lineno for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_expression_dataset"
        and any(keyword.arg == "include_labels"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
                for keyword in node.keywords)
    ]
    target_label_load_lines = [
        node.lineno for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_labels"
    ]
    target_prediction_context_lines = [
        node.lineno for node in ast.walk(runner_tree)
        if isinstance(node, ast.With)
        and any(isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "no_grad"
                for item in node.items)
    ]
    assert len(unlabeled_target_load_lines) == 1, runner_filename
    assert len(target_label_load_lines) == 1, runner_filename
    assert target_prediction_context_lines, runner_filename
    assert target_label_load_lines[0] > max(target_prediction_context_lines), runner_filename
results["runner_target_label_withholding"] = {
    "runners": ["run_tvsn_experiment.py", "run_combat_tvsn_experiment.py"],
    "parser_skips_target_label_row_during_fit_and_prediction": True,
    "labels_loaded_only_after_prediction_for_final_scoring": True,
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
