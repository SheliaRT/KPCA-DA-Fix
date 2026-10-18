"""Run a frozen grid of matched experiments with all target labels withheld.

The JSON spec is written and hashed before training starts. Every run gets the
same source/target cohorts, seed, patient map and common arguments; per-run
arguments declare the candidate differences (for example scaling policy).
The script refuses to reuse an output directory and verifies identical held-
out patient/sample manifests and code/environment fingerprints at completion.
It never scores target-test labels; use the one-time scorer only after the grid
is complete and the candidate set is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "Scripts/run_tvsn_experiment.py"
PYTHON = ROOT / ".venv/bin/python"
PROTECTED_FLAGS = {
    "--arm", "--seed", "--task-name", "--results-root", "--source-path",
    "--target-path", "--source-patient-map", "--target-patient-map",
    "--withhold-target-test-labels", "--split-only", "--preflight-only",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_spec(path: Path) -> tuple[dict[str, Any], str]:
    """Load and hash one exact spec snapshot before dependency waiting."""
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    return json.loads(payload.decode("utf-8")), digest


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def wait_for_job(job: str) -> None:
    marker = ROOT / "audit" / f"{job}.done"
    while not marker.is_file():
        time.sleep(15)
    result = json.loads(marker.read_text(encoding="utf-8"))
    if result.get("exit_code") != 0:
        raise RuntimeError(f"Required preceding job {job!r} failed; see {result.get('log')}")


def flatten_options(options: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for flag, value in options.items():
        if not flag.startswith("--") or flag in PROTECTED_FLAGS:
            raise ValueError(f"Candidate option may not override protected runner flag: {flag}")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, list):
            argv.append(flag)
            argv.extend(str(item) for item in value)
        elif value is not None:
            argv.extend([flag, str(value)])
    return argv


def verify_scaler_fit_ids(
    splits: dict[str, np.ndarray], policy: str, run_name: str
) -> np.ndarray:
    """Require the scaler's recorded IDs to equal its declared fit partition."""
    if policy not in {"source", "target_adaptation"}:
        raise ValueError(f"Unknown shared scaling policy for {run_name}: {policy!r}")
    if "shared_scaler_fit_sample_ids" not in splits:
        raise ValueError(
            f"Run {run_name} does not save shared_scaler_fit_sample_ids; "
            "the grid requires auditable scaler fit IDs"
        )
    fit_ids = np.asarray(splits["shared_scaler_fit_sample_ids"]).astype(str)
    expected_ids = np.asarray(
        splits["source_train" if policy == "source" else "target_adaptation"]
    ).astype(str)
    if not np.array_equal(fit_ids, expected_ids):
        raise ValueError(
            f"Run {run_name} shared scaler was not fit on exactly the declared "
            f"{policy!r} rows"
        )
    return fit_ids


def verify_matched_grid_configs(
    reference: dict[str, Any], candidate: dict[str, Any], *,
    allowed_differences: set[str], reference_name: str, candidate_name: str,
) -> None:
    """Reject undeclared configuration differences between frozen candidates."""
    ignored = set(allowed_differences) | {"results_root"}
    ref = {key: value for key, value in reference.items() if key not in ignored}
    cur = {key: value for key, value in candidate.items() if key not in ignored}
    if ref != cur:
        differing = sorted(
            key for key in set(ref) | set(cur) if ref.get(key) != cur.get(key)
        )
        raise ValueError(
            f"Grid configurations differ outside declared factors for "
            f"{reference_name} vs {candidate_name}: {differing}"
        )


def declared_config_differences(spec: dict[str, Any]) -> set[str]:
    """Resolve which saved config fields may vary across this frozen grid.

    Older queued specs lack the explicit allowlist. Preserve compatibility for
    the existing scaling grid by deriving only its named scaling variation;
    all other variation must be written explicitly into a new spec.
    """
    configured = spec.get("allowed_config_differences")
    if configured is not None:
        if (not isinstance(configured, list)
                or any(not isinstance(item, str) for item in configured)):
            raise ValueError(
                "allowed_config_differences must be a list of config field names"
            )
        return set(configured)
    allowed = {"arm"}
    if any(
        "--shared-scaling-policy" in run.get("options", {})
        for run in spec.get("runs", [])
    ):
        allowed.add("shared_scaling_policy")
    return allowed


def run_one(spec: dict[str, Any], run_spec: dict[str, Any], run_root: Path) -> dict[str, Any]:
    name = str(run_spec["name"])
    arm = str(run_spec["arm"])
    if arm not in {"baseline", "kpcada", "raw_deepinsight"}:
        raise ValueError(f"Unsupported arm for {name}: {arm}")
    output_root = run_root / name
    output_root.mkdir(parents=True, exist_ok=False)
    run_parent = output_root / "tvsn"
    run_parent.mkdir()
    command = [
        str(PYTHON), str(RUNNER), "--arm", arm,
        "--seed", str(spec["seed"]), "--task-name", str(spec["task_name"]),
        "--source-path", str(spec["source_path"]),
        "--target-path", str(spec["target_path"]),
        "--target-patient-map", str(spec["target_patient_map"]),
        "--results-root", str(output_root.relative_to(ROOT)),
        "--withhold-target-test-labels",
    ]
    if spec.get("source_patient_map"):
        command.extend(["--source-patient-map", str(spec["source_patient_map"])])
    command.extend(flatten_options(spec.get("common_options", {})))
    command.extend(flatten_options(run_spec.get("options", {})))
    print(f"Starting candidate {name}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    candidates = [path for path in run_parent.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one run artifact for {name}, found {candidates}")
    run_dir = candidates[0]
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.npz"
    splits_path = run_dir / "split_ids.npz"
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metadata.get("target_test_labels_scored") is not False:
        raise ValueError(f"Target test labels were not withheld for {name}")
    if metadata.get("test_metrics") is not None:
        raise ValueError(f"Target test metrics unexpectedly exist for {name}")
    if not metadata.get("config", {}).get("withhold_target_test_labels"):
        raise ValueError(f"Run configuration does not attest withholding for {name}")
    with np.load(predictions_path, allow_pickle=False) as predictions:
        if "y_true" in predictions.files:
            raise ValueError(f"Prediction artifact contains target labels for {name}")
        required = {"sample_ids", "patient_ids", "y_pred", "probabilities"}
        if not required.issubset(predictions.files):
            raise ValueError(f"Prediction artifact is incomplete for {name}")
        predicted_samples = predictions["sample_ids"].astype(str)
        predicted_patients = predictions["patient_ids"].astype(str)
    with np.load(splits_path, allow_pickle=False) as splits:
        test_samples = splits["target_test"].astype(str)
        test_patients = splits["target_test_patient"].astype(str)
        common_partitions = {
            key: splits[key].astype(str).tolist()
            for key in (
                "source_train", "source_validation", "target_adaptation",
                "target_test", "source_train_patient",
                "source_validation_patient", "target_adaptation_patient",
                "target_test_patient",
            )
        }
        scaling_policy = run_spec.get("options", {}).get(
            "--shared-scaling-policy",
            spec.get("common_options", {}).get("--shared-scaling-policy", "source"),
        )
        if metadata.get("config", {}).get("shared_scaling_policy", "source") != scaling_policy:
            raise ValueError(
                f"Run {name} saved a shared scaling policy different from the frozen spec"
            )
        scaler_fit_ids = verify_scaler_fit_ids(splits, scaling_policy, name)
    if not np.array_equal(predicted_samples, test_samples):
        raise ValueError(f"Saved prediction IDs do not match split IDs for {name}")
    if not np.array_equal(predicted_patients, test_patients):
        raise ValueError(f"Saved prediction patient IDs do not match split IDs for {name}")
    return {
        "name": name,
        "arm": arm,
        "variant": str(run_spec.get("variant", name)),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "config": metadata.get("config"),
        "source_file_sha256": metadata.get("source_file_sha256"),
        "target_file_sha256": metadata.get("target_file_sha256"),
        "target_patient_map_sha256": metadata.get("patient_mapping", {}).get(
            "target", {}
        ).get("mapping_file_sha256"),
        "python_source_tree_sha256": metadata.get("python_source_tree_sha256"),
        "environment_lock_sha256": metadata.get("environment_lock_sha256"),
        "label_classes": metadata.get("label_classes"),
        "predictions_sha256": sha256(predictions_path),
        "split_ids_sha256": sha256(splits_path),
        "partitions": common_partitions,
        "shared_scaling_policy": scaling_policy,
        "shared_scaler_fit_sample_ids": scaler_fit_ids.tolist(),
        "target_test_sample_ids": test_samples.tolist(),
        "target_test_patient_ids": test_patients.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="frozen JSON experiment grid specification")
    parser.add_argument("--after-job", action="append", default=[],
                        help="wait for a detached job to complete before training; repeatable")
    args = parser.parse_args()
    spec_path = resolve_path(args.spec)
    # Freeze the exact bytes consumed before any dependency wait. The file may
    # be edited while the job waits; hashing it after the wait would attest to
    # a spec the worker never loaded.
    spec, spec_sha256 = load_frozen_spec(spec_path)
    required = {"name", "task_name", "seed", "source_path", "target_path",
                "target_patient_map", "runs"}
    if not required.issubset(spec):
        raise ValueError(f"Spec is missing fields: {sorted(required - set(spec))}")
    if not isinstance(spec["runs"], list) or len(spec["runs"]) < 2:
        raise ValueError("The frozen grid must contain at least two candidate runs")
    allowed_config_differences = declared_config_differences(spec)
    names = [str(run.get("name", "")) for run in spec["runs"]]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Run names must be non-empty and unique")
    for key in ("source_path", "target_path", "target_patient_map"):
        path = resolve_path(spec[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        spec[key] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    if spec.get("source_patient_map"):
        path = resolve_path(spec["source_patient_map"])
        if not path.is_file():
            raise FileNotFoundError(path)
        spec["source_patient_map"] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)

    run_root = ROOT / "results" / "withheld_grids" / str(spec["name"])
    run_root.mkdir(parents=True, exist_ok=False)
    for job in args.after_job:
        wait_for_job(job)
    frozen = {
        "spec": spec,
        "spec_path": str(spec_path.relative_to(ROOT) if spec_path.is_relative_to(ROOT) else spec_path),
        "spec_sha256": spec_sha256,
        "source_file_sha256": sha256(resolve_path(spec["source_path"])),
        "target_file_sha256": sha256(resolve_path(spec["target_path"])),
        "target_patient_map_sha256": sha256(resolve_path(spec["target_patient_map"])),
        "source_patient_map_sha256": (
            sha256(resolve_path(spec["source_patient_map"]))
            if spec.get("source_patient_map") else None
        ),
        "target_test_labels_withheld": True,
    }
    (run_root / "frozen_spec.json").write_text(json.dumps(frozen, indent=2) + "\n",
                                               encoding="utf-8")
    runs = [run_one(spec, run_spec, run_root) for run_spec in spec["runs"]]
    reference = runs[0]
    for run in runs[1:]:
        verify_matched_grid_configs(
            reference["config"], run["config"],
            allowed_differences=allowed_config_differences,
            reference_name=reference["name"], candidate_name=run["name"],
        )
        for field in ("source_file_sha256", "target_file_sha256",
                      "python_source_tree_sha256", "environment_lock_sha256",
                      "target_patient_map_sha256", "label_classes", "partitions",
                      "target_test_sample_ids", "target_test_patient_ids"):
            if run[field] != reference[field]:
                raise ValueError(f"Grid candidates differ on {field}: {reference['name']} vs {run['name']}")
    manifest = {
        "result": "passed; all runs label-withheld and share cohort, code/environment, and patient split",
        "frozen_spec": "frozen_spec.json",
        "runs": runs,
        "target_test_labels_scored": False,
    }
    manifest_path = run_root / "withheld_grid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": manifest["result"], "manifest": str(manifest_path.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
