"""Verify both PAM50 directions against the official sample-to-patient map."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from experiment_protocol import make_patient_splits, patient_groups_from_mapping
from kpcada import load_expression_dataset

MAPS = {
    "smc": ROOT / "audit" / "pam50_smc_sample_patient.json",
    "cptac": ROOT / "audit" / "pam50_cptac_sample_patient.json",
}
FILES = {
    "smc": ROOT / "Data" / "cptacvssmc" / "smc_labels.csv",
    "cptac": ROOT / "Data" / "cptacvssmc" / "cptac_labels.csv",
}
OUTPUT = ROOT / "audit" / "pam50_patient_split_validation.json"
RUNNER = ROOT / "Scripts" / "run_tvsn_experiment.py"
PYTHON = ROOT / ".venv" / "bin" / "python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cohort(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_gxp, _genes, sample_ids, labels = load_expression_dataset(str(FILES[name]))
    ids = np.asarray(sample_ids, dtype=str)
    mapping = json.loads(MAPS[name].read_text(encoding="utf-8"))
    groups = patient_groups_from_mapping(ids, mapping).astype(str)
    if len(np.unique(groups)) != len(ids):
        raise ValueError(
            f"{name} contains repeated patient groups; split must keep each patient together"
        )
    return X_gxp, ids, np.asarray(labels), groups


def main() -> int:
    cohorts = {name: load_cohort(name) for name in FILES}
    directions = (
        ("smc_to_cptac", "smc", "cptac"),
        ("cptac_to_smc", "cptac", "smc"),
    )
    results: dict[str, object] = {}
    for name, source_name, target_name in directions:
        _Xs, source_ids, source_labels, source_groups = cohorts[source_name]
        _Xt, target_ids, _target_labels, target_groups = cohorts[target_name]
        splits = make_patient_splits(
            source_labels,
            source_ids=source_groups,
            target_ids=target_groups,
            source_validation_size=0.2,
            target_test_size=0.3,
            random_state=46,
        )
        assert not (set(source_groups[splits.source_train])
                    & set(source_groups[splits.source_validation]))
        assert not (set(target_groups[splits.target_adapt])
                    & set(target_groups[splits.target_test]))
        command = [
            str(PYTHON), str(RUNNER), "--arm", "baseline", "--split-only",
            "--allow-multiclass", "--seed", "46", "--task-name", f"pam50_{name}",
            "--source-path", str(FILES[source_name].relative_to(ROOT)),
            "--target-path", str(FILES[target_name].relative_to(ROOT)),
            "--source-patient-map", str(MAPS[source_name].relative_to(ROOT)),
            "--target-patient-map", str(MAPS[target_name].relative_to(ROOT)),
            "--exclude-source-labels", "Unknown", "--feature-scope", "all",
        ]
        runner_result = subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True
        )
        runner_report = json.loads(runner_result.stdout)
        if runner_report.get("result") != "patient split validation passed":
            raise ValueError(f"Runner did not confirm split validation for {name}")
        if runner_report["source_patient_count"] != len(np.unique(source_groups)):
            raise ValueError(f"Runner source patient count differs for {name}")
        if runner_report["target_patient_count"] != len(np.unique(target_groups)):
            raise ValueError(f"Runner target patient count differs for {name}")
        results[name] = {
            "source_samples": len(source_ids),
            "source_patients": len(np.unique(source_groups)),
            "target_samples": len(target_ids),
            "target_patients": len(np.unique(target_groups)),
            "source_train_samples": len(splits.source_train),
            "source_validation_samples": len(splits.source_validation),
            "target_adaptation_samples": len(splits.target_adapt),
            "target_test_samples": len(splits.target_test),
            "source_train_patients": source_groups[splits.source_train].tolist(),
            "source_validation_patients": source_groups[splits.source_validation].tolist(),
            "target_adaptation_patients": target_groups[splits.target_adapt].tolist(),
            "target_test_patients": target_groups[splits.target_test].tolist(),
            "target_labels_used_for_split": False,
            "patient_overlap": 0,
            "canonical_runner_split_validation": runner_report,
        }
    report = {
        "result": "passed",
        "seed": 46,
        "patient_maps": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in MAPS.items()
        },
        "cohort_files": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in FILES.items()
        },
        "directions": results,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": "passed",
        "output": str(OUTPUT.relative_to(ROOT)),
        "directions": {
            name: {
                "source_patients": value["source_patients"],
                "target_patients": value["target_patients"],
                "patient_overlap": value["patient_overlap"],
            }
            for name, value in results.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
