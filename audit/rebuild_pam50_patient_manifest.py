"""Reconcile local PAM50 sample IDs with cBioPortal patient identifiers."""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "smc": {
        "study_id": "brca_smc_2018",
        "expression_path": "Data/cptacvssmc/smc_labels.csv",
    },
    "cptac": {
        "study_id": "brca_cptac_2020",
        "expression_path": "Data/cptacvssmc/cptac_labels.csv",
    },
}
BASE_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
OUTPUT = ROOT / "audit" / "pam50_patient_manifest.json"
MAP_OUTPUTS = {
    "smc": ROOT / "audit" / "pam50_smc_sample_patient.json",
    "cptac": ROOT / "audit" / "pam50_cptac_sample_patient.json",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_local_samples(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [str(value).strip() for value in next(csv.reader(stream))[1:]]


def canonical_subtype(label: str) -> str:
    compact = "".join(character for character in label.casefold() if character.isalnum())
    return {"luma": "luminala", "lumb": "luminalb"}.get(compact, compact)


def read_local_labels(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        last_row = None
        for last_row in reader:
            pass
    if not last_row or len(last_row) != len(header):
        raise ValueError(f"Local label row does not align to sample columns in {path}")
    return {str(sample).strip(): str(label).strip()
            for sample, label in zip(header[1:], last_row[1:])}


def get_clinical_samples(study_id: str) -> tuple[bytes, dict[str, str], dict[str, str], str]:
    url = f"{BASE_URL}/{study_id}/data_clinical_sample.txt"
    request = Request(url, headers={"User-Agent": "KPCA-DA-paper-code-audit/1.0"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig")), delimiter="\t"))
    if len(rows) < 6:
        raise ValueError(f"Unexpected cBioPortal clinical-sample file for {study_id}")
    header = rows[4]
    try:
        sample_index = header.index("SAMPLE_ID")
        patient_index = header.index("PATIENT_ID")
    except ValueError as error:
        raise ValueError(f"Missing SAMPLE_ID/PATIENT_ID in {study_id} metadata") from error
    subtype_field = next(
        (field for field in ("PAM50_SUBTYPE", "PAM50") if field in header), None
    )
    if subtype_field is None:
        raise ValueError(f"Missing PAM50 subtype field in {study_id} metadata")
    subtype_index = header.index(subtype_field)
    result: dict[str, str] = {}
    subtypes: dict[str, str] = {}
    for row in rows[5:]:
        if len(row) <= max(sample_index, patient_index):
            continue
        sample_id, patient_id = row[sample_index].strip(), row[patient_index].strip()
        if sample_id in result:
            raise ValueError(f"Duplicate sample ID {sample_id!r} in {study_id}")
        result[sample_id] = patient_id
        subtypes[sample_id] = row[subtype_index].strip()
    return content, result, subtypes, subtype_field


def main() -> int:
    cohorts: dict[str, object] = {}
    for name, config in COHORTS.items():
        local_path = ROOT / config["expression_path"]
        local_ids = read_local_samples(local_path)
        if len(local_ids) != len(set(local_ids)):
            raise ValueError(f"Duplicate local sample IDs in {config['expression_path']}")
        clinical_bytes, official_map, official_subtypes, subtype_field = get_clinical_samples(
            config["study_id"]
        )
        missing = sorted(set(local_ids) - set(official_map))
        if missing:
            raise ValueError(
                f"{name} local expression samples are absent from cBioPortal: {missing[:10]}"
            )
        patient_map = {sample: official_map[sample] for sample in local_ids}
        patient_ids = list(patient_map.values())
        samples_per_patient = Counter(Counter(patient_ids).values())
        local_labels = read_local_labels(local_path)
        official_labels = {sample: official_subtypes[sample] for sample in local_ids}
        not_in_local = sorted(set(official_map) - set(local_ids))
        excluded_official_labels = Counter(official_subtypes[sample] for sample in not_in_local)
        label_mismatches = [
            {
                "sample_id": sample,
                "local_label": local_labels[sample],
                "official_label": official_labels[sample],
            }
            for sample in local_ids
            if canonical_subtype(local_labels[sample])
            != canonical_subtype(official_labels[sample])
        ]
        if label_mismatches:
            raise ValueError(
                f"{name} local PAM50 labels differ from official metadata: "
                f"{label_mismatches[:5]}"
            )
        cohorts[name] = {
            "study_id": config["study_id"],
            "expression_path": config["expression_path"],
            "expression_sha256": file_sha256(local_path),
            "local_sample_count": len(local_ids),
            "official_study_sample_count": len(official_map),
            "official_samples_not_in_local_subset": len(not_in_local),
            "official_not_in_local_pam50_label_counts": dict(
                sorted(excluded_official_labels.items())
            ),
            "official_not_in_local_sample_ids_by_pam50_label": {
                label: sorted(
                    sample for sample in not_in_local
                    if official_subtypes[sample] == label
                )
                for label in sorted(excluded_official_labels)
            },
            "official_sample_metadata_url": (
                f"{BASE_URL}/{config['study_id']}/data_clinical_sample.txt"
            ),
            "official_sample_metadata_sha256": sha256(clinical_bytes),
            "official_pam50_label_field": subtype_field,
            "local_pam50_label_counts": dict(sorted(Counter(local_labels.values()).items())),
            "official_pam50_label_counts_for_local_samples": dict(
                sorted(Counter(official_labels.values()).items())
            ),
            "pam50_label_mismatch_count": len(label_mismatches),
            "pam50_label_mismatch_examples": label_mismatches,
            "local_samples_missing_official_mapping": missing,
            "unique_patient_count_for_local_samples": len(set(patient_ids)),
            "samples_per_patient_counts": {
                str(sample_count): int(patient_count)
                for sample_count, patient_count in sorted(samples_per_patient.items())
            },
            "sample_to_patient": patient_map,
        }
        MAP_OUTPUTS[name].write_text(
            json.dumps(patient_map, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report = {
        "result": "passed",
        "mapping_source": "cBioPortal public datahub clinical sample tables",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohorts": cohorts,
        "interpretation": (
            "All local expression sample IDs must map to official patient IDs. "
            "The local SMC/CPTAC files currently map one sample per patient; the "
            "runner should consume this explicit mapping rather than assume it."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": "passed",
        "output": str(OUTPUT.relative_to(ROOT)),
        "patient_maps": {
            name: str(path.relative_to(ROOT)) for name, path in MAP_OUTPUTS.items()
        },
        "cohorts": {
            name: {
                "samples": values["local_sample_count"],
                "patients": values["unique_patient_count_for_local_samples"],
                "official_study_samples": values["official_study_sample_count"],
                "official_not_in_local_labels": values[
                    "official_not_in_local_pam50_label_counts"
                ],
                "label_mismatches": values["pam50_label_mismatch_count"],
                "multi_sample_patients": sum(
                    1 for count, patient_count in values["samples_per_patient_counts"].items()
                    if int(count) > 1 and patient_count > 0
                ),
            }
            for name, values in cohorts.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
