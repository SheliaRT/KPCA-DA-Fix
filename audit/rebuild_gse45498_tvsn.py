"""Rebuild an auditable GSE45498 primary/normal matrix from GEO records.

The repository's `GSE45498_NvT.csv` values match the official supplementary
non-normalized matrix, but duplicate sample headers have lost identity and 52
stored labels disagree with the official sample profiles. This utility joins
the supplementary expression columns to the GPL16299 Series Matrix by their
236-gene expression fingerprints, keeps primary and normal breast samples,
and writes unique GEO sample IDs plus metadata-derived patient IDs/labels.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPPLEMENT = ROOT / "Data/TvsN/GSE45498_mRNA_non-normalized_data.txt.gz"
DEFAULT_SERIES = ROOT / "Data/TvsN/GSE45498-GPL16299_series_matrix.txt.gz"
DEFAULT_EXISTING = ROOT / "Data/TvsN/GSE45498_NvT.csv"
DEFAULT_OUTPUT = ROOT / "Data/TvsN/GSE45498_TvsN_official_200.csv"
DEFAULT_MANIFEST = ROOT / "audit/gse45498_tvsn_rebuild_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_supplement(path: Path) -> tuple[list[str], list[str], list[list[str]], np.ndarray]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if len(header) < 4:
            raise ValueError("GEO supplementary file has an unexpected header")
        sample_ids = [value.strip() for value in header[3:]]
        rows = [row for row in reader if row and row[0] == "Endogenous"]
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError("GEO endogenous feature rows have inconsistent widths")
    genes = [row[1] for row in rows]
    if len(set(genes)) != len(genes):
        raise ValueError("GEO endogenous gene names are not unique")
    values = np.asarray([
        [float(value) if value.strip() else np.nan for value in row[3:]]
        for row in rows
    ], dtype=float)
    return genes, sample_ids, rows, values


def read_series(path: Path) -> tuple[dict[str, list[str]], list[list[str]], dict[str, list[str]]]:
    metadata: dict[str, list[str]] = {}
    characteristics: dict[str, list[str]] = {}
    matrix_rows: list[list[str]] = []
    in_table = False
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quotechar='"')
        for row in reader:
            if not row:
                continue
            if row[0] == "!series_matrix_table_begin":
                in_table = True
                continue
            if row[0] == "!series_matrix_table_end":
                break
            if in_table:
                matrix_rows.append(row)
            elif row[0] == "!Sample_characteristics_ch1":
                key = row[1].split(":", 1)[0].strip()
                characteristics[key] = [
                    value.split(":", 1)[1].strip() if ":" in value else value
                    for value in row[1:]
                ]
            elif row[0] in {
                "!Sample_geo_accession", "!Sample_title", "!Sample_source_name_ch1"
            }:
                metadata[row[0].removeprefix("!")] = row[1:]
    required = {"Sample_geo_accession", "Sample_title"}
    if not required.issubset(metadata) or "status" not in characteristics:
        raise ValueError("GEO Series Matrix lacks sample IDs, titles, or status metadata")
    if not matrix_rows:
        raise ValueError("GEO Series Matrix has no expression table")
    n = len(metadata["Sample_geo_accession"])
    if any(len(row) != n + 1 for row in matrix_rows):
        raise ValueError("GEO Series Matrix expression table width differs from metadata")
    if any(len(values) != n for values in characteristics.values()):
        raise ValueError("GEO Series Matrix sample characteristics are inconsistent")
    return metadata, matrix_rows, characteristics


def expression_fingerprint_match(
    genes: list[str], raw_values: np.ndarray, matrix_rows: list[list[str]],
) -> tuple[np.ndarray, np.ndarray]:
    matrix_by_gene = {row[0]: row[1:] for row in matrix_rows[1:]}
    if set(genes) - set(matrix_by_gene):
        raise ValueError("Series Matrix does not contain every supplementary endogenous gene")
    processed = np.asarray([
        [float(value) if value.strip() else np.nan for value in matrix_by_gene[gene]]
        for gene in genes
    ], dtype=float)
    if processed.shape[0] != raw_values.shape[0]:
        raise ValueError("Raw and processed matrices have different gene dimensions")

    # GEO's processed table differs in scale from the non-normalized file.
    # Correlation across the same 236 genes identifies sample columns without
    # relying on duplicate-prone display names.
    for values in (raw_values, processed):
        means = np.nanmean(values, axis=1)
        for row_index in range(values.shape[0]):
            values[row_index, np.isnan(values[row_index])] = means[row_index]
    raw_sd = np.maximum(raw_values.std(axis=0), 1e-12)
    processed_sd = np.maximum(processed.std(axis=0), 1e-12)
    raw_z = (raw_values - raw_values.mean(axis=0)) / raw_sd
    processed_z = (processed - processed.mean(axis=0)) / processed_sd
    correlation = raw_z.T @ processed_z / len(genes)
    best_index = correlation.argmax(axis=1)
    best_correlation = correlation.max(axis=1)
    return best_index, best_correlation


def build(args: argparse.Namespace) -> dict[str, Any]:
    genes, raw_ids, raw_rows, raw_values = read_supplement(args.supplement)
    metadata, matrix_rows, characteristics = read_series(args.series_matrix)
    if len(raw_ids) != len(metadata["Sample_geo_accession"]):
        raise ValueError("Supplementary and Series Matrix sample counts differ")
    best_index, confidence = expression_fingerprint_match(genes, raw_values, matrix_rows)
    official_status = characteristics["status"]

    selected: list[int] = []
    labels: list[str] = []
    patients: list[str] = []
    official_samples: list[dict[str, Any]] = []
    metastatic_indices = [i for i, sample_id in enumerate(raw_ids)
                          if sample_id.startswith("TAS10-15-")]
    for raw_index, raw_id in enumerate(raw_ids):
        if raw_index in set(metastatic_indices):
            continue
        series_index = int(best_index[raw_index])
        status = official_status[series_index].strip().lower()
        if status == "normal":
            label = "Normal"
        elif status == "primary triple negative cancer":
            label = "Tumor"
        else:
            raise ValueError(
                f"Non-metastatic raw sample {raw_id!r} maps to unexpected status {status!r}"
            )
        if confidence[raw_index] < args.minimum_correlation:
            raise ValueError(
                f"Sample profile match is ambiguous for {raw_id!r}: "
                f"r={confidence[raw_index]:.4f}"
            )
        title = metadata["Sample_title"][series_index].removesuffix("_mRNA")
        match = re.match(r"^(\d+)(?:N|T)(?:\s|$)", title)
        if not match:
            raise ValueError(f"Cannot derive patient number from official title {title!r}")
        patient = match.group(1)
        selected.append(raw_index)
        labels.append(label)
        patients.append(patient)
        official_samples.append({
            "raw_column_index": raw_index,
            "supplementary_sample_name": raw_id,
            "gsm": metadata["Sample_geo_accession"][series_index],
            "series_title": metadata["Sample_title"][series_index],
            "status": status,
            "label": label,
            "patient_id": patient,
            "profile_correlation": float(confidence[raw_index]),
        })

    if len(set(sample["gsm"] for sample in official_samples)) != len(official_samples):
        raise ValueError("Expression fingerprint mapping did not produce unique GEO samples")
    counts = Counter(labels)
    if counts != Counter({"Tumor": 160, "Normal": 40}):
        raise ValueError(f"Unexpected primary/normal mRNA sample counts: {counts}")
    # Keep profile/sample counts separate from patient counts. GEO has 160
    # primary tumor profiles, while two patient IDs each have two profiles.
    # This may explain the article's reported 158 tumors as a patient count,
    # but it does not prove that the article counted unique patients.
    tumor_profiles_by_patient: dict[str, list[dict[str, Any]]] = {}
    for sample in official_samples:
        if sample["label"] == "Tumor":
            tumor_profiles_by_patient.setdefault(sample["patient_id"], []).append(sample)
    repeated_tumor_profiles = [
        {
            "patient_id": patient_id,
            "profile_count": len(samples),
            "samples": [
                {"gsm": sample["gsm"], "series_title": sample["series_title"]}
                for sample in samples
            ],
        }
        for patient_id, samples in sorted(tumor_profiles_by_patient.items())
        if len(samples) > 1
    ]
    unique_patients = len(set(patients))
    tumor_patient_count = len(tumor_profiles_by_patient)
    if (unique_patients, tumor_patient_count, len(repeated_tumor_profiles)) != (158, 158, 2):
        raise ValueError(
            "Unexpected patient/profile reconciliation: "
            f"{unique_patients} total patients, {tumor_patient_count} tumor patients, "
            f"{len(repeated_tumor_profiles)} repeated tumor patients"
        )

    # Audit the previous matrix by exact expression vectors, then compare its
    # stored labels to official status. It is not used to assign new labels.
    old_audit: dict[str, Any] = {"audited": False}
    if args.existing and args.existing.exists():
        import sys
        sys.path.insert(0, str(ROOT / "Scripts"))
        from kpcada import load_expression_dataset

        old_values, old_genes, old_ids, old_labels = load_expression_dataset(str(args.existing))
        if list(old_genes) != genes:
            raise ValueError("Existing target feature order differs from GEO supplement")
        raw_lookup: dict[bytes, list[int]] = {}
        for column in range(raw_values.shape[1]):
            raw_lookup.setdefault(raw_values[:, column].tobytes(), []).append(column)
        errors = []
        mapped = 0
        for column, (sample_id, label) in enumerate(zip(old_ids, old_labels)):
            candidates = raw_lookup.get(np.asarray(old_values[:, column], dtype=float).tobytes(), [])
            if len(candidates) != 1:
                raise ValueError(f"Cannot uniquely map existing sample {sample_id!r} to GEO values")
            raw_index = candidates[0]
            if raw_index in set(metastatic_indices):
                official_label = "Metastatic"
            else:
                status = official_status[int(best_index[raw_index])].strip().lower()
                official_label = "Normal" if status == "normal" else "Tumor"
            mapped += 1
            if str(label) != official_label:
                errors.append({
                    "existing_sample_id": str(sample_id),
                    "stored_label": str(label),
                    "official_label": official_label,
                    "official_title": metadata["Sample_title"][int(best_index[raw_index])],
                    "gsm": metadata["Sample_geo_accession"][int(best_index[raw_index])],
                })
        old_audit = {
            "audited": True,
            "existing_sha256": sha256(args.existing),
            "samples_exactly_matched_to_official_supplement": mapped,
            "stored_label_disagreements": len(errors),
            "disagreement_examples": errors[:100],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = [
        f"GSE45498:{sample['patient_id']}:{sample['gsm']}"
        for sample in official_samples
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene", *sample_ids])
        for row_index, gene in enumerate(genes):
            writer.writerow([gene, *[raw_rows[row_index][3 + index] for index in selected]])
        writer.writerow(["Sample_Type", *labels])

    manifest = {
        "result": "passed",
        "source": "NCBI GEO GSE45498 supplementary non-normalized mRNA data plus GPL16299 Series Matrix sample metadata",
        "source_urls": {
            "supplement": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE45nnn/GSE45498/suppl/GSE45498_mRNA_non-normalized_data.txt.gz",
            "series_matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE45nnn/GSE45498/matrix/GSE45498-GPL16299_series_matrix.txt.gz",
        },
        "supplement_sha256": sha256(args.supplement),
        "series_matrix_sha256": sha256(args.series_matrix),
        "output_sha256": sha256(args.output),
        "output_path": str(args.output),
        "gene_count": len(genes),
        "sample_count": len(selected),
        "label_counts": dict(counts),
        "unique_patient_count": unique_patients,
        "tumor_profile_patient_reconciliation": {
            "tumor_profile_count": counts["Tumor"],
            "unique_tumor_patient_count": tumor_patient_count,
            "additional_profiles_beyond_one_per_patient": counts["Tumor"] - tumor_patient_count,
            "repeated_tumor_profiles": repeated_tumor_profiles,
            "paper_158_count_interpretation": (
                "The public GEO data contain 160 primary tumor profiles from 158 patient IDs. "
                "This is consistent with, but does not prove, interpreting the article's 158 "
                "tumors as unique patients; the article's counting unit and exclusions remain unverified."
            ),
        },
        "excluded_metastatic_samples": len(metastatic_indices),
        "minimum_profile_correlation": float(min(item["profile_correlation"] for item in official_samples)),
        "sample_identity_and_label_disagreements_in_previous_file": old_audit,
        "samples": official_samples,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "samples"}, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--series-matrix", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--minimum-correlation", type=float, default=0.8)
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
