"""Build a patient-paired external TvsN target from GEO GSE183947.

The official supplementary matrix uses submitter IDs ``CA.<donor>`` for
tumors and ``CAP.<donor>`` for paired normal tissue. This utility crosswalks
those IDs to the donor and tissue fields in the official GEO family SOFT,
then writes the canonical expression CSV and explicit sample-to-patient map
used by the matched runner. It does not split patients or fit a model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "Data/TvsN/GSE183947_fpkm.csv.gz"
DEFAULT_SOFT = ROOT / "Data/TvsN/GSE183947_family.soft.gz"
DEFAULT_OUTPUT = ROOT / "Data/TvsN/GSE183947_TvsN_official_60.csv"
DEFAULT_MANIFEST = ROOT / "audit/gse183947_external_target_manifest.json"
DEFAULT_PATIENT_MAP = ROOT / "audit/gse183947_patient_map.json"
SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE183nnn/GSE183947/"
    "soft/GSE183947_family.soft.gz"
)
MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE183nnn/GSE183947/"
    "suppl/GSE183947_fpkm.csv.gz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_if_missing(path: Path, url: str) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "KPCA-DA-audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)


def parse_geo_samples(path: Path) -> dict[str, dict[str, str]]:
    """Return GSM -> donor/tissue/title fields from official GEO SOFT."""
    samples: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line.startswith("^SAMPLE = "):
                if current is not None:
                    samples[current["gsm"]] = current
                current = {"gsm": line.split("=", 1)[1].strip()}
                continue
            if current is None or not line.startswith("!Sample_"):
                continue
            key, _, value = line.partition(" = ")
            value = value.strip().strip('"')
            if key == "!Sample_title":
                current["title"] = value
            elif key == "!Sample_characteristics_ch1":
                field, sep, content = value.partition(":")
                if sep:
                    field = field.strip().lower()
                    if field in {"donor", "tissue"}:
                        current[field] = content.strip()
        if current is not None:
            samples[current["gsm"]] = current
    required = {"gsm", "donor", "tissue", "title"}
    incomplete = [gsm for gsm, record in samples.items() if not required <= record.keys()]
    if incomplete:
        raise ValueError(f"GEO SOFT has incomplete donor/tissue metadata: {incomplete[:5]}")
    return samples


def read_expression(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Read the supplied gene-by-sample FPKM matrix without changing values."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) < 2:
            raise ValueError("FPKM matrix has no sample columns")
        sample_ids = header[1:]
        if len(sample_ids) != 60 or len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"Expected 60 unique sample columns; got {len(sample_ids)}")
        genes: list[str] = []
        values_by_gene: dict[str, list[str]] = {}
        for line_no, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Malformed FPKM row {line_no}: expected {len(header)} columns")
            gene = row[0].strip()
            if not gene or gene in values_by_gene:
                raise ValueError(f"Empty or duplicate gene symbol at row {line_no}: {gene!r}")
            values = [float(x) for x in row[1:]]
            if any(not (v >= 0 and v < float("inf")) for v in values):
                raise ValueError(f"Non-finite or negative FPKM value for {gene}")
            genes.append(gene)
            values_by_gene[gene] = row[1:]
    return sample_ids, {"genes": genes, "values": values_by_gene}


def build(matrix_path: Path, soft_path: Path, output_path: Path,
          manifest_path: Path, patient_map_path: Path) -> dict[str, Any]:
    fetch_if_missing(soft_path, SOFT_URL)
    if not matrix_path.is_file():
        fetch_if_missing(matrix_path, MATRIX_URL)

    geo = parse_geo_samples(soft_path)
    sample_ids, expression = read_expression(matrix_path)
    genes = expression["genes"]
    values_by_gene = expression["values"]
    source_columns: dict[tuple[str, str], str] = {}
    for sample_id in sample_ids:
        match = re.fullmatch(r"(CA|CAP)\.(\d+)", sample_id)
        if not match:
            raise ValueError(f"Unexpected GSE183947 submitter sample ID: {sample_id}")
        tissue_code, donor = match.groups()
        source_columns[(tissue_code, donor)] = sample_id

    donor_to_geo: dict[tuple[str, str], str] = {}
    for gsm, record in geo.items():
        donor = record["donor"].strip()
        tissue_text = record["tissue"].strip().lower()
        if "tumor" in tissue_text or "cancer" in tissue_text:
            label, tissue_code = "Tumor", "CA"
        elif "normal" in tissue_text:
            label, tissue_code = "Normal", "CAP"
        else:
            raise ValueError(f"Unrecognized GEO tissue description for {gsm}: {tissue_text!r}")
        key = (tissue_code, donor)
        if key in donor_to_geo:
            raise ValueError(f"Multiple GEO samples for donor/tissue {key}")
        donor_to_geo[key] = gsm

    source_keys = set(source_columns)
    geo_keys = set(donor_to_geo)
    if source_keys != geo_keys:
        raise ValueError(
            f"Submitter/GEO donor crosswalk differs: matrix-only={sorted(source_keys-geo_keys)[:5]}, "
            f"GEO-only={sorted(geo_keys-source_keys)[:5]}"
        )
    tumor_donors = {donor for code, donor in source_keys if code == "CA"}
    normal_donors = {donor for code, donor in source_keys if code == "CAP"}
    if len(tumor_donors) != 30 or len(normal_donors) != 30:
        raise ValueError("Expected 30 GEO tumor and 30 GEO normal donor records")
    paired_donors = tumor_donors & normal_donors
    tumor_only_donors = sorted(tumor_donors - normal_donors)
    normal_only_donors = sorted(normal_donors - tumor_donors)

    # Preserve GEO's authoritative sample order, including pair members.
    records: list[tuple[str, str, str, str]] = []
    for gsm, geo_record in geo.items():
        tissue_text = geo_record["tissue"].lower()
        label = "Tumor" if ("tumor" in tissue_text or "cancer" in tissue_text) else "Normal"
        code = "CA" if label == "Tumor" else "CAP"
        donor = geo_record["donor"].strip()
        records.append((gsm, donor, label, source_columns[(code, donor)]))
    records.sort(key=lambda row: int(row[0].removeprefix("GSM")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene", *[row[0] for row in records]])
        sample_positions = [sample_ids.index(row[3]) for row in records]
        for gene in genes:
            vals = values_by_gene[gene]
            writer.writerow([gene, *[vals[i] for i in sample_positions]])
        writer.writerow(["Sample_Type", *[row[2] for row in records]])

    patient_map = {
        row[0]: f"GSE183947:{row[1]}" for row in records
    }
    patient_map_path.parent.mkdir(parents=True, exist_ok=True)
    patient_map_path.write_text(json.dumps(patient_map, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    counts = Counter(row[2] for row in records)
    manifest = {
        "result": "passed",
        "cohort": "independent paired breast tumor/normal RNA-seq target",
        "geo_accession": "GSE183947",
        "source_urls": {"expression_supplement": MATRIX_URL, "sample_metadata": SOFT_URL},
        "matrix_path": str(matrix_path.relative_to(ROOT) if matrix_path.is_relative_to(ROOT) else matrix_path),
        "matrix_sha256": sha256(matrix_path),
        "geo_soft_path": str(soft_path.relative_to(ROOT) if soft_path.is_relative_to(ROOT) else soft_path),
        "geo_soft_sha256": sha256(soft_path),
        "output_path": str(output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path),
        "output_sha256": sha256(output_path),
        "patient_map_path": str(patient_map_path.relative_to(ROOT) if patient_map_path.is_relative_to(ROOT) else patient_map_path),
        "patient_map_sha256": sha256(patient_map_path),
        "expression_measure": "FPKM as supplied; values were not transformed",
        "gene_count": len(genes),
        "unique_gene_count": len(set(genes)),
        "sample_count": len(records),
        "label_counts": dict(counts),
        "patient_count": len(tumor_donors | normal_donors),
        "complete_paired_patient_count": len(paired_donors),
        "tumor_only_donor_ids": tumor_only_donors,
        "normal_only_donor_ids": normal_only_donors,
        "sample_metadata_crosswalk": "GEO GSM donor/tissue characteristics matched exactly to CA.<donor>/CAP.<donor> matrix columns",
        "samples": [
            {"gsm": gsm, "donor_id": donor, "label": label,
             "matrix_sample_id": matrix_id}
            for gsm, donor, label, matrix_id in records
        ],
        "crosswalk_checks": {
            "all_60_matrix_columns_matched_to_unique_geo_samples": len(records) == 60,
            "30_tumor_and_30_normal_records_crosswalked": len(records) == 60,
            "all_records_grouped_by_official_donor_characteristic": True,
            "all_expression_values_finite_and_nonnegative": True,
            "target_labels_used_for_patient_mapping": False,
            "patient_grouping_source": "official GEO donor characteristic",
        },
        "evaluation_note": (
            "This is an independent external development cohort, not a paper cohort. "
            "The GEO series description says there are 30 pairs, but detailed donor "
            "characteristics yield 29 complete pairs plus a tumor-only donor "
            "(2045012) and a normal-only donor (2046641). Do not infer a correction. "
            "Group samples by the official donor characteristics before splitting. "
            "Keep test labels withheld until the candidate and preprocessing policy are frozen."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--soft", type=Path, default=DEFAULT_SOFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--patient-map", type=Path, default=DEFAULT_PATIENT_MAP)
    args = parser.parse_args()
    print(json.dumps(build(args.matrix, args.soft, args.output,
                           args.manifest, args.patient_map), indent=2))


if __name__ == "__main__":
    main()
