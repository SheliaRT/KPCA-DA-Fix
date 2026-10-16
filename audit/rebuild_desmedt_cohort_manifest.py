"""Recover GSE7390 sample annotations and audit local Desmedt sample mapping.

This script does not choose survival labels. GEO exposes overall-survival and
distant-metastasis-free endpoints, while the manuscript says disease-specific
survival. It emits candidate threshold counts and a sample-level metadata
mapping so that endpoint differences remain explicit rather than silently
substituted.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE7nnn/GSE7390/"
    "matrix/GSE7390_series_matrix.txt.gz"
)
DEFAULT_MATRIX = ROOT / "Data/Survival_TCGA_VS_Desmd/GSE7390_series_matrix.txt.gz"
DEFAULT_OUTPUT = ROOT / "audit/desmedt_cohort_manifest.json"
LOCAL_CANDIDATES = {
    "filtered_desmd_average": ROOT / "Data/Survival_TCGA_VS_Desmd/filtered_desmd_average.csv",
    "filtered_desmd_average_ver1": ROOT / "Data/Survival_TCGA_VS_Desmd/filtered_desmd_average_ver1.csv",
    "tcgavsdesmd_filtered": ROOT / "Data/TCGAvsDesmd/filtered_desmd_average.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_if_missing(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(GEO_URL, headers={"User-Agent": "KPCA-DA-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)


def parse_series_matrix(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    digest = sha256(path)
    open_file = gzip.open if path.suffix == ".gz" else open
    samples: dict[str, dict[str, str]] = {}
    with open_file(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        fields: dict[str, list[str]] = {}
        characteristics: list[list[str]] = []
        for row in reader:
            if not row:
                continue
            key = row[0]
            if key == "!series_matrix_table_begin":
                break
            if key in {"!Sample_geo_accession", "!Sample_title"}:
                fields[key] = row[1:]
            elif key.startswith("!Sample_characteristics_ch1"):
                characteristics.append(row[1:])
        accessions = fields.get("!Sample_geo_accession", [])
        titles = fields.get("!Sample_title", [])
        if not accessions or len(accessions) != len(titles):
            raise ValueError("GSE7390 series matrix has missing or misaligned sample accession/title rows")
        for index, accession in enumerate(accessions):
            entry = {"geo_accession": accession, "title": titles[index]}
            for values in characteristics:
                if index < len(values):
                    value = values[index]
                    if ":" in value:
                        name, item = value.split(":", 1)
                        entry[name.strip().lower()] = item.strip()
            samples[titles[index]] = entry
    return samples, digest


def local_sample_headers(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) < 2:
        raise ValueError(f"Expected gene-by-sample matrix: {path}")
    return [item.strip() for item in header[1:]]


def numeric(metadata: dict[str, str], name: str) -> float | None:
    value = metadata.get(name)
    if value is None or value.lower() in {"", "na", "nan", "null", "not available"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def endpoint_counts(records: list[dict[str, object]], time_key: str, event_key: str) -> dict[str, object]:
    upper = 10 * 365.25
    lower = 5 * 365.25
    aggressive = []
    indolent = []
    excluded = []
    for record in records:
        time = record.get(time_key)
        event = record.get(event_key)
        if not isinstance(time, (float, int)) or not isinstance(event, (float, int)):
            excluded.append(record["title"])
        elif int(event) == 1 and time <= lower:
            aggressive.append(record["title"])
        elif time >= upper:
            indolent.append(record["title"])
        else:
            excluded.append(record["title"])
    return {
        "rule": f"aggressive={event_key}==1 and {time_key}<=5y; indolent={time_key}>=10y; otherwise excluded",
        "aggressive_n": len(aggressive),
        "indolent_n": len(indolent),
        "excluded_n": len(excluded),
        "aggressive_ids": aggressive,
        "indolent_ids": indolent,
        "excluded_ids": excluded,
    }


def build_manifest(matrix_path: Path, output: Path) -> dict[str, object]:
    download_if_missing(matrix_path)
    geo, matrix_hash = parse_series_matrix(matrix_path)
    titles = set(geo)
    local: dict[str, object] = {}
    union: dict[str, dict[str, object]] = {}
    for name, path in LOCAL_CANDIDATES.items():
        ids = local_sample_headers(path)
        matched = [sample for sample in ids if sample in geo]
        records = []
        for sample in matched:
            raw = geo[sample]
            record: dict[str, object] = {"title": sample, "geo_accession": raw["geo_accession"]}
            for key in ("t.rfs", "e.rfs", "t.os", "e.os", "t.dmfs", "e.dmfs", "t.tdm", "e.tdm", "id"):
                record[key] = numeric(raw, key)
            records.append(record)
            union[sample] = record
        local[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "sample_count": len(ids),
            "geo_title_match_count": len(matched),
            "unmatched_sample_ids": [sample for sample in ids if sample not in geo],
            "matched_records": records,
        }

    all_records = list(union.values())
    candidate_rules = {}
    endpoints = (("t.rfs", "e.rfs"), ("t.os", "e.os"),
                 ("t.dmfs", "e.dmfs"), ("t.tdm", "e.tdm"))
    for time_key, event_key in endpoints:
        candidate_rules[f"{time_key}/{event_key}"] = endpoint_counts(all_records, time_key, event_key)
    candidate_rules_by_matrix = {}
    for name, matrix in local.items():
        matrix_records = matrix["matched_records"]
        candidate_rules_by_matrix[name] = {
            f"{time_key}/{event_key}": endpoint_counts(matrix_records, time_key, event_key)
            for time_key, event_key in endpoints
        }
    manifest: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "authority": "NCBI GEO GSE7390 series matrix",
        "source_url": GEO_URL,
        "source_sha256": matrix_hash,
        "geo_sample_count": len(geo),
        "local_matrices": local,
        "candidate_threshold_rules_over_unique_local_samples": candidate_rules,
        "candidate_threshold_rules_by_local_matrix": candidate_rules_by_matrix,
        "endpoint_caveat": (
            "The paper says disease-specific survival. GEO supplies overall survival and distant-metastasis-free endpoints; "
            "these candidates are not interchangeable with disease-specific survival. Counts are diagnostics, not adopted labels."
        ),
        "paper_cohort_counts": {"desmedt_patients": 164},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest(args.matrix, args.output)
    print(json.dumps({
        "output": str(args.output),
        "geo_sample_count": manifest["geo_sample_count"],
        "local_match_counts": {
            name: values["geo_title_match_count"]
            for name, values in manifest["local_matrices"].items()
        },
        "candidate_rules": {
            name: {key: value for key, value in rule.items() if not key.endswith("_ids")}
            for name, rule in manifest["candidate_threshold_rules_over_unique_local_samples"].items()
        },
        "endpoint_caveat": manifest["endpoint_caveat"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
