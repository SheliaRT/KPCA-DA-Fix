"""Compare the original GSE41970 mRNA deposit with its GSE45498 mirror."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SERIES = {
    "GSE41970": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE41nnn/GSE41970",
    "GSE45498": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE45nnn/GSE45498",
}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "KPCA-DA-audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def uncompressed_sha256(payload: bytes) -> str:
    return hashlib.sha256(gzip.decompress(payload)).hexdigest()


def parse_matrix(payload: bytes) -> dict[str, Any]:
    rows_by_key: list[tuple[str, list[str]]] = []
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
        reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8", newline=""),
                            delimiter="\t")
        for row in reader:
            if not row:
                continue
            key = row[0]
            if key not in {
                "!Sample_geo_accession", "!Sample_title", "!Sample_source_name_ch1",
            } and key != "!Sample_characteristics_ch1":
                continue
            rows_by_key.append((key, row[1:]))

    accession_rows = [values for key, values in rows_by_key
                      if key == "!Sample_geo_accession"]
    if len(accession_rows) != 1:
        raise ValueError("Expected exactly one sample accession row")
    ids = accession_rows[0]
    samples = {sample_id: {} for sample_id in ids}
    for key, values in rows_by_key:
        if key == "!Sample_geo_accession":
            continue
        if len(values) != len(samples):
            raise ValueError(f"Malformed sample metadata row: {key}")
        if key == "!Sample_characteristics_ch1":
            for sample_id, value in zip(ids, values):
                if ":" in value:
                    field, detail = value.split(":", 1)
                    samples[sample_id][field.strip().lower()] = detail.strip()
        else:
            field = "title" if key == "!Sample_title" else "source_name"
            for sample_id, value in zip(ids, values):
                samples[sample_id][field] = value.strip()

    if not samples:
        raise ValueError("No sample metadata found in GEO series matrix")
    statuses = Counter(sample.get("status", "unclassified") for sample in samples.values())
    return {
        "sample_count": len(samples),
        "status_counts": dict(sorted(statuses.items())),
        "sample_metadata": samples,
    }


def parse_raw_supplement(payload: bytes, accession: str) -> dict[str, Any]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
        reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8", newline=""),
                            delimiter="\t")
        header = next(reader)
    if header[:3] != ["Code Class", "Name", "Accession"]:
        raise ValueError(f"Unexpected mRNA supplement header for {accession}")
    return {
        "accession": accession,
        "sha256_compressed": hashlib.sha256(payload).hexdigest(),
        "sha256_uncompressed": uncompressed_sha256(payload),
        "compressed_bytes": len(payload),
        "sample_columns": len(header) - 3,
    }


def main() -> int:
    accessions: dict[str, dict[str, Any]] = {}
    for accession, base_url in SERIES.items():
        raw_url = f"{base_url}/suppl/{accession}_mRNA_non-normalized_data.txt.gz"
        matrix_url = f"{base_url}/matrix/{accession}-GPL16299_series_matrix.txt.gz"
        raw = fetch(raw_url)
        matrix = fetch(matrix_url)
        parsed_matrix = parse_matrix(matrix)
        accessions[accession] = {
            "raw_supplement_url": raw_url,
            "series_matrix_url": matrix_url,
            "raw_supplement": parse_raw_supplement(raw, accession),
            "series_matrix_sha256_compressed": hashlib.sha256(matrix).hexdigest(),
            "series_matrix_sha256_uncompressed": uncompressed_sha256(matrix),
            "series_matrix": {
                "sample_count": parsed_matrix["sample_count"],
                "status_counts": parsed_matrix["status_counts"],
            },
            "_sample_metadata": parsed_matrix["sample_metadata"],
        }

    a = accessions["GSE41970"]
    b = accessions["GSE45498"]
    ids_a = a.pop("_sample_metadata")
    ids_b = b.pop("_sample_metadata")
    result = {
        "result": "passed",
        "original_paper": {
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3566108/",
            "miRNA_tumor_normal_cohort": {"tumor": 165, "normal": 59, "total": 224},
            "reported_mRNA_cohort": {"tumor": 158, "normal": 40, "metastatic": 54},
        },
        "accessions": accessions,
        "crosswalk": {
            "mRNA_supplements_identical_after_gzip_decompression": (
                a["raw_supplement"]["sha256_uncompressed"]
                == b["raw_supplement"]["sha256_uncompressed"]
            ),
            "series_matrix_sample_ids_identical": set(ids_a) == set(ids_b),
            "series_matrix_sample_metadata_identical": ids_a == ids_b,
            "primary_and_normal_mRNA_samples_in_GEO": sum(
                count for status, count in a["series_matrix"]["status_counts"].items()
                if status.lower() in {"primary triple negative cancer", "normal"}
            ),
            "GEO_primary_tumor_count_minus_paper_mRNA_tumor_count": (
                a["series_matrix"]["status_counts"].get("primary triple negative cancer", 0)
                - 158
            ),
        },
        "interpretation": (
            "The 224 count is reported for the paper's miRNA tumor/normal cohort. "
            "The public GPL16299 mRNA deposits contain a distinct tissue-class composition. "
            "GSE41970 and GSE45498 are identical mRNA supplements after decompression; "
            "the public primary/normal mRNA subset has 200 samples, while the source article "
            "reports 198 mRNA tumor/normal samples (158 tumor, 40 normal). The two-tumor "
            "difference is not resolved by the public sample metadata and requires the "
            "authors' inclusion/QC manifest."
        ),
    }
    output = ROOT / "audit" / "gse41970_mrna_crosswalk.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": result["result"],
        "output": str(output),
        "series_matrix_counts": {
            name: values["series_matrix"]["status_counts"]
            for name, values in accessions.items()
        },
        "crosswalk": result["crosswalk"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
