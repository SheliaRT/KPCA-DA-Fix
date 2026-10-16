"""Audit the second official GSE45498 supplement's modality and sample coverage."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE45nnn/GSE45498/"
    "suppl/GSE45498_raw_data.txt.gz"
)


def inspect(payload: bytes) -> dict[str, object]:
    digest = hashlib.sha256(payload).hexdigest()
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
        text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
        reader = csv.reader(text, delimiter="\t")
        header = next(reader)
        if header[:3] != ["Code Class", "Name", "Accession"]:
            raise ValueError("Unexpected GEO raw supplement columns")
        sample_names = [name.strip() for name in header[3:]]
        sample_classes = Counter(name.split("_", 1)[0] for name in sample_names)
        feature_classes: Counter[str] = Counter()
        endogenous_accessions: list[str] = []
        endogenous_names: list[str] = []
        row_count = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError("Raw supplement row width differs from the header")
            row_count += 1
            feature_classes[row[0]] += 1
            if row[0].startswith("Endogenous"):
                endogenous_names.append(row[1])
                endogenous_accessions.append(row[2])
        if not endogenous_accessions or not all(
            accession.startswith("nmiR") for accession in endogenous_accessions
        ):
            raise ValueError("Raw supplement endogenous features are not all nmiR IDs")
    return {
        "source_url": RAW_URL,
        "sha256": digest,
        "compressed_bytes": len(payload),
        "format": "tab-separated NanoString raw data",
        "sample_count": len(sample_names),
        "sample_class_counts_from_column_names": dict(sorted(sample_classes.items())),
        "row_count": row_count,
        "feature_class_counts": dict(sorted(feature_classes.items())),
        "endogenous_feature_count": len(endogenous_accessions),
        "endogenous_accession_prefixes": ["nmiR"],
        "example_endogenous_names": endogenous_names[:8],
        "assessment": (
            "This supplement contains microRNA nmiR features, not GPL16299 mRNA "
            "expression values. Do not transfer its 224-sample miRNA cohort count "
            "to the mRNA classification cohort or use these values to infer missing "
            "mRNA measurements."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "audit/gse45498_raw_data_audit.json")
    args = parser.parse_args()
    with urllib.request.urlopen(RAW_URL, timeout=30) as response:
        result = inspect(response.read())
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": "passed", "output": str(output),
                      "sample_count": result["sample_count"],
                      "feature_class_counts": result["feature_class_counts"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
