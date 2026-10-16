"""Focused regression tests for paired-run provenance compatibility."""

import unittest

from audit.compare_tvsn_runs import patient_mapping_provenance_status


def built_in_mapping():
    return {
        domain: {
            "source": "built-in sample-ID parser",
            "mapping_file": None,
            "mapping_file_sha256": None,
        }
        for domain in ("source", "target")
    }


class PatientMappingProvenanceTests(unittest.TestCase):
    def test_matching_mapping_metadata_is_accepted(self):
        mapping = built_in_mapping()
        self.assertEqual(
            patient_mapping_provenance_status(mapping, mapping.copy()),
            "mapping metadata matched",
        )

    def test_missing_legacy_metadata_accepts_only_the_builtin_parser(self):
        self.assertIn(
            "legacy artifact",
            patient_mapping_provenance_status(None, built_in_mapping()),
        )
        mapping_file = built_in_mapping()
        mapping_file["target"] = {
            "source": "mapping file",
            "mapping_file": "target-patients.csv",
            "mapping_file_sha256": "example",
        }
        with self.assertRaisesRegex(ValueError, "not the built-in sample-ID parser"):
            patient_mapping_provenance_status(None, mapping_file)

    def test_conflicting_recorded_mappings_are_rejected(self):
        baseline = built_in_mapping()
        adapted = built_in_mapping()
        adapted["target"]["source"] = "mapping file"
        with self.assertRaisesRegex(ValueError, "differ in sample-to-patient"):
            patient_mapping_provenance_status(baseline, adapted)


if __name__ == "__main__":
    unittest.main()
