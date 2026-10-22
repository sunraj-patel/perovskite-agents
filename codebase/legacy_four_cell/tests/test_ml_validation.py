"""ML validation artifact tests: results/ml/validation.json is present,
sane, group-aware, and actually gates real predictions.

The numbers themselves come from `ml/compute_validation.py` (GroupKFold
by DOI, repo feature pipeline); these tests pin the artifact contract,
not the science.
"""
import json
import os
import unittest

from perovagents import config
from perovagents.cells.ml import attach_validation

PATH = os.path.join(config.ensure_results("ml"), "validation.json")


class TestValidationArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(PATH, encoding="utf-8") as f:
            cls.rec = json.load(f)

    def test_all_architectures_recorded(self):
        self.assertEqual(sorted(self.rec), ["NIP", "Overall", "PIN"])

    def test_scores_sane_and_group_aware(self):
        for arch, r in self.rec.items():
            self.assertEqual(r["metric"], "rmse")
            self.assertGreater(r["score"], 0)
            self.assertLess(r["score"], 10)
            self.assertGreaterEqual(r["r2"], 0.0)
            self.assertLessEqual(r["r2"], 1.0)
            self.assertTrue(r["split"].startswith("group-doi"),
                            f"{arch} split is not group-aware")
            self.assertGreater(r["n_groups"], 1000)
            self.assertGreater(r["n"], 10000)

    def test_attach_wiring(self):
        res = {"architecture": "Overall", "provenance": {"model": "m.cbm"}}
        out = attach_validation(res, PATH)
        val = out["provenance"]["validation"]
        self.assertEqual(val["split"], "group-doi-5fold")
        # missing file leaves the result untouched -> gate fails closed
        res2 = {"architecture": "Overall", "provenance": {"model": "m.cbm"}}
        self.assertNotIn("validation",
                         attach_validation(res2, "/tmp/nope.json")
                         ["provenance"])

    def test_random_split_rejected_by_gate(self):
        from perovagents.verify.ml_checks import check_model_attached
        bad = {"provenance": {"model": "m.cbm",
                              "validation": {"metric": "rmse", "score": 1.0,
                                             "split": "random"}}}
        self.assertFalse(check_model_attached(bad).passed)


if __name__ == "__main__":
    unittest.main()
