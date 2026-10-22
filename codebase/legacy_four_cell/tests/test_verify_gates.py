"""S2 gate tests: each named check has at least one pass and one fail.

Includes the spec section 9 examples: unknown SQL column, PCE-as-fraction,
number-not-in-source, OOD input, uncontrollable what-if variable, and
SCAPS PCE above the Shockley-Queisser limit.
"""
import unittest

from perovagents.verify import sql_checks as sq
from perovagents.verify import evidence_checks as ev
from perovagents.verify import ml_checks as ml
from perovagents.verify import physics_checks as ph

SCHEMA = {"devices": ["JV_default_PCE", "JV_default_Voc", "Cell_architecture",
                       "ETL_stack_sequence", "Ref_DOI_number"],
          "papers": ["doi", "title"]}


class TestSQLChecks(unittest.TestCase):
    q = ("SELECT JV_default_PCE, Cell_architecture FROM devices "
         "WHERE ETL_stack_sequence LIKE '%SnO2%' LIMIT 20")

    def test_select_only_pass(self):
        self.assertTrue(sq.check_select_only(self.q).passed)

    def test_select_only_rejects_insert(self):
        r = sq.check_select_only("INSERT INTO devices VALUES (1)")
        self.assertFalse(r.passed)

    def test_select_only_rejects_stacked(self):
        r = sq.check_select_only(self.q + "; DROP TABLE devices")
        self.assertFalse(r.passed)

    def test_known_tables_pass(self):
        self.assertTrue(sq.check_known_tables(self.q, SCHEMA).passed)

    def test_known_tables_unknown(self):
        r = sq.check_known_tables("SELECT x FROM proteins", SCHEMA)
        self.assertFalse(r.passed)
        self.assertIn("proteins", r.offending)

    def test_known_columns_pass(self):
        self.assertTrue(sq.check_known_columns(self.q, SCHEMA).passed)

    def test_known_columns_unknown(self):
        # spec 9: SQL query with an unknown column must fail
        r = sq.check_known_columns(
            "SELECT JV_default_PCE, unicorn_column FROM devices", SCHEMA)
        self.assertFalse(r.passed)
        self.assertIn("unicorn_column", r.offending)

    def test_units_sane_pass(self):
        self.assertTrue(
            sq.check_units_sane({"rows": [{"pce": 21.5}]}).passed)

    def test_units_sane_fraction_fails(self):
        # spec 9: PCE reported as a fraction instead of % must fail
        r = sq.check_units_sane({"rows": [{"pce": 0.215}]})
        self.assertFalse(r.passed)

    def test_shape_pass(self):
        res = {"rows": [{"JV_default_PCE": 20.0, "doi": "10/x"}]}
        self.assertTrue(
            sq.check_shape(res, expected_cols=["JV_default_PCE"]).passed)

    def test_shape_empty_fails(self):
        self.assertFalse(sq.check_shape({"rows": []}).passed)

    def test_shape_all_null_fails(self):
        res = {"rows": [{"a": None}, {"a": None}]}
        self.assertFalse(sq.check_shape(res).passed)

    def test_verify_sql_all_pass(self):
        res = {"rows": [{"JV_default_PCE": 20.0}]}
        rep = sq.verify_sql(self.q, res, SCHEMA,
                            expected_cols=["JV_default_PCE"])
        self.assertTrue(rep.passed)


class TestEvidenceChecks(unittest.TestCase):
    def test_citation_resolvable_pass(self):
        c = {"paper_id": "10.1002/adfm.201604733", "locator": "p3",
             "passage": "SnO2 ETLs average above 20% PCE in our devices."}
        self.assertTrue(ev.check_citation_resolvable(c).passed)

    def test_citation_missing_locator_fails(self):
        self.assertFalse(
            ev.check_citation_resolvable({"paper_id": "10/x"}).passed)

    def test_number_in_source_pass(self):
        self.assertTrue(
            ev.check_number_in_source(20.5, "devices reached 20.5% PCE").passed)

    def test_number_not_in_source_fails(self):
        # spec 9: literature number not present in the cited text
        self.assertFalse(
            ev.check_number_in_source(25.1, "devices reached 20.5% PCE").passed)

    def test_support_pass_and_fail(self):
        good = ("SnO2 electron transport layers enable efficient charge "
                "extraction in perovskite solar cells")
        self.assertTrue(ev.check_support(
            "SnO2 ETLs enable efficient charge extraction", good).passed)
        self.assertFalse(ev.check_support(
            "quantum dot lasers lase at room temperature", good).passed)

    def test_context_match_and_mismatch(self):
        q = {"architecture": "n-i-p", "composition": "MAPbI3"}
        self.assertTrue(ev.check_context(
            q, {"architecture": "n-i-p device", "composition": "MAPbI3"}).passed)
        bad = ev.check_context(q, {"architecture": "p-i-n"})
        self.assertFalse(bad.passed)

    def test_verify_claim(self):
        claim = {"text": "SnO2 ETLs average above 20%",
                 "number": 20,
                 "citations": [{"paper_id": "10/x", "locator": "p1",
                                "passage": "SnO2 ETLs average above 20% PCE.",
                                "context": {}}]}
        self.assertTrue(ev.verify_claim(claim, {}).passed)
        claim["citations"][0]["passage"] = "unrelated text here"
        self.assertFalse(ev.verify_claim(claim, {}).passed)


class TestMLChecks(unittest.TestCase):
    good = {"provenance": {"model": "model_Overall.cbm",
                           "validation": {"metric": "rmse", "score": 1.8,
                                          "split": "group-doi"}},
            "hood": {"applied": True, "ood": False}}

    def test_model_attached_pass(self):
        self.assertTrue(ml.check_model_attached(self.good).passed)

    def test_model_missing_perf_fails(self):
        r = ml.check_model_attached({"provenance": {"model": "m.cbm"}})
        self.assertFalse(r.passed)

    def test_model_random_split_fails(self):
        bad = {"provenance": {"model": "m.cbm",
                              "validation": {"metric": "rmse", "score": 1.0,
                                             "split": "random"}}}
        self.assertFalse(ml.check_model_attached(bad).passed)

    def test_in_domain_pass_and_ood_fail(self):
        self.assertTrue(ml.check_in_domain(self.good).passed)
        # spec 9: OOD input must fail the gate
        ood = {"hood": {"applied": True, "ood": True, "fraction_out": 0.5}}
        self.assertFalse(ml.check_in_domain(ood).passed)

    def test_controllable_pass_and_fail(self):
        self.assertTrue(
            ml.check_controllable("Perovskite_thickness_nm").passed)
        # spec 9: uncontrollable what-if variable must fail
        self.assertFalse(ml.check_controllable("JV_default_Voc").passed)

    def test_verify_ml(self):
        self.assertTrue(ml.verify_ml(self.good, "ETL_thickness_nm").passed)


class TestPhysicsChecks(unittest.TestCase):
    # consistent 1-sun row: 1.1 * 24 * 0.78 = 20.59
    row = {"parameter": "perovskite_thickness", "value": 500.0,
           "unit": "nm", "Voc": 1.1, "Jsc": 24.0, "FF": 0.78, "PCE": 20.59,
           "converged": True}

    def test_bounds_pass(self):
        self.assertTrue(ph.check_bounds(self.row, band_gap_ev=1.55).passed)

    def test_bounds_above_sq_fails(self):
        # spec 9: SCAPS PCE above the S-Q limit must fail
        bad = dict(self.row, PCE=45.0)
        r = ph.check_bounds(bad, band_gap_ev=1.55)
        self.assertFalse(r.passed)

    def test_consistency_pass(self):
        self.assertTrue(ph.check_consistency(self.row).passed)

    def test_consistency_mismatch_fails(self):
        bad = dict(self.row, PCE=10.0)
        self.assertFalse(ph.check_consistency(bad).passed)

    def test_iv_monotone(self):
        iv = [{"v": 0.0, "j": 24.0}, {"v": 0.5, "j": 23.0},
              {"v": 1.0, "j": 5.0}]
        self.assertTrue(ph.check_iv_monotone(iv).passed)
        self.assertTrue(ph.check_iv_monotone(None).passed)  # explicit skip
        bad = [{"v": 0.0, "j": 5.0}, {"v": 0.5, "j": 23.0}]
        self.assertFalse(ph.check_iv_monotone(bad).passed)

    def test_complete(self):
        self.assertTrue(ph.check_complete(self.row).passed)
        self.assertFalse(ph.check_complete({"Voc": 1.0}).passed)

    def test_converged(self):
        self.assertTrue(ph.check_converged(self.row).passed)
        self.assertFalse(
            ph.check_converged({"converged": False}).passed)
        self.assertFalse(
            ph.check_converged({"convergence_ok": True}).passed)  # assumed

    def test_verify_physics(self):
        rep = ph.verify_physics(self.row, band_gap_ev=1.55)
        self.assertTrue(rep.passed)


if __name__ == "__main__":
    unittest.main()
