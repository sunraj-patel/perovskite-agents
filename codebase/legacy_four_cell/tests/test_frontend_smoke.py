"""Frontend smoke tests.

Run with:  python tests/test_frontend_smoke.py

Covers the live-trace rendering pipeline without a browser:
  * AppTest loads the Streamlit page with no exceptions
  * submitting a query produces user + assistant messages and an OK step badge
  * a second query renders the persisted history trace (nested expander-safe)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFrontendSmoke(unittest.TestCase):
    def test_page_loads_clean(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("frontend/app.py", default_timeout=120)
        at.run()
        self.assertEqual(len(at.exception), 0)
        titles = [t.value for t in at.title]
        self.assertIn("PerovAgents", titles)

    def test_query_runs(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("frontend/app.py", default_timeout=240)
        at.run()
        self.assertGreater(len(at.chat_input), 0)
        at.chat_input[0].set_value("what is the average PCE for SnO2 ETL devices?").run()
        self.assertEqual(len(at.exception), 0)
        self.assertGreaterEqual(len(at.chat_message), 2)
        has_ok = any("OK ·" in m.value for m in at.markdown)
        self.assertTrue(has_ok, "expected at least one completed step card")

    def test_history_persists(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("frontend/app.py", default_timeout=240)
        at.run()
        at.chat_input[0].set_value("how many records use PTAA?").run()
        self.assertEqual(len(at.exception), 0)
        at.chat_input[0].set_value("top 3 spiro devices by PCE").run()
        self.assertEqual(len(at.exception), 0)
        # user(2) + assistant(2) messages persisted across runs
        self.assertGreaterEqual(len(at.chat_message), 4)

    @unittest.skipUnless(os.path.exists(os.path.join("results", "ml", "model_PIN.cbm")),
                         "ML model artifacts not present")
    def test_prediction_metrics_render(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("frontend/app.py", default_timeout=240)
        at.run()
        at.chat_input[0].set_value(
            "predict the PCE of a p-i-n cell with SnO2 ETL, PTAA HTL and CsMAFA").run()
        self.assertEqual(len(at.exception), 0)
        labels = {m.label for m in at.metric}
        self.assertIn("Predicted PCE (%)", labels)
        self.assertIn("95% interval low (%)", labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)