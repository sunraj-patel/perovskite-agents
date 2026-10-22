"""S4 SQL-tool tests: mirror build, read-only SELECT, limits, gate wiring."""
import os
import unittest

from perovagents.tools.sql_tool import SQLTool, build_sqlite
from perovagents.verify import sql_checks as sq

TMP = "/tmp/perovagents_s4_test.db"


def _tool():
    if not os.path.exists(TMP):
        build_sqlite(max_rows=3000, db_path=TMP)
    return SQLTool(db_path=TMP, row_limit=500, timeout_s=10.0)


class TestSQLTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _tool()

    def test_schema_served(self):
        schema = self.tool.schema
        self.assertIn("devices", schema)
        self.assertIn("JV_default_PCE", schema["devices"])
        self.assertGreater(len(schema["devices"]), 400)

    def test_select_where_limit(self):
        out = self.tool.call({"sql": (
            "SELECT JV_default_PCE, Cell_architecture FROM devices "
            "WHERE CAST(JV_default_PCE AS REAL) > 20 LIMIT 5")})
        self.assertNotIn("error", out)
        self.assertEqual(len(out["rows"]), 5)
        self.assertFalse(out["truncated"])
        self.assertTrue(all(float(r["JV_default_PCE"]) > 20
                            for r in out["rows"]))

    def test_aggregation(self):
        out = self.tool.call({"sql": (
            "SELECT AVG(JV_default_PCE) AS avg_pce FROM devices")})
        self.assertNotIn("error", out)
        self.assertGreater(out["rows"][0]["avg_pce"], 5.0)

    def test_unknown_column_errors(self):
        out = self.tool.call(
            {"sql": "SELECT unicorn_column FROM devices LIMIT 1"})
        self.assertIn("error", out)

    def test_insert_rejected(self):
        out = self.tool.call({"sql": "INSERT INTO devices VALUES (1)"})
        self.assertIn("error", out)
        # read-only connection is a second wall: no write API exists
        self.assertNotIn("rows", out)

    def test_row_limit_truncates(self):
        out = self.tool.call({"sql": "SELECT * FROM devices"})
        self.assertTrue(out["truncated"])
        self.assertEqual(len(out["rows"]), 500)

    def test_timeout_errors(self):
        tool = SQLTool(db_path=TMP, row_limit=500, timeout_s=0.0)
        out = tool.call({"sql": "SELECT * FROM devices"})
        self.assertIn("timed out", out.get("error", ""))

    def test_gate_passes_good_query(self):
        sql = ("SELECT JV_default_PCE AS pce, Cell_architecture FROM devices "
               "WHERE JV_default_PCE > 15 LIMIT 10")
        out = self.tool.call({"sql": sql})
        rep = sq.verify_sql(sql, out, self.tool.schema,
                            expected_cols=["pce"])
        self.assertTrue(rep.passed, rep.to_dict())

    def test_gate_flags_unknown_column(self):
        # spec 9 example, end to end through the tool
        sql = "SELECT JV_default_PCE, unicorn_column FROM devices LIMIT 2"
        out = self.tool.call({"sql": sql})
        rep = sq.verify_sql(sql, out, self.tool.schema)
        self.assertFalse(rep.passed)
        self.assertIn("known_columns", rep.failed_checks())


if __name__ == "__main__":
    unittest.main()
