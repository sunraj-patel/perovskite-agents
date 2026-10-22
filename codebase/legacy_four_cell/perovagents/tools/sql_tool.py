"""Read-only SQL tool over a SQLite mirror of the Perovskite Database (S4).

The PDB lives as CSV (`config.PDB_CSV`); this module mirrors it into SQLite
once (`build_sqlite`, column types inferred: all-numeric → REAL else TEXT)
and serves queries through `SQLTool`:

- read-only connection (`mode=ro`; missing DB is an error, never created)
- SELECT/WITH only (rejected otherwise — the S2 gate re-checks the text)
- row cap enforced by fetching at most `row_limit + 1` rows (`truncated`)
- wall-clock cap via a SQLite progress handler (`timed out` error)

The tool is intentionally NOT registered in `layers/` yet: the old planner's
prompt catalog must not change until cell (a) passes acceptance (S9).
"""
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from .. import config
from ..core.tool import Tool
from ..verify.sql_checks import check_select_only

TABLE = "devices"


def infer_types(csv_path: str) -> Dict[str, str]:
    """Per-column SQLite type: REAL when every non-null value parses as a
    number, else TEXT."""
    import pandas as pd
    df = pd.read_csv(csv_path, low_memory=False)
    types = {}
    for col in df.columns:
        series = df[col]
        non_null = series[series.notna()]
        if len(non_null) == 0:
            types[col] = "TEXT"
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        types[col] = ("REAL" if coerced.notna().sum() == len(non_null)
                      else "TEXT")
    return types


def build_sqlite(csv_path: Optional[str] = None,
                 db_path: Optional[str] = None,
                 max_rows: Optional[int] = None) -> str:
    """Mirror the PDB CSV into SQLite. Returns the db path. Idempotent
    inputs: an existing db file is rebuilt (delete first to force)."""
    import pandas as pd
    csv_path = csv_path or config.PDB_CSV
    db_path = db_path or default_db_path()
    df = pd.read_csv(csv_path, low_memory=False, nrows=max_rows)
    if max_rows is None:
        types = infer_types(csv_path)  # full mirror: inferred REAL/TEXT
    else:
        types = {c: "TEXT" for c in df.columns}  # sampled: fast TEXT path
    cols = list(df.columns)
    decl = ", ".join(f'"{c}" {types[c]}' for c in cols)
    if os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(f'CREATE TABLE "{TABLE}" ({decl})')
        rows = df.where(pd.notnull(df), None).values.tolist()
        ph = ", ".join(["?"] * len(cols))
        names = ", ".join(f'"{c}"' for c in cols)
        for i in range(0, len(rows), 2000):
            conn.executemany(
                f'INSERT INTO "{TABLE}" ({names}) VALUES ({ph})',
                rows[i:i + 2000])
        conn.commit()
    finally:
        conn.close()
    return db_path


def default_db_path() -> str:
    """Canonical mirror location under the results dir."""
    return os.path.join(config.ensure_results("sqlite"), "perovskite.db")


class SQLTool(Tool):
    """Read-only SQL over the PDB mirror. Deliberately unregistered (S4)."""
    name = "sql"
    description = (
        "Run read-only SELECT queries against the Perovskite Database "
        "(experimental device records). Returns rows + columns with row "
        "and time limits enforced."
    )
    layer = "materials"
    version = "0.1.0"
    capabilities = {"sql", "database", "read_only"}

    def __init__(self, db_path: Optional[str] = None,
                 row_limit: int = 1000, timeout_s: float = 10.0,
                 csv_path: Optional[str] = None):
        self.db_path = db_path or default_db_path()
        self.row_limit = row_limit
        self.timeout_s = timeout_s
        if not os.path.exists(self.db_path):
            build_sqlite(csv_path or config.PDB_CSV, self.db_path)

    @property
    def schema(self) -> Dict[str, List[str]]:
        """Live {table: [columns]} for the gate (S2 checks)."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            cols = [r[1] for r in
                    conn.execute(f'PRAGMA table_info("{TABLE}")')]
        finally:
            conn.close()
        return {TABLE: cols}

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sql = str(params.get("sql", "") or "").strip()
        if not sql:
            return {"error": "sql required"}
        verdict = check_select_only(sql)
        if not verdict.passed:
            return {"error": verdict.message, "sql": sql[:200]}
        limit = int(params.get("row_limit", self.row_limit))
        timeout = float(params.get("timeout_s", self.timeout_s))
        t0 = time.time()

        def _abort() -> int:
            return 1 if time.time() - t0 > timeout else 0

        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        except Exception as exc:
            return {"error": f"cannot open read-only db: {exc}"}
        try:
            conn.set_progress_handler(_abort, 5000)
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            fetched = cur.fetchmany(limit + 1)
            truncated = len(fetched) > limit
            rows = [dict(zip(cols, r)) for r in fetched[:limit]]
        except Exception as exc:
            msg = str(exc)
            if "interrupted" in msg:
                return {"error": f"query timed out after {timeout}s",
                        "sql": sql[:200]}
            return {"error": f"query failed: {msg}", "sql": sql[:200]}
        finally:
            conn.close()
        elapsed = time.time() - t0
        return {"rows": rows, "columns": cols, "n_rows": len(rows),
                "truncated": truncated, "row_limit": limit,
                "elapsed_s": round(elapsed, 3),
                "provenance": {"kind": "sql_query", "table": TABLE,
                               "sql": sql[:500], "truncated": truncated}}
