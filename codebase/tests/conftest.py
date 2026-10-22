import json, os, sqlite3, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ScriptedLLM:
    """Replays canned replies (list, or f(prompt) -> str) and keeps every prompt it saw."""
    def __init__(self, replies):
        self.replies, self.prompts = replies, []

    def chat(self, prompt, temperature=None):
        self.prompts.append(prompt)
        if callable(self.replies):
            return self.replies(prompt)
        return self.replies[len(self.prompts) - 1] if len(self.prompts) <= len(self.replies) else "FINAL ANSWER: (script ended)"


@pytest.fixture
def fake_db(tmp_path, monkeypatch):
    from perovagents import config
    path = str(tmp_path / "pdb.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE devices (Ref_ID REAL, Ref_DOI_number TEXT, JV_default_PCE REAL, ETL_stack_sequence TEXT, Cell_architecture TEXT, HTL_stack_sequence TEXT)")
    con.executemany("INSERT INTO devices VALUES (?,?,?,?,?,?)",
                    [(1, "10.1/a", 14.0, "SnO2", "nip", "Spiro-MeOTAD"), (2, "10.1/a", 16.0, "SnO2", "nip", "Spiro-MeOTAD"), (3, "10.1/b", 18.0, "TiO2-c", "nip", "PTAA"),
                     (4, "10.1/c", 20.0, "SnO2", "nip", "PTAA")])
    con.commit(); con.close()
    monkeypatch.setattr(config, "PDB_SQLITE", path)
    return path
