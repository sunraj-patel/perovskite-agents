"""Freeze (version) the benchmark before the main comparison.

Writes FREEZE.json: SHA-256 of the question bank, reference values, scorer,
model configuration and the source data, plus the budgets. ``run.py`` refuses a
main-stage run if these hashes no longer match (so scoring rules and questions
cannot drift between episodes). Re-freezing after an intentional change must be
done explicitly and bumps ``freeze_id``.

    python -m benchmarks.agentic_paper.freeze            # write / verify
    python -m benchmarks.agentic_paper.freeze --verify
"""
import hashlib
import json
import sys
import time

from .paths import (CONFIG_DIR, FREEZE_PATH, HERE, QUESTIONS_DIR, QUESTIONS_FILE, REFERENCE_DIR, REPO,
                    SPLIT_FILE, VERSION, setup_env)

setup_env()
FILES = {
    "questions": QUESTIONS_FILE,
    "reference_values": REFERENCE_DIR / "reference_values.json",
    "known_dois": REFERENCE_DIR / "known_dois.json",
    "scorer": HERE / "scorer.py",
    "models": CONFIG_DIR / "models.json",
    "episode_runner": HERE / "episode.py",
    "providers": HERE / "providers.py",
}
if VERSION != "v2":                       # v3+: the framework itself is part of the configuration
    FILES.update({
        "providers_instrumented": HERE / "providers_instrumented.py",
        "run_policy": HERE / "run.py",
        "analysis_audit": HERE / "audit.py",
        "split": SPLIT_FILE,
        "framework_agent": REPO / "perovagents" / "agent.py",
        "framework_progress": REPO / "perovagents" / "progress.py",
        "framework_team": REPO / "perovagents" / "team.py",
    })
if VERSION not in ("v2", "v3"):           # v4+: the baseline protocol and the new references
    FILES.update({
        "baseline_protocol": HERE / "baseline_v4.py",
        "baseline_harness": REPO / "benchmarks" / "harness.py",
        "validity": HERE / "validity.py",
        "reference_additions_v4": REFERENCE_DIR / "reference_values_v4_additions.json",
    })


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def current():
    from benchmarks.tool_layer import MAX_LLM_TURNS, MAX_TOOL_CALLS, RUN_TIMEOUT_S
    ref = json.load(open(REFERENCE_DIR / "reference_values.json"))
    bank = json.load(open(QUESTIONS_FILE))["questions"]
    return {
        "hashes": {k: sha(p) for k, p in FILES.items()},
        "source_data_sha256": ref["sources"],
        "budgets": {"max_tool_calls": MAX_TOOL_CALLS, "max_model_turns": MAX_LLM_TURNS,
                    "timeout_s": RUN_TIMEOUT_S},
        "n_questions_total": len(bank),
        "main_set": [q["id"] for q in bank if q["set"] == "main"],
        "case_study_set": [q["id"] for q in bank if q["set"] != "main"],
    }


def verify():
    if not FREEZE_PATH.exists():
        return False, "not frozen"
    frozen = json.load(open(FREEZE_PATH))
    cur = current()
    bad = [k for k, v in cur["hashes"].items() if frozen["hashes"].get(k) != v]
    return (not bad), (f"changed since freeze: {bad}" if bad else "ok")


def main():
    if "--verify" in sys.argv:
        ok, msg = verify()
        print(msg)
        sys.exit(0 if ok else 1)
    cur = current()
    prev = json.load(open(FREEZE_PATH)) if FREEZE_PATH.exists() else None
    cur["freeze_id"] = (prev["freeze_id"] + 1) if prev else 1
    cur["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(cur, open(FREEZE_PATH, "w"), indent=1)
    print(f"froze benchmark (freeze_id={cur['freeze_id']}): {len(cur['main_set'])} main questions")


if __name__ == "__main__":
    main()
