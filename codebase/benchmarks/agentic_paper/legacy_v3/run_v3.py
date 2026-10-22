"""Resumable benchmark runner.

    python -m benchmarks.agentic_paper.run --status
    python -m benchmarks.agentic_paper.run --stage smoke
    python -m benchmarks.agentic_paper.run --stage pass --reps 1          # one full pass
    python -m benchmarks.agentic_paper.run --stage pass --reps 3          # fill up to 3 reps
    python -m benchmarks.agentic_paper.run --stage pass --reps 3 --backbones gemini --workers 4

Every episode is one (question, backbone, condition, repetition) cell with a
unique ``episode_id``. Completed cells are never repeated.

Retry policy (validity.py): only an ``infra_invalid`` attempt (provider failure, or
all replies empty AND explained by provider-side evidence) is replaced, up to
``--max-attempts`` attempts in total. ``unclassified_empty`` attempts are kept and are
re-queued only with ``--retry-unclassified``. The original attempt is never overwritten:
every attempt is stored as ``episode.attempt<n>.json`` and its manifest row carries
``attempt`` and ``replaces`` (the previous attempt's file), so chains are auditable. If a backbone keeps
failing the runner stops that backbone (no fallback to another model) and
prints the resume command.

Order is repetition-major, then question, with the two conditions of a
question adjacent so that partial runs keep paired comparisons.
"""
import argparse
import json
import shutil
import queue
import signal
import threading
import time
from pathlib import Path

from .paths import CONFIG_DIR, RUNS_DIR, setup_env

setup_env()

CONDITIONS = ("direct", "perovagents")
INFRA_STREAK_STOP = 4                    # consecutive infra failures before a backbone stops


def load_bank():
    from .scorer import load_bank
    return load_bank()[1]


def manifest_path(stage):
    return RUNS_DIR / ("manifest.jsonl" if stage == "main" else "smoke_manifest.jsonl")


def read_manifest(stage):
    p = manifest_path(stage)
    rows = []
    if p.exists():
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def episode_id(qid, backbone, condition, rep):
    return f"{qid}__{backbone}__{condition}__r{rep}"


def cell_state(rows):
    """episode_id -> {'done': bool, 'attempts': n, 'last_validity': str, 'last_path': str}"""
    st = {}
    for r in rows:
        s = st.setdefault(r["episode_id"], {"done": False, "attempts": 0})
        s["attempts"] += 1
        s["last_validity"] = r.get("validity")
        s["last_path"] = r.get("attempt_path")
        s["done"] = r["status"] == "completed"          # the latest attempt decides
    return st


def needs_run(state, max_attempts, retry_unclassified):
    """Explicit retry policy for one cell (state = cell_state entry or None)."""
    from .validity import should_retry
    if not state:
        return True
    if state["done"] and not (retry_unclassified and state.get("last_validity") == "unclassified_empty"):
        return False
    return should_retry(state.get("last_validity") or "infra_invalid", state["attempts"], max_attempts,
                        retry_unclassified)


def build_plan(qids, backbones, conditions, reps):
    plan = []
    for rep in reps:
        for qid in qids:
            for bb in backbones:
                for cond in conditions:
                    plan.append((qid, bb, cond, rep))
    return plan


def run_cell(cell, bank, backbone_cfg, stage, lock):
    from .episode import run_episode
    qid, bb, cond, rep = cell
    eid = episode_id(qid, bb, cond, rep)
    root = RUNS_DIR / ("episodes" if stage == "main" else "smoke_episodes") / eid
    with lock:
        prior = cell_state(read_manifest(stage)).get(eid)
    attempt = (prior["attempts"] if prior else 0) + 1
    replaces = prior["last_path"] if prior else None
    t0 = time.time()
    try:
        ep = run_episode(bank[qid], backbone_cfg, cond, rep, root)
        status = ("infra_failed" if ep["infra_failure"] else
                  "harness_error" if ep["termination"] == "harness_error" else "completed")
        term, secs = ep["termination"], ep["seconds"]
    except Exception as exc:                            # runner-level crash; recorded, not hidden
        status, term, secs = "harness_error", "harness_error", round(time.time() - t0, 1)
        root.mkdir(parents=True, exist_ok=True)
        (root / "runner_error.txt").write_text(f"{type(exc).__name__}: {exc}")
    validity = None
    attempt_rel = None
    if (root / "episode.json").exists():                 # immutable per-attempt copy (never overwritten)
        att = root / f"episode.attempt{attempt}.json"
        shutil.copy(root / "episode.json", att)
        attempt_rel = str(att.relative_to(RUNS_DIR.parent))
        try:
            validity = json.load(open(att)).get("validity")
        except Exception:
            pass
    row = {"episode_id": eid, "qid": qid, "category": bank[qid]["category"], "backbone": bb,
           "provider": backbone_cfg["route"], "model": backbone_cfg["model"],
           "condition": cond, "rep": rep, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "status": status, "termination": term, "seconds": secs, "validity": validity,
           "attempt": attempt, "replaces": replaces, "config_version": "v3",
           "attempt_path": attempt_rel,
           "episode_path": attempt_rel or str((root / "episode.json").relative_to(RUNS_DIR.parent))}
    with lock:
        with open(manifest_path(stage), "a") as f:
            f.write(json.dumps(row) + "\n")
    return row


def worker(bb, q, bank, cfg, stage, lock, stop, streak, log):
    while not stop.is_set():
        try:
            cell = q.get_nowait()
        except queue.Empty:
            return
        if streak[bb] >= INFRA_STREAK_STOP:
            log(f"[{bb}] stopped: {streak[bb]} consecutive infrastructure failures; "
                f"remaining cells left for a later --resume run")
            return
        row = run_cell(cell, bank, cfg, stage, lock)
        streak[bb] = streak[bb] + 1 if row["status"] == "infra_failed" else 0
        log(f"{row['timestamp']} {row['episode_id']} -> {row['status']}/{row['termination']} "
            f"{row['seconds']}s")


def status(stage="main"):
    from .scorer import load_bank as lb
    bank = lb()[1]
    main_ids = [q for q, v in bank.items() if v["set"] == "main"]
    from .providers import load_models
    bbs = [b["id"] for b in load_models()]
    st = cell_state(read_manifest(stage))
    out = {}
    for bb in bbs:
        for cond in CONDITIONS:
            done = {r: sum(1 for q in main_ids if st.get(episode_id(q, bb, cond, r), {}).get("done"))
                    for r in (0, 1, 2)}
            out[f"{bb}/{cond}"] = done
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="benchmarks.agentic_paper.run")
    ap.add_argument("--stage", choices=["smoke", "pass"], default="pass")
    ap.add_argument("--reps", type=int, default=1, help="repetitions per cell (0..reps-1)")
    ap.add_argument("--backbones", default="all")
    ap.add_argument("--conditions", default="direct,perovagents")
    ap.add_argument("--questions", default="main", help="'main' or comma-separated IDs")
    ap.add_argument("--workers", type=int, default=3, help="concurrent episodes per backbone")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="total attempts per cell (the original counts as attempt 1)")
    ap.add_argument("--retry-unclassified", action="store_true",
                    help="also replace (once) attempts whose empty replies could not be classified")
    ap.add_argument("--limit", type=int, default=0, help="stop after N cells (debug)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-unfrozen", action="store_true")
    a = ap.parse_args(argv)
    if a.status:
        print(json.dumps(status(), indent=1))
        return
    stage = "main" if a.stage == "pass" else "smoke"
    from . import freeze
    ok, msg = freeze.verify()
    if stage == "main" and not ok and not a.allow_unfrozen:
        raise SystemExit(f"benchmark not frozen / changed ({msg}); run "
                         f"`python -m benchmarks.agentic_paper.freeze` after reviewing the change")
    from .providers import load_models
    models = {b["id"]: b for b in load_models()}
    bbs = list(models) if a.backbones == "all" else a.backbones.split(",")
    bank = load_bank()
    if stage == "smoke":
        qids = (json.load(open(CONFIG_DIR / "split_v3.json"))["smoke_v3"] if a.questions == "main"
                else a.questions.split(","))
        reps = [0]
    else:
        qids = ([q for q, v in bank.items() if v["set"] == "main"] if a.questions == "main"
                else a.questions.split(","))
        reps = list(range(a.reps))
    conds = a.conditions.split(",")
    plan = build_plan(qids, bbs, conds, reps)
    st = cell_state(read_manifest(stage))
    todo = [c for c in plan if needs_run(st.get(episode_id(*c)), a.max_attempts, a.retry_unclassified)]
    print(f"plan: {len(plan)} cells, {len(plan) - len(todo)} done/exhausted, {len(todo)} to run "
          f"({', '.join(bbs)} x {', '.join(conds)} x reps {reps[0]}..{reps[-1]})", flush=True)
    if a.limit:
        todo = todo[:a.limit]
    if a.dry_run or not todo:
        return
    queues = {bb: queue.Queue() for bb in bbs}
    for c in todo:
        queues[c[1]].put(c)
    stop, lock = threading.Event(), threading.Lock()
    streak = {bb: 0 for bb in bbs}

    def log(msg):
        with lock:
            print(msg, flush=True)

    signal.signal(signal.SIGINT, lambda *_: (stop.set(), print("stopping after current episodes...")))
    threads = []
    for bb in bbs:
        for _ in range(a.workers):
            t = threading.Thread(target=worker, daemon=True,
                                 args=(bb, queues[bb], bank, models[bb], stage, lock, stop, streak, log))
            t.start()
            threads.append(t)
    for t in threads:
        while t.is_alive():
            t.join(1.0)
    left = sum(q.qsize() for q in queues.values())
    print(f"finished; {left} cells left. Resume with the same command.", flush=True)


if __name__ == "__main__":
    main()
