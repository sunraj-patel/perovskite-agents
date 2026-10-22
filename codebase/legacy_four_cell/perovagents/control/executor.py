"""Executor v2: RunState + parallel dispatch + dependency passing (S7).

- Tasks with no unmet dependencies run concurrently (one thread per cell).
- A task with `depends_on` waits; it receives its dependencies' validated
  payloads injected as `task.params["dep_results"]`.
- Only `ValidatedResult` objects are collected. A cell that exhausts its
  budget returns `FailedResult`: surfaced in the summary, dependents marked
  `skipped`, never silently dropped (spec C8/D5).
- Per-task wall-clock timeout (a timed-out cell keeps running in its thread;
  the run moves on — documented limitation, DECISIONS.md D13).
- One structured JSONL trace per run (plan, dispatch, verdict, critique,
  skip, timeout, done) for replay.
- `utility` tasks run helper tools directly with a trivially-passing
  `utility_ok` check (no gate).
"""
import concurrent.futures as futures
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from ..cells.base import DomainCell
from ..cells.schemas import (FailedResult, Plan, RunState, Task,
                             ValidatedResult)
from ..verify.base import check, report

UTILITY_TOOLS = ("calculator", "python_exec", "symbolic")


def _utc_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


class ExecutorV2:
    """Parallel dependency-aware executor over DomainCell instances."""

    def __init__(self, cells: Dict[str, DomainCell],
                 utility_runner=None,
                 timeout_s: float = 300.0,
                 trace_dir: Optional[str] = None,
                 seed: int = 42):
        self.cells = dict(cells)
        self.utility_runner = utility_runner
        self.timeout_s = timeout_s
        self.trace_dir = trace_dir
        self.seed = seed
        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []

    # ---- trace ------------------------------------------------------
    def _log(self, event: str, **fields) -> None:
        with self._lock:
            self._events.append({"t": time.time(), "event": event, **fields})

    def _flush_trace(self, run_id: str) -> Optional[str]:
        if not self.trace_dir:
            return None
        os.makedirs(self.trace_dir, exist_ok=True)
        path = os.path.join(self.trace_dir, f"run_{run_id}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for e in self._events:
                f.write(json.dumps(e, default=str) + "\n")
        return path

    # ---- single task ------------------------------------------------
    def _run_task(self, state: RunState, task: Task):
        """Execute one ready task (called in a worker thread)."""
        if task.cell == "utility":
            return self._run_utility(task)
        cell = self.cells.get(task.cell)
        if cell is None:
            err = f"no cell registered for {task.cell!r}"
            return FailedResult(task_id=task.id, cell=task.cell,
                                attempts=[], last_critique=None), err
        params = dict(task.params)
        params["dep_results"] = {
            d: state.results[d].payload
            for d in task.depends_on
            if isinstance(state.results.get(d), ValidatedResult)}
        child = Task(id=task.id, cell=task.cell, request=task.request,
                     params=params, depends_on=list(task.depends_on))
        return cell.run(child), ""

    def _run_utility(self, task: Task):
        """Helper-tool call without a gate (DECISIONS.md D2)."""
        if self.utility_runner is None:
            return FailedResult(task_id=task.id, cell="utility",
                                attempts=[], last_critique=None), \
                "no utility runner configured"
        try:
            payload = self.utility_runner(task)
        except Exception as exc:
            return FailedResult(task_id=task.id, cell="utility",
                                attempts=[], last_critique=None), str(exc)
        return ValidatedResult(
            task_id=task.id, cell="utility", payload=dict(payload),
            report=report([check("utility_ok", True, "helper ran")]),
            provenance={"cell": "utility", "task_id": task.id}), ""

    # ---- finishing --------------------------------------------------
    def _finish_task(self, state: RunState, plan: Plan, by_id: Dict,
                     tid: str, task: Task, result, err: str) -> None:
        """Record one task outcome; fail dependents on FailedResult."""
        state.results[tid] = result
        if isinstance(result, ValidatedResult):
            state.mark(tid, "validated")
            self._log("validated", task=tid, cell=task.cell,
                      attempts=result.provenance.get("attempts"),
                      report=result.report.to_dict())
        else:
            state.mark(tid, "failed")
            msg = err or (result.last_critique.explanation
                          if result.last_critique else "failed")
            state.errors[tid] = msg
            self._log("failed", task=tid, cell=task.cell,
                      reason=msg[:300])
            for dep in plan.tasks:  # dependents can never run
                if tid in dep.depends_on and \
                        state.task_status.get(dep.id) == "pending":
                    state.mark(dep.id, "skipped")
                    state.errors[dep.id] = \
                        f"skipped: dependency {tid} failed"
                    self._log("skipped", task=dep.id, blocked_by=tid)

    # ---- main loop --------------------------------------------------
    def run(self, plan: Plan) -> Dict[str, Any]:
        """Dispatch the plan to completion. Returns run summary + state."""
        run_id = f"{_utc_ts()}_{os.getpid()}"
        state = RunState(plan=plan)
        for t in plan.tasks:
            state.task_status.setdefault(t.id, "pending")
        self._log("plan", run_id=run_id, seed=self.seed,
                  tasks=[t.to_dict() for t in plan.tasks])
        by_id = {t.id: t for t in plan.tasks}
        # Unknown dependency ids fail fast (planner bug, not a cell fault).
        for t in plan.tasks:
            for d in t.depends_on:
                if d not in by_id:
                    state.mark(t.id, "failed")
                    state.errors[t.id] = f"unknown dependency {d!r}"
                    self._log("failed", task=t.id, reason="unknown dep")

        with futures.ThreadPoolExecutor(
                max_workers=max(1, len(self.cells))) as pool:
            pending_futures: Dict[Any, str] = {}
            started_at: Dict[Any, float] = {}
            poll = min(self.timeout_s, 1.0)
            while True:
                ready = [by_id[tid] for tid in state.ready()
                         if tid not in pending_futures.values()]
                for task in ready:
                    state.mark(task.id, "running")
                    self._log("dispatch", task=task.id, cell=task.cell)
                    fut = pool.submit(self._run_task, state, task)
                    pending_futures[fut] = task.id
                    started_at[fut] = time.time()
                if not pending_futures:
                    break
                done, _ = futures.wait(
                    pending_futures, timeout=poll,
                    return_when=futures.FIRST_COMPLETED)
                now = time.time()
                for fut in list(done):
                    tid = pending_futures.pop(fut)
                    started_at.pop(fut, None)
                    task = by_id[tid]
                    try:
                        result, err = fut.result(timeout=0)
                    except Exception as exc:  # worker bug: failed, not lost
                        from ..cells.schemas import Critique
                        result = FailedResult(
                            task_id=tid, cell=task.cell, attempts=[],
                            last_critique=Critique(
                                failed_checks=["worker_error"],
                                explanation=str(exc)[:300],
                                suggested_change=""))
                        err = str(exc)[:200]
                    self._finish_task(state, plan, by_id, tid, task,
                                      result, err)
                # per-task wall-clock timeout: abandon slow workers
                for fut, tid in list(pending_futures.items()):
                    if now - started_at.get(fut, now) > self.timeout_s:
                        from ..cells.schemas import Critique
                        pending_futures.pop(fut)
                        started_at.pop(fut, None)
                        task = by_id[tid]
                        result = FailedResult(
                            task_id=tid, cell=task.cell, attempts=[],
                            last_critique=Critique(
                                failed_checks=["timeout"],
                                explanation=(f"task exceeded {self.timeout_s}s;"
                                             " worker thread abandoned"),
                                suggested_change="raise timeout_s"))
                        self._log("timeout", task=tid)
                        self._finish_task(state, plan, by_id, tid, task,
                                          result, "timeout")
        trace = self._flush_trace(run_id)
        summary = {
            "run_id": run_id,
            "status": dict(state.task_status),
            "validated": sorted(
                tid for tid, r in state.results.items()
                if isinstance(r, ValidatedResult)),
            "failed": {tid: state.errors.get(tid, "")
                       for tid, r in state.results.items()
                       if isinstance(r, FailedResult)},
            "skipped": sorted(
                tid for tid, s in state.task_status.items()
                if s == "skipped"),
            "trace": trace,
        }
        self._log("done", run_id=run_id, summary={
            k: v for k, v in summary.items() if k != "trace"})
        # re-flush so the file also contains the done event
        trace = self._flush_trace(run_id)
        summary["trace"] = trace
        return {"summary": summary, "state": state}
