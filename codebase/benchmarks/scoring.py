"""Deterministic scoring for the cross-LLM benchmark (spec sections 9-11).

No LLM judge anywhere: every metric is computed from the run record, the
tool-call trace and the trusted gold artifact. Heuristic proxies (e.g. the
M quality composite) are labeled as such in the output.

Failure taxonomy (spec 10):
    technical_tool_failure | wrong_tool_selection | insufficient_result |
    evidence_mismatch | repeated_action | no_progress | incomplete_synthesis |
    scientifically_incorrect | system_unavailable | harness_error
"""
import math
import re

# tool name -> benchmark action (spec "expected_actions" vocabulary).
TOOL_TO_ACTION = {
    "sql": "database_query", "find_columns": "database_query",
    "column_values": "database_query",
    "search": "literature_search", "read_paper": "evidence_verify",
    "predict": "ml.predict", "explain_global": "ml.explain_global",
    "explain_local": "ml.explain_local", "what_if": "ml.what_if",
    "optimize": "ml.optimize",
    "run_point": "simulate", "sweep": "sweep", "get_sweep": "sweep",
    "reference_cell": "device_construct", "read_def": "device_construct",
    "list_parameters": "device_construct",
    "design_stack": "device_construct", "simulate_stack": "simulate",
    "simulate_defect_profile": "simulate", "write_execution_script": "device_construct",
    "_calc": "calculate",
}

FAILURES = ("technical_tool_failure", "wrong_tool_selection",
            "insufficient_result", "evidence_mismatch", "repeated_action",
            "no_progress", "incomplete_synthesis", "scientifically_incorrect",
            "system_unavailable", "harness_error", "none")

# Outcome-level non-establishment declarations. When the final answer itself
# says the required output was not established, correctness is 0 however
# many incidental keyword matches the text contains (e.g. an honest
# "the ranking could not be established" must not pass on the back of
# HTL names mentioned in the failure narrative). Narrow by design:
# recovery narratives ("the first run failed, then converged to PCE=X")
# do not match.
# Outcome-level non-establishment declarations. Only the LEAD of the answer
# (first 300 chars) counts: orchestrator failure verdicts and model hedges
# open with the non-delivery ("The ranking could not be established..."),
# while delivered answers relegate gaps to a trailing section ("What could
# NOT be established: std dev..."). Scoping to the lead keeps delivered
# content (MRS Q01: count+mean present, std-dev gap at the end) scoring
# while true non-deliveries (Q02, Q06, gemini-Q01 rerun) do not pass on
# incidental keyword matches. Recovery narratives ending in a delivered
# value do not match.
FAILURE_DECLARATIONS = (
    "could not be established",
    "could not be finalized",
    "could not be determined",
    "no qualifying",
    "nothing could be established",
    "was never established",
    "was unable to",
    "were unable to",
    "could not find",
    "could not locate",
    "no evidence was found",
    "cannot answer",
    "cannot be answered",
    "gave up",
)

LEAD_CHARS = 300


def declares_failure(answer, lead=True) -> bool:
    """True when non-establishment is declared (in the lead by default)."""
    text = str(answer or "")
    low = (text[:LEAD_CHARS] if lead else text).lower()
    return any(d in low for d in FAILURE_DECLARATIONS)


# Words marking a DOI mention as a REJECTION rather than supporting evidence
# ("candidate DOI X was examined but rejected as it did not fulfil...").
# A cited-and-rejected paper must not pass citation checks (Q06 rerun:
# the gold DOI was mentioned only to reject it, yet the naive substring
# check scored it as cited evidence).
REJECTION_WORDS = (
    "reject", "exclud", "disqualif", "insufficient", "lacks", "lacking",
    "failed", "failure", "did not", "does not", "do not", "not meet",
    "not match", "not fulfil", "not fulfill", "ineligible", "ruled out",
    "set aside", "passed over", "not qualifi",
)


def doi_cited_as_evidence(answer, doi) -> bool:
    """True when the DOI appears in a sentence free of rejection language."""
    if not doi:
        return False
    for sent in re.split(r"(?<=[.!?])\s+", str(answer or "")):
        if doi.lower() in sent.lower():
            if not any(w in sent.lower() for w in REJECTION_WORDS):
                return True
    return False


def _norm_args(args) -> str:
    if isinstance(args, dict):
        return json_dumps_sorted(args).lower()
    return str(args or "").strip().lower()


def json_dumps_sorted(args) -> str:
    import json
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except Exception:
        return str(args)


def repeated_action_rate(tool_calls) -> float:
    """Fraction of tool calls repeating an earlier (tool, normalized args)."""
    seen, repeats = set(), 0
    for tc in tool_calls or []:
        key = (str(tc.get("tool")), _norm_args(tc.get("args")))
        if key in seen:
            repeats += 1
        seen.add(key)
    n = len(tool_calls or [])
    return round(repeats / n, 3) if n else 0.0


# expected `layer.tool` entries -> benchmark action (same vocabulary as
# TOOL_TO_ACTION values, so precision/recall compare actions, not spellings).
EXPECTED_TO_ACTION = {
    "pdb.sql": "database_query",
    "literature.search": "literature_search",
    "literature.read_paper": "evidence_verify",
    "scaps.build": "device_construct",
    "scaps.run": "simulate",
    "scaps.sweep": "sweep",
}


def tool_f1(expected_tools, tool_calls):
    """Precision/recall/F1 over benchmark actions (spec metric C)."""
    want = set()
    for t in (expected_tools or []):
        t = str(t)
        if t in EXPECTED_TO_ACTION:
            want.add(EXPECTED_TO_ACTION[t])
        elif "." in t and t.split(".")[-1] in TOOL_TO_ACTION:
            want.add(TOOL_TO_ACTION[t.split(".")[-1]])
        else:
            want.add(TOOL_TO_ACTION.get(t, t))
    got = [TOOL_TO_ACTION.get(str(tc.get("tool")), str(tc.get("tool")))
           for tc in (tool_calls or [])]
    got_set = set(got)
    tp = len(want & got_set)
    precision = tp / len(got_set) if got_set else 0.0
    recall = tp / len(want) if want else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "expected": sorted(want), "used": sorted(got_set)}


def _numbers(text):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(text or ""))]


def numeric_hit(answer, gold_value, rel_tol=0.05, abs_tol=None):
    """True when any number in the answer matches gold within tolerance."""
    if gold_value is None:
        return None
    for v in _numbers(answer):
        tol = abs_tol if abs_tol is not None else abs(gold_value) * rel_tol
        if abs(v - gold_value) <= max(tol, 1e-9):
            return True
    return False


def scientific_correctness(qid, meta, gold, record):
    """Per-question numeric/categorical checks against gold. Returns
    {"score": 0..1 or None, "checks": {...}}; None = not auto-checkable.

    Checks run on the FINAL ANSWER text only: required outputs must be
    stated in the answer, not merely present somewhere in tool outputs.
    (Scoring tool-result blobs instead lets raw-row dumps pass Q02-style
    aggregation checks without ever reporting a ranking.)
    """
    blob = str(record.get("final_answer") or "")
    checks, g = {}, (gold or {})
    if qid == "Q01":
        gg = (g.get("pdb") or {}).get("Q01", {})
        # count tolerance admits NULL-PCE filter variants (gold counts
        # non-null PCE only); the mean must still match tightly.
        checks["count"] = numeric_hit(blob, gg.get("count"), rel_tol=0.05)
        checks["mean_pce"] = numeric_hit(blob, gg.get("mean_pce"), rel_tol=0.05)
    elif qid == "Q02":
        want = [r["htl"] for r in ((g.get("pdb") or {}).get("Q02", {}).get("ranking", []))[:5]]
        checks["top_htl_recall"] = (sum(1 for h in want if h and h.lower() in blob.lower())
                                    / len(want)) if want else None
        # value-anchored: a count must sit in a table cell or next to a
        # unit word ("| 736", "736 devices") - "top 10" or "PCE > 20" in
        # prose does not count as reporting a frequency.
        checks["has_counts"] = bool(re.search(
            r"\|\s*\d+|\b\d{2,}\s*(devices|papers|records|counts?)\b", blob, re.I))
    elif qid == "Q03":
        gg = (g.get("pdb") or {}).get("Q03", {})
        checks["nip_mean"] = numeric_hit(blob, (gg.get("nip") or {}).get("mean_pce"), rel_tol=0.05)
        checks["pin_mean"] = numeric_hit(blob, (gg.get("pin") or {}).get("mean_pce"), rel_tol=0.05)
        checks["both_arch"] = ("pin" in blob.lower() and "nip" in blob.lower().replace("p-i-n", ""))
    elif qid == "Q04":
        top = ((g.get("pdb") or {}).get("Q04", {}).get("top_devices", []) or [{}])[0]
        checks["top_pce"] = numeric_hit(blob, top.get("pce"), rel_tol=0.05)
        checks["has_fields"] = all(k in blob.lower() for k in ("voc", "jsc", "ff"))
    elif qid in ("Q05", "Q06", "Q07", "Q08"):
        ev = ((g.get("literature") or {}).get(qid, {}).get("gold_evidence") or {})
        doi = (ev.get("doi") or "").lower()
        checks["gold_doi_cited"] = doi_cited_as_evidence(blob, doi)
        if qid in ("Q07", "Q08"):
            checks["gold_number"] = numeric_hit(blob, ev.get("champion_pce"), rel_tol=0.03)
    elif qid == "Q09":
        gg = (g.get("ml") or {}).get("Q09", {})
        checks["predicted_pce"] = numeric_hit(blob, gg.get("predicted_pce"), rel_tol=0.10)
        checks["has_interval"] = bool(re.search(r"\[?\d+\.\d+\s*[,–-]\s*\d+\.\d+\]?", blob))
    elif qid == "Q10":
        feats = [f.get("feature", "") for f in
                 ((g.get("ml") or {}).get("Q10", {}).get("top_features", []) or
                  (g.get("ml") or {}).get("Q10", {}).get("features", []))]
        if not feats and isinstance((g.get("ml") or {}).get("Q10"), dict):
            feats = _shap_features((g.get("ml") or {})["Q10"])
        checks["feature_recall"] = (sum(1 for f in feats[:5] if f and f.lower() in blob.lower())
                                    / 5) if feats else None
    elif qid == "Q11":
        gg = (g.get("ml") or {}).get("Q11", {})
        pred = ((gg.get("predict") or {}).get("predicted_pce")
                if isinstance(gg, dict) else None)
        checks["predicted_pce"] = numeric_hit(blob, pred, rel_tol=0.10)
    elif qid == "Q12":
        checks["has_baseline"] = "baseline" in blob.lower()
        checks["has_predicted_change"] = bool(_numbers(blob))
    elif qid == "Q13":
        gg = (g.get("scaps") or {}).get("Q13", {})
        checks["pce"] = numeric_hit(blob, gg.get("PCE_percent"), rel_tol=0.05)
        checks["voc"] = numeric_hit(blob, gg.get("Voc_V"), rel_tol=0.05)
        checks["converged_claim"] = "converg" in blob.lower()
    elif qid == "Q14":
        checks["has_trend"] = any(w in blob.lower() for w in ("trend", "increas", "decreas", "peak", "falls", "rises"))
        checks["has_values"] = len(_numbers(blob)) >= 3
    elif qid == "Q15":
        checks["converged_claim"] = "converg" in blob.lower()
        checks["mentions_change"] = any(w in blob.lower() for w in
                                        ("chang", "adjust", "retr", "fix", "instead", "reduc", "increas"))
    elif qid == "Q16":
        checks["names_parameter"] = any(w in blob.lower() for w in
                                        ("thickness", "defect", "doping", "mobility", "gap", "affinity"))
    elif qid in ("Q17", "Q18", "Q19", "Q20"):
        # Value-anchored: a part counts only with reported numbers, so a
        # 0-tool "plan" answer (names the tools it would call, states no
        # values) scores nothing. Residual: open-device Qs have no numeric
        # gold; full credit needs the human grading pass over saved traces.
        low = blob.lower()
        checks["has_model_part"] = bool(
            re.search(r"(predict|shap|optim|what.if).{0,250}?\d+\.\d+", low)
            or re.search(r"\d+\.\d+.{0,250}?(predict|shap|optim)", low))
        checks["has_db_or_sim_part"] = bool(
            re.search(r"(databas\w*|records).{0,150}?\d+", low)
            or re.search(r"simulat\w*[^.]{0,150}?(voc|jsc|ff|pce)[^.]{0,60}?\d+\.\d+", low))
        if qid in ("Q17", "Q19", "Q20"):
            checks["has_literature_part"] = bool(
                re.search(r"10\.\d{3,}/[^\s)]+", blob)
                and re.search(r"quot|passage|evidence|report(s|ed)|findings?", low))
    vals = [v for v in checks.values() if isinstance(v, bool)]
    score = round(sum(vals) / len(vals), 3) if vals else None
    return {"score": score, "checks": checks}


def _shap_features(obj):
    feats = []
    if isinstance(obj, dict):
        for k in ("top_features", "features", "global_importance", "importance"):
            v = obj.get(k)
            if isinstance(v, list):
                for f in v:
                    if isinstance(f, dict) and (f.get("feature") or f.get("name")):
                        feats.append(f.get("feature") or f.get("name"))
                    elif isinstance(f, str):
                        feats.append(f)
    return feats


def completion(meta, correctness, record):
    """1 when the answer terminates AND all required outputs are evidenced."""
    if record.get("termination") != "answer":
        return 0
    if declares_failure(record.get("final_answer")):
        return 0
    req = meta.get("required_outputs") or []
    if not req or correctness.get("score") is None:
        return 1 if str(record.get("final_answer") or "").strip() else 0
    return 1 if correctness.get("score", 0) >= 0.5 else 0


def contract_satisfaction(meta, correctness):
    """1 when required outputs hold and no forbidden substitution is visible."""
    if correctness.get("score") is None:
        return None
    return 1 if correctness["score"] >= 0.5 else 0


def failure_class(record, correctness, tool_f1_result, rep_rate, meta=None):
    """Single primary failure label (spec 10 taxonomy)."""
    term = record.get("termination", "")
    if term in ("system_unavailable",):
        return "system_unavailable"
    if term in ("harness_error", "provider_error"):
        return "harness_error"
    if declares_failure(record.get("final_answer")):
        return "incomplete_synthesis"
    tools = record.get("tool_calls") or []
    if any(str(t.get("result", "")).startswith("ERROR") for t in tools) and not tools:
        return "technical_tool_failure"
    if tool_f1_result["recall"] == 0 and tools:
        return "wrong_tool_selection"
    if rep_rate >= 0.5 and len(tools) >= 2:
        return "repeated_action"
    if term in ("timeout", "turn_budget", "tool_budget"):
        return "no_progress"
    if correctness.get("score") is not None and correctness["score"] < 0.5:
        c = correctness.get("checks", {})
        if c.get("gold_doi_cited") is False:
            return "evidence_mismatch"
        # Multi-tool answers are scored on structural delivery (no numeric
        # golds exist for their open devices): partial structure is omission,
        # not an incorrect scientific claim.
        if (meta or {}).get("category") == "multi_tool":
            return "incomplete_synthesis"
        return "scientifically_incorrect"
    if not str(record.get("final_answer") or "").strip():
        return "incomplete_synthesis"
    return "none"


def recovery_record(meta, record, correctness):
    """For recovery-sensitive Qs: failure seen? recovered? extra cost.

    Recovery is evidence-gated (final score >= 0.5 after a blocked or
    failed attempt) with one coherence rule: a lead declaration of
    non-delivery defeats recovery, since the system's own verdict is that
    nothing usable was established.
    """
    tools = record.get("tool_calls") or []
    texts = " ".join(str(t.get("result", "")) for t in tools)
    saw_block = any(s in texts for s in
                    ("REPEATED_ACTION", "BLOCKED", "NO_PROGRESS", "INCOMPLETE",
                     "did not converge", "ERROR"))
    final_ok = bool((correctness.get("score") or 0) >= 0.5)
    # A lead declaration of non-delivery defeats recovery even when partial
    # evidence scores: the system's own verdict is that nothing usable was
    # established (Q06 rerun rejected the gold paper and delivered no
    # alternative). D stays evidence-based; A/F reflect the verdict.
    recovered = bool(final_ok and saw_block
                     and not declares_failure(record.get("final_answer"), lead=True))
    return {"recovery_required": bool(meta.get("recovery_required")),
            "initial_failure_seen": saw_block,
            "recovered": recovered,
            "final_correct": final_ok,
            "extra_tool_calls": len(tools)}


def score_run(meta, gold, record):
    """Full metric set A-M for one question x system record."""
    tools = record.get("tool_calls") or []
    f1 = tool_f1(meta.get("expected_tools"), tools)
    rep = repeated_action_rate(tools)
    corr = scientific_correctness(meta["id"], meta, gold, record)
    lead_declared = declares_failure(record.get("final_answer"), lead=True)
    comp = completion(meta, corr, record)
    contract = 0 if lead_declared else contract_satisfaction(meta, corr)
    fail = failure_class(record, corr, f1, rep, meta)
    rec = recovery_record(meta, record, corr)
    quality = None
    parts = [x for x in (comp, contract, corr.get("score")) if isinstance(x, (int, float))]
    if parts:
        quality = round(sum(parts) / len(parts), 3)   # deterministic proxy, NOT an LLM judge
    return {
        "qid": meta["id"], "system": record.get("system"),
        "A_completion": comp, "B_contract": contract,
        "C_tool_precision": f1["precision"], "C_tool_recall": f1["recall"],
        "C_tool_f1": f1["f1"],
        "D_scientific": corr.get("score"), "D_checks": corr.get("checks"),
        "E_citation_ok": corr.get("checks", {}).get("gold_doi_cited"),
        "F_recovered": rec["recovered"], "F_detail": rec,
        "G_repeat_rate": rep,
        "H_tool_calls": len(tools), "I_llm_calls": record.get("n_llm_calls", 0),
        "J_prompt_tokens": record.get("prompt_tokens", 0),
        "K_completion_tokens": record.get("completion_tokens", 0),
        "L_cost_usd": record.get("estimated_cost_usd"),
        "M_quality_proxy": quality,
        "termination": record.get("termination"),
        "failure": fail,
        "lead_declared": lead_declared,
        "seconds": record.get("seconds"),
    }


def paired_stats(scores_a, scores_b, key="A_completion"):
    """Paired comparison on shared qids: wins/losses/ties (n=20 too small
    for significance claims; report counts + exact binomial CI on win rate)."""
    pairs = []
    by_b = {s["qid"]: s for s in scores_b}
    for s in scores_a:
        o = by_b.get(s["qid"])
        if o is None:
            continue
        va, vb = s.get(key), o.get(key)
        if va is None or vb is None:
            continue
        pairs.append((va > vb, va < vb, va == vb))
    wins = sum(1 for w, _l, _t in pairs if w)
    loss = sum(1 for _w, l, _t in pairs if l)
    n = len(pairs)
    lo, hi = _binom_ci(wins, n) if n else (0.0, 1.0)
    return {"n_pairs": n, "wins": wins, "losses": loss, "ties": n - wins - loss,
            "win_rate": round(wins / n, 3) if n else None,
            "win_rate_ci95": [round(lo, 3), round(hi, 3)]}


def _binom_ci(k, n, z=1.96):
    """Wilson 95% interval for a binomial proportion."""
    if not n:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - m) / d), min(1.0, (c + m) / d)
