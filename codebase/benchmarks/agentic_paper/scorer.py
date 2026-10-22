"""Deterministic, transparent scorer. No LLM judge anywhere.

Every dimension is computed independently from (a) the frozen question spec
(questions/questions_v2.json), (b) the episode record (final answer, raw tool
calls, provider events) and (c) known-DOI sets built from the data.

Definitions (also in README.md):

  completion   mandatory output requirements are PRESENT in the final answer:
               the episode ended with an answer, did not declare failure (for
               answer-type questions), and every mandatory check's *presence*
               rule holds (a number in a loose band around the reference, the
               DOI string, the required terms, an explicit decline statement for
               insufficient-evidence questions). A model saying it is finished
               is not sufficient.
  correct      binary: completion AND every mandatory check passes its *value*
               rule (tolerances, units, DOI cited as supporting evidence, names
               recalled, no forbidden fabricated claim). correct implies complete.
  partial      fraction of ALL checks (mandatory and optional) that pass.
  tool F1      precision / recall / F1 of the SET of canonical tool actions used
               against the best-matching acceptable reference plan (required +
               optional actions). Auxiliary calculator calls are excluded.
               Sequencing constraints and missing dependencies are recorded
               separately.
"""
import json
import re
from functools import lru_cache

from .paths import QUESTIONS_FILE, REFERENCE_DIR

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s)\]>,;\"'|]+", re.I)
NUM_RE = re.compile(r"(?<![\w.])[-−]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")

# Rejection language: a DOI mentioned in a sentence with these words is not
# counted as supporting evidence (carried over from the original scorer).
REJECTION_WORDS = (
    "reject", "exclud", "disqualif", "insufficient", "lacks", "lacking",
    "failed", "failure", "did not", "does not", "do not", "not meet",
    "not match", "not fulfil", "not fulfill", "ineligible", "ruled out",
    "set aside", "passed over", "not qualifi")

# Lead-of-answer non-delivery declarations (carried over from the original scorer).
FAILURE_DECLARATIONS = (
    "could not be established", "could not be finalized", "could not be determined",
    "no qualifying", "nothing could be established", "was never established",
    "was unable to", "were unable to", "could not find", "could not locate",
    "no evidence was found", "cannot answer", "cannot be answered", "gave up")
LEAD_CHARS = 300


@lru_cache(maxsize=1)
def load_bank():
    doc = json.load(open(QUESTIONS_FILE))
    return doc["tool_action_taxonomy"], {q["id"]: q for q in doc["questions"]}


@lru_cache(maxsize=1)
def load_known_dois():
    k = json.load(open(REFERENCE_DIR / "known_dois.json"))
    return {d.lower() for d in k["corpus"]}, {d.lower() for d in k["database"]}


# ------------------------------------------------------------------ text utils
def clean(text):
    return str(text or "").replace("−", "-").replace("–", "-")


def strip_dois(text):
    return DOI_RE.sub(" ", text)


def parse_numbers(text):
    """[(value, start, end)] of numbers in the text, DOIs removed first."""
    t = strip_dois(clean(text))
    out = []
    for m in NUM_RE.finditer(t):
        raw = m.group(0).replace(",", "").replace("−", "-")
        try:
            out.append((float(raw), m.start(), m.end()))
        except ValueError:
            pass
    return out, t


def alnum(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def declares_failure(answer, lead=True):
    text = str(answer or "")
    low = (text[:LEAD_CHARS] if lead else text).lower()
    return any(d in low for d in FAILURE_DECLARATIONS)


def doi_cited_as_evidence(answer, doi):
    for sent in re.split(r"(?<=[.!?])\s+|\n+", str(answer or "")):
        if doi.lower() in sent.lower():
            if not any(w in sent.lower() for w in REJECTION_WORDS):
                return True
    return False


# ---------------------------------------------------------------------- checks
def _windows(t, near):
    """Text windows around occurrences of near['term'] (regex)."""
    spans = []
    for m in re.finditer(near["term"], t, re.I):
        spans.append((max(0, m.start() - near.get("before", 0)), m.end() + near.get("after", 100)))
    return spans


def check_number(c, text):
    nums, t = parse_numbers(text)
    if "near" in c:
        spans = _windows(t, c["near"])
        nums = [n for n in nums if any(a <= n[1] < b for a, b in spans)]
    scales = [1.0] + list(c.get("scale_alts", []))
    pres = ok = False
    for v in c["values"]:
        for s in scales:
            ref = v * s
            tol = max(c.get("abs_tol", 0.0), c.get("rel_tol", 0.0) * abs(ref))
            if "abs_tol" not in c and "rel_tol" not in c:
                tol = 0.05 * abs(ref)
            loose = max(0.5 * abs(ref), 0.5)
            for n, _a, _b in nums:
                d = abs(n - ref)
                if d <= tol + 1e-9:
                    ok = pres = True
                if d <= loose:
                    pres = True
    return pres, ok


def _norm_terms(text):
    low = clean(text).lower()
    return low, alnum(low)


def check_terms(c, text):
    low, flat = _norm_terms(text)

    def hit(term):
        tl = term.lower()
        return tl in low or (len(alnum(tl)) >= 4 and alnum(tl) in flat)

    ok = all(hit(t) for t in c.get("all_of", [])) and \
        (not c.get("any_of") or any(hit(t) for t in c["any_of"]))
    return ok, ok


def check_groups(c, text):
    ok = all(check_terms({"any_of": g}, text)[1] for g in c["groups"])
    return ok, ok


def check_recall(c, text):
    flat = alnum(clean(text))
    got = sum(1 for alts in c["items"] if any(alnum(a) and alnum(a) in flat for a in alts))
    frac = got / max(1, len(c["items"]))
    ok = frac + 1e-9 >= c["min_fraction"]
    return ok, ok


def check_doi_evidence(c, text):
    pres = any(d.lower() in str(text).lower() for d in c["dois"])
    ok = any(doi_cited_as_evidence(text, d) for d in c["dois"])
    return pres, ok


def dois_in(text):
    return {m.group(0).rstrip(".").lower() for m in DOI_RE.finditer(str(text or ""))}


def check_doi_known(c, text):
    found = dois_in(text)
    corpus, db = load_known_dois()
    pres = len(found) >= c["min_count"]
    ok = pres and all(d in corpus or d in db for d in found)
    return pres, ok


def check_range(c, text):
    nums, _ = parse_numbers(text)
    ok = any(c["lo"] <= n <= c["hi"] for n, _a, _b in nums)
    return ok, ok


def check_min_numbers(c, text):
    nums, t = parse_numbers(text)
    dec = [n for n in nums if "." in t[n[1]:n[2]]]
    ok = len(dec) >= c["count"]
    return ok, ok


def check_decline(c, text):
    low = clean(text).lower()
    pres = any(p in low for p in c["phrases"])
    forbidden = [r for r in c.get("forbidden_regex", []) if re.search(r, low)]
    return pres, bool(pres and not forbidden)


CHECKS = {"number": check_number, "terms": check_terms, "term_groups": check_groups,
          "set_recall": check_recall, "doi_evidence": check_doi_evidence,
          "doi_known": check_doi_known, "number_in_range": check_range,
          "min_numbers": check_min_numbers, "decline": check_decline}


# ----------------------------------------------------------------- tool metrics
def tool_metrics(q, tool_calls, taxonomy):
    used_seq = []
    n_unknown = 0
    for c in tool_calls:
        a = taxonomy.get(c["tool"])
        if a is None:
            n_unknown += 1
        elif a != "calc":
            used_seq.append(a)
    used = set(used_seq)
    best = None
    for plan in q["reference_plans"]:
        req, opt = set(plan["required"]), set(plan["optional"])
        allowed = req | opt
        if not used and not req:
            p = r = f = 1.0
        else:
            tp_p = len(used & allowed)
            p = tp_p / len(used) if used else 0.0
            r = len(used & req) / len(req) if req else 1.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
        cand = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                "plan": plan, "missing": sorted(req - used), "extra": sorted(used - allowed)}
        if best is None or (cand["f1"], cand["recall"]) > (best["f1"], best["recall"]):
            best = cand
    allowed = set(best["plan"]["required"]) | set(best["plan"]["optional"])
    call_prec = (sum(1 for a in used_seq if a in allowed) / len(used_seq)) if used_seq else None
    # sequencing: first occurrence order for constraints where both actions were used
    seq_ok, evaluated = True, 0
    for a, b in q.get("order_constraints", []):
        if a in used_seq and b in used_seq:
            evaluated += 1
            if used_seq.index(a) > used_seq.index(b):
                seq_ok = False
    return {"tool_precision": best["precision"], "tool_recall": best["recall"], "tool_f1": best["f1"],
            "actions_used": sorted(used), "actions_missing": best["missing"],
            "actions_extra": best["extra"], "call_precision": None if call_prec is None else round(call_prec, 3),
            "dependencies_satisfied": not best["missing"],
            "sequencing_satisfied": (seq_ok if evaluated else None),
            "n_unknown_tool_names": n_unknown}


def grounding(answer, tool_calls):
    """Share of decimal numbers in the answer that occur in some raw tool output.
    Heuristic diagnostic (derived numbers such as differences are flagged too)."""
    nums, t = parse_numbers(answer)
    dec = [(n, t[a:b]) for n, a, b in nums if "." in t[a:b]]
    if not dec:
        return None, 0
    pool = []
    for c in tool_calls:
        pool.extend(n for n, _a, _b in parse_numbers(c.get("result", ""))[0])
    hit = 0
    for n, raw in dec:
        d = len(raw.split(".")[1].rstrip("0123456789eE-+")) if "." in raw else 0
        d = len(raw.split(".")[1]) if "." in raw else 0
        tol = max(0.51 * 10 ** (-d), abs(n) * 0.005)
        if any(abs(n - p) <= tol for p in pool):
            hit += 1
    return round(hit / len(dec), 3), len(dec)


# -------------------------------------------------------------------- scoring
def score_episode(ep):
    taxonomy, bank = load_bank()
    q = bank[ep["qid"]]
    answer = clean(ep.get("final_answer"))
    term = ep.get("termination")
    results, behaviour = [], q["expected_behaviour"]
    infra = bool(ep.get("infra_failure")) or term in ("system_unavailable",)
    for c in q["checks"]:
        pres, ok = CHECKS[c["type"]](c, answer)
        results.append({"id": c["id"], "type": c["type"], "mandatory": c["mandatory"],
                        "present": bool(pres), "pass": bool(ok)})
    mandatory = [r for r in results if r["mandatory"]]
    answered = term == "answer" and bool(answer.strip())
    lead_fail = declares_failure(answer) and behaviour == "answer"
    complete = bool(answered and not lead_fail and mandatory and all(r["present"] for r in mandatory))
    correct = bool(complete and all(r["pass"] for r in mandatory))
    partial = round(sum(r["pass"] for r in results) / len(results), 3) if results else None
    tm = tool_metrics(q, ep.get("tool_calls") or [], taxonomy)
    g_frac, g_n = grounding(answer, ep.get("tool_calls") or [])
    prov = ep.get("provider") or {}
    n_calls = len(ep.get("tool_calls") or [])
    rep = ep.get("repeated_tool_calls", 0)
    errs = ep.get("tool_errors", 0)
    # ------------------------------------------------ failure analysis (multi-label)
    labels, evid = [], []
    unknown_dois = [d for d in dois_in(answer) if d not in load_known_dois()[0] and d not in load_known_dois()[1]]
    if infra:
        labels.append("infrastructure_failure")
        evid.append(f"provider: {prov.get('infra_reason') or term}")
    if not correct:
        if term in ("turn_budget", "tool_budget", "timeout"):
            labels.append("budget_exhaustion")
            evid.append(f"termination={term} after {n_calls} tool calls, {ep.get('n_model_calls')} model calls")
        if errs and (errs >= max(1, n_calls // 2)):
            labels.append("tool_errors")
            evid.append(f"{errs}/{n_calls} tool calls returned errors")
        if rep >= 2 or (n_calls >= 3 and rep / n_calls >= 0.34):
            labels.append("repeated_actions")
            evid.append(f"{rep} repeated identical tool calls out of {n_calls}")
        if tm["actions_missing"] and not infra:
            labels.append("incorrect_tool_selection")
            evid.append(f"required actions never used: {tm['actions_missing']}; extra: {tm['actions_extra']}")
        if not complete and not infra:
            labels.append("incomplete_answer")
            why = [r["id"] for r in mandatory if not r["present"]]
            evid.append("answer missing required outputs: " + (", ".join(why) or "no answer / declared failure"))
        if complete:
            bad = [r for r in mandatory if not r["pass"]]
            cit = [r["id"] for r in bad if r["type"] in ("doi_evidence", "doi_known")]
            val = [r["id"] for r in bad if r["type"] not in ("doi_evidence", "doi_known")]
            if cit:
                labels.append("incorrect_citation")
                evid.append(f"citation checks failed: {cit}")
            if val:
                labels.append("incorrect_value")
                evid.append(f"value checks failed: {val}")
    if unknown_dois:
        labels.append("unsupported_claim")
        evid.append(f"cites DOI(s) not in corpus or database: {unknown_dois[:3]}")
    if behaviour in ("decline", "partial_decline"):
        for c in q["checks"]:
            if c["type"] == "decline" and any(re.search(r, answer.lower()) for r in c.get("forbidden_regex", [])):
                labels.append("unsupported_claim")
                evid.append("asserts a value the available evidence cannot support")
    labels = list(dict.fromkeys(labels))
    order = ["infrastructure_failure", "budget_exhaustion", "tool_errors", "repeated_actions",
             "incorrect_tool_selection", "incomplete_answer", "incorrect_citation", "incorrect_value",
             "unsupported_claim"]
    primary = next((l for l in order if l in labels), "none") if (not correct or infra) else "none"
    toks_ok = (prov.get("usage_reported_calls") or 0) == (prov.get("model_calls") or -1) and \
        prov.get("model_calls", 0) > 0
    return {
        "episode_id": ep["episode_id"], "qid": ep["qid"], "category": q["category"],
        "set": q["set"], "condition": ep["condition"], "backbone": ep["backbone"],
        "model": ep["model"], "rep": ep["rep"], "behaviour": behaviour,
        "evaluation_mode": q["evaluation_mode"], "infra_failure": infra,
        "termination": term, "completed": int(complete), "correct": int(correct),
        "partial": partial, "checks": results,
        **tm,
        "grounded_frac": g_frac, "n_decimal_numbers": g_n,
        "seconds": ep.get("seconds"), "n_model_calls": ep.get("n_model_calls"),
        "n_tool_calls": n_calls, "repeated_tool_calls": rep, "tool_errors": errs,
        "retries": ep.get("retries"), "timeout": int(term == "timeout"),
        "rate_limit_events": prov.get("rate_limit_events"),
        "provider_error_events": prov.get("provider_error_events"),
        "pool_wait_s": prov.get("pool_wait_s"),
        "prompt_tokens": prov.get("real_prompt_tokens") if toks_ok else None,
        "completion_tokens": prov.get("real_completion_tokens") if toks_ok else None,
        "failure_labels": labels, "primary_failure": primary, "failure_evidence": evid,
        "manual_review": q["evaluation_mode"] != "deterministic",
    }
