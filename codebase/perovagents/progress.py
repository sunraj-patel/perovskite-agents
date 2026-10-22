"""Progress-aware execution: contracts that give every domain agent an explicit
subtask state, completion requirements, and an anti-loop guard.

Why: the generic agent loop previously knew only "keep calling tools until you
are satisfied". A free-text task carried no model of *required outputs*, *what is
already established*, or *what call could resolve the deficiency*, so agents could
re-run semantically equivalent calls (same search/read for literature, raw-row SQL
for pdb) until the turn cap without progressing.

This module adds a generic, deterministic control layer on top of the loop:

    build_state(task)   -> SubtaskState          (per-domain requirements)
    guard(state, call)  -> BLOCK reason | None    (before executing; stops loops)
    assess(state, call, result) -> assessment     (progress / partial / no-progress)
    review(state)       -> escalation | None      (when the agent declares FINAL)

Rules are keyword/result-shape based, not learned, and add no extra LLM calls.
"""
import re

from typing import Any, Dict, List, Optional

# Consecutive no-progress calls that stop a subtask. Small: a loop has no reason
# to run more than ~2 identical no-gain steps before control returns upstream.
MAX_NO_PROGRESS = 2
# Jaccard token similarity for two calls of the same tool to count as the same
# action. 0.8 tolerates re-typed/re-worded variants, not genuinely new ones.
REPEAT_SIM = 0.8

# Statuses emitted by progress assessment / review.
PROGRESS = "PROGRESS"
PARTIAL = "PARTIAL_PROGRESS"
NO_PROGRESS = "NO_PROGRESS"
SUFFICIENT = "SUFFICIENT"
BLOCKED = "BLOCKED"
INVALID = "INVALID_RESULT"
INCOMPLETE = "INCOMPLETE"

ESCALATIONS = {NO_PROGRESS, BLOCKED, INVALID, INCOMPLETE, "REPEATED_ACTION"}


def norm_text(text: str) -> str:
    """Normalise free text for duplicate detection (whitespace/case)."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def token_set(args: Any) -> frozenset:
    """Bag of lower-case tokens in an args dict/list/str, for similarity."""
    toks = set()

    def add(v):
        if isinstance(v, dict):
            for k, val in v.items():
                add(k)
                add(val)
        elif isinstance(v, (list, tuple)):
            for x in v:
                add(x)
        elif isinstance(v, str):
            toks.update(w for w in re.split(r"\W+", v.lower()) if len(w) > 1)
        else:
            toks.add(str(v).lower())

    add(args)
    return frozenset(toks)


def token_sim(a: Any, b: Any) -> float:
    """Jaccard similarity of two args' token sets (0..1)."""
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------- state

class Requirement:
    __slots__ = ("rid", "text", "satisfied")

    def __init__(self, rid: str, text: str):
        self.rid = rid
        self.text = text
        self.satisfied = False


class HistoryEntry:
    __slots__ = ("fp", "tool", "args", "status", "gained", "rejected", "note")

    def __init__(self, fp, tool, args, status, gained, rejected, note=""):
        self.fp = fp
        self.tool = tool
        self.args = args
        self.status = status
        self.gained = gained                      # requirement ids satisfied
        self.rejected = rejected                  # (target, reason) pairs
        self.note = note


class SubtaskState:
    """Everything the progress layer knows about one domain subtask."""

    def __init__(self, agent: str, task: str):
        self.agent = agent
        self.task = task
        self.requirements: List[Requirement] = []
        self.rejected: List[tuple] = []            # (target, reason, requirement_id)
        self.history: List[HistoryEntry] = []
        self.status = "RUNNING"

    # ---- requirements ----------------------------------------------------
    def add_requirement(self, rid: str, text: str) -> Requirement:
        self.requirements.append(Requirement(rid, text))
        return self.requirements[-1]

    def requirement(self, rid: str) -> Optional[Requirement]:
        return next((r for r in self.requirements if r.rid == rid), None)

    def satisfied(self) -> List[str]:
        return [r.rid for r in self.requirements if r.satisfied]

    def missing(self) -> List[tuple]:
        return [(r.rid, r.text) for r in self.requirements if not r.satisfied]

    # ---- results ---------------------------------------------------------
    def apply(self, tool, args, fp, status, newly_satisfied, rejected, note=""):
        self.history.append(HistoryEntry(fp, tool, args, status, newly_satisfied, rejected, note))
        for rid in newly_satisfied:
            r = self.requirement(rid)
            if r is not None:
                r.satisfied = True
        for target, reason, rid in rejected:
            self.rejected.append((target, reason, rid))

    def is_repeat(self, tool, args, fp) -> bool:
        """Same tool, and either identical args or very similar args, seen before."""
        if fp is None:
            return False                    # unparseable call: nothing to compare against
        identical = any(h.fp == fp for h in self.history)
        if identical:
            return True
        return any(h.tool == tool and token_sim(h.args, args) >= REPEAT_SIM
                   for h in self.history)

    def tools_called(self) -> List[str]:
        return list(dict.fromkeys(h.tool for h in self.history))

    def no_progress_streak(self) -> int:
        streak = 0
        for h in reversed(self.history):
            if h.status == NO_PROGRESS:
                streak += 1
            else:
                break
        return streak

    def successful_raw_selects(self) -> int:
        return sum(1 for h in self.history
                   if h.tool == "sql" and not h.gained and not h.rejected)


# --------------------------------------------------------------------------- contracts

class Contract:
    """Generic hooks a domain agent provides. All rules are deterministic."""

    name = "generic"

    def new_state(self, task: str) -> SubtaskState:
        state = SubtaskState(self.name, task)
        self._requirements(task, state)
        return state

    def _requirements(self, task: str, state: SubtaskState):
        pass                          # subclasses add Requirement entries

    def _futile_repeat(self, state, tool, args, fp) -> Optional[str]:
        """The call re-does something whose earlier attempt gained nothing.

        Calls that ended in a technical failure (ERROR_RETRY) or an invalid
        result (INVALID) are skipped: a bounded retry after those is legitimate
        (the numeric REPEAT_LIMIT still caps identical retries).
        """
        if not state.history or tool not in state.tools_called():
            return None
        if fp is None:
            return None                     # unparseable call: nothing to compare against
        for h in state.history:
            if h.status in ("ERROR_RETRY", INVALID) or h.gained:
                continue
            if h.fp == fp:
                return (f"REPEATED_ACTION: {tool}({args}) was called before with the same "
                        f"arguments and added nothing new. Repeating it cannot answer the missing "
                        f"requirements. Change the query/strategy before trying again.")
        for h in state.history:
            if h.status in ("ERROR_RETRY", INVALID) or h.gained:
                continue
            if h.tool == tool and token_sim(h.args, args) >= REPEAT_SIM \
                    and not h.rejected:
                return (f"REPEATED_ACTION: {tool}({args}) with nearly identical arguments "
                        f"({h.note or 'previous call'}) already produced no progress. "
                        f"This is a semantic retry; change the strategy.")
        return None

    def guard(self, state: SubtaskState, tool: str, args: dict, fp: str) -> Optional[str]:
        """Return a diagnostic to STOP with, or None to allow the call."""
        return self._futile_repeat(state, tool, args, fp)

    def assess(self, state: SubtaskState, tool: str, args: dict, result: str, error: bool,
               fp: str) -> dict:
        """Classify one tool result. Returns {status, newly_satisfied, rejected, note}."""
        if error:
            return {"status": "ERROR_RETRY", "newly_satisfied": [], "rejected": [],
                    "note": "technical failure; a retry with corrected args is allowed"}
        if all(r.satisfied for r in state.requirements):
            return {"status": SUFFICIENT, "newly_satisfied": [], "rejected": [], "note": ""}
        return {"status": PROGRESS, "newly_satisfied": [], "rejected": [], "note": ""}

    def review(self, state: SubtaskState) -> Optional[str]:
        """Escalation when the agent declares FINAL but the contract is unmet."""
        missing = state.missing()
        if not missing:
            return None
        return f"{INCOMPLETE}: still missing {missing}; report what you could not establish."


# --------------------------------------------------------------------------- PDB

# aggregate families the DB actually supports (SQLite has no MEDIAN).
# v3: every pattern is word-bounded. The v2 `vs\.?` matched inside "perovskite"
# (perovs-kite), so EVERY pdb task demanded an aggregate statistic and a correct
# ORDER BY ... LIMIT row ranking was rejected as "raw rows" (benchmark Q04 trace).
_AGG_FAMILIES = {
    "ranking":    re.compile(r"\btop\b|\bmost\s+(common|frequent|used|popular)|\branked?\b|\bbest\b", re.I),
    "mean":       re.compile(r"\bmean\b|\baverage\b|\bavg\b", re.I),
    "count":      re.compile(r"\bhow\s+many\b|\bcount\b|\bnumber\s+of\b", re.I),
    "distribution": re.compile(r"\bdistribut\w*|\bhistogram\b|\bspread\b", re.I),
    "comparison": re.compile(r"\bcompar\w*|\bvs\.?(?!\w)|\bversus\b", re.I),
}
# Statistics that REQUIRE an aggregate function (GROUP BY/AVG/COUNT ...).
_STAT_FAMILIES = ("mean", "count", "distribution", "comparison")
# "highest-PCE devices", "top 5 cells": a ranking of individual records. An ordered,
# limited SELECT of the rows is the right answer; no aggregate function is needed.
_CATEGORY = r"(?:materials?|htls?|etls?|compositions?|architectures?|solvents?|additives?|layers?|stacks?|dopants?)"
_ROW_RANK = re.compile(
    r"\b(?:top|highest|lowest|best|worst)\b[\s-]*"
    rf"(?:(?!{_CATEGORY}\b)[\w-]+\W+){{0,6}}?"                 # a few modifiers, never a category noun
    r"(?:devices?|cells?|records?|entries)\b"                    # "top 5 devices", "highest-PCE reported devices"
    r"|\b(?:highest|lowest|best|worst)[\s-]+(?:pce|efficiency|voc|jsc|ff|fill factor)\b", re.I)
_FREQ_RANK = re.compile(r"\bmost\s+(common|frequent|used|popular)\b|\bhow\s+many\b|\bcount\b|"
                        r"\bnumber\s+of\b|\bfrequen\w*\b", re.I)
_PDB_AGG_RID = "aggregate"          # an aggregate statistic requested
_PDB_RANK_RID = "ranking"           # an ordered, limited row ranking requested


def wants_row_ranking(task: str) -> bool:
    """True for a ranking of individual devices ("highest-PCE devices"), False for
    a ranking of categories ("top 5 ETL materials", "most common HTL"), which needs
    GROUP BY. A statistic request in the same task (mean/count/...) also wins."""
    low = task.lower()
    if any(_AGG_FAMILIES[f].search(low) for f in _STAT_FAMILIES) or _FREQ_RANK.search(low):
        return False
    return bool(_ROW_RANK.search(low))


class PdbContract(Contract):
    name = "pdb"

    def _requirements(self, task, state):
        low = task.lower()
        if wants_row_ranking(task):
            state.add_requirement(_PDB_RANK_RID,
                                  "a ranking of the devices: ONE SELECT ... ORDER BY <metric> DESC "
                                  "LIMIT n returning the requested columns (no GROUP BY needed)")
        elif any(pat.search(low) for pat in _AGG_FAMILIES.values()):
            state.add_requirement(_PDB_AGG_RID,
                                  "an aggregate statistic (GROUP BY/AVG/COUNT + ORDER BY + LIMIT "
                                  "for rankings), not just raw rows")
        if re.search(r"n-?i-?p", task):
            state.add_requirement("filter:arch-nip", "the query filters Cell_architecture = 'nip'")
        if re.search(r"p-?i-?n", task):
            state.add_requirement("filter:arch-pin", "the query filters Cell_architecture = 'pin'")
        for m in re.finditer(r"(?:PCE|efficiency)[^\d]*(\d+(?:\.\d+)?)\s*%", low):
            state.add_requirement("filter:pce", f"the query filters JV_default_PCE >= {m.group(1)}%")

    @staticmethod
    def _sql_has_aggregate(sql: str) -> bool:
        s = re.sub(r"\s+", " ", (sql or "").lower())
        return bool(re.search(r"\b(group\s+by|avg\(|sum\(|count\(|min\(|max\(|count\s*\*)", s))

    @staticmethod
    def _sql_has_ranking(sql: str) -> bool:
        s = re.sub(r"\s+", " ", (sql or "").lower())
        return bool(re.search(r"\border\s+by", s)) and bool(re.search(r"\blimit\b", s))

    def guard(self, state, tool, args, fp):
        if tool != "sql":
            return super().guard(state, tool, args, fp)
        sql = (args.get("query") or "")
        # a raw (non-aggregate) SELECT when an aggregate is due and rows were
        # already fetched once: do NOT fetch more rows, aggregate the ones you have.
        # NOTE: the count covers EVERY successful raw SELECT, including ones
        # that satisfied a filter requirement (Q02 pilot hole: a raw SELECT
        # satisfying filter:arch-nip left successful_raw_selects() at 0, so
        # the block never fired and the agent looped raw selects forever).
        agg = state.requirement(_PDB_AGG_RID)
        if agg and not agg.satisfied and not self._sql_has_aggregate(sql):
            raw = sum(1 for h in state.history
                      if h.tool == "sql" and h.status not in ("ERROR_RETRY", INVALID)
                      and not self._sql_has_aggregate(
                          (h.args if isinstance(h.args, dict) else {}).get("query", "")))
            if raw >= 1:
                return (f"BLOCKED: aggregation required ({agg.text}), but this SQL only fetches "
                        f"raw rows again. Aggregate the rows you already have with ONE query: "
                        f"e.g. GROUP BY 1 + ORDER BY 2 DESC + LIMIT, or AVG/COUNT. Do not fetch "
                        f"more rows.")
        return super().guard(state, tool, args, fp)

    def assess(self, state, tool, args, result, error, fp):
        if error:
            return {"status": "ERROR_RETRY", "newly_satisfied": [], "rejected": [],
                    "note": "technical failure; retry with corrected SQL"}
        sat = []
        if tool == "sql":
            sql = (args.get("query") or "")
            agg = state.requirement(_PDB_AGG_RID)
            if agg and not agg.satisfied:
                if self._sql_has_aggregate(sql):
                    if agg.rid == "aggregate" and not self._sql_has_ranking(sql) \
                            and not re.match(r"top|most", state.task, re.I) \
                            and not _AGG_FAMILIES["ranking"].search(state.task):
                        pass                     # plain mean/count needs no ORDER BY
                    sat.append(agg.rid)
            rk = state.requirement(_PDB_RANK_RID)
            if rk and not rk.satisfied and self._sql_has_ranking(sql) \
                    and "'n_rows': 0" not in str(result)[:200]:
                sat.append(rk.rid)
            for rid in ("filter:arch-nip", "filter:arch-pin"):
                r = state.requirement(rid)
                if r and not r.satisfied:
                    fam = "nip" if rid.endswith("nip") else "pin"
                    if "cell_architecture" in sql.lower() and fam in sql.lower().replace("-", ""):
                        sat.append(rid)
            r = state.requirement("filter:pce")
            if r and not r.satisfied and "jv_default_pce" in sql.lower() \
                    and re.search(r"[><=]", sql):
                sat.append(r.rid)
        # no new info and it repeats a no-gain action -> negative signal
        note = ""
        if not sat:
            if state.is_repeat(tool, args, fp):
                return {"status": NO_PROGRESS, "newly_satisfied": [], "rejected": [],
                        "note": "same/very similar call already made; no new requirement met"}
            note = "distinct exploratory call; nothing new satisfied yet"
        return {"status": PROGRESS if sat else PARTIAL,
                "newly_satisfied": sat, "rejected": [], "note": note}

    def review(self, state):
        m = state.missing()
        if not m:
            return None
        rk = state.requirement(_PDB_RANK_RID)
        if rk and not rk.satisfied:
            return (f"{INCOMPLETE}: a row ranking was required ({rk.text}) but no ordered, limited "
                    f"SELECT returned rows. Recommended next action: one SELECT with the filters, "
                    f"ORDER BY the metric DESC LIMIT n.")
        agg = state.requirement(_PDB_AGG_RID)
        if agg and not agg.satisfied:
            return (f"{INCOMPLETE}: aggregation was required ({agg.text}) but never produced. "
                    f"Recommended next action: one SELECT with the filters, then GROUP BY 1 "
                    f"ORDER BY 2 DESC LIMIT (or AVG/COUNT).")
        return super().review(state)


# --------------------------------------------------------------------------- Literature

_ARCH_NIP = re.compile(r"\bn-?i-?p\b|\bnip\b", re.I)
_ARCH_PIN = re.compile(r"\bp-?i-?n\b|\bpin\b", re.I)
_MATERIAL_RE = re.compile(
    r"\b(sno2|tio2|spiro[- ]?meotad|spiro[- ]?ometad|ptaa|pedot[- ]?pss|fapbi3|mapbi3|csmafa|"
    r"ni ?o|nio ?x|cu ?scn|p3ht|c60|pcbm|pbi ?(?:_|)2|fapbi|thiourea|cds)\b", re.I)
_PROPERTY_RE = re.compile(
    r"\b(work function|mobilit(?:y|ies)|thickness|band ?gap|"                 # noqa: W605
    r"(?:electron|hole).*mobility|defect densit|stabilit)\b|cm2?/ ?vs|\beV\b", re.I)
# v3: a bare "W" symbol and an unbounded "eV" suffix matched DOI tails ("...03285-w") and words
# ending in "ev" (dev, prev); DOIs are also removed from the task before property detection.
_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+", re.I)


class LiteratureContract(Contract):
    name = "literature"

    def _requirements(self, task, state):
        low = task.lower()
        if _ARCH_NIP.search(task):
            state.add_requirement("arch:nip", "architecture evidence for an n-i-p device")
        if _ARCH_PIN.search(task):
            state.add_requirement("arch:pin", "architecture evidence for a p-i-n device")
        for m in _MATERIAL_RE.finditer(task):
            mat = re.sub(r"[-_ ]+", "", m.group(1).lower())
            if not state.requirement(f"material:{mat}"):
                state.add_requirement(f"material:{mat}", f"the paper contains material {m.group(1)}")
        prop = _PROPERTY_RE.search(_DOI_RE.sub(" ", task))
        if prop:
            key = re.sub(r"\W+", "_", prop.group(0).lower()).strip("_")[:24]
            state.add_requirement(f"property:{key}", f"numerical {prop.group(0)} evidence")

    def _paper_ok(self, state, doi, result):
        """Mark rejected when a paper cannot back the *architecture* requirement."""
        out = []
        low = str(result).lower().replace("-i-", "i")
        for rid in ("arch:nip", "arch:pin"):
            r = state.requirement(rid)
            if not r or r.satisfied:
                continue
            fam = rid.split(":")[1]                       # 'nip' / 'pin'
            fam_re = _ARCH_NIP if fam == "nip" else _ARCH_PIN
            if fam_re.search(low):
                continue                                   # evidence found
            other_re = _ARCH_PIN if fam == "nip" else _ARCH_NIP
            other = "pin" if fam == "nip" else "nip"
            if other_re.search(low):
                out.append((doi, f"passages are p-{other if other=='nip' else 'i'}-n, "
                                  f"task requires {rid.split(':')[1]}", rid))
            else:
                out.append((doi, f"passages carry no architecture evidence for {rid}", rid))
        return out

    def guard(self, state, tool, args, fp):
        if tool == "read_paper":
            doi = (args.get("doi") or "").strip().lower()
            rejected = [t for (t, _reason, _rid) in state.rejected
                        if str(t).strip().lower() == doi]
            if rejected and any(not r.satisfied for r in state.requirements
                                if r.rid.startswith("arch:")):
                return (f"BLOCKED: DOI {doi} was already read and rejected "
                        f"({rejected[0]} evidence). Reading it again returns the same "
                        f"passages. Read a different DOI, or change the search so it returns "
                        f"a paper that can satisfy the missing requirement.")
        return super().guard(state, tool, args, fp)

    def assess(self, state, tool, args, result, error, fp):
        if error:
            return {"status": "ERROR_RETRY", "newly_satisfied": [], "rejected": [],
                    "note": "technical failure; retry with a valid DOI/query"}
        low = str(result).lower().replace("-i-", "i")
        sat, rejected = [], []
        doi = (args.get("doi") or "").strip().lower() if tool == "read_paper" else None
        if tool == "read_paper" and doi:
            if doi == "error" or "no paper with doi" in low:
                return {"status": INVALID, "newly_satisfied": [], "rejected": [],
                        "note": f"DOI {doi} not in the corpus"}
            rejected = self._paper_ok(state, doi, str(result))
        for r in state.requirements:
            if r.satisfied:
                continue
            rid = r.rid
            if rid.startswith("arch:"):
                fam = rid.split(":")[1]
                fam_re = _ARCH_NIP if fam == "nip" else _ARCH_PIN
                if tool == "read_paper" and fam_re.search(low):
                    sat.append(rid)
            elif rid.startswith("material:"):
                mat = rid.split(":")[1]
                if tool == "read_paper" and mat in re.sub(r"[-_ ]+", "", low):
                    sat.append(rid)
            elif rid.startswith("property:"):
                if tool == "read_paper" and _PROPERTY_RE.search(str(result)):
                    sat.append(rid)
        if not sat and not rejected:
            if state.is_repeat(tool, args, fp):
                return {"status": NO_PROGRESS, "newly_satisfied": [], "rejected": [],
                        "note": "same/very similar search or read already made; no new evidence"}
            return {"status": PARTIAL, "newly_satisfied": [], "rejected": [],
                    "note": "distinct step; evidence pending"}
        return {"status": PROGRESS, "newly_satisfied": sat, "rejected": rejected, "note": ""}

    def review(self, state):
        m = state.missing()
        if not m:
            return None
        return (f"{INCOMPLETE}: {m} could not be established. Corpus papers read: "
                f"{[h.args.get('doi') for h in state.history if h.tool == 'read_paper']}. "
                f"Rejected: {[t for (t, _r, _ri) in state.rejected]}. Report the gap and "
                f"cite only papers you actually read.")


# --------------------------------------------------------------------------- ML

_ML_FLAGS = {
    "predict":         (re.compile(r"\b(predict|predicts|predicted pce|estimate|estimate.*pce|"
                                   r"what pce|would .{0,30}reach|forecast)\b", re.I),
                       {"predict"}),
    "global":          (re.compile(r"\b(most important|important parameters?|matters most|"
                                   r"drive[sd]? .*pce|dominant .*parameters?|globally|across .*"
                                   r"(devices|designs|dataset|models?)|overall .*important)\b", re.I),
                       {"explain_global"}),
    "local":           (re.compile(r"\b(explain|why|because).{0,60}(this|that|single|the model|"
                                   r"device)s?\b|\bfor this device\b|local\b", re.I),
                       {"explain_local"}),
    "what_if":         (re.compile(r"\b(what if|sweep|sensitive|sensitivity|vary|how .{0,20}"
                                   r"change|increase|decrease |from \d|to \d)\b", re.I),
                       {"what_if"}),
    "optimize":        (re.compile(r"\b(optimiz|which (parameters?|changes?|interventions?|"
                                   r"thickness).{0,20}(improve|maximize|reach|increase)|"
                                   r"(best|better) way|how (can|should) i (change|increase|push)|"
                                   r"recommend a better)\w*\b", re.I),
                       {"optimize", "what_if"}),
}


class MLContract(Contract):
    name = "ml"

    def _requirements(self, task, state):
        for rid, (pat, _tools) in _ML_FLAGS.items():
            if pat.search(task):
                state.add_requirement(f"intent:{rid}",
                                      f"answer the {rid} question (tool(s): {sorted(_tools)})")

    def assess(self, state, tool, args, result, error, fp):
        if error:
            return {"status": "ERROR_RETRY", "newly_satisfied": [], "rejected": [],
                    "note": "technical failure; retry with a parseable description"}
        sat = []
        for r in state.requirements:
            if r.satisfied or not r.rid.startswith("intent:"):
                continue
            allowed = _ML_FLAGS[r.rid.split(":")[1]][1]
            if tool in allowed:
                sat.append(r.rid)
        if not sat:
            if state.is_repeat(tool, args, fp):
                return {"status": NO_PROGRESS, "newly_satisfied": [], "rejected": [],
                        "note": "same/very similar ML call repeated; nothing new satisfied"}
            return {"status": PARTIAL, "newly_satisfied": [], "rejected": [],
                    "note": f"{tool} does not satisfy any open requirement"}
        return {"status": PROGRESS, "newly_satisfied": sat, "rejected": [], "note": ""}

    def review(self, state):
        missing = state.missing()
        if not missing:
            return None
        parts = []
        for rid, text in missing:
            flag = rid.split(":")[1]
            allowed = sorted(_ML_FLAGS[flag][1])
            called = [h.tool for h in state.history]
            if not any(t in called for t in allowed):
                parts.append(f"{text}; call {allowed} (none of them was tried)")
            else:
                parts.append(f"{text} (tool tried but result was insufficient)")
        if not parts:
            return None
        return (f"{INCOMPLETE}: the task asked for ML outputs the model has tools for, but "
                f"the final report did not establish: {'; '.join(parts)}. Recommended: run the "
                f"missing tool(s) before reporting.")


# --------------------------------------------------------------------------- SCAPS

_SCAPS_PARAM_FAMILIES = {
    "thickness":       re.compile(r"thickness|thick\b|\bhtl_d\b|\betl_d\b", re.I),
    "band_gap":        re.compile(r"band[_\s]?gap|\beg\b", re.I),
    "mobility":        re.compile(r"mobilit|\bmun\b|\bmup\b", re.I),
    "defect_density":  re.compile(r"defect[_\s]?densit|\bnt\b|recombination", re.I),
    "work_function":   re.compile(r"work[_\s]?function|\bchi\b|affinit", re.I),
    "doping":          re.compile(r"doping|\bna\b|\bnd\b", re.I),
}


class ScapsContract(Contract):
    name = "scaps"

    def _requirements(self, task, state):
        low = task.lower()
        found = False
        for fam, pat in _SCAPS_PARAM_FAMILIES.items():
            if pat.search(low):
                state.add_requirement(f"sw:{fam}", f"a simulation over {fam} (or a converged "
                                                   f"single point for it)")
                found = True
        if re.search(r"\b(voc|jsc|ff|pce|performance|sensit)\b", low) or re.search(r"sweep", low):
            state.add_requirement("sw:run", "a solver output (get_sweep/sweep/run_point) being "
                                            "parsed and reported")
            found = True
        if not found:
            state.add_requirement("sw:read", "the reference cell parameters used")

    @staticmethod
    def _has_data(result: str) -> bool:
        low = result.lower()
        return "pce" in low and ("data" in low or "converged" in low)

    def assess(self, state, tool, args, result, error, fp):
        if error:
            return {"status": "ERROR_RETRY", "newly_satisfied": [], "rejected": [],
                    "note": "technical failure; correct the parameter/value"}
        low = str(result).lower()
        sat = []
        for r in state.requirements:
            if r.satisfied:
                continue
            if r.rid == "sw:read":
                if tool == "read_def" and "found" in low:
                    sat.append(r.rid)
                continue
            family = r.rid.split(":", 1)[1]
            if family == "run":
                if tool in ("get_sweep", "sweep", "run_sweep_live") and self._has_data(low):
                    sat.append(r.rid)
                elif tool == "run_point" and "converged" in low:
                    sat.append(r.rid)
                continue
            if tool in ("get_sweep", "run_sweep_live", "sweep", "run_point") \
                    and self._has_data(low):
                fam_pat = _SCAPS_PARAM_FAMILIES.get(family)
                if fam_pat and fam_pat.search(low):
                    sat.append(r.rid)
                elif not fam_pat:
                    sat.append(r.rid)
        if "convergence failure" in low or "no iv curve" in low:
            return {"status": INVALID, "newly_satisfied": sat, "rejected": [],
                    "note": "solver did not converge; retry with corrected parameters (bounded)"}
        if not sat:
            if state.is_repeat(tool, args, fp):
                return {"status": NO_PROGRESS, "newly_satisfied": [], "rejected": [],
                        "note": "same sweep/set repeated; nothing new satisfied"}
            return {"status": PARTIAL, "newly_satisfied": [], "rejected": [],
                    "note": "distinct simulator step; result pending"}
        return {"status": PROGRESS, "newly_satisfied": sat, "rejected": [], "note": ""}

    def review(self, state):
        m = state.missing()
        if not m:
            return None
        return (f"{INCOMPLETE}: {m}. Run the appropriate get_sweep/sweep/run_point (or a "
                f"converged run_point for a single point) before reporting a performance "
                f"claim.")


CONTRACTS: Dict[str, Contract] = {
    "pdb": PdbContract(),
    "literature": LiteratureContract(),
    "ml": MLContract(),
    "scaps": ScapsContract(),
}


def contract_for(agent_name: str) -> Optional[Contract]:
    return CONTRACTS.get(agent_name)