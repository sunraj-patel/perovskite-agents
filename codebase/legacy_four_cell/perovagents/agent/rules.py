"""Rule-based planner (baseline).

A deterministic keyword-based planner with no LLM calls. Its job:
  1. a rule baseline against which to measure LLM planning (benchmark later),
  2. an offline fallback when no API key is available.

It returns `PlanStep` objects structured exactly like the LLM planner's.
"""
import re
from dataclasses import dataclass

from .models import Plan, PlanStep

RULES = [
    # (regex pattern, [tools in order])
    (r"simulate|SCAPS|physics\s*model|sensitivity|parametric|test\s+whether|does\s+the\s+physics",
     ["scaps"]),
    (r"what\s+should\s+i\s+change|recommend|optimize|improve\w*\s+.*device|"
     r"boost\s+.*pce|best\s+parameter|optimal|optimum|should\s+i\s+use\b|"
     r"to\s+(maximize|reach|achieve)\b",
     ["ml"]),
    (r"what\s+happens\s+if|how\s+sensitive|affect.*(thickness|PCE|mobility|property)|"
     r"sweep|vary\s+|change\w*\s+with\b|how\s+\w+\s+changes?\s+with\b",
     ["ml"]),
    (r"this\b.{0,25}\b(device|sample|stack)|this\s+stack",
     ["ml"]),
    (r"most\s+(relevant|important)|"
     r"which\s+(parameters|features)\s+matter|"
     r"what\s+(drives|influences)\s+(the\s+)?(pce|performance)",
     ["ml"]),
    (r"band\s*gap|formation\s*energy|stability|ionic\s*radius|lattice|dielectric",
     ["material"]),
    (r"literature|paper|published|reported|mechanism|why\s+does|what\s+explains",
     ["literature"]),
    (r"backing\s+.*claim|claim\s+.*(source|back)|source\s+.*(backing|support)|"
     r"verify\s+.*claim",
     ["citations"]),
    (r"how\s+many\s+devices|how\s+many\s+records|what\s+is\s+the\s+average|count|top\s+\d|most\s+common|unique\s+materials",
     ["pdb"]),
    (r"predict|estimate|forecast|would\s+achieve|expected\s+PCE",
     ["ml"]),
    (r"correlation|trend|relationship\s+between",
     ["pdb"]),
    # two-tool patterns
    (r"why\s+does.*(average|count|performance|PCE).*correlat",
     ["pdb", "literature"]),
    (r"why\s+does.*(predict|expected|PCE)",
     ["ml", "literature"]),
    (r"(average|count|top).*(why|mechanism|explanation)",
     ["pdb", "literature"]),
    (r"does\s+(the\s+)?physics\s+(model|simulation).*correlat|explain.*data\s+trend",
     ["pdb", "scaps"]),
    (r"what\s+should\s+i\s+change|recommend|optimize|improve.*device",
     ["pdb", "material", "literature"]),
    (r"(predict|estimate).*(why|mechanism|literature)",
     ["ml", "literature"]),
    (r"(simulate|SCAPS).*(data|database|correlat|empirical)",
     ["pdb", "scaps"]),
    # 3+ tools
    (r"full\s+investigation|comprehensive|end-to-end|complete\s+analysis",
     ["pdb", "literature", "ml", "scaps"]),
    (r"why\s+does\s+absorber\s+thickness",
     ["pdb", "ml", "literature", "scaps"]),
    (r"what\s+HTL.*and\s+why",
     ["pdb", "material", "literature"]),
]

ACTION_MAP = {"literature": "search", "pdb": "query", "material": "lookup",
              "ml": "predict", "scaps": "get_sweep", "calculator": "eval",
              "fulltext": "fetch", "doi": "resolve", "citations": "verify",
              "symbolic": "fit", "extraction": "extract"}

# Default params per tool so a rule plan is always executable (a baseline plan,
# not a tuned one — the LLM planner produces the real query plans).
DEFAULT_PARAMS = {
    "pdb": {"operation": "global_aggregate",
            "aggregation": {"column": "PCE", "agg": "mean"}},
    "literature": {"k": 5},
    "calculator": {"action": "help"},
    "scaps": {"sweep_name": "pvk_Eg"},
    "ml": {"architecture": "overall"},
}

ML_ACTION_RULES = [
    (r"what\s+should\s+i\s+change|recommend|optimize|improve\w*\s+.*device|"
     r"boost\s+.*pce|best\s+parameter|optimal|optimum|should\s+i\s+use\b|"
     r"to\s+(maximize|reach|achieve)\b",
     "optimize"),
    (r"what\s+happens\s+if|how\s+sensitive|affect.*(thickness|PCE|mobility|property)|"
     r"sweep|vary\s+|change\w*\s+with\b|how\s+\w+\s+changes?\s+with\b",
     "what_if"),
    (r"this\b.{0,25}\b(device|sample|stack)|this\s+stack|"
     r"why\s+is\s+this\s+device|this\s+device.*predict",
     "explain_local"),
    (r"why|which (parameters|features).*(important|drive|matter|relevant|influence)"
     r"|most\s+(relevant|important)"
     r"|what\s+(drives|influences)\s+(pce|performance)"
     r"|explain", "explain_global"),
]


class RulePlanner:
    name = "rules"

    def plan(self, question: str) -> Plan:
        """Keyword-rule planning: first matching rule wins; falls back to a
        generic factual database query."""
        q = question.lower()
        mat = extract_material(q)
        plan = Plan(question=question, planner=self.name)
        for pattern, tools in RULES:
            if re.search(pattern, q):
                for t in tools:
                    params = dict(DEFAULT_PARAMS.get(t, {}))
                    action = ACTION_MAP.get(t, "")
                    if t == "material" and mat:
                        params = {"material": mat}
                    if t == "ml":
                        action = self._ml_action(q)
                        params["architecture"] = self._infer_arch(q)
                        if action in ("predict", "explain_local",
                                      "what_if", "optimize") \
                                and "description" not in params:
                            params["description"] = question[:200]
                        if action == "what_if":
                            param, values = self._whatif_spec(q)
                            params.setdefault("parameter", param)
                            params.setdefault("values", values)
                    if t == "pdb" and self._wants_count(q):
                        params = {"operation": "count"}
                    plan.steps.append(PlanStep(
                        tool=t, action=action,
                        params=params,
                        reason=f"rule: {pattern[:40]}",
                        purpose=f"rule baseline for '{t}': {pattern[:40]}",
                        expected_outputs=[]))
                return plan
        # default: factual fallback
        plan.steps.append(PlanStep(tool="pdb", action="query",
                                   params=dict(DEFAULT_PARAMS["pdb"]),
                                   reason="default rule (factual)",
                                   purpose="default database query",
                                   expected_outputs=["aggregate", "records"]))
        return plan

    @staticmethod
    def _ml_action(q: str) -> str:
        for pattern, action in ML_ACTION_RULES:
            if re.search(pattern, q):
                return action
        return "predict"

    @staticmethod
    def _whatif_spec(q: str):
        """(parameter, values-grid) for a rule-level what_if sweep, chosen
        from the ontology's known experimental knobs."""
        if "anneal" in q:
            return "Perov_anneal_temp_C", [80, 100, 120, 150, 180, 200]
        if "thick" in q:
            return "Perovskite_thickness_nm", [100, 200, 300, 400, 600, 800, 1000]
        if "band" in q and "gap" in q:
            return "Perovskite_band_gap", [1.4, 1.5, 1.6, 1.7, 1.8]
        if "mobility" in q:
            return "Perovskite_hole_mobility", [10, 30, 50, 70, 90]
        return "Perovskite_thickness_nm", [100, 200, 300, 400, 600, 800]

    @staticmethod
    def _infer_arch(q: str) -> str:
        """Architecture from the question text, else 'overall'."""
        if re.search(r"\bp-?i-?n\b|\bpin\b", q):
            return "pin"
        if re.search(r"\bn-?i-?p\b|\bnip\b", q):
            return "nip"
        return "overall"

    @staticmethod
    def _wants_count(q: str) -> bool:
        """Count intent: explicit 'count'/'how many' questions should not be
        answered with a mean-aggregate default."""
        return bool(re.search(r"\bcount\b|how\s+many", q))


# singleton convenience
def default_rule_planner() -> RulePlanner:
    return RulePlanner()


def extract_material(q: str) -> str:
    """Best-effort material/formula extraction for rule plans.

    Checks the molecule/formula alias table first (strong matches like 'tio2',
    'spiro', 'csmafa'), then falls back to tokenized matches against a small
    vocabulary. Returns '' when nothing resembles a material.
    """
    ql = q.lower()
    # deterministic lookup over canonical materials (longest names first)
    CANDIDATES = [
        "spiro-meotad", "me-4pacz", "pedot:pss", "csmafa", "mapbi3", "fapbi3",
        "csfapbi3", "fama", "perovskite", "sno2", "tio2", "nio", "zno", "c60",
        "pcbm", "p3ht", "ptaa", "cu2o", "cui", "sno2",
    ]
    for name in sorted(CANDIDATES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", ql):
            return {"spiro-meotad": "Spiro-MeOTAD", "pedot:pss": "PEDOT:PSS",
                    "csmafa": "CsMAFA", "mapbi3": "MAPbI3", "fapbi3": "FAPbI3",
                    "csfapbi3": "CsFAPbI3", "fama": "FA_MA", "sno2": "SnO2",
                    "tio2": "TiO2", "nio": "NiO", "zno": "ZnO", "c60": "C60",
                    "pcbm": "PCBM", "p3ht": "P3HT", "ptaa": "PTAA",
                    "cu2o": "Cu2O", "cui": "CuI"}.get(name, name.title())
    return ""