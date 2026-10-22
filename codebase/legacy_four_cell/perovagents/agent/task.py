"""Task-intent layer — a semantic specification of *what the user wants*,
before any tool is chosen.

Instead of letting the planner guess intent from keywords, the agent first
runs a **task analyzer** that converts the question into a `TaskSpec`:

    {
      "task_type": "scientific_interpretation" | "prediction" | ...,
      "architecture": "n-i-p" | ...,
      "objective": "...",
      "requested_output": [output tokens the answer must contain],
      "requires_prediction": bool,
      "requires_explanation": bool,
      "requires_intervention": bool,
      "requires_evidence": bool,
      "materials": [...],
    }

Task types follow the planner contract (factual_lookup, literature_retrieval,
database_query, prediction, interpretation, feature_importance,
sensitivity_analysis, optimization, comparison, verification, calculation,
simulation, multi_tool_investigation).

The analyzer is LLM-assisted when a key is present and falls back to the
deterministic keyword analyzer otherwise, so `analyze_task()` always returns.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..llm import LLM, extract_json

TASK_TYPES = [
    "factual_lookup", "literature_retrieval", "literature_extraction",
    "database_query", "prediction", "interpretation", "feature_importance",
    "sensitivity_analysis", "optimization", "comparison", "verification",
    "calculation", "simulation", "multi_tool_investigation",
]

ANALYZER_PROMPT = """You analyze the scientific intent of a perovskite solar
cell research question before any tool is chosen. Output ONLY a JSON object:

{{
  "task_type": one of {TYPES},
  "architecture": "p-i-n" | "n-i-p" | "overall" | null,
  "objective": "one sentence restating what evidence the user wants",
  "requested_output": ["output tokens the final answer must contain"],
  "requires_prediction": bool,
  "requires_explanation": bool,
  "requires_intervention": bool,
  "requires_sensitivity": bool,
  "requires_optimization": bool,
  "requires_evidence": bool,
  "materials": ["extracted material names or []"]
}}

Intent is decided BEFORE tools; the planner later maps intent to the correct
*scientific operation*. Distinguish these interpretation families:

- "What parameters are most relevant/important/influential?" /
  "which features matter" / "what drives PCE" -> task_type
  feature_importance, requires_explanation=true. The user does NOT need to
  say "SHAP", "feature importance" or "explain the model".
- "Why is THIS device predicted poorly" / "what is driving this device's
  prediction" -> interpretation, requires_explanation=true.
- "What happens if I increase/change X" / "how sensitive is PCE to X" ->
  sensitivity_analysis, requires_prediction=true, requires_sensitivity=true.
- "What should I change to improve the device" -> optimization,
  requires_prediction=true, requires_optimization=true.
- "Predict/estimate the PCE..." -> prediction, requires_prediction=true.
- "average/count/top/compare-most-common" -> database_query; do NOT set
  requires_prediction just because PCE is mentioned.
- "band gap / formation energy / density of X" -> factual_lookup.
- "why does / mechanism / literature says" -> literature_retrieval.

Feature importance is NOT prediction and is NOT intervention.
Use requested_output tokens from this vocabulary:
feature_name, importance, direction, prediction, interval, ood_status,
parameter_value, predicted_pce, count, aggregate, table, distribution,
columns, records, papers, passages, material_property, properties,
doi_metadata, citation_support, sweep, equation, calculation, exec_output,
architecture.

Question: {Q}
TaskSpec (JSON only):"""


@dataclass
class TaskSpec:
    task_type: str = "database_query"
    architecture: Optional[str] = None
    objective: str = ""
    requested_output: List[str] = field(default_factory=list)
    requires_prediction: bool = False
    requires_explanation: bool = False
    requires_intervention: bool = False
    requires_sensitivity: bool = False
    requires_optimization: bool = False
    requires_evidence: bool = True
    materials: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"task_type": self.task_type, "architecture": self.architecture,
                "objective": self.objective,
                "requested_output": list(self.requested_output),
                "requires_prediction": self.requires_prediction,
                "requires_explanation": self.requires_explanation,
                "requires_intervention": self.requires_intervention,
                "requires_sensitivity": self.requires_sensitivity,
                "requires_optimization": self.requires_optimization,
                "requires_evidence": self.requires_evidence,
                "materials": list(self.materials)}


# ---------------------------------------------------------------------------
# Deterministic fallback analyzer (no LLM)
# ---------------------------------------------------------------------------
_RULES = [
    (re.compile(r"\b(simulate|SCAPS|physics model|parametric sweep)\b", re.I),
     "simulation"),
    (re.compile(r"\b(optimize|what should i change|recommend|improve .*device|"
                r"best parameter|suggest.*(change|parameter)|should i use|"
                r"to (maximize|reach|achieve))\b", re.I),
     "optimization"),
    (re.compile(r"\bwhat if|how sensitive|if .* (increases|decreases|changes)|"
                r"sweep|vary|thickness affect|change\w* with|how \w+ changes? "
                r"with\b", re.I),
     "sensitivity_analysis"),
    (re.compile(r"\bthis\b.{0,30}\b(device|sample|stack)\b", re.I),
     "interpretation"),
    (re.compile(r"\b(which|what)\b.{0,40}?(parameters?|features?|inputs?).{0,60}\b"
                r"(important|influential|relevant|matter|drives?|contribute|correlat)"
                r"|most\s+(relevant|important)\s+(parameters?|features?)"
                r"|what\s+(drives|influences)\s+(the\s+)?(pce|performance)\b", re.I),
     "feature_importance"),
    (re.compile(r"\b(explain|why|interpret|interpretability|understand)\b", re.I),
     "interpretation"),
    (re.compile(r"\b(predict|estimate|forecast|expected pce|would (a |an )?.* "
                r"(pce|perform))|what pce\b", re.I), "prediction"),
    (re.compile(r"\b(calculate|convert|compute|what is .* (in )?(eV|nm|cm|mv))\b",
                re.I), "calculation"),
    (re.compile(r"\b(band gap|formation energy|density|dielectric|mobility|"
                r"work function|homo|lumo|stability of)\b", re.I),
     "factual_lookup"),
    (re.compile(r"\b(compare|vs\.?|versus|difference between)\b", re.I),
     "comparison"),
    (re.compile(r"\b(verify|validate|check.*(claim|citation|source)|is it true|"
                r"backing.*claim|claim.*(source|back)|source.*(backing|support))\b",
                re.I), "verification"),
    (re.compile(r"\b(extract|table .*paper|pull out)\b", re.I),
     "literature_extraction"),
    (re.compile(r"\b(literature|reported|paper|mechanism|why does|what explains"
                r"|say about)\b", re.I), "literature_retrieval"),
    (re.compile(r"\b(how many|count|top \d|average|most common|unique|"
                r"mean|median)\b", re.I), "database_query"),
]

# task_type -> which output tokens a complete answer needs
_TASK_OUTPUTS = {
    "prediction": ["prediction", "interval"],
    "feature_importance": ["feature_name", "importance", "direction"],
    "sensitivity_analysis": ["parameter_value", "predicted_pce"],
    "optimization": ["parameter_value", "predicted_pce", "interval"],
    "database_query": ["records", "papers", "aggregate"],
    "comparison": ["aggregate", "records"],
    "factual_lookup": ["material_property"],
    "literature_retrieval": ["passages"],
    "literature_extraction": ["table", "columns"],
    "verification": ["citation_support"],
    "calculation": ["calculation"],
    "simulation": ["sweep"],
    "interpretation": ["feature_name", "importance", "direction"],
    "multi_tool_investigation": ["aggregate", "records", "passages"],
}

_REQ_FLAGS = {
    "prediction": {"requires_prediction": True},
    "feature_importance": {"requires_explanation": True},
    "interpretation": {"requires_explanation": True},
    "sensitivity_analysis": {"requires_prediction": True,
                             "requires_sensitivity": True},
    "optimization": {"requires_prediction": True,
                     "requires_optimization": True,
                     "requires_explanation": True,
                     "requires_intervention": True},
}

_VOCAB = {"feature_name", "importance", "direction", "prediction", "interval",
          "ood_status", "parameter_value", "predicted_pce", "count", "aggregate",
          "table", "distribution", "columns", "records", "papers", "passages",
          "material_property", "properties", "doi_metadata", "citation_support",
          "sweep", "equation", "calculation", "exec_output", "architecture"}
_INTERVENTION_TOKENS = {"parameter_value", "predicted_pce"}
_EXPLAIN_TOKENS = {"feature_name", "importance", "direction"}
_PREDICT_TOKENS = {"prediction", "interval"}


def _keyword_spec(question: str) -> TaskSpec:
    import re as _re
    q = _re.sub(r"\s+", " ", question).strip()
    for pattern, ttype in _RULES:
        if pattern.search(q):
            arch = None
            if _re.search(r"\bp-?i-?n\b", q):
                arch = "p-i-n" if "p-i-n" in q else ("pin" if "pin" in q else None)
            elif _re.search(r"\bn-?i-?p\b", q):
                arch = "n-i-p" if "n-i-p" in q else ("nip" if "nip" in q else None)
            outputs = list(_TASK_OUTPUTS.get(ttype, ["records"]))
            flags = dict(_REQ_FLAGS.get(ttype, {}))
            mats = extract_materials(q)
            return TaskSpec(task_type=ttype, architecture=arch,
                            objective=f"answer '{question[:80]}'",
                            requested_output=outputs, materials=mats, **flags)
    return TaskSpec(task_type="database_query", objective=question[:120],
                    requested_output=["records", "aggregate"], materials=extract_materials(q))


def extract_materials(question: str) -> List[str]:
    """Canonical material names mentioned in a question (keyword match)."""
    ql = question.lower()
    CANON = {"sno2": "SnO2", "tio2": "TiO2", "zno": "ZnO", "nio": "NiO",
             "ptaa": "PTAA", "spiro": "Spiro-MeOTAD", "spiro-meotad": "Spiro-MeOTAD",
             "pedot:pss": "PEDOT:PSS", "p3ht": "P3HT", "mapbi3": "MAPbI3",
             "fapbi3": "FAPbI3", "csmafa": "CsMAFA", "pcbm": "PCBM", "c60": "C60",
             "cui": "CuI", "cu2o": "Cu2O"}
    out = [c for k, c in CANON.items() if k in ql]
    return out


def analyze_task(question: str, llm: Optional[LLM] = None,
                 use_llm: bool = True) -> TaskSpec:
    """Return a TaskSpec for `question`. LLM-assisted when available.

    `use_llm=False` forces the deterministic keyword analyzer (0 cost, used
    by benchmark rule-pipeline runs and offline CI)."""
    if use_llm and llm is not None and llm.usable:
        prompt = (ANALYZER_PROMPT.replace("{TYPES}", json.dumps(TASK_TYPES))
                                .replace("{Q}", question))
        parsed = extract_json(llm.chat(prompt, temperature=0.0))
        if isinstance(parsed, dict):
            ttype = str(parsed.get("task_type") or "database_query").lower()
            if ttype not in TASK_TYPES:
                ttype = "database_query"
            ro = parsed.get("requested_output")
            outputs = _sanitize_outputs(ro, parsed) \
                if isinstance(ro, list) else list(_TASK_OUTPUTS.get(ttype, ["records"]))
            return TaskSpec(
                task_type=ttype,
                architecture=str(parsed["architecture"]) if parsed.get("architecture") else None,
                objective=str(parsed.get("objective") or "")[:200],
                requested_output=outputs,
                requires_prediction=bool(parsed.get("requires_prediction", False)),
                requires_explanation=bool(parsed.get("requires_explanation", False)),
                requires_intervention=bool(parsed.get("requires_intervention", False)),
                requires_sensitivity=bool(parsed.get("requires_sensitivity", False)),
                requires_optimization=bool(parsed.get("requires_optimization", False)),
                requires_evidence=bool(parsed.get("requires_evidence", True)),
                materials=[str(m) for m in parsed.get("materials") or []])
    return _keyword_spec(question)


def _sanitize_outputs(ro, parsed) -> List[str]:
    """Drop tokens the analyzer's own flags contradict (e.g. parameter_value
    on a question it marked requires_intervention=false)."""
    out = []
    for t in [str(x) for x in ro]:
        if t not in _VOCAB:
            continue
        if t in _INTERVENTION_TOKENS and not parsed.get("requires_intervention", False):
            continue
        if t in _EXPLAIN_TOKENS and not parsed.get("requires_explanation", False):
            continue
        if t in _PREDICT_TOKENS and not parsed.get("requires_prediction", False):
            continue
        if t not in out:
            out.append(t)
    return out or list(_TASK_OUTPUTS.get(str(parsed.get("task_type") or ""),
                                         ["records"]))