"""Cross-LLM benchmark: fixed question set for the MRS paper.

Research question:
    DOES THE MRS AGENTIC ARCHITECTURE IMPROVE SCIENTIFIC TASK EXECUTION
    OVER DIRECT LLM TOOL USE, WHEN THE UNDERLYING SCIENTIFIC TOOLS ARE
    HELD CONSTANT?

Fairness principle (enforced by the runner, not by trust):
    SAME QUESTIONS, SAME TOOLS, SAME SCIENTIFIC DATA, SAME TOOL OUTPUTS,
    FIXED CALL/TOKEN BUDGET, DIFFERENT ORCHESTRATION.

The point is NOT "model X is worse than MRS". The point is whether
structured agentic orchestration improves scientific task execution
given comparable tool access.

Status note: provider rollout is stepwise. Only Gemini is wired for now
(Direct Gemini vs MRS-on-Gemini); Claude/GPT adapters are stubbed for a
later step. Question texts below are provider-independent and frozen.

20 fixed questions (Q01..Q20), each with stable metadata. Question text is
part of the benchmark artifact and must remain unchanged. Ground truth
lives separately in benchmarks/cross_llm_gold.json (built by trusted
tools, never by an LLM, never fabricated).
"""

QUESTIONS = [
    # ---------------------------------------------------------------- PDB
    {
        "id": "Q01",
        "category": "pdb",
        "question": (
            "What is the average reported PCE of n-i-p perovskite solar cells "
            "using SnO2 as the ETL?"
        ),
        "difficulty": "normal",
        "expected_task_type": "database_aggregation",
        "expected_tools": ["pdb.sql"],
        "expected_actions": ["database_query"],
        "required_outputs": ["count", "mean_pce"],
        "forbidden_substitutions": ["raw_rows", "single_record_lookup", "ml_predict"],
        "recovery_required": False,
    },
    {
        "id": "Q02",
        "category": "pdb",
        "question": (
            "What are the 10 most frequently reported HTL materials in n-i-p "
            "perovskite solar cells with PCE above 20%?"
        ),
        "difficulty": "adversarial",
        "expected_task_type": "database_aggregation",
        "expected_tools": ["pdb.sql"],
        "expected_actions": ["database_query"],
        "required_outputs": ["top_k", "counts"],
        "forbidden_substitutions": ["raw_rows", "single_record_lookup"],
        "recovery_required": True,
    },
    {
        "id": "Q03",
        "category": "pdb",
        "question": (
            "Compare the reported PCE distributions of p-i-n and n-i-p "
            "perovskite solar cells in the database. Report the number of "
            "records and the mean and median PCE for each architecture."
        ),
        "difficulty": "normal",
        "expected_task_type": "database_comparison",
        "expected_tools": ["pdb.sql"],
        "expected_actions": ["database_query"],
        "required_outputs": ["count_per_group", "mean_pce_per_group",
                             "median_pce_per_group"],
        "forbidden_substitutions": ["single_architecture_only", "ml_predict"],
        "recovery_required": False,
    },
    {
        "id": "Q04",
        "category": "pdb",
        "question": (
            "Among n-i-p perovskite solar cells using SnO2 as the ETL, identify "
            "the highest-PCE reported devices and report their HTL, "
            "absorber/material stack, PCE, Voc, Jsc and FF."
        ),
        "difficulty": "normal",
        "expected_task_type": "database_ranked_lookup",
        "expected_tools": ["pdb.sql"],
        "expected_actions": ["database_query"],
        "required_outputs": ["ranked_records", "requested_fields"],
        "forbidden_substitutions": ["ml_predict_as_database_value",
                                    "unranked_dump"],
        "recovery_required": False,
    },
    # ------------------------------------------------------- LITERATURE
    {
        "id": "Q05",
        "category": "literature",
        "question": (
            "Find literature evidence describing strategies for improving the "
            "performance of SnO2-based n-i-p perovskite solar cells. Give the "
            "strategy, mechanism, and supporting citation."
        ),
        "difficulty": "normal",
        "expected_task_type": "literature_retrieval",
        "expected_tools": ["literature.search", "literature.read_paper"],
        "expected_actions": ["literature_search", "evidence_verify"],
        "required_outputs": ["strategy", "mechanism", "citation"],
        "forbidden_substitutions": ["pin_paper_for_nip_claim", "uncited_claim"],
        "recovery_required": False,
    },
    {
        "id": "Q06",
        "category": "literature",
        "question": (
            "Find a paper supporting an experimentally demonstrated method to "
            "improve SnO2-based n-i-p perovskite solar-cell performance. Verify "
            "that the paper actually studies an n-i-p SnO2 device before using "
            "it as evidence."
        ),
        "difficulty": "adversarial",
        "expected_task_type": "literature_retrieval_with_recovery",
        "expected_tools": ["literature.search", "literature.read_paper"],
        "expected_actions": ["literature_search", "evidence_verify",
                             "reject_and_research"],
        "required_outputs": ["verified_citation", "architecture_check"],
        "forbidden_substitutions": ["unverified_first_hit",
                                    "reread_rejected_paper"],
        "recovery_required": True,
    },
    {
        "id": "Q07",
        "category": "literature",
        "question": (
            "Find a relevant paper reporting a quantitative performance "
            "improvement for an SnO2-based n-i-p perovskite solar cell. Report "
            "the baseline/result value, the intervention, the quantitative "
            "improvement, and the DOI."
        ),
        "difficulty": "normal",
        "expected_task_type": "literature_numerical_extraction",
        "expected_tools": ["literature.search", "literature.read_paper"],
        "expected_actions": ["literature_search", "evidence_verify"],
        "required_outputs": ["baseline_value", "result_value", "intervention",
                             "doi"],
        "forbidden_substitutions": ["invented_doi", "invented_value",
                                    "qualitative_only"],
        "recovery_required": False,
    },
    {
        "id": "Q08",
        "category": "literature",
        "question": (
            "Find literature evidence for improving an SnO2 n-i-p device, "
            "including a numerical value for the relevant improvement. Do not "
            "cite a paper as supporting the claim unless the required "
            "numerical evidence can actually be located in the paper."
        ),
        "difficulty": "adversarial",
        "expected_task_type": "literature_retrieval_with_recovery",
        "expected_tools": ["literature.search", "literature.read_paper"],
        "expected_actions": ["literature_search", "evidence_verify",
                             "reject_and_research"],
        "required_outputs": ["numerical_evidence", "verified_citation"],
        "forbidden_substitutions": ["citation_without_number",
                                    "reread_rejected_paper"],
        "recovery_required": True,
    },
    # ---------------------------------------------------------------- ML
    {
        "id": "Q09",
        "category": "ml",
        "question": (
            "What PCE does the ML model predict for an n-i-p cell with SnO2 "
            "ETL, Spiro-MeOTAD HTL and MAPbI3 perovskite 500 nm? Report the "
            "prediction with its 95% interval."
        ),
        "difficulty": "normal",
        "expected_task_type": "prediction",
        "expected_tools": ["ml.predict"],
        "expected_actions": ["ml.predict"],
        "required_outputs": ["predicted_pce", "interval_95"],
        "forbidden_substitutions": ["database_aggregate", "ml.explain_global"],
        "recovery_required": False,
    },
    {
        "id": "Q10",
        "category": "ml",
        "question": (
            "What are the most important parameters for improving PCE in n-i-p "
            "perovskite solar cells according to the ML model?"
        ),
        "difficulty": "normal",
        "expected_task_type": "feature_importance",
        "expected_tools": ["ml.explain_global"],
        "expected_actions": ["ml.explain_global"],
        "required_outputs": ["feature_name", "importance"],
        "forbidden_substitutions": ["ml.predict", "database_aggregate",
                                    "simple_correlation"],
        "recovery_required": False,
    },
    {
        "id": "Q11",
        "category": "ml",
        "question": (
            "For an n-i-p cell with SnO2 ETL, Spiro-MeOTAD HTL and MAPbI3 "
            "perovskite 500 nm, why did the ML model give its predicted PCE? "
            "Report the prediction, the major contributing features and their "
            "direction of contribution."
        ),
        "difficulty": "normal",
        "expected_task_type": "local_explanation",
        "expected_tools": ["ml.predict", "ml.explain_local"],
        "expected_actions": ["ml.predict", "ml.explain_local"],
        "required_outputs": ["predicted_pce", "contributing_features",
                             "contribution_direction"],
        "forbidden_substitutions": ["global_shap_only"],
        "recovery_required": False,
    },
    {
        "id": "Q12",
        "category": "ml",
        "question": (
            "For an n-i-p cell with SnO2 ETL, Spiro-MeOTAD HTL and MAPbI3 "
            "perovskite 500 nm, what controllable parameters can be changed "
            "to improve the predicted PCE, and what does the ML model predict "
            "will happen? Give the baseline, candidate changes and predicted "
            "PCEs."
        ),
        "difficulty": "normal",
        "expected_task_type": "optimization",
        "expected_tools": ["ml.what_if", "ml.optimize"],
        "expected_actions": ["ml.what_if", "ml.optimize"],
        "required_outputs": ["baseline", "candidate_change", "predicted_pce",
                             "optimization_status"],
        "forbidden_substitutions": ["shap_importance_as_causal"],
        "recovery_required": False,
    },
    # -------------------------------------------------------------- SCAPS
    {
        "id": "Q13",
        "category": "scaps",
        "question": (
            "Design and run a p-i-n cell with a SnO2 ETL, a PTAA HTL and a "
            "600 nm absorber (FTO / PTAA / FAPbI3 / SnO2 / Ag; PTAA 200 nm, "
            "absorber 600 nm, SnO2 50 nm), then report Voc, Jsc, FF and PCE."
        ),
        "difficulty": "normal",
        "expected_task_type": "device_simulation",
        "expected_tools": ["scaps.build", "scaps.run"],
        "expected_actions": ["device_construct", "simulate", "validate"],
        "required_outputs": ["voc", "jsc", "ff", "pce", "convergence_status"],
        "forbidden_substitutions": ["hardcoded_metrics", "ml_predict_as_sim"],
        "recovery_required": False,
    },
    {
        "id": "Q14",
        "category": "scaps",
        "question": (
            "For the p-i-n device with SnO2 ETL, PTAA HTL and 600 nm absorber "
            "from the previous design, determine how the PCE changes when the "
            "perovskite absorber thickness is varied over a physically "
            "reasonable range."
        ),
        "difficulty": "normal",
        "expected_task_type": "parameter_sweep",
        "expected_tools": ["scaps.sweep"],
        "expected_actions": ["device_construct", "sweep", "trend_interpret"],
        "required_outputs": ["sweep_values", "pce_trend"],
        "forbidden_substitutions": ["single_point_only", "ml_trend_as_sim"],
        "recovery_required": False,
    },
    {
        "id": "Q15",
        "category": "scaps",
        "question": (
            "Run a SCAPS simulation for the requested p-i-n device (SnO2 ETL, "
            "PTAA HTL, 600 nm absorber) and recover if the initial "
            "configuration does not converge. Report the final validated "
            "result and what was changed to reach convergence."
        ),
        "difficulty": "adversarial",
        "expected_task_type": "simulation_with_recovery",
        "expected_tools": ["scaps.build", "scaps.run"],
        "expected_actions": ["device_construct", "simulate", "recover_rebuild",
                             "validate"],
        "required_outputs": ["final_metrics", "convergence_status",
                             "recovery_action"],
        "forbidden_substitutions": ["repeat_identical_run",
                                    "report_nonconverged_as_valid"],
        "recovery_required": True,
    },
    {
        "id": "Q16",
        "category": "scaps",
        "question": (
            "For the simulated p-i-n device (SnO2 ETL, PTAA HTL, 600 nm "
            "absorber), identify the parameters that most strongly affect the "
            "simulated PCE and explain what component of the device "
            "performance they influence."
        ),
        "difficulty": "normal",
        "expected_task_type": "simulation_interpretation",
        "expected_tools": ["scaps.sweep"],
        "expected_actions": ["sweep", "trend_interpret"],
        "required_outputs": ["sensitive_parameters", "affected_component"],
        "forbidden_substitutions": ["ml_shap_as_scaps", "invented_causality"],
        "recovery_required": False,
    },
    # --------------------------------------------------------- MULTI-TOOL
    {
        "id": "Q17",
        "category": "multi_tool",
        "question": (
            "I am developing an SnO2 n-i-p perovskite solar cell. What can I "
            "do to improve the PCE, and what does the literature say about "
            "improving these devices?"
        ),
        "difficulty": "multi_tool",
        "expected_task_type": "ml_plus_literature_synthesis",
        "expected_tools": ["ml.optimize", "ml.explain_global",
                           "literature.search", "literature.read_paper"],
        "expected_actions": ["ml.optimize", "literature_search",
                             "evidence_verify", "synthesize"],
        "required_outputs": ["model_recommendations", "literature_evidence",
                             "provenance_split"],
        "forbidden_substitutions": ["model_claim_cited_as_literature",
                                    "literature_claim_cited_as_model"],
        "recovery_required": False,
    },
    {
        "id": "Q18",
        "category": "multi_tool",
        "question": (
            "For n-i-p devices using SnO2 as the ETL, first identify "
            "representative high-performing devices in the database, then "
            "estimate the PCE of an n-i-p SnO2 / MAPbI3 500 nm / Spiro-MeOTAD "
            "target device with the ML model and explain which parameters "
            "most affect the prediction."
        ),
        "difficulty": "multi_tool",
        "expected_task_type": "database_then_ml",
        "expected_tools": ["pdb.sql", "ml.predict", "ml.explain_local"],
        "expected_actions": ["database_query", "ml.predict",
                             "ml.explain_local", "synthesize"],
        "required_outputs": ["representative_devices", "predicted_pce",
                             "key_parameters"],
        "forbidden_substitutions": ["database_mean_as_ml_prediction"],
        "recovery_required": False,
    },
    {
        "id": "Q19",
        "category": "multi_tool",
        "question": (
            "I am considering a new n-i-p device using a candidate ETL and "
            "HTL. Find relevant literature evidence, retrieve the relevant "
            "material properties, estimate the expected PCE, and identify the "
            "parameters that most strongly affect the prediction."
        ),
        "difficulty": "multi_tool",
        "expected_task_type": "literature_property_ml_synthesis",
        "expected_tools": ["literature.search", "ml.predict",
                           "ml.explain_local"],
        "expected_actions": ["literature_search", "evidence_verify",
                             "ml.predict", "ml.explain_local", "synthesize"],
        "required_outputs": ["literature_evidence", "material_properties",
                             "predicted_pce", "key_parameters"],
        "forbidden_substitutions": ["unprovenanced_synthesis"],
        "recovery_required": False,
    },
    {
        "id": "Q20",
        "category": "multi_tool",
        "question": (
            "Design and evaluate a p-i-n perovskite solar cell using PTAA and "
            "SnO2 with a 600 nm absorber. Use the available scientific tools "
            "to construct the device, simulate it, report its performance, "
            "and explain the most important factors affecting the result. "
            "Where appropriate, support the interpretation using literature "
            "evidence."
        ),
        "difficulty": "multi_tool",
        "expected_task_type": "full_workflow",
        "expected_tools": ["scaps.build", "scaps.run", "ml.explain_global",
                           "literature.search", "literature.read_paper"],
        "expected_actions": ["device_construct", "simulate", "validate",
                             "ml.explain_global", "literature_search",
                             "synthesize"],
        "required_outputs": ["device_spec", "simulated_metrics",
                             "key_factors", "supporting_evidence"],
        "forbidden_substitutions": ["single_tool_answer"],
        "recovery_required": True,
    },
]

# Recovery-sensitive subset (spec section 11): initial failure expected or
# explicitly probed; runner records failure type, recovery action, cost.
RECOVERY_SENSITIVE = {"Q02", "Q06", "Q08", "Q15", "Q20"}

# Pilot subset (spec section 20): run first across systems, inspect traces
# manually, confirm tool interfaces / fairness / token use / golds, and only
# then run the full 20 x systems grid.
PILOT = ["Q01", "Q02", "Q06", "Q10", "Q13", "Q17", "Q20"]

CATEGORIES = ("pdb", "literature", "ml", "scaps", "multi_tool")


def get_question(qid: str) -> dict:
    """Stable lookup by ID (Q01..Q20); raises KeyError on unknown ID."""
    for q in QUESTIONS:
        if q["id"] == qid:
            return q
    raise KeyError(f"unknown benchmark question {qid!r}")


def by_category(category: str) -> list:
    """All questions in one category, in benchmark order."""
    return [q for q in QUESTIONS if q["category"] == category]
