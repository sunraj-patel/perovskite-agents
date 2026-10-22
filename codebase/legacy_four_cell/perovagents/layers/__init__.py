"""Aggregates all PerovAgents layers and registers their tools.

Layer structure (mirrors the tool ecosystem table):

  Orchestration : calculator, python_exec      (+ registry/planner infra)
  Literature    : literature, fulltext, extraction, citations, doi
  Materials     : pdb, material, materials_project
  ML            : ml
  Physics       : scaps, symbolic
  Verification  : the SanityChecker is applied by the agent, not a tool
  Provenance    : the ProvenanceStore is applied by the agent, not a tool

Every tool registers per-*action* schemas (not a flat tool schema):

    tool
     ├── action 1 ── input schema / task_types / outputs / produces /
     │               compatible_with / requires / constraints
     ├── action 2 ...
     └── ...

The planner plans against these actions, so it has semantic information (what
a step will produce, which task types it satisfies) instead of a flat namespace.
`register_all(registry)` is the single entry point the framework uses to make
every capability available to the planner.
"""
from ..core.registry import ActionSpec
from ..core.registry import ToolRegistry

A = ActionSpec

ML_OUTPUTS = ["prediction", "interval", "ood_status", "architecture"]


def register_all(registry: ToolRegistry) -> ToolRegistry:
    """Register every tool class into `registry` (lazy factories)."""
    ARC = {"type": "string", "enum": ["p-i-n", "n-i-p", "overall"]}

    ACTIONS = {
        # --- Orchestration layer -------------------------------------------
        "calculator": {
            "eval": A("eval", task_types=["calculation"],
                      params={"action": {"type": "string", "enum": ["convert", "pce", "eval"]},
                              "value": {"type": "number"}, "from": {"type": "string"},
                              "to": {"type": "string"}, "voc": {"type": "number"},
                              "jsc": {"type": "number"}, "ff": {"type": "number"},
                              "expression": {"type": "string"}},
                      outputs=["calculation"], produces=["calculation"]),
        },
        "python_exec": {
            "exec": A("exec", task_types=["calculation"],
                      params={"code": {"type": "string", "required": True},
                              "timeout": {"type": "number"}},
                      outputs=["exec_output"], produces=["exec_output"]),
        },
        # --- Literature layer ----------------------------------------------
        "literature": {
            "search": A("search", task_types=["literature_retrieval", "interpretation"],
                        params={"query": {"type": "string", "required": True},
                                "k": {"type": "integer"}},
                        outputs=["passages"], produces=["passages", "doi_metadata"]),
        },
        "fulltext": {
            "fetch": A("fetch", task_types=["literature_retrieval", "literature_extraction"],
                       params={"doi": {"type": "string", "required": True}},
                       outputs=["passages"], produces=["passages"]),
        },
        "extraction": {
            "extract": A("extract", task_types=["literature_extraction"],
                         params={"doi": {"type": "string", "required": True},
                                 "kind": {"type": "string", "enum": ["table", "figure"]}},
                         outputs=["columns", "table"], produces=["columns"]),
        },
        "citations": {
            "verify": A("verify", task_types=["verification"],
                        params={"claim": {"type": "string", "required": True},
                                "doi": {"type": "string", "required": True}},
                        outputs=["citation_support"], produces=["citation_support"]),
        },
        "doi": {
            "resolve": A("resolve", task_types=["factual_lookup"],
                         params={"doi": {"type": "string", "required": True}},
                         outputs=["doi_metadata"], produces=["doi_metadata"]),
        },
        # --- Materials layer -----------------------------------------------
        "pdb": {
            "query": A("query", task_types=["database_query", "factual_lookup",
                                            "comparison"],
                       params={"operation": {"type": "string",
                                             "enum": ["count", "filter", "top_k",
                                                      "groupby_aggregate",
                                                      "global_aggregate", "unique",
                                                      "distribution", "sample",
                                                      "describe_columns"],
                                             "required": True},
                               "filters": {"type": "array"}, "group_by": {"type": "string"},
                               "aggregation": {"type": "object"},
                               "sort_by": {"type": "string"}, "ascending": {"type": "boolean"},
                               "k": {"type": "integer"}, "limit": {"type": "integer"},
                               "select_columns": {"type": "array"},
                               "column": {"type": "string"}},
                       outputs=["records", "papers", "count", "aggregate", "table",
                                "distribution", "columns"],
                       produces=["records"], compatible_with=["ml.predict", "material.lookup"]),
        },
        "material": {
            "lookup": A("lookup", task_types=["factual_lookup", "prediction"],
                        params={"material": {"type": "string", "required": True},
                                "property": {"type": "string",
                                             "enum": ["band_gap", "LUMO", "HOMO",
                                                      "electron_mobility",
                                                      "hole_mobility",
                                                      "dielectric_constant",
                                                      "min_radius", "max_radius",
                                                      "avg_radius",
                                                      "tolerance_factor",
                                                      "octahedral_factor",
                                                      "tau_descriptor"]}},
                        outputs=["material_property"], produces=["material_property"],
                        compatible_with=["ml.predict", "ml.what_if"]),
        },
        "materials_project": {
            "lookup": A("lookup", task_types=["factual_lookup"],
                        params={"formula": {"type": "string", "required": True}},
                        outputs=["material_property", "properties"],
                        produces=["material_property"]),
        },
        # --- ML layer (discrete actions, no cross-action flags) ------------
        "ml": {
            "predict": A("predict",
                         task_types=["prediction", "estimation", "database_query"],
                         params={"_strict": True,
                                 "description": {"type": "string", "required": True},
                                 "architecture": ARC,
                                 "features": {"type": "object"},
                                 "uncertainty": {"type": "boolean"}},
                         outputs=list(ML_OUTPUTS), produces=["prediction"],
                         compatible_with=["material.lookup", "pdb.query"],
                         requires=["trained_model"],
                         constraints=["description must describe a device and its layers"]),
            "explain_global": A("explain_global",
                                task_types=["feature_importance", "interpretation"],
                                params={"_strict": True,
                                        "architecture": ARC,
                                        "method": {"type": "string", "enum": ["shap"]},
                                        "top_k": {"type": "integer"}},
                                outputs=["feature_name", "importance", "direction"],
                                produces=["feature_name", "importance", "direction"],
                                compatible_with=["ml.what_if", "pdb.query", "ml.optimize"],
                                requires=["trained_model"],
                                constraints=["returns model associations, not causal "
                                             "evidence; global across architectures"]),
            "explain_local": A("explain_local",
                               task_types=["feature_importance", "interpretation",
                                           "explanation"],
                               params={"_strict": True,
                                       "description": {"type": "string", "required": True},
                                       "architecture": ARC,
                                       "method": {"type": "string", "enum": ["shap"]},
                                       "top_k": {"type": "integer"}},
                               outputs=["feature_name", "importance", "direction"],
                               produces=["feature_name", "importance", "direction"],
                               compatible_with=["ml.what_if", "ml.optimize"],
                               requires=["trained_model"],
                               constraints=["SHAP associations are not causal"]),
            "what_if": A("what_if",
                         task_types=["sensitivity_analysis", "intervention",
                                     "simulation"],
                         params={"_strict": True,
                                 "description": {"type": "string", "required": True},
                                 "architecture": ARC,
                                 "parameter": {"type": "string", "required": True},
                                 "values": {"type": "array", "required": True}},
                         outputs=["parameter_value", "predicted_pce", "uncertainty",
                                  "ood_status"],
                         produces=["parameter_value", "predicted_pce"],
                         compatible_with=["pdb.query"],
                         requires=["trained_model", "controllable_feature"],
                         constraints=["parameter must be a controllable input "
                                      "feature (not measurement-derived)"]),
            "optimize": A("optimize",
                          task_types=["optimization", "recommendation",
                                      "intervention"],
                          params={"_strict": True,
                                  "description": {"type": "string"},
                                  "architecture": ARC,
                                  "top_k": {"type": "integer"}},
                          outputs=["parameter_value", "predicted_pce", "uncertainty"],
                          produces=["parameter_value", "predicted_pce"],
                          compatible_with=["pdb.query"],
                          requires=["trained_model", "controllable_feature"],
                          constraints=["recommendations are model-derived, not "
                                       "causal or experimental"]),
        },
        # --- Physics layer --------------------------------------------------
        "scaps": {
            "get_sweep": A("get_sweep", task_types=["simulation", "sensitivity_analysis"],
                           params={"sweep_name": {"type": "string", "enum": ["eg", "d", "mue",
                                                          "muh", "htlchi", "htlEg"]},
                                   "parameter": {"type": "string"},
                                   "values": {"type": "array"}},
                           outputs=["sweep"], produces=["sweep"],
                           compatible_with=["pdb.query"],
                           constraints=["SCAPS results are model sensitivities, "
                                        "not experimental reproductions"]),
        },
        "symbolic": {
            "fit": A("fit", task_types=["calculation"],
                     params={"x": {"type": "array"}, "y": {"type": "array"},
                             "kind": {"type": "string", "enum": ["linear", "power",
                                                                 "exponential"]}},
                     outputs=["equation"], produces=["equation"]),
        },
    }

    # --- Orchestration layer -----------------------------------------------
    from ..core.calculator import CalculatorTool
    from ..core.python_exec import PythonExecTool
    registry.register(CalculatorTool, actions=ACTIONS["calculator"])
    registry.register(PythonExecTool, actions=ACTIONS["python_exec"])

    # --- Literature layer --------------------------------------------------
    from ..literature.search import LiteratureSearchTool
    from ..literature.fulltext import FullTextTool
    from ..literature.extraction import TableFigureExtractor
    from ..literature.citations import CitationVerifier
    from ..literature.doi import DOIMetadataTool
    registry.register(LiteratureSearchTool, actions=ACTIONS["literature"])
    registry.register(FullTextTool, actions=ACTIONS["fulltext"])
    registry.register(TableFigureExtractor, actions=ACTIONS["extraction"])
    registry.register(CitationVerifier, actions=ACTIONS["citations"])
    registry.register(DOIMetadataTool, actions=ACTIONS["doi"])

    # --- Materials layer ---------------------------------------------------
    from ..materials.pdb import PDBTool
    from ..materials.properties import MaterialPropertyTool
    from ..materials.materials_project import MaterialsProjectTool
    registry.register(PDBTool, actions=ACTIONS["pdb"])
    registry.register(MaterialPropertyTool, actions=ACTIONS["material"])
    registry.register(MaterialsProjectTool, actions=ACTIONS["materials_project"])

    # --- ML layer ----------------------------------------------------------
    from ..ml.predictor import MLPredictorTool
    registry.register(MLPredictorTool, actions=ACTIONS["ml"])

    # --- Physics layer -----------------------------------------------------
    from ..physics.scaps import SCAPSTool
    from ..physics.symbolic import EquationFitTool
    registry.register(SCAPSTool, actions=ACTIONS["scaps"])
    registry.register(EquationFitTool, actions=ACTIONS["symbolic"])

    return registry