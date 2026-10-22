"""Planner-failure / scientific-intent benchmark question sets.

Three sets, each question carrying a ground-truth plan (a list of *acceptable
action sequences*; a plan scores a hit when one GT sequence appears, in order,
among its required-for-final-answer steps):

  SIMPLE   10 categories x 8-12 questions   -> planner failures under normal
           INTENT_ROUTE questions
  AMBIGUOUS  stiff scientific phrasing that tempts class rewrite
           ("What improves n-i-p performance?", "What matters most?",
            "Which properties should I focus on?", "How can I increase
            efficiency?")
  DISAMBIGUATION  paired questions differing by a few words but needing a
           different answer class (prediction / global interpretation / local
           interpretation / sensitivity / optimization); 5 topics x 3 variants

GT sequences use "tool.action" strings. Multiple accepted sequences encode
legitimately different but science-equivalent plans.
"""
from typing import List, Dict

SIMPLE: List[Dict] = [
    # ---- prediction ------------------------------------------------------
    {"qid": "pred_01", "category": "prediction",
     "question": "What PCE would a p-i-n device with FA0.8MA0.2PbI3 perovskite "
                 "and 400 nm thickness achieve?",
     "gt": [["ml.predict"]]},
    {"qid": "pred_02", "category": "prediction",
     "question": "Predict the efficiency of an n-i-p cell using a 500 nm "
                 "CsFAPbI3 absorber with SnO2 ETL.",
     "gt": [["ml.predict"]]},
    {"qid": "pred_03", "category": "prediction",
     "question": "Estimate the PCE of a 600 nm MAPbI3 planar device without "
                 "an HTL.",
     "gt": [["ml.predict"]]},
    {"qid": "pred_04", "category": "prediction",
     "question": "What is the model prediction for a FAPbI3 p-i-n cell with "
                 "PEDOT:PSS as HTL?",
     "gt": [["ml.predict"]]},
    {"qid": "pred_05", "category": "prediction",
     "question": "Given a device with 80 nm TiO2 and 200 nm perovskite, what "
                 "JV efficiency is predicted?",
     "gt": [["ml.predict"]]},
    {"qid": "pred_06", "category": "prediction",
     "question": "Use the trained model to get PCE and interval for a 2D-3D "
                 "mixed p-i-n stack.",
     "gt": [["ml.predict"]]},
    {"qid": "pred_07", "category": "prediction",
     "question": "Would the model predict above 20% for a 1200 nm FASnI3 "
                 "device?",
     "gt": [["ml.predict"]]},
    {"qid": "pred_08", "category": "prediction",
     "question": "Forecast the open probability of success (predicted PCE) "
                 "for a CsMAFA triple-cation n-i-p cell.",
     "gt": [["ml.predict"]]},
    {"qid": "pred_09", "category": "prediction",
     "question": "Predict the band-gap-limited maximum Jsc for a 1.55 eV "
                 "halide absorber and the resulting ideal PCE.",
     "gt": [["ml.predict"], ["calculator.pce"]]},
    {"qid": "pred_10", "category": "prediction",
     "question": "Predict PCE for a device identical to a TiO2 p-i-n, with "
                 "the perovskite thickness unknown — assume 300 nm.",
     "gt": [["ml.predict"]]},
    # ---- global interpretation ------------------------------------------
    {"qid": "glob_01", "category": "global_interpretation",
     "question": "What are the most important parameters for improving an "
                 "n-i-p device?",
     "gt": [["ml.explain_global"]]},
    {"qid": "glob_02", "category": "global_interpretation",
     "question": "Which features does the model say matter most for PCE across "
                 "the whole dataset?",
     "gt": [["ml.explain_global"]]},
    {"qid": "glob_03", "category": "global_interpretation",
     "question": "List the top features associated with higher efficiency in "
                 "overall devices.",
     "gt": [["ml.explain_global"]]},
    {"qid": "glob_04", "category": "global_interpretation",
     "question": "What drives the predicted PCE in the n-i-p architecture "
                 "according to the model?",
     "gt": [["ml.explain_global"]]},
    {"qid": "glob_05", "category": "global_interpretation",
     "question": "Show the SHAP ranking of influential inputs for the overall "
                 "model.",
     "gt": [["ml.explain_global"]]},
    {"qid": "glob_06", "category": "global_interpretation",
     "question": "Identify the dominant device parameters across all recorded "
                 "cells.",
     "gt": [["ml.explain_global"], ["pdb.query"]]},
    {"qid": "glob_07", "category": "global_interpretation",
     "question": "Which ETL/HTL choices does the model associate with high PCE "
                 "globally?",
     "gt": [["ml.explain_global"]]},
    {"qid": "glob_08", "category": "global_interpretation",
     "question": "For an n-i-p stack, report which inputs the model weights "
                 "most in its PCE prediction.",
     "gt": [["ml.explain_global"]]},
    {"qid": "glob_09", "category": "global_interpretation",
     "question": "Global feature summary: what contributes most to predicted "
                 "efficiency?",
     "gt": [["ml.explain_global"]]},
    {"qid": "glob_10", "category": "global_interpretation",
     "question": "Across n-i-p devices, which measured inputs drive the model?",
     "gt": [["ml.explain_global"]]},
    # ---- local interpretation -------------------------------------------
    {"qid": "loc_01", "category": "local_interpretation",
     "question": "For a 480 nm MAPbI3 p-i-n cell, which parameters move the "
                 "model's PCE prediction for THAT device?",
     "gt": [["ml.explain_local"]]},
    {"qid": "loc_02", "category": "local_interpretation",
     "question": "Explain the model decision for a SnO2 n-i-p device with "
                 "FAPbI3.",
     "gt": [["ml.explain_local"]]},
    {"qid": "loc_03", "category": "local_interpretation",
     "question": "Which features explain the single-device prediction of "
                 "18.2%?",
     "gt": [["ml.explain_local"]]},
    {"qid": "loc_04", "category": "local_interpretation",
     "question": "What made the model predict a low PCE for this CsPbBr3 p-i-n "
                 "device?",
     "gt": [["ml.explain_local"]]},
    {"qid": "loc_05", "category": "local_interpretation",
     "question": "Local SHAP for the PTAA/ITO p-i-n cell — what raised its "
                 "score?",
     "gt": [["ml.explain_local"]]},
    {"qid": "loc_06", "category": "local_interpretation",
     "question": "Why does the model think a 300 nm device with 20 nm ETL "
                 "performs well?",
     "gt": [["ml.explain_local"]]},
    {"qid": "loc_07", "category": "local_interpretation",
     "question": "Explain the prediction for a flexible substrate n-i-p cell "
                 "with 12% predicted PCE.",
     "gt": [["ml.explain_local"]]},
    {"qid": "loc_08", "category": "local_interpretation",
     "question": "Instance-level attribution: which three features moved this "
                 "device's PCE forecast?",
     "gt": [["ml.explain_local"]]},
    # ---- sensitivity / what-if ------------------------------------------
    {"qid": "sen_01", "category": "sensitivity_analysis",
     "question": "What happens to predicted PCE if I increase perovskite "
                 "thickness from 300 to 500 nm?",
     "gt": [["ml.what_if"]]},
    {"qid": "sen_02", "category": "sensitivity_analysis",
     "question": "How sensitive is the model to annealing temperature for an "
                 "n-i-p device?",
     "gt": [["ml.what_if"]]},
    {"qid": "sen_03", "category": "sensitivity_analysis",
     "question": "Sweep ETL thickness from 20 to 100 nm and report predicted "
                 "PCE.",
     "gt": [["ml.what_if"]]},
    {"qid": "sen_04", "category": "sensitivity_analysis",
     "question": "If the HTL is removed, how does the predicted efficiency "
                 "change?",
     "gt": [["ml.what_if"], ["ml.predict", "ml.predict"]]},
    {"qid": "sen_05", "category": "sensitivity_analysis",
     "question": "Does doubling hole mobility improve the model's PCE "
                 "prediction for a p-i-n stack?",
     "gt": [["ml.what_if"]]},
    {"qid": "sen_06", "category": "sensitivity_analysis",
     "question": "Vary perovskite thickness in steps of 50 nm from 200 to "
                 "800 nm.",
     "gt": [["ml.what_if"]]},
    {"qid": "sen_07", "category": "sensitivity_analysis",
     "question": "What is the parametric response of PCE to annealing time "
                 "for FAPbI3?",
     "gt": [["ml.what_if"]]},
    {"qid": "sen_08", "category": "sensitivity_analysis",
     "question": "Sweep ETL band gap across candidate transport materials and "
                 "return the predicted PCE curve.",
     "gt": [["ml.what_if"], ["material.lookup", "ml.what_if"]]},
    {"qid": "sen_09", "category": "sensitivity_analysis",
     "question": "Run a physics sensitivity sweep of absorber thickness using "
                 "SCAPS-style simulation.",
     "gt": [["scaps.get_sweep"]]},
    # ---- optimization ----------------------------------------------------
    {"qid": "opt_01", "category": "optimization",
     "question": "Which parameters should I change to improve the predicted "
                 "PCE of my n-i-p device?",
     "gt": [["ml.optimize"]]},
    {"qid": "opt_02", "category": "optimization",
     "question": "What is the best way to increase efficiency of a p-i-n "
                 "cell with PTAA?",
     "gt": [["ml.optimize"]]},
    {"qid": "opt_03", "category": "optimization",
     "question": "Recommend a better ETL thickness for my 15% MAPbI3 device.",
     "gt": [["ml.optimize"], ["ml.what_if"]]},
    {"qid": "opt_04", "category": "optimization",
     "question": "How can I push this device above 20% according to the "
                 "model?",
     "gt": [["ml.optimize"], ["ml.what_if"]]},
    {"qid": "opt_05", "category": "optimization",
     "question": "Optimize the absorber thickness and annealing conditions for "
                 "a SnO2 n-i-p stack.",
     "gt": [["ml.optimize"], ["ml.what_if"]]},
    {"qid": "opt_06", "category": "optimization",
     "question": "Suggest interventions that the model predicts would "
                 "increase PCE for a CsMAFA cell.",
     "gt": [["ml.optimize"]]},
    {"qid": "opt_07", "category": "optimization",
     "question": "Scan feedforward candidates: increased thickness vs "
                 "thinner ETL — which wins?",
     "gt": [["ml.what_if"], ["ml.optimize"]]},
    {"qid": "opt_08", "category": "optimization",
     "question": "What change most improves the predicted efficiency of an "
                 "n-i-p device?",
     "gt": [["ml.optimize"]]},
    # ---- comparison ------------------------------------------------------
    {"qid": "cmp_01", "category": "comparison",
     "question": "Compare the average PCE of p-i-n vs n-i-p devices in the "
                 "database.",
     "gt": [["pdb.query"]]},
    {"qid": "cmp_02", "category": "comparison",
     "question": "Which architecture shows higher reported efficiency: TiO2 "
                 "or SnO2 ETL?",
     "gt": [["pdb.query"], ["pdb.query", "pdb.query"]]},
    {"qid": "cmp_03", "category": "comparison",
     "question": "Spiro-MeOTAD vs PEDOT:PSS as HTL: which has broader "
                 "efficiency spread in DB1.0?",
     "gt": [["pdb.query"]]},
    {"qid": "cmp_04", "category": "comparison",
     "question": "Are flexible cells on average less efficient than rigid "
                 "ones in the dataset?",
     "gt": [["pdb.query"]]},
    {"qid": "cmp_05", "category": "comparison",
     "question": "Compare predicted PCE for MAPbI3 vs FAPbI3 identical "
                 "stacks.",
     "gt": [["ml.predict", "ml.predict"]]},
    {"qid": "cmp_06", "category": "comparison",
     "question": "How does the recorded median Voc differ between triple and "
                 "single cation devices?",
     "gt": [["pdb.query"]]},
    {"qid": "cmp_07", "category": "comparison",
     "question": "Which TiO2-based p-i-n devices in the database report top-"
                 "10 PCE?",
     "gt": [["pdb.query"]]},
    {"qid": "cmp_08", "category": "comparison",
     "question": "Contrast stability loss between encapsulated and open "
                 "devices.",
     "gt": [["pdb.query"]]},
    # ---- literature ------------------------------------------------------
    {"qid": "lit_01", "category": "literature_retrieval",
     "question": "Find papers on buried interface passivation for p-i-n "
                 "perovskite cells.",
     "gt": [["literature.search"]]},
    {"qid": "lit_02", "category": "literature_retrieval",
     "question": "Retrieve literature evidence on the origin of "
                 "ion migration in FAPbI3.",
     "gt": [["literature.search"]]},
    {"qid": "lit_03", "category": "literature_retrieval",
     "question": "What do recent publications say about Pb-Sn mixed absorbers?",
     "gt": [["literature.search"]]},
    {"qid": "lit_04", "category": "literature_retrieval",
     "question": "Find the DOI supporting NiOx as a p-i-n HTL.",
     "gt": [["literature.search"]]},
    {"qid": "lit_05", "category": "literature_retrieval",
     "question": "Which article discusses triple-cation CsMAFA stability "
                 "under humidity?",
     "gt": [["literature.search"]]},
    {"qid": "lit_06", "category": "literature_retrieval",
     "question": "Summarize evidence on passivation of surface defects with "
                 "organic halides.",
     "gt": [["literature.search"]]},
    {"qid": "lit_07", "category": "literature_retrieval",
     "question": "Find review sources on flexible perovskite photovoltaics.",
     "gt": [["literature.search"]]},
    {"qid": "lit_08", "category": "literature_retrieval",
     "question": "Retrieve the full text's key PCE claim for a given DOI's "
                 "n-i-p SnO2 study.",
     "gt": [["literature.search", "extraction.extract"]]},
    # ---- database query --------------------------------------------------
    {"qid": "db_01", "category": "database_query",
     "question": "What is the average PCE for devices using SnO2 as an ETL?",
     "gt": [["pdb.query"]]},
    {"qid": "db_02", "category": "database_query",
     "question": "How many perovskite devices are in the database?",
     "gt": [["pdb.query"]]},
    {"qid": "db_03", "category": "database_query",
     "question": "Top-10 highest PCE recorded devices.",
     "gt": [["pdb.query"]]},
    {"qid": "db_04", "category": "database_query",
     "question": "What is the median Voc of n-i-p cells with Spiro-MeOTAD "
                 "from DB1.0?",
     "gt": [["pdb.query"]]},
    {"qid": "db_05", "category": "database_query",
     "question": "Sample 5 devices using the FA0.83MA0.17PbI3 composition.",
     "gt": [["pdb.query"]]},
    {"qid": "db_06", "category": "database_query",
     "question": "Which architectures are recorded for flexible substrates — "
                 "list the distribution.",
     "gt": [["pdb.query"]]},
    {"qid": "db_07", "category": "database_query",
     "question": "Average annealing temperature used in top-100 PCE cells.",
     "gt": [["pdb.query"]]},
    {"qid": "db_08", "category": "database_query",
     "question": "Count devices with a reported stability T80 greater than "
                 "500 h.",
     "gt": [["pdb.query"]]},
    {"qid": "db_09", "category": "database_query",
     "question": "What fraction of p-i-n devices use copper-based HTLs?",
     "gt": [["pdb.query"]]},
    {"qid": "db_10", "category": "database_query",
     "question": "Show the distribution of Spiro-MeOTAD device efficiencies.",
     "gt": [["pdb.query"]]},
    {"qid": "db_11", "category": "database_query",
     "question": "How many devices report both Voc and Jsc above 1.1 V and "
                 "24 mA/cm2?",
     "gt": [["pdb.query"]]},
    {"qid": "db_12", "category": "database_query",
     "question": "Which substrate is most common among n-i-p devices?",
     "gt": [["pdb.query"]]},
    # ---- verification ----------------------------------------------------
    {"qid": "ver_01", "category": "verification",
     "question": "Verify the claim 'CH3NH3PbI3 band gap is 1.55 eV' against "
                 "the sources.",
     "gt": [["citations.verify"]]},
    {"qid": "ver_02", "category": "verification",
     "question": "Is 'PCE above 30% for single-junction cells' a defensible "
                 "claim per the database?",
     "gt": [["citations.verify"], ["pdb.query"]]},
    {"qid": "ver_03", "category": "verification",
     "question": "Check whether the retrieved DOI actually reports a p-i-n "
                 "architecture.",
     "gt": [["citations.verify"], ["literature.search"]]},
    {"qid": "ver_04", "category": "verification",
     "question": "Confirm the Voc of 1.1 V claimed in the abstract with a "
                 "reliable source.",
     "gt": [["citations.verify"]]},
    {"qid": "ver_05", "category": "verification",
     "question": "Validate the statement that SnO2 ETL yields higher average "
                 "PCE than TiO2 in DB1.0.",
     "gt": [["citations.verify"], ["pdb.query"]]},
    {"qid": "ver_06", "category": "verification",
     "question": "Does the literature support using 20% MA in triple-cation "
                 "absorbers?",
     "gt": [["citations.verify"], ["literature.search"]]},
    # ---- multi-tool investigation ----------------------------------------
    {"qid": "mt_01", "category": "multi_tool_investigation",
     "question": "Combine database records and ML feature importance: which "
                 "transport-layer properties are associated with high n-i-p "
                 "PCE?",
     "gt": [["ml.explain_global", "pdb.query"],
            ["pdb.query", "ml.explain_global"]]},
    {"qid": "mt_02", "category": "multi_tool_investigation",
     "question": "Use both the materials leg and the ML model to pick a "
                 "suitable ETL for a 1.6 eV absorber.",
     "gt": [["material.lookup", "ml.what_if"],
            ["materials_project.lookup", "ml.what_if"],
            ["material.lookup", "ml.predict"]]},
    {"qid": "mt_03", "category": "multi_tool_investigation",
     "question": "Literature + database: is NiOx a common p-i-n HTL and does "
                 "the model associate it with high PCE?",
     "gt": [["literature.search", "pdb.query"],
            ["pdb.query", "ml.explain_global"]]},
    {"qid": "mt_04", "category": "multi_tool_investigation",
     "question": "For a fresh device idea, predict its PCE and explain which "
                 "inputs drive the prediction.",
     "gt": [["ml.predict", "ml.explain_local"]]},
    {"qid": "mt_05", "category": "multi_tool_investigation",
     "question": "Verify arXiv claim and cross-check it against the "
                 "database record count.",
     "gt": [["citations.verify", "pdb.query"]]},
]

AMBIGUOUS: List[Dict] = [
    {"qid": "amb_01", "category": "ambiguous",
     "question": "What improves n-i-p performance?",
     "gt": [["ml.optimize"], ["ml.what_if"], ["ml.explain_global"]]},
    {"qid": "amb_02", "category": "ambiguous",
     "question": "What matters most for PCE?",
     "gt": [["ml.explain_global"], ["ml.optimize"]]},
    {"qid": "amb_03", "category": "ambiguous",
     "question": "Which properties should I focus on?",
     "gt": [["ml.explain_global"], ["ml.optimize"]]},
    {"qid": "amb_04", "category": "ambiguous",
     "question": "How can I increase efficiency?",
     "gt": [["ml.optimize"], ["ml.what_if"], ["ml.explain_global"]]},
    {"qid": "amb_05", "category": "ambiguous",
     "question": "Will thicker perovskite always help?",
     "gt": [["ml.what_if"], ["ml.explain_global"]]},
    {"qid": "amb_06", "category": "ambiguous",
     "question": "Is annealing important?",
     "gt": [["ml.explain_global"], ["ml.what_if"]]},
]

DISAMBIGUATION: List[Dict] = [
    # topic 1: prediction vs global interpretation
    {"qid": "dis_01a", "category": "disambiguation", "topic": "predict-vs-global",
     "question": "What PCE does the model predict for an n-i-p SnO2 device?",
     "gt": [["ml.predict"]]},
    {"qid": "dis_01b", "category": "disambiguation", "topic": "predict-vs-global",
     "question": "Which parameters does the model predict drive PCE for n-i-p "
                 "SnO2 devices?",
     "gt": [["ml.explain_global"]]},
    {"qid": "dis_01c", "category": "disambiguation", "topic": "predict-vs-global",
     "question": "What PCE does the model predict drives n-i-p devices?",
     "gt": [["ml.explain_global"], ["ml.predict"]]},
    # topic 2: global vs local interpretation
    {"qid": "dis_02a", "category": "disambiguation", "topic": "global-vs-local",
     "question": "Which features matter for n-i-p devices overall?",
     "gt": [["ml.explain_global"]]},
    {"qid": "dis_02b", "category": "disambiguation", "topic": "global-vs-local",
     "question": "Which features matter for THIS n-i-p device?",
     "gt": [["ml.explain_local"]]},
    {"qid": "dis_02c", "category": "disambiguation", "topic": "global-vs-local",
     "question": "This device predicts 18.2% — which features matter?",
     "gt": [["ml.explain_local"]]},
    # topic 3: sensitivity vs optimization
    {"qid": "dis_03a", "category": "disambiguation", "topic": "sensitivity-vs-optimization",
     "question": "How does predicted PCE change with perovskite thickness?",
     "gt": [["ml.what_if"]]},
    {"qid": "dis_03b", "category": "disambiguation", "topic": "sensitivity-vs-optimization",
     "question": "What thickness should I use to maximize predicted PCE?",
     "gt": [["ml.optimize"], ["ml.what_if"]]},
    {"qid": "dis_03c", "category": "disambiguation", "topic": "sensitivity-vs-optimization",
     "question": "What thickness should I use to reach a predicted PCE of "
                 "20%?",
     "gt": [["ml.what_if"], ["ml.optimize"]]},
    # topic 4: what-if semantic — output vs input resolution
    {"qid": "dis_04a", "category": "disambiguation", "topic": "parameter-meaning",
     "question": "How sensitive is PCE to band gap widening?",
     "gt": [["ml.what_if"], ["scaps.get_sweep"]]},
    {"qid": "dis_04b", "category": "disambiguation", "topic": "parameter-meaning",
     "question": "How sensitive is band gap to alloy composition?",
     "gt": [["material.lookup"], ["ml.what_if"]]},
    {"qid": "dis_04c", "category": "disambiguation", "topic": "parameter-meaning",
     "question": "Which compositions can reach a 1.7 eV band gap?",
     "gt": [["material.lookup"], ["materials_project.lookup"]]},
    # topic 5: verification vs database
    {"qid": "dis_05a", "category": "disambiguation", "topic": "verify-vs-query",
     "question": "Check whether SnO2 ETLs really average above 20% PCE.",
     "gt": [["citations.verify"], ["pdb.query"]]},
    {"qid": "dis_05b", "category": "disambiguation", "topic": "verify-vs-query",
     "question": "What do SnO2 ETLs average in PCE?",
     "gt": [["pdb.query"]]},
    {"qid": "dis_05c", "category": "disambiguation", "topic": "verify-vs-query",
     "question": "Find the source backing the claim that SnO2 ETLs average "
                 "above 20%.",
     "gt": [["literature.search"], ["citations.verify"]]},
]


def all_questions() -> List[Dict]:
    return SIMPLE + AMBIGUOUS + DISAMBIGUATION