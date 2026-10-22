"""PerovAgents — chat + live pipeline trace UI.

This Streamlit app wraps the agent in a chat interface and renders the full
execution trace live: the plan (tools planned), each step as it runs (current
step highlighted), every tool's input/output, provenance DOIs, sanity warnings,
and the final synthesized answer.

Run with:
    cd perovagents
    streamlit run frontend/app.py
"""
import json
import sys
import os
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perovagents.core.logsetup import setup_logging

# make the agentic flow observable (console + results/logs/agent.log)
setup_logging()

LAYER_LABEL = {
    "orchestration": "Orchestration",
    "literature": "Literature",
    "materials": "Materials",
    "ml": "ML",
    "physics": "Physics",
}

LAYER_COLOR = {
    "orchestration": "#9E9E9E",
    "literature": "#1976D2",
    "materials": "#2E7D32",
    "ml": "#7B1FA2",
    "physics": "#E65100",
}


# ---------------------------------------------------------------- result rendering
def render_result(result: dict):
    """Render a tool result dict in a human-friendly way."""
    if not isinstance(result, dict):
        st.write(result)
        return
    rt = result.get("result_type")
    if result.get("status") == "error":
        st.error(result.get("error", "tool error"))
        return

    if rt == "count":
        st.metric("Records matched", result.get("value"))
        st.caption(f"Across {result.get('n_papers', 0)} papers")
    elif rt == "scalar":
        v = result.get("value")
        st.metric(result.get("column", "value"),
                  "n/a" if v is None else round(v, 4))
    elif rt in ("table", "top_k"):
        data = result.get("data") or []
        if data:
            st.dataframe(data, use_container_width=True)
            st.caption(f"{len(data)} rows · {result.get('n_papers', 0)} papers")
        else:
            st.info("No rows returned.")
    elif rt == "distribution":
        for m in ("mean", "median", "min", "max"):
            if result.get(m) is not None:
                st.metric(m.title(), round(result[m], 4))
        st.caption(f"column={result.get('column')} · n={result.get('n', 0)}")
    elif rt == "unique":
        data = result.get("data") or []
        st.write(" · ".join(f"`{x}`" for x in data[:20]))
        st.caption(f"{len(data)} unique values")
    elif rt == "material":
        col_left, col_right = st.columns(2)
        col_left.metric("Material", result.get("material"))
        col_right.metric("Property", result.get("property"))
        st.metric("Value", f"{result.get('value')} {result.get('unit', '')}")
    elif rt == "material_lookup":
        props = result.get("data") or []
        if props:
            cols = st.columns(min(len(props), 4))
            for i, p in enumerate(props):
                cols[i % len(cols)].metric(
                    p.get("property", "?"),
                    f"{p.get('value')} {p.get('unit', '')}")
        st.caption(f"source: {result.get('source', 'n/a')}")
    elif rt == "literature_passages":
        hits = result.get("hits") or []
        for h in hits:
            with st.expander(f"{h.get('score', 0):.2f} — {h.get('paper_title', h.get('doi','hit'))}"):
                st.caption(h.get("doi", ""))
                st.write(h.get("snippet", ""))
        if not hits:
            st.info("No passages matched.")
    elif rt == "prediction":
        pce = result.get("predicted_pce")
        lo, hi = (result.get("pce_95_interval") or [None, None])
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted PCE (%)", pce)
        c2.metric("95% interval low (%)", lo)
        c3.metric("95% interval high (%)", hi)
        st.caption(f"architecture={result.get('architecture')} · "
                   f"seed-bags={result.get('n_bags')}")
        hood = result.get("hood") or {}
        if hood.get("applied"):
            if hood.get("ood"):
                st.warning(f"Out-of-domain: {hood.get('n_out_of_envelope')} features "
                           f"beyond training envelope")
            else:
                st.success("Within training domain")
    elif rt == "explanation":
        rows = (result.get("features")
                or result.get("shap_top_features") or [])
        if rows:
            df = [{"feature": r.get("name") or r.get("feature"),
                   "importance": r.get("importance"),
                   "direction": r.get("direction")}
                  for r in rows if isinstance(r, dict)]
            st.dataframe(df, use_container_width=True)
            st.caption(f"method={result.get('method', 'n/a')} · "
                       f"scope={result.get('scope', 'n/a')} · "
                       f"architecture={result.get('architecture', 'n/a')}")
            controllable = result.get("controllable") or {}
            if controllable:
                st.caption("controllable: " + ", ".join(
                    f"`{k}`" for k, v in controllable.items() if v)
                    or "none of the top features are experimental inputs")
    elif rt == "what_if":
        rows = result.get("data") or []
        if rows:
            df = [{"value": r.get("parameter_value"),
                   "predicted PCE (%)": r.get("predicted_pce"),
                   "95% lo": r.get("interval_lo"),
                   "95% hi": r.get("interval_hi")} for r in rows]
            st.dataframe(df, use_container_width=True)
            st.caption(f"parameter=`{result.get('parameter')}` · "
                       f"category={result.get('category', 'n/a')}")
        else:
            st.info("No sweep rows returned.")
    elif rt == "optimization":
        recs = result.get("recommendations") or []
        if recs:
            df = [{"parameter": r.get("parameter"),
                   "kind": r.get("kind", "numeric"),
                   "direction": r.get("direction"),
                   "recommended": r.get("recommended_value"),
                   "base PCE": r.get("base_predicted_pce"),
                   "best PCE": r.get("best_predicted_pce"),
                   "delta": r.get("delta_predicted_pce")} for r in recs]
            st.dataframe(df, use_container_width=True)
            with st.expander("Prediction intervals"):
                for r in recs:
                    st.caption(f"{r.get('parameter')}: " + ", ".join(
                        f"`{v.get('parameter_value')}` → {v.get('predicted_pce')}% "
                        f"[{v.get('interval_lo')}, {v.get('interval_hi')}]"
                        for v in r.get("values") or []))
        else:
            st.metric("Predicted PCE (%)",
                      result.get("predicted_pce") or result.get("base_predicted_pce"))
            st.info(result.get("note", "No improvement found over baseline."))
        if recs and result.get("note"):
            st.caption(result.get("note", ""))
    elif rt == "citation_verdict":
        verdict = result.get("verdict")
        icon = {"PASS": ":**green[PASS]**", "WARN": ":**orange[WARN]**",
                "FAIL": ":**red[FAIL]**"}.get(verdict, "")
        st.markdown(f"Verdict {icon}")
        st.caption(result.get("detail", ""))
    elif rt == "scaps_sweep":
        st.code(json.dumps(result.get("summary", result), indent=2, default=str))
    elif rt == "docstring":
        st.code(result.get("help", ""))
    elif rt is None:
        for k, v in result.items():
            if k in {"result_type", "provenance", "warnings", "rows_before",
                     "rows_after"}:
                continue
            st.write(f"**{k}:** {v}")
    else:
        st.code(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------- step rendering
def render_step(card, step: dict, status: str, reason: str = "",
                nested: bool = False):
    """Render one tool step inside the placeholder `card`.

    status: "pending" | "running" | "ok" | "error"
    nested: set True when this card is itself inside an expander (e.g. the
            history trace), since Streamlit forbids nested expanders.
    """
    tool = step.get("tool", "?")
    action = step.get("action", "")
    layer = ""
    from perovagents.core import default_registry
    try:
        spec = next((s for s in default_registry().available()
                     if s.name == tool), None)
        layer = spec.layer if spec else ""
    except Exception:
        pass

    marker = {
        "pending": "PENDING",
        "running": "RUNNING",
        "ok": "OK",
        "error": "FAILED",
    }[status]
    badge = f"**{tool}**" + (f" → {action}" if action else "")
    lbl = LAYER_LABEL.get(layer, layer or "core")

    with card.container(border=True):
        hc1, hc2 = st.columns([6, 1])
        hc1.markdown(f"{marker} · {badge}")
        hc2.caption(lbl)

        if status == "running":
            st.progress(0.5, text="executing…")

        purpose = step.get("purpose") or step.get("reason") or ""
        if purpose and status in ("pending", "running"):
            st.caption(f"_Why:_ {purpose}")
        elif purpose and status == "ok":
            st.caption(f"_Purpose:_ {purpose}")

        if step.get("params"):
            if nested:
                st.code(json.dumps(step["params"], indent=2, default=str))
            else:
                with st.expander("Parameters", expanded=False):
                    st.code(json.dumps(step["params"], indent=2, default=str))

        if status == "error":
            st.error(step.get("error", "step failed"))
        elif status == "ok" and isinstance(step.get("result"), dict):
            render_result(step["result"])

        if status == "ok" and step.get("warnings"):
            for w in step["warnings"]:
                st.warning(w)
        if status == "ok" and step.get("elapsed_seconds"):
            st.caption(f"elapsed {step['elapsed_seconds']:.2f}s")


def render_plan(plan: dict):
    """Show the planned tool sequence (before execution)."""
    steps = plan.get("steps") or []
    st.markdown("#### Planned tools")
    st.caption(f"Planner: `{plan.get('planner', 'unknown')}`")
    for i, s in enumerate(steps, 1):
        tool = s.get("tool", "?")
        action = s.get("action", "")
        purpose = s.get("purpose") or s.get("reason", "")
        st.markdown(f"{i}. **{tool}**" + (f" → {action}" if action else ""))
        if purpose:
            st.caption(f"   _Why:_ {purpose}")


# ---------------------------------------------------------------- provenance
def render_verdict(verdict: dict):
    """Render the final task-sufficiency verdict."""
    if not verdict:
        return
    status = verdict.get("status", "UNRESOLVED")
    icon = {"SUFFICIENT": ":**green[SUFFICIENT]**",
            "RECOVERABLE": ":**orange[RECOVERABLE]**",
            "UNRESOLVED": ":**red[UNRESOLVED]**"}.get(status, f"**[{status}]**")
    st.markdown(f"Task verification {icon}")
    if verdict.get("missing_requirements"):
        st.caption("missing outputs: " + ", ".join(verdict["missing_requirements"]))


def render_plan_iterations(plan_iterations):
    """Short summary of replan attempts (iteration → planner → verdict)."""
    if not plan_iterations or len(plan_iterations) == 1:
        return
    st.markdown("**Replanning**")
    for it in plan_iterations:
        st.caption(
            f"`iter {it.get('iteration')}` {it.get('planner')} → "
            f"{it.get('verdict', {}).get('status', '?')}")


def render_sources(sources):
    if not sources:
        st.caption("Sources: none")
        return
    st.markdown("**Sources**")
    uniq = list(dict.fromkeys(sources))
    cols = st.columns(min(len(uniq), 3))
    for i, doi in enumerate(uniq[:12]):
        with cols[i % 3]:
            st.code(doi, language="text")
    if len(uniq) > 12:
        st.caption(f"... and {len(uniq) - 12} more")


# ---------------------------------------------------------------- session setup
def get_agent():
    if "agent" not in st.session_state:
        from perovagents import Agent
        st.session_state.agent = Agent(pipeline="llm", max_steps=6)
        st.session_state.agent_created_at = time.time()
    return st.session_state.agent


def main():
    st.set_page_config(page_title="PerovAgents", layout="wide")
    st.title("PerovAgents")
    st.caption("Chat with a perovskite solar cell research agent. Watch every "
               "planned tool, the current step, and each tool's output.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    agent = get_agent()

    # ------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Session")
        st.write(f"Agent: **{agent.pipeline}** planner")
        st.write(f"LLM available: {'yes' if agent.llm.available else 'no (rules offline)'}")
        st.caption(f"API key pool: {agent.llm.num_keys} keys "
                   f"(rotating) · model `{agent.llm.model}`")
        if st.button("Reset conversation"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.header("Tool registry")
        from perovagents.core import default_registry
        reg = default_registry()
        for layer in ("orchestration", "literature", "materials", "ml", "physics"):
            specs = [s for s in reg.available() if s.layer == layer]
            if not specs:
                continue
            with st.expander(LAYER_LABEL[layer], expanded=False):
                for s in specs:
                    st.markdown(f"**`{s.name}`**")
                    st.caption(s.description)
                    if s.capabilities:
                        st.caption("· " + ", ".join(sorted(s.capabilities)))

        st.divider()
        st.caption("Tip: try “average PCE for SnO2 devices”, "
                   "“why do these parameters matter?”, "
                   "“what if I increase the thickness?”, or "
                   "“what should I change to improve PCE?”")

    # ------------------------------------------------------------ chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("trace"):
                with st.expander("View full processing trace"):
                    render_plan(msg["trace"]["plan"])
                    render_plan_iterations(msg["trace"].get("plan_iterations") or [])
                    render_verdict(msg["trace"].get("verdict") or {})
                    for i, step in enumerate(msg["trace"]["execution"]["steps"]):
                        render_step(st.empty(), step, "ok" if step["success"] else "error",
                                    nested=True)
                    render_sources(msg["trace"]["sources"])

    # ------------------------------------------------------------ input
    if prompt := st.chat_input("Ask about perovskite solar cells…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # ---- run the agent with live callbacks --------------------------
        with st.chat_message("assistant"):
            status_panel = st.empty()
            with status_panel:
                st.markdown("_Planning…_")

            plan_holder = st.empty()
            step_holders = []
            result = None

            def on_plan(plan_dict):
                with plan_holder:
                    render_plan(plan_dict)
                # pre-paint pending cards for every planned step so the user
                # sees the full tool sequence even before step 1 finishes
                for s in plan_dict.get("steps") or []:
                    holder = st.empty()
                    step_holders.append(holder)
                    with holder:
                        render_step(holder, s, "pending",
                                    reason=s.get("reason", ""))

            def on_step_start(i, step_dict):
                with step_holders[i]:
                    render_step(step_holders[i], step_dict, "running",
                                reason=step_dict.get("reason", ""))

            def on_step_done(i, step_dict):
                with step_holders[i]:
                    render_step(step_holders[i], step_dict,
                                "ok" if step_dict.get("success") else "error",
                                reason=step_dict.get("reason", ""))

            with st.spinner("Agent is working…"):
                result = agent.answer(prompt, on_plan=on_plan,
                                      on_step_start=on_step_start,
                                      on_step_done=on_step_done)

            with status_panel:
                done = result["execution"]["total_elapsed_seconds"]
                st.markdown(f"_Agent finished in {done:.2f}s · "
                            f"{result['execution']['n_ok']} steps OK_")
                render_verdict(result.get("verdict") or {})
                render_plan_iterations(result.get("plan_iterations") or [])

            st.markdown("## Answer")
            st.write(result["answer"])
            render_sources(result["sources"])
            if result.get("flow_log"):
                st.caption(f"full flow log → `{result['flow_log']}`")
            if result["warnings"]:
                st.markdown("**Warnings**")
                for w in result["warnings"]:
                    st.warning(w)

            # store for history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "trace": result,
            })


main()