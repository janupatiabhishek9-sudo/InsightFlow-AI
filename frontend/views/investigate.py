"""Investigate page: dataset, question, live progress and the evidence-backed result."""

from __future__ import annotations

import pandas as pd
import plotly.io as pio
import streamlit as st

from app.api.schemas import DatasetInfo, InvestigationView
from ui_core import NODE_LABELS, STATUS_STYLE, access_store, ai_label, backend, gate, hero, pill

EXAMPLES = [
    ("📉 Why the Q3 drop?", "Why did European revenue decrease in Q3, and which products contributed most to the decline?"),
    ("🌍 Which countries?", "Which countries contributed most to the decline?"),
    ("🧮 Orders, price or discount?", "Was the decline caused by fewer orders, lower quantity, lower prices, or discounts?"),
    ("👥 Customer segments", "Which customer segments were affected by the European revenue decline in Q3?"),
    ("📈 Monthly trend", "Show the monthly revenue trend for Germany in 2024"),
    ("🧹 Clean data (needs approval)", "Delete rows with negative revenue"),
]


def run_with_progress(label: str, action):
    """Run a backend call, listing each workflow step live as it completes."""
    with st.status(label, expanded=True) as status:
        def on_progress(event: dict) -> None:
            icon = {"ok": "✅", "error": "❌", "skipped": "⏭️"}.get(event.get("status"), "•")
            name = NODE_LABELS.get(event["node"], event["node"])
            st.write(f"{icon} {name} · {event.get('duration_ms', 0) / 1000:.1f}s")

        try:
            view = action(on_progress)
        except ValueError as e:
            status.update(label=f"Failed: {e}", state="error")
            st.error(str(e))
            return None
        ok = view.status in ("completed", "awaiting_approval", "needs_clarification")
        status.update(label=f"{label}: {view.status.replace('_', ' ')}", state="complete" if ok else "error",
                      expanded=False)
        return view


def sidebar(be) -> None:
    with st.sidebar:
        tags = [pill(ai_label(be), "green" if "offline" not in ai_label(be) else "amber")]
        if st.session_state.get("coupon"):
            tags.append(pill(f"code {st.session_state.coupon}"))
        st.markdown(" ".join(tags), unsafe_allow_html=True)
        if st.session_state.get("coupon") and st.button("Sign out", use_container_width=True):
            for key in ("coupon", "dataset", "view", "question"):
                st.session_state.pop(key, None)
            st.rerun()
        st.divider()
        st.subheader("📁 Dataset")
        if st.button("Use example sales dataset", use_container_width=True, type="primary"):
            st.session_state.dataset = be.example()
            st.session_state.uploaded_name = None
            st.session_state.pop("view", None)
        upload = st.file_uploader("…or upload CSV / Excel", type=["csv", "xlsx", "xls"])
        if upload is not None and st.session_state.get("uploaded_name") != upload.name:
            try:
                st.session_state.dataset = be.upload(upload.name, upload.getvalue())
                st.session_state.uploaded_name = upload.name
                st.session_state.pop("view", None)
            except ValueError as e:
                st.error(str(e))


def dataset_overview(info: DatasetInfo) -> None:
    q = info.quality
    st.markdown(f"#### 📊 {info.filename}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{q.row_count:,}")
    c2.metric("Columns", q.column_count)
    c3.metric("Date range", f"{q.date_range[0][:7]} → {q.date_range[1][:7]}" if q.date_range else "n/a")
    c4.metric("Quality findings", len(q.warnings))
    with st.expander("Data profile & quality findings"):
        for w in q.warnings:
            st.markdown(f"- ⚠️ {w}")
        st.dataframe(pd.DataFrame([c.model_dump(exclude={"top_values"}) for c in q.columns]),
                     use_container_width=True, hide_index=True)
        st.caption("Sample rows")
        st.dataframe(pd.DataFrame(q.sample_rows), use_container_width=True, hide_index=True)


def ask(be, info: DatasetInfo) -> None:
    st.markdown("#### 💬 Ask a question")
    cols = st.columns(3)
    for i, (label, question) in enumerate(EXAMPLES):
        if cols[i % 3].button(label, key=f"ex{i}", use_container_width=True):
            st.session_state.question = question
    question = st.text_area("Your question", key="question", height=80, label_visibility="collapsed",
                            placeholder="e.g. Why did European revenue decline in Q3?")
    if st.button("🚀 Start investigation", type="primary", disabled=not (question or "").strip()):
        new = run_with_progress("Investigating", lambda cb: be.start(info.dataset_id, question, cb))
        if new is not None:
            access_store().record_investigation(st.session_state.get("coupon"))
            st.session_state.view = new


def headline(view: InvestigationView) -> None:
    getattr(st, STATUS_STYLE.get(view.status, "info"))(
        f"**{view.status.replace('_', ' ').capitalize()}** - {view.status_reason}")
    if view.report and view.report.executive_finding:
        st.markdown(f'<div class="if-finding">{view.report.executive_finding}</div>', unsafe_allow_html=True)
    comparison = next((r for r in view.step_results if r.kind == "period_comparison" and r.rows), None)
    if comparison:
        r = comparison.rows[0]
        c1, c2, c3 = st.columns(3)
        c1.metric(r["comparison_period"], f"{r['comparison_value']:,.0f}")
        c2.metric(r["period"], f"{r['current_value']:,.0f}",
                  delta=f"{r['pct_change']:+.1f}%" if r.get("pct_change") is not None else None)
        c3.metric("Change", f"{r['abs_change']:,.0f}")


def approval(view: InvestigationView, be) -> None:
    st.warning("⚠️ **ACTION REQUIRES APPROVAL**")
    for a in view.pending_actions:
        st.markdown(f"**Tool:** `{a.tool}`  \n**Operation:** {a.operation}  \n**Reason:** {a.reason}  \n**Risk:** {a.risk}")
        if a.command:
            st.code(a.command, language="sql")
    reviewer = st.text_input("Reviewer name", value="analyst")
    comment = st.text_input("Comment (optional)")
    c1, c2 = st.columns(2)
    for column, label, approved in ((c1, "APPROVE", True), (c2, "REJECT", False)):
        if column.button(label, type="primary" if approved else "secondary", use_container_width=True):
            new = run_with_progress("Resuming investigation",
                                    lambda cb: be.decide(view.investigation_id, approved, reviewer, comment, cb))
            if new is not None:
                st.session_state.view = new
                st.rerun()


def results(view: InvestigationView, be) -> None:
    headline(view)
    if view.status == "awaiting_approval":
        approval(view, be)
    tabs = st.tabs(["📄 Report", "📈 Charts", "🧭 Plan & SQL", "🔬 Evidence", "📚 Context", "🧾 Trace"])
    with tabs[0]:
        if view.report:
            st.markdown(view.report.markdown)
    with tabs[1]:
        if not view.charts:
            st.caption("No charts for this investigation.")
        for c in view.charts:
            st.plotly_chart(pio.from_json(c["figure_json"]), use_container_width=True)
    with tabs[2]:
        if view.plan:
            st.markdown(f"**Objective:** {view.plan.objective}")
            for s in view.plan.steps:
                res = next((r for r in view.step_results if r.step == s.step), None)
                badge = {"ok": "✅", "error": "❌", "skipped": "⏭️"}.get(res.status, "") if res else "⏳"
                st.markdown(f"{badge} **{s.step}. {s.purpose}** · `{s.tool}`  \n_{s.rationale}_")
                if res and res.sql:
                    with st.expander("SQL"):
                        st.code(res.sql, language="sql")
        if view.plan_validation:
            st.markdown(f"**Plan validation:** {view.plan_validation.status}")
        if view.risk:
            st.markdown(f"**Risk:** {view.risk.level} - " + "; ".join(view.risk.reasons))
        if view.understanding:
            with st.expander("How the question was understood"):
                st.json(view.understanding.model_dump(mode="json"), expanded=False)
    with tabs[3]:
        for e in view.evidence:
            with st.expander(f"{e.id} · {e.description} ({e.evidence_type})"):
                st.caption(f"source: {e.source}")
                if e.query:
                    st.code(e.query, language="sql")
                if isinstance(e.result, list) and e.result and isinstance(e.result[0], dict):
                    st.dataframe(pd.DataFrame(e.result), use_container_width=True, hide_index=True)
                else:
                    st.write(e.result)
        if view.validation:
            st.markdown("**Result validation checks**")
            st.dataframe(pd.DataFrame([c.model_dump() for c in view.validation.checks]),
                         use_container_width=True, hide_index=True)
    with tabs[4]:
        if not view.retrieved_context:
            st.caption("No context retrieved.")
        for e in view.retrieved_context:
            st.markdown(f"**{e.description}** · `{e.source}`")
            st.markdown(f"> {str(e.result)[:600]}")
    with tabs[5]:
        if view.evaluation:
            cols = st.columns(4)
            for i, (k, v) in enumerate(view.evaluation.metrics.items()):
                cols[i % 4].metric(k.replace("_", " "), v)
        st.markdown(f"**Model:** {view.model}  \n**Prompt versions:** {view.prompt_versions}  \n**Tokens:** {view.token_usage}")
        with st.expander("Workflow steps"):
            for e in view.trace_events:
                icon = {"ok": "✅", "error": "❌", "skipped": "⏭️"}.get(e.get("status"), "•")
                st.markdown(f"{icon} {NODE_LABELS.get(e['node'], e['node'])} · {e.get('duration_ms', 0):.0f} ms")
        st.dataframe(pd.DataFrame([t.model_dump() for t in view.tool_calls]), use_container_width=True, hide_index=True)
        for err in view.errors:
            st.markdown(f"- {err}")
        if st.button("Load full trace JSON"):
            st.json(be.trace(view.investigation_id), expanded=False)


def main() -> None:
    if not gate():
        return
    be = backend()
    sidebar(be)
    hero("InsightFlow AI", "Ask your data why - every answer comes with the evidence behind it.", small=True)
    info: DatasetInfo | None = st.session_state.get("dataset")
    if info is None:
        st.info("👈 Pick **Use example sales dataset** or upload your own file in the sidebar to begin.")
        return
    dataset_overview(info)
    ask(be, info)
    view: InvestigationView | None = st.session_state.get("view")
    if view is not None:
        results(view, be)


main()
