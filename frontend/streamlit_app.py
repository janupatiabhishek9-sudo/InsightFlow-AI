"""InsightFlow AI - Streamlit UI.

Run:  streamlit run frontend/streamlit_app.py
UI_BACKEND=local (default) runs the service in this process (one process, lowest RAM);
UI_BACKEND=http talks to the FastAPI server at API_URL. Both return the same Pydantic views.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
import plotly.io as pio
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.schemas import DatasetInfo, InvestigationView  # noqa: E402
from app.config import get_settings  # noqa: E402

EXAMPLES = [
    "Why did European revenue decrease in Q3, and which products contributed most to the decline?",
    "Which countries contributed most to the decline?",
    "Was the decline caused by fewer orders, lower quantity, lower prices, or discounts?",
    "Which customer segments were affected?",
    "Show the monthly revenue trend for Germany in 2024",
    "Delete rows with negative revenue",
]
STATUS_STYLE = {"completed": "success", "awaiting_approval": "warning", "needs_clarification": "info",
                "blocked": "error", "rejected": "error", "failed": "error", "invalid_plan": "warning"}


class LocalBackend:
    def __init__(self):
        from app.service import InsightFlowService

        self.svc = InsightFlowService()

    def example(self) -> DatasetInfo:
        return self.svc.example_dataset()

    def upload(self, name: str, data: bytes) -> DatasetInfo:
        return self.svc.register_dataset(name, data)

    def start(self, dataset_id: str, question: str) -> InvestigationView:
        return self.svc.start_investigation(dataset_id, question)

    def decide(self, inv_id: str, approved: bool, reviewer: str, comment: str) -> InvestigationView:
        return self.svc.decide(inv_id, approved, reviewer, comment)

    def trace(self, inv_id: str) -> dict:
        return self.svc.trace(inv_id)


class HttpBackend:
    def __init__(self, url: str):
        self.http = httpx.Client(base_url=url, timeout=180)

    def _ok(self, r: httpx.Response) -> dict:
        if r.status_code >= 400:
            raise ValueError(r.json().get("detail", r.text))
        return r.json()

    def example(self) -> DatasetInfo:
        return DatasetInfo.model_validate(self._ok(self.http.get("/datasets/example")))

    def upload(self, name: str, data: bytes) -> DatasetInfo:
        return DatasetInfo.model_validate(self._ok(self.http.post("/datasets", files={"file": (name, data)})))

    def start(self, dataset_id: str, question: str) -> InvestigationView:
        return InvestigationView.model_validate(
            self._ok(self.http.post("/investigations", json={"dataset_id": dataset_id, "question": question})))

    def decide(self, inv_id: str, approved: bool, reviewer: str, comment: str) -> InvestigationView:
        body = {"approved": approved, "reviewer": reviewer, "comment": comment}
        return InvestigationView.model_validate(self._ok(self.http.post(f"/investigations/{inv_id}/decision", json=body)))

    def trace(self, inv_id: str) -> dict:
        return self._ok(self.http.get(f"/investigations/{inv_id}/trace"))


@st.cache_resource(show_spinner="Starting InsightFlow AI...")
def backend():
    s = get_settings()
    return HttpBackend(s.api_url) if s.ui_backend == "http" else LocalBackend()


def show_profile(info: DatasetInfo) -> None:
    q = info.quality
    with st.expander(f"Dataset profile - {info.filename}", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{q.row_count:,}")
        c2.metric("Columns", q.column_count)
        c3.metric("Duplicate rows", q.duplicate_rows)
        c4.metric("Date range", f"{q.date_range[0]} to {q.date_range[1]}" if q.date_range else "n/a")
        if q.warnings:
            st.markdown("**Data-quality findings**")
            for w in q.warnings:
                st.markdown(f"- {w}")
        st.dataframe(pd.DataFrame([c.model_dump(exclude={"top_values"}) for c in q.columns]), use_container_width=True, hide_index=True)
        st.caption("Sample rows")
        st.dataframe(pd.DataFrame(q.sample_rows), use_container_width=True, hide_index=True)


def show_approval(view: InvestigationView, be) -> None:
    st.warning("⚠ ACTION REQUIRES APPROVAL")
    for a in view.pending_actions:
        st.markdown(f"**Tool:** `{a.tool}`  \n**Operation:** {a.operation}  \n**Reason:** {a.reason}  \n**Risk:** {a.risk}")
        if a.command:
            st.code(a.command, language="sql")
    reviewer = st.text_input("Reviewer name", value="analyst")
    comment = st.text_input("Comment (optional)")
    c1, c2 = st.columns(2)
    if c1.button("APPROVE", type="primary", use_container_width=True):
        st.session_state.view = be.decide(view.investigation_id, True, reviewer, comment)
        st.rerun()
    if c2.button("REJECT", use_container_width=True):
        st.session_state.view = be.decide(view.investigation_id, False, reviewer, comment)
        st.rerun()


def show_view(view: InvestigationView, be) -> None:
    getattr(st, STATUS_STYLE.get(view.status, "info"))(f"**Status: {view.status.replace('_', ' ')}** - {view.status_reason}")

    with st.expander("Investigation progress", expanded=view.status != "completed"):
        for e in view.trace_events:
            icon = {"ok": "✅", "error": "❌", "skipped": "⏭️"}.get(e.get("status"), "•")
            detail = f" - {e['detail']}" if e.get("detail") else ""
            st.markdown(f"{icon} `{e['node']}` ({e.get('duration_ms', 0):.0f} ms){detail}")

    if view.status == "awaiting_approval":
        show_approval(view, be)

    tabs = st.tabs(["Report", "Plan", "Charts", "Evidence", "Context", "Evaluation & Trace"])
    with tabs[0]:
        if view.report:
            st.markdown(view.report.markdown)
    with tabs[1]:
        if view.understanding:
            st.markdown("**Question understanding**")
            st.json(view.understanding.model_dump(mode="json"), expanded=False)
        if view.plan:
            st.markdown(f"**Objective:** {view.plan.objective}")
            for s in view.plan.steps:
                res = next((r for r in view.step_results if r.step == s.step), None)
                badge = {"ok": "✅", "error": "❌", "skipped": "⏭️"}.get(res.status, "") if res else "⏳"
                st.markdown(f"{badge} **{s.step}. {s.purpose}** - `{s.tool}`  \n_{s.rationale}_")
                if res and res.sql:
                    with st.expander("SQL", expanded=False):
                        st.code(res.sql, language="sql")
        if view.plan_validation:
            st.markdown(f"**Plan validation:** {view.plan_validation.status}")
            for i in view.plan_validation.issues:
                st.markdown(f"- {i.severity}: {i.message}")
        if view.risk:
            st.markdown(f"**Risk:** {view.risk.level} - " + "; ".join(view.risk.reasons))
    with tabs[2]:
        if not view.charts:
            st.caption("No charts for this investigation.")
        for c in view.charts:
            st.plotly_chart(pio.from_json(c["figure_json"]), use_container_width=True)
    with tabs[3]:
        for e in view.evidence:
            with st.expander(f"{e.id} - {e.description} ({e.evidence_type})"):
                st.caption(f"source: {e.source}")
                if e.query:
                    st.code(e.query, language="sql")
                if isinstance(e.result, list) and e.result and isinstance(e.result[0], dict):
                    st.dataframe(pd.DataFrame(e.result), use_container_width=True, hide_index=True)
                else:
                    st.write(e.result)
        if view.validation:
            st.markdown("**Result validation checks**")
            st.dataframe(pd.DataFrame([c.model_dump() for c in view.validation.checks]), use_container_width=True, hide_index=True)
    with tabs[4]:
        if not view.retrieved_context:
            st.caption("No context retrieved.")
        for e in view.retrieved_context:
            st.markdown(f"**{e.description}** - `{e.source}`")
            st.markdown(f"> {str(e.result)[:600]}")
    with tabs[5]:
        if view.evaluation:
            cols = st.columns(4)
            for i, (k, v) in enumerate(view.evaluation.metrics.items()):
                cols[i % 4].metric(k.replace("_", " "), v)
        st.markdown(f"**Model:** {view.model}  \n**Prompt versions:** {view.prompt_versions}  \n**Tokens:** {view.token_usage}")
        st.markdown("**Tool calls**")
        st.dataframe(pd.DataFrame([t.model_dump() for t in view.tool_calls]), use_container_width=True, hide_index=True)
        if view.errors:
            st.markdown("**Errors**")
            for err in view.errors:
                st.markdown(f"- {err}")
        if st.button("Load full trace JSON"):
            st.json(be.trace(view.investigation_id), expanded=False)


def main() -> None:
    st.set_page_config(page_title="InsightFlow AI", page_icon="📊", layout="wide")
    st.title("InsightFlow AI")
    st.caption("Governed agentic analytics: deterministic computation, evidence-backed answers, human oversight.")
    be = backend()

    with st.sidebar:
        st.header("Dataset")
        upload = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if upload is not None and st.session_state.get("uploaded_name") != upload.name:
            try:
                st.session_state.dataset = be.upload(upload.name, upload.getvalue())
                st.session_state.uploaded_name = upload.name
            except ValueError as e:
                st.error(str(e))
        if st.button("Use example sales dataset", use_container_width=True):
            st.session_state.dataset = be.example()
            st.session_state.uploaded_name = None
        st.divider()
        st.header("Example questions")
        for q in EXAMPLES:
            if st.button(q, use_container_width=True):
                st.session_state.question = q
        s = get_settings()
        st.divider()
        st.caption(f"LLM: {s.llm_provider} · backend: {s.ui_backend} · clearance: {s.default_user_clearance}")

    info: DatasetInfo | None = st.session_state.get("dataset")
    if info is None:
        st.info("Upload a dataset or click **Use example sales dataset** in the sidebar to begin.")
        return
    show_profile(info)

    question = st.text_area("Ask a question", key="question", height=80,
                            placeholder="Why did European revenue decline in Q3?")
    if st.button("Start Investigation", type="primary", disabled=not (question or "").strip()):
        with st.spinner("Investigating..."):
            try:
                st.session_state.view = be.start(info.dataset_id, question)
            except ValueError as e:
                st.error(str(e))
    view: InvestigationView | None = st.session_state.get("view")
    if view is not None:
        show_view(view, be)


main()
