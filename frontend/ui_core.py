"""Shared UI plumbing: backends, cached service/access store, styling and the coupon gate."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.access import AccessStore  # noqa: E402
from app.api.schemas import DatasetInfo, InvestigationView  # noqa: E402
from app.config import get_settings  # noqa: E402

STATUS_STYLE = {"completed": "success", "awaiting_approval": "warning", "needs_clarification": "info",
                "blocked": "error", "rejected": "error", "failed": "error", "invalid_plan": "warning"}
NODE_LABELS = {
    "validate_request": "Checking the request", "profile_dataset": "Profiling the dataset",
    "understand_question": "Understanding the question", "retrieve_context": "Reading business definitions",
    "create_plan": "Planning the investigation", "validate_plan": "Validating the plan",
    "revise_plan": "Revising the plan", "risk_check": "Assessing risk", "human_review": "Waiting for approval",
    "execute_analysis": "Running the analysis", "validate_results": "Double-checking the numbers",
    "generate_report": "Writing the report", "evaluate": "Scoring the investigation", "finalize": "Saving the trace",
}


class LocalBackend:
    """Runs the service in this process (default: one process, lowest RAM)."""

    local = True

    def __init__(self):
        from app.service import InsightFlowService

        self.svc = InsightFlowService()

    def example(self) -> DatasetInfo:
        return self.svc.example_dataset()

    def upload(self, name: str, data: bytes) -> DatasetInfo:
        return self.svc.register_dataset(name, data)

    def start(self, dataset_id: str, question: str, on_progress=None) -> InvestigationView:
        return self.svc.start_investigation(dataset_id, question, on_progress=on_progress)

    def decide(self, inv_id: str, approved: bool, reviewer: str, comment: str, on_progress=None) -> InvestigationView:
        return self.svc.decide(inv_id, approved, reviewer, comment, on_progress=on_progress)

    def trace(self, inv_id: str) -> dict:
        return self.svc.trace(inv_id)


class HttpBackend:
    """Talks to the FastAPI server (UI_BACKEND=http). Progress is shown after completion."""

    local = False

    def __init__(self, url: str, api_key: str | None):
        headers = {"X-API-Key": api_key} if api_key else {}
        self.http = httpx.Client(base_url=url, timeout=180, headers=headers)

    def _ok(self, r: httpx.Response) -> dict:
        if r.status_code >= 400:
            raise ValueError(r.json().get("detail", r.text))
        return r.json()

    def example(self) -> DatasetInfo:
        return DatasetInfo.model_validate(self._ok(self.http.get("/datasets/example")))

    def upload(self, name: str, data: bytes) -> DatasetInfo:
        return DatasetInfo.model_validate(self._ok(self.http.post("/datasets", files={"file": (name, data)})))

    def start(self, dataset_id: str, question: str, on_progress=None) -> InvestigationView:
        return InvestigationView.model_validate(
            self._ok(self.http.post("/investigations", json={"dataset_id": dataset_id, "question": question})))

    def decide(self, inv_id: str, approved: bool, reviewer: str, comment: str, on_progress=None) -> InvestigationView:
        body = {"approved": approved, "reviewer": reviewer, "comment": comment}
        return InvestigationView.model_validate(self._ok(self.http.post(f"/investigations/{inv_id}/decision", json=body)))

    def trace(self, inv_id: str) -> dict:
        return self._ok(self.http.get(f"/investigations/{inv_id}/trace"))


@st.cache_resource(show_spinner=False)
def access_store() -> AccessStore:
    return AccessStore(get_settings())


@st.cache_resource(show_spinner="Starting InsightFlow AI...")
def backend():
    s = get_settings()
    if s.ui_backend == "http":
        return HttpBackend(s.api_url, s.api_key.get_secret_value() if s.api_key else None)
    be = LocalBackend()
    store = access_store()
    be.svc.apply_ai_settings(store.ai_enabled, store.groq_key_override)  # honour the admin's AI switch
    return be


def ai_label(be) -> str:
    if not getattr(be, "local", False):
        return "API server"
    llm = be.svc.deps.llm
    return f"AI: {llm.provider} · {llm.model}" if llm else "AI: offline (rule-based)"


# ---- styling ------------------------------------------------------------------------------------
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.if-hero {border-radius: 18px; padding: 28px 32px; margin-bottom: 18px; color: #fff;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
  box-shadow: 0 10px 30px rgba(79,70,229,.25);}
.if-hero h1 {color: #fff; margin: 0 0 6px 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.5px;}
.if-hero p {margin: 0; opacity: .92; font-size: 1.05rem;}
.if-hero.small {padding: 18px 24px;}
.if-hero.small h1 {font-size: 1.6rem;}
.if-card {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 16px 18px; height: 100%;
  background: rgba(127,127,127,.04);}
.if-card h4 {margin: 0 0 6px 0; font-size: 1.02rem;}
.if-card p {margin: 0; opacity: .8; font-size: .92rem;}
.if-finding {border-left: 5px solid #4f46e5; border-radius: 12px; padding: 16px 20px; margin: 8px 0 14px 0;
  background: rgba(79,70,229,.07); font-size: 1.08rem;}
.if-pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
  background: rgba(79,70,229,.12); color: #4f46e5; margin-right: 6px;}
.if-pill.green {background: rgba(22,163,74,.13); color: #15803d;}
.if-pill.red {background: rgba(220,38,38,.12); color: #b91c1c;}
.if-pill.amber {background: rgba(217,119,6,.14); color: #b45309;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2); border-radius: 12px; padding: 10px 14px;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, small: bool = False) -> None:
    st.markdown(f'<div class="if-hero{" small" if small else ""}"><h1>{html.escape(title)}</h1>'
                f"<p>{html.escape(subtitle)}</p></div>", unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="if-card"><h4>{html.escape(title)}</h4><p>{html.escape(body)}</p></div>',
                unsafe_allow_html=True)


def pill(text: str, tone: str = "") -> str:
    return f'<span class="if-pill {tone}">{html.escape(text)}</span>'


# ---- access gate --------------------------------------------------------------------------------
def gate() -> bool:
    """True if this session may use the app; otherwise renders the landing/lock screen."""
    store = access_store()
    is_admin = st.session_state.get("is_admin", False)
    if not store.project_enabled and not is_admin:
        hero("InsightFlow AI", "Ask your data why - and get answers you can verify.")
        st.warning("🔧 InsightFlow AI is currently switched off by the administrator. Please check back later.")
        return False
    if is_admin and not store.project_enabled:
        st.info("The project is switched OFF for visitors. You can still use it because you are signed in as admin.")
    if not store.coupon_required or is_admin:
        return True

    code = st.session_state.get("coupon")
    if code:
        problem = store.check(code)
        if problem is None:
            return True
        st.session_state.pop("coupon", None)
        st.warning(f"Your access code no longer works: {problem}")

    landing()
    return False


def landing() -> None:
    hero("InsightFlow AI", "Ask your data why - and get answers you can verify.")
    c1, c2, c3 = st.columns(3)
    with c1:
        card("🔎 Evidence-backed", "Every number comes from a real query and links to its source. No guesses.")
    with c2:
        card("🛡️ Governed & safe", "Guarded SQL, a sandbox for code, and risky actions wait for human approval.")
    with c3:
        card("⚡ Plain English", "Upload a CSV or Excel file and ask why a metric changed.")
    st.write("")
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        with st.form("coupon_form", border=True):
            st.markdown("#### 🎟️ Enter your access code")
            code = st.text_input("Access code", placeholder="e.g. PYTHON2026", label_visibility="collapsed")
            if st.form_submit_button("Unlock InsightFlow AI", type="primary", use_container_width=True):
                problem = access_store().redeem(code)
                if problem:
                    st.error(problem)
                else:
                    st.session_state.coupon = code.strip().upper()
                    st.rerun()
        st.caption("Don't have a code? Ask the administrator of this InsightFlow AI instance.")
