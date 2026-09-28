"""Persistence, live progress, API auth/roles, risk engine, Excel upload, sandbox limits, tool retries."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, SecretStr

from app.config import Settings
from app.domain.data import DataQualityReport
from app.domain.plan import InvestigationPlan, PlanStep, PlanValidation
from app.graph.nodes.governance import assess_risk
from app.main import create_app
from app.security.policies import ToolPolicy
from app.security.sandbox import Sandbox
from app.service import InsightFlowService
from app.tools.gateway import ToolGateway
from app.tools.registry import TOOLS, NoInput, ToolContext, ToolSpec

pytestmark = pytest.mark.integration
DELETE_Q = "Delete rows with negative revenue"


# ---- persistence & progress ------------------------------------------------------------------
def test_paused_approval_survives_a_restart(settings, example):
    first = InsightFlowService(settings)
    paused = first.start_investigation(example.dataset_id, DELETE_Q)
    assert paused.status == "awaiting_approval"
    restarted = InsightFlowService(settings)  # new process: fresh engines, same SQLite checkpoints
    assert restarted.get(paused.investigation_id).status == "awaiting_approval"
    done = restarted.decide(paused.investigation_id, True, "carol")
    assert done.status == "completed" and "4 rows affected" in done.report.executive_finding


def test_progress_callback_streams_every_node(service, example):
    events = []
    view = service.start_investigation(example.dataset_id, "What was total revenue in Q3 2024?", on_progress=events.append)
    assert [e["node"] for e in events] == [e["node"] for e in view.trace_events]
    assert events[0]["node"] == "validate_request" and events[-1]["node"] == "finalize"


# ---- API authentication and roles ---------------------------------------------------------------
KEYS = {"viewer": "v" * 20, "analyst": "a" * 20, "approver": "p" * 20}


@pytest.fixture()
def secured(settings):
    s = settings.model_copy(update={"api_keys": SecretStr(
        f"vic:{KEYS['viewer']}:viewer:public,ann:{KEYS['analyst']}:analyst:internal,"
        f"pat:{KEYS['approver']}:approver:internal")})
    with TestClient(create_app(InsightFlowService(s))) as client:
        yield client


def _h(role: str) -> dict:
    return {"X-API-Key": KEYS[role]}


def test_api_requires_a_valid_key(secured):
    assert secured.get("/health").status_code == 200  # liveness stays open
    assert secured.get("/datasets/example").status_code == 401
    assert secured.get("/datasets/example", headers={"X-API-Key": "wrong-key-wrong-key"}).status_code == 401
    assert secured.get("/me", headers=_h("viewer")).json()["name"] == "vic"


def test_roles_are_enforced_and_reviewer_is_the_caller(secured):
    ds = secured.get("/datasets/example", headers=_h("viewer")).json()["dataset_id"]
    body = {"dataset_id": ds, "question": DELETE_Q}
    assert secured.post("/investigations", json=body, headers=_h("viewer")).status_code == 403
    v = secured.post("/investigations", json=body, headers=_h("analyst")).json()
    assert v["status"] == "awaiting_approval"
    url = f"/investigations/{v['investigation_id']}/decision"
    assert secured.post(url, json={"approved": True}, headers=_h("analyst")).status_code == 403
    d = secured.post(url, json={"approved": False, "reviewer": "someone-else"}, headers=_h("approver")).json()
    assert d["status"] == "rejected" and d["human_decision"]["reviewer"] == "pat"  # identity, not the typed name


def test_caller_clearance_caps_retrieval(secured):
    ds = secured.get("/datasets/example", headers=_h("viewer")).json()["dataset_id"]
    body = {"dataset_id": ds, "question": "What was total revenue in Q3 2024?", "clearance": "restricted"}
    assert secured.post("/investigations", json=body, headers=_h("analyst")).status_code == 400


# ---- risk engine ---------------------------------------------------------------------------------
def _risk_state(steps, approval=(), suspicious=0, quarantined=()):
    return {"investigation_plan": InvestigationPlan(objective="x", steps=steps),
            "plan_validation": PlanValidation(status="valid", approval_required_steps=list(approval)),
            "data_quality_report": DataQualityReport(row_count=1, column_count=1, suspicious_text_values=suspicious),
            "quarantined_sources": list(quarantined)}


def test_risk_engine_levels():
    read = PlanStep(step=1, purpose="q", rationale="r", tool="execute_sql", sql="SELECT 1 FROM dataset")
    code = PlanStep(step=2, purpose="c", rationale="r", tool="run_analysis", code="result = 1")
    write = PlanStep(step=3, purpose="d", rationale="r", tool="execute_write_sql", sql="DELETE FROM dataset")
    assert assess_risk(_risk_state([read])).level == "LOW"
    assert assess_risk(_risk_state([read, code])).level == "MEDIUM"
    assert assess_risk(_risk_state([read], suspicious=2)).level == "MEDIUM"
    assert assess_risk(_risk_state([read], quarantined=["x.md"])).level == "MEDIUM"
    high = assess_risk(_risk_state([read, write], approval=[3]))
    assert high.level == "HIGH" and high.requires_approval and high.pending_actions[0].command == "DELETE FROM dataset"


# ---- Excel upload ----------------------------------------------------------------------------------
def test_excel_upload_end_to_end(service, sales_csv, tmp_path):
    xlsx = tmp_path / "sales.xlsx"
    pd.read_csv(sales_csv).head(4000).to_excel(xlsx, index=False)
    info = service.register_dataset("sales.xlsx", xlsx.read_bytes())
    assert info.quality.row_count == 4000
    v = service.start_investigation(info.dataset_id, "What was total revenue in Q1 2023?")
    assert v.status == "completed" and v.validation.passed


# ---- sandbox resource limits ---------------------------------------------------------------------
def test_sandbox_memory_limit(tmp_path):
    out = Sandbox(tmp_path, timeout=60, memory_mb=300).run("import numpy as np\nx = np.ones((200, 1000, 1000))\nresult = 1")
    assert not out.ok and "Memory" in (out.error or "")
    ok = Sandbox(tmp_path, timeout=60, memory_mb=300).run("result = sum(range(10))")
    assert ok.ok and ok.result == 45


# ---- gateway: typed outputs and controlled retry ---------------------------------------------------
def test_every_tool_has_typed_input_and_output():
    for spec in TOOLS.values():
        assert issubclass(spec.input_model, BaseModel) and issubclass(spec.output_model, BaseModel)


class _Out(BaseModel):
    value: int


def _flaky_gateway(engine, tmp_path, fail_times: int, read_only: bool = True, output=None):
    calls = {"n": 0}

    def handler(ctx, args):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise ConnectionError("transient glitch")
        return output if output is not None else {"value": 7}

    spec = ToolSpec(ToolPolicy(name="flaky", permissions=[], risk_level="LOW", read_only=read_only),
                    "analytics", "test tool", NoInput, _Out, handler)
    ctx = ToolContext(engine=engine, knowledge=None, sandbox=Sandbox(tmp_path), clearance="internal")
    return ToolGateway(ctx, max_calls=5, tools={"flaky": spec}, max_retries=2), calls


def test_transient_failures_are_retried_with_backoff(engine, tmp_path):
    gw, calls = _flaky_gateway(engine, tmp_path, fail_times=2)
    rec, out = gw.call("flaky", {})
    assert rec.status == "ok" and rec.attempts == 3 and out == {"value": 7} and gw.calls_made == 1
    gw, _ = _flaky_gateway(engine, tmp_path, fail_times=5)
    rec, out = gw.call("flaky", {})
    assert out is None and rec.attempts == 3 and "after 3 attempt" in rec.error


def test_writes_are_never_retried_and_outputs_are_validated(engine, tmp_path):
    gw, calls = _flaky_gateway(engine, tmp_path, fail_times=1, read_only=False)
    rec, _ = gw.call("flaky", {})
    assert rec.status == "error" and calls["n"] == 1
    gw, _ = _flaky_gateway(engine, tmp_path, fail_times=0, output={"value": "not-a-number"})
    rec, out = gw.call("flaky", {})
    assert out is None and "does not match its schema" in rec.error


def test_settings_default_to_auto_provider():
    assert Settings.model_fields["llm_provider"].default == "auto"
