"""create_plan, validate_plan and revise_plan nodes."""

from __future__ import annotations

from app.graph.deps import Dependencies
from app.graph.nodes.common import with_fallback
from app.graph.state import InvestigationState
from app.prompts.contracts import PLANNER_V1
from app.reasoning.llm_stages import plan_with_llm
from app.reasoning.planner import plan_rule_based, revise_plan as repair_plan, validate_plan as check_plan
from app.security.input_guard import wrap_untrusted

REPAIRABLE = {"unknown_tool", "unsafe_sql", "unsafe_code", "bad_chart_source", "unknown_dimension", "unknown_filter",
              "redundant_step", "missing_query", "bad_dependency", "duplicate_step_number"}


def create_plan(state: InvestigationState, deps: Dependencies) -> dict:
    u = state["understanding"]
    rag_text = "\n".join(wrap_untrusted(e.source, f"{e.description}: {e.result}") for e in state.get("retrieved_context", []))
    plan, update = with_fallback(
        state, deps, "planner", PLANNER_V1.id,
        lambda llm: plan_with_llm(llm, u, state["catalog"], state["data_quality_report"], rag_text),
        lambda: plan_rule_based(u, deps.settings.max_plan_steps),
    )
    return {**update, "investigation_plan": plan}


def validate_plan(state: InvestigationState, deps: Dependencies) -> dict:
    v = check_plan(state["investigation_plan"], state["understanding"], state["catalog"], deps.settings.max_plan_steps)
    update: dict = {"plan_validation": v}
    reasons = "; ".join(i.message for i in v.errors)
    if v.status == "blocked":
        update.update(status="blocked", status_reason=f"Plan blocked by policy: {reasons}")
    elif v.status == "invalid":
        update.update(status="invalid_plan", status_reason=f"The question cannot be answered with this dataset: {reasons}")
    elif v.status == "revisable" and state.get("plan_revisions", 0) >= deps.settings.max_retries:
        update.update(status="invalid_plan", status_reason=f"Plan still invalid after {deps.settings.max_retries} revisions: {reasons}")
    return update


def revise_plan(state: InvestigationState, deps: Dependencies) -> dict:
    """Two entry points: an invalid plan (drop invalid steps) or failed results (drop failed steps)."""
    plan = state["investigation_plan"]
    validation = state.get("validation_results")
    if validation is not None and not validation.passed and state["plan_validation"].status == "valid":
        drop, reason = set(validation.failed_steps), "its result failed validation"
        counter = {"result_retries": state.get("result_retries", 0) + 1}
    else:
        v = state["plan_validation"]
        drop = {i.step for i in v.errors if i.step is not None and i.code in REPAIRABLE}
        drop |= {i.step for i in v.issues if i.code == "redundant_step" and i.step is not None}
        reason = "it failed plan validation"
        counter = {"plan_revisions": state.get("plan_revisions", 0) + 1}
    revised, notes = repair_plan(plan, drop, deps.settings.max_plan_steps, reason)
    return {**counter, "investigation_plan": revised, "extra_limitations": notes, "validation_results": None}
