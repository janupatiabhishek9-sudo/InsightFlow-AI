"""execute_analysis and validate_results nodes."""

from __future__ import annotations

from app.domain.governance import ValidationCheck
from app.graph.deps import Dependencies
from app.graph.nodes.common import active_llm, gateway
from app.graph.state import InvestigationState
from app.llm.client import LLMError, Usage
from app.prompts.contracts import RESULT_VALIDATOR_V1, SQL_GENERATION_V1
from app.reasoning.execution import execute_plan
from app.reasoning.llm_stages import repair_sql_with_llm, review_results_with_llm
from app.reasoning.validation import validate_results as check_results


def execute_analysis(state: InvestigationState, deps: Dependencies) -> dict:
    gw = gateway(state, deps)
    llm = active_llm(state, deps)
    usage = Usage()
    repair = repair_sql_with_llm(llm, state["catalog"], state["data_quality_report"], usage) if llm else None
    evidence = list(state.get("evidence") or [])
    outcome = execute_plan(
        state["investigation_plan"], state["dataset_schema"], gw, dict(state.get("step_results") or {}),
        set(state.get("approved_steps") or []), evidence_start=len(evidence) + 1,
        max_retries=deps.settings.max_retries, repair=repair,
    )
    update = {
        "step_results": outcome.results,
        "evidence": evidence + outcome.evidence,
        "charts": list(state.get("charts") or []) + outcome.charts,
        "tool_calls": outcome.records,
        "errors": outcome.errors,
    }
    if repair is not None and usage.total:
        versions = dict(state.get("prompt_versions") or {})
        versions["sql_generation"] = SQL_GENERATION_V1.id
        update.update(prompt_versions=versions, token_usage=state.get("token_usage", 0) + usage.total)
    return update


def validate_results(state: InvestigationState, deps: Dependencies) -> dict:
    engine = deps.engines.get(state["investigation_id"], state["dataset_path"])
    plan = state["investigation_plan"]
    v = check_results(plan, state["step_results"], engine, state["dataset_schema"])
    update: dict = {"validation_results": v}
    llm = active_llm(state, deps)
    if llm is None:
        return update
    # Optional semantic review. Deterministic checks stay authoritative: the model can only add warnings.
    results = [{"step": r.step, "kind": r.kind, "status": r.status, "rows": r.rows[:5]}
               for r in state["step_results"].values()]
    failed = [f"{c.name}: {c.detail}" for c in v.checks if not c.passed]
    versions = dict(state.get("prompt_versions") or {})
    try:
        review, usage = review_results_with_llm(llm, state["understanding"], plan, results, failed)
    except (LLMError, ValueError) as e:
        versions["result_validator"] = "deterministic_only (LLM review failed)"
        return {**update, "prompt_versions": versions, "errors": [f"result_validator: {e}"]}
    extra = [ValidationCheck(name="llm_review", passed=False, severity="warning", step=c.step, detail=c.concern)
             for c in review.concerns]
    versions["result_validator"] = RESULT_VALIDATOR_V1.id
    return {"validation_results": v.model_copy(update={"checks": v.checks + extra}), "prompt_versions": versions,
            "token_usage": state.get("token_usage", 0) + usage.total}
