"""generate_report, evaluate and finalize nodes."""

from __future__ import annotations

from app.domain.report import EvaluationResult, Report
from app.graph.deps import Dependencies
from app.graph.nodes.common import with_fallback
from app.graph.state import InvestigationState
from app.observability.tracing import TraceStore
from app.prompts.contracts import REPORT_GENERATOR_V1
from app.reasoning.llm_stages import report_with_llm
from app.reasoning.reporting import citations, draft_claims_rule_based, render_markdown
from app.security.output_guard import guard_claims


def generate_report(state: InvestigationState, deps: Dependencies) -> dict:
    u, plan = state["understanding"], state["investigation_plan"]
    evidence = list(state.get("retrieved_context") or []) + list(state.get("evidence") or [])
    extra = list(state.get("extra_limitations") or [])

    # The deterministic draft is always the backbone: every computed fact, contributor and driver.
    base = draft_claims_rule_based(u, plan, state.get("step_results") or {}, evidence,
                                   state["data_quality_report"], state["validation_results"], extra)
    llm_draft, update = with_fallback(
        state, deps, "report_generator", REPORT_GENERATOR_V1.id,
        lambda llm: report_with_llm(llm, u, plan.objective, evidence, base.limitations),
        lambda: None,
    )
    claims, next_analyses, limitations = list(base.claims), list(base.next_analyses), list(base.limitations)
    if llm_draft is not None:
        # The LLM adds interpretation on top of the verified facts; it never replaces them.
        claims += [c for c in llm_draft.claims if c.kind in ("interpretation", "hypothesis")]
        next_analyses = list(dict.fromkeys(llm_draft.next_analyses + next_analyses))[:6]
        limitations += [l for l in llm_draft.limitations if l not in limitations]
    guarded = guard_claims(claims, evidence)
    # The executive finding must itself be a verified fact.
    exec_facts = [c for c in guarded.claims if c.section == "executive" and c.kind == "fact"]
    executive = exec_facts[0].text if exec_facts else base.executive_finding
    limitations += [f"Output guard: {i}" for i in guarded.issues]
    report = Report(
        title=base.title, executive_finding=executive, claims=guarded.claims,
        business_context=citations([e for e in evidence if e.evidence_type == "rag"]),
        limitations=list(dict.fromkeys(limitations)), next_analyses=next_analyses, assumptions=u.assumptions,
    )
    report.markdown = render_markdown(report, evidence)
    return {**update, "final_report": report}


def evaluate(state: InvestigationState, deps: Dependencies) -> dict:
    """Online self-evaluation of this run's trajectory (golden-set evaluation lives in app.evaluation)."""
    report = state["final_report"]
    claims = report.claims
    facts = [c for c in claims if c.kind == "fact"]
    downgraded = [c for c in claims if c.note]
    validation = state["validation_results"]
    calls = state.get("tool_calls", [])
    executed = [c for c in calls if c.status != "denied"]
    metrics = {
        "evidence_coverage": round(sum(1 for c in facts if c.evidence_ids) / len(facts), 3) if facts else 0.0,
        "unsupported_claim_rate": round(len(downgraded) / len(claims), 3) if claims else 0.0,
        "validation_pass_rate": round(sum(c.passed for c in validation.checks) / len(validation.checks), 3) if validation.checks else 1.0,
        "tool_success_rate": round(sum(c.status == "ok" for c in executed) / len(executed), 3) if executed else 1.0,
        "tool_calls": float(len(calls)),
        "denied_tool_calls": float(sum(c.status == "denied" for c in calls)),
        "plan_steps": float(len(state["investigation_plan"].steps)),
        "latency_s": round(sum(e.get("duration_ms", 0) for e in state.get("trace_events", [])) / 1000, 3),
        "hypotheses": float(sum(c.kind == "hypothesis" for c in claims)),
    }
    notes = []
    if metrics["unsupported_claim_rate"] > 0:
        notes.append("some claims were downgraded by the output guard")
    if not validation.passed:
        notes.append("result validation reported errors")
    passed = validation.passed and metrics["evidence_coverage"] == 1.0 and metrics["unsupported_claim_rate"] == 0
    return {"evaluation_results": EvaluationResult(metrics=metrics, passed=passed, notes=notes)}


def finalize(state: InvestigationState, deps: Dependencies) -> dict:
    status = state.get("status", "running")
    update: dict = {}
    if status == "running":
        update = {"status": "completed", "status_reason": "investigation completed"}
    elif "final_report" not in state:
        # Terminated early (clarification, blocked, rejected, invalid): still produce a short, honest report.
        title = {"needs_clarification": "Clarification needed", "blocked": "Request blocked",
                 "rejected": "Action rejected by reviewer", "invalid_plan": "Question cannot be answered",
                 "failed": "Investigation failed"}.get(status, "Investigation stopped")
        u = state.get("understanding")
        report = Report(title=title, executive_finding=state.get("status_reason", ""),
                        assumptions=u.assumptions if u else [],
                        limitations=[i.message for i in state["plan_validation"].issues] if state.get("plan_validation") else [])
        report.markdown = f"# {title}\n\n{report.executive_finding}\n"
        update["final_report"] = report
    TraceStore(deps.settings.trace_dir).save({**state, **update})
    return update
