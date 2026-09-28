"""risk_check and human_review nodes. Risk is decided by policy code, never by the model."""

from __future__ import annotations

from langgraph.types import interrupt

from app.domain.governance import HumanDecision, PendingAction, RiskAssessment
from app.graph.deps import Dependencies
from app.graph.state import InvestigationState
from app.tools.registry import TOOLS

ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def assess_risk(state: InvestigationState) -> RiskAssessment:
    plan, validation, quality = state["investigation_plan"], state["plan_validation"], state["data_quality_report"]
    level, reasons, pending = "LOW", [], []

    def raise_to(new: str) -> None:
        nonlocal level
        if ORDER[new] > ORDER[level]:
            level = new

    for step in plan.steps:
        policy = TOOLS[step.tool].policy
        if step.step in validation.approval_required_steps or policy.requires_approval or not policy.read_only:
            pending.append(PendingAction(
                step=step.step, tool=step.tool,
                operation="Non-read-only SQL" if step.tool == "execute_write_sql" else step.tool,
                command=step.sql, risk="HIGH",
                reason=f"{step.purpose}. The requested operation modifies data (working copy only).",
            ))
            raise_to("HIGH")
        elif policy.risk_level == "MEDIUM":
            reasons.append(f"step {step.step} uses {step.tool} (sandboxed code execution)")
            raise_to("MEDIUM")
    if quality.suspicious_text_values:
        reasons.append(f"dataset contains {quality.suspicious_text_values} instruction-like text values (treated as data)")
        raise_to("MEDIUM")
    if state.get("quarantined_sources"):
        reasons.append("some retrieved documents were quarantined as possible prompt injection")
        raise_to("MEDIUM")
    reasons += [f"approval required for step {p.step}: {p.operation}" for p in pending]
    if not reasons:
        reasons.append("read-only analysis with approved tools only")
    return RiskAssessment(level=level, reasons=reasons, requires_approval=bool(pending), pending_actions=pending)


def risk_check(state: InvestigationState, deps: Dependencies) -> dict:
    risk = assess_risk(state)
    already = set(state.get("approved_steps") or [])
    if risk.requires_approval and all(p.step in already for p in risk.pending_actions):
        risk = risk.model_copy(update={"requires_approval": False,
                                       "reasons": risk.reasons + ["pending actions were already approved"]})
    return {"risk_assessment": risk}


def human_review(state: InvestigationState, deps: Dependencies) -> dict:
    risk = state["risk_assessment"]
    # Pauses the graph. The checkpoint keeps the state until a reviewer resumes with a decision.
    raw = interrupt({
        "type": "approval_required",
        "investigation_id": state["investigation_id"],
        "risk": risk.level,
        "actions": [a.model_dump() for a in risk.pending_actions],
    })
    decision = HumanDecision.model_validate(raw)
    if not decision.approved:
        return {"human_decision": decision, "status": "rejected",
                "status_reason": f"Reviewer '{decision.reviewer}' rejected the action. {decision.comment}".strip()}
    return {"human_decision": decision, "approved_steps": [a.step for a in risk.pending_actions]}
