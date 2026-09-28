"""Request/response models shared by the API, the Streamlit UI and the evaluation runner."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import AccessLevel
from app.domain.data import DataQualityReport
from app.domain.evidence import Evidence
from app.domain.governance import HumanDecision, PendingAction, ResultValidation, RiskAssessment, StepResult, ToolCallRecord
from app.domain.plan import InvestigationPlan, PlanValidation
from app.domain.question import QuestionUnderstanding
from app.domain.report import EvaluationResult, Report


class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    path: str
    quality: DataQualityReport


class InvestigationRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=3, max_length=1000)
    # May only *lower* the configured clearance; raising it requires real authentication (future work).
    clearance: AccessLevel | None = None


class DecisionRequest(BaseModel):
    approved: bool
    reviewer: str = Field("anonymous", max_length=100)
    comment: str = Field("", max_length=1000)


ViewStatus = Literal["running", "awaiting_approval", "completed", "needs_clarification", "blocked", "rejected",
                     "failed", "invalid_plan"]


class InvestigationView(BaseModel):
    investigation_id: str
    status: ViewStatus
    status_reason: str = ""
    question: str = ""
    model: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    data_quality: DataQualityReport | None = None
    understanding: QuestionUnderstanding | None = None
    retrieved_context: list[Evidence] = Field(default_factory=list)
    plan: InvestigationPlan | None = None
    plan_validation: PlanValidation | None = None
    risk: RiskAssessment | None = None
    pending_actions: list[PendingAction] = Field(default_factory=list)
    human_decision: HumanDecision | None = None
    step_results: list[StepResult] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    validation: ResultValidation | None = None
    report: Report | None = None
    evaluation: EvaluationResult | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    token_usage: int = 0
    errors: list[str] = Field(default_factory=list)
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
