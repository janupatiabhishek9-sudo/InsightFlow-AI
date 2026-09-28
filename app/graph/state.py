"""Strongly typed investigation state. Values are Pydantic models, not free-form dicts."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from app.domain.data import DataQualityReport, DatasetSchema
from app.domain.evidence import Evidence
from app.domain.governance import HumanDecision, ResultValidation, RiskAssessment, StepResult, ToolCallRecord
from app.domain.plan import InvestigationPlan, PlanValidation
from app.domain.question import QuestionUnderstanding
from app.domain.report import EvaluationResult, Report
from app.reasoning.catalog import DataCatalog

Status = Literal["running", "completed", "needs_clarification", "blocked", "rejected", "failed", "invalid_plan"]


class InvestigationState(TypedDict, total=False):
    # request
    investigation_id: str
    user_question: str
    dataset_path: str
    clearance: str
    deadline: float
    status: Status
    status_reason: str

    # data
    dataset_schema: DatasetSchema
    data_quality_report: DataQualityReport
    catalog: DataCatalog

    # reasoning
    understanding: QuestionUnderstanding
    retrieved_context: list[Evidence]
    quarantined_sources: list[str]
    investigation_plan: InvestigationPlan
    plan_validation: PlanValidation
    plan_revisions: int
    result_retries: int
    extra_limitations: Annotated[list[str], operator.add]

    # execution
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    step_results: dict[int, StepResult]
    evidence: list[Evidence]
    charts: list[dict[str, Any]]
    validation_results: ResultValidation

    # governance
    risk_assessment: RiskAssessment
    human_decision: HumanDecision | None
    approved_steps: list[int]

    # output
    final_report: Report
    evaluation_results: EvaluationResult

    # observability
    model: dict[str, str]
    prompt_versions: dict[str, str]
    token_usage: int
    errors: Annotated[list[str], operator.add]
    trace_events: Annotated[list[dict[str, Any]], operator.add]
