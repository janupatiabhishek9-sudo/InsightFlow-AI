"""Investigation plan produced by the planner and checked by the plan validator."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.question import Period

AnalysisKind = Literal[
    "period_comparison",  # metric total in period vs comparison period
    "dimension_breakdown",  # change (or level) of the metric per dimension value
    "driver_decomposition",  # orders / volume / price / discount drivers of a revenue change
    "time_trend",  # metric per month
    "metric_summary",  # metric total for a period
]


class AnalysisSpec(BaseModel):
    """Declarative description of an analysis. SQL is generated from it deterministically."""

    kind: AnalysisKind
    metric: str
    dimension: str | None = None
    filters: dict[str, list[str]] = Field(default_factory=dict)
    period: Period | None = None
    comparison_period: Period | None = None
    top_n: int = Field(10, ge=1, le=50)
    sort: Literal["change_asc", "change_desc", "abs_change_desc", "value_desc", "value_asc"] = "abs_change_desc"


class PlanStep(BaseModel):
    step: int = Field(ge=1)
    purpose: str
    rationale: str = Field(description="Why this step is needed to answer the question")
    tool: str
    analysis: AnalysisSpec | None = None
    sql: str | None = Field(None, description="Explicit SQL (custom analysis or approved modification)")
    code: str | None = Field(None, description="Python for the sandbox (run_analysis only)")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Validated against the tool's input schema")
    depends_on: list[int] = Field(default_factory=list)


class InvestigationPlan(BaseModel):
    objective: str
    steps: list[PlanStep]
    assumptions: list[str] = Field(default_factory=list)


class PlanIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    step: int | None = None


class PlanValidation(BaseModel):
    status: Literal["valid", "revisable", "invalid", "blocked"]
    issues: list[PlanIssue] = Field(default_factory=list)
    approval_required_steps: list[int] = Field(default_factory=list)

    @property
    def errors(self) -> list[PlanIssue]:
        return [i for i in self.issues if i.severity == "error"]
