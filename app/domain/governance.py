"""Tool-call records, validation, risk and human-decision models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class ToolCallRecord(BaseModel):
    call_id: str
    tool: str
    step: int | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "denied", "error", "timeout"]
    error: str | None = None
    duration_ms: float = 0.0
    attempts: int = 1
    evidence_id: str | None = None


class StepResult(BaseModel):
    """Output of one executed plan step."""

    step: int
    tool: str
    kind: str | None = None
    status: Literal["ok", "error", "skipped"]
    sql: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = False
    error: str | None = None
    evidence_id: str | None = None


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    step: int | None = None
    severity: Literal["error", "warning"] = "error"


class ResultValidation(BaseModel):
    passed: bool
    checks: list[ValidationCheck] = Field(default_factory=list)
    failed_steps: list[int] = Field(default_factory=list)


class PendingAction(BaseModel):
    step: int
    tool: str
    operation: str
    command: str | None = None
    reason: str
    risk: RiskLevel


class RiskAssessment(BaseModel):
    level: RiskLevel
    reasons: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    pending_actions: list[PendingAction] = Field(default_factory=list)


class HumanDecision(BaseModel):
    approved: bool
    reviewer: str = "anonymous"
    comment: str = ""
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
