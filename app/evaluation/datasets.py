"""Golden evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.config import PROJECT_ROOT

DEFAULT_GOLDEN = PROJECT_ROOT / "evaluations" / "datasets" / "golden_questions.json"


class ExpectedUnderstanding(BaseModel):
    intent: str | None = None
    metric: str | None = None
    filters: dict[str, list[str]] | None = None
    period: str | None = None
    comparison_period: str | None = None
    dimensions: list[str] | None = None  # must be a subset of what the system chose


class ExpectedAnswer(BaseModel):
    dimension: str
    top: str = Field(description="Expected top value, or 'oracle' to compute it independently")
    order: Literal["desc", "asc"] = "desc"


class GoldenCase(BaseModel):
    id: str
    category: str
    question: str
    expected_status: str
    expected_behavior: Literal["read_only", "requires_approval", "blocked", "clarification"]
    expected: ExpectedUnderstanding | None = None
    expected_analyses: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    forbidden_sources: list[str] = Field(default_factory=list)
    expected_answer: list[ExpectedAnswer] = Field(default_factory=list)
    hitl_decision: Literal["approve", "reject"] | None = None
    expected_final_status: str | None = None


def load_cases(path: Path = DEFAULT_GOLDEN) -> list[GoldenCase]:
    return [GoldenCase.model_validate(c) for c in json.loads(path.read_text(encoding="utf-8"))]
