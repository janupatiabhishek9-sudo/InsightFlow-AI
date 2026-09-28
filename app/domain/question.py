"""Structured understanding of the user's analytical question."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Intent(str, Enum):
    CHANGE_ANALYSIS = "change_analysis"  # why did X change between two periods
    RANKING = "ranking"  # which X had the largest Y
    TREND = "trend"  # how did X evolve over time
    SUMMARY = "summary"  # what was X in a period
    DATA_MODIFICATION = "data_modification"  # change the working data (needs approval)


class Period(BaseModel):
    """Half-open date interval [start, end)."""

    label: str
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> "Period":
        if self.end <= self.start:
            raise ValueError(f"period {self.label} ends before it starts")
        return self


class Entity(BaseModel):
    text: str = Field(description="Text span in the question")
    column: str
    value: str


class ModificationRequest(BaseModel):
    operation: Literal["delete", "update"]
    description: str
    sql: str = Field(description="Proposed data-modifying SQL, applied only to the investigation's working copy")


ComparisonType = Literal["previous_period", "same_period_last_year", "explicit", "none"]


class QuestionUnderstanding(BaseModel):
    intent: Intent
    direction: Literal["decrease", "increase", "any"] = "any"
    metric: str | None = None
    dimensions: list[str] = Field(default_factory=list, description="Breakdown dimensions requested")
    filters: dict[str, list[str]] = Field(default_factory=dict)
    entities: list[Entity] = Field(default_factory=list)
    time_period: Period | None = None
    comparison_period: Period | None = None
    comparison_type: ComparisonType = "none"
    requested_output: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    required_context: list[str] = Field(default_factory=list)
    modification: ModificationRequest | None = None

    @property
    def needs_clarification(self) -> bool:
        return bool(self.ambiguities)
