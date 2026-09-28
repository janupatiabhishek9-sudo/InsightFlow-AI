"""Final report and evaluation models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.evidence import Claim


class ContextCitation(BaseModel):
    source: str
    section: str
    excerpt: str


class Report(BaseModel):
    title: str
    executive_finding: str
    claims: list[Claim] = Field(default_factory=list)
    business_context: list[ContextCitation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_analyses: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    markdown: str = ""


class EvaluationResult(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    passed: bool = True
    notes: list[str] = Field(default_factory=list)
