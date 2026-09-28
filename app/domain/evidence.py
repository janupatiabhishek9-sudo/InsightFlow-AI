"""Evidence and claims: every important statement in a report must point to evidence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EvidenceType = Literal["computed", "rag", "user_provided", "derived", "hypothesis"]
ClaimKind = Literal["fact", "interpretation", "hypothesis"]


class Evidence(BaseModel):
    id: str
    evidence_type: EvidenceType
    source: str = Field(description="e.g. duckdb_query_3, knowledge/business/revenue.md")
    description: str
    query: str | None = None
    result: Any = None
    values: dict[str, float] = Field(default_factory=dict, description="Key numbers used to verify claims")
    step: int | None = None
    confidence: float = Field(1.0, ge=0, le=1)


class Claim(BaseModel):
    text: str
    kind: ClaimKind
    section: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0, le=1)
    note: str | None = Field(None, description="Set by the output guard when a claim is downgraded")
