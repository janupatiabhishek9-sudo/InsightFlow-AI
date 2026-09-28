"""LLM implementations of the reasoning stages.

Each function renders a versioned prompt contract, asks for schema-validated JSON, then hands the
result to the same deterministic grounding/validation as the rule-based path. Callers fall back to
the rule-based implementation on any LLMError.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.data import DataQualityReport
from app.domain.evidence import Claim, Evidence
from app.domain.plan import InvestigationPlan
from app.domain.question import ComparisonType, Intent, QuestionUnderstanding
from app.llm.client import LLMClient, Usage
from app.prompts.contracts import (
    PLANNER_V1,
    QUESTION_UNDERSTANDING_V1,
    REPORT_GENERATOR_V1,
    RESULT_VALIDATOR_V1,
    SQL_GENERATION_V1,
)
from app.reasoning.catalog import DataCatalog
from app.reasoning.reporting import ReportDraft
from app.reasoning.understanding import ground
from app.security.input_guard import wrap_untrusted
from app.tools.periods import parse_periods, previous_period, same_period_last_year
from app.tools.registry import TOOLS


MAX_VALUES_PER_EVIDENCE = 40


def schema_context(catalog: DataCatalog, quality: DataQualityReport) -> str:
    cols = [f"- {c.name} ({c.dtype}, {c.kind}){': ' + c.description if c.description else ''}" for c in catalog.schema_.columns]
    values = {k: v[:40] for k, v in catalog.values.items()}
    return "\n".join([
        f"Table `dataset`, {quality.row_count:,} rows, dates {catalog.date_min} to {catalog.date_max}.",
        "Columns:", *cols,
        "Metrics: " + ", ".join(catalog.metrics),
        wrap_untrusted("dataset_category_values", json.dumps(values)),
        "Data-quality warnings: " + "; ".join(quality.warnings[:8]),
    ])


def tools_context() -> str:
    return "\n".join(f"- {t.name}: {t.description} (risk {t.policy.risk_level})" for t in TOOLS.values())


class LLMUnderstanding(BaseModel):
    intent: Intent
    direction: Literal["decrease", "increase", "any"] = "any"
    metric: str | None = None
    dimensions: list[str] = Field(default_factory=list)
    filters: dict[str, list[str]] = Field(default_factory=dict)
    period_mentions: list[str] = Field(default_factory=list)
    comparison_type: ComparisonType = "none"
    requested_output: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)


def understand_with_llm(llm: LLMClient, question: str, catalog: DataCatalog, quality: DataQualityReport) -> tuple[QuestionUnderstanding, Usage]:
    messages = QUESTION_UNDERSTANDING_V1.render(
        output_schema=LLMUnderstanding.model_json_schema(), schema_context=schema_context(catalog, quality),
        task_input=f"Question: {question}",
    )
    draft, usage = llm.structured(messages, LLMUnderstanding)
    periods, notes = parse_periods(" ; ".join(draft.period_mentions), catalog.date_min, catalog.date_max) \
        if catalog.date_min else ([], [])
    period = periods[0] if periods else None
    comparison = None
    if len(periods) > 1:
        comparison = periods[1]
    elif period and draft.comparison_type == "same_period_last_year":
        comparison = same_period_last_year(period)
    elif period and draft.intent == Intent.CHANGE_ANALYSIS:
        comparison = previous_period(period)
    u = QuestionUnderstanding(
        intent=draft.intent, direction=draft.direction, metric=draft.metric, dimensions=draft.dimensions,
        filters=draft.filters, time_period=period, comparison_period=comparison,
        comparison_type=draft.comparison_type if comparison else "none", requested_output=draft.requested_output,
        ambiguities=draft.ambiguities, assumptions=notes,
    )
    return ground(u, catalog), usage


def plan_with_llm(llm: LLMClient, u: QuestionUnderstanding, catalog: DataCatalog, quality: DataQualityReport,
                  rag_text: str) -> tuple[InvestigationPlan, Usage]:
    messages = PLANNER_V1.render(
        output_schema=InvestigationPlan.model_json_schema(), tools=tools_context(),
        schema_context=schema_context(catalog, quality), rag_context=rag_text,
        state=f"Structured question: {u.model_dump_json()}",
    )
    plan, usage = llm.structured(messages, InvestigationPlan)
    # Fill in scope the model omitted from the grounded understanding (never the other way round).
    steps = []
    for s in plan.steps:
        a = s.analysis
        if a is not None:
            a = a.model_copy(update={
                "filters": a.filters or u.filters,
                "period": a.period or u.time_period,
                "comparison_period": a.comparison_period or (u.comparison_period if a.kind != "time_trend" else None),
            })
        steps.append(s.model_copy(update={"analysis": a}))
    return plan.model_copy(update={"steps": steps}), usage


class SQLOut(BaseModel):
    sql: str


def repair_sql_with_llm(llm: LLMClient, catalog: DataCatalog, quality: DataQualityReport, usage: Usage):
    def repair(sql: str, error: str) -> str | None:
        messages = SQL_GENERATION_V1.render(
            output_schema=SQLOut.model_json_schema(), schema_context=schema_context(catalog, quality),
            task_input=f"Failing query:\n{sql}\n\nError:\n{error}",
        )
        out, u = llm.structured(messages, SQLOut)
        usage.prompt_tokens += u.prompt_tokens
        usage.completion_tokens += u.completion_tokens
        return out.sql
    return repair


class ReviewConcern(BaseModel):
    step: int | None = None
    concern: str = Field(max_length=500)


class LLMResultReview(BaseModel):
    concerns: list[ReviewConcern] = Field(default_factory=list, max_length=8)


def review_results_with_llm(llm: LLMClient, u: QuestionUnderstanding, plan: InvestigationPlan,
                            results: list[dict], failed_checks: list[str]) -> tuple[LLMResultReview, Usage]:
    """Semantic review of results (small samples, misleading comparisons). Can only add warnings."""
    messages = RESULT_VALIDATOR_V1.render(
        output_schema=LLMResultReview.model_json_schema(),
        state=f"Question: {u.model_dump_json(include={'intent', 'metric', 'filters'})}\n"
              f"Plan: {[s.purpose for s in plan.steps]}\nDeterministic checks that failed: {failed_checks}",
        task_input="Results (first rows per step):\n" + json.dumps(results, default=str)[:6000],
    )
    return llm.structured(messages, LLMResultReview)


class LLMReport(BaseModel):
    executive_finding: str
    claims: list[Claim]
    limitations: list[str] = Field(default_factory=list)
    next_analyses: list[str] = Field(default_factory=list)


def report_with_llm(llm: LLMClient, u: QuestionUnderstanding, objective: str, evidence: list[Evidence],
                    limitations: list[str]) -> tuple[ReportDraft, Usage]:
    # Keep prompts small (free-tier token limits): breakdown rows are sorted, so the first values
    # hold the main contributors. Claims citing a dropped number are caught by the output guard.
    ev = [{"id": e.id, "type": e.evidence_type, "description": e.description,
           "values": dict(list(e.values.items())[:MAX_VALUES_PER_EVIDENCE]),
           "excerpt": str(e.result)[:400] if e.evidence_type == "rag" else None} for e in evidence]
    messages = REPORT_GENERATOR_V1.render(
        output_schema=LLMReport.model_json_schema(),
        rag_context=wrap_untrusted("retrieved_documents", json.dumps([x for x in ev if x["type"] == "rag"])),
        state=f"Question: {u.model_dump_json(include={'intent', 'metric', 'filters', 'direction'})}\n"
              f"Known limitations: {limitations}",
        task_input="Evidence:\n" + json.dumps([x for x in ev if x["type"] != "rag"], default=str),
    )
    out, usage = llm.structured(messages, LLMReport)
    return ReportDraft(title=f"Investigation: {objective}", executive_finding=out.executive_finding,
                       claims=out.claims, limitations=limitations + out.limitations, next_analyses=out.next_analyses), usage
