"""Execute plan steps through the tool gateway and turn results into evidence."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field

from app.domain.data import DatasetSchema
from app.domain.evidence import Evidence
from app.domain.governance import StepResult, ToolCallRecord
from app.domain.plan import InvestigationPlan, PlanStep
from app.tools.gateway import ToolGateway
from app.tools.sql_builder import SQLBuilder, SQLBuildError, decompose_drivers

RATE_METRICS = {"discount"}

# (failing_sql, error) -> repaired SQL or None. Supplied by the LLM layer in LLM mode.
SQLRepair = Callable[[str, str], str | None]


class ExecutionOutcome(BaseModel):
    results: dict[int, StepResult]
    evidence: list[Evidence]
    charts: list[dict[str, Any]] = Field(default_factory=list)
    records: list[ToolCallRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _numbers(prefix: str, row: dict[str, Any]) -> dict[str, float]:
    return {f"{prefix}{k}": float(v) for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _values_for(kind: str | None, rows: list[dict], summary: dict, metric: str | None = None) -> dict[str, float]:
    values = _raw_values(kind, rows, summary)
    if metric in RATE_METRICS:  # rates are reported as percentages; keep that derived form verifiable too
        values.update({f"{k}_pct": v * 100 for k, v in list(values.items()) if abs(v) <= 1})
    return values


def _raw_values(kind: str | None, rows: list[dict], summary: dict) -> dict[str, float]:
    if kind == "dimension_breakdown":
        out: dict[str, float] = {}
        for r in rows:
            out.update(_numbers(f"{r.get('dim_value')}.", r))
        return out
    if kind == "time_trend":
        return {f"{r['month']}.value": float(r["value"]) for r in rows if r.get("value") is not None}
    if kind == "driver_decomposition":
        return {k: float(v) for k, v in summary.items()}
    return _numbers("", rows[0]) if rows else {}


def execute_plan(
    plan: InvestigationPlan,
    schema: DatasetSchema,
    gateway: ToolGateway,
    previous: dict[int, StepResult],
    approved_steps: set[int],
    evidence_start: int = 1,
    max_retries: int = 2,
    repair: SQLRepair | None = None,
) -> ExecutionOutcome:
    """Run every step without a successful result yet. Never raises for step failures."""
    builder = SQLBuilder(schema)
    results = dict(previous)
    gateway.context.step_results = results
    evidence: list[Evidence] = []
    charts: list[dict[str, Any]] = []
    errors: list[str] = []
    counter = [evidence_start]

    def new_evidence(**kw) -> Evidence:
        ev = Evidence(id=f"E{counter[0]}", **kw)
        counter[0] += 1
        evidence.append(ev)
        return ev

    for step in plan.steps:
        if step.step in results and results[step.step].status == "ok":
            continue
        failed_deps = [d for d in step.depends_on if d not in results or results[d].status != "ok"]
        if failed_deps:
            results[step.step] = StepResult(step=step.step, tool=step.tool, status="skipped",
                                            error=f"depends on failed step(s) {failed_deps}")
            continue
        results[step.step] = _run_step(step, builder, gateway, approved_steps, new_evidence, charts, max_retries, repair)
        if results[step.step].status == "error":
            errors.append(f"step {step.step} ({step.tool}): {results[step.step].error}")

    return ExecutionOutcome(results=results, evidence=evidence, charts=charts, records=gateway.records, errors=errors)


def _run_step(step: PlanStep, builder: SQLBuilder, gateway: ToolGateway, approved: set[int],
              new_evidence, charts: list, max_retries: int, repair: SQLRepair | None) -> StepResult:
    kind = step.analysis.kind if step.analysis else None

    if step.tool == "execute_sql":
        try:
            sql = builder.build(step.analysis) if step.analysis else step.sql or ""
        except SQLBuildError as e:
            return StepResult(step=step.step, tool=step.tool, kind=kind, status="error", error=str(e))
        attempts = 0
        while True:
            record, out = gateway.call("execute_sql", {"sql": sql}, step=step.step)
            if out is not None or repair is None or attempts >= max_retries or record.status == "denied":
                break
            attempts += 1  # SQL error -> diagnose & repair (LLM mode) -> guard runs again -> retry
            fixed = repair(sql, record.error or "")
            if not fixed or fixed == sql:
                break
            sql = fixed
        if out is None:
            return StepResult(step=step.step, tool=step.tool, kind=kind, status="error", sql=sql, error=record.error)
        rows = out["rows"]
        summary: dict[str, Any] = rows[0] if kind in ("period_comparison", "metric_summary") and rows else {}
        if kind == "driver_decomposition":
            try:
                summary = decompose_drivers(rows)
            except ValueError as e:
                return StepResult(step=step.step, tool=step.tool, kind=kind, status="error", sql=sql, rows=rows, error=str(e))
        ev = new_evidence(evidence_type="computed", source=f"duckdb_query_{record.call_id}", description=step.purpose,
                          query=sql, result=rows[:25], step=step.step,
                          values=_values_for(kind, rows, summary, step.analysis.metric if step.analysis else None))
        record.evidence_id = ev.id
        return StepResult(step=step.step, tool=step.tool, kind=kind, status="ok", sql=sql, rows=rows,
                          summary=summary, truncated=out["truncated"], evidence_id=ev.id)

    if step.tool == "execute_write_sql":
        record, out = gateway.call("execute_write_sql", {"sql": step.sql or ""}, step=step.step,
                                   step_approved=step.step in approved)
        if out is None:
            return StepResult(step=step.step, tool=step.tool, status="error", sql=step.sql, error=record.error)
        ev = new_evidence(evidence_type="computed", source=f"duckdb_write_{record.call_id}",
                          description=f"{step.purpose} (working copy only)", query=step.sql, result=out,
                          values={k: float(v) for k, v in out.items()}, step=step.step)
        return StepResult(step=step.step, tool=step.tool, status="ok", sql=step.sql, summary=out, evidence_id=ev.id)

    args = dict(step.arguments)
    if step.tool == "run_analysis" and step.code:
        args.setdefault("code", step.code)
    record, out = gateway.call(step.tool, args, step=step.step)
    if out is None:
        return StepResult(step=step.step, tool=step.tool, status="error", error=record.error)

    if step.tool in ("get_kpi_definition", "search_business_docs"):
        ids = []
        for hit in out.get("results", []):
            c = hit["chunk"]
            ev = new_evidence(evidence_type="rag", source=c["metadata"]["source"], description=f"{c['title']} - {c['heading']}",
                              result=c["text"], confidence=min(1.0, 0.5 + hit["score"]), step=step.step)
            ids.append(ev.id)
        return StepResult(step=step.step, tool=step.tool, status="ok", summary={"evidence_ids": ids, **out},
                          evidence_id=ids[0] if ids else None)
    if step.tool == "create_chart":
        charts.append({"step": step.step, "title": out.get("title", ""), "figure_json": out["figure_json"]})
        return StepResult(step=step.step, tool=step.tool, status="ok", summary={"title": out.get("title", "")})
    if step.tool == "run_statistical_test":
        ev = new_evidence(evidence_type="computed", source=f"scipy_{out['test']}_{record.call_id}", description=step.purpose,
                          result=out, values={k: float(v) for k, v in out.items() if isinstance(v, (int, float))}, step=step.step)
        return StepResult(step=step.step, tool=step.tool, status="ok", summary=out, evidence_id=ev.id)

    ev = new_evidence(evidence_type="computed", source=f"{step.tool}_{record.call_id}", description=step.purpose,
                      result=out, values=_numbers("", out) if isinstance(out, dict) else {}, step=step.step)
    return StepResult(step=step.step, tool=step.tool, status="ok", summary=out if isinstance(out, dict) else {"result": out},
                      evidence_id=ev.id)
