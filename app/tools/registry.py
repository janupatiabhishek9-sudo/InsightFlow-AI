"""Tool catalogue: every tool has a description, typed input, policy (permissions, risk, timeout).

The same catalogue backs the in-process gateway and the MCP servers, so there is one source of
truth for what a tool does and who may call it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.config import AccessLevel
from app.domain.data import DataQualityReport, DatasetSchema
from app.domain.governance import StepResult
from app.domain.question import Period
from app.rag.retrieval import KnowledgeBase, SearchResult
from app.security.policies import Permission, ToolPolicy
from app.security.sandbox import Sandbox
from app.security.sql_guard import SQLGuard
from app.tools.duckdb_engine import TABLE, DuckDBEngine, QueryResult, quote
from app.tools.profiler import profile_dataset
from app.tools.sql_builder import filter_predicate
from app.tools.stats_charts import build_chart, compare_distributions

P = Permission


class ToolError(RuntimeError):
    """Raised by tools for expected, user-explainable failures."""


@dataclass
class ToolContext:
    """Resources a tool may touch. Built per investigation; nothing global or mutable is shared."""

    engine: DuckDBEngine
    knowledge: KnowledgeBase
    sandbox: Sandbox
    clearance: AccessLevel
    step_results: dict[int, StepResult] = field(default_factory=dict)

    @property
    def sql_guard(self) -> SQLGuard:
        return SQLGuard({TABLE})


# ---- typed outputs ------------------------------------------------------------------------------
class StatisticsOutput(BaseModel):
    n: int
    mean: float | None = None
    std: float | None = None
    min: float | str | None = None
    median: float | None = None
    max: float | str | None = None


class WriteOutput(BaseModel):
    affected_rows: int
    rows_before: int
    rows_after: int


class AnalysisOutput(BaseModel):
    result: Any = Field(description="JSON-serialisable value assigned to `result` by the sandboxed code")


class ChartOutput(BaseModel):
    figure_json: str = Field(description="Plotly figure serialised as JSON")
    title: str = ""


class StatTestOutput(BaseModel):
    test: Literal["mann_whitney", "welch_t"]
    metric: str
    period_a: str
    period_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    statistic: float
    p_value: float
    significant_at_5pct: bool


class DictionaryOutput(BaseModel):
    columns: dict[str, str]


# ---- typed inputs -------------------------------------------------------------------------------
class NoInput(BaseModel):
    pass


class SQLInput(BaseModel):
    sql: str = Field(description="A single read-only SELECT/WITH query over table `dataset`")


class WriteSQLInput(BaseModel):
    sql: str = Field(description="A single UPDATE or DELETE on table `dataset` (working copy only)")


class StatisticsInput(BaseModel):
    column: str
    filters: dict[str, list[str]] = Field(default_factory=dict)


class AnalysisInput(BaseModel):
    code: str = Field(description="Python using pandas/numpy/scipy; read `data`, assign `result`")
    input_steps: list[int] = Field(default_factory=list, description="Plan steps whose rows become data['step_<n>']")


class ChartInput(BaseModel):
    source_step: int
    chart_type: Literal["bar", "line"] = "bar"
    x: str
    y: str
    title: str = ""


class StatTestInput(BaseModel):
    metric: str
    filters: dict[str, list[str]] = Field(default_factory=dict)
    period_a: Period
    period_b: Period
    test: Literal["mann_whitney", "welch_t"] = "mann_whitney"


class SearchInput(BaseModel):
    query: str = Field(max_length=500)
    k: int = Field(4, ge=1, le=8)


class KPIInput(BaseModel):
    name: str = Field(max_length=100)


# ---- handlers -----------------------------------------------------------------------------------
def _get_schema(ctx: ToolContext, _: NoInput) -> dict:
    schema, _report = profile_dataset(ctx.engine)
    return schema.model_dump()


def _profile(ctx: ToolContext, _: NoInput) -> dict:
    _schema, report = profile_dataset(ctx.engine)
    return report.model_dump()


def _execute_sql(ctx: ToolContext, args: SQLInput) -> dict:
    verdict = ctx.sql_guard.check_read(args.sql)
    if not verdict.allowed:
        raise ToolError(f"SQL guard rejected query: {verdict.reason}")
    return ctx.engine.query(args.sql).model_dump()


def _execute_write_sql(ctx: ToolContext, args: WriteSQLInput) -> dict:
    verdict = ctx.sql_guard.check_write(args.sql)
    if not verdict.allowed:
        raise ToolError(f"SQL guard rejected modification: {verdict.reason}")
    before = ctx.engine.row_count()
    affected = ctx.engine.execute_write(args.sql)
    return {"affected_rows": affected, "rows_before": before, "rows_after": ctx.engine.row_count()}


def _get_statistics(ctx: ToolContext, args: StatisticsInput) -> dict:
    if args.column not in ctx.engine.columns():
        raise ToolError(f"unknown column '{args.column}'")
    col = quote(args.column)
    sql = (f"SELECT count({col}) AS n, avg({col}) AS mean, stddev_samp({col}) AS std, min({col}) AS min, "
           f"median({col}) AS median, max({col}) AS max FROM {TABLE} WHERE {filter_predicate(args.filters)}")
    return ctx.engine.query(sql).rows[0]


def _run_analysis(ctx: ToolContext, args: AnalysisInput) -> dict:
    inputs = {}
    for step in args.input_steps:
        res = ctx.step_results.get(step)
        if res is None or res.status != "ok":
            raise ToolError(f"input step {step} has no successful result")
        inputs[f"step_{step}"] = pd.DataFrame(res.rows)
    out = ctx.sandbox.run(args.code, inputs)
    if not out.ok:
        raise ToolError(out.error or "sandbox failure")
    return {"result": out.result}


def _create_chart(ctx: ToolContext, args: ChartInput) -> dict:
    res = ctx.step_results.get(args.source_step)
    if res is None or res.status != "ok":
        raise ToolError(f"step {args.source_step} has no successful result to chart")
    try:
        return {"figure_json": build_chart(res.rows, args.chart_type, args.x, args.y, args.title), "title": args.title}
    except ValueError as e:
        raise ToolError(str(e)) from e


def _stat_test(ctx: ToolContext, args: StatTestInput) -> dict:
    types = ctx.engine.column_types()
    dcol = next((c for c, t in types.items() if t.startswith(("DATE", "TIMESTAMP"))), None)
    if dcol is None or args.metric not in types:
        raise ToolError("statistical test needs a date column and a valid metric column")
    try:
        return compare_distributions(ctx.engine, args.metric, dcol, args.filters, args.period_a, args.period_b, args.test)
    except ValueError as e:
        raise ToolError(str(e)) from e


def _search_docs(ctx: ToolContext, args: SearchInput) -> dict:
    return ctx.knowledge.search(args.query, ctx.clearance, k=args.k).model_dump()


def _kpi(ctx: ToolContext, args: KPIInput) -> dict:
    return ctx.knowledge.kpi_definition(args.name, ctx.clearance).model_dump()


def _dictionary(ctx: ToolContext, _: NoInput) -> dict:
    return {"columns": ctx.knowledge.data_dictionary(ctx.clearance)}


# ---- catalogue ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolSpec:
    policy: ToolPolicy
    server: Literal["analytics", "python", "knowledge"]
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[ToolContext, Any], dict]

    @property
    def name(self) -> str:
        return self.policy.name


def _spec(name, server, description, io, handler, perms, risk, read_only=True, approval=False, timeout=30):
    policy = ToolPolicy(name=name, permissions=perms, risk_level=risk, read_only=read_only,
                        requires_approval=approval, timeout_seconds=timeout)
    return ToolSpec(policy, server, description, io[0], io[1], handler)


TOOLS: dict[str, ToolSpec] = {
    s.name: s
    for s in [
        _spec("get_schema", "analytics", "Column names, types and kinds of the dataset.",
              (NoInput, DatasetSchema), _get_schema, [P.READ_DATA], "LOW"),
        _spec("profile_dataset", "analytics", "Data-quality profile: missing values, duplicates, outliers, suspicious values.",
              (NoInput, DataQualityReport), _profile, [P.READ_DATA], "LOW"),
        _spec("execute_sql", "analytics", "Run one guarded read-only SQL query on table `dataset`.",
              (SQLInput, QueryResult), _execute_sql, [P.READ_DATA], "LOW"),
        _spec("get_statistics", "analytics", "Summary statistics for one column, optionally filtered.",
              (StatisticsInput, StatisticsOutput), _get_statistics, [P.READ_DATA], "LOW"),
        _spec("execute_write_sql", "analytics", "Apply an UPDATE/DELETE to the investigation's working copy. Needs MODIFY_DATA and human approval.",
              (WriteSQLInput, WriteOutput), _execute_write_sql, [P.MODIFY_DATA], "HIGH", read_only=False, approval=True),
        _spec("run_analysis", "python", "Run guarded Python in an isolated sandbox over previous step results.",
              (AnalysisInput, AnalysisOutput), _run_analysis, [P.RUN_ANALYSIS], "MEDIUM", timeout=40),
        _spec("create_chart", "python", "Build a Plotly chart from a previous step result.",
              (ChartInput, ChartOutput), _create_chart, [P.RUN_ANALYSIS], "LOW"),
        _spec("run_statistical_test", "python", "Compare a metric's distribution between two periods (Mann-Whitney or Welch t).",
              (StatTestInput, StatTestOutput), _stat_test, [P.READ_DATA, P.RUN_ANALYSIS], "LOW"),
        _spec("search_business_docs", "knowledge", "Search business, data and policy documents the user may read.",
              (SearchInput, SearchResult), _search_docs, [P.READ_KNOWLEDGE], "LOW"),
        _spec("get_kpi_definition", "knowledge", "Retrieve the authoritative definition of a KPI.",
              (KPIInput, SearchResult), _kpi, [P.READ_KNOWLEDGE], "LOW"),
        _spec("get_data_dictionary", "knowledge", "Column descriptions from the data dictionary.",
              (NoInput, DictionaryOutput), _dictionary, [P.READ_KNOWLEDGE], "LOW"),
    ]
}
