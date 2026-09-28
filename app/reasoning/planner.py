"""Investigation planning, plan validation and plan repair.

The planner only *proposes* steps. Validation is deterministic and decides whether the plan may
run, must be revised, needs human approval, or is blocked.
"""

from __future__ import annotations

from datetime import timedelta

from app.domain.plan import AnalysisSpec, InvestigationPlan, PlanIssue, PlanStep, PlanValidation
from app.domain.question import Intent, Period, QuestionUnderstanding
from app.reasoning.catalog import DataCatalog
from app.security.code_guard import check_code
from app.security.policies import APPROVAL_GRANTABLE, DEFAULT_GRANTS
from app.security.sql_guard import SQLGuard
from app.tools.duckdb_engine import TABLE
from app.tools.registry import TOOLS
from app.tools.sql_builder import resolve_metric

LABELS = {"country": "country", "product": "product", "region": "region",
          "customer_segment": "customer segment", "product_category": "product category"}


def _label(dim: str) -> str:
    return LABELS.get(dim, dim.replace("_", " "))


class _Steps:
    def __init__(self) -> None:
        self.steps: list[PlanStep] = []

    def add(self, **kw) -> int:
        n = len(self.steps) + 1
        self.steps.append(PlanStep(step=n, **kw))
        return n


def plan_rule_based(u: QuestionUnderstanding, max_steps: int) -> InvestigationPlan:
    s = _Steps()
    metric = u.metric or "revenue"
    base = dict(metric=metric, filters=u.filters)
    scope = ", ".join(v for vals in u.filters.values() for v in vals) or "all markets"

    if u.intent == Intent.DATA_MODIFICATION and u.modification:
        s.add(purpose=u.modification.description, tool="execute_write_sql", sql=u.modification.sql,
              rationale="The user explicitly asked to change the data; applied to this investigation's working copy only.")
        s.add(purpose="Re-profile the working copy after the modification", tool="profile_dataset",
              rationale="Verify the effect of the change on row counts and data quality.", depends_on=[1])
        return InvestigationPlan(objective=f"Apply requested data change: {u.modification.description}", steps=s.steps)

    s.add(purpose=f"Retrieve the authoritative definition of {metric}", tool="get_kpi_definition",
          arguments={"name": metric}, rationale="Use the company's definition instead of assuming one.")

    if u.intent == Intent.CHANGE_ANALYSIS and u.time_period and u.comparison_period:
        cmp = s.add(purpose=f"Compare {metric} in {u.time_period.label} with {u.comparison_period.label} ({scope})",
                    tool="execute_sql",
                    analysis=AnalysisSpec(kind="period_comparison", period=u.time_period,
                                          comparison_period=u.comparison_period, **base),
                    rationale="Establishes whether and by how much the metric changed; every other step explains this number.")
        sort = {"decrease": "change_asc", "increase": "change_desc"}.get(u.direction, "abs_change_desc")
        first_breakdown = None
        for dim in u.dimensions:
            if dim in u.filters and len(u.filters[dim]) == 1:
                continue  # breaking down by a single filtered value is pointless
            n = s.add(purpose=f"Break the {metric} change down by {_label(dim)}", tool="execute_sql",
                      analysis=AnalysisSpec(kind="dimension_breakdown", dimension=dim, period=u.time_period,
                                            comparison_period=u.comparison_period, sort=sort, **base),
                      rationale=f"Locates which {_label(dim)} values account for the change.", depends_on=[cmp])
            first_breakdown = first_breakdown or n
        if metric == "revenue" and "drivers" in u.requested_output:
            s.add(purpose="Decompose the revenue change into orders, basket size, list price and discount effects",
                  tool="execute_sql",
                  analysis=AnalysisSpec(kind="driver_decomposition", period=u.time_period,
                                        comparison_period=u.comparison_period, **base),
                  rationale="Separates fewer orders, smaller orders, lower prices and heavier discounting.", depends_on=[cmp])
            s.add(purpose=f"Test whether per-order {metric} differs between the two periods", tool="run_statistical_test",
                  arguments={"metric": metric, "filters": u.filters, "period_a": u.comparison_period.model_dump(mode="json"),
                             "period_b": u.time_period.model_dump(mode="json"), "test": "mann_whitney"},
                  rationale="Checks whether the change in order values is larger than random variation.")
        if first_breakdown:
            dim = s.steps[first_breakdown - 1].analysis.dimension
            s.add(purpose=f"Chart the {metric} change by {_label(dim)}", tool="create_chart",
                  arguments={"source_step": first_breakdown, "chart_type": "bar", "x": "dim_value", "y": "abs_change",
                             "title": f"{metric.title()} change by {_label(dim)}: {u.time_period.label} vs {u.comparison_period.label}"},
                  rationale="Makes the concentration of the change visible.", depends_on=[first_breakdown])

    elif u.intent == Intent.TREND:
        n = s.add(purpose=f"Monthly {metric} trend ({scope})", tool="execute_sql",
                  analysis=AnalysisSpec(kind="time_trend", period=u.time_period, **base),
                  rationale="Shows how the metric evolved month by month.")
        s.add(purpose=f"Chart the monthly {metric} trend", tool="create_chart",
              arguments={"source_step": n, "chart_type": "line", "x": "month", "y": "value", "title": f"Monthly {metric}"},
              rationale="A line chart is the clearest view of a trend.", depends_on=[n])

    elif u.intent == Intent.RANKING and u.dimensions:
        sort = "value_asc" if "lowest" in u.requested_output else "value_desc"
        first = None
        for dim in u.dimensions:
            n = s.add(purpose=f"Rank {_label(dim)} values by {metric}" + (f" in {u.time_period.label}" if u.time_period else ""),
                      tool="execute_sql",
                      analysis=AnalysisSpec(kind="dimension_breakdown", dimension=dim, period=u.time_period, sort=sort, **base),
                      rationale=f"Directly answers which {_label(dim)} ranks {'lowest' if sort == 'value_asc' else 'highest'}.")
            first = first or n
        s.add(purpose=f"Chart {metric} by {_label(u.dimensions[0])}", tool="create_chart",
              arguments={"source_step": first, "chart_type": "bar", "x": "dim_value", "y": "value", "title": f"{metric.title()} by {_label(u.dimensions[0])}"},
              rationale="Visual comparison of the ranking.", depends_on=[first])

    else:  # summary (and change questions that lack a comparison period)
        s.add(purpose=f"Total {metric}" + (f" in {u.time_period.label}" if u.time_period else "") + f" ({scope})",
              tool="execute_sql", analysis=AnalysisSpec(kind="metric_summary", period=u.time_period, **base),
              rationale="Computes the requested figure deterministically.")
        if u.comparison_period and u.time_period:
            s.add(purpose=f"Compare with {u.comparison_period.label}", tool="execute_sql",
                  analysis=AnalysisSpec(kind="period_comparison", period=u.time_period, comparison_period=u.comparison_period, **base),
                  rationale="Puts the figure in context.")

    steps = s.steps[:max_steps]
    return InvestigationPlan(
        objective=_objective(u, metric, scope),
        steps=steps,
        assumptions=list(u.assumptions),
    )


def _objective(u: QuestionUnderstanding, metric: str, scope: str) -> str:
    if u.intent == Intent.CHANGE_ANALYSIS and u.time_period and u.comparison_period:
        verb = {"decrease": "decline", "increase": "increase"}.get(u.direction, "change")
        return f"Explain the {metric} {verb} for {scope} in {u.time_period.label} vs {u.comparison_period.label}"
    return f"Answer a {u.intent.value.replace('_', ' ')} question about {metric} for {scope}"


# ---- validation -----------------------------------------------------------------------------
def _period_ok(p: Period, catalog: DataCatalog) -> str | None:
    if catalog.date_min is None:
        return "dataset has no date column"
    last_day = p.end - timedelta(days=1)
    if p.end <= catalog.date_min or p.start > catalog.date_max:
        return f"{p.label} is outside the data range {catalog.date_min} to {catalog.date_max}"
    if p.start < catalog.date_min or last_day > catalog.date_max:
        return f"partial:{p.label} is only partially covered by the data"
    return None


def validate_plan(plan: InvestigationPlan, u: QuestionUnderstanding, catalog: DataCatalog, max_steps: int) -> PlanValidation:
    issues: list[PlanIssue] = []
    approval: list[int] = []
    blocked = False
    guard = SQLGuard({TABLE})

    def err(code: str, msg: str, step: int | None = None, severity: str = "error") -> None:
        issues.append(PlanIssue(severity=severity, code=code, message=msg, step=step))

    if not plan.steps:
        err("empty_plan", "the plan has no steps")
    if len(plan.steps) > max_steps:
        err("too_many_steps", f"plan has {len(plan.steps)} steps; the limit is {max_steps}")
    seen_numbers: set[int] = set()
    signatures: set[str] = set()
    for step in plan.steps:
        n = step.step
        if n in seen_numbers:
            err("duplicate_step_number", f"step number {n} used twice", n)
        seen_numbers.add(n)
        if any(d >= n for d in step.depends_on):
            err("bad_dependency", "a step may only depend on earlier steps", n)
        sig = step.model_dump_json(exclude={"step", "purpose", "rationale", "depends_on"})
        if sig in signatures:
            err("redundant_step", "identical to an earlier step (unnecessary tool call)", n, "warning")
        signatures.add(sig)

        spec = TOOLS.get(step.tool)
        if spec is None:
            err("unknown_tool", f"tool '{step.tool}' is not registered", n)
            continue
        ungrantable = [p for p in spec.policy.permissions if p not in DEFAULT_GRANTS and p not in APPROVAL_GRANTABLE]
        if ungrantable:
            blocked = True
            err("unauthorized_tool", f"tool '{step.tool}' needs {[p.value for p in ungrantable]}, which cannot be granted", n)

        if step.tool == "execute_write_sql":
            verdict = guard.check_write(step.sql or "")
            if not verdict.allowed:
                blocked = True
                err("dangerous_sql", f"modification rejected by SQL guard: {verdict.reason}", n)
            else:
                approval.append(n)
        elif step.tool == "execute_sql":
            if step.analysis is None and not step.sql:
                err("missing_query", "execute_sql step has neither an analysis spec nor SQL", n)
            if step.sql and step.analysis is None:
                verdict = guard.check_read(step.sql)
                if not verdict.allowed:
                    err("unsafe_sql", f"SQL guard: {verdict.reason}", n)
        elif step.tool == "run_analysis":
            verdict = check_code(step.code or step.arguments.get("code", ""))
            if not verdict.allowed:
                err("unsafe_code", "; ".join(verdict.violations), n)
        elif step.tool == "create_chart":
            src = step.arguments.get("source_step")
            if not isinstance(src, int) or src >= n or src not in seen_numbers:
                err("bad_chart_source", "chart must reference an earlier step", n)

        a = step.analysis
        if a is not None:
            if resolve_metric(a.metric, catalog.schema_) is None:
                err("unknown_metric", f"metric '{a.metric}' is not defined for this dataset", n)
            if a.dimension and not catalog.has_dimension(a.dimension):
                err("unknown_dimension", f"column '{a.dimension}' is not available", n)
            for col, values in a.filters.items():
                known = set(catalog.values.get(col, []))
                missing = [v for v in values if v not in known]
                if not known or missing:
                    err("unknown_filter", f"filter {col}={missing or values} does not exist in the data", n)
            for p in (a.period, a.comparison_period):
                if p is None:
                    continue
                problem = _period_ok(p, catalog)
                if problem and problem.startswith("partial:"):
                    err("partial_period", problem.split(":", 1)[1], n, "warning")
                elif problem:
                    err("period_unavailable", problem, n)
            if a.kind in ("period_comparison", "driver_decomposition") and (a.period is None or a.comparison_period is None):
                err("missing_period", f"{a.kind} needs both periods", n)

    if u.intent == Intent.CHANGE_ANALYSIS and not any(
        s.analysis and s.analysis.kind == "period_comparison" for s in plan.steps
    ):
        err("insufficient_plan", "a change analysis must compare the two periods")
    for q in u.ambiguities:
        err("unresolved_ambiguity", q)

    errors = [i for i in issues if i.severity == "error"]
    fatal = {"unknown_metric", "period_unavailable", "empty_plan", "unresolved_ambiguity", "missing_period"}
    if blocked:
        status = "blocked"
    elif not errors:
        status = "valid"
    elif any(i.code in fatal and (i.step is None or _is_core(plan, i.step)) for i in errors):
        status = "invalid"
    else:
        status = "revisable"
    return PlanValidation(status=status, issues=issues, approval_required_steps=approval)


def _is_core(plan: InvestigationPlan, step_no: int) -> bool:
    """A step is core if the question cannot be answered without it."""
    step = next((s for s in plan.steps if s.step == step_no), None)
    return bool(step and step.analysis and step.analysis.kind in ("period_comparison", "metric_summary", "time_trend"))


def revise_plan(plan: InvestigationPlan, drop_steps: set[int], max_steps: int, reason: str) -> tuple[InvestigationPlan, list[str]]:
    """Deterministic repair: remove failing/invalid steps and their dependents, then trim.

    Step numbers are kept stable so results of already-executed steps stay attached to them.
    """
    removed = set(drop_steps)
    changed = True
    while changed:  # cascade to dependents and to charts of removed steps
        changed = False
        for s in plan.steps:
            if s.step in removed:
                continue
            deps = set(s.depends_on) | ({s.arguments.get("source_step")} if s.tool == "create_chart" else set())
            if deps & removed:
                removed.add(s.step)
                changed = True
    kept = [s for s in plan.steps if s.step not in removed]
    trimmed = {s.step for s in kept[max_steps:]}
    kept = kept[:max_steps]
    notes = [f"Removed step '{s.purpose}': {reason}" for s in plan.steps if s.step in removed]
    notes += [f"Removed step '{s.purpose}': plan exceeded {max_steps} steps" for s in plan.steps if s.step in trimmed]
    return plan.model_copy(update={"steps": kept}), notes
