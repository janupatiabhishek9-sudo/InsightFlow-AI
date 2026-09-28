"""Result validation: never let the report blindly trust tool output.

Checks result schema, empty results, missing/impossible values, divide-by-zero, percentage
arithmetic, aggregation consistency across steps, exactness of the driver bridge, time-period
coverage, and independently recalculates critical totals through a separate query path.
"""

from __future__ import annotations

import math

from app.domain.data import DatasetSchema
from app.domain.governance import ResultValidation, StepResult, ValidationCheck
from app.domain.plan import InvestigationPlan
from app.tools.duckdb_engine import DuckDBEngine
from app.tools.sql_builder import SQLBuilder, resolve_metric

EXPECTED_COLUMNS = {
    "period_comparison": {"current_value", "comparison_value", "abs_change", "pct_change", "current_rows", "comparison_rows"},
    "dimension_breakdown": {"dim_value"},
    "driver_decomposition": {"period", "orders", "units", "revenue", "gross_revenue"},
    "time_trend": {"month", "value"},
    "metric_summary": {"value"},
}


def _close(a: float | None, b: float | None, rel: float = 1e-6, abs_: float = 0.01) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def validate_results(
    plan: InvestigationPlan, results: dict[int, StepResult], engine: DuckDBEngine, schema: DatasetSchema,
) -> ResultValidation:
    checks: list[ValidationCheck] = []
    builder = SQLBuilder(schema)

    def check(name: str, passed: bool, detail: str = "", step: int | None = None, severity: str = "error") -> None:
        checks.append(ValidationCheck(name=name, passed=passed, detail=detail, step=step, severity=severity))

    comparison_step = None
    for step in plan.steps:
        res = results.get(step.step)
        if res is None:
            check("step_executed", False, "step was not executed", step.step)
            continue
        if res.status == "error":
            check("step_succeeded", False, res.error or "unknown error", step.step)
            continue
        if res.status == "skipped":
            check("step_succeeded", False, res.error or "skipped", step.step, "warning")
            continue
        a = step.analysis
        if step.tool != "execute_sql" or a is None:
            continue
        cols = set(res.rows[0]) if res.rows else set()
        missing = EXPECTED_COLUMNS[a.kind] - cols
        check("result_schema", not missing or not res.rows, f"missing columns {sorted(missing)}" if missing else "ok", step.step)
        core = a.kind in ("period_comparison", "metric_summary", "time_trend")
        check("non_empty_result", bool(res.rows), "no rows returned" if not res.rows else f"{len(res.rows)} rows",
              step.step, "error" if core else "warning")
        if res.truncated:
            check("complete_result", False, "result was truncated by the row limit; totals may be incomplete", step.step, "warning")
        if not res.rows:
            continue

        if a.kind == "period_comparison":
            comparison_step = (step, res)
            r = res.rows[0]
            check("time_period_coverage", (r["current_rows"] or 0) > 0 and (r["comparison_rows"] or 0) > 0,
                  f"rows: current={r['current_rows']}, comparison={r['comparison_rows']}", step.step)
            if r["current_value"] is None or r["comparison_value"] is None:
                check("values_present", False, "a period has no value", step.step)
                continue
            if r["comparison_value"] == 0:
                check("divide_by_zero", False, "comparison value is 0; percentage change undefined", step.step, "warning")
            else:
                pct = 100 * (r["current_value"] - r["comparison_value"]) / r["comparison_value"]
                check("percentage_arithmetic", _close(pct, r["pct_change"], abs_=0.01), f"recomputed {pct:.2f}% vs {r['pct_change']}%", step.step)
            check("change_arithmetic", _close(r["current_value"] - r["comparison_value"], r["abs_change"]), "", step.step)
            metric = resolve_metric(a.metric, schema)
            if metric and metric.additive and a.metric != "orders":
                if r["current_value"] < 0 or r["comparison_value"] < 0:
                    check("impossible_values", False, f"negative total {a.metric}", step.step, "warning")
            # Independent recalculation through a separate, simpler query.
            for label, period, reported in (("current", a.period, r["current_value"]), ("comparison", a.comparison_period, r["comparison_value"])):
                recomputed = engine.query(builder.total(a.metric, a.filters, period)).rows[0]["value"]
                check("independent_recalculation", _close(recomputed, reported, rel=1e-9),
                      f"{label}: {reported} vs independent {recomputed}", step.step)
            if a.metric == "revenue":
                sql = builder.revenue_definition_check(a.filters, a.period)
                if sql:
                    d = engine.query(sql).rows[0]
                    dev = abs((d["reported"] or 0) - (d["recomputed"] or 0)) / max(abs(d["reported"] or 1), 1)
                    check("revenue_definition_consistency", dev < 0.005,
                          f"revenue column deviates {dev:.3%} from quantity x unit_price x (1 - discount)", step.step, "warning")

        elif a.kind == "dimension_breakdown" and a.comparison_period is not None:
            for r in res.rows:
                if r["comparison_value"]:
                    pct = 100 * r["abs_change"] / r["comparison_value"]
                    if not _close(pct, r["pct_change"], abs_=0.01):
                        check("percentage_arithmetic", False, f"{r['dim_value']}: {pct:.2f} vs {r['pct_change']}", step.step)
                        break
            else:
                check("percentage_arithmetic", True, "all rows", step.step)
            shares = [r["share_of_total_change"] for r in res.rows if r.get("share_of_total_change") is not None]
            if shares and not res.truncated:
                check("shares_sum_to_100", abs(sum(shares) - 100) < 0.5, f"sum of shares {sum(shares):.2f}%", step.step, "warning")

        elif a.kind == "driver_decomposition":
            s = res.summary
            bridge = s["orders_effect"] + s["basket_size_effect"] + s["list_price_effect"] + s["discount_effect"]
            check("decomposition_reconciles", _close(bridge, s["total_change"], abs_=0.05),
                  f"effects sum {bridge:,.2f} vs total change {s['total_change']:,.2f}", step.step)

    # Cross-step aggregation consistency: breakdown totals must equal the period comparison.
    if comparison_step:
        cstep, cres = comparison_step
        crow = cres.rows[0]
        metric = resolve_metric(cstep.analysis.metric, schema)
        for step in plan.steps:
            res = results.get(step.step)
            a = step.analysis
            if not (res and res.status == "ok" and a and res.rows and not res.truncated):
                continue
            same_scope = a.filters == cstep.analysis.filters and a.period == cstep.analysis.period \
                and a.comparison_period == cstep.analysis.comparison_period and a.metric == cstep.analysis.metric
            if a.kind == "dimension_breakdown" and same_scope and metric and metric.additive:
                total_cur = sum(r["current_value"] or 0 for r in res.rows)
                total_cmp = sum(r["comparison_value"] or 0 for r in res.rows)
                ok = _close(total_cur, crow["current_value"], rel=1e-6) and _close(total_cmp, crow["comparison_value"], rel=1e-6)
                check("aggregation_consistency", ok, f"sum over {a.dimension}: {total_cur:,.2f} / {total_cmp:,.2f}", step.step)
            if a.kind == "driver_decomposition" and same_scope:
                check("aggregation_consistency", _close(res.summary["total_change"], crow["abs_change"], rel=1e-6),
                      "decomposition total equals period comparison", step.step)

    failed = sorted({c.step for c in checks if not c.passed and c.severity == "error" and c.step is not None})
    passed = not any(not c.passed and c.severity == "error" for c in checks)
    return ResultValidation(passed=passed, checks=checks, failed_steps=failed)
