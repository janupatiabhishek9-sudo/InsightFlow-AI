"""Deterministic SQL generation from declarative AnalysisSpecs, plus exact driver decomposition.

The planner (LLM or rule-based) decides *what* to analyse; this module decides *how* to compute it.
Identifiers come from the validated schema and literals are escaped, so generated SQL is safe
and reproducible. It still passes through the SQL guard like any other query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.data import DatasetSchema
from app.domain.plan import AnalysisSpec
from app.domain.question import Period
from app.tools.duckdb_engine import TABLE, quote

AVERAGED_HINTS = ("price", "discount", "rate", "pct", "percent", "ratio", "margin")
DERIVED_METRICS = {"orders": "COUNT(*)"}


@dataclass(frozen=True)
class MetricDef:
    name: str
    expression: str
    additive: bool  # sums across groups (contribution shares are meaningful)
    description: str


def resolve_metric(name: str, schema: DatasetSchema) -> MetricDef | None:
    if name in DERIVED_METRICS:
        return MetricDef(name, DERIVED_METRICS[name], True, f"{name} = count of order rows")
    col = schema.column(name)
    if col is None or col.kind != "numeric":
        return None
    if any(h in name for h in AVERAGED_HINTS):
        return MetricDef(name, f"AVG({quote(name)})", False, f"average {name}")
    return MetricDef(name, f"SUM({quote(name)})", True, f"sum of {name}")


def date_column(schema: DatasetSchema) -> str | None:
    temporal = schema.by_kind("temporal")
    return next((c for c in temporal if "date" in c), temporal[0] if temporal else None)


def literal(value: Any) -> str:
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def period_predicate(date_col: str, period: Period) -> str:
    return f"({quote(date_col)} >= DATE '{period.start.isoformat()}' AND {quote(date_col)} < DATE '{period.end.isoformat()}')"


def filter_predicate(filters: dict[str, list[str]]) -> str:
    parts = [f"{quote(col)} IN ({', '.join(literal(v) for v in values)})" for col, values in sorted(filters.items()) if values]
    return " AND ".join(parts) if parts else "TRUE"


class SQLBuildError(ValueError):
    pass


class SQLBuilder:
    def __init__(self, schema: DatasetSchema):
        self.schema = schema
        self.date_col = date_column(schema)

    def build(self, spec: AnalysisSpec) -> str:
        metric = resolve_metric(spec.metric, self.schema)
        if metric is None:
            raise SQLBuildError(f"metric '{spec.metric}' is not a numeric column or known derived metric")
        builder = getattr(self, f"_{spec.kind}")
        return builder(spec, metric)

    # ---- helpers ----------------------------------------------------------------------------
    def _need_date(self) -> str:
        if not self.date_col:
            raise SQLBuildError("dataset has no date column, time-based analysis is impossible")
        return self.date_col

    def _base(self, spec: AnalysisSpec, periods: list[Period]) -> str:
        where = [filter_predicate(spec.filters)]
        if periods:
            d = self._need_date()
            where.append("(" + " OR ".join(period_predicate(d, p) for p in periods) + ")")
        return f"base AS (SELECT * FROM {TABLE} WHERE {' AND '.join(where)})"

    def _periods(self, spec: AnalysisSpec) -> tuple[Period, Period]:
        if spec.period is None or spec.comparison_period is None:
            raise SQLBuildError(f"{spec.kind} needs both a period and a comparison period")
        return spec.period, spec.comparison_period

    # ---- analyses ---------------------------------------------------------------------------
    def _period_comparison(self, spec: AnalysisSpec, m: MetricDef) -> str:
        cur, cmp_ = self._periods(spec)
        d = self._need_date()
        pc, pp = period_predicate(d, cur), period_predicate(d, cmp_)
        return (
            f"WITH {self._base(spec, [cur, cmp_])},\n"
            f"agg AS (SELECT {m.expression} FILTER (WHERE {pc}) AS current_value,\n"
            f"               {m.expression} FILTER (WHERE {pp}) AS comparison_value,\n"
            f"               COUNT(*) FILTER (WHERE {pc}) AS current_rows,\n"
            f"               COUNT(*) FILTER (WHERE {pp}) AS comparison_rows FROM base)\n"
            f"SELECT '{cur.label}' AS period, '{cmp_.label}' AS comparison_period, current_value, comparison_value,\n"
            f"       current_value - comparison_value AS abs_change,\n"
            f"       ROUND(100.0 * (current_value - comparison_value) / NULLIF(comparison_value, 0), 2) AS pct_change,\n"
            f"       current_rows, comparison_rows\nFROM agg"
        )

    def _dimension_breakdown(self, spec: AnalysisSpec, m: MetricDef) -> str:
        if not spec.dimension or self.schema.column(spec.dimension) is None:
            raise SQLBuildError(f"unknown breakdown dimension '{spec.dimension}'")
        dim = quote(spec.dimension)
        if spec.comparison_period is None:
            periods = [spec.period] if spec.period else []
            order = "value ASC" if spec.sort == "value_asc" else "value DESC"
            share = "ROUND(100.0 * value / NULLIF(SUM(value) OVER (), 0), 2)" if m.additive else "NULL"
            return (
                f"WITH {self._base(spec, periods)}\n"
                f"SELECT dim_value, value, {share} AS share_pct FROM (\n"
                f"  SELECT {dim} AS dim_value, {m.expression} AS value FROM base GROUP BY 1)\n"
                f"ORDER BY {order}, dim_value"
            )
        cur, cmp_ = self._periods(spec)
        d = self._need_date()
        pc, pp = period_predicate(d, cur), period_predicate(d, cmp_)
        order = {"change_asc": "abs_change ASC", "change_desc": "abs_change DESC"}.get(spec.sort, "ABS(abs_change) DESC")
        share = "ROUND(100.0 * abs_change / NULLIF(SUM(abs_change) OVER (), 0), 2)" if m.additive else "NULL"
        return (
            f"WITH {self._base(spec, [cur, cmp_])},\n"
            f"g AS (SELECT {dim} AS dim_value,\n"
            f"             COALESCE({m.expression} FILTER (WHERE {pc}), 0) AS current_value,\n"
            f"             COALESCE({m.expression} FILTER (WHERE {pp}), 0) AS comparison_value\n"
            f"      FROM base GROUP BY 1),\n"
            f"c AS (SELECT *, current_value - comparison_value AS abs_change FROM g)\n"
            f"SELECT dim_value, current_value, comparison_value, abs_change,\n"
            f"       ROUND(100.0 * abs_change / NULLIF(comparison_value, 0), 2) AS pct_change,\n"
            f"       {share} AS share_of_total_change\n"
            f"FROM c ORDER BY {order}, dim_value"
        )

    def _driver_decomposition(self, spec: AnalysisSpec, m: MetricDef) -> str:
        for col in ("quantity", "unit_price", spec.metric):
            if self.schema.column(col) is None:
                raise SQLBuildError(f"driver decomposition needs column '{col}'")
        cur, cmp_ = self._periods(spec)
        d = self._need_date()
        rev, qty, price = quote(spec.metric), quote("quantity"), quote("unit_price")

        def agg(label: str, p: Period) -> str:
            return (
                f"SELECT '{label}' AS period, COUNT(*) AS orders, SUM({qty}) AS units, SUM({rev}) AS revenue,\n"
                f"       SUM({qty} * {price}) AS gross_revenue FROM base WHERE {period_predicate(d, p)}"
            )

        return f"WITH {self._base(spec, [cur, cmp_])}\n{agg('comparison', cmp_)}\nUNION ALL\n{agg('current', cur)}"

    def _time_trend(self, spec: AnalysisSpec, m: MetricDef) -> str:
        d = self._need_date()
        periods = [p for p in (spec.comparison_period, spec.period) if p]
        return (
            f"WITH {self._base(spec, periods)}\n"
            f"SELECT CAST(date_trunc('month', {quote(d)}) AS DATE) AS month, {m.expression} AS value\n"
            f"FROM base GROUP BY 1 ORDER BY 1"
        )

    def _metric_summary(self, spec: AnalysisSpec, m: MetricDef) -> str:
        periods = [spec.period] if spec.period else []
        return f"WITH {self._base(spec, periods)}\nSELECT {m.expression} AS value, COUNT(*) AS row_count FROM base"

    # ---- independent recalculation (used by the result validator) ---------------------------
    def total(self, metric: str, filters: dict[str, list[str]], period: Period | None) -> str:
        """A deliberately simple, separate query path for cross-checking totals."""
        m = resolve_metric(metric, self.schema)
        if m is None:
            raise SQLBuildError(f"unknown metric {metric}")
        where = filter_predicate(filters)
        if period is not None:
            where += " AND " + period_predicate(self._need_date(), period)
        return f"SELECT {m.expression} AS value FROM {TABLE} WHERE {where}"

    def revenue_definition_check(self, filters: dict[str, list[str]], period: Period | None) -> str | None:
        """Compare the revenue column with its documented definition quantity x unit_price x (1 - discount)."""
        if not all(self.schema.column(c) for c in ("revenue", "quantity", "unit_price", "discount")):
            return None
        where = filter_predicate(filters)
        if period is not None:
            where += " AND " + period_predicate(self._need_date(), period)
        return (
            f"SELECT SUM(revenue) AS reported, SUM(quantity * unit_price * (1 - COALESCE(discount, 0))) AS recomputed "
            f"FROM {TABLE} WHERE {where}"
        )


def decompose_drivers(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Exact revenue bridge: orders, basket size, list price and discount effects.

    revenue = orders x units_per_order x list_price x (1 - effective_discount)
    Effects are computed sequentially so they sum exactly to the total change.
    """
    by = {r["period"]: r for r in rows}
    if not {"current", "comparison"} <= by.keys():
        raise ValueError("decomposition needs rows for both periods")
    c0, c1 = by["comparison"], by["current"]
    o0, o1 = float(c0["orders"] or 0), float(c1["orders"] or 0)
    u0, u1 = float(c0["units"] or 0), float(c1["units"] or 0)
    r0, r1 = float(c0["revenue"] or 0), float(c1["revenue"] or 0)
    g0, g1 = float(c0["gross_revenue"] or 0), float(c1["gross_revenue"] or 0)
    if min(o0, u0, g0, o1, u1, g1) <= 0:
        raise ValueError("cannot decompose: a period has zero orders, units or gross revenue")
    upo0, upo1 = u0 / o0, u1 / o1
    lp0, lp1 = g0 / u0, g1 / u1
    kept0, kept1 = r0 / g0, r1 / g1  # 1 - effective discount
    orders_effect = (o1 - o0) * upo0 * lp0 * kept0
    basket_effect = o1 * (upo1 - upo0) * lp0 * kept0
    list_price_effect = u1 * (lp1 - lp0) * kept0
    discount_effect = u1 * lp1 * (kept1 - kept0)
    return {
        "comparison_revenue": r0,
        "current_revenue": r1,
        "total_change": r1 - r0,
        "orders_comparison": o0,
        "orders_current": o1,
        "units_per_order_comparison": upo0,
        "units_per_order_current": upo1,
        "avg_list_price_comparison": lp0,
        "avg_list_price_current": lp1,
        "effective_discount_pct_comparison": (1 - kept0) * 100,
        "effective_discount_pct_current": (1 - kept1) * 100,
        "orders_effect": orders_effect,
        "basket_size_effect": basket_effect,
        "list_price_effect": list_price_effect,
        "discount_effect": discount_effect,
    }
