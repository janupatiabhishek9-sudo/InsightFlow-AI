"""Deterministic statistical tests and chart generation."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from scipy import stats

from app.domain.question import Period
from app.tools.duckdb_engine import TABLE, DuckDBEngine, quote
from app.tools.sql_builder import filter_predicate, period_predicate


def compare_distributions(
    engine: DuckDBEngine,
    metric: str,
    date_col: str,
    filters: dict[str, list[str]],
    period_a: Period,
    period_b: Period,
    test: str = "mann_whitney",
) -> dict[str, Any]:
    """Compare per-row metric values between two periods (e.g. order values Q2 vs Q3)."""

    def values(p: Period) -> list[float]:
        sql = f"SELECT {quote(metric)} FROM {TABLE} WHERE {filter_predicate(filters)} AND {period_predicate(date_col, p)} AND {quote(metric)} IS NOT NULL"
        return [float(r[0]) for r in engine.con.execute(sql).fetchall()]

    a, b = values(period_a), values(period_b)
    if len(a) < 5 or len(b) < 5:
        raise ValueError(f"not enough observations for a test (n_a={len(a)}, n_b={len(b)})")
    if test == "welch_t":
        res = stats.ttest_ind(a, b, equal_var=False)
    else:
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "test": test,
        "metric": metric,
        "period_a": period_a.label,
        "period_b": period_b.label,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": sum(a) / len(a),
        "mean_b": sum(b) / len(b),
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "significant_at_5pct": bool(res.pvalue < 0.05),
    }


def build_chart(rows: list[dict[str, Any]], chart_type: str, x: str, y: str, title: str) -> str:
    """Return a Plotly figure as JSON. Data comes only from executed, validated step results."""
    if not rows or x not in rows[0] or y not in rows[0]:
        raise ValueError(f"chart needs columns '{x}' and '{y}' in the step result")
    xs = [r[x] for r in rows]
    ys = [r[y] for r in rows]
    if chart_type == "line":
        trace = go.Scatter(x=xs, y=ys, mode="lines+markers")
    else:
        colors = ["#c0392b" if (v or 0) < 0 else "#2e86c1" for v in ys]
        trace = go.Bar(x=xs, y=ys, marker_color=colors)
    fig = go.Figure(trace)
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, template="plotly_white", height=380,
                      margin=dict(l=40, r=20, t=50, b=40))
    return fig.to_json()
