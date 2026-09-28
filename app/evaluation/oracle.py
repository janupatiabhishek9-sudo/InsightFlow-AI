"""Independent ground truth computed with pandas (a different engine and code path than DuckDB/SQL)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from app.domain.question import Period


class Oracle:
    def __init__(self, dataset_path: Path):
        self.df = pd.read_csv(dataset_path, parse_dates=["order_date"])

    def _slice(self, filters: dict[str, list[str]], period: Period | None) -> pd.DataFrame:
        d = self.df
        for col, values in filters.items():
            d = d[d[col].isin(values)]
        if period is not None:
            d = d[(d["order_date"] >= pd.Timestamp(period.start)) & (d["order_date"] < pd.Timestamp(period.end))]
        return d

    @staticmethod
    def _agg(d: pd.DataFrame, metric: str) -> float:
        if metric == "orders":
            return float(len(d))
        if metric in ("discount", "unit_price"):
            return float(d[metric].mean())
        return float(d[metric].sum())

    def total(self, metric: str, filters: dict[str, list[str]], period: Period | None) -> float:
        return self._agg(self._slice(filters, period), metric)

    def top_change(self, metric: str, dimension: str, filters: dict[str, list[str]], period: Period, comparison: Period) -> str:
        cur = self._slice(filters, period).groupby(dimension).apply(lambda g: self._agg(g, metric), include_groups=False)
        prev = self._slice(filters, comparison).groupby(dimension).apply(lambda g: self._agg(g, metric), include_groups=False)
        diff = cur.sub(prev, fill_value=0)
        total = diff.sum()
        return str(diff.idxmin() if total < 0 else diff.idxmax())

    def top_level(self, metric: str, dimension: str, filters: dict[str, list[str]], period: Period | None, order: str) -> str:
        levels = self._slice(filters, period).groupby(dimension).apply(lambda g: self._agg(g, metric), include_groups=False)
        return str(levels.idxmin() if order == "asc" else levels.idxmax())


def as_period(value: dict | Period | None) -> Period | None:
    if value is None or isinstance(value, Period):
        return value
    return Period(label=value["label"], start=date.fromisoformat(str(value["start"])), end=date.fromisoformat(str(value["end"])))
