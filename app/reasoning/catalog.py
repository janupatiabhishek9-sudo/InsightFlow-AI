"""DataCatalog: the grounded vocabulary of a dataset (columns, category values, date range)."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from app.domain.data import DatasetSchema
from app.tools.duckdb_engine import TABLE, DuckDBEngine, quote
from app.tools.sql_builder import DERIVED_METRICS, date_column, resolve_metric


class DataCatalog(BaseModel):
    schema_: DatasetSchema
    values: dict[str, list[str]]  # categorical column -> distinct values
    date_min: date | None
    date_max: date | None

    @property
    def metrics(self) -> list[str]:
        numeric = [c for c in self.schema_.by_kind("numeric")]
        return numeric + list(DERIVED_METRICS)

    def has_metric(self, name: str) -> bool:
        return resolve_metric(name, self.schema_) is not None

    def has_dimension(self, name: str) -> bool:
        col = self.schema_.column(name)
        return col is not None and col.kind == "categorical"


def build_catalog(engine: DuckDBEngine, schema: DatasetSchema) -> DataCatalog:
    values: dict[str, list[str]] = {}
    for col in schema.by_kind("categorical"):
        rows = engine.con.execute(
            f"SELECT DISTINCT CAST({quote(col)} AS VARCHAR) FROM {TABLE} WHERE {quote(col)} IS NOT NULL ORDER BY 1 LIMIT 200"
        ).fetchall()
        values[col] = [r[0] for r in rows]
    dmin = dmax = None
    dcol = date_column(schema)
    if dcol:
        dmin, dmax = engine.con.execute(f"SELECT CAST(min({quote(dcol)}) AS DATE), CAST(max({quote(dcol)}) AS DATE) FROM {TABLE}").fetchone()
    return DataCatalog(schema_=schema, values=values, date_min=dmin, date_max=dmax)
