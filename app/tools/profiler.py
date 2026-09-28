"""Deterministic dataset profiling and data-quality checks."""

from __future__ import annotations

from datetime import date

from app.domain.data import ColumnInfo, ColumnProfile, DataQualityReport, DatasetSchema
from app.security.input_guard import scan_text
from app.tools.duckdb_engine import TABLE, DuckDBEngine, quote

NUMERIC_PREFIXES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
                    "UINTEGER", "UBIGINT", "FLOAT", "DOUBLE", "DECIMAL", "REAL")
TEMPORAL_PREFIXES = ("DATE", "TIMESTAMP")
# Columns where negative values are legitimate and should not be flagged.
SIGNED_HINTS = ("profit", "margin", "change", "delta", "diff", "balance")
CATEGORICAL_MAX_DISTINCT = 100


def _kind(name: str, dtype: str, distinct: int, non_null: int) -> str:
    if dtype.startswith(NUMERIC_PREFIXES):
        return "identifier" if name.endswith("_id") else "numeric"
    if dtype.startswith(TEMPORAL_PREFIXES):
        return "temporal"
    if name.endswith("_id") or (non_null > 50 and distinct == non_null):
        return "identifier"
    if distinct <= CATEGORICAL_MAX_DISTINCT or (non_null and distinct / non_null < 0.05):
        return "categorical"
    return "text"


def profile_dataset(engine: DuckDBEngine, descriptions: dict[str, str] | None = None) -> tuple[DatasetSchema, DataQualityReport]:
    descriptions = descriptions or {}
    rows = engine.row_count()
    types = engine.column_types()
    warnings: list[str] = []
    profiles: list[ColumnProfile] = []
    infos: list[ColumnInfo] = []
    missing_columns: dict[str, int] = {}
    invalid_dates = 0
    date_range: tuple[str, str] | None = None
    suspicious = 0

    for name, dtype in types.items():
        col = quote(name)
        missing, distinct = engine.con.execute(
            f"SELECT count(*) - count({col}), count(DISTINCT {col}) FROM {TABLE}"
        ).fetchone()
        kind = _kind(name, dtype, distinct, rows - missing)
        prof = ColumnProfile(name=name, dtype=dtype, kind=kind, missing=missing, distinct=distinct)
        if missing:
            missing_columns[name] = missing
            warnings.append(f"column '{name}' has {missing} missing values")

        if kind == "numeric":
            mn, mx, mean, median, q1, q3, neg = engine.con.execute(
                f"SELECT min({col}), max({col}), avg({col}), median({col}), quantile_cont({col}, 0.25), "
                f"quantile_cont({col}, 0.75), count(*) FILTER (WHERE {col} < 0) FROM {TABLE}"
            ).fetchone()
            prof.min, prof.max = _num(mn), _num(mx)
            prof.mean, prof.median, prof.negative_values = _num(mean), _num(median), neg
            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                prof.outliers = engine.con.execute(
                    f"SELECT count(*) FROM {TABLE} WHERE {col} < ? OR {col} > ?", [q1 - 3 * iqr, q3 + 3 * iqr]
                ).fetchone()[0]
                # A skewed distribution (many "outliers") is normal for sales data; only rare ones are anomalies.
                if prof.outliers and prof.outliers <= 0.01 * rows:
                    warnings.append(f"column '{name}' has {prof.outliers} extreme outliers (beyond 3 x IQR)")
            if neg and not any(h in name for h in SIGNED_HINTS):
                warnings.append(f"{neg} rows have negative {name}")
        elif kind == "temporal":
            mn, mx = engine.con.execute(f"SELECT min({col}), max({col}) FROM {TABLE}").fetchone()
            prof.min, prof.max = str(mn), str(mx)
            bad = engine.con.execute(
                f"SELECT count(*) FROM {TABLE} WHERE {col} < DATE '1990-01-01' OR {col} > ?",
                [date(date.today().year + 1, 12, 31)],
            ).fetchone()[0]
            invalid_dates += bad
            if date_range is None and mn is not None:
                date_range = (str(mn)[:10], str(mx)[:10])
        elif kind == "categorical":
            top = engine.con.execute(
                f"SELECT {col}, count(*) c FROM {TABLE} WHERE {col} IS NOT NULL GROUP BY 1 ORDER BY c DESC LIMIT 10"
            ).fetchall()
            prof.top_values = [str(v) for v, _ in top]
            values = [r[0] for r in engine.con.execute(f"SELECT DISTINCT {col} FROM {TABLE} WHERE {col} IS NOT NULL LIMIT 1000").fetchall()]
            warnings.extend(_categorical_warnings(name, [str(v) for v in values]))

        if dtype == "VARCHAR":
            texts = [r[0] for r in engine.con.execute(f"SELECT DISTINCT {col} FROM {TABLE} WHERE {col} IS NOT NULL LIMIT 2000").fetchall()]
            hits = sum(1 for t in texts if scan_text(str(t)))
            if hits:
                suspicious += hits
                warnings.append(f"column '{name}' contains {hits} value(s) that look like embedded instructions; treated as data only")

        profiles.append(prof)
        infos.append(ColumnInfo(name=name, dtype=dtype, kind=kind, description=descriptions.get(name)))

    distinct_rows = engine.con.execute(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {TABLE})").fetchone()[0]
    duplicates = rows - distinct_rows
    if duplicates:
        warnings.append(f"{duplicates} fully duplicated rows")
    if invalid_dates:
        warnings.append(f"{invalid_dates} dates fall outside the plausible range")

    sample = engine.query(f"SELECT * FROM {TABLE} LIMIT 5").rows
    schema = DatasetSchema(table=TABLE, row_count=rows, columns=infos)
    report = DataQualityReport(
        row_count=rows,
        column_count=len(types),
        missing_columns=missing_columns,
        duplicate_rows=duplicates,
        invalid_dates=invalid_dates,
        date_range=date_range,
        columns=profiles,
        warnings=warnings,
        suspicious_text_values=suspicious,
        sample_rows=sample,
    )
    return schema, report


def _num(v) -> float | None:
    return None if v is None else round(float(v), 4)


def _categorical_warnings(name: str, values: list[str]) -> list[str]:
    out = []
    padded = [v for v in values if v != v.strip()]
    if padded:
        out.append(f"column '{name}' has values with leading/trailing whitespace: {padded[:3]}")
    seen: dict[str, str] = {}
    variants = set()
    for v in values:
        key = v.strip().lower()
        if key in seen and seen[key] != v:
            variants.add(key)
        seen.setdefault(key, v)
    if variants:
        out.append(f"column '{name}' has case/whitespace variants of the same value: {sorted(variants)[:3]}")
    return out
