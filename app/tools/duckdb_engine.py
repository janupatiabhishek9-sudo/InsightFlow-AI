"""DuckDB analytical engine: one in-memory database per investigation.

The dataset is loaded into table `dataset`; afterwards external access (files, network,
extensions) is disabled and the configuration locked, so even SQL that slips past the guard
cannot read other files or exfiltrate data.
"""

from __future__ import annotations

import datetime as dt
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from pydantic import BaseModel

TABLE = "dataset"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


class QueryTimeout(RuntimeError):
    pass


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    duration_ms: float


def to_identifier(name: str) -> str:
    """Normalise a column name to a safe snake_case identifier."""
    ident = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower() or "col"
    return f"c_{ident}" if ident[0].isdigit() else ident


def quote(ident: str) -> str:
    if not _IDENT.match(ident):
        raise ValueError(f"unsafe identifier: {ident!r}")
    return f'"{ident}"'


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


class DuckDBEngine:
    def __init__(self, dataset_path: Path, query_timeout: float = 15, max_rows: int = 500, memory_mb: int = 512):
        self.dataset_path = Path(dataset_path)
        self.query_timeout = query_timeout
        self.max_rows = max_rows
        self._lock = threading.Lock()
        self.con = duckdb.connect(":memory:", config={"threads": 2, "memory_limit": f"{memory_mb}MB"})
        self._load()
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    def _load(self) -> None:
        suffix = self.dataset_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"unsupported dataset type {suffix}; expected one of {sorted(SUPPORTED_EXTENSIONS)}")
        if suffix == ".csv":
            rel = self.con.read_csv(str(self.dataset_path))
            rel.create(TABLE)
        else:
            df = pd.read_excel(self.dataset_path)
            self.con.register("_upload", df)
            self.con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM _upload")
            self.con.unregister("_upload")
        for col in self.columns():
            safe = to_identifier(col)
            if safe != col:
                self.con.execute(f'ALTER TABLE {TABLE} RENAME COLUMN "{col.replace(chr(34), "")}" TO {quote(safe)}')
        # Promote VARCHAR date-like columns to DATE when every non-null value parses.
        for name, dtype in self.column_types().items():
            if dtype == "VARCHAR" and "date" in name:
                bad = self.con.execute(
                    f"SELECT count(*) FROM {TABLE} WHERE {quote(name)} IS NOT NULL AND TRY_CAST({quote(name)} AS DATE) IS NULL"
                ).fetchone()[0]
                if bad == 0:
                    self.con.execute(f"ALTER TABLE {TABLE} ALTER {quote(name)} TYPE DATE")

    # ---- metadata ---------------------------------------------------------------------------
    def columns(self) -> list[str]:
        return [r[0] for r in self.con.execute(f"DESCRIBE {TABLE}").fetchall()]

    def column_types(self) -> dict[str, str]:
        return {r[0]: r[1] for r in self.con.execute(f"DESCRIBE {TABLE}").fetchall()}

    def row_count(self) -> int:
        return self.con.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]

    # ---- execution --------------------------------------------------------------------------
    def query(self, sql: str, params: list[Any] | None = None, max_rows: int | None = None) -> QueryResult:
        """Run a read query with timeout and row limit. Callers must guard model-generated SQL first."""
        limit = max_rows or self.max_rows
        start = time.perf_counter()
        with self._lock, self._timeout():
            cur = self.con.execute(sql, params or [])
            columns = [d[0] for d in cur.description]
            raw = cur.fetchmany(limit + 1)
        truncated = len(raw) > limit
        rows = [{c: _jsonable(v) for c, v in zip(columns, r)} for r in raw[:limit]]
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    def execute_write(self, sql: str) -> int:
        """Apply an approved UPDATE/DELETE to this investigation's working copy. Returns affected rows."""
        before = self.row_count()
        with self._lock, self._timeout():
            cur = self.con.execute(sql)
            changed = cur.fetchone()
        after = self.row_count()
        return int(changed[0]) if changed and changed[0] is not None else abs(before - after)

    def dataframe(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        with self._lock, self._timeout():
            return self.con.execute(sql, params or []).df()

    @contextmanager
    def _timeout(self) -> Iterator[None]:
        """Interrupt the query from a timer thread if it runs longer than query_timeout."""
        timer = threading.Timer(self.query_timeout, self.con.interrupt)
        timer.start()
        try:
            yield
        except Exception as exc:
            if not timer.is_alive():  # the timer fired, so the failure is our interrupt
                raise QueryTimeout(f"query exceeded {self.query_timeout}s") from exc
            raise
        finally:
            timer.cancel()

    def close(self) -> None:
        self.con.close()
