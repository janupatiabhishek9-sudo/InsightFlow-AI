"""Dataset schema and data-quality models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    kind: str = Field(description="numeric | categorical | temporal | text")
    description: str | None = None


class DatasetSchema(BaseModel):
    table: str
    row_count: int
    columns: list[ColumnInfo]

    def column(self, name: str) -> ColumnInfo | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def by_kind(self, kind: str) -> list[str]:
        return [c.name for c in self.columns if c.kind == kind]


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    kind: str
    missing: int
    distinct: int
    min: float | str | None = None
    max: float | str | None = None
    mean: float | None = None
    median: float | None = None
    outliers: int | None = None
    negative_values: int | None = None
    top_values: list[str] = Field(default_factory=list)


class DataQualityReport(BaseModel):
    row_count: int
    column_count: int
    missing_columns: dict[str, int] = Field(default_factory=dict)
    duplicate_rows: int = 0
    invalid_dates: int = 0
    date_range: tuple[str, str] | None = None
    columns: list[ColumnProfile] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suspicious_text_values: int = Field(0, description="Cells that look like prompt-injection attempts")
    sample_rows: list[dict] = Field(default_factory=list)
