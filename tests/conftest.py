"""Shared fixtures. Every test session uses temporary data/trace/vector directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings
from app.datagen import generate
from app.service import InsightFlowService
from app.tools.duckdb_engine import DuckDBEngine
from app.tools.profiler import profile_dataset


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("data")
    (d / "examples").mkdir()
    generate().to_csv(d / "examples" / "sales.csv", index=False)
    return d


@pytest.fixture(scope="session")
def sales_csv(data_dir) -> Path:
    return data_dir / "examples" / "sales.csv"


@pytest.fixture(scope="session")
def settings(data_dir, tmp_path_factory) -> Settings:
    return Settings(
        _env_file=None, llm_provider="rule_based", data_dir=data_dir, knowledge_dir=PROJECT_ROOT / "knowledge",
        vector_db_path=tmp_path_factory.mktemp("vectors"), default_user_clearance="internal", log_level="WARNING",
    )


@pytest.fixture(scope="session")
def service(settings) -> InsightFlowService:
    return InsightFlowService(settings)


@pytest.fixture(scope="session")
def example(service):
    return service.example_dataset()


@pytest.fixture()
def engine(sales_csv):
    e = DuckDBEngine(sales_csv)
    yield e
    e.close()


@pytest.fixture(scope="session")
def profiled(sales_csv):
    e = DuckDBEngine(sales_csv)
    schema, report = profile_dataset(e)
    yield e, schema, report
    e.close()


@pytest.fixture(scope="session")
def catalog(profiled):
    from app.reasoning.catalog import build_catalog

    e, schema, _ = profiled
    return build_catalog(e, schema)
