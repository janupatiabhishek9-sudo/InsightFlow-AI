import pytest

from app.security.sql_guard import SQLGuard
from app.tools.duckdb_engine import DuckDBEngine

guard = SQLGuard({"dataset"})


@pytest.mark.parametrize("sql", [
    "SELECT * FROM dataset",
    "select region, sum(revenue) from dataset group by 1",
    "WITH a AS (SELECT * FROM dataset) SELECT count(*) FROM a",
    "SELECT * FROM dataset WHERE country IN ('Germany', 'France') ORDER BY revenue DESC LIMIT 5",
])
def test_read_queries_allowed(sql):
    assert guard.check_read(sql).allowed


@pytest.mark.parametrize("sql,reason", [
    ("DROP TABLE dataset", "only SELECT"),
    ("DELETE FROM dataset", "only SELECT"),
    ("UPDATE dataset SET revenue = 0", "only SELECT"),
    ("INSERT INTO dataset VALUES (1)", "only SELECT"),
    ("ALTER TABLE dataset ADD COLUMN x INT", "only SELECT"),
    ("TRUNCATE dataset", ""),
    ("CREATE TABLE x AS SELECT 1", "only SELECT"),
    ("ATTACH 'other.db'", "only SELECT"),
    ("COPY dataset TO 'out.csv'", "only SELECT"),
    ("SELECT 1; DROP TABLE dataset", "exactly one statement"),
    ("SELECT * FROM read_csv('C:/secrets.csv')", "not allowed"),
    ("SELECT * FROM 'C:/secrets.csv'", "not allowed"),
    ("SELECT * FROM read_text('/etc/passwd')", "not allowed"),
    ("SELECT getenv('OPENAI_API_KEY')", "getenv"),
    ("SELECT * FROM other_table", "not in the allowed tables"),
    ("SELECT * FROM main.dataset", "schema-qualified"),
    ("SELECT * FROM duckdb_settings()", "not allowed"),
    ("", "empty"),
])
def test_dangerous_queries_blocked(sql, reason):
    res = guard.check_read(sql)
    assert not res.allowed
    assert reason.lower() in res.reason.lower()


def test_write_guard_allows_only_update_delete():
    assert guard.check_write("DELETE FROM dataset WHERE revenue < 0").allowed
    assert guard.check_write("UPDATE dataset SET discount = 0 WHERE discount IS NULL").allowed
    assert not guard.check_write("DROP TABLE dataset").allowed
    assert not guard.check_write("DELETE FROM other").allowed


def test_engine_blocks_external_access_even_if_guard_is_bypassed(sales_csv, tmp_path):
    """Defense in depth: after loading, DuckDB itself refuses file access."""
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n")
    engine = DuckDBEngine(sales_csv)
    with pytest.raises(Exception):
        engine.query(f"SELECT * FROM read_csv('{secret.as_posix()}')")
    with pytest.raises(Exception):
        engine.con.execute("SET enable_external_access = true")
    engine.close()


def test_engine_row_limit_and_truncation(sales_csv):
    engine = DuckDBEngine(sales_csv, max_rows=10)
    res = engine.query("SELECT * FROM dataset")
    assert res.row_count == 10 and res.truncated
    engine.close()
