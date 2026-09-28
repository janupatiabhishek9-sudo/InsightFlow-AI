"""Regression gate: a representative slice of the golden set must keep passing."""

import pytest

from app.evaluation.datasets import load_cases
from app.evaluation.runner import compare, run

pytestmark = pytest.mark.integration
SLICE = {"chg-01", "chg-05", "rank-03", "trend-01", "sum-06", "sec-02", "sec-08", "hitl-01", "hitl-03", "clar-02", "inv-01"}


def test_golden_set_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 50
    assert len({c.id for c in cases}) == len(cases)
    assert {c.category for c in cases} >= {"change_analysis", "ranking", "trend", "summary", "safety", "hitl", "clarification"}


def test_golden_slice_passes(service):
    results, summary = run([c for c in load_cases() if c.id in SLICE], service)
    failed = {r["id"]: r["scores"] for r in results if not r["passed"]}
    assert not failed, failed
    assert summary["numerical_accuracy"] == 1.0 and summary["safety"] == 1.0


def test_regression_detection():
    base = {"numerical_accuracy": 1.0, "unsupported_claim_rate": 0.0, "mean_latency_s": 0.1}
    assert compare({"numerical_accuracy": 1.0, "unsupported_claim_rate": 0.0, "mean_latency_s": 9}, base) == []
    assert compare({"numerical_accuracy": 0.9, "unsupported_claim_rate": 0.1}, base) == [
        "numerical_accuracy: 1.0 -> 0.9", "unsupported_claim_rate: 0.0 -> 0.1"]
