"""Question understanding, planning/plan validation, result validation and the output guard."""

import pytest

from app.domain.evidence import Claim, Evidence
from app.domain.plan import AnalysisSpec, InvestigationPlan, PlanStep
from app.domain.question import Intent
from app.reasoning.execution import execute_plan
from app.reasoning.planner import plan_rule_based, revise_plan, validate_plan
from app.reasoning.understanding import ground, understand_rule_based
from app.reasoning.validation import validate_results
from app.security.output_guard import extract_numbers, guard_claims
from app.tools.gateway import ToolGateway
from app.tools.periods import quarter
from app.tools.registry import ToolContext


# ---- question understanding ------------------------------------------------------------------
def test_understands_the_flagship_question(catalog):
    u = understand_rule_based("Why did European revenue decrease in Q3, and which products contributed most?", catalog)
    assert u.intent == Intent.CHANGE_ANALYSIS and u.direction == "decrease"
    assert u.metric == "revenue" and u.filters == {"region": ["Europe"]}
    assert (u.time_period.label, u.comparison_period.label) == ("Q3 2024", "Q2 2024")
    assert {"country", "product"} <= set(u.dimensions)
    assert "drivers" in u.requested_output and not u.ambiguities


@pytest.mark.parametrize("question", ["Why did it go down?", "Why did Brazil revenue drop in Q3?", "Tell me something"])
def test_ambiguous_questions_ask_for_clarification(catalog, question):
    assert understand_rule_based(question, catalog).ambiguities


def test_driver_question_maps_to_revenue(catalog):
    u = understand_rule_based("Was the decline caused by fewer orders, lower quantity, lower prices, or discounts?", catalog)
    assert u.metric == "revenue" and "drivers" in u.requested_output


def test_modification_request_is_recognised(catalog):
    u = understand_rule_based("Delete rows with negative revenue", catalog)
    assert u.intent == Intent.DATA_MODIFICATION
    assert u.modification.sql == 'DELETE FROM dataset WHERE "revenue" < 0'


def test_grounding_rejects_hallucinated_values(catalog):
    draft = understand_rule_based("Why did European revenue decrease in Q3?", catalog)
    draft = draft.model_copy(update={"metric": "happiness", "filters": {"region": ["Atlantis"]}})
    grounded = ground(draft, catalog)
    assert grounded.metric is None and grounded.filters == {}
    assert any("happiness" in a for a in grounded.ambiguities) and any("Atlantis" in a for a in grounded.ambiguities)


# ---- planning & plan validation --------------------------------------------------------------
def _understanding(catalog, q="Why did European revenue decrease in Q3?"):
    return understand_rule_based(q, catalog)


def test_plan_is_valid_and_bounded(catalog):
    u = _understanding(catalog)
    plan = plan_rule_based(u, max_steps=8)
    kinds = [s.analysis.kind for s in plan.steps if s.analysis]
    assert kinds[0] == "period_comparison" and "driver_decomposition" in kinds
    assert all(s.rationale for s in plan.steps) and len(plan.steps) <= 8
    assert validate_plan(plan, u, catalog, 8).status == "valid"


def test_plan_validation_catches_bad_steps(catalog):
    u = _understanding(catalog)
    q3, q2 = u.time_period, u.comparison_period
    bad = InvestigationPlan(objective="x", steps=[
        PlanStep(step=1, purpose="compare", rationale="r", tool="execute_sql",
                 analysis=AnalysisSpec(kind="period_comparison", metric="revenue", period=q3, comparison_period=q2)),
        PlanStep(step=2, purpose="bad dim", rationale="r", tool="execute_sql",
                 analysis=AnalysisSpec(kind="dimension_breakdown", metric="revenue", dimension="planet", period=q3, comparison_period=q2)),
        PlanStep(step=3, purpose="no such tool", rationale="r", tool="send_email"),
        PlanStep(step=4, purpose="unsafe", rationale="r", tool="execute_sql", sql="SELECT * FROM read_csv('x')"),
    ])
    v = validate_plan(bad, u, catalog, 8)
    assert v.status == "revisable"
    assert {i.code for i in v.errors} >= {"unknown_dimension", "unknown_tool", "unsafe_sql"}
    revised, notes = revise_plan(bad, {i.step for i in v.errors}, 8, "invalid")
    assert [s.step for s in revised.steps] == [1] and len(notes) == 3
    assert validate_plan(revised, u, catalog, 8).status == "valid"


def test_unavailable_period_makes_plan_invalid(catalog):
    u = _understanding(catalog, "Why did European revenue decline in Q3 2019?")
    v = validate_plan(plan_rule_based(u, 8), u, catalog, 8)
    assert v.status == "invalid" and any(i.code == "period_unavailable" for i in v.errors)


def test_write_step_needs_approval_and_destructive_sql_is_blocked(catalog):
    u = _understanding(catalog, "Delete rows with negative revenue")
    v = validate_plan(plan_rule_based(u, 8), u, catalog, 8)
    assert v.status == "valid" and v.approval_required_steps == [1]
    u2 = _understanding(catalog, "Drop the dataset table")
    assert validate_plan(plan_rule_based(u2, 8), u2, catalog, 8).status == "blocked"


# ---- execution + result validation -----------------------------------------------------------
@pytest.fixture()
def executed(profiled, catalog, tmp_path):
    from app.config import PROJECT_ROOT
    from app.rag.embeddings import HashingEmbedder
    from app.rag.retrieval import KnowledgeBase
    from app.security.sandbox import Sandbox

    engine, schema, _ = profiled
    kb = KnowledgeBase(PROJECT_ROOT / "knowledge", tmp_path / "vs", HashingEmbedder())
    u = _understanding(catalog)
    plan = plan_rule_based(u, 8)
    gw = ToolGateway(ToolContext(engine=engine, knowledge=kb, sandbox=Sandbox(tmp_path), clearance="internal"), max_calls=20)
    outcome = execute_plan(plan, schema, gw, {}, set())
    return engine, schema, plan, outcome


def test_execution_produces_evidence_and_passes_validation(executed):
    engine, schema, plan, outcome = executed
    assert not outcome.errors
    assert all(r.status == "ok" for r in outcome.results.values())
    computed = [e for e in outcome.evidence if e.evidence_type == "computed"]
    assert computed and all(e.query or e.source.startswith("scipy") for e in computed)
    v = validate_results(plan, outcome.results, engine, schema)
    assert v.passed, [c for c in v.checks if not c.passed]
    names = {c.name for c in v.checks}
    assert {"independent_recalculation", "aggregation_consistency", "decomposition_reconciles", "percentage_arithmetic"} <= names


def test_validation_detects_tampered_results(executed):
    engine, schema, plan, outcome = executed
    results = {k: v.model_copy(deep=True) for k, v in outcome.results.items()}
    cmp_step = next(s.step for s in plan.steps if s.analysis and s.analysis.kind == "period_comparison")
    results[cmp_step].rows[0]["current_value"] += 1000  # a wrong number from a buggy tool
    v = validate_results(plan, results, engine, schema)
    assert not v.passed and cmp_step in v.failed_steps


# ---- output guard ----------------------------------------------------------------------------
EV = [Evidence(id="E1", evidence_type="computed", source="q1", description="d", values={"pct_change": -27.79, "abs_change": -471866.4}),
      Evidence(id="C1", evidence_type="rag", source="knowledge/business/revenue.md", description="Revenue", result="Revenue = ...")]


def test_output_guard_keeps_supported_facts():
    c = Claim(text="Revenue decreased by 27.8% (change: -$471,866).", kind="fact", section="executive", evidence_ids=["E1"])
    assert guard_claims([c], EV).claims[0].kind == "fact"


def test_output_guard_downgrades_fabricated_numbers_and_uncited_facts():
    wrong = Claim(text="Revenue decreased by 17.4%.", kind="fact", section="executive", evidence_ids=["E1"])
    uncited = Claim(text="Germany caused the decline.", kind="fact", section="contributors")
    out = guard_claims([wrong, uncited], EV)
    assert [c.kind for c in out.claims] == ["hypothesis", "hypothesis"] and out.downgraded == 2


def test_citing_a_document_does_not_launder_a_number():
    laundered = Claim(text="Revenue fell by 99.9%.", kind="fact", section="executive", evidence_ids=["C1"])
    assert guard_claims([laundered], EV).claims[0].kind == "hypothesis"
    doc = [Evidence(id="C2", evidence_type="rag", source="s", description="d", result="more than 500 employees")]
    quoted = Claim(text="Enterprise means more than 500 employees.", kind="fact", section="detail", evidence_ids=["C2"])
    assert guard_claims([quoted], doc).claims[0].kind == "fact"


def test_output_guard_removes_hallucinated_definitions():
    fake = Claim(text="Revenue excludes all software sales.", kind="fact", section="context", evidence_ids=["E1"])
    real = Claim(text="Revenue = quantity x unit price x (1 - discount).", kind="fact", section="context", evidence_ids=["C1"])
    out = guard_claims([fake, real], EV)
    assert out.removed == 1 and [c.text for c in out.claims] == [real.text]


def test_number_extraction_ignores_periods_and_names():
    nums = [n[0] for n in extract_numbers("Monitor 27in fell 12.5% in Q3 2024 (2024-07) to $1.2M")]
    assert nums == [12.5, 1_200_000.0]
    assert quarter(2024, 3).label == "Q3 2024"
