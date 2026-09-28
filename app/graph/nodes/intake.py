"""validate_request, profile_dataset, understand_question and retrieve_context nodes."""

from __future__ import annotations

import time
from pathlib import Path

from app.domain.data import ColumnInfo, DataQualityReport, DatasetSchema
from app.domain.evidence import Evidence
from app.graph.deps import Dependencies
from app.graph.nodes.common import gateway, with_fallback
from app.graph.state import InvestigationState
from app.llm.client import describe
from app.prompts.contracts import QUESTION_UNDERSTANDING_V2
from app.reasoning.catalog import build_catalog
from app.reasoning.llm_stages import understand_with_llm
from app.reasoning.understanding import grounded_filters, understand_rule_based
from app.security.input_guard import check_user_question
from app.tools.duckdb_engine import SUPPORTED_EXTENSIONS, TABLE
from app.tools.sql_builder import SQLBuilder


def validate_request(state: InvestigationState, deps: Dependencies) -> dict:
    provider, model = describe(deps.llm)
    base = {"status": "running", "deadline": time.time() + deps.settings.max_execution_time,
            "model": {"provider": provider, "model": model}, "token_usage": 0, "plan_revisions": 0, "result_retries": 0,
            "approved_steps": [], "step_results": {}, "evidence": [], "charts": [], "human_decision": None}
    verdict = check_user_question(state.get("user_question", ""))
    if not verdict.allowed:
        return {**base, "status": "blocked", "status_reason": f"Input guardrail: {verdict.reason}",
                "errors": [f"input guard flags: {verdict.flags}"]}
    path = Path(state.get("dataset_path", "")).resolve()
    data_root = deps.settings.resolve(deps.settings.data_dir).resolve()
    if not path.is_file():
        return {**base, "status": "blocked", "status_reason": "dataset file not found"}
    if data_root not in path.parents:
        return {**base, "status": "blocked", "status_reason": "dataset must live inside the configured DATA_DIR"}
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {**base, "status": "blocked", "status_reason": f"unsupported file type {path.suffix}"}
    if path.stat().st_size > deps.settings.max_upload_mb * 1024 * 1024:
        return {**base, "status": "blocked", "status_reason": f"dataset larger than {deps.settings.max_upload_mb} MB"}
    return base


def profile_dataset(state: InvestigationState, deps: Dependencies) -> dict:
    gw = gateway(state, deps)
    _, report = gw.call("profile_dataset", {})
    _, dictionary = gw.call("get_data_dictionary", {})
    if report is None:
        return {"status": "failed", "status_reason": "dataset profiling failed", "tool_calls": gw.records}
    quality = DataQualityReport.model_validate(report)
    descriptions = (dictionary or {}).get("columns", {})
    schema = DatasetSchema(table=TABLE, row_count=quality.row_count, columns=[
        ColumnInfo(name=c.name, dtype=c.dtype, kind=c.kind, description=descriptions.get(c.name)) for c in quality.columns
    ])
    catalog = build_catalog(gw.context.engine, schema)
    return {"dataset_schema": schema, "data_quality_report": quality, "catalog": catalog, "tool_calls": gw.records}


def understand_question(state: InvestigationState, deps: Dependencies) -> dict:
    catalog, quality, question = state["catalog"], state["data_quality_report"], state["user_question"]
    engine = deps.engines.get(state["investigation_id"], state["dataset_path"])
    builder = SQLBuilder(catalog.schema_)

    filters = grounded_filters(question, catalog)

    def probe(cur, prev):
        """Deterministic check so 'the decline' without a period resolves to a real decline in the data."""
        if not catalog.has_metric("revenue"):
            return None
        a = engine.query(builder.total("revenue", filters, cur)).rows[0]["value"]
        b = engine.query(builder.total("revenue", filters, prev)).rows[0]["value"]
        return None if a is None or b is None else a - b

    def rule_based():
        return understand_rule_based(question, catalog, probe)

    u, update = with_fallback(state, deps, "question_understanding", QUESTION_UNDERSTANDING_V2.id,
                              lambda llm: understand_with_llm(llm, question, catalog, quality), rule_based)
    if u.ambiguities:
        return {**update, "understanding": u, "status": "needs_clarification",
                "status_reason": " ".join(u.ambiguities)}
    return {**update, "understanding": u}


def retrieve_context(state: InvestigationState, deps: Dependencies) -> dict:
    u = state["understanding"]
    query = " ".join([state["user_question"], *u.required_context, u.metric or ""])
    gw = gateway(state, deps)
    _, out = gw.call("search_business_docs", {"query": query, "k": 4})
    evidence = []
    for i, hit in enumerate((out or {}).get("results", []), start=1):
        c = hit["chunk"]
        evidence.append(Evidence(id=f"C{i}", evidence_type="rag", source=c["metadata"]["source"],
                                 description=f"{c['title']} - {c['heading']}", result=c["text"],
                                 confidence=min(1.0, 0.5 + hit["score"])))
    quarantined = (out or {}).get("quarantined", [])
    extra = [f"Retrieved document(s) {quarantined} contained instruction-like text and were excluded."] if quarantined else []
    return {"retrieved_context": evidence, "quarantined_sources": quarantined, "tool_calls": gw.records,
            "extra_limitations": extra}
