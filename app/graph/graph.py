"""The investigation workflow as an explicit LangGraph state machine.

validate_request -> profile_dataset -> understand_question -> retrieve_context -> create_plan
-> validate_plan -(revisable)-> revise_plan -> validate_plan
                 -(valid)-> risk_check -(HIGH)-> human_review -(approve)-> execute_analysis
                                       -(LOW/MEDIUM)-------------------> execute_analysis
-> validate_results -(failed, retries left)-> revise_plan
                    -(ok)-> generate_report -> evaluate -> finalize
Any terminal status (blocked, needs_clarification, invalid_plan, rejected, failed) routes to finalize.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.graph.deps import Dependencies
from app.graph.nodes import execution, governance, intake, planning, reporting
from app.graph.state import InvestigationState
from app.observability.tracing import traced

NODES: dict[str, Callable] = {
    "validate_request": intake.validate_request,
    "profile_dataset": intake.profile_dataset,
    "understand_question": intake.understand_question,
    "retrieve_context": intake.retrieve_context,
    "create_plan": planning.create_plan,
    "validate_plan": planning.validate_plan,
    "revise_plan": planning.revise_plan,
    "risk_check": governance.risk_check,
    "human_review": governance.human_review,
    "execute_analysis": execution.execute_analysis,
    "validate_results": execution.validate_results,
    "generate_report": reporting.generate_report,
    "evaluate": reporting.evaluate,
    "finalize": reporting.finalize,
}


def _stopped(state: InvestigationState) -> bool:
    return state.get("status") not in (None, "running")


def _next(target: str) -> Callable[[InvestigationState], str]:
    return lambda state: "finalize" if _stopped(state) else target


def _after_plan_validation(state: InvestigationState) -> str:
    if _stopped(state):
        return "finalize"
    return "revise_plan" if state["plan_validation"].status == "revisable" else "risk_check"


def _after_risk(state: InvestigationState) -> str:
    if _stopped(state):
        return "finalize"
    return "human_review" if state["risk_assessment"].requires_approval else "execute_analysis"


def _after_results(max_retries: int) -> Callable[[InvestigationState], str]:
    def route(state: InvestigationState) -> str:
        if _stopped(state):
            return "finalize"
        v = state["validation_results"]
        if not v.passed and v.failed_steps and state.get("result_retries", 0) < max_retries:
            return "revise_plan"
        return "generate_report"
    return route


def _checkpointer(db_path: Path | None):
    """Checkpoints with an explicit allow-list of the types that may be deserialized.

    With a path, state is persisted in SQLite, so an investigation paused for human approval
    survives a process restart. Without one, checkpoints live in memory.
    """
    import app.domain.data as data
    import app.domain.evidence as evidence
    import app.domain.governance as governance
    import app.domain.plan as plan
    import app.domain.question as question
    import app.domain.report as report
    import app.reasoning.catalog as catalog

    allowed = [
        (m.__name__, name)
        for m in (data, evidence, governance, plan, question, report, catalog)
        for name, obj in vars(m).items()
        if isinstance(obj, type) and obj.__module__ == m.__name__
    ]
    serde = JsonPlusSerializer(allowed_msgpack_modules=allowed)
    if db_path is None:
        return InMemorySaver(serde=serde)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # One connection shared by the app's threads; SqliteSaver serialises access with its own lock.
    return SqliteSaver(sqlite3.connect(db_path, check_same_thread=False), serde=serde)


def build_graph(deps: Dependencies, checkpointer=None):
    g = StateGraph(InvestigationState)
    for name, fn in NODES.items():
        g.add_node(name, traced(name)(lambda state, fn=fn: fn(state, deps)))

    g.add_edge(START, "validate_request")
    for src, dst in [("validate_request", "profile_dataset"), ("profile_dataset", "understand_question"),
                     ("understand_question", "retrieve_context"), ("retrieve_context", "create_plan"),
                     ("create_plan", "validate_plan"), ("revise_plan", "validate_plan"),
                     ("human_review", "execute_analysis"), ("execute_analysis", "validate_results"),
                     ("generate_report", "evaluate"), ("evaluate", "finalize")]:
        g.add_conditional_edges(src, _next(dst), [dst, "finalize"])
    g.add_conditional_edges("validate_plan", _after_plan_validation, ["revise_plan", "risk_check", "finalize"])
    g.add_conditional_edges("risk_check", _after_risk, ["human_review", "execute_analysis", "finalize"])
    g.add_conditional_edges("validate_results", _after_results(deps.settings.max_retries),
                            ["revise_plan", "generate_report", "finalize"])
    g.add_edge("finalize", END)
    db = deps.settings.checkpoint_db.strip()
    db_path = None if db in ("", "memory") else deps.settings.resolve(Path(db))
    return g.compile(checkpointer=checkpointer or _checkpointer(db_path))
