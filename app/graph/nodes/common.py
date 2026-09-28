"""Helpers shared by nodes: tool context/gateway construction and LLM budget handling."""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

from app.graph.deps import Dependencies
from app.graph.state import InvestigationState
from app.llm.client import LLMClient, LLMError, Usage
from app.tools.gateway import ToolGateway
from app.tools.registry import ToolContext

log = logging.getLogger(__name__)
T = TypeVar("T")


def tool_context(state: InvestigationState, deps: Dependencies) -> ToolContext:
    engine = deps.engines.get(state["investigation_id"], state["dataset_path"])
    return ToolContext(engine=engine, knowledge=deps.knowledge, sandbox=deps.sandbox,
                       clearance=state.get("clearance", deps.settings.default_user_clearance),
                       step_results=dict(state.get("step_results") or {}))


def gateway(state: InvestigationState, deps: Dependencies) -> ToolGateway:
    return ToolGateway(tool_context(state, deps), max_calls=deps.settings.max_tool_calls,
                       calls_made=sum(1 for r in state.get("tool_calls", []) if r.status in ("ok", "error", "timeout")))


def active_llm(state: InvestigationState, deps: Dependencies) -> LLMClient | None:
    """The LLM, unless the investigation's token budget is spent."""
    if deps.llm is None or state.get("token_usage", 0) >= deps.settings.token_budget:
        return None
    return deps.llm


def with_fallback(
    state: InvestigationState, deps: Dependencies, stage: str, contract_id: str,
    llm_call: Callable[[LLMClient], tuple[T, Usage]], fallback: Callable[[], T],
) -> tuple[T, dict]:
    """Run the LLM implementation of a stage; on any LLM failure use the deterministic one.

    Returns the result plus a state update recording which implementation and prompt version ran.
    """
    versions = dict(state.get("prompt_versions") or {})
    llm = active_llm(state, deps)
    if llm is not None:
        try:
            result, usage = llm_call(llm)
            versions[stage] = contract_id
            return result, {"prompt_versions": versions, "token_usage": state.get("token_usage", 0) + usage.total}
        except (LLMError, ValueError) as e:
            log.warning("llm stage failed, using deterministic fallback", extra={"stage": stage, "error": str(e)})
            versions[stage] = f"rule_based_v1 (fallback: {str(e)[:120]})"
            return fallback(), {"prompt_versions": versions, "errors": [f"{stage}: LLM failed, used rule-based fallback: {e}"]}
    versions[stage] = "rule_based_v1"
    return fallback(), {"prompt_versions": versions}
