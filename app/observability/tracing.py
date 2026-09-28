"""Local trace store (one JSON file per investigation) and node instrumentation.

Trace events live in the graph state (append-only), so they survive human-review interrupts and
are checkpointed together with everything else. LangSmith can be enabled on top via configuration.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from langgraph.errors import GraphInterrupt
from pydantic import BaseModel

from app.config import Settings

log = logging.getLogger(__name__)
TERMINAL_STATUSES = {"completed", "needs_clarification", "blocked", "rejected", "failed", "invalid_plan"}


def traced(name: str) -> Callable:
    """Wrap a graph node: time it, record a trace event, enforce the deadline, never let it crash the graph."""

    def decorator(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        @wraps(fn)
        def wrapper(state: dict) -> dict:
            started = time.perf_counter()
            event: dict[str, Any] = {"node": name, "started_at": datetime.now(timezone.utc).isoformat()}
            deadline = state.get("deadline")
            if deadline and time.time() > deadline and name != "finalize":
                update = {"status": "failed", "status_reason": "execution time budget exceeded",
                          "errors": [f"{name}: execution time budget exceeded"]}
                event.update(status="skipped", detail="deadline exceeded")
            else:
                try:
                    update = fn(state) or {}
                    event["status"] = "ok"
                    if update.get("status") in TERMINAL_STATUSES:
                        event["detail"] = f"status -> {update['status']}: {update.get('status_reason', '')}"
                except GraphInterrupt:
                    raise  # human-review interrupts must propagate to LangGraph
                except Exception as exc:  # noqa: BLE001 - node failures become traceable state, not crashes
                    log.exception("node failed", extra={"node": name})
                    update = {"status": "failed", "status_reason": f"{name} failed: {exc}",
                              "errors": [f"{name}: {type(exc).__name__}: {exc}"]}
                    event.update(status="error", detail=str(exc))
            event["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return {**update, "trace_events": [event]}

        return wrapper

    return decorator


def _dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
    return value


class TraceStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, investigation_id: str) -> Path:
        safe = "".join(ch for ch in investigation_id if ch.isalnum() or ch in "-_")
        return self.directory / f"{safe}.json"

    def save(self, state: dict) -> Path:
        plan = state.get("investigation_plan")
        trace = {
            "investigation_id": state.get("investigation_id"),
            "status": state.get("status"),
            "status_reason": state.get("status_reason"),
            "user_question": state.get("user_question"),
            "model": (state.get("model") or {}).get("provider"),
            "model_version": (state.get("model") or {}).get("model"),
            "prompt_versions": state.get("prompt_versions"),
            "retries": {
                "plan_revisions": state.get("plan_revisions", 0),
                "result_retries": state.get("result_retries", 0),
                "tool_retries": sum(max(getattr(c, "attempts", 1) - 1, 0) for c in state.get("tool_calls", [])),
            },
            "dataset": state.get("dataset_path"),
            "schema": _dump(state.get("dataset_schema")),
            "understanding": _dump(state.get("understanding")),
            "retrieved_documents": _dump(state.get("retrieved_context")),
            "plan": _dump(plan),
            "plan_validation": _dump(state.get("plan_validation")),
            "tool_calls": _dump(state.get("tool_calls", [])),
            "sql": [r.sql for r in (state.get("step_results") or {}).values() if getattr(r, "sql", None)],
            "results": _dump(state.get("step_results")),
            "validation_results": _dump(state.get("validation_results")),
            "risk_decisions": _dump(state.get("risk_assessment")),
            "human_decisions": _dump(state.get("human_decision")),
            "final_report": _dump(state.get("final_report")),
            "evaluation_results": _dump(state.get("evaluation_results")),
            "token_usage": state.get("token_usage", 0),
            "errors": state.get("errors", []),
            "events": state.get("trace_events", []),
            "latency_ms": round(sum(e.get("duration_ms", 0) for e in state.get("trace_events", [])), 2),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.path(trace["investigation_id"] or "unknown")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def load(self, investigation_id: str) -> dict | None:
        p = self.path(investigation_id)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def list(self, limit: int = 50) -> list[dict]:
        files = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        out = []
        for f in files:
            t = json.loads(f.read_text(encoding="utf-8"))
            out.append({k: t.get(k) for k in ("investigation_id", "status", "user_question", "latency_ms", "saved_at")})
        return out


def configure_langsmith(settings: Settings) -> None:
    """LangGraph traces to LangSmith automatically when these variables are set."""
    if settings.enable_langsmith and settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key.get_secret_value())
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
