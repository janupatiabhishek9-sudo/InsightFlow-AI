"""The single choke point for tool execution: lookup -> budget -> authorization -> input
validation -> guarded execution with timeout and controlled retry -> output validation -> audit
record. The model never calls a tool directly."""

from __future__ import annotations

import concurrent.futures as cf
import logging
import time
import uuid
from typing import Any

from pydantic import ValidationError

from app.domain.governance import ToolCallRecord
from app.security.policies import DEFAULT_GRANTS, Permission, authorize
from app.tools.registry import TOOLS, ToolContext, ToolError, ToolSpec

log = logging.getLogger(__name__)
_POOL = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="tool")
BACKOFF_SECONDS = 0.2


class _Transient(Exception):
    """A failure worth retrying (timeout or unexpected error), as opposed to a deliberate ToolError."""


class ToolGateway:
    def __init__(
        self,
        context: ToolContext,
        max_calls: int,
        calls_made: int = 0,
        grants: frozenset[Permission] = DEFAULT_GRANTS,
        tools: dict[str, ToolSpec] | None = None,
        max_retries: int = 2,
    ):
        self.context = context
        self.max_calls = max_calls
        self.calls_made = calls_made
        self.grants = grants
        self.tools = tools if tools is not None else TOOLS
        self.max_retries = max_retries
        self.records: list[ToolCallRecord] = []

    def spec(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def call(
        self, tool: str, arguments: dict[str, Any], step: int | None = None, step_approved: bool = False
    ) -> tuple[ToolCallRecord, dict | None]:
        call_id = uuid.uuid4().hex[:10]
        start = time.perf_counter()
        attempts = 0

        def record(status: str, error: str | None = None) -> ToolCallRecord:
            rec = ToolCallRecord(
                call_id=call_id, tool=tool, step=step, arguments=_redact(arguments), status=status, error=error,
                duration_ms=round((time.perf_counter() - start) * 1000, 2), attempts=max(attempts, 1),
            )
            self.records.append(rec)
            log.info("tool call", extra={"tool": tool, "status": status, "step": step, "error": error, "attempts": attempts})
            return rec

        spec = self.tools.get(tool)
        if spec is None:
            return record("denied", f"unknown tool '{tool}'"), None
        if self.calls_made >= self.max_calls:
            return record("denied", f"tool-call budget of {self.max_calls} exhausted"), None
        decision = authorize(spec.policy, self.grants, step_approved)
        if not decision.allowed:
            return record("denied", decision.reason), None
        try:
            args = spec.input_model.model_validate(arguments)
        except ValidationError as e:
            return record("error", f"invalid arguments: {e.errors()[0]['msg']}"), None

        self.calls_made += 1
        # Only read-only tools are retried: re-running a write could apply it twice.
        max_attempts = 1 + (self.max_retries if spec.policy.read_only else 0)
        last_error, last_status = "", "error"
        while attempts < max_attempts:
            attempts += 1
            try:
                raw = self._run_once(spec, args)
            except ToolError as e:  # deliberate, explainable refusal: never retried
                return record("error", str(e)), None
            except _Transient as e:
                last_error, last_status = str(e), ("timeout" if "exceeded" in str(e) else "error")
                if attempts < max_attempts:
                    time.sleep(BACKOFF_SECONDS * 2 ** (attempts - 1))
                continue
            try:
                output = spec.output_model.model_validate(raw).model_dump(mode="json")
            except ValidationError as e:
                return record("error", f"tool returned output that does not match its schema: {e.errors()[0]['msg']}"), None
            return record("ok"), output
        return record(last_status, f"{last_error} (after {attempts} attempt(s))"), None

    def _run_once(self, spec: ToolSpec, args) -> Any:
        future = _POOL.submit(spec.handler, self.context, args)
        try:
            return future.result(timeout=spec.policy.timeout_seconds)
        except cf.TimeoutError as e:
            raise _Transient(f"exceeded {spec.policy.timeout_seconds}s") from e
        except ToolError:
            raise
        except Exception as e:  # unexpected failure: keep the message, never crash the graph
            log.warning("tool failure", extra={"tool": spec.name, "error": str(e)})
            raise _Transient(f"{type(e).__name__}: {e}") from e


def _redact(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep traces small and free of large code blobs."""
    out = {}
    for k, v in arguments.items():
        if isinstance(v, str) and len(v) > 2000:
            out[k] = v[:2000] + "...[truncated]"
        else:
            out[k] = v
    return out
