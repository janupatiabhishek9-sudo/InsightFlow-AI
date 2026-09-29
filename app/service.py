"""InsightFlowService: the application layer used by the API, the UI and the evaluation runner.

Runs investigations synchronously (a TaskExecutor-style seam: swap `_run` for a queue later).
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Callable

from langgraph.types import Command

from app.api.schemas import DatasetInfo, InvestigationView
from app.config import AccessLevel, Settings, get_settings
from app.domain.governance import HumanDecision
from app.graph.deps import Dependencies, EngineCache
from app.graph.graph import build_graph
from app.llm.client import OPENAI_COMPATIBLE, LLMError, OpenAICompatibleClient, build_llm, describe, resolve_provider
from app.logging_config import configure_logging
from app.observability.tracing import TraceStore, configure_langsmith
from app.rag.embeddings import get_embedder
from app.rag.retrieval import ACCESS_RANK, KnowledgeBase
from app.security.sandbox import Sandbox
from app.tools.duckdb_engine import SUPPORTED_EXTENSIONS, DuckDBEngine
from app.tools.profiler import profile_dataset

log = logging.getLogger(__name__)
EXAMPLE_ID = "example-sales"
ProgressCallback = Callable[[dict[str, Any]], None]


class ServiceError(ValueError):
    pass


class InsightFlowService:
    def __init__(self, settings: Settings | None = None, llm=None):
        self.settings = settings or get_settings()
        configure_logging(self.settings.log_level)
        configure_langsmith(self.settings)
        if llm is None:
            try:
                llm = build_llm(self.settings)
            except LLMError as e:
                log.error("LLM misconfigured; falling back to rule_based", extra={"error": str(e)})
        knowledge = KnowledgeBase(self.settings.resolve(self.settings.knowledge_dir),
                                  self.settings.resolve(self.settings.vector_db_path),
                                  get_embedder(self.settings.embedding_model))
        self.deps = Dependencies(settings=self.settings, knowledge=knowledge, engines=EngineCache(self.settings),
                                 sandbox=Sandbox(self.settings.sandbox_dir, self.settings.sandbox_timeout,
                                                 self.settings.sandbox_memory_mb), llm=llm)
        self.graph = build_graph(self.deps)
        self.traces = TraceStore(self.settings.trace_dir)
        self.settings.raw_dir.mkdir(parents=True, exist_ok=True)

    # ---- AI switch ----------------------------------------------------------------------------
    def apply_ai_settings(self, enabled: bool, groq_key: str | None = None) -> str:
        """Switch AI mode at runtime (admin page). Returns a description of the model now in use."""
        if not enabled:
            self.deps.llm = None
        elif groq_key:
            groq = OPENAI_COMPATIBLE["groq"]
            model = self.settings.llm_model.strip() if resolve_provider(self.settings) == "groq" else ""
            self.deps.llm = OpenAICompatibleClient("groq", model or groq.default_model, groq_key, groq.base_url,
                                                   self.settings.llm_timeout_seconds)
        else:
            try:
                self.deps.llm = build_llm(self.settings)
            except LLMError as e:
                log.error("LLM misconfigured; using rule_based", extra={"error": str(e)})
                self.deps.llm = None
        provider, model = describe(self.deps.llm)
        return f"{provider}/{model}"

    # ---- datasets ---------------------------------------------------------------------------
    def _meta_path(self, dataset_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9-]{1,64}", dataset_id):
            raise ServiceError("invalid dataset id")
        return self.settings.raw_dir / f"{dataset_id}.meta.json"

    def register_dataset(self, filename: str, content: bytes) -> DatasetInfo:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ServiceError(f"unsupported file type '{suffix}'; upload CSV or Excel")
        if len(content) > self.settings.max_upload_mb * 1024 * 1024:
            raise ServiceError(f"file larger than {self.settings.max_upload_mb} MB")
        dataset_id = uuid.uuid4().hex[:12]
        path = self.settings.raw_dir / f"{dataset_id}{suffix}"  # never trust the uploaded name as a path
        path.write_bytes(content)
        try:
            return self._describe(dataset_id, Path(filename).name, path)
        except Exception as e:
            path.unlink(missing_ok=True)
            raise ServiceError(f"could not read the dataset: {e}") from e

    def example_dataset(self) -> DatasetInfo:
        path = self.settings.resolve(self.settings.data_dir) / "examples" / "sales.csv"
        if not path.exists():
            from app.datagen import generate

            path.parent.mkdir(parents=True, exist_ok=True)
            generate().to_csv(path, index=False)
        meta = self._meta_path(EXAMPLE_ID)
        if meta.exists():
            return DatasetInfo.model_validate_json(meta.read_text(encoding="utf-8"))
        return self._describe(EXAMPLE_ID, "sales.csv", path)

    def _describe(self, dataset_id: str, filename: str, path: Path) -> DatasetInfo:
        engine = DuckDBEngine(path, self.settings.query_timeout, self.settings.max_result_rows, self.settings.duckdb_memory_mb)
        try:
            _, quality = profile_dataset(engine)
        finally:
            engine.close()
        info = DatasetInfo(dataset_id=dataset_id, filename=filename, path=str(path), quality=quality)
        self._meta_path(dataset_id).write_text(info.model_dump_json(), encoding="utf-8")
        return info

    def dataset(self, dataset_id: str) -> DatasetInfo:
        if dataset_id == EXAMPLE_ID:
            return self.example_dataset()
        meta = self._meta_path(dataset_id)
        if not meta.exists():
            raise ServiceError(f"unknown dataset '{dataset_id}'")
        return DatasetInfo.model_validate_json(meta.read_text(encoding="utf-8"))

    # ---- investigations ---------------------------------------------------------------------
    def _config(self, investigation_id: str) -> dict:
        return {"configurable": {"thread_id": investigation_id}, "recursion_limit": 60}

    def start_investigation(
        self,
        dataset_id: str,
        question: str,
        clearance: AccessLevel | None = None,
        max_clearance: AccessLevel | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> InvestigationView:
        """Run an investigation. `max_clearance` comes from the authenticated caller (default: config);
        `clearance` may only lower it. `on_progress` receives each node's trace event as it completes."""
        info = self.dataset(dataset_id)
        ceiling = max_clearance or self.settings.default_user_clearance
        if clearance is not None and ACCESS_RANK[clearance] > ACCESS_RANK[ceiling]:
            raise ServiceError("requested clearance exceeds what this caller may read")
        investigation_id = uuid.uuid4().hex[:16]
        state = {"investigation_id": investigation_id, "user_question": question, "dataset_path": info.path,
                 "clearance": clearance or ceiling}
        return self._run(investigation_id, state, on_progress)

    def decide(self, investigation_id: str, approved: bool, reviewer: str = "anonymous", comment: str = "",
               on_progress: ProgressCallback | None = None) -> InvestigationView:
        view = self.get(investigation_id)
        if view.status != "awaiting_approval":
            raise ServiceError(f"investigation is not awaiting approval (status: {view.status})")
        decision = HumanDecision(approved=approved, reviewer=reviewer, comment=comment)
        return self._run(investigation_id, Command(resume=decision.model_dump(mode="json")), on_progress)

    def _run(self, investigation_id: str, payload, on_progress: ProgressCallback | None = None) -> InvestigationView:
        # Streaming node updates lets callers show live progress; the result is identical to invoke().
        for chunk in self.graph.stream(payload, self._config(investigation_id), stream_mode="updates"):
            if on_progress is None:
                continue
            for update in chunk.values():
                for event in (update or {}).get("trace_events", []) if isinstance(update, dict) else []:
                    on_progress(event)
        view = self.get(investigation_id)
        if view.status == "awaiting_approval":
            values = self.graph.get_state(self._config(investigation_id)).values
            self.traces.save({**values, "status": "awaiting_approval"})
        elif view.status != "running":
            self.deps.engines.release(investigation_id)
        return view

    def get(self, investigation_id: str) -> InvestigationView:
        snapshot = self.graph.get_state(self._config(investigation_id))
        s = snapshot.values
        if not s:
            raise ServiceError(f"unknown investigation '{investigation_id}'")
        interrupted = any(t.interrupts for t in snapshot.tasks)
        status = "awaiting_approval" if interrupted else s.get("status", "running")
        risk = s.get("risk_assessment")
        return InvestigationView(
            investigation_id=investigation_id, status=status, status_reason=s.get("status_reason", ""),
            question=s.get("user_question", ""), model=s.get("model") or {}, prompt_versions=s.get("prompt_versions") or {},
            data_quality=s.get("data_quality_report"), understanding=s.get("understanding"),
            retrieved_context=s.get("retrieved_context") or [], plan=s.get("investigation_plan"),
            plan_validation=s.get("plan_validation"), risk=risk,
            pending_actions=risk.pending_actions if (risk and interrupted) else [],
            human_decision=s.get("human_decision"),
            step_results=[v for _, v in sorted((s.get("step_results") or {}).items())],
            evidence=s.get("evidence") or [], charts=s.get("charts") or [], validation=s.get("validation_results"),
            report=s.get("final_report"), evaluation=s.get("evaluation_results"), tool_calls=s.get("tool_calls") or [],
            token_usage=s.get("token_usage", 0), errors=s.get("errors") or [], trace_events=s.get("trace_events") or [],
        )

    def trace(self, investigation_id: str) -> dict:
        t = self.traces.load(investigation_id)
        if t is None:
            raise ServiceError(f"no trace for '{investigation_id}'")
        return t
