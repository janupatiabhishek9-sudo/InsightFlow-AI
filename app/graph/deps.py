"""Dependencies injected into graph nodes (no module-level mutable state)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.llm.client import LLMClient
from app.rag.retrieval import KnowledgeBase
from app.security.sandbox import Sandbox
from app.tools.duckdb_engine import DuckDBEngine


class EngineCache:
    """One DuckDB working copy per investigation, bounded to keep RAM low on an 8 GB laptop.

    An evicted investigation reloads its dataset from disk (approved modifications are then lost,
    which is safe: the source file is never modified).
    """

    def __init__(self, settings: Settings, max_engines: int = 3):
        self.settings = settings
        self.max_engines = max_engines
        self._engines: OrderedDict[str, DuckDBEngine] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, investigation_id: str, dataset_path: str) -> DuckDBEngine:
        with self._lock:
            engine = self._engines.get(investigation_id)
            if engine is None:
                engine = DuckDBEngine(Path(dataset_path), self.settings.query_timeout, self.settings.max_result_rows)
                self._engines[investigation_id] = engine
                while len(self._engines) > self.max_engines:
                    _, old = self._engines.popitem(last=False)
                    old.close()
            self._engines.move_to_end(investigation_id)
            return engine

    def release(self, investigation_id: str) -> None:
        with self._lock:
            engine = self._engines.pop(investigation_id, None)
        if engine:
            engine.close()


@dataclass
class Dependencies:
    settings: Settings
    knowledge: KnowledgeBase
    engines: EngineCache
    sandbox: Sandbox
    llm: LLMClient | None = None
