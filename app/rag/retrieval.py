"""Authorization-aware retrieval over a small persisted vector store (numpy + JSON).

Access control is applied *before* ranking, so restricted text never reaches a prompt.
Chunks that look like prompt injections are quarantined and returned separately for the trace.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from app.config import AccessLevel
from app.rag.embeddings import Embedder
from app.rag.ingestion import Chunk, fingerprint, load_chunks
from app.security.input_guard import scan_text

log = logging.getLogger(__name__)
ACCESS_RANK: dict[str, int] = {"public": 0, "internal": 1, "restricted": 2}


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class SearchResult(BaseModel):
    results: list[RetrievedChunk]
    quarantined: list[str] = []  # sources dropped because they contained instruction-like text
    filtered_by_access: int = 0


class KnowledgeBase:
    def __init__(self, knowledge_dir: Path, store_dir: Path, embedder: Embedder):
        self.knowledge_dir = knowledge_dir
        self.store_dir = store_dir
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 1), dtype=np.float32)
        self._load_or_build()

    # ---- index ------------------------------------------------------------------------------
    def _load_or_build(self) -> None:
        fp = fingerprint(self.knowledge_dir, self.embedder.name)
        meta_path, vec_path = self.store_dir / "chunks.json", self.store_dir / "vectors.npy"
        if meta_path.exists() and vec_path.exists():
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            if stored.get("fingerprint") == fp:
                self.chunks = [Chunk.model_validate(c) for c in stored["chunks"]]
                self.matrix = np.load(vec_path)
                return
        self.chunks = load_chunks(self.knowledge_dir)
        texts = [f"{c.title}. {c.heading}. {c.text}" for c in self.chunks]
        self.matrix = self.embedder.embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        np.save(vec_path, self.matrix)
        meta_path.write_text(
            json.dumps({"fingerprint": fp, "chunks": [c.model_dump() for c in self.chunks]}), encoding="utf-8"
        )
        log.info("knowledge index built", extra={"chunks": len(self.chunks), "embedder": self.embedder.name})

    # ---- queries ----------------------------------------------------------------------------
    def search(
        self,
        query: str,
        clearance: AccessLevel,
        k: int = 4,
        document_types: set[str] | None = None,
        min_score: float = 0.05,
    ) -> SearchResult:
        allowed = [
            i for i, c in enumerate(self.chunks)
            if ACCESS_RANK[c.metadata.access_level] <= ACCESS_RANK[clearance]
            and (document_types is None or c.metadata.document_type in document_types)
        ]
        filtered = sum(
            1 for c in self.chunks
            if ACCESS_RANK[c.metadata.access_level] > ACCESS_RANK[clearance]
        )
        if not allowed:
            return SearchResult(results=[], filtered_by_access=filtered)
        q = self.embedder.embed([query])[0]
        scores = self.matrix[allowed] @ q
        order = np.argsort(-scores)
        results, quarantined = [], []
        for pos in order:
            if len(results) >= k or scores[pos] < min_score:
                break
            chunk = self.chunks[allowed[pos]]
            if scan_text(chunk.text):
                quarantined.append(chunk.metadata.source)
                continue
            results.append(RetrievedChunk(chunk=chunk, score=round(float(scores[pos]), 4)))
        return SearchResult(results=results, quarantined=sorted(set(quarantined)), filtered_by_access=filtered)

    def kpi_definition(self, name: str, clearance: AccessLevel) -> SearchResult:
        return self.search(f"{name} definition formula", clearance, k=2, document_types={"definition"})

    def data_dictionary(self, clearance: AccessLevel) -> dict[str, str]:
        """Column -> description parsed from data-dictionary tables the user may read."""
        out: dict[str, str] = {}
        for c in self.chunks:
            if c.metadata.document_type != "data_dictionary":
                continue
            if ACCESS_RANK[c.metadata.access_level] > ACCESS_RANK[clearance]:
                continue
            for line in c.text.splitlines():
                m = re.match(r"^\|\s*([a-z_][a-z0-9_]*)\s*\|\s*(.+?)\s*\|$", line.strip())
                if m and m.group(1) != "column":
                    out[m.group(1)] = m.group(2)
        return out
