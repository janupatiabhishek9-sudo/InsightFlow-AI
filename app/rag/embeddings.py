"""Embedding models behind one interface.

Default `hashing`: a dependency-free, deterministic feature-hashing embedder (unigrams + bigrams).
It uses ~0 RAM, needs no download and is plenty for a small curated knowledge base.
`fastembed:<model>` swaps in a small ONNX neural model (e.g. BAAI/bge-small-en-v1.5) without torch.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "for", "on", "by", "with", "be",
         "as", "it", "that", "this", "was", "did", "why", "what", "which", "how", "do", "does"}


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _stem(token: str) -> str:
    for suffix in ("ies", "es", "s"):
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _index(self, feature: str) -> tuple[int, float]:
        digest = hashlib.md5(feature.encode()).digest()
        return int.from_bytes(digest[:4], "little") % self.dim, (1.0 if digest[4] & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]
            counts: dict[str, int] = {}
            for feat in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]:
                counts[feat] = counts.get(feat, 0) + 1
            for feat, n in counts.items():
                idx, sign = self._index(feat)
                out[row, idx] += sign * (1 + math.log(n))
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class FastEmbedEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding  # optional dependency

        self._model = TextEmbedding(model_name=model)
        self.name = f"fastembed-{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)


def get_embedder(spec: str) -> Embedder:
    if spec.startswith("fastembed:"):
        return FastEmbedEmbedder(spec.split(":", 1)[1])
    if spec == "hashing" or spec.startswith("hashing"):
        return HashingEmbedder()
    raise ValueError(f"unknown EMBEDDING_MODEL '{spec}' (use 'hashing' or 'fastembed:<model>')")
