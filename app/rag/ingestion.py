"""Load markdown knowledge documents (with front-matter metadata) and chunk them by section."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel

from app.config import AccessLevel

REQUIRED_METADATA = ("document_type", "department", "version", "access_level", "effective_date")


class DocumentMetadata(BaseModel):
    source: str
    document_type: str
    department: str
    version: str
    access_level: AccessLevel
    effective_date: str


class Chunk(BaseModel):
    id: str
    text: str
    heading: str
    title: str
    metadata: DocumentMetadata


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, raw[match.end():]


def chunk_document(path: Path, root: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    if missing:
        raise ValueError(f"{path.name}: missing metadata {missing}")
    # Unknown access levels fall back to the most restrictive level (fail closed).
    if meta["access_level"] not in ("public", "internal", "restricted"):
        meta["access_level"] = "restricted"
    source = path.relative_to(root).as_posix()
    metadata = DocumentMetadata(source=f"knowledge/{source}", **{k: meta[k] for k in REQUIRED_METADATA})

    title_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    sections = re.split(r"^##\s+", body, flags=re.MULTILINE)
    chunks = []
    for section in sections[1:] or sections:
        heading, _, text = section.partition("\n")
        text = text.strip()
        if not text:
            continue
        cid = hashlib.sha1(f"{source}#{heading}".encode()).hexdigest()[:12]
        chunks.append(Chunk(id=cid, text=text, heading=heading.strip(), title=title, metadata=metadata))
    return chunks


def load_chunks(knowledge_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.rglob("*.md")):
        chunks.extend(chunk_document(path, knowledge_dir))
    return chunks


def fingerprint(knowledge_dir: Path, embedder_name: str) -> str:
    h = hashlib.sha256(embedder_name.encode())
    for path in sorted(knowledge_dir.rglob("*.md")):
        h.update(path.as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]
