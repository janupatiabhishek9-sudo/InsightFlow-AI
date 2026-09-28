"""Build an MCP server for one logical group of tools from the shared catalogue.

Each MCP tool gets a typed signature generated from the tool's Pydantic input model, and every call
is routed through ToolGateway, so MCP clients get exactly the same authorization, guards, timeouts
and budget as the in-process agent. MCP clients receive only DEFAULT_GRANTS: approval-only tools
(execute_write_sql) are listed for transparency but always denied over MCP.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from app.config import PROJECT_ROOT, get_settings
from app.rag.embeddings import get_embedder
from app.rag.retrieval import KnowledgeBase
from app.security.sandbox import Sandbox
from app.tools.duckdb_engine import DuckDBEngine
from app.tools.gateway import ToolGateway
from app.tools.registry import TOOLS, ToolContext, ToolSpec


def _signature_fn(spec: ToolSpec, gateway: ToolGateway):
    def call(**kwargs: Any):
        record, out = gateway.call(spec.name, kwargs)
        if out is None:  # ToolError messages are shown to the client (denials are explained, not hidden)
            raise ToolError(f"{record.status}: {record.error}")
        return spec.output_model.model_validate(out)

    params = []
    for name, field in spec.input_model.model_fields.items():
        default = inspect.Parameter.empty if field.is_required() else field.get_default(call_default_factory=True)
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=field.annotation))
    # Input *and* output schemas come from the Pydantic models, so MCP clients see both.
    call.__signature__ = inspect.Signature(params, return_annotation=spec.output_model)
    call.__annotations__ = {p.name: p.annotation for p in params} | {"return": spec.output_model}
    call.__name__ = spec.name
    return call


def build_server(group: str, dataset: Path, clearance: str | None = None) -> MCPServer:
    settings = get_settings()
    engine = DuckDBEngine(dataset, settings.query_timeout, settings.max_result_rows)
    knowledge = KnowledgeBase(settings.resolve(settings.knowledge_dir), settings.resolve(settings.vector_db_path),
                              get_embedder(settings.embedding_model))
    ctx = ToolContext(engine=engine, knowledge=knowledge, sandbox=Sandbox(settings.sandbox_dir, settings.sandbox_timeout, settings.sandbox_memory_mb),
                      clearance=clearance or settings.default_user_clearance)
    gateway = ToolGateway(ctx, max_calls=10_000)  # per-session budget; the agent applies its own tighter budget
    server = MCPServer(name=f"insightflow-{group}", instructions=f"InsightFlow AI {group} tools. All calls are policy-checked.")
    for spec in TOOLS.values():
        if spec.server != group:
            continue
        p = spec.policy
        description = (f"{spec.description} [permissions: {', '.join(x.value for x in p.permissions)}; risk: {p.risk_level}; "
                       f"timeout: {p.timeout_seconds}s{'; requires human approval' if p.requires_approval else ''}]")
        server.add_tool(_signature_fn(spec, gateway), name=spec.name, description=description,
                        annotations=ToolAnnotations(readOnlyHint=p.read_only, destructiveHint=not p.read_only))
    return server


def main(group: str) -> None:
    parser = argparse.ArgumentParser(description=f"InsightFlow {group} MCP server (stdio)")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "examples" / "sales.csv")
    args = parser.parse_args()
    build_server(group, args.dataset).run("stdio")
