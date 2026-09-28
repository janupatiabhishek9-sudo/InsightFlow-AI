"""FastAPI application.  Run:  uvicorn app.main:app --port 8000"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes import router
from app.service import InsightFlowService


def create_app(service: InsightFlowService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or InsightFlowService()
        yield

    app = FastAPI(title="InsightFlow AI", version=__version__, lifespan=lifespan,
                  description="Governed agentic analytics: evidence-backed investigations of tabular data.")
    app.include_router(router)
    return app


app = create_app()
