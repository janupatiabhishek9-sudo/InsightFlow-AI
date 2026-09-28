"""REST endpoints. Thin: validation + delegation to InsightFlowService."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.api.schemas import DatasetInfo, DecisionRequest, InvestigationRequest, InvestigationView
from app.service import InsightFlowService, ServiceError

router = APIRouter()


def service(request: Request) -> InsightFlowService:
    return request.app.state.service


def _bad_request(e: ServiceError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(e))


@router.get("/health")
def health(svc: InsightFlowService = Depends(service)) -> dict:
    return {"status": "ok", "llm_provider": svc.settings.llm_provider, "knowledge_chunks": len(svc.deps.knowledge.chunks)}


@router.post("/datasets", response_model=DatasetInfo)
async def upload_dataset(file: UploadFile = File(...), svc: InsightFlowService = Depends(service)) -> DatasetInfo:
    content = await file.read(svc.settings.max_upload_mb * 1024 * 1024 + 1)
    try:
        return svc.register_dataset(file.filename or "upload.csv", content)
    except ServiceError as e:
        raise _bad_request(e) from e


@router.get("/datasets/example", response_model=DatasetInfo)
def example_dataset(svc: InsightFlowService = Depends(service)) -> DatasetInfo:
    return svc.example_dataset()


@router.get("/datasets/{dataset_id}", response_model=DatasetInfo)
def get_dataset(dataset_id: str, svc: InsightFlowService = Depends(service)) -> DatasetInfo:
    try:
        return svc.dataset(dataset_id)
    except ServiceError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/investigations", response_model=InvestigationView)
def start_investigation(req: InvestigationRequest, svc: InsightFlowService = Depends(service)) -> InvestigationView:
    try:
        return svc.start_investigation(req.dataset_id, req.question, req.clearance)
    except ServiceError as e:
        raise _bad_request(e) from e


@router.get("/investigations/{investigation_id}", response_model=InvestigationView)
def get_investigation(investigation_id: str, svc: InsightFlowService = Depends(service)) -> InvestigationView:
    try:
        return svc.get(investigation_id)
    except ServiceError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/investigations/{investigation_id}/decision", response_model=InvestigationView)
def decide(investigation_id: str, req: DecisionRequest, svc: InsightFlowService = Depends(service)) -> InvestigationView:
    try:
        return svc.decide(investigation_id, req.approved, req.reviewer, req.comment)
    except ServiceError as e:
        raise _bad_request(e) from e


@router.get("/investigations/{investigation_id}/trace")
def trace(investigation_id: str, svc: InsightFlowService = Depends(service)) -> dict:
    try:
        return svc.trace(investigation_id)
    except ServiceError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/traces")
def traces(limit: int = 50, svc: InsightFlowService = Depends(service)) -> list[dict]:
    return svc.traces.list(min(limit, 200))
