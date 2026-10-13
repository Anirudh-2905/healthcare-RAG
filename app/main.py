from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status

from app.catalog import DocumentCatalog
from app.config import PROJECT_ROOT, settings
from app.embeddings import create_embedding_provider
from app.evaluation import run_evaluation
from app.llm import create_llm_provider
from app.schemas import (
    DocumentRecord,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.service import HealthcareRAGService
from app.vector_store import ChromaVectorStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Local evidence lookup over approved healthcare guidelines. "
        "Educational use only; not medical advice."
    ),
)


@lru_cache(maxsize=1)
def build_service() -> HealthcareRAGService:
    settings.ensure_directories()
    embedding_provider = create_embedding_provider(
        settings.embedding_backend, settings.embedding_model
    )
    vector_store = ChromaVectorStore(
        settings.chroma_dir, settings.chroma_collection
    )
    llm_provider = create_llm_provider(
        settings.llm_backend, settings.ollama_base_url, settings.ollama_model
    )
    return HealthcareRAGService(
        source_docs_dir=settings.source_docs_dir,
        catalog=DocumentCatalog(settings.catalog_db),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_provider=llm_provider,
        chunk_words=settings.chunk_words,
        chunk_overlap_words=settings.chunk_overlap_words,
        min_retrieval_score=settings.min_retrieval_score,
        max_agent_steps=settings.max_agent_steps,
    )


def get_service() -> HealthcareRAGService:
    try:
        return build_service()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Local model or vector-store setup is unavailable: {exc}",
        ) from exc


@app.get("/health", response_model=HealthResponse)
def health(service: HealthcareRAGService = Depends(get_service)) -> HealthResponse:
    llm_available = service.llm_provider.available()
    return HealthResponse(
        status="healthy" if llm_available else "degraded",
        embedding_backend=service.embedding_provider.name,
        llm_backend=service.llm_provider.name,
        llm_available=llm_available,
        generation_mode="ollama" if llm_available else "extractive-fallback",
        indexed_documents=len(service.catalog.list()),
    )


@app.post(
    "/documents/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_documents(
    request: IngestRequest,
    service: HealthcareRAGService = Depends(get_service),
) -> IngestResponse:
    try:
        documents = service.ingest(request.paths, request.rebuild)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(
        documents=documents,
        total_chunks=sum(document.chunks for document in documents),
    )


@app.get("/documents", response_model=list[DocumentRecord])
def list_documents(
    service: HealthcareRAGService = Depends(get_service),
) -> list[DocumentRecord]:
    return [DocumentRecord(**record) for record in service.catalog.list()]


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    service: HealthcareRAGService = Depends(get_service),
) -> QueryResponse:
    return service.query(request.question, request.top_k or settings.top_k)


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(
    request: EvaluationRequest,
    service: HealthcareRAGService = Depends(get_service),
) -> EvaluationResponse:
    dataset_path = Path(request.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    dataset_path = dataset_path.resolve()
    evaluation_dir = (PROJECT_ROOT / "data" / "evaluation").resolve()
    if evaluation_dir not in dataset_path.parents:
        raise HTTPException(status_code=400, detail="Dataset must be in data/evaluation")
    if not dataset_path.is_file():
        raise HTTPException(status_code=404, detail="Evaluation dataset not found")
    try:
        return run_evaluation(service, dataset_path, settings.top_k)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
