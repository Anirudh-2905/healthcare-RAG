from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    paths: list[str] = Field(
        default_factory=list,
        description="PDF filenames relative to data/source_docs; empty means all PDFs.",
    )
    rebuild: bool = False


class IngestedDocument(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    skipped: bool = False


class IngestResponse(BaseModel):
    documents: list[IngestedDocument]
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Citation(BaseModel):
    document_id: str
    filename: str
    page: int
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    retrieval_score: float
    latency_ms: int


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    sha256: str
    ingested_at: str


class HealthResponse(BaseModel):
    status: str
    embedding_backend: str
    llm_backend: str
    llm_available: bool
    generation_mode: str
    indexed_documents: int


class EvaluationRequest(BaseModel):
    dataset_path: str = "data/evaluation/questions.json"


class EvaluationResponse(BaseModel):
    total: int
    retrieval_hit_at_k: float
    citation_presence: float
    citation_correctness: float
    expected_document_coverage: float
    abstention_accuracy: float
    keyword_coverage: float
    average_latency_ms: float
    cases: list[dict]

