from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from app.agent import EvidenceAgent
from app.catalog import DocumentCatalog
from app.chunking import TextChunk, chunk_page
from app.embeddings import EmbeddingProvider
from app.llm import LLMProvider
from app.pdf_loader import extract_pdf
from app.safety import (
    contains_prompt_injection,
    requests_personal_medical_advice,
    safe_excerpt,
)
from app.schemas import Citation, IngestedDocument, QueryResponse
from app.vector_store import VectorStore


logger = logging.getLogger(__name__)


class HealthcareRAGService:
    def __init__(
        self,
        *,
        source_docs_dir: Path,
        catalog: DocumentCatalog,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        chunk_words: int,
        chunk_overlap_words: int,
        min_retrieval_score: float,
        max_agent_steps: int,
    ):
        self.source_docs_dir = source_docs_dir.resolve()
        self.catalog = catalog
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.chunk_words = chunk_words
        self.chunk_overlap_words = chunk_overlap_words
        self.min_retrieval_score = min_retrieval_score
        self.agent = EvidenceAgent(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            catalog=catalog,
            max_steps=max_agent_steps,
        )

    def _resolve_paths(self, requested: list[str]) -> list[Path]:
        if not requested:
            candidates = sorted(self.source_docs_dir.glob("*.pdf"))
        else:
            candidates = [self.source_docs_dir / item for item in requested]
        resolved: list[Path] = []
        for candidate in candidates:
            path = candidate.resolve()
            if self.source_docs_dir not in path.parents:
                raise ValueError(f"Path is outside approved source directory: {candidate}")
            if path.suffix.lower() != ".pdf" or not path.is_file():
                raise ValueError(f"Approved PDF not found: {candidate.name}")
            resolved.append(path)
        return resolved

    def ingest(self, requested_paths: list[str], rebuild: bool) -> list[IngestedDocument]:
        if rebuild:
            self.vector_store.clear()
            self.catalog.clear()

        results: list[IngestedDocument] = []
        for path in self._resolve_paths(requested_paths):
            document = extract_pdf(path)
            existing = self.catalog.get(document.document_id)
            if existing and not rebuild:
                results.append(
                    IngestedDocument(
                        document_id=document.document_id,
                        filename=document.filename,
                        pages=int(existing["pages"]),
                        chunks=int(existing["chunks"]),
                        skipped=True,
                    )
                )
                continue

            chunks: list[TextChunk] = []
            for page in document.pages:
                chunks.extend(
                    chunk_page(
                        text=page.text,
                        document_id=document.document_id,
                        filename=document.filename,
                        page=page.page,
                        chunk_words=self.chunk_words,
                        overlap_words=self.chunk_overlap_words,
                    )
                )
            if not chunks:
                raise ValueError(
                    f"No extractable text found in {document.filename}; OCR is required."
                )
            embeddings = self.embedding_provider.embed([chunk.text for chunk in chunks])
            self.vector_store.add(chunks, embeddings)
            self.catalog.upsert(
                document_id=document.document_id,
                filename=document.filename,
                path=str(document.path),
                sha256=document.sha256,
                pages=len(document.pages),
                chunks=len(chunks),
            )
            results.append(
                IngestedDocument(
                    document_id=document.document_id,
                    filename=document.filename,
                    pages=len(document.pages),
                    chunks=len(chunks),
                )
            )
        return results

    def query(self, question: str, top_k: int) -> QueryResponse:
        started = time.perf_counter()
        request_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
        logger.info("query_received request_hash=%s", request_hash)

        if contains_prompt_injection(question):
            return self._abstention(
                "I cannot follow instructions that attempt to override the assistant's safety rules.",
                started,
            )

        if requests_personal_medical_advice(question):
            return self._abstention(
                "I can summarize indexed evidence, but I cannot diagnose, prescribe, "
                "or provide personal medical advice. Please consult a qualified healthcare professional.",
                started,
            )

        agent_result = self.agent.run(question, top_k)
        safe_evidence = [
            item for item in agent_result.evidence if not contains_prompt_injection(item.text)
        ]
        if len(safe_evidence) != len(agent_result.evidence):
            logger.warning("unsafe_retrieved_content_filtered request_hash=%s", request_hash)

        best_score = safe_evidence[0].score if safe_evidence else 0.0
        if not safe_evidence or best_score < self.min_retrieval_score:
            return self._abstention(
                "Insufficient evidence in the indexed documents.", started, best_score
            )

        answer = self.llm_provider.generate(question, safe_evidence)
        citations = [
            Citation(
                document_id=item.document_id,
                filename=item.filename,
                page=item.page,
                excerpt=safe_excerpt(item.text),
            )
            for item in safe_evidence[:3]
        ]
        return QueryResponse(
            answer=answer,
            citations=citations,
            abstained=False,
            retrieval_score=round(best_score, 4),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _abstention(
        answer: str, started: float, retrieval_score: float = 0.0
    ) -> QueryResponse:
        return QueryResponse(
            answer=answer,
            citations=[],
            abstained=True,
            retrieval_score=round(retrieval_score, 4),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

