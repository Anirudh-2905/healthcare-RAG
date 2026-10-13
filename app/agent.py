from __future__ import annotations

import logging
from dataclasses import dataclass

from app.catalog import DocumentCatalog
from app.embeddings import EmbeddingProvider
from app.vector_store import SearchResult, VectorStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResult:
    evidence: list[SearchResult]
    tool_trace: list[str]


class EvidenceAgent:
    """A bounded, explicit tool loop suitable for inspection and testing."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        catalog: DocumentCatalog,
        max_steps: int = 2,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.catalog = catalog
        self.max_steps = max_steps

    def search_guidelines(self, query: str, top_k: int) -> list[SearchResult]:
        embedding = self.embedding_provider.embed([query])[0]
        return self.vector_store.query(embedding, top_k)

    def get_document_details(self, document_id: str) -> dict | None:
        return self.catalog.get(document_id)

    def run(self, question: str, top_k: int) -> AgentResult:
        trace: list[str] = []
        evidence: list[SearchResult] = []
        for step in range(self.max_steps):
            if step == 0:
                trace.append("search_guidelines")
                evidence = self.search_guidelines(question, top_k)
                logger.info("agent_tool=search_guidelines result_count=%d", len(evidence))
                if not evidence:
                    break
            elif evidence:
                trace.append("get_document_details")
                self.get_document_details(evidence[0].document_id)
                logger.info("agent_tool=get_document_details")
                break
        return AgentResult(evidence=evidence, tool_trace=trace)

