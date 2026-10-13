from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.chunking import TextChunk


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    page: int
    text: str
    score: float


class VectorStore(Protocol):
    def add(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None: ...

    def query(self, embedding: list[float], top_k: int) -> list[SearchResult]: ...

    def clear(self) -> None: ...


class ChromaVectorStore:
    def __init__(self, persist_path: Path, collection_name: str):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        ids = [chunk.chunk_id for chunk in chunks]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def query(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        raw = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        return [
            SearchResult(
                chunk_id=chunk_id,
                document_id=str(metadata["document_id"]),
                filename=str(metadata["filename"]),
                page=int(metadata["page"]),
                text=document,
                score=max(0.0, min(1.0, 1.0 - float(distance))),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )


class InMemoryVectorStore:
    """Small cosine store for tests; production defaults to persistent Chroma."""

    def __init__(self):
        self._items: list[tuple[TextChunk, list[float]]] = []

    def add(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        existing = {chunk.chunk_id for chunk, _ in self._items}
        self._items.extend(
            (chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings)
            if chunk.chunk_id not in existing
        )

    def query(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        scored = [
            (self._cosine(embedding, candidate), chunk)
            for chunk, candidate in self._items
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                text=chunk.text,
                score=score,
            )
            for score, chunk in scored[:top_k]
        ]

    def clear(self) -> None:
        self._items.clear()

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
        right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
        return dot / (left_norm * right_norm)

