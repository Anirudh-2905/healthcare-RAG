from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    filename: str
    page: int
    text: str
    chunk_index: int


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_page(
    *,
    text: str,
    document_id: str,
    filename: str,
    page: int,
    chunk_words: int,
    overlap_words: int,
) -> list[TextChunk]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be between 0 and chunk_words - 1")

    words = normalize_text(text).split()
    if not words:
        return []

    chunks: list[TextChunk] = []
    step = chunk_words - overlap_words
    for chunk_index, start in enumerate(range(0, len(words), step)):
        segment = words[start : start + chunk_words]
        if not segment:
            break
        chunk_id = f"{document_id}:p{page}:c{chunk_index}"
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                filename=filename,
                page=page,
                text=" ".join(segment),
                chunk_index=chunk_index,
            )
        )
        if start + chunk_words >= len(words):
            break
    return chunks

