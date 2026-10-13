from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ExtractedPage:
    page: int
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    document_id: str
    filename: str
    path: Path
    sha256: str
    pages: list[ExtractedPage]


def extract_pdf(path: Path) -> ExtractedDocument:
    # Keep PDF support lazy so the lightweight retrieval and safety tests can run
    # before the optional project environment has been installed.
    from pypdf import PdfReader

    payload = path.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    document_id = sha256[:16]
    reader = PdfReader(path)
    pages = [
        ExtractedPage(page=index, text=page.extract_text() or "")
        for index, page in enumerate(reader.pages, start=1)
    ]
    return ExtractedDocument(
        document_id=document_id,
        filename=path.name,
        path=path,
        sha256=sha256,
        pages=pages,
    )

