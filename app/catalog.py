from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class DocumentCatalog:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    pages INTEGER NOT NULL,
                    chunks INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL
                )
                """
            )

    def upsert(
        self,
        *,
        document_id: str,
        filename: str,
        path: str,
        sha256: str,
        pages: int,
        chunks: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents
                    (document_id, filename, path, sha256, pages, chunks, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    filename = excluded.filename,
                    path = excluded.path,
                    sha256 = excluded.sha256,
                    pages = excluded.pages,
                    chunks = excluded.chunks,
                    ingested_at = excluded.ingested_at
                """,
                (
                    document_id,
                    filename,
                    path,
                    sha256,
                    pages,
                    chunks,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get(self, document_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return dict(row) if row else None

    def list(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents")

