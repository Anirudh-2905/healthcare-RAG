from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Load a tiny subset of dotenv syntax without adding a runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Settings:
    app_name: str
    source_docs_dir: Path
    chroma_dir: Path
    catalog_db: Path
    chroma_collection: str
    embedding_backend: str
    embedding_model: str
    llm_backend: str
    ollama_base_url: str
    ollama_model: str
    top_k: int
    min_retrieval_score: float
    chunk_words: int
    chunk_overlap_words: int
    max_agent_steps: int

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            app_name=os.getenv(
                "APP_NAME", "Evidence-Grounded Healthcare Research Assistant"
            ),
            source_docs_dir=_path_from_env("SOURCE_DOCS_DIR", "data/source_docs"),
            chroma_dir=_path_from_env("CHROMA_DIR", "data/chroma"),
            catalog_db=_path_from_env("CATALOG_DB", "data/catalog.db"),
            chroma_collection=os.getenv(
                "CHROMA_COLLECTION", "healthcare_guidelines"
            ),
            embedding_backend=os.getenv(
                "EMBEDDING_BACKEND", "sentence_transformer"
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm_backend=os.getenv("LLM_BACKEND", "ollama"),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            top_k=int(os.getenv("TOP_K", "4")),
            min_retrieval_score=float(os.getenv("MIN_RETRIEVAL_SCORE", "0.35")),
            chunk_words=int(os.getenv("CHUNK_WORDS", "220")),
            chunk_overlap_words=int(os.getenv("CHUNK_OVERLAP_WORDS", "40")),
            max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "2")),
        )

    def ensure_directories(self) -> None:
        self.source_docs_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_db.parent.mkdir(parents=True, exist_ok=True)


settings = Settings.from_env()

