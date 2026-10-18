from dataclasses import replace
from pathlib import Path

import pytest

from app.config import Settings


def valid_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="Test RAG",
        source_docs_dir=tmp_path / "source_docs",
        chroma_dir=tmp_path / "chroma",
        catalog_db=tmp_path / "catalog.db",
        chroma_collection="test_collection",
        embedding_backend="hashing",
        embedding_model="test-model",
        llm_backend="extractive",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="test-model",
        top_k=4,
        min_retrieval_score=0.40,
        chunk_words=220,
        chunk_overlap_words=40,
        max_agent_steps=2,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"top_k": 0}, "TOP_K must be at least 1"),
        (
            {"min_retrieval_score": 1.01},
            "MIN_RETRIEVAL_SCORE must be between 0 and 1",
        ),
        ({"chunk_words": 0}, "CHUNK_WORDS must be at least 1"),
        (
            {"chunk_overlap_words": 220},
            "CHUNK_OVERLAP_WORDS must be non-negative and smaller than CHUNK_WORDS",
        ),
        ({"max_agent_steps": 0}, "MAX_AGENT_STEPS must be at least 1"),
        (
            {"ollama_base_url": "127.0.0.1:11434"},
            "OLLAMA_BASE_URL must be an absolute http:// or https:// URL",
        ),
    ],
)
def test_invalid_settings_fail_at_startup(
    tmp_path: Path, overrides: dict, message: str
):
    settings = valid_settings(tmp_path)

    with pytest.raises(ValueError, match=message):
        replace(settings, **overrides)


def test_boundary_configuration_is_valid(tmp_path: Path):
    settings = replace(
        valid_settings(tmp_path),
        top_k=1,
        min_retrieval_score=1.0,
        chunk_words=1,
        chunk_overlap_words=0,
        max_agent_steps=1,
        ollama_base_url="https://ollama.example.test",
    )

    assert settings.min_retrieval_score == 1.0


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("TOP_K", "many", "TOP_K must be an integer"),
        (
            "MIN_RETRIEVAL_SCORE",
            "high",
            "MIN_RETRIEVAL_SCORE must be a number",
        ),
    ],
)
def test_malformed_numeric_environment_values_have_clear_errors(
    monkeypatch, name: str, value: str, message: str
):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        Settings.from_env()
