from typer.testing import CliRunner

from app.cli import APIClient, app


runner = CliRunner()


def test_health_command(monkeypatch):
    monkeypatch.setattr(
        APIClient,
        "request",
        lambda *args, **kwargs: {
            "status": "healthy",
            "generation_mode": "ollama",
            "llm_available": True,
            "llm_backend": "ollama:qwen2.5:1.5b",
            "embedding_backend": "all-MiniLM-L6-v2",
            "indexed_documents": 5,
        },
    )

    result = runner.invoke(app, ["health"])

    assert result.exit_code == 0
    assert "Status: healthy" in result.output
    assert "Indexed documents: 5" in result.output


def test_ask_command_prints_answer_and_source(monkeypatch):
    monkeypatch.setattr(
        APIClient,
        "request",
        lambda *args, **kwargs: {
            "answer": "Adults should complete 150–300 minutes each week. [S1]",
            "citations": [
                {
                    "filename": "who_physical_activity_guideline.pdf",
                    "page": 42,
                    "excerpt": "Adults should undertake regular physical activity.",
                }
            ],
            "abstained": False,
            "retrieval_score": 0.72,
            "latency_ms": 1200,
        },
    )

    result = runner.invoke(app, ["ask", "How much activity is recommended?"])

    assert result.exit_code == 0
    assert "150–300 minutes" in result.output
    assert "who_physical_activity_guideline.pdf — page 42" in result.output


def test_connection_error_is_actionable(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError(
            "Cannot reach the RAG API. Start it with: uvicorn app.main:app --reload"
        )

    monkeypatch.setattr(APIClient, "request", fail)

    result = runner.invoke(app, ["health"])

    assert result.exit_code == 1
    assert "uvicorn app.main:app --reload" in result.output
