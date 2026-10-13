from __future__ import annotations

import os
from typing import Annotated, Any

import requests
import typer


DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

app = typer.Typer(
    name="healthcare-rag",
    help="Query the evidence-grounded healthcare assistant from the terminal.",
    no_args_is_help=True,
)


class APIClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> Any:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                json=json,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach the RAG API at {self.base_url}. "
                "Start it with: uvicorn app.main:app --reload"
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError("The RAG API request timed out.") from exc
        except requests.HTTPError as exc:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(
                f"The RAG API returned HTTP {response.status_code}: {detail}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"RAG API request failed: {exc}") from exc


@app.callback()
def configure(
    ctx: typer.Context,
    api_url: Annotated[
        str,
        typer.Option(
            "--api-url",
            envvar="RAG_API_URL",
            help="Base URL of the running FastAPI application.",
        ),
    ] = DEFAULT_API_URL,
) -> None:
    ctx.obj = APIClient(api_url)


def _client(ctx: typer.Context) -> APIClient:
    return ctx.ensure_object(APIClient)


def _fail(message: str) -> None:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _print_query_response(payload: dict[str, Any], show_excerpts: bool) -> None:
    if payload["abstained"]:
        typer.secho("\nAbstained", fg=typer.colors.YELLOW, bold=True)
    else:
        typer.secho("\nAnswer", fg=typer.colors.GREEN, bold=True)
    typer.echo(payload["answer"])

    citations = payload.get("citations", [])
    if citations:
        typer.secho("\nSources", fg=typer.colors.CYAN, bold=True)
        for index, citation in enumerate(citations, start=1):
            typer.echo(
                f"  {index}. {citation['filename']} — page {citation['page']}"
            )
            if show_excerpts:
                typer.echo(f"     {citation['excerpt']}")

    typer.secho("\nRetrieval", fg=typer.colors.BLUE, bold=True)
    typer.echo(
        f"  score={payload['retrieval_score']:.4f}  "
        f"latency={payload['latency_ms']} ms"
    )


@app.command()
def health(ctx: typer.Context) -> None:
    """Show API, model, index, and generation status."""
    try:
        payload = _client(ctx).request("GET", "/health")
    except RuntimeError as exc:
        _fail(str(exc))
    color = typer.colors.GREEN if payload["status"] == "healthy" else typer.colors.YELLOW
    typer.secho(f"Status: {payload['status']}", fg=color, bold=True)
    typer.echo(f"Generation mode: {payload['generation_mode']}")
    typer.echo(f"LLM available: {payload['llm_available']}")
    typer.echo(f"LLM backend: {payload['llm_backend']}")
    typer.echo(f"Embedding backend: {payload['embedding_backend']}")
    typer.echo(f"Indexed documents: {payload['indexed_documents']}")


@app.command()
def documents(ctx: typer.Context) -> None:
    """List the documents currently indexed by the RAG system."""
    try:
        payload = _client(ctx).request("GET", "/documents")
    except RuntimeError as exc:
        _fail(str(exc))
    if not payload:
        typer.secho("No documents are indexed.", fg=typer.colors.YELLOW)
        return
    typer.secho(f"Indexed documents: {len(payload)}", fg=typer.colors.CYAN, bold=True)
    for document in payload:
        typer.echo(
            f"- {document['filename']} "
            f"({document['pages']} pages, {document['chunks']} chunks)"
        )


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument(help="Question to ask the assistant.")],
    top_k: Annotated[
        int, typer.Option("--top-k", min=1, max=10, help="Chunks to retrieve.")
    ] = 4,
    show_excerpts: Annotated[
        bool,
        typer.Option("--show-excerpts", help="Print citation excerpts."),
    ] = False,
) -> None:
    """Ask one question and print its answer and citations."""
    try:
        payload = _client(ctx).request(
            "POST", "/query", json={"question": question, "top_k": top_k}
        )
    except RuntimeError as exc:
        _fail(str(exc))
    _print_query_response(payload, show_excerpts)


@app.command()
def chat(
    ctx: typer.Context,
    top_k: Annotated[
        int, typer.Option("--top-k", min=1, max=10, help="Chunks to retrieve.")
    ] = 4,
    show_excerpts: Annotated[
        bool,
        typer.Option("--show-excerpts", help="Print citation excerpts."),
    ] = False,
) -> None:
    """Open an interactive question-and-answer session."""
    typer.secho("Healthcare Evidence RAG", fg=typer.colors.CYAN, bold=True)
    typer.echo("Educational evidence lookup only; not medical advice.")
    typer.echo("Type /exit to stop.\n")
    while True:
        try:
            question = typer.prompt("You").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nGoodbye.")
            return
        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            typer.echo("Goodbye.")
            return
        if len(question) < 3:
            typer.secho("Please enter at least three characters.", fg=typer.colors.YELLOW)
            continue
        try:
            payload = _client(ctx).request(
                "POST", "/query", json={"question": question, "top_k": top_k}
            )
        except RuntimeError as exc:
            _fail(str(exc))
        _print_query_response(payload, show_excerpts)
        typer.echo()


@app.command()
def ingest(
    ctx: typer.Context,
    paths: Annotated[
        list[str] | None,
        typer.Argument(help="Optional PDF filenames from data/source_docs."),
    ] = None,
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Clear and rebuild the complete index."),
    ] = False,
) -> None:
    """Ingest selected PDFs, or every approved PDF when none are listed."""
    try:
        payload = _client(ctx).request(
            "POST",
            "/documents/ingest",
            json={"paths": paths or [], "rebuild": rebuild},
        )
    except RuntimeError as exc:
        _fail(str(exc))
    typer.secho("Ingestion complete", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Total chunks: {payload['total_chunks']}")
    for document in payload["documents"]:
        state = "skipped" if document["skipped"] else "indexed"
        typer.echo(
            f"- {document['filename']}: {state}, "
            f"{document['pages']} pages, {document['chunks']} chunks"
        )


@app.command()
def evaluate(ctx: typer.Context) -> None:
    """Run the fixed 20-case evaluation through the API."""
    typer.echo("Running evaluation. This can take a minute with a local model...")
    try:
        payload = _client(ctx).request("POST", "/evaluate", json={})
    except RuntimeError as exc:
        _fail(str(exc))
    typer.secho("\nEvaluation results", fg=typer.colors.GREEN, bold=True)
    labels = {
        "total": "Cases",
        "retrieval_hit_at_k": "Retrieval hit@k",
        "citation_presence": "Citation presence",
        "citation_correctness": "Citation correctness",
        "expected_document_coverage": "Expected-document coverage",
        "abstention_accuracy": "Abstention accuracy",
        "keyword_coverage": "Keyword coverage",
        "average_latency_ms": "Average latency (ms)",
    }
    for key, label in labels.items():
        typer.echo(f"{label}: {payload[key]}")


if __name__ == "__main__":
    app()
