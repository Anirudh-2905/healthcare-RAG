# Evidence-Grounded Healthcare Research Assistant

A local-first RAG application that answers questions from approved healthcare guideline PDFs, provides page-level citations, abstains when evidence is insufficient, and includes a small explicit tool-using agent, safety checks, and an evaluation harness.

This is an educational evidence-retrieval system. It does not diagnose conditions, prescribe treatment, or replace a qualified healthcare professional.

## Why this project exists

Healthcare teams often need to locate statements across long guideline documents. This project demonstrates how a controlled retrieval pipeline can make that work faster while keeping the source evidence visible and measuring failure modes.

## Architecture

```text
Approved PDFs
  -> page extraction
  -> word-aware chunking
  -> MiniLM embeddings
  -> persistent Chroma collection
  -> retrieval tools
  -> guarded agent loop
  -> Ollama open-source LLM or extractive fallback
  -> answer + page citations + abstention signal
```

Document metadata is stored in SQLite. Chroma stores the vector index. Raw prompts are not logged.

## API

- `POST /documents/ingest` indexes specified PDFs or every PDF in `data/source_docs`.
- `GET /documents` lists indexed documents.
- `POST /query` returns a cited answer or explicit abstention.
- `GET /health` reports dependency and backend status.
- `POST /evaluate` runs the fixed evaluation set.

Interactive documentation is available at `http://127.0.0.1:8000/docs`.

## Terminal CLI

Keep the FastAPI server running, then open a second activated terminal. Start an interactive session:

```bash
python -m app.cli chat
```

Useful one-shot commands:

```bash
python -m app.cli health
python -m app.cli documents
python -m app.cli ask "How much physical activity should adults perform?"
python -m app.cli ask --show-excerpts "What does the guideline say about salt?"
python -m app.cli evaluate
```

After installing the project in editable mode, the shorter `healthcare-rag` command is also available:

```bash
healthcare-rag chat
```

Set `RAG_API_URL` or pass `--api-url` before the command when the API is not running at `http://127.0.0.1:8000`.

## Local setup

```bash
cd zs_healthcare_rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Place approved public-health PDFs in `data/source_docs/`. A source manifest and download helper are included in `scripts/download_sources.py`.

See [`data/README.md`](data/README.md) for the exact five official sources, required filenames, and an explanation of the evaluation dataset.

```bash
python scripts/download_sources.py
```

The helper verifies that each response is actually a PDF. If an official link changes, it reports only that document so it can be downloaded manually without blocking the rest of the project.

For local generation, install Ollama separately and run:

```bash
ollama pull qwen2.5:1.5b
ollama serve
```

If Ollama is unavailable, the application automatically returns a citation-first extractive answer. Retrieval and evaluation still work.

Configuration is validated when the application starts. `TOP_K`, chunk sizes,
overlap, retrieval threshold and agent-step settings must remain within valid
ranges, and `OLLAMA_BASE_URL` must be an absolute `http://` or `https://` URL.
Invalid configuration stops startup with a setting-specific error instead of
failing later during a request.

Run the API:

```bash
uvicorn app.main:app --reload
```

Run tests:

```bash
pytest
```

Run the fixed evaluation suite and save `reports/evaluation_latest.json`:

```bash
python -m scripts.run_evaluation
```

For a download-free smoke test, set these two values in `.env`:

```ini
EMBEDDING_BACKEND=hashing
LLM_BACKEND=extractive
```

This mode is intentionally less accurate, but exercises the API, retrieval, citations, abstention, and tests without Ollama or a model download.

## Example requests

```bash
curl -X POST http://127.0.0.1:8000/documents/ingest \
  -H 'Content-Type: application/json' \
  -d '{"paths": [], "rebuild": true}'
```

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does the guideline recommend?", "top_k": 4}'
```

## Evaluation

The fixed evaluation set covers answerable, multi-document, unanswerable, and prompt-injection cases. Reported metrics include retrieval hit@k, citation presence, citation correctness, expected-document coverage, abstention accuracy, keyword coverage, and latency.

The initial retrieval threshold is `0.40`. It was selected from the first corpus baseline: the weakest supported test scored above it, while the strongest unrelated test scored below it. This is a corpus-specific engineering threshold, not a medical-confidence probability, and should be recalibrated whenever the documents or embedding model change.

## Known limitations

- Small local models can produce weak synthesis even when retrieval is correct.
- PDF extraction quality depends on the source document; scanned PDFs require OCR, which is not included in v0.1.
- The confidence signal is a retrieval score, not a calibrated probability of correctness.
- The prompt-injection filter is deliberately simple and should be combined with stronger isolation and policy enforcement in production.
- The application is designed for a small curated corpus, not unrestricted medical search.

## Responsible-use rules

- Use only approved public documents.
- Do not provide personal medical advice.
- Do not place patient identifiers, secrets, or credentials in prompts or logs.
- Treat retrieved text as untrusted data, never as system instructions.
- Prefer abstention over unsupported claims.

