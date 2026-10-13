from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import PROJECT_ROOT, settings
from app.evaluation import run_evaluation
from app.main import build_service


def main() -> None:
    service = build_service()
    dataset = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
    report = run_evaluation(service, dataset, settings.top_k)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_backend": service.embedding_provider.name,
        "generation_backend": service.llm_provider.name,
        "primary_llm_available": service.llm_provider.available(),
        "top_k": settings.top_k,
        "minimum_retrieval_score": settings.min_retrieval_score,
        "metrics": report.model_dump(),
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    destination = reports_dir / "evaluation_latest.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved report to {destination}")


if __name__ == "__main__":
    main()
