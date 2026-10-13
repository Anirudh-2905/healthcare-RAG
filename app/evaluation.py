from __future__ import annotations

import json
from pathlib import Path

from app.schemas import EvaluationResponse
from app.service import HealthcareRAGService


def run_evaluation(
    service: HealthcareRAGService, dataset_path: Path, top_k: int
) -> EvaluationResponse:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must be a non-empty JSON list")

    results: list[dict] = []
    retrieval_hits = 0
    retrieval_cases = 0
    citation_hits = 0
    citation_cases = 0
    citation_precision_scores: list[float] = []
    document_coverage_scores: list[float] = []
    abstention_hits = 0
    keyword_scores: list[float] = []
    latencies: list[int] = []

    for case in cases:
        response = service.query(case["question"], top_k=top_k)
        cited_files = {citation.filename for citation in response.citations}
        expected_files = set(case.get("expected_documents", []))
        retrieval_hit = not expected_files or bool(cited_files & expected_files)
        expected_abstention = bool(case.get("should_abstain", False))
        abstention_correct = response.abstained == expected_abstention

        citation_precision = (
            len(cited_files & expected_files) / len(cited_files)
            if cited_files
            else float(expected_abstention)
        )
        document_coverage = (
            len(cited_files & expected_files) / len(expected_files)
            if expected_files
            else 1.0
        )

        if expected_files:
            retrieval_cases += 1
            retrieval_hits += int(retrieval_hit)
        if not expected_abstention:
            citation_cases += 1
            citation_hits += int(bool(response.citations))
            citation_precision_scores.append(citation_precision)
            document_coverage_scores.append(document_coverage)

        keywords = [item.lower() for item in case.get("expected_keywords", [])]
        answer_lower = response.answer.lower()
        keyword_coverage = (
            sum(keyword in answer_lower for keyword in keywords) / len(keywords)
            if keywords
            else 1.0
        )

        abstention_hits += int(abstention_correct)
        keyword_scores.append(keyword_coverage)
        latencies.append(response.latency_ms)
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "retrieval_hit": retrieval_hit,
                "citation_present": bool(response.citations),
                "citation_correctness": round(citation_precision, 3),
                "expected_document_coverage": round(document_coverage, 3),
                "abstention_correct": abstention_correct,
                "keyword_coverage": round(keyword_coverage, 3),
                "latency_ms": response.latency_ms,
            }
        )

    total = len(cases)
    return EvaluationResponse(
        total=total,
        retrieval_hit_at_k=round(retrieval_hits / max(1, retrieval_cases), 3),
        citation_presence=round(citation_hits / max(1, citation_cases), 3),
        citation_correctness=round(
            sum(citation_precision_scores) / max(1, len(citation_precision_scores)), 3
        ),
        expected_document_coverage=round(
            sum(document_coverage_scores) / max(1, len(document_coverage_scores)), 3
        ),
        abstention_accuracy=round(abstention_hits / total, 3),
        keyword_coverage=round(sum(keyword_scores) / total, 3),
        average_latency_ms=round(sum(latencies) / total, 2),
        cases=results,
    )
