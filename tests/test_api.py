from pathlib import Path

import pytest


pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from app.main import app, get_service
from tests.test_service import make_service


@pytest.fixture
def client(tmp_path: Path):
    service = make_service(tmp_path)
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_successful_query_end_to_end(client: TestClient):
    response = client.post(
        "/query",
        json={"question": "How many minutes of physical activity are recommended?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["abstained"] is False
    assert payload["citations"][0]["page"] == 1


def test_abstaining_query_end_to_end(client: TestClient):
    response = client.post(
        "/query",
        json={"question": "Diagnose me and tell me which medicine to take."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["abstained"] is True
    assert payload["citations"] == []
