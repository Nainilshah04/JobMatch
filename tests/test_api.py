"""API Integration tests for FastAPI endpoints."""

from unittest.mock import patch
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.models import JobSearchResult, JobExplanation

client = TestClient(app)


def test_health_endpoint():
    """Verify that /health responds with 200 and expected diagnostic structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "JobMatch"
    assert "database" in data
    assert "models" in data
    assert data["models"]["embedding_dimension"] == 384


def test_root_serves_html():
    """Verify that GET / returns the frontend HTML application."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "JobMatch" in response.text
    assert "Candidate Fit Assessment" in response.text or "pgvector" in response.text


def test_search_validation_empty_query():
    """Verify that search rejects empty payloads with 400."""
    response = client.post("/search", json={"query": "", "resume_text": ""})
    assert response.status_code == 400
    assert "Must provide either 'query' or 'resume_text'" in response.json()["detail"]


def test_search_validation_invalid_mode():
    """Verify that search rejects unsupported retrieval modes with 400."""
    response = client.post("/search", json={"query": "Python Engineer", "mode": "quantum_search"})
    assert response.status_code == 400
    assert "Invalid search mode" in response.json()["detail"]


@patch("app.main.search_hybrid")
def test_search_hybrid_mock_pipeline(mock_hybrid):
    """Verify end-to-end search pipeline and response structure."""
    mock_job = JobSearchResult(
        id=1,
        title="Python Backend Engineer",
        company="Elip AI",
        location="Bengaluru, India",
        description="Build scalable microservices and AI agent career recommenders.",
        skills="Python, FastAPI, Docker",
        remote=True,
        score=0.032,
        retrieval_mode="hybrid",
    )
    mock_hybrid.return_value = [mock_job]

    response = client.post(
        "/search",
        json={
            "query": "Python and FastAPI",
            "mode": "hybrid",
            "top_k": 5,
            "explain": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Python and FastAPI"
    assert data["mode"] == "hybrid"
    assert data["total"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["title"] == "Python Backend Engineer"
    # Ensure explanation is integrated into result
    assert data["results"][0]["match_explanation"] is not None
    assert "Python" in data["results"][0]["match_explanation"]["matched_skills"]


def test_upload_resume_invalid_format():
    """Verify that non-PDF files are rejected with 400."""
    response = client.post(
        "/upload-resume",
        files={"file": ("resume.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF" in response.json()["detail"]
