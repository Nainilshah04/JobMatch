"""Unit tests for Phase 3 reranking (Cross-Encoder and LLM scoring)."""

import pytest
from app.models import JobSearchResult
from app.rerank import rerank_candidates, _sigmoid


@pytest.fixture
def sample_candidates():
    """Create sample candidate jobs with varying relevance to Python backend engineering."""
    return [
        JobSearchResult(
            id=1,
            title="Junior Frontend Developer",
            company="WebCraft",
            location="Remote",
            description="Developing user interfaces with React, CSS, and HTML.",
            skills="React, HTML, CSS, JavaScript",
            remote=True,
            score=0.015,
            retrieval_mode="hybrid",
        ),
        JobSearchResult(
            id=2,
            title="Senior Python Backend Engineer",
            company="Elip AI",
            location="Bengaluru, India",
            description="Architecting high-throughput microservices using Python, FastAPI, and PostgreSQL.",
            skills="Python, FastAPI, Docker, PostgreSQL, Redis",
            remote=False,
            score=0.030,
            retrieval_mode="hybrid",
        ),
        JobSearchResult(
            id=3,
            title="Quantitative Data Analyst",
            company="FinMetrics",
            location="New York, NY",
            description="Analyzing financial datasets with Excel and Tableau.",
            skills="Excel, Tableau, SQL",
            remote=False,
            score=0.012,
            retrieval_mode="hybrid",
        ),
    ]


def test_sigmoid_calibration():
    """Verify sigmoid calibration produces probabilities in [0, 1]."""
    assert round(_sigmoid(0.0), 2) == 0.50
    assert _sigmoid(5.0) > 0.99
    assert _sigmoid(-5.0) < 0.01


def test_cross_encoder_rerank_accuracy(sample_candidates):
    """Verify that Cross-Encoder ranks the Python Backend role #1 for a Python query."""
    query = "Senior Python Engineer with FastAPI and PostgreSQL"
    reranked = rerank_candidates(query, sample_candidates, reranker_type="cross_encoder")

    assert len(reranked) == len(sample_candidates)
    # The Senior Python Backend Engineer must be ranked #1
    assert reranked[0].id == 2
    assert reranked[0].retrieval_mode == "rerank_cross_encoder"
    # Scores must be sorted in descending order
    assert reranked[0].score >= reranked[1].score >= reranked[2].score


def test_llm_rerank_fallback(sample_candidates):
    """Verify that LLM reranking strategy produces sorted results."""
    query = "React frontend web developer"
    reranked = rerank_candidates(query, sample_candidates, reranker_type="llm")

    assert len(reranked) == len(sample_candidates)
    assert reranked[0].retrieval_mode == "rerank_llm"
    # Frontend role should score higher for a frontend query
    assert reranked[0].id == 1
    assert reranked[0].score >= reranked[1].score
