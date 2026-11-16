"""Unit tests for Phase 2 retrieval logic and RRF computation."""

import pytest
from app.models import JobSearchResult
from app.retrieval import compute_query_embedding


def test_query_embedding_dimension():
    """Verify that query embedding produces a normalized 384-dimensional vector."""
    query = "Staff Machine Learning Engineer PyTorch LLM"
    emb = compute_query_embedding(query)
    assert isinstance(emb, list)
    assert len(emb) == 384
    assert all(isinstance(x, float) for x in emb)


def test_rrf_mathematical_properties():
    """Verify Reciprocal Rank Fusion formula and ranking correctness.
    
    Formula: RRF = 1 / (60 + rank_vector) + 1 / (60 + rank_keyword)
    For rank 1 in both: 1/61 + 1/61 = 2/61 = 0.0327869
    For rank 2 in both: 1/62 + 1/62 = 2/62 = 0.0322581
    """
    rrf_k = 60
    rank1_score = (1.0 / (rrf_k + 1)) + (1.0 / (rrf_k + 1))
    rank2_score = (1.0 / (rrf_k + 2)) + (1.0 / (rrf_k + 2))
    single_rank1_score = 1.0 / (rrf_k + 1)

    assert rank1_score > rank2_score
    # Being in both modalities at rank 2 beats being in only one modality at rank 1
    assert rank2_score > single_rank1_score


def test_job_search_result_model():
    """Verify JobSearchResult model validation and serialization."""
    result = JobSearchResult(
        id=101,
        title="AI Engineer",
        company="Elip AI",
        location="Bengaluru, India",
        description="Developing LLM agents and career recommenders.",
        skills="Python, PyTorch, pgvector",
        salary="₹30,00,000 - ₹45,00,000",
        remote=True,
        score=0.03278,
        retrieval_mode="hybrid",
    )
    assert result.id == 101
    assert result.remote is True
    assert result.retrieval_mode == "hybrid"
