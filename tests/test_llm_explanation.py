"""Unit tests for Phase 4 LLM structured explanation and Pydantic validation."""

import pytest
from pydantic import ValidationError
from app.models import JobSearchResult, JobExplanation
from app.llm import _extract_json_snippet, explain_job_match, explain_top_jobs


@pytest.fixture
def sample_job():
    return JobSearchResult(
        id=42,
        title="Senior Python Backend Engineer",
        company="Elip AI",
        location="Bengaluru, India",
        description="Build scalable microservices and AI agent career recommenders.",
        skills="Python, FastAPI, Docker, PostgreSQL, Redis, Kafka",
        remote=True,
        score=0.88,
        retrieval_mode="rerank_cross_encoder",
    )


def test_job_explanation_pydantic_schema():
    """Verify that JobExplanation validates valid data and rejects invalid score ranges."""
    valid_data = {
        "match_score": 85,
        "matched_skills": ["Python", "FastAPI"],
        "missing_skills": ["Kafka"],
        "reason": "Strong backend background, needs Kafka streaming experience.",
    }
    explanation = JobExplanation.model_validate(valid_data)
    assert explanation.match_score == 85
    assert "Python" in explanation.matched_skills
    assert "Kafka" in explanation.missing_skills

    # Invalid score (> 100) must fail validation
    invalid_data = {**valid_data, "match_score": 120}
    with pytest.raises(ValidationError):
        JobExplanation.model_validate(invalid_data)

    # Invalid score (< 0) must fail validation
    invalid_data_negative = {**valid_data, "match_score": -10}
    with pytest.raises(ValidationError):
        JobExplanation.model_validate(invalid_data_negative)


def test_json_snippet_extraction():
    """Verify JSON extraction from markdown code fences and noisy LLM responses."""
    markdown_wrapped = "Here is the result:\n```json\n{\"match_score\": 90, \"matched_skills\": [\"Python\"], \"missing_skills\": [], \"reason\": \"Great fit.\"}\n```"
    extracted = _extract_json_snippet(markdown_wrapped)
    assert extracted.startswith("{")
    assert extracted.endswith("}")
    assert '"match_score": 90' in extracted


def test_explain_job_match_structure(sample_job):
    """Verify explain_job_match generates valid JobExplanation schema and detects skills."""
    resume_text = "Experienced backend developer with 4 years in Python, FastAPI, Docker, and PostgreSQL."
    explanation = explain_job_match(resume_text, sample_job)

    assert isinstance(explanation, JobExplanation)
    assert 0 <= explanation.match_score <= 100
    assert "Python" in explanation.matched_skills
    assert "FastAPI" in explanation.matched_skills
    # Kafka was in job skills but not in the resume text
    assert "Kafka" in explanation.missing_skills
    assert len(explanation.reason) > 10


def test_explain_top_jobs_enrichment(sample_job):
    """Verify that explain_top_jobs correctly enriches only the top_n results."""
    job2 = sample_job.model_copy(update={"id": 43, "title": "DevOps Engineer"})
    job3 = sample_job.model_copy(update={"id": 44, "title": "Data Scientist"})

    jobs_list = [sample_job, job2, job3]
    enriched = explain_top_jobs("Python and Docker", jobs_list, top_n=2)

    assert len(enriched) == 3
    # Top 2 must have explanations
    assert enriched[0].match_explanation is not None
    assert enriched[1].match_explanation is not None
    # 3rd job must not have explanation
    assert enriched[2].match_explanation is None
