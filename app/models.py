"""Pydantic models and schemas for JobMatch."""

from typing import List, Optional
from pydantic import BaseModel, Field


class JobBase(BaseModel):
    """Base schema for a job posting."""

    title: str = Field(..., description="Job role title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Geographic location or city")
    description: str = Field(..., description="Detailed job description and responsibilities")
    skills: str = Field(..., description="Comma-separated or list of key skills required")
    salary: Optional[str] = Field(None, description="Reported or estimated salary range")
    remote: bool = Field(False, description="Whether the role is remote-friendly")


class JobRecord(JobBase):
    """Full database job record including primary key identifier."""

    id: int = Field(..., description="Unique job ID")


class JobExplanation(BaseModel):
    """Structured LLM explanation of candidate-job fit."""

    match_score: int = Field(..., ge=0, le=100, description="Fit score between 0 and 100")
    matched_skills: List[str] = Field(default_factory=list, description="Skills present in both job and query/resume")
    missing_skills: List[str] = Field(default_factory=list, description="Skills required by the job but missing in profile")
    reason: str = Field(..., description="Concise explanation justifying the score and fit")


class JobSearchResult(JobRecord):
    """Job search result enriched with relevance scoring and retrieval mode."""

    score: float = Field(..., description="Relevance score (similarity, ts_rank, or RRF)")
    retrieval_mode: str = Field(..., description="Retrieval method used: vector, keyword, hybrid, or rerank")
    match_explanation: Optional[JobExplanation] = Field(
        None, description="Optional LLM-generated skill match explanation"
    )


class SearchRequest(BaseModel):
    """Search query payload supporting both keyword queries and full resume text."""

    query: Optional[str] = Field(None, description="Search keywords, job title, or role query")
    resume_text: Optional[str] = Field(None, description="Full resume or candidate background text")
    mode: str = Field("hybrid", description="Search mode: vector, keyword, hybrid, or rerank")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    location: Optional[str] = Field(None, description="Filter by location (case-insensitive substring)")
    remote: Optional[bool] = Field(None, description="Filter for remote roles only")
    explain: bool = Field(True, description="Whether to generate structured fit explanations for top results")
    reranker_type: str = Field("cross_encoder", description="Reranker strategy: cross_encoder or llm")


class SearchResponse(BaseModel):
    """Encapsulates search results and execution metadata."""

    query: str = Field(..., description="Effective query or resume text used for retrieval")
    mode: str = Field(..., description="Retrieval mode executed")
    total: int = Field(..., description="Number of jobs retrieved")
    latency_ms: float = Field(..., description="Total execution latency in milliseconds")
    results: List[JobSearchResult] = Field(default_factory=list, description="Ranked job search results")


class ExplainRequest(BaseModel):
    """Request payload for standalone candidate fit explanation."""

    query_or_resume: str = Field(..., description="Candidate query or resume text")
    job_id: int = Field(..., description="Target job posting ID to explain fit against")
