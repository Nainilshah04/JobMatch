"""Main FastAPI application for JobMatch.

Provides production-ready REST endpoints:
- GET /health: Health check and database status
- POST /search: Semantic, keyword, hybrid, and reranked job retrieval with LLM explanation
- POST /explain: Standalone candidate-job fit explanation
- GET /: Serves modern job search interface
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.db import init_db, get_db_cursor
from app.llm import explain_job_match, explain_top_jobs
from app.models import (
    ExplainRequest,
    JobExplanation,
    JobSearchResult,
    SearchRequest,
    SearchResponse,
)
from app.rerank import rerank_candidates
from app.retrieval import search_hybrid, search_keyword, search_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler verifying database schema on startup."""
    try:
        init_db()
        logger.info("Application started: Database schema verified.")
    except Exception as e:
        logger.warning(f"Could not connect to database on startup ({e}). App running in offline/readiness state.")
    yield


app = FastAPI(
    title="JobMatch API",
    description="Production-style AI Job-Matching Backend with Semantic Search, Hybrid RRF, Reranking, and Structured LLM Explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local development and external UI clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health", tags=["System"])
def health_check():
    """Service health check and database connectivity diagnostic."""
    settings = get_settings()
    db_status = "disconnected"
    job_count = 0

    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM jobs;")
            row = cur.fetchone()
            if row:
                job_count = row[0]
                db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "ok",
        "service": "JobMatch",
        "version": "1.0.0",
        "database": {
            "status": db_status,
            "total_jobs": job_count,
        },
        "models": {
            "embedding": settings.EMBEDDING_MODEL_NAME,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "llm_provider": settings.LLM_PROVIDER,
            "llm_configured": bool(settings.LLM_API_KEY and settings.LLM_API_KEY != "your_llm_api_key_here"),
        },
    }


@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
def search_jobs(payload: SearchRequest):
    """Execute job search across vector, keyword, hybrid, or reranked pipelines.
    
    Supports:
    - query: short keywords or job titles
    - resume_text: unstructured candidate resume or career summary
    - mode: 'vector', 'keyword', 'hybrid', or 'rerank'
    - filters: location (case-insensitive substring) and remote (boolean)
    - explain: generates structured LLM fit analysis for top 5 candidates
    """
    effective_query = (payload.query or "").strip()
    if not effective_query and payload.resume_text:
        effective_query = payload.resume_text.strip()

    if not effective_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'query' or 'resume_text' in search payload.",
        )

    start_time = time.time()
    mode = payload.mode.lower().strip()
    results: List[JobSearchResult] = []

    try:
        if mode == "vector":
            results = search_vector(
                query_text=effective_query,
                top_k=payload.top_k,
                location=payload.location,
                remote=payload.remote,
            )
        elif mode == "keyword":
            results = search_keyword(
                query_text=effective_query,
                top_k=payload.top_k,
                location=payload.location,
                remote=payload.remote,
            )
        elif mode == "hybrid":
            results = search_hybrid(
                query_text=effective_query,
                top_k=payload.top_k,
                location=payload.location,
                remote=payload.remote,
            )
        elif mode == "rerank":
            # Retrieve top 20 candidates via hybrid RRF search
            candidate_pool = search_hybrid(
                query_text=effective_query,
                top_k=20,
                location=payload.location,
                remote=payload.remote,
            )
            # Rerank candidate pool using cross-encoder or LLM
            reranked_pool = rerank_candidates(
                query=effective_query,
                candidates=candidate_pool,
                reranker_type="llm" if payload.reranker_type == "llm" else "cross_encoder",
            )
            results = reranked_pool[: payload.top_k]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid search mode: '{payload.mode}'. Valid modes: vector, keyword, hybrid, rerank.",
            )

        # Generate structured explanations for top results if requested
        if payload.explain and results:
            results = explain_top_jobs(
                query_or_resume=effective_query,
                jobs=results,
                top_n=min(5, len(results)),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during search execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database or retrieval engine error: {str(e)}",
        )

    latency_ms = round((time.time() - start_time) * 1000.0, 2)

    return SearchResponse(
        query=effective_query,
        mode=mode,
        total=len(results),
        latency_ms=latency_ms,
        results=results,
    )


@app.post("/explain", response_model=JobExplanation, tags=["Analysis"])
def explain_single_job(payload: ExplainRequest):
    """Generate structured candidate-job fit analysis for a specific job ID."""
    with get_db_cursor() as cur:
        cur.execute(
            """
            SELECT id, title, company, location, description, skills, salary, remote
            FROM jobs
            WHERE id = %s;
            """,
            (payload.job_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job ID {payload.job_id} not found in database.",
            )

        job_result = JobSearchResult(
            id=row[0],
            title=row[1],
            company=row[2],
            location=row[3],
            description=row[4],
            skills=row[5],
            salary=row[6],
            remote=row[7],
            score=1.0,
            retrieval_mode="direct_lookup",
        )

    return explain_job_match(payload.query_or_resume, job_result)


@app.get("/", tags=["Frontend"])
def serve_ui():
    """Serve the single-page job search application."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "JobMatch API is running. Visit /docs for Swagger API documentation."}
