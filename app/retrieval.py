"""Retrieval module implementing Vector, Keyword, and Hybrid (RRF) search.

Provides three distinct retrieval modes over the PostgreSQL `jobs` table,
with an automatic graceful fallback to the local dataset if the database is offline.
"""

import csv
from functools import lru_cache
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Limit PyTorch CPU threads to 1 to stay strictly below 512MB RAM on Render
torch.set_num_threads(1)

from app.config import get_settings
from app.db import get_db_connection, is_db_connected
from app.models import JobSearchResult

logger = logging.getLogger(__name__)


def get_active_backend() -> str:
    """Return the active storage/retrieval backend identifier ('postgres+pgvector' or 'local-fallback')."""
    return "postgres+pgvector" if is_db_connected() else "local-fallback"


# Cached in-memory dataset for offline/local resilience
_LOCAL_JOBS_CACHE: Optional[List[dict]] = None
_LOCAL_EMBEDDINGS_CACHE: Optional[np.ndarray] = None


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model singleton."""
    settings = get_settings()
    logger.info(f"Loading cached embedding model: {settings.EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


def build_job_embedding_text(
    title: str,
    company: str,
    location: str,
    skills: str,
    description: str,
) -> str:
    """Format a dense, structured semantic text representation of a job posting for embedding."""
    clean_desc = " ".join(description[:400].split())
    return (
        f"Job Title: {title}. "
        f"Company: {company}. "
        f"Location: {location}. "
        f"Key Skills: {skills}. "
        f"Summary: {clean_desc}"
    )


def compute_query_embedding(text: str) -> List[float]:
    """Generate a normalized 384-dimensional vector embedding for search input."""
    clean_text = " ".join(text[:500].split())
    model = get_embedding_model()
    vector = model.encode(clean_text, normalize_embeddings=True)
    return vector.tolist()


def _load_local_jobs() -> Tuple[List[dict], np.ndarray]:
    """Load and precompute local dataset embeddings for offline resilience."""
    global _LOCAL_JOBS_CACHE, _LOCAL_EMBEDDINGS_CACHE
    if _LOCAL_JOBS_CACHE is not None and _LOCAL_EMBEDDINGS_CACHE is not None:
        return _LOCAL_JOBS_CACHE, _LOCAL_EMBEDDINGS_CACHE

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jobs.csv")
    if not os.path.exists(csv_path):
        from scripts.generate_jobs import generate_jobs_dataset
        generate_jobs_dataset(csv_path, count=850)

    jobs: List[dict] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "company": row["company"],
                "location": row["location"],
                "description": row["description"],
                "skills": row["skills"],
                "salary": row["salary"] if row.get("salary") else None,
                "remote": str(row.get("remote", "")).lower() == "true",
            })

    _LOCAL_JOBS_CACHE = jobs

    # Load precomputed embeddings via memory-mapping to prevent runtime memory spikes
    emb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "job_embeddings.npy")
    if os.path.exists(emb_path):
        _LOCAL_EMBEDDINGS_CACHE = np.load(emb_path, mmap_mode="r")
    else:
        texts = [
            build_job_embedding_text(
                title=j["title"],
                company=j["company"],
                location=j["location"],
                skills=j["skills"],
                description=j["description"],
            )
            for j in jobs
        ]
        model = get_embedding_model()
        _LOCAL_EMBEDDINGS_CACHE = model.encode(texts, batch_size=32, normalize_embeddings=True)
    return _LOCAL_JOBS_CACHE, _LOCAL_EMBEDDINGS_CACHE


def _search_local(
    query_text: str,
    top_k: int = 10,
    mode: str = "hybrid",
    location: Optional[str] = None,
    remote: Optional[bool] = None,
) -> List[JobSearchResult]:
    """Perform in-memory retrieval when PostgreSQL database is unreachable."""
    jobs, embeddings = _load_local_jobs()

    # Filter candidate indices
    candidate_indices = []
    for idx, j in enumerate(jobs):
        if location and location.lower() not in j["location"].lower():
            continue
        if remote is not None and j["remote"] != remote:
            continue
        candidate_indices.append(idx)

    if not candidate_indices:
        return []

    # Vector scoring
    query_vec = np.array(compute_query_embedding(query_text))
    candidate_embeddings = embeddings[candidate_indices]
    cosine_sims = np.dot(candidate_embeddings, query_vec)

    # Keyword scoring
    query_tokens = set(re.findall(r"\w+", query_text.lower()))
    keyword_scores = []
    for idx in candidate_indices:
        j = jobs[idx]
        title_toks = set(re.findall(r"\w+", j["title"].lower()))
        skill_toks = set(re.findall(r"\w+", j["skills"].lower()))
        desc_toks = set(re.findall(r"\w+", j["description"].lower()))

        score = (
            len(query_tokens & title_toks) * 3.0
            + len(query_tokens & skill_toks) * 2.0
            + len(query_tokens & desc_toks) * 0.5
        )
        keyword_scores.append(score)

    keyword_scores = np.array(keyword_scores)

    # Produce ranking according to requested mode
    if mode == "vector":
        sorted_order = np.argsort(-cosine_sims)[:top_k]
        results = []
        for i in sorted_order:
            j = jobs[candidate_indices[i]]
            results.append(
                JobSearchResult(
                    id=j["id"],
                    title=j["title"],
                    company=j["company"],
                    location=j["location"],
                    description=j["description"],
                    skills=j["skills"],
                    salary=j["salary"],
                    remote=j["remote"],
                    score=round(float(cosine_sims[i]), 4),
                    retrieval_mode="vector",
                )
            )
        return results

    if mode == "keyword":
        sorted_order = np.argsort(-keyword_scores)
        results = []
        for i in sorted_order:
            if keyword_scores[i] <= 0:
                continue
            if len(results) >= top_k:
                break
            j = jobs[candidate_indices[i]]
            results.append(
                JobSearchResult(
                    id=j["id"],
                    title=j["title"],
                    company=j["company"],
                    location=j["location"],
                    description=j["description"],
                    skills=j["skills"],
                    salary=j["salary"],
                    remote=j["remote"],
                    score=round(float(keyword_scores[i]), 4),
                    retrieval_mode="keyword",
                )
            )
        return results

    # Hybrid RRF (k=60)
    vec_rank_order = np.argsort(-cosine_sims)
    kw_rank_order = np.argsort(-keyword_scores)

    rrf_scores: Dict[int, float] = {}
    rrf_k = 60

    for rank, pos in enumerate(vec_rank_order, start=1):
        rrf_scores[pos] = rrf_scores.get(pos, 0.0) + (1.0 / (rrf_k + rank))

    # Drop candidates with keyword score == 0 before RRF so they don't add noise to the fusion
    for rank, pos in enumerate(kw_rank_order, start=1):
        if keyword_scores[pos] > 0:
            rrf_scores[pos] = rrf_scores.get(pos, 0.0) + (1.0 / (rrf_k + rank))

    sorted_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for pos, rrf_val in sorted_by_rrf:
        j = jobs[candidate_indices[pos]]
        results.append(
            JobSearchResult(
                id=j["id"],
                title=j["title"],
                company=j["company"],
                location=j["location"],
                description=j["description"],
                skills=j["skills"],
                salary=j["salary"],
                remote=j["remote"],
                score=round(float(rrf_val), 6),
                retrieval_mode="hybrid",
            )
        )
    return results


def search_vector(
    query_text: str,
    top_k: int = 10,
    location: Optional[str] = None,
    remote: Optional[bool] = None,
) -> List[JobSearchResult]:
    """Retrieve jobs using dense cosine similarity via pgvector (or fallback)."""
    if not is_db_connected():
        logger.info("Database unreachable. Executing search_vector via local fallback engine.")
        return _search_local(query_text, top_k=top_k, mode="vector", location=location, remote=remote)

    query_vector = compute_query_embedding(query_text)
    loc_filter = f"%{location.strip()}%" if location else None

    sql = """
        SELECT 
            id, title, company, location, description, skills, salary, remote,
            ROUND((1.0 - (embedding <=> %s::vector))::numeric, 4) AS score
        FROM jobs
        WHERE embedding IS NOT NULL
          AND (%s::varchar IS NULL OR location ILIKE %s)
          AND (%s::boolean IS NULL OR remote = %s)
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
    """

    results: List[JobSearchResult] = []
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_vector,
                    loc_filter,
                    loc_filter,
                    remote,
                    remote,
                    query_vector,
                    top_k,
                ),
            )
            rows = cur.fetchall()
            for r in rows:
                results.append(
                    JobSearchResult(
                        id=r[0],
                        title=r[1],
                        company=r[2],
                        location=r[3],
                        description=r[4],
                        skills=r[5],
                        salary=r[6],
                        remote=r[7],
                        score=float(r[8]),
                        retrieval_mode="vector",
                    )
                )

    return results


def search_keyword(
    query_text: str,
    top_k: int = 10,
    location: Optional[str] = None,
    remote: Optional[bool] = None,
) -> List[JobSearchResult]:
    """Retrieve jobs using PostgreSQL full-text search and ts_rank (or fallback)."""
    if not is_db_connected():
        logger.info("Database unreachable. Executing search_keyword via local fallback engine.")
        return _search_local(query_text, top_k=top_k, mode="keyword", location=location, remote=remote)

    clean_query = query_text.strip()
    if not clean_query:
        return []

    loc_filter = f"%{location.strip()}%" if location else None

    sql = """
        WITH query_parsed AS (
            SELECT COALESCE(
                NULLIF(websearch_to_tsquery('english', %s), ''::tsquery),
                plainto_tsquery('english', %s)
            ) AS q
        )
        SELECT 
            j.id, j.title, j.company, j.location, j.description, j.skills, j.salary, j.remote,
            ROUND(ts_rank(j.tsv, qp.q)::numeric, 4) AS score
        FROM jobs j, query_parsed qp
        WHERE qp.q IS NOT NULL
          AND j.tsv @@ qp.q
          AND (%s::varchar IS NULL OR j.location ILIKE %s)
          AND (%s::boolean IS NULL OR j.remote = %s)
        ORDER BY score DESC
        LIMIT %s;
    """

    results: List[JobSearchResult] = []
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    clean_query,
                    clean_query,
                    loc_filter,
                    loc_filter,
                    remote,
                    remote,
                    top_k,
                ),
            )
            rows = cur.fetchall()
            for r in rows:
                results.append(
                    JobSearchResult(
                        id=r[0],
                        title=r[1],
                        company=r[2],
                        location=r[3],
                        description=r[4],
                        skills=r[5],
                        salary=r[6],
                        remote=r[7],
                        score=float(r[8]),
                        retrieval_mode="keyword",
                    )
                )

    return results


def search_hybrid(
    query_text: str,
    top_k: int = 10,
    rrf_k: int = 60,
    candidate_pool_size: int = 50,
    location: Optional[str] = None,
    remote: Optional[bool] = None,
) -> List[JobSearchResult]:
    """Retrieve jobs using Reciprocal Rank Fusion (RRF, k=60) combining vector and keyword search."""
    if not is_db_connected():
        logger.info("Database unreachable. Executing search_hybrid via local fallback engine.")
        return _search_local(query_text, top_k=top_k, mode="hybrid", location=location, remote=remote)

    # 1. Fetch top candidates from both modalities
    vector_results = search_vector(
        query_text=query_text,
        top_k=candidate_pool_size,
        location=location,
        remote=remote,
    )
    keyword_results = search_keyword(
        query_text=query_text,
        top_k=candidate_pool_size,
        location=location,
        remote=remote,
    )

    # 2. Accumulate RRF scores
    candidate_map: Dict[int, Tuple[JobSearchResult, float]] = {}

    for rank, item in enumerate(vector_results, start=1):
        rrf_increment = 1.0 / (rrf_k + rank)
        if item.id in candidate_map:
            record, current_score = candidate_map[item.id]
            candidate_map[item.id] = (record, current_score + rrf_increment)
        else:
            candidate_map[item.id] = (item, rrf_increment)

    for rank, item in enumerate(keyword_results, start=1):
        rrf_increment = 1.0 / (rrf_k + rank)
        if item.id in candidate_map:
            record, current_score = candidate_map[item.id]
            candidate_map[item.id] = (record, current_score + rrf_increment)
        else:
            candidate_map[item.id] = (item, rrf_increment)

    # 3. Sort candidates by RRF score descending
    sorted_candidates = sorted(
        candidate_map.values(),
        key=lambda pair: pair[1],
        reverse=True,
    )

    # 4. Construct final top_k results
    hybrid_results: List[JobSearchResult] = []
    for record, score in sorted_candidates[:top_k]:
        hybrid_results.append(
            JobSearchResult(
                id=record.id,
                title=record.title,
                company=record.company,
                location=record.location,
                description=record.description,
                skills=record.skills,
                salary=record.salary,
                remote=record.remote,
                score=round(score, 6),
                retrieval_mode="hybrid",
            )
        )

    return hybrid_results
