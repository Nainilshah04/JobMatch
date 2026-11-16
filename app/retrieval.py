"""Retrieval module implementing Vector, Keyword, and Hybrid (RRF) search.

Provides three distinct retrieval modes over the PostgreSQL `jobs` table:
1. Vector Search: Dense semantic cosine similarity using pgvector.
2. Keyword Search: Full-text search with ts_rank and GIN indexing.
3. Hybrid Search: Reciprocal Rank Fusion (RRF, k=60) combining vector and keyword ranks.
"""

from functools import lru_cache
import logging
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.db import get_db_connection
from app.models import JobSearchResult

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model singleton."""
    settings = get_settings()
    logger.info(f"Loading cached embedding model: {settings.EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


def compute_query_embedding(text: str) -> List[float]:
    """Generate a normalized 384-dimensional vector embedding for search input.
    
    If full resume text is passed, truncate to the first 500 characters
    to retain key skills and role summary within the token limit.
    """
    clean_text = " ".join(text[:500].split())
    model = get_embedding_model()
    vector = model.encode(clean_text, normalize_embeddings=True)
    return vector.tolist()


def search_vector(
    query_text: str,
    top_k: int = 10,
    location: Optional[str] = None,
    remote: Optional[bool] = None,
) -> List[JobSearchResult]:
    """Retrieve jobs using dense cosine similarity via pgvector.
    
    Design Decision (Score calculation):
    pgvector cosine distance operator `<=>` yields (1 - cosine_similarity).
    We calculate similarity score as: `1.0 - (embedding <=> query_vec)`,
    providing an intuitive 0.0 to 1.0 relevance metric.
    """
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
    """Retrieve jobs using PostgreSQL full-text search and ts_rank on weighted tsv.
    
    Uses `websearch_to_tsquery` to support natural search phrasing, boolean operators,
    and quoted phrases without throwing syntax errors on freeform user input.
    """
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
    """Retrieve jobs using Reciprocal Rank Fusion (RRF, k=60) combining vector and keyword search.
    
    Design Decision (Reciprocal Rank Fusion):
    Cosine similarity scores (bounded [0, 1]) and ts_rank scores (unbounded [0, >10])
    exist on fundamentally different scales and cannot be directly summed or multiplied.
    RRF solves this scale-mismatch by relying strictly on ordinal rank positions:
        RRF_Score(d) = SUM_m [ 1 / (k + rank_m(d)) ]
    where k=60 (Cormack et al.) penalizes lower-ranked items while stabilizing
    outlier ranks across disparate retrieval modalities.
    """
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
    # candidate_map stores: job_id -> (job_record, cumulative_rrf_score)
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
