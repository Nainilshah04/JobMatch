"""Reranking module supporting Cross-Encoder and LLM scoring.

Takes top candidates from initial retrieval (e.g. top 20 from hybrid search)
and rescores them using a high-capacity cross-encoder or LLM scoring call.
"""

from functools import lru_cache
import logging
import math
from typing import List, Literal, Tuple
from sentence_transformers import CrossEncoder

from app.llm import rerank_with_llm
from app.models import JobSearchResult

logger = logging.getLogger(__name__)

RerankerType = Literal["cross_encoder", "llm"]
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_cross_encoder_model() -> CrossEncoder:
    """Load and cache the CrossEncoder model singleton."""
    logger.info(f"Loading cached CrossEncoder model: {CROSS_ENCODER_MODEL_NAME}")
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


def _sigmoid(x: float) -> float:
    """Map unbounded cross-encoder logit to calibrated [0, 1] probability."""
    return 1.0 / (1.0 + math.exp(-x))


def rerank_with_cross_encoder(
    query: str,
    candidates: List[JobSearchResult],
) -> List[JobSearchResult]:
    """Rerank candidates using a full-attention Cross-Encoder.
    
    Design Decision (Cross-Encoder vs Bi-Encoder):
    Bi-encoders compute query and document representations independently, enabling
    fast vector index search but missing cross-token interactions.
    A Cross-Encoder performs joint cross-attention across all (query, document) token pairs.
    While computationally too heavy to evaluate across thousands of jobs in the database,
    evaluating it on the top 20 hybrid candidates provides state-of-the-art precision
    with sub-50ms latency.
    """
    if not candidates:
        return []

    model = get_cross_encoder_model()

    # Prepare (query, document) text pairs
    pairs: List[Tuple[str, str]] = []
    for c in candidates:
        doc_text = (
            f"Title: {c.title}. Company: {c.company}. Location: {c.location}. "
            f"Required Skills: {c.skills}. Overview: {c.description[:300]}"
        )
        pairs.append((query, doc_text))

    # Predict raw logits
    raw_scores = model.predict(pairs)

    reranked: List[JobSearchResult] = []
    for idx, c in enumerate(candidates):
        score_val = float(raw_scores[idx])
        calibrated_score = round(_sigmoid(score_val), 4)

        reranked.append(
            JobSearchResult(
                id=c.id,
                title=c.title,
                company=c.company,
                location=c.location,
                description=c.description,
                skills=c.skills,
                salary=c.salary,
                remote=c.remote,
                score=calibrated_score,
                retrieval_mode="rerank_cross_encoder",
                match_explanation=c.match_explanation,
            )
        )

    # Sort descending by cross-encoder calibrated score
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked


def rerank_candidates(
    query: str,
    candidates: List[JobSearchResult],
    reranker_type: RerankerType = "cross_encoder",
) -> List[JobSearchResult]:
    """Rerank a candidate list using the designated reranker strategy."""
    if not candidates:
        return []

    if reranker_type == "llm":
        logger.info(f"Reranking {len(candidates)} candidates using LLM scoring...")
        return rerank_with_llm(query, candidates)
    else:
        logger.info(f"Reranking {len(candidates)} candidates using Cross-Encoder...")
        return rerank_with_cross_encoder(query, candidates)
