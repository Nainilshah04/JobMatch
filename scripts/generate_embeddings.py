"""Generate and store dense vector embeddings for job records in PostgreSQL.

Uses sentence-transformers 'all-MiniLM-L6-v2' (384 dimensions) to generate
dense semantic vector representations for fast cosine retrieval via pgvector.
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2.extras import execute_batch
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.db import get_db_connection
from app.retrieval import build_job_embedding_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_and_store_embeddings(batch_size: int = 64, force: bool = False) -> int:
    """Compute embeddings for jobs with missing embeddings and update Postgres."""
    settings = get_settings()
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    logger.info(f"Model loaded. Dimension: {settings.EMBEDDING_DIMENSION}")

    query_filter = "" if force else "WHERE embedding IS NULL"
    select_sql = f"""
        SELECT id, title, company, location, skills, description
        FROM jobs
        {query_filter}
        ORDER BY id ASC;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(select_sql)
            rows = cur.fetchall()

    total = len(rows)
    if total == 0:
        logger.info("No jobs require embedding generation.")
        return 0

    logger.info(f"Found {total} jobs to embed (batch size: {batch_size}).")
    start_time = time.time()

    update_sql = """
        UPDATE jobs
        SET embedding = %s
        WHERE id = %s;
    """

    processed = 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for i in range(0, total, batch_size):
                batch_rows = rows[i : i + batch_size]
                
                # Format text representations
                texts = [
                    build_job_embedding_text(
                        title=r[1],
                        company=r[2],
                        location=r[3],
                        skills=r[4],
                        description=r[5],
                    )
                    for r in batch_rows
                ]
                
                # Compute embeddings (normalized for cosine distance)
                embeddings = model.encode(
                    texts,
                    batch_size=len(texts),
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )

                # Prepare payload: (vector, job_id)
                update_params: List[Tuple[List[float], int]] = [
                    (emb.tolist(), batch_rows[idx][0])
                    for idx, emb in enumerate(embeddings)
                ]

                execute_batch(cur, update_sql, update_params)
                conn.commit()
                processed += len(batch_rows)
                logger.info(f"Embedded & stored {processed}/{total} jobs...")

    elapsed = time.time() - start_time
    logger.info(f"Finished embedding {processed} jobs in {elapsed:.2f}s ({processed / elapsed:.1f} jobs/sec).")
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and update job embeddings in Postgres.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding inference",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate embeddings for all jobs even if already present",
    )
    args = parser.parse_args()

    try:
        generate_and_store_embeddings(batch_size=args.batch_size, force=args.force)
    except psycopg2.OperationalError as e:
        logger.error(
            "\n[Database Connection Error]: Could not connect to PostgreSQL.\n"
            "Please ensure DATABASE_URL is configured in your .env file.\n"
            f"Details: {e}"
        )
        sys.exit(1)
