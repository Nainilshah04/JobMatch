"""Database connection and schema management with pgvector and full-text search.

This module sets up PostgreSQL connection handling, pgvector extension registration,
and schema initialization with HNSW vector indexing and weighted tsvector GIN indexing.
"""

import contextlib
import logging
from typing import Generator
import psycopg2
from psycopg2.extensions import connection as PgConnection, cursor as PgCursor
from psycopg2.pool import ThreadedConnectionPool
from pgvector.psycopg2 import register_vector

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level connection pool
_pool: ThreadedConnectionPool | None = None


def get_connection_pool() -> ThreadedConnectionPool:
    """Initialize or return the existing threaded PostgreSQL connection pool."""
    global _pool
    if _pool is None:
        settings = get_settings()
        logger.info("Initializing database connection pool...")
        _pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=settings.DATABASE_URL,
        )
    return _pool


@contextlib.contextmanager
def get_db_connection() -> Generator[PgConnection, None, None]:
    """Yield a database connection from the pool, with pgvector registered."""
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        # Register pgvector type handler on this connection
        register_vector(conn)
        yield conn
    finally:
        pool.putconn(conn)


@contextlib.contextmanager
def get_db_cursor(commit: bool = False) -> Generator[PgCursor, None, None]:
    """Yield a database cursor, handling transactions and connection return."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise


def init_db() -> None:
    """Create pgvector extension, jobs table, and required search indexes.
    
    Design Decisions:
    1. pgvector extension: Enables native 384-dimensional cosine vector storage and distance operators (<=>).
    2. Generated tsvector column: Precomputes weighted text tokens (Title=A, Skills=B, Company=C, Description=D)
       avoiding runtime overhead during full-text queries.
    3. HNSW index on embedding: Provides fast approximate nearest-neighbors (ANN) search for cosine distance.
    4. GIN index on tsv: Enables fast full-text inverted index searches.
    """
    settings = get_settings()
    dimension = settings.EMBEDDING_DIMENSION

    # Schema definition statements
    ddl_statements = [
        # 1. Enable pgvector extension
        "CREATE EXTENSION IF NOT EXISTS vector;",
        
        # 2. Create jobs table with structured fields and generated tsvector
        f"""
        CREATE TABLE IF NOT EXISTS jobs (
            id SERIAL PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            company VARCHAR(255) NOT NULL,
            location VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            skills TEXT NOT NULL,
            salary VARCHAR(100),
            remote BOOLEAN DEFAULT FALSE,
            embedding vector({dimension}),
            tsv tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(skills, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(company, '')), 'C') ||
                setweight(to_tsvector('english', coalesce(description, '')), 'D')
            ) STORED
        );
        """,

        # 3. Full-text search GIN index on weighted tsv
        "CREATE INDEX IF NOT EXISTS jobs_tsv_idx ON jobs USING GIN (tsv);",

        # 4. Fast HNSW vector index for cosine similarity (<=>)
        """
        CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw_idx 
        ON jobs USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """,

        # 5. Filter indexes for metadata filtering
        "CREATE INDEX IF NOT EXISTS jobs_remote_idx ON jobs (remote);",
        "CREATE INDEX IF NOT EXISTS jobs_location_idx ON jobs (location);"
    ]

    logger.info("Initializing database schema and indexes...")
    with get_db_cursor(commit=True) as cur:
        for stmt in ddl_statements:
            cur.execute(stmt)
    logger.info("Database schema initialized successfully.")
