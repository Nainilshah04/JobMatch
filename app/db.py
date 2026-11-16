"""Database connection and schema management with pgvector and full-text search.

This module sets up PostgreSQL connection handling, pgvector extension registration,
schema initialization, and connection health diagnostics.
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
_db_connected: bool | None = None


def is_db_connected() -> bool:
    """Check whether PostgreSQL database is reachable."""
    global _db_connected
    settings = get_settings()
    try:
        conn = psycopg2.connect(settings.DATABASE_URL, connect_timeout=2)
        conn.close()
        _db_connected = True
        return True
    except Exception as e:
        logger.debug(f"Database reachability check failed: {e}")
        _db_connected = False
        return False


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
            connect_timeout=3,
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
    """Create pgvector extension, jobs table, and required search indexes."""
    settings = get_settings()
    dimension = settings.EMBEDDING_DIMENSION

    ddl_statements = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
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
        "CREATE INDEX IF NOT EXISTS jobs_tsv_idx ON jobs USING GIN (tsv);",
        """
        CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw_idx 
        ON jobs USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """,
        "CREATE INDEX IF NOT EXISTS jobs_remote_idx ON jobs (remote);",
        "CREATE INDEX IF NOT EXISTS jobs_location_idx ON jobs (location);"
    ]

    with get_db_cursor(commit=True) as cur:
        for stmt in ddl_statements:
            cur.execute(stmt)
    logger.info("Database schema initialized successfully.")
