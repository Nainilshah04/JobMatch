"""Configuration management for JobMatch.

Loads environment variables with fallbacks and provides strongly-typed
access to database and model configurations.
"""

import os
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


class Settings:
    """Application runtime settings."""

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/jobmatch",
    )

    # Embedding Model
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = 384

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
