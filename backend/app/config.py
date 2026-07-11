"""Application settings, environment-driven (prefix ``VS_``)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Override any field via ``VS_``-prefixed env vars."""

    model_config = SettingsConfigDict(env_prefix="VS_", env_file=".env", extra="ignore")

    app_name: str = "virtual-staging-api"
    version: str = "0.1.0"
    environment: str = "development"

    # PostgreSQL 16 (docker compose service `db`).
    database_url: str = "postgresql+psycopg://vs:vs@localhost:5432/virtual_staging"

    # Raw asset storage root (local filesystem in dev; same key layout as the S3 bucket).
    storage_root: Path = Path("data/storage")

    # If true, ingestion rejects uploads missing Matterpak or cubemap files.
    require_complete_ingest: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
