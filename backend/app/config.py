"""Application settings, environment-driven (prefix ``VS_``)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Override any field via ``VS_``-prefixed env vars."""

    model_config = SettingsConfigDict(env_prefix="VS_", env_file=".env", extra="ignore")

    app_name: str = "virtual-staging-api"
    version: str = "0.1.0"
    environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
