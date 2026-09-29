"""Pytest configuration for backend tests.

Ensures test runs execute deterministically and hermetically against
in-memory SQLite with isolated database engine singletons.
"""

from collections.abc import Generator

import pytest

import app.db
from app.config import get_settings


@pytest.fixture(autouse=True)
def _isolate_test_db(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Ensures each test gets an isolated in-memory database and clean engine singletons."""
    monkeypatch.setenv("VS_DATABASE_URL", "sqlite:///:memory:")
    app.db._engine = None
    app.db._session_factory = None
    get_settings.cache_clear()
    try:
        yield
    finally:
        app.db._engine = None
        app.db._session_factory = None
        get_settings.cache_clear()
