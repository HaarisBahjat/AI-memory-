"""
============================================================
tests/conftest.py — Pytest Configuration & Global Fixtures
============================================================
PURPOSE:
    Sets up the test environment before any test module is
    collected or imported. This is the correct place to inject
    environment variables that config.py reads at import time.

    Without this, every test file that imports from app.*
    would fail with a Pydantic ValidationError because
    SUPABASE_DB_URL and OPENAI_API_KEY are required fields
    in Settings and have no defaults.

    The os.environ mutations must happen BEFORE any import of
    app modules — conftest.py is loaded first by pytest, making
    it the right place for this initialization.

DEPENDENCY OVERRIDES:
    We use FastAPI's dependency_overrides mechanism to replace
    get_db and get_redis with non-connecting mock versions.
    This is the correct, non-fragile way to mock FastAPI
    dependencies — it goes through the same DI resolution
    path as production code, not bypassing it with patch().
============================================================
"""
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# Set required env vars before any app module is imported.
# These are test-only stub values — no real services are called
# in unit tests because all external I/O is mocked.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql+asyncpg://test:test@localhost/testdb")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-not-real")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-minimum-32-characters-1234")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")


def make_mock_db(user_row=None):
    """
    Returns a factory that yields a mock AsyncSession.
    The mock responds to .execute() with the provided user_row
    for get_current_user dependency lookups.
    """
    async def _mock_get_db():
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        if user_row is not None:
            result = MagicMock()
            result.mappings.return_value.first.return_value = user_row
            mock_session.execute = AsyncMock(return_value=result)
        else:
            # Default: return a mock that can be configured per-test
            result = MagicMock()
            result.mappings.return_value.first.return_value = None
            result.first.return_value = None
            mock_session.execute = AsyncMock(return_value=result)

        yield mock_session

    return _mock_get_db


def make_mock_redis():
    """Returns a factory that yields a mock Redis client."""
    async def _mock_get_redis():
        mock_redis = AsyncMock()
        mock_redis.lrange = AsyncMock(return_value=[])
        mock_redis.rpush = AsyncMock(return_value=1)
        mock_redis.ltrim = AsyncMock(return_value=True)
        mock_redis.expire = AsyncMock(return_value=True)
        mock_redis.exists = AsyncMock(return_value=0)
        mock_redis.delete = AsyncMock(return_value=1)
        yield mock_redis

    return _mock_get_redis


# Default authenticated user row (returned by get_current_user dep lookups)
DEFAULT_TEST_USER = {
    "user_id": "test-user-uuid",
    "email": "test@example.com",
}


@pytest.fixture(autouse=False)
def mock_auth_app():
    """
    Fixture that overrides get_db and get_redis with mocks
    for tests that need JWT auth to work without live infra.

    Usage:
        def test_something(mock_auth_app):
            client = mock_auth_app
            response = client.get("/api/v1/user/me", headers=auth_header())
    """
    from fastapi.testclient import TestClient
    from main import app
    from app.core.database import get_db
    from app.core.redis_client import get_redis

    app.dependency_overrides[get_db] = make_mock_db(DEFAULT_TEST_USER)
    app.dependency_overrides[get_redis] = make_mock_redis()

    yield TestClient(app)

    app.dependency_overrides.clear()
