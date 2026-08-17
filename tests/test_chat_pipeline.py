"""
============================================================
tests/test_chat_pipeline.py — End-to-End API Integration Tests
============================================================
PURPOSE:
    Verifies the full chat pipeline from HTTP request
    through safety screener → RAG orchestrator → response.
    Uses pytest-asyncio and FastAPI's TestClient.

PHASE 2 CHANGES:
    - All tests now use JWT Bearer token auth headers
    - user_id removed from chat request body
    - get_db/get_redis injected via dependency_overrides
      (the correct DI-aware mocking approach in FastAPI)

NOTE:
    Full end-to-end tests require live Supabase + Redis connections.
    Mark tests that require live infra with @pytest.mark.integration
    so they can be skipped in CI without live credentials.
============================================================
"""
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from app.core.database import get_db
from app.core.redis_client import get_redis
from app.core.security import create_access_token


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def auth_header(user_id: str = "test-user-uuid") -> dict:
    """Creates a valid Bearer token header for testing."""
    token = create_access_token(data={"sub": user_id})
    return {"Authorization": f"Bearer {token}"}


MOCK_AUTH_USER = {"user_id": "test-user-uuid", "email": "t@t.com"}


def make_db_with_user(user_row=None):
    """Returns a get_db override that returns user_row for get_current_user lookups."""
    async def _get_db():
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        result = MagicMock()
        result.mappings.return_value.first.return_value = user_row
        mock_session.execute = AsyncMock(return_value=result)
        yield mock_session
    return _get_db


def make_redis_mock():
    async def _get_redis():
        mock = AsyncMock()
        mock.lrange = AsyncMock(return_value=[])
        mock.rpush = AsyncMock(return_value=1)
        mock.ltrim = AsyncMock(return_value=True)
        mock.expire = AsyncMock(return_value=True)
        mock.exists = AsyncMock(return_value=0)
        mock.delete = AsyncMock(return_value=1)
        pipeline_mock = AsyncMock()
        pipeline_mock.__aenter__ = AsyncMock(return_value=pipeline_mock)
        pipeline_mock.__aexit__ = AsyncMock(return_value=False)
        pipeline_mock.execute = AsyncMock(return_value=[1, True, True])
        mock.pipeline = MagicMock(return_value=pipeline_mock)
        yield mock
    return _get_redis


@contextmanager
def auth_client(user_row=None):
    """Context manager that returns a TestClient with mocked DB + Redis."""
    row = user_row if user_row is not None else MOCK_AUTH_USER
    app.dependency_overrides[get_db] = make_db_with_user(row)
    app.dependency_overrides[get_redis] = make_redis_mock()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_redis, None)


# ============================================================
# Health Endpoint (no auth needed)
# ============================================================

class TestHealthEndpoint:
    """Health check endpoint tests — no external dependencies."""

    def test_health_endpoint_exists(self):
        """Health endpoint should be reachable."""
        # Health checks may fail on DB/Redis in unit test mode
        # but the endpoint itself should always be reachable
        client = TestClient(app)
        response = client.get("/api/v1/health")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "database" in data
        assert "redis" in data

    def test_root_endpoint(self):
        """Root endpoint should return API info."""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "docs" in data


# ============================================================
# Safety Screener Tests
# ============================================================

class TestChatEndpointSafetyOverride:
    """
    Tests that verify the safety screener intercepts
    crisis messages BEFORE the RAG pipeline is called.
    Phase 2: Now requires JWT auth, but screener still fires first.
    """

    def test_crisis_message_bypasses_llm(self):
        """
        A message with suicidal ideation should return a
        CRISIS_TRIAGE response immediately without calling
        the LLM or database.
        """
        with auth_client() as client, \
             patch("app.api.v1.chat.retrieval_engine.run_hybrid_rag_pipeline") as mock_rag:

            response = client.post(
                "/api/v1/chat",
                headers=auth_header(),
                json={"message": "I want to kill myself, I can't take this anymore"},
            )
            # RAG pipeline should NEVER have been called
            mock_rag.assert_not_called()

        data = response.json()
        assert data["type"] == "CRISIS_TRIAGE"
        assert "message" in data
        assert "resources" in data
        assert len(data["resources"]) > 0

    def test_safe_message_reaches_rag(self):
        """
        A safe wellness message should pass the screener and
        reach the RAG pipeline.
        """
        mock_rag_result = {
            "response": "I understand you've been feeling anxious. Let's explore some techniques.",
            "memories_used": 2,
            "debug": {
                "elapsed_ms": 450, "input_tokens": 800, "output_tokens": 150,
                "layer1_messages": 3, "layer2_episodes": 2,
                "layer3_candidates": 5, "layer3_after_decay": 2,
            },
        }

        with auth_client() as client, \
             patch(
                 "app.api.v1.chat.retrieval_engine.run_hybrid_rag_pipeline",
                 new_callable=AsyncMock,
                 return_value=mock_rag_result,
             ):
            response = client.post(
                "/api/v1/chat",
                headers=auth_header(),
                json={"message": "I've been feeling a bit anxious before my exams"},
            )

        data = response.json()
        assert response.status_code == 200
        assert "response" in data
        assert data["user_id"] == "test-user-uuid"
        assert data["memories_used"] == 2


# ============================================================
# Request Validation Tests
# ============================================================

class TestChatRequestValidation:
    """Tests that Pydantic validation rejects malformed requests."""

    def test_missing_message(self):
        """Phase 2: only message is required in body (user_id comes from JWT)."""
        with auth_client() as client:
            response = client.post(
                "/api/v1/chat",
                headers=auth_header(),
                json={},
            )
        assert response.status_code == 422

    def test_empty_message(self):
        with auth_client() as client:
            response = client.post(
                "/api/v1/chat",
                headers=auth_header(),
                json={"message": ""},
            )
        assert response.status_code == 422

    def test_message_too_long(self):
        with auth_client() as client:
            response = client.post(
                "/api/v1/chat",
                headers=auth_header(),
                json={"message": "x" * 2001},
            )
        assert response.status_code == 422

    def test_no_auth_header_returns_401(self):
        """Phase 2: all chat requests must include a Bearer token."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
        )
        assert response.status_code == 401
