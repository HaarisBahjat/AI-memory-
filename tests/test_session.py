"""
============================================================
tests/test_session.py — Phase 3 Session Lifecycle Test Suite
============================================================
PURPOSE:
    Full unit + integration coverage for Phase 3:
    - Mood scoring math (pure Python, no mocks needed)
    - Mood drop detection
    - Session lifecycle service (mocked Redis)
    - Session API endpoints (FastAPI TestClient + DI overrides)
    - Session boundary detection
    - Session end + synthesis stub dispatch

TEST STRATEGY:
    - Pure logic tests (TestMoodTracker) need no mocks
    - Service-level tests (TestSessionLifecycle) mock Redis via AsyncMock
    - HTTP tests (TestSessionEndpoints) use FastAPI dependency_overrides
      for both get_db and get_redis — same pattern as test_auth.py

COVERAGE GROUPS:
    TestMoodTracker          — valence scoring, amplifiers, mood drop detection
    TestSessionLifecycle     — boundary check, meta create/update, close
    TestSessionEndpoints     — GET /active, POST /end with JWT auth
    TestBoundaryDetection    — new session flag in chat response
============================================================
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.services.mood_tracker import (
    MOOD_DROP_THRESHOLD,
    detect_mood_drop,
    score_message,
)

# ============================================================
# TestMoodTracker — Pure Python, No Mocks
# ============================================================

class TestMoodTracker:
    """Tests for app/services/mood_tracker.py — zero dependencies."""

    def test_clearly_positive_message(self):
        """'Feeling great and hopeful' should score positive."""
        score = score_message("I'm feeling great today and really hopeful about things")
        assert score > 0.0

    def test_clearly_negative_message(self):
        """'Anxious and stressed' should score negative."""
        score = score_message("I've been feeling very anxious and stressed all week")
        assert score < 0.0

    def test_neutral_message_scores_near_zero(self):
        """A message with no sentiment words should be close to zero."""
        score = score_message("My name is Haaris and I study at KAUST")
        assert -0.2 <= score <= 0.2

    def test_crisis_amplifier_increases_negative_weight(self):
        """
        Crisis amplifiers boost the weight of adjacent negative words.
        Test: 'can't anxious' should score lower than 'feel anxious'
        because 'can't' is an amplifier but 'feel' is not.
        Both phrases have the same word count so normalization is equal.
        """
        plain = score_message("feel anxious")       # -1 / 2 words = -0.5
        amplified = score_message("can't anxious")  # -1.5 / 2 words = -0.75
        assert amplified < plain, f"Expected {amplified} < {plain}"

    def test_empty_string_returns_zero(self):
        assert score_message("") == 0.0

    def test_whitespace_only_returns_zero(self):
        assert score_message("   ") == 0.0

    def test_score_clamps_to_range(self):
        """Extreme input should never exceed [-1.0, +1.0]."""
        very_positive = " ".join(["better", "good", "calm", "happy"] * 50)
        very_negative = " ".join(["anxious", "hopeless", "exhausted", "scared"] * 50)
        assert -1.0 <= score_message(very_positive) <= 1.0
        assert -1.0 <= score_message(very_negative) <= 1.0

    def test_case_insensitive(self):
        """Scoring should be case-insensitive."""
        lower = score_message("feeling anxious and stressed")
        upper = score_message("FEELING ANXIOUS AND STRESSED")
        assert lower == upper

    def test_mood_drop_not_detected_with_one_sample(self):
        """Mood drop requires at least 2 data points."""
        result = detect_mood_drop(mood_sum=-0.8, mood_count=1, first_score=0.5)
        assert result is False

    def test_mood_drop_detected_when_threshold_exceeded(self):
        """
        First score = +0.5 (positive message)
        Current average = -0.1 (session has gotten worse)
        Delta = 0.5 - (-0.1) = 0.6 > MOOD_DROP_THRESHOLD (0.4)
        → Should flag mood drop
        """
        # mood_sum / mood_count = -0.2 / 2 = -0.1 average
        result = detect_mood_drop(
            mood_sum=-0.2,
            mood_count=2,
            first_score=0.5,
        )
        assert result is True

    def test_mood_drop_not_detected_when_below_threshold(self):
        """
        First score = +0.3
        Current average = +0.1
        Delta = 0.3 - 0.1 = 0.2 < MOOD_DROP_THRESHOLD (0.4)
        → Should NOT flag mood drop
        """
        result = detect_mood_drop(
            mood_sum=0.2,
            mood_count=2,
            first_score=0.3,
        )
        assert result is False

    def test_mood_drop_none_first_score(self):
        """If first_score is None, mood drop cannot be calculated."""
        result = detect_mood_drop(mood_sum=-0.8, mood_count=3, first_score=None)
        assert result is False

    def test_mood_drop_threshold_value(self):
        """MOOD_DROP_THRESHOLD should be 0.4."""
        assert MOOD_DROP_THRESHOLD == 0.4


# ============================================================
# TestSessionLifecycle — Mocked Redis
# ============================================================

class TestSessionLifecycle:
    """Tests for app/services/session_lifecycle.py with mocked Redis."""

    def _make_redis(self, meta_data: dict = None, ttl: int = 1500):
        """
        Returns a mock Redis client.
        meta_data: dict returned by hgetall (None = empty HASH = no session)
        """
        mock = AsyncMock()
        mock.exists = AsyncMock(return_value=1 if meta_data else 0)
        mock.hgetall = AsyncMock(return_value=meta_data or {})
        mock.hget = AsyncMock(return_value=None)
        mock.ttl = AsyncMock(return_value=ttl)
        mock.delete = AsyncMock(return_value=1)
        pipeline = AsyncMock()
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=False)
        # Pipeline command methods are called synchronously (not awaited)
        # inside `async with pipe`, so they must be regular MagicMock
        pipeline.hmset = MagicMock(return_value=None)
        pipeline.hset = MagicMock(return_value=None)
        pipeline.expire = MagicMock(return_value=None)
        pipeline.execute = AsyncMock(return_value=[1, True, True])
        mock.pipeline = MagicMock(return_value=pipeline)
        return mock

    @pytest.mark.asyncio
    async def test_check_session_boundary_no_meta_is_new(self):
        """Missing meta key → new session boundary."""
        from app.services.session_lifecycle import check_session_boundary
        redis = self._make_redis(meta_data=None)
        result = await check_session_boundary(redis, "user-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_session_boundary_meta_exists_is_not_new(self):
        """Existing meta key → active session, not a boundary."""
        from app.services.session_lifecycle import check_session_boundary
        redis = self._make_redis(meta_data={"start_time": "1000", "message_count": "4"})
        result = await check_session_boundary(redis, "user-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_or_create_creates_meta_when_missing(self):
        """When no meta exists, a new SessionMetadata should be initialized."""
        from app.services.session_lifecycle import get_or_create_session_meta
        redis = self._make_redis(meta_data=None)
        meta = await get_or_create_session_meta(redis, "user-123")
        assert meta.message_count == 0
        assert meta.mood_sum == 0.0
        assert meta.mood_drop_flag is False
        assert meta.start_time > 0

    @pytest.mark.asyncio
    async def test_get_or_create_reads_existing_meta(self):
        """When meta exists, it should parse the HASH into SessionMetadata."""
        from app.services.session_lifecycle import get_or_create_session_meta
        existing = {
            "start_time": "1700000000",
            "last_active": "1700001000",
            "message_count": "6",
            "mood_sum": "-0.3",
            "mood_count": "3",
            "mood_drop_flag": "false",
        }
        redis = self._make_redis(meta_data=existing)
        meta = await get_or_create_session_meta(redis, "user-123")
        assert meta.message_count == 6
        assert meta.mood_count == 3
        assert meta.mood_drop_flag is False

    @pytest.mark.asyncio
    async def test_get_session_state_no_session(self):
        """No active session → is_new_session=True, nulls for all state."""
        from app.services.session_lifecycle import get_session_state
        redis = AsyncMock()
        redis.hgetall = AsyncMock(return_value={})
        redis.ttl = AsyncMock(return_value=-2)  # key does not exist
        state = await get_session_state(redis, "user-123")
        assert state["is_new_session"] is True
        assert state["message_count"] == 0
        assert state["mood_delta"] is None
        assert state["ttl_seconds"] is None

    @pytest.mark.asyncio
    async def test_get_session_state_active_session(self):
        """Active session → is_new_session=False, populated counts."""
        from app.services.session_lifecycle import get_session_state
        redis = AsyncMock()
        redis.hgetall = AsyncMock(return_value={
            "start_time": "1700000000",
            "last_active": "1700001000",
            "message_count": "4",
            "mood_sum": "0.6",
            "mood_count": "2",
            "mood_drop_flag": "false",
        })
        redis.ttl = AsyncMock(return_value=1200)
        state = await get_session_state(redis, "user-123")
        assert state["is_new_session"] is False
        assert state["message_count"] == 4
        assert state["ttl_seconds"] == 1200
        assert state["mood_delta"] == 0.3   # 0.6 / 2

    @pytest.mark.asyncio
    async def test_session_meta_mood_delta_property(self):
        """mood_delta property returns average when count > 0."""
        from app.schemas.session import SessionMetadata
        meta = SessionMetadata(
            start_time=1000,
            last_active=2000,
            message_count=4,
            mood_sum=-0.6,
            mood_count=3,
        )
        assert meta.mood_delta == pytest.approx(-0.2, abs=0.01)

    @pytest.mark.asyncio
    async def test_session_meta_mood_delta_none_when_no_data(self):
        """mood_delta returns None when mood_count=0."""
        from app.schemas.session import SessionMetadata
        meta = SessionMetadata(start_time=1000, last_active=2000)
        assert meta.mood_delta is None


# ============================================================
# Helpers for HTTP tests
# ============================================================

MOCK_AUTH_USER = {"user_id": "session-test-uuid", "email": "haaris@test.com"}


def make_db_with_user(user_row=None):
    """Returns a get_db override returning user_row on first execute."""
    async def _get_db():
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        result = MagicMock()
        result.mappings.return_value.first.return_value = user_row or MOCK_AUTH_USER
        mock_session.execute = AsyncMock(return_value=result)
        yield mock_session
    return _get_db


def make_redis_mock(
    meta_data: dict = None,
    ttl: int = 1500,
):
    """Returns a get_redis override with a configurable mock Redis client."""
    async def _get_redis():
        mock = AsyncMock()
        mock.exists = AsyncMock(return_value=1 if meta_data else 0)
        mock.hgetall = AsyncMock(return_value=meta_data or {})
        mock.hget = AsyncMock(return_value=None)
        mock.ttl = AsyncMock(return_value=ttl)
        mock.lrange = AsyncMock(return_value=[])
        mock.rpush = AsyncMock(return_value=1)
        mock.ltrim = AsyncMock(return_value=True)
        mock.expire = AsyncMock(return_value=True)
        mock.delete = AsyncMock(return_value=1)
        pipeline = AsyncMock()
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=False)
        # Pipeline command methods called synchronously inside `async with pipe`
        pipeline.hmset = MagicMock(return_value=None)
        pipeline.hset = MagicMock(return_value=None)
        pipeline.rpush = MagicMock(return_value=None)
        pipeline.ltrim = MagicMock(return_value=None)
        pipeline.expire = MagicMock(return_value=None)
        pipeline.execute = AsyncMock(return_value=[1, True, True])
        mock.pipeline = MagicMock(return_value=pipeline)
        yield mock
    return _get_redis


@contextmanager
def session_client(meta_data=None, ttl=1500):
    """Context manager returning a TestClient with mocked DB + Redis."""
    from main import app
    from app.core.database import get_db
    from app.core.redis_client import get_redis
    from app.core.security import create_access_token

    app.dependency_overrides[get_db] = make_db_with_user(MOCK_AUTH_USER)
    app.dependency_overrides[get_redis] = make_redis_mock(meta_data=meta_data, ttl=ttl)
    try:
        client = TestClient(app)
        token = create_access_token({"sub": "session-test-uuid"})
        yield client, {"Authorization": f"Bearer {token}"}
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_redis, None)


# ============================================================
# TestSessionEndpoints — HTTP Tests
# ============================================================

class TestSessionEndpoints:
    """HTTP-level tests for GET /session/active and POST /session/end."""

    def test_get_active_session_no_session(self):
        """GET /session/active → is_new_session=True when no meta exists."""
        with session_client(meta_data=None) as (client, headers):
            response = client.get("/api/v1/session/active", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["is_new_session"] is True
        assert data["message_count"] == 0
        assert data["mood_delta"] is None
        assert data["ttl_seconds"] is None

    def test_get_active_session_with_active_session(self):
        """GET /session/active → is_new_session=False when session exists."""
        meta = {
            "start_time": "1700000000",
            "last_active": "1700001200",
            "message_count": "4",
            "mood_sum": "0.4",
            "mood_count": "2",
            "mood_drop_flag": "false",
        }
        with session_client(meta_data=meta, ttl=1200) as (client, headers):
            response = client.get("/api/v1/session/active", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["is_new_session"] is False
        assert data["message_count"] == 4
        assert data["ttl_seconds"] == 1200
        assert data["mood_delta"] == pytest.approx(0.2, abs=0.01)

    def test_get_active_session_requires_auth(self):
        """GET /session/active without token → 401."""
        from main import app
        client = TestClient(app)
        response = client.get("/api/v1/session/active")
        assert response.status_code == 401

    def test_post_session_end_no_active_session(self):
        """POST /session/end when no session → 200 with message_count=0."""
        with session_client(meta_data=None) as (client, headers):
            response = client.post(
                "/api/v1/session/end",
                headers=headers,
                json={"reason": "explicit"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["message_count"] == 0
        assert data["synthesis_triggered"] is False

    def test_post_session_end_with_active_session(self):
        """POST /session/end with an active session → synthesis triggered."""
        import json as _json
        meta = {
            "start_time": "1700000000",
            "last_active": "1700001800",
            "message_count": "6",
            "mood_sum": "-0.4",
            "mood_count": "3",
            "mood_drop_flag": "true",
        }

        # Return mock messages from lrange for flush_session
        from app.schemas.chat import SessionMessage
        import time
        messages = [
            _json.dumps({"role": "user", "content": "Hello", "timestamp": int(time.time())}),
            _json.dumps({"role": "assistant", "content": "Hi there!", "timestamp": int(time.time())}),
        ]

        async def _get_redis():
            mock = AsyncMock()
            mock.exists = AsyncMock(return_value=1)
            mock.hgetall = AsyncMock(return_value=meta)
            mock.hget = AsyncMock(return_value=None)
            mock.ttl = AsyncMock(return_value=600)
            mock.lrange = AsyncMock(return_value=messages)
            mock.delete = AsyncMock(return_value=2)
            pipeline = AsyncMock()
            pipeline.__aenter__ = AsyncMock(return_value=pipeline)
            pipeline.__aexit__ = AsyncMock(return_value=False)
            pipeline.execute = AsyncMock(return_value=[1, True, True])
            mock.pipeline = MagicMock(return_value=pipeline)
            yield mock

        from main import app
        from app.core.database import get_db
        from app.core.redis_client import get_redis
        from app.core.security import create_access_token

        app.dependency_overrides[get_db] = make_db_with_user(MOCK_AUTH_USER)
        app.dependency_overrides[get_redis] = _get_redis
        try:
            client = TestClient(app)
            token = create_access_token({"sub": "session-test-uuid"})
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post(
                "/api/v1/session/end",
                headers=headers,
                json={"reason": "explicit"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)
            app.dependency_overrides.pop(get_redis, None)

        assert response.status_code == 200
        data = response.json()
        # message_count comes from the meta HASH (what was stored)
        assert data["message_count"] == 6
        assert data["synthesis_triggered"] is True
        assert data["mood_drop_alert"] is True
        assert data["reason"] == "explicit"

    def test_post_session_end_requires_auth(self):
        """POST /session/end without token → 401."""
        from main import app
        client = TestClient(app)
        response = client.post("/api/v1/session/end", json={"reason": "explicit"})
        assert response.status_code == 401

    def test_post_session_end_default_reason(self):
        """POST /session/end with no body should default reason='explicit'."""
        with session_client(meta_data=None) as (client, headers):
            response = client.post("/api/v1/session/end", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["reason"] == "explicit"

    def test_post_session_end_invalid_reason_422(self):
        """POST /session/end with invalid reason → 422 Pydantic validation error."""
        with session_client(meta_data=None) as (client, headers):
            response = client.post(
                "/api/v1/session/end",
                headers=headers,
                json={"reason": "invalid_reason_value"},
            )
        assert response.status_code == 422


# ============================================================
# TestBoundaryDetection — is_new_session in Chat Response
# ============================================================

class TestBoundaryDetection:
    """
    Verifies that the chat endpoint correctly sets is_new_session
    based on the session lifecycle boundary check.
    """

    def test_chat_response_has_is_new_session_field(self):
        """ChatResponse schema must include is_new_session field."""
        from app.schemas.chat import ChatResponse
        assert "is_new_session" in ChatResponse.model_fields

    def test_chat_response_is_new_session_defaults_false(self):
        """is_new_session defaults to False."""
        from app.schemas.chat import ChatResponse
        r = ChatResponse(
            user_id="u1",
            response="hello",
            session_id="sensory:u1:session",
            memories_used=0,
        )
        assert r.is_new_session is False

    def test_chat_response_is_new_session_can_be_true(self):
        """is_new_session can be set to True explicitly."""
        from app.schemas.chat import ChatResponse
        r = ChatResponse(
            user_id="u1",
            response="hello",
            session_id="sensory:u1:session",
            memories_used=0,
            is_new_session=True,
        )
        assert r.is_new_session is True

    def test_chat_returns_is_new_session_false_when_session_active(self):
        """
        When meta HASH exists (session active), chat should return
        is_new_session=False in the response.
        """
        meta = {
            "start_time": "1700000000",
            "last_active": "1700001000",
            "message_count": "2",
            "mood_sum": "0.1",
            "mood_count": "1",
            "mood_drop_flag": "false",
        }
        mock_rag_result = {
            "response": "I understand your concern.",
            "memories_used": 0,
            "debug": {"elapsed_ms": 200, "input_tokens": 100, "output_tokens": 50,
                      "layer1_messages": 0, "layer2_episodes": 0,
                      "layer3_candidates": 0, "layer3_after_decay": 0},
        }
        with session_client(meta_data=meta) as (client, headers), \
             patch(
                 "app.api.v1.chat.retrieval_engine.run_hybrid_rag_pipeline",
                 new_callable=AsyncMock,
                 return_value=mock_rag_result,
             ):
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "I have been feeling anxious lately"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["is_new_session"] is False

    def test_chat_returns_is_new_session_true_when_no_meta(self):
        """
        When meta HASH is absent (new/expired session), chat should return
        is_new_session=True in the response.
        """
        mock_rag_result = {
            "response": "Welcome! Let's talk about how you're feeling.",
            "memories_used": 0,
            "debug": {"elapsed_ms": 210, "input_tokens": 80, "output_tokens": 60,
                      "layer1_messages": 0, "layer2_episodes": 0,
                      "layer3_candidates": 0, "layer3_after_decay": 0},
        }
        with session_client(meta_data=None) as (client, headers), \
             patch(
                 "app.api.v1.chat.retrieval_engine.run_hybrid_rag_pipeline",
                 new_callable=AsyncMock,
                 return_value=mock_rag_result,
             ):
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "Hello, I want to talk"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["is_new_session"] is True
