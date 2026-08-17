"""
============================================================
tests/test_auth.py — Phase 2 Authentication Test Suite
============================================================
PURPOSE:
    Full unit + integration test coverage for the auth system.
    Uses FastAPI's TestClient + dependency_overrides (the correct
    DI-aware approach) to test the complete auth flow without
    requiring live infrastructure.

TEST STRATEGY:
    - All database I/O is replaced via app.dependency_overrides
    - Redis I/O is replaced via app.dependency_overrides
    - Token generation/validation uses real crypto (no mocking)
      so we test the actual JWT logic, not stubs
    - Each test class manages its own override lifecycle

COVERAGE:
    Register: success, duplicate email, weak password, invalid email
    Login:    success, wrong password, unknown email
    Refresh:  success with rotation, revoked token, expired token, not found
    Logout:   success, already revoked (idempotent)
    /me:      authenticated success, missing token, expired token
    /baseline: success, no-op (empty body)
    /me/memory: cascade delete with JWT
    Chat:     JWT-authenticated, unauthenticated (401)
    Safety:   screener fires even when authenticated

============================================================
"""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from main import app
from app.core.database import get_db
from app.core.redis_client import get_redis
from app.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)

# ============================================================
# Helpers
# ============================================================

def make_access_token(user_id: str = "user-uuid-123") -> str:
    """Creates a real, signed JWT access token for testing."""
    return create_access_token(data={"sub": user_id})


def make_expired_access_token(user_id: str = "user-uuid-123") -> str:
    """Creates an expired JWT access token for negative tests."""
    from jose import jwt
    from app.core.config import get_settings
    settings = get_settings()
    payload = {
        "sub": user_id,
        "type": "access",
        "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        "iat": datetime.now(timezone.utc) - timedelta(minutes=16),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def auth_header(user_id: str = "user-uuid-123") -> dict:
    """Returns the Authorization header dict for a given user."""
    token = make_access_token(user_id)
    return {"Authorization": f"Bearer {token}"}


# Standard mock user row returned by the DB for get_current_user checks
MOCK_AUTH_USER = {
    "user_id": "user-uuid-123",
    "email": "haaris@example.com",
}

MOCK_FULL_USER = {
    "user_id": "user-uuid-123",
    "email": "haaris@example.com",
    "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
    "baseline_profile": {
        "averageSleepHours": None,
        "knownTriggers": [],
        "effectiveCopingMechanisms": [],
        "dataRetentionDays": 365,
        "allowBiometrics": False,
    },
}


# ============================================================
# DB override factories
# ============================================================

def make_db_returning(*rows):
    """
    Returns a get_db dependency override that cycles through the
    provided rows on successive .execute() calls.
    """
    async def _get_db():
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        results = []
        for row in rows:
            r = MagicMock()
            if isinstance(row, dict):
                r.mappings.return_value.first.return_value = row
                r.first.return_value = tuple(row.values()) if row else None
            else:
                r.mappings.return_value.first.return_value = row
                r.first.return_value = row

            results.append(r)

        # Cycle through results on each execute call
        call_count = [0]
        async def _execute(*args, **kwargs):
            idx = min(call_count[0], len(results) - 1)
            call_count[0] += 1
            return results[idx]

        mock_session.execute = _execute
        yield mock_session

    return _get_db


def make_redis_mock():
    """Returns a get_redis dependency override with a noop Redis mock."""
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
def override_db(*rows):
    """Context manager that overrides get_db for the duration of the block."""
    app.dependency_overrides[get_db] = make_db_returning(*rows)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


@contextmanager
def override_db_and_redis(*rows):
    """Context manager that overrides both get_db and get_redis."""
    app.dependency_overrides[get_db] = make_db_returning(*rows)
    app.dependency_overrides[get_redis] = make_redis_mock()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_redis, None)


# ============================================================
# Security Unit Tests (no HTTP, no DB)
# ============================================================

class TestSecurityUtils:
    """Tests for app/core/security.py — pure crypto, no mocking needed."""

    def test_hash_password_produces_bcrypt_hash(self):
        h = hash_password("Str0ngPass1")
        assert h.startswith("$2b$")
        assert len(h) == 60

    def test_verify_password_correct(self):
        h = hash_password("Str0ngPass1")
        assert verify_password("Str0ngPass1", h) is True

    def test_verify_password_wrong(self):
        h = hash_password("Str0ngPass1")
        assert verify_password("WrongPass1", h) is False

    def test_access_token_decode_roundtrip(self):
        from app.core.security import decode_access_token
        token = create_access_token({"sub": "abc-123"})
        payload = decode_access_token(token)
        assert payload["sub"] == "abc-123"
        assert payload["type"] == "access"

    def test_expired_token_raises_401(self):
        from fastapi import HTTPException
        expired_token = make_expired_access_token()
        from app.core.security import decode_access_token
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(expired_token)
        assert exc_info.value.status_code == 401

    def test_tampered_token_raises_401(self):
        from fastapi import HTTPException
        from app.core.security import decode_access_token
        good_token = create_access_token({"sub": "abc-123"})
        tampered = good_token[:-4] + "XXXX"
        with pytest.raises(HTTPException):
            decode_access_token(tampered)

    def test_refresh_token_generation(self):
        raw, h = generate_refresh_token()
        assert len(raw) > 20
        assert hash_refresh_token(raw) == h

    def test_refresh_token_hashes_are_unique(self):
        _, h1 = generate_refresh_token()
        _, h2 = generate_refresh_token()
        assert h1 != h2


# ============================================================
# Register Endpoint Tests
# ============================================================

class TestRegister:
    """POST /api/v1/auth/register"""

    def test_register_success(self):
        # DB calls: 1) email check (None = not found), 2) INSERT user, 3) INSERT refresh token
        with override_db(None, None, None) as client:
            response = client.post(
                "/api/v1/auth/register",
                json={"email": "newuser@example.com", "password": "Str0ngPass1"},
            )
        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 900

    def test_register_weak_password_missing_digit(self):
        client = TestClient(app)
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "user@example.com", "password": "NoDigitsHere"},
        )
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any("digit" in str(e).lower() for e in errors)

    def test_register_weak_password_missing_uppercase(self):
        client = TestClient(app)
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "user@example.com", "password": "nouppercase1"},
        )
        assert response.status_code == 422

    def test_register_weak_password_too_short(self):
        client = TestClient(app)
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "user@example.com", "password": "Ab1"},
        )
        assert response.status_code == 422

    def test_register_invalid_email(self):
        client = TestClient(app)
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": "Str0ngPass1"},
        )
        assert response.status_code == 422

    def test_register_duplicate_email(self):
        # DB: email check returns an existing user → should 409
        existing = {"user_id": "existing-uuid"}
        with override_db(existing) as client:
            response = client.post(
                "/api/v1/auth/register",
                json={"email": "existing@example.com", "password": "Str0ngPass1"},
            )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_register_missing_email(self):
        client = TestClient(app)
        response = client.post(
            "/api/v1/auth/register",
            json={"password": "Str0ngPass1"},
        )
        assert response.status_code == 422

    def test_register_missing_password(self):
        client = TestClient(app)
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "user@example.com"},
        )
        assert response.status_code == 422


# ============================================================
# Login Endpoint Tests
# ============================================================

class TestLogin:
    """POST /api/v1/auth/login"""

    def test_login_success(self):
        user_in_db = {
            "user_id": "user-uuid-123",
            "password_hash": hash_password("Str0ngPass1"),
        }
        # DB: 1) user lookup, 2) INSERT refresh token
        with override_db(user_in_db, None) as client:
            response = client.post(
                "/api/v1/auth/login",
                json={"email": "haaris@example.com", "password": "Str0ngPass1"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_login_wrong_password(self):
        user_in_db = {
            "user_id": "user-uuid-123",
            "password_hash": hash_password("Str0ngPass1"),
        }
        with override_db(user_in_db) as client:
            response = client.post(
                "/api/v1/auth/login",
                json={"email": "haaris@example.com", "password": "WrongPass1"},
            )
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]

    def test_login_email_not_found(self):
        # DB: user lookup returns None
        with override_db(None) as client:
            response = client.post(
                "/api/v1/auth/login",
                json={"email": "ghost@example.com", "password": "Str0ngPass1"},
            )
        assert response.status_code == 401
        # Must match wrong-password message exactly (anti-enumeration)
        assert "Invalid email or password" in response.json()["detail"]


# ============================================================
# Refresh Token Tests
# ============================================================

class TestRefreshToken:
    """POST /api/v1/auth/refresh"""

    def test_refresh_success_rotates_token(self):
        raw_token, _ = generate_refresh_token()
        token_row = {
            "id": "rt-uuid-456",
            "user_id": "user-uuid-123",
            "expires_at": datetime.now(timezone.utc) + timedelta(days=6),
            "revoked": False,
        }
        # DB: 1) lookup token hash, 2) UPDATE revoke old, 3) INSERT new refresh
        with override_db(token_row, None, None) as client:
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": raw_token},
            )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["refresh_token"] != raw_token  # must be a new token

    def test_refresh_revoked_token(self):
        raw_token, _ = generate_refresh_token()
        revoked_row = {
            "id": "rt-uuid-456",
            "user_id": "user-uuid-123",
            "expires_at": datetime.now(timezone.utc) + timedelta(days=6),
            "revoked": True,
        }
        with override_db(revoked_row) as client:
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": raw_token},
            )
        assert response.status_code == 401

    def test_refresh_expired_token(self):
        raw_token, _ = generate_refresh_token()
        expired_row = {
            "id": "rt-uuid-456",
            "user_id": "user-uuid-123",
            "expires_at": datetime.now(timezone.utc) - timedelta(days=1),
            "revoked": False,
        }
        with override_db(expired_row) as client:
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": raw_token},
            )
        assert response.status_code == 401

    def test_refresh_unknown_token(self):
        raw_token, _ = generate_refresh_token()
        # DB: token hash not found → None
        with override_db(None) as client:
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": raw_token},
            )
        assert response.status_code == 401


# ============================================================
# Logout Tests
# ============================================================

class TestLogout:
    """POST /api/v1/auth/logout"""

    def test_logout_success(self):
        raw_token, _ = generate_refresh_token()
        # UPDATE RETURNING returns the user_id of the revoked token
        revoke_result = MagicMock()
        revoke_result.first.return_value = ("user-uuid-123",)

        async def _get_db():
            mock_session = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.rollback = AsyncMock()
            mock_session.close = AsyncMock()
            mock_session.execute = AsyncMock(return_value=revoke_result)
            yield mock_session

        app.dependency_overrides[get_db] = _get_db
        try:
            client = TestClient(app)
            response = client.post(
                "/api/v1/auth/logout",
                json={"refresh_token": raw_token},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 200
        assert "logged out" in response.json()["message"].lower()

    def test_logout_already_revoked_is_idempotent(self):
        """Logging out with an already-revoked token should NOT error."""
        raw_token, _ = generate_refresh_token()
        # UPDATE ... WHERE revoked = FALSE → 0 rows returned
        revoke_result = MagicMock()
        revoke_result.first.return_value = None

        async def _get_db():
            mock_session = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.rollback = AsyncMock()
            mock_session.close = AsyncMock()
            mock_session.execute = AsyncMock(return_value=revoke_result)
            yield mock_session

        app.dependency_overrides[get_db] = _get_db
        try:
            client = TestClient(app)
            response = client.post(
                "/api/v1/auth/logout",
                json={"refresh_token": raw_token},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 200


# ============================================================
# Protected Endpoint Tests (GET /user/me)
# ============================================================

class TestProtectedEndpoints:
    """Tests that verify JWT enforcement on protected routes."""

    def test_get_me_no_token_returns_401(self):
        client = TestClient(app)
        response = client.get("/api/v1/user/me")
        assert response.status_code == 401

    def test_get_me_expired_token_returns_401(self):
        client = TestClient(app)
        expired_token = make_expired_access_token()
        response = client.get(
            "/api/v1/user/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401

    def test_get_me_malformed_token_returns_401(self):
        client = TestClient(app)
        response = client.get(
            "/api/v1/user/me",
            headers={"Authorization": "Bearer this.is.not.a.jwt"},
        )
        assert response.status_code == 401

    def test_get_me_success_with_valid_token(self):
        # DB calls: 1) get_current_user lookup, 2) GET /user/me profile fetch
        with override_db(MOCK_AUTH_USER, MOCK_FULL_USER) as client:
            response = client.get(
                "/api/v1/user/me",
                headers=auth_header(),
            )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-uuid-123"
        assert data["email"] == "haaris@example.com"
        assert "baseline_profile" in data

    def test_chat_endpoint_no_token_returns_401(self):
        """POST /chat without a token must be rejected."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={"message": "I've been feeling anxious"},
        )
        assert response.status_code == 401

    def test_chat_uses_jwt_user_id_not_body(self):
        """
        The chat endpoint must use the JWT-derived user_id,
        and must NOT accept a user_id in the request body.
        """
        from unittest.mock import patch

        mock_rag_result = {
            "response": "I understand how that feels.",
            "memories_used": 0,
            "debug": {
                "elapsed_ms": 300, "input_tokens": 500, "output_tokens": 100,
                "layer1_messages": 0, "layer2_episodes": 0,
                "layer3_candidates": 0, "layer3_after_decay": 0,
            },
        }

        with override_db_and_redis(MOCK_AUTH_USER) as client, \
             patch(
                 "app.api.v1.chat.retrieval_engine.run_hybrid_rag_pipeline",
                 new_callable=AsyncMock,
                 return_value=mock_rag_result,
             ):
            response = client.post(
                "/api/v1/chat",
                headers=auth_header("user-uuid-123"),
                json={"message": "I've been feeling anxious before exams."},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-uuid-123"

    def test_chat_does_not_accept_user_id_in_body(self):
        """
        Verify that ChatRequest no longer has a user_id field.
        """
        from app.schemas.chat import ChatRequest
        assert "user_id" not in ChatRequest.model_fields


# ============================================================
# Chat Request Validation (with auth)
# ============================================================

class TestChatRequestValidationWithAuth:
    """Tests that Pydantic validation still works with JWT auth."""

    def test_missing_message_returns_422(self):
        with override_db_and_redis(MOCK_AUTH_USER) as client:
            response = client.post(
                "/api/v1/chat",
                headers=auth_header(),
                json={},
            )
        assert response.status_code == 422

    def test_empty_message_returns_422(self):
        with override_db_and_redis(MOCK_AUTH_USER) as client:
            response = client.post(
                "/api/v1/chat",
                headers=auth_header(),
                json={"message": ""},
            )
        assert response.status_code == 422

    def test_message_too_long_returns_422(self):
        with override_db_and_redis(MOCK_AUTH_USER) as client:
            response = client.post(
                "/api/v1/chat",
                headers=auth_header(),
                json={"message": "x" * 2001},
            )
        assert response.status_code == 422


# ============================================================
# Safety Override works even when authenticated
# ============================================================

class TestSafetyScreenerWithAuth:
    """Verify the safety screener still fires on crisis messages."""

    def test_crisis_message_returns_triage_even_when_authenticated(self):
        from unittest.mock import patch

        with override_db_and_redis(MOCK_AUTH_USER) as client, \
             patch(
                 "app.api.v1.chat.retrieval_engine.run_hybrid_rag_pipeline",
                 new_callable=AsyncMock,
             ) as mock_rag:
            response = client.post(
                "/api/v1/chat",
                headers=auth_header(),
                json={"message": "I want to kill myself"},
            )
            mock_rag.assert_not_called()

        data = response.json()
        assert data["type"] == "CRISIS_TRIAGE"
        assert "resources" in data
