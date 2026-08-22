"""
============================================================
tests/test_triage.py — Phase 6 Safety Triage Test Suite
============================================================
"""
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.safety_triage import SafetyResult, TriageResponse
from app.services import alert_dispatcher, triage_service


# -------------------------------------------------------
# Helpers / Fixtures
# -------------------------------------------------------

def _make_safe_result() -> SafetyResult:
    return SafetyResult(is_safe=True)


def _make_crisis_result(crisis_type: str = "self_harm") -> SafetyResult:
    triage = TriageResponse(
        is_crisis=True,
        crisis_type=crisis_type,
        message="Please reach out.",
        resources=[{"name": "iCall", "contact": "1234"}],
        follow_up="Call 112.",
        triggered_by=r"\bsuicid\b",
    )
    return SafetyResult(is_safe=False, triage_response=triage)


# -------------------------------------------------------
# 1. Triage Service Unit Tests
# -------------------------------------------------------

class TestTriageService:

    @pytest.mark.asyncio
    async def test_safe_message_returns_safe_result_no_db_write(self):
        """A safe message must not touch the DB or alert dispatcher."""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch(
            "app.services.triage_service.evaluate_clinical_safety",
            return_value=_make_safe_result(),
        ):
            result = await triage_service.evaluate_and_store(
                message="I feel okay today.",
                user_id="user-123",
                session_id="sensory:user-123:session",
                db=mock_db,
                redis=mock_redis,
            )

        assert result.is_safe is True
        mock_db.execute.assert_not_called()
        mock_db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_crisis_message_attempts_db_write(self):
        """A crisis message must attempt to persist a triage_events row."""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_db.execute.return_value.fetchone.return_value = (str(uuid.uuid4()),)

        with (
            patch(
                "app.services.triage_service.evaluate_clinical_safety",
                return_value=_make_crisis_result("self_harm"),
            ),
            patch(
                "app.services.triage_service.alert_dispatcher.dispatch",
                new=AsyncMock(return_value=True),
            ),
        ):
            result = await triage_service.evaluate_and_store(
                message="I want to kill myself",
                user_id="user-456",
                session_id="sensory:user-456:session",
                db=mock_db,
                redis=mock_redis,
            )

        assert result.is_safe is False
        assert result.triage_response.crisis_type == "self_harm"
        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_failure_does_not_raise(self):
        """DB write failure must not propagate — SafetyResult is still returned."""
        mock_db = AsyncMock()
        mock_db.execute.side_effect = Exception("DB connection lost")
        mock_redis = AsyncMock()

        with (
            patch(
                "app.services.triage_service.evaluate_clinical_safety",
                return_value=_make_crisis_result(),
            ),
            patch(
                "app.services.triage_service.alert_dispatcher.dispatch",
                new=AsyncMock(return_value=False),
            ),
        ):
            result = await triage_service.evaluate_and_store(
                message="I want to end it all",
                user_id="user-789",
                session_id="sensory:user-789:session",
                db=mock_db,
                redis=mock_redis,
            )

        assert result.is_safe is False

    @pytest.mark.asyncio
    async def test_alert_dispatch_failure_does_not_raise(self):
        """Alert dispatch failure must not propagate — DB write still proceeds."""
        mock_db = AsyncMock()
        mock_db.execute.return_value.fetchone.return_value = (str(uuid.uuid4()),)
        mock_redis = AsyncMock()

        with (
            patch(
                "app.services.triage_service.evaluate_clinical_safety",
                return_value=_make_crisis_result(),
            ),
            patch(
                "app.services.triage_service.alert_dispatcher.dispatch",
                new=AsyncMock(side_effect=Exception("Slack is down")),
            ),
        ):
            result = await triage_service.evaluate_and_store(
                message="I want to kill myself",
                user_id="user-999",
                session_id="sensory:user-999:session",
                db=mock_db,
                redis=mock_redis,
            )

        assert result.is_safe is False
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_hash_computed_when_enabled(self):
        """When SAFETY_SESSION_HASH_ENABLED=True, session_hash is in the INSERT params."""
        mock_db = AsyncMock()
        mock_db.execute.return_value.fetchone.return_value = (str(uuid.uuid4()),)
        mock_redis = AsyncMock()

        with (
            patch(
                "app.services.triage_service.evaluate_clinical_safety",
                return_value=_make_crisis_result("eating_disorder"),
            ),
            patch(
                "app.services.triage_service.alert_dispatcher.dispatch",
                new=AsyncMock(return_value=False),
            ),
            patch.object(
                triage_service.settings,
                "SAFETY_SESSION_HASH_ENABLED",
                True,
            ),
        ):
            await triage_service.evaluate_and_store(
                message="I have been purging",
                user_id="user-hash",
                session_id="sess-hash",
                db=mock_db,
                redis=mock_redis,
            )

        call_kwargs = mock_db.execute.call_args
        params = call_kwargs[0][1]
        assert params["session_hash"] is not None
        assert len(params["session_hash"]) == 64

    def test_compute_session_hash_deterministic(self):
        """Same inputs must always produce the same hash."""
        h1 = triage_service._compute_session_hash("u1", "s1", "self_harm")
        h2 = triage_service._compute_session_hash("u1", "s1", "self_harm")
        h3 = triage_service._compute_session_hash("u1", "s1", "eating_disorder")
        assert h1 == h2
        assert h1 != h3


# -------------------------------------------------------
# 2. Alert Dispatcher Unit Tests
# -------------------------------------------------------

class TestAlertDispatcher:

    @pytest.mark.asyncio
    async def test_none_channel_returns_false(self):
        """NONE channel must short-circuit without touching Redis."""
        mock_redis = AsyncMock()
        with patch.object(alert_dispatcher.settings, "SAFETY_ALERT_CHANNEL", "NONE"):
            result = await alert_dispatcher.dispatch(
                redis=mock_redis, user_id="u1", session_id="s1", crisis_type="self_harm"
            )
        assert result is False
        mock_redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limited_skips_send(self):
        """If Redis key exists, dispatch must return False without sending."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = None

        with patch.object(alert_dispatcher.settings, "SAFETY_ALERT_CHANNEL", "EMAIL"):
            result = await alert_dispatcher.dispatch(
                redis=mock_redis, user_id="u1", session_id="s1", crisis_type="self_harm"
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_not_rate_limited_sends_email(self):
        """If Redis key is new (set=True), EMAIL channel must send."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True

        with (
            patch.object(alert_dispatcher.settings, "SAFETY_ALERT_CHANNEL", "EMAIL"),
            patch("app.services.alert_dispatcher.asyncio.to_thread", new=AsyncMock()) as mock_thread,
        ):
            result = await alert_dispatcher.dispatch(
                redis=mock_redis, user_id="u1", session_id="s1", crisis_type="self_harm"
            )

        assert result is True
        mock_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_rate_limited_sends_slack(self):
        """If Redis key is new, SLACK channel must call httpx POST."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with (
            patch.object(alert_dispatcher.settings, "SAFETY_ALERT_CHANNEL", "SLACK"),
            patch.object(
                alert_dispatcher.settings, "SAFETY_ALERT_SLACK_WEBHOOK_URL", "https://hooks.slack.com/test"
            ),
            patch("app.services.alert_dispatcher.httpx.AsyncClient") as mock_client,
        ):
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            result = await alert_dispatcher.dispatch(
                redis=mock_redis, user_id="u2", session_id="s2", crisis_type="acute_medical"
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_email_send_failure_returns_false(self):
        """SMTP failure must be caught — returns False, never raises."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True

        with (
            patch.object(alert_dispatcher.settings, "SAFETY_ALERT_CHANNEL", "EMAIL"),
            patch(
                "app.services.alert_dispatcher.asyncio.to_thread",
                new=AsyncMock(side_effect=Exception("SMTP connection refused")),
            ),
        ):
            result = await alert_dispatcher.dispatch(
                redis=mock_redis, user_id="u3", session_id="s3", crisis_type="self_harm"
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_redis_failure_degrades_gracefully(self):
        """Redis failure during rate-limit check must not prevent the alert."""
        mock_redis = AsyncMock()
        mock_redis.set.side_effect = Exception("Redis timeout")

        with (
            patch.object(alert_dispatcher.settings, "SAFETY_ALERT_CHANNEL", "EMAIL"),
            patch("app.services.alert_dispatcher.asyncio.to_thread", new=AsyncMock()) as mock_thread,
        ):
            result = await alert_dispatcher.dispatch(
                redis=mock_redis, user_id="u4", session_id="s4", crisis_type="self_harm"
            )

        assert result is True
        mock_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_channel_returns_false(self):
        """Unknown channel value returns False, no exceptions."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True

        with patch.object(alert_dispatcher.settings, "SAFETY_ALERT_CHANNEL", "TELEGRAM"):
            result = await alert_dispatcher.dispatch(
                redis=mock_redis, user_id="u5", session_id="s5", crisis_type="self_harm"
            )

        assert result is False


# -------------------------------------------------------
# 3. Triage API Integration Tests
# -------------------------------------------------------

class TestTriageAPI:
    """Integration tests against the FastAPI TestClient, with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def setup_overrides(self):
        from main import app
        from app.api.deps import get_current_user
        from app.core.database import get_db
        from app.core.redis_client import get_redis
        from app.schemas.auth import CurrentUser

        # Mock auth
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(
            user_id="admin-123", email="admin@test.com", is_active=True
        )

        # Mock DB
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_result.mappings().all.return_value = []
        mock_result.mappings().first.return_value = None
        mock_result.fetchone.return_value = None
        mock_db.execute.return_value = mock_result
        app.dependency_overrides[get_db] = lambda: mock_db

        # Mock Redis
        app.dependency_overrides[get_redis] = lambda: AsyncMock()

        yield

        app.dependency_overrides.clear()

    def test_list_triage_empty(self):
        """GET /triage with no events returns empty list."""
        from main import app
        client = TestClient(app)
        resp = client.get("/api/v1/triage")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["total"] >= 0
        assert data["page"] == 1

    def test_list_triage_requires_auth(self):
        """GET /triage without token returns 401."""
        from main import app
        from app.api.deps import get_current_user
        app.dependency_overrides.pop(get_current_user, None)
        client = TestClient(app)
        resp = client.get("/api/v1/triage")
        assert resp.status_code == 401

    def test_get_triage_event_not_found(self):
        """GET /triage/{id} with unknown UUID returns 404."""
        from main import app
        client = TestClient(app)
        resp = client.get(f"/api/v1/triage/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_delete_triage_event_not_found(self):
        """DELETE /triage/{id} with unknown UUID returns 404."""
        from main import app
        client = TestClient(app)
        resp = client.delete(f"/api/v1/triage/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_delete_triage_requires_auth(self):
        """DELETE /triage/{id} without token returns 401."""
        from main import app
        from app.api.deps import get_current_user
        app.dependency_overrides.pop(get_current_user, None)
        client = TestClient(app)
        resp = client.delete(f"/api/v1/triage/{uuid.uuid4()}")
        assert resp.status_code == 401

    def test_get_single_triage_requires_auth(self):
        """GET /triage/{id} without token returns 401."""
        from main import app
        from app.api.deps import get_current_user
        app.dependency_overrides.pop(get_current_user, None)
        client = TestClient(app)
        resp = client.get(f"/api/v1/triage/{uuid.uuid4()}")
        assert resp.status_code == 401

    def test_list_triage_pagination_params(self):
        """GET /triage accepts page and page_size query params."""
        from main import app
        client = TestClient(app)
        resp = client.get("/api/v1/triage?page=1&page_size=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page_size"] == 5

    def test_list_triage_invalid_page(self):
        """GET /triage with page=0 returns 422 validation error."""
        from main import app
        client = TestClient(app)
        resp = client.get("/api/v1/triage?page=0")
        assert resp.status_code == 422

    def test_list_triage_invalid_page_size(self):
        """GET /triage with page_size=200 returns 422 (max is 100)."""
        from main import app
        client = TestClient(app)
        resp = client.get("/api/v1/triage?page_size=200")
        assert resp.status_code == 422

    def test_list_triage_active_only_false(self):
        """GET /triage?active_only=False is accepted and returns 200."""
        from main import app
        client = TestClient(app)
        resp = client.get("/api/v1/triage?active_only=false")
        assert resp.status_code == 200

    def test_list_triage_crisis_type_filter(self):
        """GET /triage?crisis_type=self_harm returns 200."""
        from main import app
        client = TestClient(app)
        resp = client.get("/api/v1/triage?crisis_type=self_harm")
        assert resp.status_code == 200


# -------------------------------------------------------
# 4. Chat Integration Tests — triage_service wiring
# -------------------------------------------------------

class TestChatTriageIntegration:

    @pytest.fixture(autouse=True)
    def setup_overrides(self):
        from main import app
        from app.api.deps import get_current_user
        from app.core.database import get_db
        from app.core.redis_client import get_redis
        from app.schemas.auth import CurrentUser

        app.dependency_overrides[get_current_user] = lambda: CurrentUser(
            user_id="user-123", email="user@test.com", is_active=True
        )
        app.dependency_overrides[get_db] = lambda: AsyncMock()
        app.dependency_overrides[get_redis] = lambda: AsyncMock()

        yield

        app.dependency_overrides.clear()

    def test_crisis_chat_calls_triage_service(self):
        """
        A crisis message must go through triage_service.evaluate_and_store,
        not bare evaluate_clinical_safety, and still return CRISIS_TRIAGE.
        """
        from main import app

        client = TestClient(app)
        with patch(
            "app.api.v1.chat.triage_service.evaluate_and_store",
            new=AsyncMock(return_value=_make_crisis_result("self_harm")),
        ) as mock_eval:
            resp = client.post(
                "/api/v1/chat",
                json={"message": "I want to kill myself"},
            )
        # Crisis response returned
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "CRISIS_TRIAGE"
        mock_eval.assert_called_once()

    def test_safe_chat_does_not_early_return(self):
        """A safe message must NOT return a CRISIS_TRIAGE type."""
        from main import app

        client = TestClient(app)
        with patch(
            "app.api.v1.chat.triage_service.evaluate_and_store",
            new=AsyncMock(return_value=_make_safe_result()),
        ):
            with patch(
                "app.services.retrieval_engine.run_hybrid_rag_pipeline",
                new=AsyncMock(
                    return_value={
                        "response": "I'm here to help.",
                        "memories_used": 0,
                        "debug": {},
                    }
                ),
            ):
                with patch(
                    "app.api.v1.chat.session_lifecycle.get_or_create_session_meta",
                    new=AsyncMock(return_value="sensory:user-123:session"),
                ):
                    with patch(
                        "app.api.v1.chat.session_lifecycle.check_session_boundary",
                        new=AsyncMock(return_value=False),
                    ):
                        with patch(
                            "app.api.v1.chat.sensory_service.append_message",
                            new=AsyncMock(),
                        ):
                            resp = client.post(
                                "/api/v1/chat",
                                json={"message": "I feel a bit tired today."},
                            )

        assert resp.status_code == 200
        data = resp.json()
        assert data.get("type") != "CRISIS_TRIAGE"
        assert "response" in data
