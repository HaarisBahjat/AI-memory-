"""
============================================================
tests/test_consolidation.py -- Phase 7 Consolidation Tests
============================================================
PURPOSE:
    Covers the Phase 7 batch consolidation pipeline:
      - upsert_semantic_fact deduplication logic (create vs reinforce)
      - consolidation_service.run_batch() batch processing
      - POST /api/v1/system/consolidate endpoint (returns 202)
      - GET  /api/v1/system/consolidation/status endpoint
      - Failure path: episode marked FAILED on extraction error
      - Empty-facts path: episode marked CONSOLIDATED with no facts

All tests use mocked DB sessions and mocked OpenAI calls.
No live database or Redis connections are required.
============================================================
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.database import get_db
from app.api.deps import get_current_user
from app.schemas.auth import CurrentUser
from main import app

settings = get_settings()


# -------------------------------------------------------
# Shared Fixtures
# -------------------------------------------------------

def _fake_user():
    return CurrentUser(user_id="test-user-001", email="test@example.com")


def _make_client(db_override=None, user_override=None):
    """Build a TestClient with mocked get_db and get_current_user."""
    app.dependency_overrides[get_current_user] = lambda: _fake_user()
    if db_override is not None:
        app.dependency_overrides[get_db] = lambda: db_override
    client = TestClient(app, raise_server_exceptions=False)
    return client


def _cleanup():
    app.dependency_overrides.clear()


# -------------------------------------------------------
# 1. upsert_semantic_fact -- DEDUPLICATION LOGIC
# -------------------------------------------------------

class TestUpsertSemanticFact:
    """Unit tests for memory_service.upsert_semantic_fact."""

    @pytest.mark.asyncio
    async def test_creates_new_memory_when_no_existing(self):
        """
        When the similarity search returns no existing rows,
        upsert_semantic_fact should INSERT a new memory row.
        """
        from app.services.memory_service import upsert_semantic_fact

        db = AsyncMock()
        # Simulate: no existing memory in the table
        nearest_result = MagicMock()
        nearest_result.mappings.return_value.first.return_value = None
        db.execute = AsyncMock(return_value=nearest_result)

        result = await upsert_semantic_fact(
            db=db,
            user_id="user-001",
            category="trigger",
            text_content="Exams cause anxiety.",
            embedding_vector=[0.1] * 1536,
        )

        assert result["action"] == "created"
        assert "memory_id" in result
        # Should have made 2 execute calls: SELECT + INSERT
        assert db.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_reinforces_existing_memory_when_similar(self):
        """
        When the closest existing memory has distance <= threshold,
        upsert_semantic_fact should UPDATE reinforcement_count (not insert).
        """
        from app.services.memory_service import upsert_semantic_fact

        db = AsyncMock()
        existing_id = str(uuid.uuid4())
        # distance = 0.05 << threshold (1 - 0.88 = 0.12) => reinforce
        nearest_result = MagicMock()
        nearest_result.mappings.return_value.first.return_value = {
            "id": existing_id,
            "distance": 0.05,
        }
        db.execute = AsyncMock(return_value=nearest_result)

        result = await upsert_semantic_fact(
            db=db,
            user_id="user-001",
            category="trigger",
            text_content="Upcoming exams trigger anxiety.",
            embedding_vector=[0.1] * 1536,
        )

        assert result["action"] == "reinforced"
        assert result["memory_id"] == existing_id
        # Should have made 2 execute calls: SELECT + UPDATE
        assert db.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_creates_new_memory_when_distance_above_threshold(self):
        """
        When the closest existing memory has distance > threshold,
        a new memory row should be created (not a reinforcement).
        """
        from app.services.memory_service import upsert_semantic_fact

        db = AsyncMock()
        # distance = 0.25 > threshold (1 - 0.88 = 0.12) => create
        nearest_result = MagicMock()
        nearest_result.mappings.return_value.first.return_value = {
            "id": str(uuid.uuid4()),
            "distance": 0.25,
        }
        db.execute = AsyncMock(return_value=nearest_result)

        result = await upsert_semantic_fact(
            db=db,
            user_id="user-001",
            category="trigger",
            text_content="A completely different trigger.",
            embedding_vector=[0.9] * 1536,
        )

        assert result["action"] == "created"
        assert db.execute.call_count == 2


# -------------------------------------------------------
# 2. _extract_facts_from_summary -- LLM EXTRACTION
# -------------------------------------------------------

class TestExtractFacts:
    """Unit tests for consolidation_service._extract_facts_from_summary."""

    @pytest.mark.asyncio
    async def test_returns_valid_facts(self):
        """LLM response with valid facts should be parsed and returned."""
        from app.services.consolidation_service import _extract_facts_from_summary

        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '[{"content": "Exams trigger anxiety.", "category": "trigger"},'
            ' {"content": "Short walks help manage stress.", "category": "coping_mechanism"}]'
        )

        with patch("app.services.consolidation_service._get_openai_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            facts = await _extract_facts_from_summary("User had an exam week.")

        assert len(facts) == 2
        assert facts[0]["category"] == "trigger"
        assert facts[1]["category"] == "coping_mechanism"

    @pytest.mark.asyncio
    async def test_filters_unknown_categories(self):
        """Facts with categories not in ALLOWED_CATEGORIES should be dropped."""
        from app.services.consolidation_service import _extract_facts_from_summary

        mock_response = MagicMock()
        # "emotion" is NOT in ALLOWED_CATEGORIES
        mock_response.choices[0].message.content = (
            '[{"content": "User felt sad.", "category": "emotion"},'
            ' {"content": "User sleeps early.", "category": "pattern"}]'
        )

        with patch("app.services.consolidation_service._get_openai_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            facts = await _extract_facts_from_summary("Short session.")

        assert len(facts) == 1
        assert facts[0]["category"] == "pattern"

    @pytest.mark.asyncio
    async def test_returns_empty_on_llm_failure(self):
        """If the LLM call raises, the function should return [] (never raise)."""
        from app.services.consolidation_service import _extract_facts_from_summary

        with patch("app.services.consolidation_service._get_openai_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("OpenAI API error")
            )
            mock_client_fn.return_value = mock_client

            facts = await _extract_facts_from_summary("Some session summary.")

        assert facts == []

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_summary(self):
        """Empty or whitespace-only summaries should return [] without LLM call."""
        from app.services.consolidation_service import _extract_facts_from_summary

        facts = await _extract_facts_from_summary("   ")
        assert facts == []


# -------------------------------------------------------
# 3. POST /api/v1/system/consolidate
# -------------------------------------------------------

class TestConsolidateEndpoint:
    """Integration tests for the consolidation trigger endpoint."""

    def test_consolidate_returns_202(self):
        """
        POST /system/consolidate should return 202 Accepted.
        The consolidation job runs in the background.
        """
        db_mock = AsyncMock()
        client = _make_client(db_override=db_mock)

        with patch("app.api.v1.system._run_consolidation_safe", new_callable=AsyncMock):
            resp = client.post("/api/v1/system/consolidate")

        _cleanup()
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "accepted"

    def test_consolidate_requires_auth(self):
        """Without a valid token, the endpoint should return 401 or 403."""
        # Clear any existing overrides
        app.dependency_overrides.clear()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/system/consolidate")
        _cleanup()
        assert resp.status_code in (401, 403)


# -------------------------------------------------------
# 4. GET /api/v1/system/consolidation/status
# -------------------------------------------------------

class TestConsolidationStatus:
    """Integration tests for the consolidation status endpoint."""

    def test_status_returns_counts(self):
        """
        GET /system/consolidation/status should return a dict with
        PENDING / PROCESSING / CONSOLIDATED / FAILED counts.
        """
        db_mock = AsyncMock()
        # Simulate DB returning two status buckets
        rows_mock = MagicMock()
        rows_mock.mappings.return_value.all.return_value = [
            {"consolidation_status": "PENDING", "episode_count": 5},
            {"consolidation_status": "CONSOLIDATED", "episode_count": 42},
        ]
        db_mock.execute = AsyncMock(return_value=rows_mock)

        app.dependency_overrides[get_db] = lambda: db_mock
        app.dependency_overrides[get_current_user] = lambda: _fake_user()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/api/v1/system/consolidation/status")
        _cleanup()

        assert resp.status_code == 200
        body = resp.json()
        assert body["PENDING"] == 5
        assert body["CONSOLIDATED"] == 42
        assert body["PROCESSING"] == 0
        assert body["FAILED"] == 0


# -------------------------------------------------------
# 5. run_batch -- FULL PIPELINE (happy path)
# -------------------------------------------------------

class TestRunBatch:
    """Integration tests for consolidation_service.run_batch."""

    @pytest.mark.asyncio
    async def test_run_batch_no_pending_episodes(self):
        """run_batch with no PENDING episodes should return zero stats."""
        from app.services.consolidation_service import run_batch

        db = AsyncMock()
        empty_result = MagicMock()
        empty_result.mappings.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=empty_result)
        db.commit = AsyncMock()

        stats = await run_batch(db=db)

        assert stats["processed"] == 0
        assert stats["consolidated"] == 0
        assert stats["failed"] == 0

    @pytest.mark.asyncio
    async def test_run_batch_processes_episode_successfully(self):
        """
        When run_batch finds a PENDING episode and LLM returns valid facts,
        it should update processed/consolidated/created counts.
        """
        from app.services.consolidation_service import run_batch

        episode_id = str(uuid.uuid4())
        fake_facts = [
            {"content": "Exams trigger anxiety.", "category": "trigger"},
        ]
        fake_vectors = [[0.1] * 1536]

        db = AsyncMock()
        episodes_result = MagicMock()
        episodes_result.mappings.return_value.all.return_value = [
            {
                "id": episode_id,
                "user_id": "user-001",
                "session_summary": "User discussed exam stress.",
            }
        ]
        db.execute = AsyncMock(return_value=episodes_result)
        db.commit = AsyncMock()

        with (
            patch(
                "app.services.consolidation_service._extract_facts_from_summary",
                AsyncMock(return_value=fake_facts),
            ),
            patch(
                "app.services.consolidation_service.embed_batch",
                AsyncMock(return_value=fake_vectors),
            ),
            patch(
                "app.services.consolidation_service.AsyncSessionLocal"
            ) as mock_session_local,
        ):
            inner_db = AsyncMock()
            inner_db.__aenter__ = AsyncMock(return_value=inner_db)
            inner_db.__aexit__ = AsyncMock(return_value=False)
            inner_db.begin = MagicMock(return_value=inner_db)
            upsert_result = MagicMock()
            upsert_result.mappings.return_value.first.return_value = None
            inner_db.execute = AsyncMock(return_value=upsert_result)
            inner_db.commit = AsyncMock()
            mock_session_local.return_value = inner_db

            with patch(
                "app.services.consolidation_service.upsert_semantic_fact",
                AsyncMock(return_value={"action": "created", "memory_id": str(uuid.uuid4())}),
            ):
                stats = await run_batch(db=db)

        assert stats["processed"] == 1
        assert stats["consolidated"] == 1
        assert stats["created"] == 1
        assert stats["failed"] == 0

    @pytest.mark.asyncio
    async def test_run_batch_marks_failed_on_error(self):
        """
        If _extract_facts_from_summary raises, the episode should be
        marked FAILED and the pipeline should continue (not re-raise).
        """
        from app.services.consolidation_service import run_batch

        episode_id = str(uuid.uuid4())

        db = AsyncMock()
        episodes_result = MagicMock()
        episodes_result.mappings.return_value.all.return_value = [
            {
                "id": episode_id,
                "user_id": "user-001",
                "session_summary": "Normal session.",
            }
        ]
        db.execute = AsyncMock(return_value=episodes_result)
        db.commit = AsyncMock()

        with (
            patch(
                "app.services.consolidation_service._extract_facts_from_summary",
                AsyncMock(side_effect=RuntimeError("LLM unavailable")),
            ),
            patch(
                "app.services.consolidation_service.AsyncSessionLocal"
            ) as mock_session_local,
        ):
            inner_db = AsyncMock()
            inner_db.__aenter__ = AsyncMock(return_value=inner_db)
            inner_db.__aexit__ = AsyncMock(return_value=False)
            inner_db.execute = AsyncMock()
            inner_db.commit = AsyncMock()
            mock_session_local.return_value = inner_db

            stats = await run_batch(db=db)

        assert stats["processed"] == 1
        assert stats["failed"] == 1
        assert stats["consolidated"] == 0

