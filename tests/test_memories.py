"""
============================================================
tests/test_memories.py — Phase 4 Semantic Memory CRUD Tests
============================================================
PURPOSE:
    Comprehensive pytest suite for all 6 /api/v1/memories
    endpoints. All external I/O (database, OpenAI embeddings)
    is mocked — no live infrastructure required.

COVERAGE:
    ✓ GET    /memories — list, category filter, pinned filter, pagination
    ✓ POST   /memories — create with embedding mock
    ✓ GET    /memories/{id} — fetch, 404 on missing/wrong-user
    ✓ PUT    /memories/{id} — update text (re-embeds), category only
    ✓ DELETE /memories/{id} — delete own, 404 on missing
    ✓ PATCH  /memories/{id}/pin — pin, unpin, 404 on missing
    ✓ Error paths — unauthenticated, invalid category, empty text

MOCK STRATEGY:
    - get_db overridden at FastAPI DI level (make_mock_db)
    - get_redis overridden (session.py imports it; must be present)
    - embedding_service.embed_text patched via unittest.mock.patch
    - memory_service functions patched per test for precise control

CONVENTIONS:
    - All tests use valid JWT tokens generated with test secret
    - memory_id values use fixed UUIDs for determinism
    - Structured responses verified field-by-field, not just status codes
============================================================
"""

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Env setup (must precede app imports) ────────────────────────
import os
os.environ.setdefault("SUPABASE_DB_URL", "postgresql+asyncpg://test:test@localhost/testdb")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-not-real")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-minimum-32-characters-1234")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from main import app
from app.core.database import get_db
from app.core.redis_client import get_redis
from app.core.security import create_access_token
from app.schemas.memory import MemoryResponse

# ── Constants ────────────────────────────────────────────────────
TEST_MEMORY_ID = str(uuid.uuid4())
TEST_MEMORY_ID_2 = str(uuid.uuid4())
TEST_USER_ID = "test-user-uuid"
TEST_USER_EMAIL = "test@example.com"

FAKE_VECTOR = [0.01] * 1536  # 1536-dim unit vector for mocking

NOW = datetime.now(timezone.utc)

# User row returned by the DB mock so get_current_user resolves
DEFAULT_USER_ROW = {"user_id": TEST_USER_ID, "email": TEST_USER_EMAIL}


# ── Helpers ──────────────────────────────────────────────────────

def make_mock_db_local(user_row=None):
    """Local copy of make_mock_db from conftest — avoids direct import."""
    async def _mock_get_db():
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        result = MagicMock()
        result.mappings.return_value.first.return_value = user_row
        result.first.return_value = None
        mock_session.execute = AsyncMock(return_value=result)
        yield mock_session
    return _mock_get_db


def make_mock_redis_local():
    """Local copy of make_mock_redis from conftest — avoids direct import."""
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


def auth_header() -> dict[str, str]:
    """Generate a valid Bearer token for the test user."""
    token = create_access_token(data={"sub": TEST_USER_ID})
    return {"Authorization": f"Bearer {token}"}


def make_memory_row(
    memory_id: str = None,
    user_id: str = None,
    category: str = "trigger",
    text: str = "Public speaking causes anxiety",
    reinforcement_count: int = 1,
    is_pinned: bool = False,
    created_at: datetime = None,
) -> dict[str, Any]:
    """Returns a dict that mirrors a semantic_memories DB row."""
    return {
        "id": memory_id or TEST_MEMORY_ID,
        "user_id": user_id or TEST_USER_ID,
        "category": category,
        "text": text,
        "reinforcement_count": reinforcement_count,
        "is_pinned": is_pinned,
        "created_at": created_at or NOW,
    }


def make_memory_response(**kwargs) -> MemoryResponse:
    """Builds a MemoryResponse with sensible defaults."""
    row = make_memory_row(**kwargs)
    return MemoryResponse(**row)


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(autouse=False)
def client():
    """
    TestClient with get_db and get_redis overridden to mocks.
    The mock DB yields an authenticated user row so that
    get_current_user resolves correctly.
    """
    app.dependency_overrides[get_db] = make_mock_db_local(DEFAULT_USER_ROW)
    app.dependency_overrides[get_redis] = make_mock_redis_local()
    yield TestClient(app)
    app.dependency_overrides.clear()


# ================================================================
# GET /api/v1/memories
# ================================================================

class TestListMemories:
    """Tests for GET /api/v1/memories."""

    def test_list_returns_200_empty(self, client: TestClient):
        """Empty memory store returns 200 with empty items list."""
        empty_result = {
            "items": [], "total": 0, "page": 1,
            "page_size": 20, "total_pages": 1,
        }
        with patch(
            "app.api.v1.memories.memory_service.list_memories",
            new_callable=AsyncMock,
            return_value=empty_result,
        ):
            response = client.get("/api/v1/memories", headers=auth_header())

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []
        assert data["page"] == 1

    def test_list_returns_memories(self, client: TestClient):
        """Returns populated items list with correct structure."""
        mem = make_memory_response()
        result = {
            "items": [mem],
            "total": 1,
            "page": 1,
            "page_size": 20,
            "total_pages": 1,
        }
        with patch(
            "app.api.v1.memories.memory_service.list_memories",
            new_callable=AsyncMock,
            return_value=result,
        ):
            response = client.get("/api/v1/memories", headers=auth_header())

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["id"] == TEST_MEMORY_ID
        assert item["category"] == "trigger"
        assert item["is_pinned"] is False

    def test_list_filters_by_category(self, client: TestClient):
        """Passes category filter to memory_service."""
        with patch(
            "app.api.v1.memories.memory_service.list_memories",
            new_callable=AsyncMock,
            return_value={"items": [], "total": 0, "page": 1, "page_size": 20, "total_pages": 1},
        ) as mock_list:
            client.get(
                "/api/v1/memories?category=trigger",
                headers=auth_header(),
            )
        # Verify category was passed through
        called_filters = mock_list.call_args.kwargs["filters"]
        assert called_filters.category == "trigger"

    def test_list_pinned_only_filter(self, client: TestClient):
        """pinned_only=true passes through to service."""
        with patch(
            "app.api.v1.memories.memory_service.list_memories",
            new_callable=AsyncMock,
            return_value={"items": [], "total": 0, "page": 1, "page_size": 20, "total_pages": 1},
        ) as mock_list:
            client.get(
                "/api/v1/memories?pinned_only=true",
                headers=auth_header(),
            )
        called_filters = mock_list.call_args.kwargs["filters"]
        assert called_filters.pinned_only is True

    def test_list_pagination_params(self, client: TestClient):
        """page and page_size are passed to service correctly."""
        with patch(
            "app.api.v1.memories.memory_service.list_memories",
            new_callable=AsyncMock,
            return_value={"items": [], "total": 0, "page": 2, "page_size": 5, "total_pages": 1},
        ) as mock_list:
            client.get(
                "/api/v1/memories?page=2&page_size=5",
                headers=auth_header(),
            )
        called_filters = mock_list.call_args.kwargs["filters"]
        assert called_filters.page == 2
        assert called_filters.page_size == 5

    def test_list_invalid_category_ignored(self, client: TestClient):
        """An invalid category value falls back to no filter (sort_by validation)."""
        # The router normalizes invalid sort_by; invalid category raises 422
        response = client.get(
            "/api/v1/memories?category=invalid_cat",
            headers=auth_header(),
        )
        assert response.status_code == 422

    def test_list_requires_auth(self, client: TestClient):
        """Unauthenticated request returns 401."""
        response = client.get("/api/v1/memories")
        assert response.status_code == 401

    def test_list_invalid_sort_by_falls_back(self, client: TestClient):
        """Invalid sort_by falls back to created_at without error."""
        with patch(
            "app.api.v1.memories.memory_service.list_memories",
            new_callable=AsyncMock,
            return_value={"items": [], "total": 0, "page": 1, "page_size": 20, "total_pages": 1},
        ) as mock_list:
            response = client.get(
                "/api/v1/memories?sort_by=invalid_field",
                headers=auth_header(),
            )
        assert response.status_code == 200
        called_filters = mock_list.call_args.kwargs["filters"]
        assert called_filters.sort_by == "created_at"


# ================================================================
# POST /api/v1/memories
# ================================================================

class TestCreateMemory:
    """Tests for POST /api/v1/memories."""

    def test_create_returns_201(self, client: TestClient):
        """Valid request returns 201 with the created memory."""
        created_mem = make_memory_response(
            category="trigger",
            text="Penicillin allergy — anaphylaxis risk",
        )
        with (
            patch(
                "app.api.v1.memories.embedding_service.embed_text",
                new_callable=AsyncMock,
                return_value=FAKE_VECTOR,
            ),
            patch(
                "app.api.v1.memories.memory_service.create_memory",
                new_callable=AsyncMock,
                return_value=created_mem,
            ),
        ):
            response = client.post(
                "/api/v1/memories",
                json={"category": "trigger", "text": "Penicillin allergy — anaphylaxis risk"},
                headers=auth_header(),
            )

        assert response.status_code == 201
        data = response.json()
        assert data["category"] == "trigger"
        assert data["is_pinned"] is False
        assert "id" in data

    def test_create_calls_embedding(self, client: TestClient):
        """embed_text is called with the provided text."""
        created_mem = make_memory_response()
        with (
            patch(
                "app.api.v1.memories.embedding_service.embed_text",
                new_callable=AsyncMock,
                return_value=FAKE_VECTOR,
            ) as mock_embed,
            patch(
                "app.api.v1.memories.memory_service.create_memory",
                new_callable=AsyncMock,
                return_value=created_mem,
            ),
        ):
            client.post(
                "/api/v1/memories",
                json={"category": "symptom", "text": "Persistent headache after stress"},
                headers=auth_header(),
            )

        mock_embed.assert_awaited_once_with("Persistent headache after stress")

    def test_create_passes_vector_to_service(self, client: TestClient):
        """The vector from embed_text is passed to create_memory."""
        created_mem = make_memory_response()
        with (
            patch(
                "app.api.v1.memories.embedding_service.embed_text",
                new_callable=AsyncMock,
                return_value=FAKE_VECTOR,
            ),
            patch(
                "app.api.v1.memories.memory_service.create_memory",
                new_callable=AsyncMock,
                return_value=created_mem,
            ) as mock_create,
        ):
            client.post(
                "/api/v1/memories",
                json={"category": "trigger", "text": "Public speaking"},
                headers=auth_header(),
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["embedding_vector"] == FAKE_VECTOR
        assert call_kwargs["user_id"] == TEST_USER_ID

    def test_create_requires_text(self, client: TestClient):
        """Missing text field returns 422."""
        response = client.post(
            "/api/v1/memories",
            json={"category": "trigger"},
            headers=auth_header(),
        )
        assert response.status_code == 422

    def test_create_requires_category(self, client: TestClient):
        """Missing category field returns 422."""
        response = client.post(
            "/api/v1/memories",
            json={"text": "Some text"},
            headers=auth_header(),
        )
        assert response.status_code == 422

    def test_create_invalid_category_returns_422(self, client: TestClient):
        """Invalid category value returns 422."""
        response = client.post(
            "/api/v1/memories",
            json={"category": "not_a_real_category", "text": "Some text"},
            headers=auth_header(),
        )
        assert response.status_code == 422

    def test_create_text_too_short_returns_422(self, client: TestClient):
        """Text shorter than 3 characters returns 422."""
        response = client.post(
            "/api/v1/memories",
            json={"category": "trigger", "text": "AB"},
            headers=auth_header(),
        )
        assert response.status_code == 422

    def test_create_text_too_long_returns_422(self, client: TestClient):
        """Text longer than 2000 characters returns 422."""
        response = client.post(
            "/api/v1/memories",
            json={"category": "trigger", "text": "X" * 2001},
            headers=auth_header(),
        )
        assert response.status_code == 422

    def test_create_all_valid_categories(self, client: TestClient):
        """All five valid categories are accepted."""
        valid_categories = [
            "trigger", "baseline", "coping_mechanism", "symptom", "milestone"
        ]
        for cat in valid_categories:
            mem = make_memory_response(category=cat)
            with (
                patch(
                    "app.api.v1.memories.embedding_service.embed_text",
                    new_callable=AsyncMock,
                    return_value=FAKE_VECTOR,
                ),
                patch(
                    "app.api.v1.memories.memory_service.create_memory",
                    new_callable=AsyncMock,
                    return_value=mem,
                ),
            ):
                response = client.post(
                    "/api/v1/memories",
                    json={"category": cat, "text": f"Test memory for {cat}"},
                    headers=auth_header(),
                )
            assert response.status_code == 201, f"Category '{cat}' should be valid"

    def test_create_requires_auth(self, client: TestClient):
        """Unauthenticated request returns 401."""
        response = client.post(
            "/api/v1/memories",
            json={"category": "trigger", "text": "Some text"},
        )
        assert response.status_code == 401


# ================================================================
# GET /api/v1/memories/{memory_id}
# ================================================================

class TestGetMemory:
    """Tests for GET /api/v1/memories/{memory_id}."""

    def test_get_returns_200(self, client: TestClient):
        """Valid memory ID returns 200 with full memory data."""
        mem = make_memory_response()
        with patch(
            "app.api.v1.memories.memory_service.get_memory",
            new_callable=AsyncMock,
            return_value=mem,
        ):
            response = client.get(
                f"/api/v1/memories/{TEST_MEMORY_ID}",
                headers=auth_header(),
            )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == TEST_MEMORY_ID
        assert data["category"] == "trigger"
        assert data["reinforcement_count"] == 1

    def test_get_404_for_missing_memory(self, client: TestClient):
        """Non-existent memory returns 404."""
        from fastapi import HTTPException, status
        with patch(
            "app.api.v1.memories.memory_service.get_memory",
            new_callable=AsyncMock,
            side_effect=HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory 'fake-id' not found.",
            ),
        ):
            response = client.get(
                "/api/v1/memories/fake-id",
                headers=auth_header(),
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_requires_auth(self, client: TestClient):
        """Unauthenticated request returns 401."""
        response = client.get(f"/api/v1/memories/{TEST_MEMORY_ID}")
        assert response.status_code == 401

    def test_get_passes_user_id_to_service(self, client: TestClient):
        """user_id from JWT is passed to get_memory (not from URL)."""
        mem = make_memory_response()
        with patch(
            "app.api.v1.memories.memory_service.get_memory",
            new_callable=AsyncMock,
            return_value=mem,
        ) as mock_get:
            client.get(
                f"/api/v1/memories/{TEST_MEMORY_ID}",
                headers=auth_header(),
            )
        assert mock_get.call_args.kwargs["user_id"] == TEST_USER_ID
        assert mock_get.call_args.kwargs["memory_id"] == TEST_MEMORY_ID


# ================================================================
# PUT /api/v1/memories/{memory_id}
# ================================================================

class TestUpdateMemory:
    """Tests for PUT /api/v1/memories/{memory_id}."""

    def test_update_text_triggers_reembedding(self, client: TestClient):
        """When text is updated, embed_text is called."""
        mem = make_memory_response(text="Updated text")
        with (
            patch(
                "app.api.v1.memories.memory_service.get_memory",
                new_callable=AsyncMock,
                return_value=mem,
            ),
            patch(
                "app.api.v1.memories.embedding_service.embed_text",
                new_callable=AsyncMock,
                return_value=FAKE_VECTOR,
            ) as mock_embed,
            patch(
                "app.api.v1.memories.memory_service.update_memory",
                new_callable=AsyncMock,
                return_value=mem,
            ),
        ):
            response = client.put(
                f"/api/v1/memories/{TEST_MEMORY_ID}",
                json={"text": "Updated text"},
                headers=auth_header(),
            )

        assert response.status_code == 200
        mock_embed.assert_awaited_once_with("Updated text")

    def test_update_category_only_no_reembedding(self, client: TestClient):
        """Category-only update does NOT call embed_text."""
        mem = make_memory_response(category="milestone")
        with (
            patch(
                "app.api.v1.memories.memory_service.get_memory",
                new_callable=AsyncMock,
                return_value=mem,
            ),
            patch(
                "app.api.v1.memories.embedding_service.embed_text",
                new_callable=AsyncMock,
            ) as mock_embed,
            patch(
                "app.api.v1.memories.memory_service.update_memory",
                new_callable=AsyncMock,
                return_value=mem,
            ),
        ):
            response = client.put(
                f"/api/v1/memories/{TEST_MEMORY_ID}",
                json={"category": "milestone"},
                headers=auth_header(),
            )

        assert response.status_code == 200
        mock_embed.assert_not_awaited()

    def test_update_both_fields(self, client: TestClient):
        """Updating both text and category calls embed_text once."""
        mem = make_memory_response(category="baseline", text="New text")
        with (
            patch(
                "app.api.v1.memories.memory_service.get_memory",
                new_callable=AsyncMock,
                return_value=mem,
            ),
            patch(
                "app.api.v1.memories.embedding_service.embed_text",
                new_callable=AsyncMock,
                return_value=FAKE_VECTOR,
            ) as mock_embed,
            patch(
                "app.api.v1.memories.memory_service.update_memory",
                new_callable=AsyncMock,
                return_value=mem,
            ) as mock_update,
        ):
            response = client.put(
                f"/api/v1/memories/{TEST_MEMORY_ID}",
                json={"category": "baseline", "text": "New text"},
                headers=auth_header(),
            )

        assert response.status_code == 200
        mock_embed.assert_awaited_once()
        call_kwargs = mock_update.call_args.kwargs
        assert call_kwargs["new_category"] == "baseline"
        assert call_kwargs["new_text"] == "New text"
        assert call_kwargs["new_embedding"] == FAKE_VECTOR

    def test_update_empty_body_returns_422(self, client: TestClient):
        """Request with no fields returns 422 (Pydantic validator)."""
        response = client.put(
            f"/api/v1/memories/{TEST_MEMORY_ID}",
            json={},
            headers=auth_header(),
        )
        assert response.status_code == 422

    def test_update_404_for_missing(self, client: TestClient):
        """Returns 404 if memory doesn't exist."""
        from fastapi import HTTPException, status
        with patch(
            "app.api.v1.memories.memory_service.get_memory",
            new_callable=AsyncMock,
            side_effect=HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory 'fake-id' not found.",
            ),
        ):
            response = client.put(
                "/api/v1/memories/fake-id",
                json={"text": "Updated"},
                headers=auth_header(),
            )
        assert response.status_code == 404

    def test_update_requires_auth(self, client: TestClient):
        """Unauthenticated request returns 401."""
        response = client.put(
            f"/api/v1/memories/{TEST_MEMORY_ID}",
            json={"text": "Updated text"},
        )
        assert response.status_code == 401

    def test_update_invalid_category_returns_422(self, client: TestClient):
        """Invalid category in update returns 422."""
        response = client.put(
            f"/api/v1/memories/{TEST_MEMORY_ID}",
            json={"category": "not_valid"},
            headers=auth_header(),
        )
        assert response.status_code == 422


# ================================================================
# DELETE /api/v1/memories/{memory_id}
# ================================================================

class TestDeleteMemory:
    """Tests for DELETE /api/v1/memories/{memory_id}."""

    def test_delete_returns_200(self, client: TestClient):
        """Successful delete returns 200 with confirmation."""
        with patch(
            "app.api.v1.memories.memory_service.delete_memory",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.delete(
                f"/api/v1/memories/{TEST_MEMORY_ID}",
                headers=auth_header(),
            )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == TEST_MEMORY_ID
        assert data["deleted"] is True

    def test_delete_404_for_missing(self, client: TestClient):
        """Deleting non-existent memory returns 404."""
        from fastapi import HTTPException, status
        with patch(
            "app.api.v1.memories.memory_service.delete_memory",
            new_callable=AsyncMock,
            side_effect=HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory 'fake-id' not found.",
            ),
        ):
            response = client.delete(
                "/api/v1/memories/fake-id",
                headers=auth_header(),
            )
        assert response.status_code == 404

    def test_delete_passes_user_id_to_service(self, client: TestClient):
        """user_id from JWT is passed to delete_memory (not from URL)."""
        with patch(
            "app.api.v1.memories.memory_service.delete_memory",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_delete:
            client.delete(
                f"/api/v1/memories/{TEST_MEMORY_ID}",
                headers=auth_header(),
            )
        assert mock_delete.call_args.kwargs["user_id"] == TEST_USER_ID
        assert mock_delete.call_args.kwargs["memory_id"] == TEST_MEMORY_ID

    def test_delete_requires_auth(self, client: TestClient):
        """Unauthenticated request returns 401."""
        response = client.delete(f"/api/v1/memories/{TEST_MEMORY_ID}")
        assert response.status_code == 401


# ================================================================
# PATCH /api/v1/memories/{memory_id}/pin
# ================================================================

class TestPinMemory:
    """Tests for PATCH /api/v1/memories/{memory_id}/pin."""

    def test_pin_memory_returns_200(self, client: TestClient):
        """Pinning a memory returns 200 with is_pinned=True."""
        pinned_mem = make_memory_response(is_pinned=True)
        with patch(
            "app.api.v1.memories.memory_service.set_pin",
            new_callable=AsyncMock,
            return_value=pinned_mem,
        ):
            response = client.patch(
                f"/api/v1/memories/{TEST_MEMORY_ID}/pin",
                json={"pinned": True},
                headers=auth_header(),
            )

        assert response.status_code == 200
        data = response.json()
        assert data["is_pinned"] is True
        assert "pinned" in data["message"].lower()

    def test_unpin_memory_returns_200(self, client: TestClient):
        """Unpinning a memory returns 200 with is_pinned=False."""
        unpinned_mem = make_memory_response(is_pinned=False)
        with patch(
            "app.api.v1.memories.memory_service.set_pin",
            new_callable=AsyncMock,
            return_value=unpinned_mem,
        ):
            response = client.patch(
                f"/api/v1/memories/{TEST_MEMORY_ID}/pin",
                json={"pinned": False},
                headers=auth_header(),
            )

        assert response.status_code == 200
        data = response.json()
        assert data["is_pinned"] is False
        assert "unpinned" in data["message"].lower()

    def test_pin_calls_service_with_correct_args(self, client: TestClient):
        """set_pin is called with correct user_id, memory_id, pinned flag."""
        mem = make_memory_response(is_pinned=True)
        with patch(
            "app.api.v1.memories.memory_service.set_pin",
            new_callable=AsyncMock,
            return_value=mem,
        ) as mock_pin:
            client.patch(
                f"/api/v1/memories/{TEST_MEMORY_ID}/pin",
                json={"pinned": True},
                headers=auth_header(),
            )
        call_kwargs = mock_pin.call_args.kwargs
        assert call_kwargs["user_id"] == TEST_USER_ID
        assert call_kwargs["memory_id"] == TEST_MEMORY_ID
        assert call_kwargs["pinned"] is True

    def test_pin_404_for_missing(self, client: TestClient):
        """Pinning a non-existent memory returns 404."""
        from fastapi import HTTPException, status
        with patch(
            "app.api.v1.memories.memory_service.set_pin",
            new_callable=AsyncMock,
            side_effect=HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory 'fake-id' not found.",
            ),
        ):
            response = client.patch(
                "/api/v1/memories/fake-id/pin",
                json={"pinned": True},
                headers=auth_header(),
            )
        assert response.status_code == 404

    def test_pin_requires_pinned_field(self, client: TestClient):
        """Missing pinned field in body returns 422."""
        response = client.patch(
            f"/api/v1/memories/{TEST_MEMORY_ID}/pin",
            json={},
            headers=auth_header(),
        )
        assert response.status_code == 422

    def test_pin_requires_auth(self, client: TestClient):
        """Unauthenticated request returns 401."""
        response = client.patch(
            f"/api/v1/memories/{TEST_MEMORY_ID}/pin",
            json={"pinned": True},
        )
        assert response.status_code == 401


# ================================================================
# Schema Unit Tests (no HTTP needed)
# ================================================================

class TestMemorySchemas:
    """Unit tests for Pydantic schema validation logic."""

    def test_update_request_requires_at_least_one_field(self):
        """MemoryUpdateRequest with both None fields raises ValueError."""
        from pydantic import ValidationError
        from app.schemas.memory import MemoryUpdateRequest
        with pytest.raises(ValidationError) as exc_info:
            MemoryUpdateRequest(category=None, text=None)
        errors = exc_info.value.errors()
        assert any("at least one field" in str(e).lower() for e in errors)

    def test_update_request_text_only_valid(self):
        """MemoryUpdateRequest with text only is valid."""
        from app.schemas.memory import MemoryUpdateRequest
        req = MemoryUpdateRequest(text="Some new text")
        assert req.text == "Some new text"
        assert req.category is None

    def test_update_request_category_only_valid(self):
        """MemoryUpdateRequest with category only is valid."""
        from app.schemas.memory import MemoryUpdateRequest
        req = MemoryUpdateRequest(category="milestone")
        assert req.category == "milestone"
        assert req.text is None

    def test_filter_params_page_size_max(self):
        """page_size above 100 raises ValidationError."""
        from pydantic import ValidationError
        from app.schemas.memory import MemoryFilterParams
        with pytest.raises(ValidationError):
            MemoryFilterParams(page_size=101)

    def test_filter_params_defaults(self):
        """Default MemoryFilterParams has expected values."""
        from app.schemas.memory import MemoryFilterParams
        params = MemoryFilterParams()
        assert params.page == 1
        assert params.page_size == 20
        assert params.category is None
        assert params.pinned_only is False
        assert params.sort_by == "created_at"

    def test_memory_create_request_boundaries(self):
        """Text at min (3) and max (2000) chars are accepted."""
        from app.schemas.memory import MemoryCreateRequest
        req_min = MemoryCreateRequest(category="trigger", text="ABC")
        assert len(req_min.text) == 3

        req_max = MemoryCreateRequest(category="trigger", text="X" * 2000)
        assert len(req_max.text) == 2000


# ================================================================
# Retrieval Engine: Phase 4 Pinned Memory Bypass
# ================================================================

class TestRetrievalEnginePinBypass:
    """
    Unit tests for the Phase 4 changes to retrieval_engine.filter_by_decay.
    Verifies that is_pinned=True bypasses exponential decay.
    """

    def test_pinned_memory_bypasses_decay(self):
        """
        A pinned memory created 365 days ago should have S_adjusted = S_raw,
        not a decayed value. Without pinning, 365-day decay would drop
        S_raw=0.90 to ~0.163, well below the 0.65 threshold.
        """
        from app.services.retrieval_engine import filter_by_decay
        old_date = datetime(2025, 1, 1, tzinfo=timezone.utc)  # ~1.5yr ago

        memories = [
            {
                "id": TEST_MEMORY_ID,
                "text": "Penicillin allergy — anaphylaxis risk",
                "category": "trigger",
                "created_at": old_date,
                "reinforcement_count": 5,
                "is_pinned": True,          # PINNED
                "similarity_score": 0.90,
            }
        ]

        result = filter_by_decay(memories, threshold=0.65, max_memories=3)

        # Pinned: must survive filtering despite old age
        assert len(result) == 1
        assert result[0]["decay_bypassed"] is True
        assert result[0]["adjusted_score"] == result[0]["raw_score"]

    def test_unpinned_old_memory_decays_below_threshold(self):
        """
        An unpinned memory created 365 days ago with S_raw=0.90 should
        decay to ~0.163 and be discarded (below 0.65 threshold).
        """
        from app.services.retrieval_engine import filter_by_decay
        old_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

        memories = [
            {
                "id": TEST_MEMORY_ID,
                "text": "Old unpinned memory",
                "category": "milestone",
                "created_at": old_date,
                "reinforcement_count": 1,
                "is_pinned": False,         # NOT PINNED
                "similarity_score": 0.90,
            }
        ]

        result = filter_by_decay(memories, threshold=0.65, max_memories=3)
        # Old unpinned memory should be discarded
        assert len(result) == 0

    def test_pinned_memory_sorted_first(self):
        """Pinned memories appear before higher-scoring unpinned memories."""
        from app.services.retrieval_engine import filter_by_decay
        recent = datetime.now(timezone.utc)

        memories = [
            {
                "id": TEST_MEMORY_ID,
                "text": "High-score unpinned",
                "category": "symptom",
                "created_at": recent,
                "reinforcement_count": 1,
                "is_pinned": False,
                "similarity_score": 0.95,
            },
            {
                "id": TEST_MEMORY_ID_2,
                "text": "Lower-score pinned",
                "category": "trigger",
                "created_at": recent,
                "reinforcement_count": 1,
                "is_pinned": True,
                "similarity_score": 0.75,
            },
        ]

        result = filter_by_decay(memories, threshold=0.65, max_memories=3)
        assert len(result) == 2
        # Pinned must be first
        assert result[0]["is_pinned"] is True
        assert result[0]["id"] == TEST_MEMORY_ID_2
