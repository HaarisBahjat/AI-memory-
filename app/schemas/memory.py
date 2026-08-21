"""
============================================================
app/schemas/memory.py — Pydantic v2 Schemas for Memory CRUD (Phase 4)
============================================================
PURPOSE:
    Defines every request and response shape for the
    /api/v1/memories endpoints. Pydantic v2 validates
    all fields at the FastAPI boundary before any service
    or database logic runs.

CATEGORIES:
    Five canonical categories map to distinct retrieval
    strategies in the retrieval engine:
        - trigger          : Known anxiety/mood triggers
        - baseline         : Resting-state health baselines
        - coping_mechanism : Effective coping strategies
        - symptom          : Observed physical/psychological symptoms
        - milestone        : Positive progress / breakthroughs

CONNECTED TO:
    Phase 4  → app/api/v1/memories.py   (request/response)
    Phase 4  → app/services/memory_service.py (domain models)
    Phase 7  → consolidation.py creates MemoryResponse-shaped records
    Phase 9  → benchmark tests assert on MemoryResponse fields
============================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# -------------------------------------------------------
# Category Enum (as a Literal for Pydantic v2 compatibility)
# -------------------------------------------------------
# Using Literal instead of enum.Enum keeps JSON serialization
# clean (no ".value" unwrapping) and plays nicely with OpenAPI.
MemoryCategory = Literal[
    "trigger",
    "baseline",
    "coping_mechanism",
    "symptom",
    "milestone",
]

VALID_SORT_BY = Literal["created_at", "reinforcement_count", "category"]


# -------------------------------------------------------
# Filter / Query Parameters
# -------------------------------------------------------

class MemoryFilterParams(BaseModel):
    """
    Query parameters for GET /api/v1/memories.

    All fields are optional — omitting them returns all
    memories for the authenticated user without filtering.

    Pagination is 1-indexed (page=1 is the first page).
    """
    category: Optional[MemoryCategory] = Field(
        default=None,
        description="Filter by memory category.",
    )
    pinned_only: bool = Field(
        default=False,
        description="If true, return only pinned memories.",
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-indexed).",
    )
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Items per page. Max 100.",
    )
    sort_by: VALID_SORT_BY = Field(
        default="created_at",
        description="Sort field. Always descending.",
    )


# -------------------------------------------------------
# Create Request
# -------------------------------------------------------

class MemoryCreateRequest(BaseModel):
    """
    Request body for POST /api/v1/memories.

    The `text` is embedded server-side via text-embedding-3-small.
    The caller only supplies human-readable text and a category.

    Max text length of 2000 characters keeps embedding tokens
    well within the 8191-token model limit while enforcing
    concise memory storage.
    """
    category: MemoryCategory = Field(
        ...,
        description="Memory category for retrieval filtering.",
        examples=["trigger"],
    )
    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Human-readable fact text to store and embed.",
        examples=["User is allergic to Penicillin and experiences anaphylaxis."],
    )


# -------------------------------------------------------
# Update Request
# -------------------------------------------------------

class MemoryUpdateRequest(BaseModel):
    """
    Request body for PUT /api/v1/memories/{memory_id}.

    Both fields are optional, but at least one must be present.
    If `text` is updated, the embedding is automatically
    re-generated via a fresh OpenAI API call before the DB write.
    If only `category` is changed, no re-embedding occurs
    (saves ~200ms and one API call).
    """
    category: Optional[MemoryCategory] = Field(
        default=None,
        description="New category for the memory.",
    )
    text: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=2000,
        description="Updated fact text. Triggers automatic re-embedding.",
    )

    @model_validator(mode="after")
    def at_least_one_field(self) -> "MemoryUpdateRequest":
        """Reject requests that send no fields to update."""
        if self.category is None and self.text is None:
            raise ValueError(
                "At least one field (category or text) must be provided."
            )
        return self


# -------------------------------------------------------
# Pin Request
# -------------------------------------------------------

class MemoryPinRequest(BaseModel):
    """
    Request body for PATCH /api/v1/memories/{memory_id}/pin.

    Setting pinned=True causes the retrieval engine to
    bypass time-decay scoring for this memory — it is always
    included in LLM context regardless of age.

    Example use cases:
        - "User has a severe Penicillin allergy" (critical safety fact)
        - "User's primary anxiety trigger is public speaking"
    """
    pinned: bool = Field(
        ...,
        description="True to pin (bypass decay). False to unpin (restore decay).",
    )


# -------------------------------------------------------
# Response Schemas
# -------------------------------------------------------

class MemoryResponse(BaseModel):
    """
    Full memory representation returned by all read/write endpoints.

    Fields map 1:1 to the `semantic_memories` table columns,
    with `is_pinned` added in Phase 4.

    `adjusted_score` is only populated on similarity-search
    results (fetch_semantic_memories); it is None for
    direct CRUD responses.
    """
    id: str = Field(description="UUID primary key.")
    user_id: str = Field(description="Owner's user_id (UUID).")
    category: MemoryCategory = Field(description="Memory category.")
    text: str = Field(description="Human-readable fact text.")
    reinforcement_count: int = Field(
        description="How many times this fact was reinforced by consolidation runs.",
    )
    is_pinned: bool = Field(
        description="Pinned memories bypass time-decay scoring in retrieval.",
    )
    created_at: datetime = Field(
        description="Creation timestamp. Reset on reinforcement (Phase 7).",
    )

    model_config = {"from_attributes": True}


class MemoryListResponse(BaseModel):
    """
    Paginated list response for GET /api/v1/memories.

    `total` reflects the count of all matching memories
    (not just the current page), enabling client-side
    pagination UI.
    """
    items: list[MemoryResponse]
    total: int = Field(description="Total matching records across all pages.")
    page: int = Field(description="Current page (1-indexed).")
    page_size: int = Field(description="Items returned per page.")
    total_pages: int = Field(description="Total number of pages.")


class MemoryPinResponse(BaseModel):
    """Minimal response confirming a pin/unpin operation."""
    id: str
    is_pinned: bool
    message: str


class MemoryDeleteResponse(BaseModel):
    """Confirmation response for DELETE /api/v1/memories/{memory_id}."""
    id: str
    deleted: bool = True
    message: str = "Memory permanently deleted."
