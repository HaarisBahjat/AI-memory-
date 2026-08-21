"""
============================================================
app/api/v1/memories.py — Layer 3 Semantic Memory CRUD API (Phase 4)
============================================================
PURPOSE:
    Six JWT-protected endpoints that give authenticated users
    full visibility and control over their Layer 3 semantic
    memory store:

    GET    /api/v1/memories              → Paginated list (filterable)
    POST   /api/v1/memories              → Create a new memory
    GET    /api/v1/memories/{memory_id}  → Fetch single memory
    PUT    /api/v1/memories/{memory_id}  → Update text / category
    DELETE /api/v1/memories/{memory_id}  → Hard delete
    PATCH  /api/v1/memories/{memory_id}/pin → Toggle pin status

SECURITY MODEL:
    - All endpoints require a valid Bearer JWT.
    - user_id is extracted exclusively from the validated token.
    - memory_service enforces ownership at the SQL level —
      callers cannot access or modify another user's memories.

EMBEDDING FLOW:
    POST /memories and PUT /memories/{id} (when text changes):
        1. API layer calls embedding_service.embed_text()
        2. Vector is passed to memory_service
        3. Service writes to DB
    This synchronous design ensures consistency (no orphan rows
    with missing embeddings) at an acceptable ~200ms cost.

CONNECTED TO:
    Phase 4  → app/services/memory_service.py  (all DB logic)
    Phase 4  → app/schemas/memory.py           (all schemas)
    Phase 4  → app/services/embedding_service.py (vectorization)
    Phase 4  → app/api/deps.py                 (get_current_user)
    Phase 4  → app/api/v1/router.py            (registered here)
    Phase 7  → Consolidation creates memories via memory_service directly
============================================================
"""

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.api.deps import get_current_user
from app.core.database import get_db
from app.schemas.auth import CurrentUser
from app.schemas.memory import (
    MemoryCategory,
    MemoryCreateRequest,
    MemoryDeleteResponse,
    MemoryFilterParams,
    MemoryListResponse,
    MemoryPinRequest,
    MemoryPinResponse,
    MemoryResponse,
    MemoryUpdateRequest,
)
from app.services import embedding_service, memory_service

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/memories", tags=["Semantic Memory"])


# -------------------------------------------------------
# GET /api/v1/memories
# -------------------------------------------------------

@router.get(
    "",
    response_model=MemoryListResponse,
    summary="List semantic memories",
    description=(
        "Returns a paginated list of the authenticated user's Layer 3 semantic "
        "memories. Optionally filter by category or pinned status. "
        "Results are sorted by: pinned first, then by sort_by DESC."
    ),
    responses={
        200: {"description": "Memories retrieved successfully."},
        401: {"description": "Missing or invalid authentication token."},
    },
)
async def list_memories(
    category: Optional[MemoryCategory] = Query(
        default=None,
        description="Filter by memory category.",
    ),
    pinned_only: bool = Query(
        default=False,
        description="Return only pinned memories.",
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
    sort_by: str = Query(
        default="created_at",
        description="Sort field: created_at | reinforcement_count | category.",
    ),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MemoryListResponse:
    """
    Paginated memory list with optional category and pin filtering.

    Use cases:
    - "Show all my known triggers" → ?category=trigger
    - "Show only my pinned memories" → ?pinned_only=true
    - "Browse all memories page 2" → ?page=2&page_size=20
    """
    # Normalize sort_by to prevent invalid values reaching service
    valid_sort = {"created_at", "reinforcement_count", "category"}
    if sort_by not in valid_sort:
        sort_by = "created_at"

    filters = MemoryFilterParams(
        category=category,
        pinned_only=pinned_only,
        page=page,
        page_size=page_size,
        sort_by=sort_by,  # type: ignore[arg-type]
    )

    result = await memory_service.list_memories(
        db=db,
        user_id=current_user.user_id,
        filters=filters,
    )

    log.info(
        "Memories list requested",
        user_id=current_user.user_id,
        total=result["total"],
        page=page,
        category=category,
    )
    return MemoryListResponse(**result)


# -------------------------------------------------------
# POST /api/v1/memories
# -------------------------------------------------------

@router.post(
    "",
    response_model=MemoryResponse,
    status_code=201,
    summary="Create a new semantic memory",
    description=(
        "Manually adds a known health fact to the user's Layer 3 semantic memory. "
        "The text is automatically embedded via text-embedding-3-small before storage. "
        "Examples: 'User is allergic to Penicillin', 'Deep breathing reduces anxiety within 5 minutes'."
    ),
    responses={
        201: {"description": "Memory created and embedded successfully."},
        400: {"description": "Invalid request body."},
        401: {"description": "Missing or invalid authentication token."},
        502: {"description": "OpenAI embedding API unavailable."},
    },
)
async def create_memory(
    body: MemoryCreateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MemoryResponse:
    """
    Create a new semantic memory fact.

    Embedding flow:
        1. Validate request (Pydantic)
        2. Call OpenAI text-embedding-3-small (~200ms)
        3. Insert row into semantic_memories with vector
        4. Return the created MemoryResponse
    """
    # Embed the text — this is the only async I/O before the DB write
    vector = await embedding_service.embed_text(body.text)

    memory = await memory_service.create_memory(
        db=db,
        user_id=current_user.user_id,
        category=body.category,
        text_content=body.text,
        embedding_vector=vector,
    )

    log.info(
        "Memory created via API",
        user_id=current_user.user_id,
        memory_id=memory.id,
        category=body.category,
    )
    return memory


# -------------------------------------------------------
# GET /api/v1/memories/{memory_id}
# -------------------------------------------------------

@router.get(
    "/{memory_id}",
    response_model=MemoryResponse,
    summary="Fetch a single semantic memory",
    description=(
        "Returns the full details of a specific semantic memory. "
        "Returns 404 if the memory does not exist or belongs to another user."
    ),
    responses={
        200: {"description": "Memory returned successfully."},
        401: {"description": "Missing or invalid authentication token."},
        404: {"description": "Memory not found."},
    },
)
async def get_memory(
    memory_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MemoryResponse:
    """
    Single memory fetch with ownership enforcement.

    Returns 404 for both "not found" and "belongs to another user"
    to prevent UUID enumeration attacks.
    """
    return await memory_service.get_memory(
        db=db,
        user_id=current_user.user_id,
        memory_id=memory_id,
    )


# -------------------------------------------------------
# PUT /api/v1/memories/{memory_id}
# -------------------------------------------------------

@router.put(
    "/{memory_id}",
    response_model=MemoryResponse,
    summary="Update a semantic memory",
    description=(
        "Updates the text and/or category of an existing memory. "
        "If text is changed, the embedding is automatically re-generated "
        "via a fresh OpenAI API call. If only category changes, no re-embedding occurs."
    ),
    responses={
        200: {"description": "Memory updated successfully."},
        400: {"description": "No update fields provided."},
        401: {"description": "Missing or invalid authentication token."},
        404: {"description": "Memory not found."},
        502: {"description": "OpenAI embedding API unavailable (text update only)."},
    },
)
async def update_memory(
    memory_id: str,
    body: MemoryUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MemoryResponse:
    """
    Partial update for a semantic memory.

    Re-embedding only occurs when body.text is not None —
    saves one OpenAI API call for category-only updates.

    Ownership is verified via get_memory() which raises 404
    if the memory doesn't exist or isn't owned by the caller.
    """
    # Verify ownership before any embedding work
    await memory_service.get_memory(
        db=db,
        user_id=current_user.user_id,
        memory_id=memory_id,
    )

    # Compute new embedding only if text changed
    new_vector: list[float] | None = None
    if body.text is not None:
        new_vector = await embedding_service.embed_text(body.text)

    updated = await memory_service.update_memory(
        db=db,
        user_id=current_user.user_id,
        memory_id=memory_id,
        new_category=body.category,
        new_text=body.text,
        new_embedding=new_vector,
    )

    log.info(
        "Memory updated via API",
        user_id=current_user.user_id,
        memory_id=memory_id,
        text_updated=(body.text is not None),
        category_updated=(body.category is not None),
    )
    return updated


# -------------------------------------------------------
# DELETE /api/v1/memories/{memory_id}
# -------------------------------------------------------

@router.delete(
    "/{memory_id}",
    response_model=MemoryDeleteResponse,
    summary="Delete a semantic memory",
    description=(
        "Permanently deletes a specific semantic memory. "
        "This action is irreversible. Returns 404 if the memory "
        "does not exist or belongs to another user."
    ),
    responses={
        200: {"description": "Memory permanently deleted."},
        401: {"description": "Missing or invalid authentication token."},
        404: {"description": "Memory not found."},
    },
)
async def delete_memory(
    memory_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MemoryDeleteResponse:
    """
    Hard delete for a semantic memory.

    Ownership enforced via SQL WHERE clause in memory_service.
    Returns 404 for both missing and foreign-owned memories.
    """
    await memory_service.delete_memory(
        db=db,
        user_id=current_user.user_id,
        memory_id=memory_id,
    )

    log.info(
        "Memory deleted via API",
        user_id=current_user.user_id,
        memory_id=memory_id,
    )
    return MemoryDeleteResponse(
        id=memory_id,
        deleted=True,
        message="Memory permanently deleted.",
    )


# -------------------------------------------------------
# PATCH /api/v1/memories/{memory_id}/pin
# -------------------------------------------------------

@router.patch(
    "/{memory_id}/pin",
    response_model=MemoryPinResponse,
    summary="Pin or unpin a semantic memory",
    description=(
        "Toggles the `is_pinned` flag on a memory. "
        "Pinned memories bypass time-decay scoring in the retrieval engine "
        "and are always injected into the LLM context regardless of age. "
        "Ideal for critical, always-relevant facts (e.g. medication allergies)."
    ),
    responses={
        200: {"description": "Pin status updated."},
        401: {"description": "Missing or invalid authentication token."},
        404: {"description": "Memory not found."},
    },
)
async def pin_memory(
    memory_id: str,
    body: MemoryPinRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MemoryPinResponse:
    """
    Toggle pin status for a semantic memory.

    Pinned = True  → Memory bypasses time-decay; always in LLM context.
    Pinned = False → Memory subject to normal time-decay scoring.
    """
    updated = await memory_service.set_pin(
        db=db,
        user_id=current_user.user_id,
        memory_id=memory_id,
        pinned=body.pinned,
    )

    action = "pinned" if body.pinned else "unpinned"
    log.info(
        f"Memory {action}",
        user_id=current_user.user_id,
        memory_id=memory_id,
        is_pinned=body.pinned,
    )

    return MemoryPinResponse(
        id=updated.id,
        is_pinned=updated.is_pinned,
        message=f"Memory successfully {action}.",
    )
