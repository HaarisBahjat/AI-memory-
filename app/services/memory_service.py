"""
============================================================
app/services/memory_service.py — Layer 3 Semantic Memory CRUD (Phase 4)
============================================================
PURPOSE:
    Encapsulates all database operations for the semantic_memories
    table. The API layer calls these functions after resolving
    auth — no business logic lives in the router.

    All queries use raw SQLAlchemy text() consistent with the
    Phase 1–3 codebase pattern. Ownership is enforced at the SQL
    level: every query includes `AND user_id = :user_id`, meaning
    a user can never read, modify, or delete another user's memories
    even if they somehow obtain a valid memory UUID.

DATABASE COLUMN REQUIREMENTS (Phase 4 migration):
    ALTER TABLE semantic_memories
    ADD COLUMN IF NOT EXISTS is_pinned BOOLEAN NOT NULL DEFAULT FALSE;

    This column must exist before this service is used.
    See schema.sql for the canonical migration block.

CONNECTED TO:
    Phase 4  → app/api/v1/memories.py  (called by all endpoints)
    Phase 4  → app/schemas/memory.py   (MemoryFilterParams, MemoryResponse)
    Phase 7  → consolidation.py calls create_memory() for extracted facts
    Phase 7  → Reinforcement calls increment_reinforcement_count() (defined here)
    Phase 9  → Benchmark tests stub this module's functions
============================================================
"""

import math
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.schemas.memory import MemoryCategory, MemoryFilterParams, MemoryResponse

log = structlog.get_logger(__name__)

# -------------------------------------------------------
# Column allowlist for sort_by to prevent SQL injection
# (We never interpolate user input directly — only map to
#  a known safe column name.)
# -------------------------------------------------------
_SORT_COLUMN_MAP: dict[str, str] = {
    "created_at": "created_at",
    "reinforcement_count": "reinforcement_count",
    "category": "category",
}


# -------------------------------------------------------
# READ: List memories (paginated + filtered)
# -------------------------------------------------------

async def list_memories(
    db: AsyncSession,
    user_id: str,
    filters: MemoryFilterParams,
) -> dict:
    """
    Returns a paginated, optionally filtered list of semantic memories
    for the given user.

    Sort order (always DESC):
        1. is_pinned — pinned memories always float to the top
        2. User's selected sort_by column
        3. created_at (tiebreaker)

    Args:
        db      : Async database session
        user_id : Authenticated user's UUID
        filters : Validated MemoryFilterParams from query string

    Returns:
        Dict with keys: items, total, page, page_size, total_pages
    """
    # ── Build WHERE clause predicates ────────────────────────────
    where_clauses = ["user_id = :user_id"]
    params: dict = {"user_id": user_id}

    if filters.category is not None:
        where_clauses.append("category = :category")
        params["category"] = filters.category

    if filters.pinned_only:
        where_clauses.append("is_pinned = TRUE")

    where_sql = " AND ".join(where_clauses)

    # ── Safe sort column ──────────────────────────────────────────
    sort_col = _SORT_COLUMN_MAP.get(filters.sort_by, "created_at")

    # ── COUNT query (for pagination metadata) ────────────────────
    count_result = await db.execute(
        text(f"SELECT COUNT(*) FROM semantic_memories WHERE {where_sql}"),
        params,
    )
    total: int = count_result.scalar_one()

    # ── Data query ───────────────────────────────────────────────
    offset = (filters.page - 1) * filters.page_size
    params["limit"] = filters.page_size
    params["offset"] = offset

    rows_result = await db.execute(
        text(f"""
            SELECT
                id,
                user_id,
                category,
                text,
                reinforcement_count,
                is_pinned,
                created_at
            FROM semantic_memories
            WHERE {where_sql}
            ORDER BY is_pinned DESC, {sort_col} DESC, created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = rows_result.mappings().all()
    items = [MemoryResponse(**dict(row)) for row in rows]

    total_pages = math.ceil(total / filters.page_size) if total > 0 else 1

    log.debug(
        "Memories listed",
        user_id=user_id,
        total=total,
        page=filters.page,
        page_size=filters.page_size,
        category=filters.category,
        pinned_only=filters.pinned_only,
    )

    return {
        "items": items,
        "total": total,
        "page": filters.page,
        "page_size": filters.page_size,
        "total_pages": total_pages,
    }


# -------------------------------------------------------
# READ: Single memory fetch
# -------------------------------------------------------

async def get_memory(
    db: AsyncSession,
    user_id: str,
    memory_id: str,
) -> MemoryResponse:
    """
    Fetches a single semantic memory by ID, enforcing ownership.

    The `user_id = :user_id` clause means a 404 is returned for
    any memory not owned by the caller — whether it doesn't exist
    or belongs to another user. This prevents enumeration attacks.

    Args:
        db         : Async database session
        user_id    : Authenticated user's UUID
        memory_id  : Target memory UUID

    Returns:
        MemoryResponse for the matching row

    Raises:
        HTTPException 404 — memory not found or not owned by user
    """
    result = await db.execute(
        text("""
            SELECT
                id,
                user_id,
                category,
                text,
                reinforcement_count,
                is_pinned,
                created_at
            FROM semantic_memories
            WHERE id = :memory_id AND user_id = :user_id
        """),
        {"memory_id": memory_id, "user_id": user_id},
    )
    row = result.mappings().first()

    if row is None:
        log.warning(
            "Memory not found or access denied",
            user_id=user_id,
            memory_id=memory_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found.",
        )

    return MemoryResponse(**dict(row))


# -------------------------------------------------------
# CREATE: Insert a new memory with pre-computed embedding
# -------------------------------------------------------

async def create_memory(
    db: AsyncSession,
    user_id: str,
    category: MemoryCategory,
    text_content: str,
    embedding_vector: list[float],
) -> MemoryResponse:
    """
    Inserts a new semantic memory row with its embedding vector.

    The embedding is computed by the API layer (calling
    embedding_service.embed_text()) BEFORE this function is
    called. This keeps the service layer pure — no async I/O
    other than the database write.

    UUID generation: uses Python's uuid4() rather than
    gen_random_uuid() server-side, so we can return the
    new ID immediately without a follow-up SELECT.

    Args:
        db               : Async database session
        user_id          : Owner's UUID
        category         : Memory category
        text_content     : Human-readable fact text
        embedding_vector : 1536-dim float list from OpenAI

    Returns:
        Newly created MemoryResponse
    """
    new_id = str(uuid.uuid4())
    vector_str = "[" + ",".join(str(v) for v in embedding_vector) + "]"

    result = await db.execute(
        text("""
            INSERT INTO semantic_memories
                (id, user_id, category, text, embedding, reinforcement_count, is_pinned)
            VALUES
                (:id, :user_id, :category, :text, :embedding::vector, 1, FALSE)
            RETURNING
                id, user_id, category, text,
                reinforcement_count, is_pinned, created_at
        """),
        {
            "id": new_id,
            "user_id": user_id,
            "category": category,
            "text": text_content,
            "embedding": vector_str,
        },
    )
    row = result.mappings().first()

    log.info(
        "Memory created",
        user_id=user_id,
        memory_id=new_id,
        category=category,
        text_preview=text_content[:80],
    )
    return MemoryResponse(**dict(row))


# -------------------------------------------------------
# UPDATE: Partial update (re-embed only if text changed)
# -------------------------------------------------------

async def update_memory(
    db: AsyncSession,
    user_id: str,
    memory_id: str,
    new_category: Optional[MemoryCategory],
    new_text: Optional[str],
    new_embedding: Optional[list[float]],
) -> MemoryResponse:
    """
    Partially updates an existing memory.

    Called after the API layer has:
        1. Validated ownership (get_memory raises 404 if not found)
        2. Optionally computed a fresh embedding if text changed

    Build the SET clause dynamically from only the fields
    that were supplied — avoids overwriting unchanged columns.

    Args:
        db            : Async database session
        user_id       : Authenticated user's UUID
        memory_id     : Target memory UUID
        new_category  : Optional new category
        new_text      : Optional new fact text
        new_embedding : Pre-computed embedding (only when new_text supplied)

    Returns:
        Updated MemoryResponse

    Raises:
        HTTPException 404 — memory not found or not owned by user
    """
    set_clauses: list[str] = []
    params: dict = {"memory_id": memory_id, "user_id": user_id}

    if new_text is not None:
        set_clauses.append("text = :text")
        params["text"] = new_text

    if new_embedding is not None:
        set_clauses.append("embedding = :embedding::vector")
        params["embedding"] = "[" + ",".join(str(v) for v in new_embedding) + "]"

    if new_category is not None:
        set_clauses.append("category = :category")
        params["category"] = new_category

    # Defensive: should never happen because MemoryUpdateRequest
    # already validates at least one field is present.
    if not set_clauses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update.",
        )

    result = await db.execute(
        text(f"""
            UPDATE semantic_memories
            SET {", ".join(set_clauses)}
            WHERE id = :memory_id AND user_id = :user_id
            RETURNING
                id, user_id, category, text,
                reinforcement_count, is_pinned, created_at
        """),
        params,
    )
    row = result.mappings().first()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found.",
        )

    log.info(
        "Memory updated",
        user_id=user_id,
        memory_id=memory_id,
        text_updated=(new_text is not None),
        category_updated=(new_category is not None),
    )
    return MemoryResponse(**dict(row))


# -------------------------------------------------------
# DELETE: Hard delete single memory
# -------------------------------------------------------

async def delete_memory(
    db: AsyncSession,
    user_id: str,
    memory_id: str,
) -> None:
    """
    Permanently deletes a semantic memory.

    Ownership is enforced via `AND user_id = :user_id`.
    If the row doesn't exist or belongs to another user,
    a 404 is returned — no information leakage.

    Args:
        db        : Async database session
        user_id   : Authenticated user's UUID
        memory_id : Target memory UUID

    Raises:
        HTTPException 404 — memory not found or not owned by user
    """
    result = await db.execute(
        text("""
            DELETE FROM semantic_memories
            WHERE id = :memory_id AND user_id = :user_id
            RETURNING id
        """),
        {"memory_id": memory_id, "user_id": user_id},
    )
    deleted_row = result.first()

    if deleted_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found.",
        )

    log.info(
        "Memory deleted",
        user_id=user_id,
        memory_id=memory_id,
    )


# -------------------------------------------------------
# PIN: Toggle is_pinned flag
# -------------------------------------------------------

async def set_pin(
    db: AsyncSession,
    user_id: str,
    memory_id: str,
    pinned: bool,
) -> MemoryResponse:
    """
    Sets the `is_pinned` flag for a semantic memory.

    Pinned memories bypass time-decay scoring in the retrieval
    engine (Phase 4 update to retrieval_engine.py). This is
    appropriate for critical, always-relevant facts such as:
        - Medication allergies
        - Primary anxiety triggers
        - Active treatment plans

    Args:
        db        : Async database session
        user_id   : Authenticated user's UUID
        memory_id : Target memory UUID
        pinned    : True to pin, False to unpin

    Returns:
        Updated MemoryResponse

    Raises:
        HTTPException 404 — memory not found or not owned by user
    """
    result = await db.execute(
        text("""
            UPDATE semantic_memories
            SET is_pinned = :pinned
            WHERE id = :memory_id AND user_id = :user_id
            RETURNING
                id, user_id, category, text,
                reinforcement_count, is_pinned, created_at
        """),
        {"pinned": pinned, "memory_id": memory_id, "user_id": user_id},
    )
    row = result.mappings().first()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found.",
        )

    log.info(
        "Memory pin status updated",
        user_id=user_id,
        memory_id=memory_id,
        is_pinned=pinned,
    )
    return MemoryResponse(**dict(row))


# -------------------------------------------------------
# Phase 7 Helper: Increment reinforcement count
# -------------------------------------------------------

async def increment_reinforcement_count(
    db: AsyncSession,
    user_id: str,
    memory_id: str,
) -> None:
    """
    Increments the reinforcement_count and resets created_at
    for a semantic memory. Called by Phase 7 consolidation
    when a near-duplicate fact (cosine sim > 0.88) is found.

    Resetting created_at effectively resets the time-decay clock,
    keeping this memory "fresh" in S_adjusted calculations.

    Args:
        db        : Async database session (within consolidation transaction)
        user_id   : Owner UUID (for ownership enforcement)
        memory_id : Target memory UUID
    """
    await db.execute(
        text("""
            UPDATE semantic_memories
            SET
                reinforcement_count = reinforcement_count + 1,
                created_at = NOW()
            WHERE id = :memory_id AND user_id = :user_id
        """),
        {"memory_id": memory_id, "user_id": user_id},
    )
    log.debug(
        "Reinforcement count incremented",
        user_id=user_id,
        memory_id=memory_id,
    )


# -------------------------------------------------------
# Phase 7: Semantic Deduplication Upsert
# -------------------------------------------------------

async def upsert_semantic_fact(
    db: AsyncSession,
    user_id: str,
    category: str,
    text_content: str,
    embedding_vector: list[float],
) -> dict:
    """
    Inserts a new semantic memory or reinforces an existing one.

    Called by the Phase 7 consolidation pipeline for each fact
    extracted from a daily episode.

    Deduplication logic:
        1. Search semantic_memories for the closest existing row
           owned by the same user in the same category.
        2. pgvector's `<=>` operator returns cosine DISTANCE
           (0 = identical, 2 = maximally different).
           Lower distance means more similar. This is NOT a
           similarity score — do NOT compare it as similarity >= X.
        3. Compare: distance <= settings.SEMANTIC_MEMORY_MAX_COSINE_DISTANCE
           (default 0.12, which corresponds to cosine similarity ~0.88)
        4. If within threshold → reinforce ONLY that specific row (by ID).
        5. Otherwise → insert a new memory row.

    Fix notes (Phase 7 patch):
        - Fix #1: distance is compared directly (not converted) against
          SEMANTIC_MEMORY_MAX_COSINE_DISTANCE. Old code converted 0.88
          similarity → 0.12 distance threshold, which was correct math
          but the config name was confusingly called "SIMILARITY_THRESHOLD".
          Now the config name matches the actual comparison value.
        - Fix #2: UPDATE targets a specific row by `id = :best_id`, not
          by re-running the distance condition across multiple rows.
          This prevents accidentally reinforcing more than one memory.

    Args:
        db               : Async database session (must be within begin() block)
        user_id          : Memory owner's UUID
        category         : Must be one of the ALLOWED_CATEGORIES enum values
        text_content     : Natural-language fact extracted by LLM
        embedding_vector : 1536-dim float vector from OpenAI

    Returns:
        dict with keys:
            action   : "reinforced" | "created"
            memory_id: UUID of the affected row
    """
    settings = get_settings()
    max_distance = settings.SEMANTIC_MEMORY_MAX_COSINE_DISTANCE  # e.g. 0.12

    vector_str = "[" + ",".join(str(v) for v in embedding_vector) + "]"

    # ── Step 1: Find the single nearest neighbour (same user + same category) ──
    # Alias the distance column unambiguously as "cos_dist".
    # ORDER BY cos_dist ASC ensures the CLOSEST memory is returned first.
    nearest = await db.execute(
        text("""
            SELECT id, (embedding <=> CAST(:emb AS vector)) AS cos_dist
            FROM semantic_memories
            WHERE user_id = :uid AND category = :cat
            ORDER BY cos_dist ASC
            LIMIT 1
        """),
        {"emb": vector_str, "uid": user_id, "cat": category},
    )
    row = nearest.mappings().first()

    # ── Step 2: Decision: reinforce the best match OR insert a new row ────────
    if row is not None and row["cos_dist"] <= max_distance:
        # Near-duplicate found.
        # Fix #2: target the specific best-match row by its primary key.
        # Never update by distance condition (could accidentally hit multiple rows).
        best_id = row["id"]
        await db.execute(
            text("""
                UPDATE semantic_memories
                SET
                    reinforcement_count = reinforcement_count + 1,
                    created_at = NOW()
                WHERE id = :best_id AND user_id = :uid
            """),
            {"best_id": best_id, "uid": user_id},
        )
        log.info(
            "Memory reinforced (deduplication hit)",
            user_id=user_id,
            memory_id=best_id,
            category=category,
            cos_dist=round(row["cos_dist"], 4),
            max_distance=max_distance,
        )
        return {"action": "reinforced", "memory_id": best_id}

    else:
        # No close enough match — insert a new semantic memory row.
        new_id = str(uuid.uuid4())
        await db.execute(
            text("""
                INSERT INTO semantic_memories
                    (id, user_id, category, text, embedding, reinforcement_count, is_pinned)
                VALUES
                    (:id, :uid, :cat, :txt, CAST(:emb AS vector), 1, FALSE)
            """),
            {
                "id": new_id,
                "uid": user_id,
                "cat": category,
                "txt": text_content,
                "emb": vector_str,
            },
        )
        log.info(
            "Memory created (new semantic fact)",
            user_id=user_id,
            memory_id=new_id,
            category=category,
            text_preview=text_content[:80],
        )
        return {"action": "created", "memory_id": new_id}
