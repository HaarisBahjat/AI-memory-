"""
============================================================
app/api/v1/episodes.py -- Phase 5 Episode Endpoints
============================================================
PURPOSE:
    JWT-protected endpoints for inspecting Layer 2 episodic
    memory (daily session summaries).

    GET  /api/v1/episodes          -- Paginated list (date-range)
    GET  /api/v1/episodes/{id}     -- Single episode detail
    POST /api/v1/episodes/search   -- Semantic similarity search

SECURITY:
    All queries are scoped to current_user.user_id.
    Users cannot enumerate or access another user's episodes.

CONNECTED TO:
    Phase 5 -> episode_service.py writes episodes
    Phase 7 -> Consolidation reads from here
    Phase 8 -> GDPR right-to-forget deletes all user episodes
============================================================
"""
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.database import get_db
from app.models.episode import Episode
from app.schemas.auth import CurrentUser
from app.schemas.episode import (
    EpisodeListResponse,
    EpisodeResponse,
    EpisodeSearchRequest,
    ExtractedMetrics,
)
from app.services.embedding_service import embed_text

log = structlog.get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/episodes", tags=["Episodes"])


# -------------------------------------------------------
# Helper: ORM row -> EpisodeResponse
# -------------------------------------------------------

def _row_to_response(row: Episode, similarity: Optional[float] = None) -> EpisodeResponse:
    """Convert a SQLAlchemy Episode ORM instance into an API response schema."""
    metrics_dict = row.extracted_metrics or {}
    # Validate / coerce the JSONB blob through Pydantic
    metrics = ExtractedMetrics.model_validate(metrics_dict)
    return EpisodeResponse(
        id=row.id,
        user_id=row.user_id,
        timestamp=row.timestamp,
        session_summary=row.session_summary,
        extracted_metrics=metrics,
        archived_at=row.archived_at,
        similarity=similarity,
    )


# -------------------------------------------------------
# GET /api/v1/episodes
# -------------------------------------------------------

@router.get(
    "",
    response_model=EpisodeListResponse,
    summary="List your session episodes",
    description=(
        "Returns a paginated, reverse-chronological list of your daily "
        "session summaries. By default only active (non-archived) episodes "
        "within the last 14 days are returned. Use ctive_only=false to "
        "include cold-storage episodes."
    ),
    responses={
        200: {"description": "Episodes returned successfully"},
        401: {"description": "Missing or invalid authentication token"},
    },
)
async def list_episodes(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page (max 100)"),
    active_only: bool = Query(
        True, description="If true, exclude cold-storage (archived) episodes"
    ),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> EpisodeListResponse:
    """Return paginated episodes for the authenticated user."""
    uid = current_user.user_id
    offset = (page - 1) * per_page

    # Build count query
    count_q = select(func.count()).select_from(Episode).where(
        Episode.user_id == uid
    )
    if active_only:
        count_q = count_q.where(Episode.archived_at.is_(None))

    # Build data query
    data_q = (
        select(Episode)
        .where(Episode.user_id == uid)
        .order_by(Episode.timestamp.desc())
        .limit(per_page)
        .offset(offset)
    )
    if active_only:
        data_q = data_q.where(Episode.archived_at.is_(None))

    try:
        total_result = await db.execute(count_q)
        total = total_result.scalar_one()

        rows_result = await db.execute(data_q)
        rows = rows_result.scalars().all()
    except Exception as e:
        log.error("Failed to list episodes", user_id=uid, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve episodes.",
        )

    items = [_row_to_response(row) for row in rows]
    log.debug("Episodes listed", user_id=uid, count=len(items), page=page)
    return EpisodeListResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        active_only=active_only,
    )


# -------------------------------------------------------
# GET /api/v1/episodes/{episode_id}
# -------------------------------------------------------

@router.get(
    "/{episode_id}",
    response_model=EpisodeResponse,
    summary="Get a single episode by ID",
    responses={
        200: {"description": "Episode returned successfully"},
        401: {"description": "Missing or invalid authentication token"},
        404: {"description": "Episode not found or does not belong to user"},
    },
)
async def get_episode(
    episode_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> EpisodeResponse:
    """Fetch a single episode. Returns 404 if missing OR owned by another user."""
    uid = current_user.user_id

    try:
        result = await db.execute(
            select(Episode).where(
                Episode.id == episode_id,
                Episode.user_id == uid,   # strict ownership check
            )
        )
        row = result.scalars().first()
    except Exception as e:
        log.error("Failed to fetch episode", user_id=uid, episode_id=episode_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve episode.",
        )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Episode not found.",
        )

    log.debug("Episode fetched", user_id=uid, episode_id=episode_id)
    return _row_to_response(row)


# -------------------------------------------------------
# POST /api/v1/episodes/search
# -------------------------------------------------------

@router.post(
    "/search",
    response_model=EpisodeListResponse,
    summary="Semantic search over your past episodes",
    description=(
        "Embeds the query string and returns the most semantically similar "
        "episode summaries using pgvector cosine distance. "
        "Episodes with a NULL embedding column are excluded from results."
    ),
    responses={
        200: {"description": "Search results returned"},
        400: {"description": "Query string is empty"},
        401: {"description": "Missing or invalid authentication token"},
        503: {"description": "Embedding service temporarily unavailable"},
    },
)
async def search_episodes(
    body: EpisodeSearchRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> EpisodeListResponse:
    """
    Vector-similarity search over Layer 2 episodic memory.

    Flow:
        1. Embed the query with text-embedding-3-small
        2. Run pgvector cosine distance query against episodes.embedding
        3. Return ranked matches with similarity scores
    """
    uid = current_user.user_id

    # Generate query embedding
    try:
        query_vector = await embed_text(body.query)
    except Exception as e:
        log.error("Search embedding failed", user_id=uid, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service temporarily unavailable. Please try again.",
        )

    vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

    archived_clause = "AND archived_at IS NULL" if body.active_only else ""

    raw_sql = text(f"""
        SELECT
            id,
            user_id,
            timestamp,
            session_summary,
            extracted_metrics,
            archived_at,
            1 - (embedding <-> :query_vector::vector) AS similarity
        FROM episodes
        WHERE
            user_id = :user_id
            AND embedding IS NOT NULL
            {archived_clause}
        ORDER BY embedding <-> :query_vector::vector
        LIMIT :limit
    """)

    try:
        result = await db.execute(
            raw_sql,
            {
                "query_vector": vector_str,
                "user_id": uid,
                "limit": body.limit,
            },
        )
        rows = result.mappings().all()
    except Exception as e:
        log.error("Episode search query failed", user_id=uid, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Episode search failed.",
        )

    items = []
    for row in rows:
        metrics = ExtractedMetrics.model_validate(row.get("extracted_metrics") or {})
        items.append(
            EpisodeResponse(
                id=row["id"],
                user_id=row["user_id"],
                timestamp=row["timestamp"],
                session_summary=row["session_summary"],
                extracted_metrics=metrics,
                archived_at=row.get("archived_at"),
                similarity=round(float(row["similarity"]), 4) if row.get("similarity") else None,
            )
        )

    log.info(
        "Episode search complete",
        user_id=uid,
        query_preview=body.query[:60],
        results=len(items),
    )
    return EpisodeListResponse(
        items=items,
        total=len(items),
        page=1,
        per_page=body.limit,
        active_only=body.active_only,
    )
