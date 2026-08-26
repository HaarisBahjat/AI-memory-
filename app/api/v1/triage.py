"""
============================================================
app/api/v1/triage.py — Phase 6 Triage Events Admin API
============================================================
PURPOSE:
    Admin-only endpoints for viewing and managing persisted
    crisis triage events. These endpoints are secured by JWT
    and restricted to admin users via the `require_admin`
    dependency (currently treats all authenticated users as
    admins — Phase 8 will add role-based access control).

    GET    /api/v1/triage              → Paginated list (filterable)
    GET    /api/v1/triage/{id}         → Single event detail
    DELETE /api/v1/triage/{id}         → Soft-delete (set archived_at)

SECURITY MODEL:
    - All endpoints require a valid Bearer JWT.
    - user_id is extracted exclusively from the validated token.
    - Phase 8 will add an admin role check here.

NOTE ON triggered_by:
    The `triggered_by` field (raw regex pattern) is NEVER included
    in API responses. It is an internal audit field stored in the DB.
    It is visible only to engineers with direct DB access.

CONNECTED TO:
    Phase 6  → app/services/triage_service.py (writes)
    Phase 6  → app/schemas/triage.py (schemas)
    Phase 6  → triage_events table
    Phase 8  → Role-based admin guard added here
============================================================
"""
import math
import structlog
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_admin
from app.core.database import get_db
from app.schemas.auth import CurrentUser
from app.schemas.triage import (
    TriageEventListResponse,
    TriageEventResponse,
    TriageEventSoftDeleteResponse,
)

log = structlog.get_logger(__name__)
router = APIRouter()


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _build_where(
    user_id_filter: Optional[str],
    crisis_type_filter: Optional[str],
    active_only: bool,
) -> tuple[str, dict]:
    """
    Builds a parameterized WHERE clause from the provided filters.
    Returns (where_sql_fragment, params_dict).
    """
    clauses = []
    params: dict = {}

    if user_id_filter:
        clauses.append("user_id = :filter_user_id")
        params["filter_user_id"] = user_id_filter

    if crisis_type_filter:
        clauses.append("crisis_type = :crisis_type")
        params["crisis_type"] = crisis_type_filter

    if active_only:
        clauses.append("archived_at IS NULL")

    where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
    return where_sql, params


# -------------------------------------------------------
# GET /triage — Paginated list
# -------------------------------------------------------

@router.get(
    "/triage",
    response_model=TriageEventListResponse,
    summary="List crisis triage events (admin)",
    description=(
        "Returns a paginated list of crisis triage events. "
        "Filterable by user_id, crisis_type, and active status. "
        "Requires a valid Bearer token. "
        "Phase 8 will restrict this to admin role only."
    ),
)
async def list_triage_events(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    user_id_filter: Optional[str] = Query(default=None, alias="user_id", description="Filter by user_id"),
    crisis_type: Optional[str] = Query(default=None, description="Filter by crisis type"),
    active_only: bool = Query(default=True, description="Exclude archived events"),
    current_user: CurrentUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    where_sql, params = _build_where(user_id_filter, crisis_type, active_only)
    params["limit"] = page_size
    params["offset"] = offset

    # Count query
    count_result = await db.execute(
        text(f"SELECT COUNT(*) FROM triage_events {where_sql}"),
        params,
    )
    total = count_result.scalar() or 0

    # Data query — triggered_by is intentionally excluded
    rows_result = await db.execute(
        text(f"""
            SELECT
                id, user_id, session_id, created_at,
                crisis_type, severity, resources,
                confidence, alert_sent, archived_at
            FROM triage_events
            {where_sql}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = rows_result.mappings().all()
    items = [TriageEventResponse(**dict(row)) for row in rows]
    total_pages = math.ceil(total / page_size) if total > 0 else 1

    log.debug(
        "Triage events listed",
        requester=current_user.user_id,
        total=total,
        page=page,
        page_size=page_size,
        active_only=active_only,
    )

    return TriageEventListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


# -------------------------------------------------------
# GET /triage/{triage_id} — Single event
# -------------------------------------------------------

@router.get(
    "/triage/{triage_id}",
    response_model=TriageEventResponse,
    summary="Get a single triage event (admin)",
)
async def get_triage_event(
    triage_id: str,
    current_user: CurrentUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        text("""
            SELECT
                id, user_id, session_id, created_at,
                crisis_type, severity, resources,
                confidence, alert_sent, archived_at
            FROM triage_events
            WHERE id = :triage_id
        """),
        {"triage_id": triage_id},
    )
    row = result.mappings().first()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Triage event {triage_id!r} not found.",
        )

    log.debug("Triage event fetched", triage_id=triage_id, requester=current_user.user_id)
    return TriageEventResponse(**dict(row))


# -------------------------------------------------------
# DELETE /triage/{triage_id} — Soft-delete (archive)
# -------------------------------------------------------

@router.delete(
    "/triage/{triage_id}",
    response_model=TriageEventSoftDeleteResponse,
    summary="Soft-delete (archive) a triage event (admin)",
    description=(
        "Sets archived_at to the current timestamp. "
        "The row is retained for audit compliance — no data is permanently deleted. "
        "Phase 8 Right-to-Forget performs hard deletes via CASCADE."
    ),
)
async def archive_triage_event(
    triage_id: str,
    current_user: CurrentUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    now = datetime.now(timezone.utc)

    result = await db.execute(
        text("""
            UPDATE triage_events
            SET archived_at = :now
            WHERE id = :triage_id AND archived_at IS NULL
            RETURNING id, archived_at
        """),
        {"triage_id": triage_id, "now": now},
    )
    row = result.fetchone()
    await db.commit()

    if row is None:
        # Either not found or already archived
        check = await db.execute(
            text("SELECT id FROM triage_events WHERE id = :triage_id"),
            {"triage_id": triage_id},
        )
        exists = check.fetchone()
        if exists is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Triage event {triage_id!r} not found.",
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Triage event {triage_id!r} is already archived.",
        )

    log.info(
        "Triage event archived",
        triage_id=triage_id,
        archived_at=str(row[1]),
        requester=current_user.user_id,
    )
    return TriageEventSoftDeleteResponse(id=row[0], archived_at=row[1])
