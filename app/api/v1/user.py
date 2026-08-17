"""
============================================================
app/api/v1/user.py — User Management & Profile API (Phase 2)
============================================================
PURPOSE:
    Provides authenticated user self-service endpoints:
    - GET  /api/v1/user/me           → Return own profile + baseline
    - PUT  /api/v1/user/me/baseline  → Update baseline health profile (JSONB patch)
    - DELETE /api/v1/user/me/memory  → GDPR Right-to-Forget (self-delete)

    All three endpoints require a valid JWT (via get_current_user).
    The user_id is extracted from the token — not from the URL —
    ensuring users can ONLY act on their own data.

PHASE HISTORY:
    Phase 1 → Had /user/{user_id}/memory (unauthenticated path param)
    Phase 2 → Replaced with /user/me/* (JWT-authenticated, no path param)

CONNECTED TO:
    Phase 2 → app/api/deps.py      (get_current_user)
    Phase 2 → app/schemas/auth.py  (UserMeResponse, BaselineUpdateRequest)
    Phase 7 → baseline_profile JSONB updated by nightly consolidation
    Phase 8 → GDPR cascade deletion + Redis purge
============================================================
"""
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.redis_client import get_redis
from app.schemas.auth import BaselineUpdateRequest, CurrentUser, UserMeResponse
from app.services import sensory_service

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/user", tags=["User Management"])


# -------------------------------------------------------
# GET /api/v1/user/me
# -------------------------------------------------------

@router.get(
    "/me",
    response_model=UserMeResponse,
    summary="Get the authenticated user's profile",
    description="Returns the profile and baseline health configuration for the currently logged-in user.",
)
async def get_me(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserMeResponse:
    """
    Returns the full user record for the authenticated caller.

    The user_id comes from the validated JWT (via get_current_user),
    so the caller can ONLY see their own profile.
    """
    result = await db.execute(
        text("""
            SELECT user_id, email, created_at, baseline_profile
            FROM users
            WHERE user_id = :uid
        """),
        {"uid": current_user.user_id},
    )
    user = result.mappings().first()

    if not user:
        # Defensive: shouldn't happen since get_current_user already verified existence
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    return UserMeResponse(**dict(user))


# -------------------------------------------------------
# PUT /api/v1/user/me/baseline
# -------------------------------------------------------

@router.put(
    "/me/baseline",
    response_model=UserMeResponse,
    summary="Update baseline health profile",
    description=(
        "Partially updates the user's baseline_profile JSONB field. "
        "Only the provided fields are updated — omitted fields retain their current values. "
        "Merge is performed with PostgreSQL's || (JSONB concatenation) operator."
    ),
)
async def update_baseline(
    updates: BaselineUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserMeResponse:
    """
    JSONB patch endpoint for the baseline health profile.

    PostgreSQL's || operator performs a shallow merge on the JSONB field:
        {"knownTriggers": ["A", "B"]} || {"knownTriggers": ["C"]}
        → {"knownTriggers": ["C"]}   ← full replacement, not array append

    If the user wants to *append* to knownTriggers, they must send
    the full new array. This is intentional — it avoids hidden state
    and gives the client full control over the baseline content.

    Only non-None fields from the request are applied to the patch.
    """
    # Build the patch dict from only the fields explicitly sent
    patch = updates.model_dump(exclude_none=True)

    if not patch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update. Send at least one field.",
        )

    # Cast the Python dict to a JSONB literal via ::jsonb
    # SQLAlchemy's :patch bind param is passed as a JSON string
    import json
    result = await db.execute(
        text("""
            UPDATE users
            SET baseline_profile = baseline_profile || :patch::jsonb
            WHERE user_id = :uid
            RETURNING user_id, email, created_at, baseline_profile
        """),
        {"uid": current_user.user_id, "patch": json.dumps(patch)},
    )
    updated = result.mappings().first()

    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    log.info("Baseline profile updated", user_id=current_user.user_id, fields=list(patch.keys()))
    return UserMeResponse(**dict(updated))


# -------------------------------------------------------
# DELETE /api/v1/user/me/memory
# -------------------------------------------------------

@router.delete(
    "/me/memory",
    summary="Right-to-Forget: Cascade delete all personal data",
    description=(
        "Executes a single atomic PostgreSQL transaction that permanently removes "
        "ALL personal data: users, episodes, semantic_memories, biometrics_stream, "
        "and refresh_tokens rows. Also purges the Redis sensory session. "
        "This action is IRREVERSIBLE and complies with GDPR Article 17."
    ),
    responses={
        200: {"description": "All data permanently deleted"},
        500: {"description": "Deletion failed — no partial delete occurred (atomic)"},
    },
)
async def delete_my_data(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> dict:
    """
    Cascading Right-to-Forget deletion for the authenticated user.

    All tables have ON DELETE CASCADE foreign key constraints on user_id,
    so deleting from `users` auto-cascades to episodes, semantic_memories,
    biometrics_stream, and refresh_tokens. The explicit child deletes
    below are belt-and-suspenders for auditability and Phase 10 safety.
    """
    user_id = current_user.user_id
    log.info("Right-to-Forget requested by authenticated user", user_id=user_id)

    try:
        # ── Atomic SQL Transaction ─────────────────────────────────
        # Execute child deletes first (belt-and-suspenders),
        # then parent. Rollback is automatic via get_db() on exception.
        await db.execute(
            text("DELETE FROM refresh_tokens WHERE user_id = :uid"),
            {"uid": user_id},
        )
        await db.execute(
            text("DELETE FROM semantic_memories WHERE user_id = :uid"),
            {"uid": user_id},
        )
        await db.execute(
            text("DELETE FROM episodes WHERE user_id = :uid"),
            {"uid": user_id},
        )
        await db.execute(
            text("DELETE FROM biometrics_stream WHERE user_id = :uid"),
            {"uid": user_id},
        )
        await db.execute(
            text("DELETE FROM users WHERE user_id = :uid"),
            {"uid": user_id},
        )
        # Commit happens via get_db() context manager after yield

        # ── Redis Purge (outside DB transaction) ───────────────────
        await sensory_service.flush_session(redis, user_id)

        deleted_at = datetime.now(timezone.utc).isoformat()
        log.info("All user data deleted", user_id=user_id, deleted_at=deleted_at)

        return {
            "success": True,
            "user_id": user_id,
            "deleted_at": deleted_at,
            "message": (
                "All personal data has been permanently removed from our systems. "
                "This includes your conversation history, health timeline, "
                "long-term memory facts, biometric data, and authentication tokens."
            ),
        }

    except Exception as e:
        log.error("Cascading deletion failed", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data deletion failed. No partial deletion occurred. Please try again.",
        )
