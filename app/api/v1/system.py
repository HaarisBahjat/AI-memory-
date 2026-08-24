"""
============================================================
app/api/v1/system.py -- Phase 7 System / Admin Endpoints
============================================================
PURPOSE:
    Exposes admin-only endpoints for triggering background
    maintenance tasks. Currently:

        POST /api/v1/system/consolidate
            Triggers the Phase 7 nightly memory consolidation
            batch. Returns 202 Accepted immediately; the actual
            consolidation runs in the background.

        GET  /api/v1/system/consolidation/status
            Returns the count of episodes in each status bucket
            (PENDING / PROCESSING / CONSOLIDATED / FAILED) so
            operators can monitor pipeline health.

SECURITY:
    Both endpoints require a valid JWT. In production, restrict
    further to an admin role (Phase 8 RBAC will address this).

TRIGGERING:
    External schedulers (cron, GitHub Actions, APScheduler) should
    call POST /consolidate on a nightly schedule. The API does NOT
    run the job on a timer itself -- FastAPI has no built-in scheduler.

CONNECTED TO:
    Phase 7  -> app/services/consolidation_service.run_batch()
    Phase 7  -> app/api/v1/router.py (registered here)
============================================================
"""

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import get_db
from app.api.deps import get_current_user

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/system")


# -------------------------------------------------------
# POST /system/consolidate
# -------------------------------------------------------

@router.post(
    "/consolidate",
    status_code=202,
    summary="Trigger nightly memory consolidation",
    description=(
        "Starts a background batch job that extracts durable semantic facts "
        "from all PENDING episodes and upserts them into semantic_memories. "
        "Returns 202 Accepted immediately; work runs asynchronously."
    ),
)
async def trigger_consolidation(
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    """
    Trigger the Phase 7 consolidation batch.

    This endpoint is designed to be called by an external scheduler
    (cron job, GitHub Actions, APScheduler, etc.) on a nightly schedule.

    It can also be called manually by admins for testing.

    Returns 202 immediately -- the consolidation runs in the background.
    """
    # Import here to avoid circular imports at module load time
    from app.services.consolidation_service import run_batch

    log.info(
        "Consolidation triggered via API",
        triggered_by=current_user.user_id,
    )

    background_tasks.add_task(_run_consolidation_safe)

    return {
        "status": "accepted",
        "message": "Consolidation batch started in the background.",
    }


async def _run_consolidation_safe() -> None:
    """
    Background task wrapper for consolidation.
    Catches all exceptions so FastAPI's background task runner
    never sees an unhandled error.
    """
    from app.services.consolidation_service import run_batch

    try:
        stats = await run_batch()
        log.info("Background consolidation complete", **stats)
    except Exception as e:
        log.error("Background consolidation raised an unexpected error", error=str(e))


# -------------------------------------------------------
# GET /system/consolidation/status
# -------------------------------------------------------

@router.get(
    "/consolidation/status",
    summary="Get consolidation pipeline status",
    description="Returns episode counts grouped by consolidation_status for monitoring.",
)
async def consolidation_status(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Returns a health snapshot of the consolidation pipeline:
        {
            "PENDING":      12,
            "PROCESSING":    0,
            "CONSOLIDATED": 87,
            "FAILED":        3
        }

    FAILED episodes are retried on the next consolidation run.
    """
    result = await db.execute(
        text("""
            SELECT consolidation_status, COUNT(*) AS episode_count
            FROM episodes
            GROUP BY consolidation_status
        """)
    )
    rows = result.mappings().all()

    # Build a complete dict with 0 defaults for missing statuses
    status_map = {
        "PENDING": 0,
        "PROCESSING": 0,
        "CONSOLIDATED": 0,
        "FAILED": 0,
    }
    for row in rows:
        key = row["consolidation_status"].upper()
        if key in status_map:
            status_map[key] = row["episode_count"]

    log.debug(
        "Consolidation status queried",
        user_id=current_user.user_id,
        **status_map,
    )
    return status_map

