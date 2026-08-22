"""
============================================================
app/services/triage_service.py — Phase 6 Safety Triage Service
============================================================
PURPOSE:
    Wraps the existing evaluate_clinical_safety() screener with
    production-grade persistence and alerting:

    1. Call evaluate_clinical_safety() (pure regex, O(1), sync)
    2. If crisis detected:
       a. Optionally compute session_hash for idempotent dedup
       b. INSERT a row into triage_events (async DB)
       c. Dispatch an external alert (email/Slack) via alert_dispatcher
    3. Return the original SafetyResult to the caller

IDEMPOTENCY:
    If SAFETY_SESSION_HASH_ENABLED=True, a SHA-256 of
    (user_id:session_id:crisis_type) is stored in session_hash.
    A UNIQUE index on session_hash prevents duplicate rows on retry.
    On IntegrityError, the function treats it as a no-op (idempotent).

FAILURE ISOLATION:
    DB write failures or alert dispatch failures are caught and
    logged at WARNING level. The function always returns the
    SafetyResult — a storage failure must never block the
    crisis response from reaching the user.

CONNECTED TO:
    Phase 6  → Called from app/api/v1/chat.py (replaces direct call)
    Phase 6  → app/core/safety_triage.evaluate_clinical_safety()
    Phase 6  → app/services/alert_dispatcher.dispatch()
    Phase 6  → triage_events table (via SQLAlchemy)
    Phase 7  → Nightly archival reads triage_events.archived_at
    Phase 8  → CASCADE delete on user deletion
============================================================
"""
import hashlib
import structlog
from typing import Optional
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.safety_triage import evaluate_clinical_safety, SafetyResult
from app.services import alert_dispatcher

log = structlog.get_logger(__name__)
settings = get_settings()


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _compute_session_hash(user_id: str, session_id: str, crisis_type: str) -> str:
    """
    Returns a deterministic SHA-256 hash of (user_id:session_id:crisis_type).
    Used to deduplicate triage_events rows on retry when
    SAFETY_SESSION_HASH_ENABLED=True.
    """
    raw = f"{user_id}:{session_id}:{crisis_type}"
    return hashlib.sha256(raw.encode()).hexdigest()


async def _persist_triage_event(
    db: AsyncSession,
    *,
    user_id: str,
    session_id: str,
    crisis_type: str,
    triggered_by: Optional[str],
    resources: Optional[list],
    alert_sent: bool,
) -> Optional[str]:
    """
    Inserts a row into triage_events.

    Returns:
        The new row's UUID string on success.
        None on IntegrityError (duplicate hash) or any other DB error.

    All exceptions are caught — a DB write failure must never
    prevent the crisis response from being returned to the user.
    """
    session_hash: Optional[str] = None
    if settings.SAFETY_SESSION_HASH_ENABLED:
        session_hash = _compute_session_hash(user_id, session_id, crisis_type)

    try:
        result = await db.execute(
            text("""
                INSERT INTO triage_events (
                    user_id, session_id, crisis_type, severity,
                    triggered_by, resources, session_hash, alert_sent
                ) VALUES (
                    :user_id, :session_id, :crisis_type, 'HIGH',
                    :triggered_by, :resources::jsonb, :session_hash, :alert_sent
                )
                ON CONFLICT (session_hash) WHERE session_hash IS NOT NULL
                DO NOTHING
                RETURNING id
            """),
            {
                "user_id": user_id,
                "session_id": session_id,
                "crisis_type": crisis_type,
                "triggered_by": triggered_by,
                "resources": __import__("json").dumps(resources) if resources else None,
                "session_hash": session_hash,
                "alert_sent": alert_sent,
            },
        )
        await db.commit()
        row = result.fetchone()
        if row:
            log.info(
                "Triage event persisted",
                triage_id=row[0],
                user_id=user_id,
                crisis_type=crisis_type,
                session_hash=session_hash,
            )
            return row[0]
        else:
            # ON CONFLICT DO NOTHING hit — duplicate, idempotent no-op
            log.info(
                "Triage event skipped (duplicate session_hash)",
                user_id=user_id,
                crisis_type=crisis_type,
                session_hash=session_hash,
            )
            return None
    except IntegrityError as e:
        await db.rollback()
        log.warning(
            "Triage event insert integrity error (duplicate?)",
            user_id=user_id,
            error=str(e),
        )
        return None
    except Exception as e:
        await db.rollback()
        log.error(
            "Triage event DB write failed — continuing without persistence",
            user_id=user_id,
            crisis_type=crisis_type,
            error=str(e),
        )
        return None


# -------------------------------------------------------
# Public API
# -------------------------------------------------------

async def evaluate_and_store(
    *,
    message: str,
    user_id: str,
    session_id: str,
    db: AsyncSession,
    redis: aioredis.Redis,
) -> SafetyResult:
    """
    Phase 6 drop-in replacement for evaluate_clinical_safety().

    Behaviour:
        1. Runs the keyword-based safety screener (synchronous, <1ms).
        2. If crisis detected:
           a. Dispatches external alert (email/Slack) via Redis rate-limit.
           b. Persists a triage_events row asynchronously.
        3. Returns SafetyResult in ALL cases — never raises.

    Args:
        message    : Raw user message to screen.
        user_id    : Authenticated user ID (from JWT).
        session_id : Active Redis session key used as an identifier.
        db         : Async DB session (from FastAPI dependency).
        redis      : Async Redis client (from FastAPI dependency).

    Returns:
        SafetyResult(is_safe=True)  — no crisis, proceed to RAG pipeline.
        SafetyResult(is_safe=False, triage_response=...) — crisis detected.
    """
    # Step 1: Run the pure-regex screener (no I/O, always fast)
    safety_result = evaluate_clinical_safety(message=message, user_id=user_id)

    if safety_result.is_safe:
        return safety_result

    # Step 2: Crisis detected — grab triage details
    triage = safety_result.triage_response
    crisis_type = triage.crisis_type

    log.warning(
        "Crisis triage fired — persisting event",
        user_id=user_id,
        crisis_type=crisis_type,
        session_id=session_id,
    )

    # Step 3: Dispatch external alert (with Redis rate-limit)
    alert_sent = False
    try:
        alert_sent = await alert_dispatcher.dispatch(
            redis=redis,
            user_id=user_id,
            session_id=session_id,
            crisis_type=crisis_type,
        )
    except Exception as e:
        log.error("Alert dispatch raised unexpectedly", error=str(e))

    # Step 4: Persist the triage event to the DB
    await _persist_triage_event(
        db=db,
        user_id=user_id,
        session_id=session_id,
        crisis_type=crisis_type,
        triggered_by=triage.triggered_by,    # Internal audit field — not shown to users
        resources=triage.resources,
        alert_sent=alert_sent,
    )

    return safety_result
