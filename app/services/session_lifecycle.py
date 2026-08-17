"""
============================================================
app/services/session_lifecycle.py — Session Lifecycle Orchestrator (Phase 3)
============================================================
PURPOSE:
    The authoritative source for all session state management logic.
    Wraps sensory_service.py (which handles the session message LIST)
    and owns the session metadata HASH key.

REDIS KEY OWNERSHIP:
    This module is responsible for:
        sensory:{user_id}:meta  → HASH  (this module creates/updates/deletes)

    sensory_service.py remains responsible for:
        sensory:{user_id}:session → LIST (message buffer, unchanged from Phase 1)

    Both keys share the same 30-minute sliding TTL. When the session
    LIST expires due to inactivity, the meta HASH expires simultaneously
    (they're set to the same TTL on every write). This means:
        - meta key MISSING → session has expired → new session on next message
        - meta key PRESENT → session is active → continue appending

SESSION BOUNDARY DETECTION (Lazy / On-Message):
    We don't use Redis keyspace notifications (requires server config).
    Instead, each new message checks if the meta key exists:
        - EXISTS  → session is active → append normally
        - MISSING → session expired (idle > 30 min) → create new meta, new session

    This "lazy expiry" approach works on all Redis tiers (including
    Upstash free tier) with zero infrastructure changes.

CONNECTED TO:
    Phase 3 → app/services/sensory_service.py (flush_session must delete meta)
    Phase 3 → app/api/v1/chat.py (check_session_boundary on every message)
    Phase 3 → app/api/v1/session.py (get_session_state, close_session)
    Phase 5 → synthesis stub receives SessionMetadata payload from close_session
============================================================
"""
import time
from typing import Optional

import redis.asyncio as aioredis
import structlog

from app.core.config import get_settings
from app.schemas.chat import SessionMessage
from app.schemas.session import SessionMetadata
from app.services import mood_tracker

log = structlog.get_logger(__name__)
settings = get_settings()


# -------------------------------------------------------
# Redis Key Helpers
# -------------------------------------------------------

def _meta_key(user_id: str) -> str:
    """Redis HASH key for session metadata."""
    return f"sensory:{user_id}:meta"


def _session_key(user_id: str) -> str:
    """Redis LIST key for session messages (owned by sensory_service)."""
    return f"sensory:{user_id}:session"


# -------------------------------------------------------
# Session Boundary Detection
# -------------------------------------------------------

async def check_session_boundary(redis: aioredis.Redis, user_id: str) -> bool:
    """
    Checks whether the current message is the start of a new session.

    A new session is indicated by the absence of the meta HASH key.
    This happens when:
        1. The user has never sent a message before (first ever message)
        2. The session TTL expired (30 minutes of inactivity)
        3. A session was explicitly closed via POST /session/end

    This is a lazy check — we don't poll; we only check on incoming messages.

    Args:
        redis  : Async Redis client
        user_id: User whose session to inspect

    Returns:
        True  → This is a new session (no meta key found)
        False → Active session is continuing (meta key present)
    """
    key = _meta_key(user_id)
    exists = await redis.exists(key)
    is_new = not bool(exists)
    if is_new:
        log.info("New session boundary detected", user_id=user_id)
    return is_new


# -------------------------------------------------------
# Session Metadata CRUD
# -------------------------------------------------------

async def get_or_create_session_meta(
    redis: aioredis.Redis,
    user_id: str,
) -> SessionMetadata:
    """
    Reads session metadata from Redis, or creates a new meta HASH
    if one doesn't exist (i.e., this is the first message of a new session).

    Called at the start of each chat turn (after check_session_boundary).
    If this is a new session, the meta HASH is created with the current
    timestamp and zeroed counters.

    Args:
        redis  : Async Redis client
        user_id: Target user

    Returns:
        SessionMetadata populated from the Redis HASH, or a fresh
        SessionMetadata with now() as start_time.
    """
    key = _meta_key(user_id)
    now = int(time.time())

    raw = await redis.hgetall(key)

    if not raw:
        # New session — initialize meta HASH
        meta = SessionMetadata(
            start_time=now,
            last_active=now,
            message_count=0,
            mood_sum=0.0,
            mood_count=0,
            mood_drop_flag=False,
        )
        await _write_meta(redis, user_id, meta)
        log.info("Session metadata initialized", user_id=user_id, start_time=now)
        return meta

    # Parse existing meta from Redis HASH (all values are strings)
    return SessionMetadata(
        start_time=int(raw.get("start_time", now)),
        last_active=int(raw.get("last_active", now)),
        message_count=int(raw.get("message_count", 0)),
        mood_sum=float(raw.get("mood_sum", 0.0)),
        mood_count=int(raw.get("mood_count", 0)),
        mood_drop_flag=raw.get("mood_drop_flag", "false").lower() == "true",
    )


async def update_session_meta(
    redis: aioredis.Redis,
    user_id: str,
    user_message_text: Optional[str] = None,
) -> SessionMetadata:
    """
    Updates session metadata after a new message pair (user + assistant turn).

    Operations performed atomically in a single pipeline:
        - Increment message_count by 2 (user + assistant)
        - Update last_active to now()
        - If user_message_text is provided: score valence, update mood accumulators
        - Reset both keys' TTL (sliding window)

    The mood drop check is performed after updating:
        - Reads first_score from meta (stored as "first_score" field)
        - Compares to current running average
        - Sets mood_drop_flag if threshold exceeded

    Args:
        redis             : Async Redis client
        user_id           : Target user
        user_message_text : The user's raw message text (for valence scoring)

    Returns:
        Updated SessionMetadata after all writes.
    """
    key = _meta_key(user_id)
    now = int(time.time())

    # Read current state first
    meta = await get_or_create_session_meta(redis, user_id)

    # Compute new values
    new_message_count = meta.message_count + 2  # user + assistant turn
    new_mood_sum = meta.mood_sum
    new_mood_count = meta.mood_count
    new_mood_drop_flag = meta.mood_drop_flag
    first_score_raw: Optional[float] = None

    if user_message_text:
        valence = mood_tracker.score_message(user_message_text)
        new_mood_sum += valence
        new_mood_count += 1

        # Track the first message's score for mood drop baseline
        first_score_str = await redis.hget(key, "first_score")
        if first_score_str is None:
            # This is the first scored message — store it as baseline
            first_score_raw = valence
        else:
            first_score_raw = float(first_score_str)

        # Check if mood drop threshold is now exceeded
        if not new_mood_drop_flag:
            new_mood_drop_flag = mood_tracker.detect_mood_drop(
                mood_sum=new_mood_sum,
                mood_count=new_mood_count,
                first_score=first_score_raw,
            )
            if new_mood_drop_flag:
                log.warning(
                    "Mood drop alert triggered",
                    user_id=user_id,
                    first_score=first_score_raw,
                    current_avg=round(new_mood_sum / new_mood_count, 3),
                )

    # Build updated meta
    updated_meta = SessionMetadata(
        start_time=meta.start_time,
        last_active=now,
        message_count=new_message_count,
        mood_sum=new_mood_sum,
        mood_count=new_mood_count,
        mood_drop_flag=new_mood_drop_flag,
    )

    # Write atomically in a pipeline
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hmset(key, {
            "start_time": updated_meta.start_time,
            "last_active": updated_meta.last_active,
            "message_count": updated_meta.message_count,
            "mood_sum": updated_meta.mood_sum,
            "mood_count": updated_meta.mood_count,
            "mood_drop_flag": str(updated_meta.mood_drop_flag).lower(),
        })
        # Store first_score baseline if this is the first user message
        if first_score_raw is not None and new_mood_count == 1:
            pipe.hset(key, "first_score", first_score_raw)
        # Reset TTL on both meta and session list keys (sliding window)
        pipe.expire(key, settings.REDIS_SESSION_TTL)
        pipe.expire(_session_key(user_id), settings.REDIS_SESSION_TTL)
        await pipe.execute()

    log.debug(
        "Session meta updated",
        user_id=user_id,
        message_count=new_message_count,
        mood_drop=new_mood_drop_flag,
    )
    return updated_meta


async def _write_meta(
    redis: aioredis.Redis,
    user_id: str,
    meta: SessionMetadata,
) -> None:
    """
    Writes a SessionMetadata object to the Redis HASH and sets TTL.
    Internal helper — callers should use get_or_create_session_meta.
    """
    key = _meta_key(user_id)
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hmset(key, {
            "start_time": meta.start_time,
            "last_active": meta.last_active,
            "message_count": meta.message_count,
            "mood_sum": meta.mood_sum,
            "mood_count": meta.mood_count,
            "mood_drop_flag": str(meta.mood_drop_flag).lower(),
        })
        pipe.expire(key, settings.REDIS_SESSION_TTL)
        await pipe.execute()


async def delete_session_meta(redis: aioredis.Redis, user_id: str) -> None:
    """
    Deletes the session metadata HASH key.

    Called by:
    1. sensory_service.flush_session() — ensures meta is cleaned up with the LIST
    2. close_session() — cleans up after explicit session end
    """
    key = _meta_key(user_id)
    await redis.delete(key)
    log.debug("Session meta deleted", user_id=user_id)


# -------------------------------------------------------
# Session State Query
# -------------------------------------------------------

async def get_session_state(
    redis: aioredis.Redis,
    user_id: str,
) -> dict:
    """
    Returns the current session state as a dict, ready for
    SessionStateResponse construction in the API layer.

    If no session exists, returns default values with is_new_session=True.

    Args:
        redis  : Async Redis client
        user_id: Target user

    Returns:
        Dict with keys: is_new_session, ttl_seconds, message_count,
        mood_delta, mood_drop_alert, start_time
    """
    key = _meta_key(user_id)

    # Get TTL and existence atomically
    raw, ttl = await redis.hgetall(key), await redis.ttl(key)

    if not raw:
        return {
            "is_new_session": True,
            "ttl_seconds": None,
            "message_count": 0,
            "mood_delta": None,
            "mood_drop_alert": False,
            "start_time": None,
        }

    meta = SessionMetadata(
        start_time=int(raw.get("start_time", 0)),
        last_active=int(raw.get("last_active", 0)),
        message_count=int(raw.get("message_count", 0)),
        mood_sum=float(raw.get("mood_sum", 0.0)),
        mood_count=int(raw.get("mood_count", 0)),
        mood_drop_flag=raw.get("mood_drop_flag", "false").lower() == "true",
    )

    return {
        "is_new_session": False,
        "ttl_seconds": max(0, int(ttl)) if ttl and ttl > 0 else 0,
        "message_count": meta.message_count,
        "mood_delta": meta.mood_delta,
        "mood_drop_alert": meta.mood_drop_flag,
        "start_time": meta.start_time,
    }


# -------------------------------------------------------
# Session Close
# -------------------------------------------------------

async def close_session(
    redis: aioredis.Redis,
    user_id: str,
    reason: str = "explicit",
) -> dict:
    """
    Performs a clean session close:
    1. Reads current metadata (for mood_drop_flag, message_count)
    2. Flushes the session message LIST (via sensory_service)
    3. Deletes the metadata HASH
    4. Returns a close summary dict for the API layer

    The caller (POST /session/end) is responsible for firing the
    synthesis background task using the returned `messages` list.

    Args:
        redis  : Async Redis client
        user_id: Target user
        reason : Why the session is ending (for logging/synthesis payload)

    Returns:
        Dict with keys: messages (list), message_count, mood_drop_flag, reason
    """
    from app.services import sensory_service  # Avoid circular import

    # Read metadata before flushing
    raw = await redis.hgetall(_meta_key(user_id))
    mood_drop_flag = raw.get("mood_drop_flag", "false").lower() == "true" if raw else False
    message_count = int(raw.get("message_count", 0)) if raw else 0

    # Flush message LIST and delete meta HASH
    messages = await sensory_service.flush_session(redis, user_id)
    # flush_session deletes the session LIST and meta HASH
    # (sensory_service.flush_session was updated in Phase 3 to call delete_session_meta)

    log.info(
        "Session closed",
        user_id=user_id,
        reason=reason,
        message_count=message_count,
        mood_drop_flag=mood_drop_flag,
        messages_flushed=len(messages),
    )

    return {
        "messages": messages,
        "message_count": message_count,
        "mood_drop_flag": mood_drop_flag,
        "reason": reason,
    }
