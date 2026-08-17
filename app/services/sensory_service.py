"""
============================================================
app/services/sensory_service.py — Layer 1: Redis Session Manager
============================================================
PURPOSE:
    Manages the active conversational buffer in Redis.
    This is Layer 1 (Sensory Memory) of the 3-tier LMS.

HOW IT WORKS:
    Every time a user sends a message, it gets appended to their
    session list in Redis. The last N messages (default 10) are
    kept as a rolling window. A 30-minute TTL is reset on every
    interaction, so the session stays alive during active chat
    but auto-expires after inactivity.

    Key pattern : sensory:{user_id}:session
    Value type  : Redis List of JSON-serialized SessionMessage objects
    TTL         : 30 minutes (REDIS_SESSION_TTL from config)
    Max size    : 10 messages (REDIS_MAX_MESSAGES from config)

HOW IT CONNECTS TO THE HYBRID RAG:
    When a user sends a message, retrieval_engine.py calls
    get_active_session() to get the last N messages. These form
    the "RECENT CONVERSATION" section of the LLM system prompt,
    giving the model immediate short-term conversational context.

CONNECTED TO:
    Phase 1  → get_active_session() used in retrieval_engine.py
    Phase 5  → append_message() called after every chat turn
    Phase 7  → flush_session() triggers consolidation when TTL expires
    Phase 8  → flush_session() called during Right-to-Forget deletion
============================================================
"""
import json
import time
from typing import Optional
import redis.asyncio as aioredis
import structlog

from app.core.config import get_settings
from app.schemas.chat import SessionMessage

log = structlog.get_logger(__name__)
settings = get_settings()


def _session_key(user_id: str) -> str:
    """Generates the Redis key for a user's sensory session buffer."""
    return f"sensory:{user_id}:session"


async def get_active_session(
    redis: aioredis.Redis,
    user_id: str,
    limit: Optional[int] = None,
) -> list[SessionMessage]:
    """
    Retrieves the last N messages from the user's active session buffer.

    Returns an empty list if no active session exists (e.g., first message
    of the day, or session expired after 30 minutes of inactivity).

    Args:
        redis   : The shared async Redis client
        user_id : User whose session to retrieve
        limit   : Max messages to return (defaults to REDIS_MAX_MESSAGES)

    Returns:
        List of SessionMessage objects (oldest → newest order)
    """
    max_msgs = limit or settings.REDIS_MAX_MESSAGES
    key = _session_key(user_id)

    try:
        # LRANGE fetches from tail (most recent messages)
        # Index -max_msgs to -1 gives us the last N messages
        raw_messages = await redis.lrange(key, -max_msgs, -1)
        if not raw_messages:
            return []

        messages = []
        for raw in raw_messages:
            try:
                data = json.loads(raw)
                messages.append(SessionMessage(**data))
            except (json.JSONDecodeError, ValueError) as e:
                log.warning("Malformed session message skipped", error=str(e))
                continue

        log.debug("Session retrieved", user_id=user_id, count=len(messages))
        return messages

    except Exception as e:
        log.error("Failed to retrieve session", user_id=user_id, error=str(e))
        return []


async def append_message(
    redis: aioredis.Redis,
    user_id: str,
    role: str,
    content: str,
) -> None:
    """
    Appends a new message to the user's session buffer and resets TTL.

    This is called TWICE per chat turn:
    1. When the user sends their message (role="user")
    2. After the LLM response is generated (role="assistant")

    The rolling window enforcement works as follows:
    - RPUSH appends the new message to the right end of the list
    - LTRIM keeps only the last REDIS_MAX_MESSAGES items
    - EXPIRE resets the 30-minute sliding TTL

    Args:
        redis   : Shared async Redis client
        user_id : Target user's session
        role    : "user" or "assistant"
        content : Message text content
    """
    key = _session_key(user_id)
    message = SessionMessage(
        role=role,
        content=content,
        timestamp=int(time.time()),
    )
    serialized = json.dumps(message.model_dump())

    try:
        # Pipeline for atomic multi-command execution
        async with redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, serialized)                   # Append to list
            pipe.ltrim(key, -settings.REDIS_MAX_MESSAGES, -1)  # Rolling window
            pipe.expire(key, settings.REDIS_SESSION_TTL)  # Reset 30-min TTL
            await pipe.execute()

        log.debug("Message appended to session", user_id=user_id, role=role)

    except Exception as e:
        log.error("Failed to append session message", user_id=user_id, error=str(e))
        raise


async def flush_session(
    redis: aioredis.Redis,
    user_id: str,
) -> list[SessionMessage]:
    """
    Atomically retrieves and clears the user's session buffer AND metadata.

    Called in two scenarios:
    1. Phase 3 session_lifecycle.close_session() — explicit/timeout session end
       (the message history is passed to the synthesis background task)
    2. Phase 8 Right-to-Forget (immediately wipes all session data)

    Phase 3 Change:
        Now also deletes `sensory:{user_id}:meta` alongside the session LIST
        to ensure no orphan metadata is left behind after a flush.

    Returns:
        All messages from the session before it was flushed.
        Empty list if no session existed.
    """
    session_key = _session_key(user_id)
    meta_key = f"sensory:{user_id}:meta"

    try:
        # Get all messages first
        messages = await get_active_session(redis, user_id, limit=1000)

        # Delete session LIST and metadata HASH atomically
        await redis.delete(session_key, meta_key)
        log.info(
            "Session flushed",
            user_id=user_id,
            message_count=len(messages),
            keys_deleted=[session_key, meta_key],
        )
        return messages

    except Exception as e:
        log.error("Failed to flush session", user_id=user_id, error=str(e))
        raise


async def session_exists(redis: aioredis.Redis, user_id: str) -> bool:
    """
    Checks if an active session exists for the user.
    Used by Phase 5 to determine whether to greet returning
    user vs. continuing an active conversation.
    """
    key = _session_key(user_id)
    return bool(await redis.exists(key))
