"""
============================================================
app/api/v1/session.py — Session Lifecycle Endpoints (Phase 3)
============================================================
PURPOSE:
    Two JWT-protected endpoints for session state inspection
    and explicit session termination:

    GET  /api/v1/session/active
        Returns the user's current session state: TTL, message count,
        mood delta, and whether the session is new. Useful for front-end
        clients to display context-window usage indicators or session timers.

    POST /api/v1/session/end
        Explicitly closes the user's current session. Flushes the Redis
        message buffer and metadata, and fires a background task stub
        (placeholder for Phase 5 Episode Synthesis). Safe to call even
        when no session is active (idempotent).

SECURITY:
    Both endpoints require a valid Bearer token. The user_id is
    extracted exclusively from the JWT — callers cannot close
    another user's session.

CONNECTED TO:
    Phase 3 → app/services/session_lifecycle.py (all session logic)
    Phase 3 → app/api/deps.py (get_current_user dependency)
    Phase 5 → synthesis_stub() background task will be replaced by
              the real episode synthesizer
    Phase 7 → Celery will also call close_session() on nightly TTL sweep
============================================================
"""
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends
from redis.asyncio import Redis

from app.api.deps import get_current_user
from app.core.redis_client import get_redis
from app.schemas.auth import CurrentUser
from app.schemas.chat import SessionMessage
from app.schemas.session import (
    SessionEndRequest,
    SessionEndResponse,
    SessionStateResponse,
)
from app.services import session_lifecycle

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/session", tags=["Session Lifecycle"])


# -------------------------------------------------------
# Phase 5 Synthesis Stub
# -------------------------------------------------------
# This function will be replaced by the real episode synthesizer
# in Phase 5. For now, it logs the session summary and returns.
# It runs as a FastAPI BackgroundTask — non-blocking.
# -------------------------------------------------------

async def _synthesis_stub(
    user_id: str,
    messages: list[SessionMessage],
    mood_drop_flag: bool,
    reason: str,
) -> None:
    """
    Phase 5 placeholder: logs the session summary that would be
    sent to the LLM episode synthesizer.

    Phase 5 will replace this with:
        - episode_synthesizer.synthesize(user_id, messages, mood_drop_flag)
        - which calls GPT-4o-mini with JSON Schema mode to extract metrics
        - and writes a structured row to the `episodes` table

    Args:
        user_id       : Owner of the session
        messages      : All flushed messages from the session buffer
        mood_drop_flag: Whether a significant mood drop was detected
        reason        : Why the session ended (for audit metadata)
    """
    log.info(
        "Phase 5 synthesis stub invoked",
        user_id=user_id,
        message_count=len(messages),
        mood_drop_flag=mood_drop_flag,
        close_reason=reason,
        stub_note="Replace this function with episode_synthesizer.synthesize() in Phase 5",
    )
    # Phase 5: await episode_synthesizer.synthesize(user_id, messages, mood_drop_flag)


# -------------------------------------------------------
# GET /api/v1/session/active
# -------------------------------------------------------

@router.get(
    "/active",
    response_model=SessionStateResponse,
    summary="Get current session state",
    description=(
        "Returns a snapshot of the authenticated user's active session: "
        "remaining TTL, message count, mood delta, and whether a new session "
        "boundary was detected. Returns `is_new_session=true` if no active "
        "session exists (expired or never started)."
    ),
    responses={
        200: {"description": "Session state returned successfully"},
        401: {"description": "Missing or invalid authentication token"},
    },
)
async def get_active_session(
    current_user: CurrentUser = Depends(get_current_user),
    redis: Redis = Depends(get_redis),
) -> SessionStateResponse:
    """
    Session state inspection endpoint.

    Reads the `sensory:{user_id}:meta` HASH from Redis and returns
    all session metadata. Does NOT modify any state.

    Use cases:
    - Front-end "You have 7 of 10 messages in your context window" indicator
    - "Session expires in 4 minutes" countdown timer
    - Detecting session resets so the UI can clear the chat log
    - Monitoring whether a mood drop alert is active
    """
    state = await session_lifecycle.get_session_state(
        redis=redis,
        user_id=current_user.user_id,
    )

    log.debug(
        "Session state queried",
        user_id=current_user.user_id,
        is_new_session=state["is_new_session"],
        message_count=state["message_count"],
    )

    return SessionStateResponse(**state)


# -------------------------------------------------------
# POST /api/v1/session/end
# -------------------------------------------------------

@router.post(
    "/end",
    response_model=SessionEndResponse,
    summary="Explicitly end the current session",
    description=(
        "Closes the user's active session: flushes the Redis message buffer "
        "and metadata HASH, then fires a background task stub (Phase 5 Episode "
        "Synthesis placeholder). Safe to call when no session is active — "
        "returns message_count=0 and synthesis_triggered=False."
    ),
    responses={
        200: {"description": "Session closed successfully"},
        401: {"description": "Missing or invalid authentication token"},
    },
)
async def end_session(
    background_tasks: BackgroundTasks,
    request: SessionEndRequest = SessionEndRequest(),
    current_user: CurrentUser = Depends(get_current_user),
    redis: Redis = Depends(get_redis),
) -> SessionEndResponse:
    """
    Explicit session termination endpoint.

    Called when:
    - User clicks "End Chat" in the UI
    - Mobile app goes to background
    - User logs out (frontend calls this before POST /auth/logout)

    If no session is active, this is a no-op with a clean 200 response.
    The synthesis background task is only fired if there were messages
    to synthesize (message_count > 0).

    Token flow:
        close_session() flushes buffer + metadata → returns messages list
        BackgroundTasks.add_task(_synthesis_stub, ...) → non-blocking
        Return SessionEndResponse immediately to the client
    """
    user_id = current_user.user_id

    close_result = await session_lifecycle.close_session(
        redis=redis,
        user_id=user_id,
        reason=request.reason,
    )

    messages = close_result["messages"]
    message_count = close_result["message_count"]
    mood_drop_flag = close_result["mood_drop_flag"]
    synthesis_triggered = False

    # Only fire synthesis if there was something to synthesize
    if messages:
        background_tasks.add_task(
            _synthesis_stub,
            user_id=user_id,
            messages=messages,
            mood_drop_flag=mood_drop_flag,
            reason=request.reason,
        )
        synthesis_triggered = True

    log.info(
        "Session ended via explicit endpoint",
        user_id=user_id,
        message_count=message_count,
        synthesis_triggered=synthesis_triggered,
        mood_drop_alert=mood_drop_flag,
        reason=request.reason,
    )

    return SessionEndResponse(
        message_count=message_count,
        synthesis_triggered=synthesis_triggered,
        mood_drop_alert=mood_drop_flag,
        reason=request.reason,
    )
