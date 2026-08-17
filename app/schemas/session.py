"""
============================================================
app/schemas/session.py — Pydantic Schemas for Session API (Phase 3)
============================================================
PURPOSE:
    Defines all request and response shapes for the session
    lifecycle endpoints:
        GET  /api/v1/session/active  → SessionStateResponse
        POST /api/v1/session/end     → SessionEndResponse

    Also defines the internal SessionMetadata model that is
    read from the Redis HASH key `sensory:{user_id}:meta`.

CONNECTED TO:
    Phase 3 → app/services/session_lifecycle.py (reads/writes meta HASH)
    Phase 3 → app/api/v1/session.py (returns these types to clients)
    Phase 5 → synthesis_stub receives SessionMetadata on session close
============================================================
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# -------------------------------------------------------
# Internal: Session Metadata (stored as Redis HASH)
# -------------------------------------------------------

class SessionMetadata(BaseModel):
    """
    In-memory representation of the `sensory:{user_id}:meta` Redis HASH.

    Written when a new session starts (first message after an idle period).
    Updated on every message (message_count incremented, last_active refreshed,
    mood accumulators updated).
    Deleted atomically with the session LIST when the session ends.

    Fields:
        start_time    : Unix timestamp when this session started
        last_active   : Unix timestamp of the most recent message
        message_count : Total messages appended this session (user + assistant)
        mood_sum      : Running sum of valence scores (-1.0 to +1.0 per message)
        mood_count    : Number of scored messages (user turns only)
        mood_drop_flag: True if cumulative valence dropped > 0.4 in this session
    """
    start_time: int = Field(description="Unix timestamp of session start")
    last_active: int = Field(description="Unix timestamp of last message")
    message_count: int = Field(default=0, ge=0)
    mood_sum: float = Field(default=0.0, description="Running sum of per-message valence scores")
    mood_count: int = Field(default=0, ge=0, description="Number of user messages scored")
    mood_drop_flag: bool = Field(
        default=False,
        description="True if mood dropped > 0.4 points since session start",
    )

    @property
    def mood_delta(self) -> Optional[float]:
        """
        Returns the average valence score for this session,
        or None if no user messages have been scored yet.

        Range: -1.0 (extremely negative) to +1.0 (extremely positive)
        """
        if self.mood_count == 0:
            return None
        return round(self.mood_sum / self.mood_count, 3)


# -------------------------------------------------------
# GET /api/v1/session/active Response
# -------------------------------------------------------

class SessionStateResponse(BaseModel):
    """
    Response for GET /api/v1/session/active.

    Gives the client a real-time snapshot of the user's current session.
    Useful for:
    - Displaying an "Active session" indicator in the UI
    - Showing turn count ("3 of 10 messages in this context window")
    - Warning "Session expires in 5 minutes" via ttl_seconds
    - Detecting session resets (is_new_session=True) to clear UI chat log
    """
    is_new_session: bool = Field(
        description="True if no active session exists (expired or never started)."
    )
    ttl_seconds: Optional[int] = Field(
        default=None,
        description="Seconds until the session expires due to inactivity. None if no session.",
    )
    message_count: int = Field(
        default=0,
        description="Total messages (user + assistant) in the current session buffer.",
    )
    mood_delta: Optional[float] = Field(
        default=None,
        description=(
            "Average valence score across user messages this session. "
            "Range: -1.0 (very negative) to +1.0 (very positive). "
            "None if no messages have been scored yet."
        ),
    )
    mood_drop_alert: bool = Field(
        default=False,
        description="True if mood dropped significantly (>0.4) within this session.",
    )
    start_time: Optional[int] = Field(
        default=None,
        description="Unix timestamp when this session started. None if no active session.",
    )


# -------------------------------------------------------
# POST /api/v1/session/end Request + Response
# -------------------------------------------------------

class SessionEndRequest(BaseModel):
    """
    Optional request body for POST /api/v1/session/end.

    The `reason` field is used for analytics and audit logging.
    All values trigger the same flush + synthesis stub behavior.
    """
    reason: Literal["user_logout", "timeout", "explicit", "app_background"] = Field(
        default="explicit",
        description="Why the session is being ended.",
    )


class SessionEndResponse(BaseModel):
    """
    Response for POST /api/v1/session/end.

    Confirms the session was flushed and informs the client
    whether the synthesis background task was dispatched.

    Fields:
        message_count       : How many messages were in the buffer at close time
        synthesis_triggered : Whether the background synthesis task was fired
        mood_drop_alert     : Whether mood drop was flagged (for client awareness)
        reason              : Echo of the close reason from the request
    """
    message_count: int
    synthesis_triggered: bool
    mood_drop_alert: bool
    reason: str
    message: str = "Session closed successfully."
