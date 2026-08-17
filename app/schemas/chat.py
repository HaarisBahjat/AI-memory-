"""
============================================================
app/schemas/chat.py — Pydantic Request/Response Schemas
============================================================
PURPOSE:
    Defines strictly typed data contracts for the Chat API.
    Pydantic v2 validates all incoming requests and outgoing
    responses at the FastAPI boundary, preventing bad data
    from ever reaching the service layer.

CONNECTED TO:
    Phase 1  → POST /api/v1/chat uses ChatRequest/ChatResponse
    Phase 5  → SessionMessage used in Layer 1 sensory memory
    Phase 6  → CrisisTriageResponse returned on safety override
    Phase 9  → DebugContext added to response for benchmarking
============================================================
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


# -------------------------------------------------------
# Incoming Chat Request
# -------------------------------------------------------

class ChatRequest(BaseModel):
    """
    Request body for POST /api/v1/chat

    Phase 2 Change:
        `user_id` has been REMOVED from the request body.
        The user identity is now extracted exclusively from the
        validated JWT Bearer token (via the get_current_user dependency),
        eliminating the ability for callers to impersonate other users.

    Fields:
        message : The raw user message to process (1–2000 chars)
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's wellness message",
        examples=["I've been feeling really anxious before my exams lately."],
    )



# -------------------------------------------------------
# Session Message (Layer 1 Redis Buffer)
# -------------------------------------------------------

class SessionMessage(BaseModel):
    """
    Represents a single message in the Layer 1 sensory buffer.
    Stored as JSON in Redis: sensory:{user_id}:session
    """
    role: Literal["user", "assistant"] = Field(
        ..., description="Message sender role"
    )
    content: str = Field(..., description="Message text content")
    timestamp: int = Field(
        ..., description="Unix timestamp when message was created"
    )


# -------------------------------------------------------
# Outgoing Chat Response
# -------------------------------------------------------

class ChatResponse(BaseModel):
    """
    Successful chat response returned from POST /api/v1/chat

    Fields:
        user_id       : Echoes the request user_id for client routing
        response      : The LLM-generated wellness response text
        session_id    : Redis session key identifier
        memories_used : Count of Layer 3 memories injected into context
        is_new_session: True if this message started a fresh session (Phase 3)
        debug         : Optional debug context (Phase 9 benchmarking only)
    """
    user_id: str
    response: str
    session_id: str
    memories_used: int = 0
    is_new_session: bool = False  # Phase 3: indicates a 30-min inactivity boundary was crossed
    debug: Optional[dict] = None  # Phase 9: token counts, decay scores, etc.


# -------------------------------------------------------
# Crisis Triage Response (Safety Override)
# -------------------------------------------------------

class CrisisResource(BaseModel):
    """A single crisis helpline resource."""
    name: str
    contact: str
    type: str  # "phone" or "emergency"


class CrisisTriageResponse(BaseModel):
    """
    Returned instead of ChatResponse when clinical safety
    screener detects a crisis signal. The LLM is NOT called.
    """
    type: Literal["CRISIS_TRIAGE"] = "CRISIS_TRIAGE"
    message: str
    resources: list[CrisisResource]
    follow_up: str


# -------------------------------------------------------
# Health Check Response
# -------------------------------------------------------

class HealthResponse(BaseModel):
    """Response for GET /api/v1/health endpoint."""
    status: str
    version: str
    database: bool
    redis: bool
    timestamp: datetime
