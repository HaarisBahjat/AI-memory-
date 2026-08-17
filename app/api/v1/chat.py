"""
============================================================
app/api/v1/chat.py — POST /api/v1/chat (Core Chat Endpoint)
============================================================
PURPOSE:
    The main API endpoint that processes all user wellness messages.

    1. Authenticate the caller (JWT Bearer token via get_current_user)
    2. Run clinical safety screener (HARD OVERRIDE — no LLM if crisis)
    3. Check session boundary (Phase 3: detect 30-min inactivity reset)
    4. Execute Hybrid RAG pipeline (retrieval_engine.py)
    5. Persist both message turns to the Redis session buffer (Phase 3)
    6. Update session metadata + mood scoring (Phase 3)
    7. Return structured ChatResponse

PHASE 2 CHANGE:
    user_id is now sourced exclusively from the validated JWT
    (current_user.user_id), NOT from the request body.

PHASE 3 CHANGE:
    - Session boundary check added (lazy expiry detection)
    - append_message() now called for BOTH user turn AND assistant turn
      (Phase 1 only retrieved from Redis; Phase 3 also writes back)
    - session_lifecycle.update_session_meta() updates mood + counters
    - ChatResponse now includes is_new_session flag

CONNECTED TO:
    Phase 1 → Orchestrates all Phase 1 services
    Phase 2 → get_current_user guards the endpoint
    Phase 3 → session_lifecycle + sensory_service manage session state
    Phase 5 → Episode insertion added here after LLM response
    Phase 6 → Safety screener extended with semantic layer
    Phase 9 → Debug context added to response for benchmarking
============================================================
"""
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.database import get_db
from app.core.redis_client import get_redis
from app.core.safety_triage import evaluate_clinical_safety
from app.schemas.auth import CurrentUser
from app.schemas.chat import ChatRequest, ChatResponse, CrisisTriageResponse
from app.services import retrieval_engine, sensory_service, session_lifecycle

log = structlog.get_logger(__name__)
settings = get_settings()
router = APIRouter()


@router.post(
    "/chat",
    summary="Send a wellness message and receive AI-powered response",
    description=(
        "Processes user wellness messages through the 3-layer Hybrid RAG pipeline. "
        "Requires a valid Bearer token in the Authorization header. "
        "The user identity is extracted from the JWT — not from the request body. "
        "Layer 1 (Redis session) + Layer 2 (Supabase JSONB episodes) + "
        "Layer 3 (pgvector semantic memories) are fetched in parallel. "
        "Time-decay scoring filters stale memories before LLM context injection. "
        "Clinical safety screener intercepts crisis signals BEFORE any LLM call. "
        "On crisis detection, returns a CrisisTriageResponse instead of ChatResponse. "
        "Phase 3: Session boundary detection, full turn persistence, and mood scoring added."
    ),
    responses={
        200: {
            "description": (
                "Successful wellness response (ChatResponse) or crisis triage "
                "(CrisisTriageResponse). `is_new_session=true` indicates the "
                "user's previous session expired and a new one was started."
            ),
        },
        401: {"description": "Missing or invalid authentication token"},
        422: {"description": "Invalid request body"},
        500: {"description": "Internal server error"},
    },
)
async def chat(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
):
    """
    Main wellness chat endpoint (Phase 3: session lifecycle integrated).

    The user_id used for all memory lookups is sourced exclusively from
    the validated JWT. Session state is managed transparently — the client
    only needs to check `is_new_session` in the response to know when a
    context reset occurred.

    Request body:
        message : User's wellness message (max 2000 characters)

    Returns:
        ChatResponse with LLM response + session metadata, or
        CrisisTriageResponse if safety screener fires.
    """
    user_id = current_user.user_id

    log.info(
        "Chat request received",
        user_id=user_id,
        message_length=len(request.message),
    )

    # ── STEP 1: CLINICAL SAFETY HARD OVERRIDE ──────────────────────────
    # Must happen before any DB queries or LLM calls.
    # Crisis detection returns immediately — no session state update,
    # no RAG, no LLM. The crisis triage response is not persisted to
    # the session buffer (it would pollute the conversation context).
    if settings.ENABLE_SAFETY_SCREENER:
        safety_result = evaluate_clinical_safety(
            message=request.message,
            user_id=user_id,
        )
        if not safety_result.is_safe:
            log.warning(
                "Crisis triage response sent",
                user_id=user_id,
                crisis_type=safety_result.triage_response.crisis_type,
            )
            return safety_result.triage_response.to_client_response()

    # ── STEP 2: SESSION BOUNDARY CHECK (Phase 3) ───────────────────────
    # Lazy inactivity detection: if the meta HASH key is absent, the
    # session TTL expired → this is a new session.
    # This does NOT block the request — it's a single Redis EXISTS call.
    try:
        is_new_session = await session_lifecycle.check_session_boundary(
            redis=redis,
            user_id=user_id,
        )
        if is_new_session:
            # Initialize fresh session metadata
            await session_lifecycle.get_or_create_session_meta(
                redis=redis,
                user_id=user_id,
            )
    except Exception as e:
        # Session lifecycle errors are non-fatal — log and continue
        log.warning("Session boundary check failed", user_id=user_id, error=str(e))
        is_new_session = False

    # ── STEP 3: HYBRID RAG PIPELINE ────────────────────────────────────
    try:
        result = await retrieval_engine.run_hybrid_rag_pipeline(
            user_id=user_id,
            message=request.message,
            db=db,
            redis=redis,
        )
    except Exception as e:
        log.error("RAG pipeline failed", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred processing your message. Please try again.",
        )

    # ── STEP 4: PERSIST SESSION TURNS (Phase 3) ────────────────────────
    # Append BOTH the user message and assistant response to the Redis
    # session buffer. Phase 1 only READ from Redis; Phase 3 also WRITES.
    # Both turns are needed in the session buffer so that:
    #   a) The RAG pipeline sees the full conversation context next turn
    #   b) Phase 5 can synthesize a complete dialogue episode
    try:
        await sensory_service.append_message(
            redis=redis,
            user_id=user_id,
            role="user",
            content=request.message,
        )
        await sensory_service.append_message(
            redis=redis,
            user_id=user_id,
            role="assistant",
            content=result["response"],
        )
    except Exception as e:
        # Session write errors are non-fatal — the response is still valid
        log.warning("Failed to append messages to session", user_id=user_id, error=str(e))

    # ── STEP 5: UPDATE SESSION METADATA + MOOD (Phase 3) ───────────────
    # update_session_meta scores the user's message valence,
    # increments the message counter, and refreshes the TTL.
    try:
        await session_lifecycle.update_session_meta(
            redis=redis,
            user_id=user_id,
            user_message_text=request.message,
        )
    except Exception as e:
        log.warning("Failed to update session metadata", user_id=user_id, error=str(e))

    # ── STEP 6: BUILD RESPONSE ─────────────────────────────────────────
    return ChatResponse(
        user_id=user_id,
        response=result["response"],
        session_id=f"sensory:{user_id}:session",
        memories_used=result["memories_used"],
        is_new_session=is_new_session,
        debug=result.get("debug") if settings.DEBUG else None,
    )
