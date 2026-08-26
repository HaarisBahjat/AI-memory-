"""
============================================================
app/services/episode_service.py -- Phase 5 Episode Synthesis
============================================================
PURPOSE:
    Produces a structured daily health episode after each chat
    session ends. Three responsibilities:

    1. synthesize_episode()  -- call GPT-4o-mini to produce a
       concise session summary + extract JSON health metrics
    2. persist_episode()     -- embed the summary, insert into DB
    3. run_synthesis()       -- top-level orchestrator called by
       the FastAPI background task; never raises (catches all errors
       so the HTTP response is never affected)

SAFETY DESIGN:
    All steps are wrapped in try/except. If OpenAI is unavailable
    or the DB write fails, the error is logged with structlog and
    the function returns None gracefully. The client response has
    already been sent before this task starts.

CONNECTED TO:
    Phase 3 -> session_lifecycle.close_session() provides messages list
    Phase 5 -> app/api/v1/session.py calls run_synthesis() as BG task
    Phase 5 -> app/api/v1/episodes.py exposes the persisted episodes
    Phase 7 -> Nightly consolidation reads from the episodes table
============================================================
"""
import json
import time
from typing import Optional

import structlog
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.episode import Episode
from app.schemas.chat import SessionMessage
from app.schemas.episode import ExtractedMetrics
from app.services.embedding_service import embed_text

log = structlog.get_logger(__name__)
settings = get_settings()

_openai_client: Optional[AsyncOpenAI] = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
    return _openai_client


# -------------------------------------------------------
# Prompt Template
# -------------------------------------------------------

_SYNTHESIS_SYSTEM_PROMPT = """You are a clinical data-extraction assistant for a mental wellness app.

Given a transcript of a chat session between a user and an AI wellness companion,
produce a JSON object with exactly these fields:
{
  "session_summary": "<2-5 sentence empathetic narrative of the session>",
  "extracted_metrics": {
    "moodScore": <integer 1-10 or null>,
    "physicalSymptoms": ["<symptom>" ...],
    "primaryStressor": "<string or null>",
    "sleepHoursLogged": <float or null>,
    "anxietyLevel": <integer 1-10 or null>,
    "energyLevel": <integer 1-10 or null>,
    "biometrics": {}
  }
}

Rules:
- ONLY output valid JSON. No markdown, no explanations.
- Set numeric fields to null when the user did not mention the metric.
- moodScore: 1=very bad, 10=very good. Infer from emotional tone if not stated.
- session_summary must be written in third-person neutral clinical style.
- physicalSymptoms must only contain symptoms the user explicitly mentioned.
- NEVER fabricate data not present in the transcript.
"""

_FALLBACK_METRICS = ExtractedMetrics(
    moodScore=None,
    physicalSymptoms=[],
    primaryStressor=None,
    sleepHoursLogged=None,
    anxietyLevel=None,
    energyLevel=None,
    biometrics={},
)


# -------------------------------------------------------
# Step 1: LLM Synthesis
# -------------------------------------------------------

async def synthesize_episode(
    messages: list[SessionMessage],
) -> tuple[str, ExtractedMetrics]:
    """
    Calls GPT-4o-mini with the session transcript and extracts:
        - session_summary : str (2-5 sentence narrative)
        - extracted_metrics : ExtractedMetrics (structured health data)

    If the LLM call fails or returns unparseable JSON, returns a
    safe fallback summary and null-filled metrics rather than raising.

    Args:
        messages : Full list of SessionMessage objects from the session buffer

    Returns:
        Tuple (session_summary, ExtractedMetrics)
    """
    if not messages:
        log.warning("synthesize_episode called with empty messages list")
        return "Session ended with no messages.", _FALLBACK_METRICS

    # Build the transcript string (role: content pairs)
    transcript_lines = []
    for msg in messages:
        role_label = "User" if msg.role == "user" else "AI Companion"
        transcript_lines.append(f"{role_label}: {msg.content}")

    transcript_text = "\n".join(transcript_lines)

    # Enforce a hard token limit by truncating transcript at 12000 chars
    # (text-embedding-3-small and gpt-4o-mini context is ~128k tokens)
    if len(transcript_text) > 12000:
        transcript_text = transcript_text[:12000] + "\n[TRANSCRIPT TRUNCATED]"
        log.warning("Episode synthesis transcript truncated", char_limit=12000)

    client = _get_openai_client()

    # Try the LLM call; on any failure return a safe fallback
    for attempt in range(2):  # Try twice before giving up
        try:
            # Determine which model to try on this attempt
            if attempt == 0:
                model = settings.OPENAI_CHAT_MODEL
            else:
                model = settings.OPENAI_CHAT_MODEL_FALLBACK or settings.OPENAI_CHAT_MODEL

            response = await client.chat.completions.create(
                model=model,
                temperature=0.2,  # Low temperature for consistent JSON extraction
                # NOTE: response_format=json_object is NOT supported by Gemini OpenAI compat.
                # Instead, the prompt explicitly instructs the model to output only JSON.
                messages=[
                    {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": f"TRANSCRIPT:\n{transcript_text}"},
                ],
            )
            raw = response.choices[0].message.content or ""
            # Strip accidental markdown code fences Gemini sometimes adds
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]  # drop opening fence line
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

            parsed = json.loads(raw)

            summary = parsed.get("session_summary", "").strip()
            if not summary:
                summary = "The session has been recorded."

            raw_metrics = parsed.get("extracted_metrics", {})
            metrics = ExtractedMetrics.model_validate(raw_metrics)

            log.info(
                "Episode synthesis successful",
                attempt=attempt + 1,
                summary_length=len(summary),
                mood_score=metrics.moodScore,
            )
            return summary, metrics

        except (json.JSONDecodeError, ValueError) as e:
            log.warning(
                "Episode synthesis JSON parse error",
                attempt=attempt + 1,
                model=model,
                error=str(e),
            )
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            log.error(
                "Episode synthesis LLM call failed",
                attempt=attempt + 1,
                model=model,
                rate_limited=is_rate_limit,
                error=str(e)[:200],
            )

    # Both attempts failed: return a minimal safe fallback
    log.error("Episode synthesis failed after 2 attempts; using fallback")
    fallback_summary = (
        f"Session with {len(messages)} messages recorded. "
        "Automated synthesis was unavailable."
    )
    return fallback_summary, _FALLBACK_METRICS


# -------------------------------------------------------
# Step 2: Embed + Persist to DB
# -------------------------------------------------------

async def persist_episode(
    db: AsyncSession,
    user_id: str,
    session_summary: str,
    metrics: ExtractedMetrics,
) -> Episode:
    """
    Embeds the session_summary and inserts a new Episode row.

    Embedding step:
        Calls embedding_service.embed_text(session_summary) to produce
        a 1536-dim vector. If this fails, the episode is still inserted
        with embedding=NULL so the row is never lost.

    DB step:
        All fields are inserted in one transaction. SQLAlchemy will
        auto-rollback on exception (handled by get_db dependency).

    Args:
        db              : Active AsyncSession from the dependency
        user_id         : Owner of the session
        session_summary : LLM-generated narrative summary
        metrics         : ExtractedMetrics Pydantic model

    Returns:
        The newly created Episode ORM instance with its UUID populated.
    """
    # Attempt to embed; gracefully fall back to NULL embedding
    embedding_vector: Optional[list[float]] = None
    try:
        embedding_vector = await embed_text(session_summary)
        log.debug("Episode summary embedded", user_id=user_id, dims=len(embedding_vector))
    except Exception as e:
        log.error(
            "Episode embedding failed; storing with NULL vector",
            user_id=user_id,
            error=str(e),
        )

    episode = Episode(
        user_id=user_id,
        session_summary=session_summary,
        extracted_metrics=metrics.model_dump(),
        embedding=embedding_vector,
    )
    db.add(episode)
    await db.flush()   # Populate the server-generated UUID without committing

    log.info(
        "Episode persisted",
        user_id=user_id,
        episode_id=episode.id,
        has_embedding=embedding_vector is not None,
        mood_score=metrics.moodScore,
    )
    return episode


# -------------------------------------------------------
# Top-Level Orchestrator (called by BG task)
# -------------------------------------------------------

async def run_synthesis(
    user_id: str,
    messages: list[SessionMessage],
    mood_drop_flag: bool,
    reason: str,
) -> None:
    """
    Top-level orchestrator invoked as a FastAPI BackgroundTask.

    Flow:
        1. synthesize_episode()  -- LLM summary + metric extraction
        2. embed + persist in a fresh DB session
        3. Log outcome

    NEVER RAISES. All errors are caught and logged so that the
    HTTP response (already sent) is never affected.

    Args:
        user_id        : Session owner
        messages       : All messages flushed from the Redis buffer
        mood_drop_flag : Whether a significant mood drop was detected
        reason         : Session close reason ("explicit" | "expired" | ...)
    """
    t_start = time.perf_counter()
    log.info(
        "Episode synthesis started",
        user_id=user_id,
        message_count=len(messages),
        mood_drop_flag=mood_drop_flag,
        close_reason=reason,
    )

    if not messages:
        log.info("No messages to synthesize; skipping episode creation", user_id=user_id)
        return

    try:
        # Step 1: LLM call (safe  never raises)
        session_summary, metrics = await synthesize_episode(messages)

        # Step 2: DB write in a self-managed session
        # (BG tasks can't use the request-scoped get_db generator)
        async with AsyncSessionLocal() as db:
            try:
                episode = await persist_episode(
                    db=db,
                    user_id=user_id,
                    session_summary=session_summary,
                    metrics=metrics,
                )
                await db.commit()
                elapsed = time.perf_counter() - t_start
                log.info(
                    "Episode synthesis complete",
                    user_id=user_id,
                    episode_id=episode.id,
                    elapsed_seconds=round(elapsed, 3),
                    mood_drop_flag=mood_drop_flag,
                )
            except Exception as e:
                await db.rollback()
                log.error(
                    "Episode DB write failed; rolled back",
                    user_id=user_id,
                    error=str(e),
                )
    except Exception as e:
        # Absolute safety net  this function must never propagate exceptions
        log.error(
            "Unexpected error in episode synthesis",
            user_id=user_id,
            error=str(e),
        )
