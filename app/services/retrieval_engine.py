"""
============================================================
app/services/retrieval_engine.py — Hybrid RAG + Time-Decay Scoring
============================================================
PURPOSE:
    This is the mathematical and logical core of Phase 1.
    It implements the full Hybrid Retrieval-Augmented Generation
    (RAG) pipeline:

    1. Vectorize the incoming user message
    2. Fetch Layer 1 (Redis session), Layer 2 (Supabase JSONB episodes),
       and Layer 3 (Supabase pgvector semantic memories) IN PARALLEL
    3. Apply exponential time-decay scoring to all Layer 3 candidates
    4. Discard memories scoring below the 0.65 threshold
    5. Assemble the structured LLM system prompt
    6. Call gpt-4o-mini and return the response

TIME-DECAY FORMULA:
    S_adjusted = S_raw × e^(-λ × Δt)

    Where:
        S_raw = raw cosine similarity score (0 to 1)
        λ     = 0.005 (configurable DECAY_LAMBDA)
        Δt    = days elapsed since memory was created (createdAt)

    Example:
        S_raw = 0.82, created 60 days ago:
        S_adjusted = 0.82 × e^(-0.005 × 60)
                   = 0.82 × e^(-0.30)
                   = 0.82 × 0.7408
                   = 0.607  ← below 0.65 threshold → DISCARDED

    A memory created TODAY with S_raw=0.82 scores:
        S_adjusted = 0.82 × e^(0) = 0.82 × 1.0 = 0.82 ← KEPT

WHY TIME-DECAY?
    Wellness state changes over time. A coping mechanism that
    worked 6 months ago may be less relevant now. Time-decay
    ensures the LLM is injected with timely, relevant context.
    Reinforced memories (Phase 7) have their created_at reset,
    keeping them "fresh" in the decay calculation.

CONNECTED TO:
    Phase 1  → Core chat pipeline (called by POST /api/v1/chat)
    Phase 7  → Nightly consolidation uses decay to rank what to reinforce
    Phase 9  → Benchmark suite measures precision/recall at different λ values
============================================================
"""
import asyncio
import math
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis
import structlog

from app.core.config import get_settings
from app.schemas.chat import SessionMessage
from app.services import sensory_service, embedding_service
from app.services.temporal_graph_engine import retrieve_graph_context

log = structlog.get_logger(__name__)
settings = get_settings()


# -------------------------------------------------------
# Time-Decay Scoring (NumPy implementation)
# -------------------------------------------------------

def apply_time_decay(
    raw_score: float,
    created_at: datetime,
    lambda_decay: float = None,
) -> float:
    """
    Applies exponential time-decay to a raw cosine similarity score.

    Formula: S_adjusted = S_raw × e^(-λ × Δt)

    Args:
        raw_score   : Raw cosine similarity score [0, 1]
        created_at  : Datetime the memory was created
        lambda_decay: Decay constant per day (default: DECAY_LAMBDA=0.005)

    Returns:
        Adjusted score after applying temporal decay penalty.

    Technical note:
        We use numpy's exp() for numerical precision, though
        Python's math.exp() is equally valid for scalars.
        For Phase 9 batch benchmarking, the numpy version
        enables vectorized operations across large arrays.
    """
    lam = lambda_decay or settings.DECAY_LAMBDA

    # Ensure created_at is timezone-aware
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    delta_days = (now - created_at).total_seconds() / 86400  # Convert seconds → days

    # Apply exponential decay
    decay_factor = float(np.exp(-lam * delta_days))
    adjusted = raw_score * decay_factor

    log.debug(
        "Time-decay applied",
        raw_score=round(raw_score, 4),
        delta_days=round(delta_days, 1),
        decay_factor=round(decay_factor, 4),
        adjusted_score=round(adjusted, 4),
    )
    return adjusted


def apply_time_decay_batch(
    raw_scores: list[float],
    created_ats: list[datetime],
    lambda_decay: float = None,
) -> list[float]:
    """
    Vectorized batch version of time-decay scoring using NumPy.

    Phase 9 uses this for benchmarking across large sets of
    memory candidates simultaneously (100x faster than a loop).

    Args:
        raw_scores  : Array of cosine similarity scores
        created_ats : Array of corresponding creation datetimes
        lambda_decay: Decay constant (default: DECAY_LAMBDA)

    Returns:
        List of time-decay adjusted scores, same order as input.
    """
    lam = lambda_decay or settings.DECAY_LAMBDA
    now = datetime.now(timezone.utc)

    # Vectorized computation over NumPy arrays
    delta_days_arr = np.array([
        (now - (ca if ca.tzinfo else ca.replace(tzinfo=timezone.utc))).total_seconds() / 86400
        for ca in created_ats
    ])
    decay_factors = np.exp(-lam * delta_days_arr)
    adjusted_scores = np.array(raw_scores) * decay_factors

    return adjusted_scores.tolist()


# -------------------------------------------------------
# Layer 2: Episodic Memory Retrieval (Supabase JSONB)
# -------------------------------------------------------

async def fetch_episodic_context(
    db: AsyncSession,
    user_id: str,
    limit: int = 5,
) -> list[dict]:
    """
    Fetches the most recent N active episodes from Layer 2.

    Query retrieves episodes within the active window (14 days by
    default) ordered by timestamp DESC (most recent first).
    The session_summary and extracted_metrics fields are returned
    for injection into the LLM system prompt.

    Args:
        db      : Async database session
        user_id : User whose episodes to retrieve
        limit   : Max number of episodes (default 5)

    Returns:
        List of episode dicts with summary and metrics for prompt assembly
    """
    query = text("""
        SELECT
            session_summary,
            extracted_metrics,
            timestamp
        FROM episodes
        WHERE
            user_id = :user_id
            AND archived_at IS NULL
            AND timestamp >= NOW() - INTERVAL ':days days'
        ORDER BY timestamp DESC
        LIMIT :limit
    """).bindparams(
        user_id=user_id,
        days=settings.EPISODIC_ACTIVE_DAYS,
        limit=limit,
    )

    # Handle Supabase INTERVAL parameter binding differently
    raw_query = text(f"""
        SELECT
            session_summary,
            extracted_metrics,
            timestamp
        FROM episodes
        WHERE
            user_id = :user_id
            AND archived_at IS NULL
            AND timestamp >= NOW() - INTERVAL '{settings.EPISODIC_ACTIVE_DAYS} days'
        ORDER BY timestamp DESC
        LIMIT :limit
    """)

    try:
        result = await db.execute(raw_query, {"user_id": user_id, "limit": limit})
        rows = result.mappings().all()
        episodes = [dict(row) for row in rows]
        log.debug("Episodic context fetched", user_id=user_id, count=len(episodes))
        return episodes
    except Exception as e:
        log.error("Failed to fetch episodic context", user_id=user_id, error=str(e))
        return []


# -------------------------------------------------------
# Layer 3: Semantic Memory Retrieval (pgvector HNSW)
# -------------------------------------------------------

async def fetch_semantic_memories(
    db: AsyncSession,
    user_id: str,
    query_vector: list[float],
    top_k: int = 10,
) -> list[dict]:
    """
    Performs HNSW approximate nearest-neighbor search against
    the user's semantic memory store in Supabase pgvector.

    The query uses cosine distance operator (<->) to rank
    memories by similarity to the user's current message.
    We fetch top_k=10 candidates (more than we need) so that
    after time-decay scoring and threshold filtering, we're
    left with the most relevant 3 memories for LLM injection.

    Args:
        db           : Async database session
        user_id      : User whose memories to search
        query_vector : 1536-dim embedding of the user's message
        top_k        : Number of candidates to retrieve pre-decay

    Returns:
        List of memory dicts with text, category, score, created_at
    """
    # Format vector as PostgreSQL array literal
    vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

    raw_query = text("""
        SELECT
            id,
            text,
            category,
            created_at,
            reinforcement_count,
            is_pinned,
            1 - (embedding <-> :query_vector::vector) AS similarity_score
        FROM semantic_memories
        WHERE user_id = :user_id
        ORDER BY is_pinned DESC, embedding <-> :query_vector::vector
        LIMIT :top_k
    """)
    # Phase 4: ORDER BY is_pinned DESC ensures pinned memories are fetched
    # first within the top_k window, guaranteeing they're never crowded out
    # by a high volume of non-pinned high-similarity candidates.

    try:
        result = await db.execute(raw_query, {
            "user_id": user_id,
            "query_vector": vector_str,
            "top_k": top_k,
        })
        rows = result.mappings().all()
        memories = [dict(row) for row in rows]
        log.debug("Semantic candidates fetched", user_id=user_id, count=len(memories))
        return memories
    except Exception as e:
        log.error("Failed to fetch semantic memories", user_id=user_id, error=str(e))
        return []


# -------------------------------------------------------
# Time-Decay Filter
# -------------------------------------------------------

def filter_by_decay(
    memories: list[dict],
    threshold: float = None,
    max_memories: int = 3,
) -> list[dict]:
    """
    Applies time-decay scoring to memory candidates and
    returns only those passing the threshold, ranked by
    adjusted score.

    Process:
    1. For each memory, compute: S_adj = S_raw × e^(-λ × Δt)
    2. Discard memories where S_adj < threshold (default 0.65)
    3. Return top max_memories by adjusted score

    Args:
        memories    : Raw memory dicts from pgvector query
        threshold   : Minimum adjusted score to keep (default 0.65)
        max_memories: Max memories to inject into LLM context

    Returns:
        Filtered, ranked list of memory dicts with decay scores added
    """
    cutoff = threshold or settings.SIMILARITY_THRESHOLD

    scored = []
    for mem in memories:
        raw_score = float(mem.get("similarity_score", 0))
        created_at = mem.get("created_at")
        is_pinned: bool = bool(mem.get("is_pinned", False))

        if not isinstance(created_at, datetime):
            # Handle string timestamps from Supabase
            created_at = datetime.fromisoformat(str(created_at))

        if is_pinned:
            # Phase 4: Pinned memories bypass time-decay entirely.
            # They are always injected into LLM context regardless of age.
            # S_adjusted = S_raw (decay factor = 1.0)
            adjusted = raw_score
            mem["decay_bypassed"] = True
        else:
            adjusted = apply_time_decay(raw_score, created_at)
            mem["decay_bypassed"] = False

        mem["adjusted_score"] = round(adjusted, 4)
        mem["raw_score"] = round(raw_score, 4)

        if adjusted >= cutoff:
            scored.append(mem)

    # Sort: pinned memories float to the top, then by adjusted score
    scored.sort(key=lambda x: (x.get("is_pinned", False), x["adjusted_score"]), reverse=True)

    # Limit to max_memories for context window budget
    kept = scored[:max_memories]

    pinned_count = sum(1 for m in kept if m.get("is_pinned", False))
    log.info(
        "Memory filter applied",
        total_candidates=len(memories),
        passed_threshold=len(scored),
        injected=len(kept),
        pinned_injected=pinned_count,
        threshold=cutoff,
    )
    return kept


# -------------------------------------------------------
# System Prompt Assembly
# -------------------------------------------------------

def assemble_system_prompt(
    session_messages: list[SessionMessage],
    episodes: list[dict],
    semantic_memories: list[dict],
    graph_context: str = "",
) -> str:
    """
    Constructs the structured LLM system prompt by combining
    all three memory layers into a token-efficient format.

    TOKEN BUDGET BREAKDOWN:
        System base instructions : ~300 tokens
        Layer 1 (session)        : ~600 tokens (10 messages × 60 tokens)
        Layer 2 (episodes)       : ~500 tokens (5 episodes × 100 tokens)
        Layer 3 (3 memories)     : ~200 tokens (3 facts × 67 tokens)
        ─────────────────────────────────────────────
        Total input budget       : ~1,600 tokens
        User message             : ~200 tokens
        LLM response budget      : ~500 tokens
        ─────────────────────────────────────────────
        Grand Total              : ~2,300 tokens
        (Well within gpt-4o-mini's 128k context window)

    CONNECTED TO:
        Phase 9 → Token counting with tiktoken validates this budget
    """
    # Layer 1: Recent conversation history
    session_text = ""
    if session_messages:
        lines = []
        for msg in session_messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")
        session_text = "\n".join(lines)
    else:
        session_text = "(No prior conversation in this session)"

    # Layer 2: Recent episodic health timeline
    episode_text = ""
    if episodes:
        lines = []
        for ep in episodes:
            metrics = ep.get("extracted_metrics", {})
            ts = ep.get("timestamp", "")
            if isinstance(ts, datetime):
                ts = ts.strftime("%Y-%m-%d")
            mood = metrics.get("moodScore", "N/A")
            stressor = metrics.get("primaryStressor", "N/A")
            sleep = metrics.get("sleepHoursLogged", "N/A")
            summary = ep.get("session_summary", "")[:200]  # Truncate for token budget
            lines.append(
                f"[{ts}] Mood: {mood}/10 | Sleep: {sleep}h | "
                f"Stressor: {stressor}\nSummary: {summary}"
            )
        episode_text = "\n\n".join(lines)
    else:
        episode_text = "(No recent health episodes on record)"

    # Layer 3: Semantic memory facts (time-decay filtered)
    memory_text = ""
    if semantic_memories:
        lines = []
        for mem in semantic_memories:
            category = mem.get("category", "fact").upper().replace("_", " ")
            fact_text = mem.get("text", "")
            reinforced = mem.get("reinforcement_count", 1)
            score = mem.get("adjusted_score", 0)
            lines.append(
                f"[{category}] (Confidence: {reinforced}x confirmed, "
                f"Relevance: {score:.2f})\n{fact_text}"
            )
        memory_text = "\n\n".join(lines)
    else:
        memory_text = "(No long-term memory facts retrieved)"

    # Layer 4: Temporal Knowledge Graph (Causal & Relational Context)
    graph_section = ""
    if graph_context:
        graph_section = f"""
--- CAUSAL KNOWLEDGE GRAPH (Layer 4: Temporal Relationships) ---
{graph_context}
"""

    prompt = f"""You are a compassionate AI wellness companion. You are speaking with a user who has been sharing their mental and physical health journey with you over time.

You have access to four layers of context about this user:

--- RECENT CONVERSATION (Layer 1: Active Session) ---
{session_text}

--- HEALTH TIMELINE (Layer 2: Last {settings.EPISODIC_ACTIVE_DAYS} Days) ---
{episode_text}

--- LONG-TERM MEMORY FACTS (Layer 3: Verified Patterns) ---
{memory_text}
{graph_section}
--- YOUR INSTRUCTIONS ---
1. Respond with genuine empathy and warmth, as a trusted wellness companion.
2. Reference specific patterns from the long-term memory facts when relevant (e.g., "I know you tend to feel more anxious before evaluations...").
3. When the causal knowledge graph is present, use it proactively to anticipate downstream effects (e.g., if the user mentions their exam trigger, warn them about the insomnia pattern it historically causes).
4. Keep responses concise, actionable, and grounded in the provided context.
5. NEVER fabricate health data, diagnoses, or facts not present in the context above.
6. NEVER provide medical diagnoses or prescription recommendations.
7. If the user shows signs of crisis or self-harm, immediately refer to emergency resources.
8. Celebrate progress and milestones visible in the episode timeline.
"""
    return prompt


# -------------------------------------------------------
# Core Chat Execution
# -------------------------------------------------------

async def run_hybrid_rag_pipeline(
    user_id: str,
    message: str,
    db: AsyncSession,
    redis: aioredis.Redis,
) -> dict:
    """
    Orchestrates the full Hybrid RAG pipeline for a single chat turn.

    Pipeline Steps:
    1. Vectorize the user's message (OpenAI embedding)
    2. PARALLEL fetch of Layer 1, Layer 2, Layer 3
    3. Apply time-decay scoring and filter Layer 3 results
    4. Assemble structured LLM system prompt
    5. Call gpt-4o-mini and return the response

    Args:
        user_id : The user sending the message
        message : The raw user message text
        db      : Async database session (Supabase)
        redis   : Async Redis client

    Returns:
        Dict with: response (str), memories_used (int), debug (dict)
    """
    start_time = time.time()

    # ── Step 1: Vectorize incoming message ─────────────────────────
    query_vector = await embedding_service.embed_text(message)

    # ── Step 2: Parallel fetch across all four layers ──────────────
    # asyncio.gather() runs all four fetches concurrently.
    # Graph retrieval target: < 100ms total (seed < 30ms + traversal < 50ms)
    session_msgs, episodes, raw_semantic, graph_context = await asyncio.gather(
        sensory_service.get_active_session(redis, user_id),
        fetch_episodic_context(db, user_id),
        fetch_semantic_memories(db, user_id, query_vector),
        retrieve_graph_context(db, user_id, query_vector),
    )

    # ── Step 3: Time-decay scoring + threshold filtering ────────────
    filtered_memories = filter_by_decay(raw_semantic)

    # ── Step 4: Assemble system prompt ─────────────────────────────
    system_prompt = assemble_system_prompt(
        session_msgs, episodes, filtered_memories, graph_context
    )

    # ── Step 5: LLM Call (with automatic fallback on 429 rate-limit) ───
    client = embedding_service.get_openai_client()
    chat_payload = dict(
        temperature=settings.OPENAI_CHAT_TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
    )

    try:
        chat_response = await client.chat.completions.create(
            model=settings.OPENAI_CHAT_MODEL,
            **chat_payload,
        )
        model_used = settings.OPENAI_CHAT_MODEL
    except Exception as primary_err:
        # Detect 429 / RESOURCE_EXHAUSTED from Gemini free tier quota
        is_rate_limit = "429" in str(primary_err) or "RESOURCE_EXHAUSTED" in str(primary_err)
        fallback = settings.OPENAI_CHAT_MODEL_FALLBACK
        if is_rate_limit and fallback:
            log.warning(
                "Primary model rate-limited, falling back",
                primary_model=settings.OPENAI_CHAT_MODEL,
                fallback_model=fallback,
            )
            chat_response = await client.chat.completions.create(
                model=fallback,
                **chat_payload,
            )
            model_used = fallback
        else:
            raise

    response_text = chat_response.choices[0].message.content
    elapsed_ms = round((time.time() - start_time) * 1000)

    log.info(
        "RAG pipeline complete",
        user_id=user_id,
        model_used=model_used,
        memories_used=len(filtered_memories),
        episodes_used=len(episodes),
        session_msgs=len(session_msgs),
        elapsed_ms=elapsed_ms,
        input_tokens=chat_response.usage.prompt_tokens,
        output_tokens=chat_response.usage.completion_tokens,
    )

    # ── Append both turns to Layer 1 session buffer ─────────────────
    await sensory_service.append_message(redis, user_id, "user", message)
    await sensory_service.append_message(redis, user_id, "assistant", response_text)

    return {
        "response": response_text,
        "memories_used": len(filtered_memories),
        "debug": {
            "elapsed_ms": elapsed_ms,
            "input_tokens": chat_response.usage.prompt_tokens,
            "output_tokens": chat_response.usage.completion_tokens,
            "layer1_messages": len(session_msgs),
            "layer2_episodes": len(episodes),
            "layer3_candidates": len(raw_semantic),
            "layer3_after_decay": len(filtered_memories),
            "layer4_graph_paths": len(graph_context.splitlines()) if graph_context else 0,
        }
    }
