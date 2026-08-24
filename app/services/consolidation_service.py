"""
============================================================
app/services/consolidation_service.py -- Phase 7 Batch Memory Consolidation
============================================================
PURPOSE:
    Processes daily Episodes and extracts durable, long-term
    semantic facts into the semantic_memories table.

    Triggered by an external scheduler via:
        POST /api/v1/system/consolidate

ARCHITECTURE (two-transaction design -- Fix #3):
    Transaction 1  (short, minimal lock time):
        SELECT PENDING episodes FOR UPDATE SKIP LOCKED
        UPDATE status -> PROCESSING
        COMMIT                        <-- lock released here

    Long work outside any DB lock:
        Call LLM  (10-30 seconds)
        Call OpenAI embedding API
        All CPU/network work

    Transaction 2 (per-episode, fully atomic -- Fix #5):
        BEGIN
          Upsert all extracted facts into semantic_memories
          UPDATE episode status -> CONSOLIDATED
        COMMIT
        -- If ANY step fails: ROLLBACK
        --   -> episode marked FAILED (no partial memories written)

WHY TWO TRANSACTIONS?
    Holding a DB row lock during a 10-30 second LLM API call is
    dangerous in production:
      - Blocks other DB connections from accessing those rows
      - Holds a DB connection open for the entire LLM call duration
      - Under high load, this can exhaust the connection pool
    By releasing the PROCESSING lock after the claim transaction commits,
    other queries can see the PROCESSING status and skip those rows cleanly.

CONCURRENT SAFETY:
    FOR UPDATE SKIP LOCKED: if Worker B sees a row already locked by
    Worker A during the claim transaction, it skips it. After Worker A
    commits PROCESSING, Worker B's own query won't select it (status
    is no longer PENDING). So each episode is processed exactly once.

CONNECTED TO:
    Phase 5  -> episodes table (source)
    Phase 4  -> semantic_memories table (destination)
    Phase 7  -> app/api/v1/system.py (trigger endpoint)
    Phase 7  -> app/services/memory_service.upsert_semantic_fact (dedup)
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
from app.services.embedding_service import embed_batch
from app.services.graph_service import update_knowledge_graph
from app.services.memory_service import upsert_semantic_fact

log = structlog.get_logger(__name__)
settings = get_settings()

# -------------------------------------------------------
# Allowed semantic fact categories (strict enum)
# The LLM is never permitted to invent new categories.
# -------------------------------------------------------
ALLOWED_CATEGORIES = frozenset({
    "trigger",
    "coping_mechanism",
    "baseline",
    "symptom",
    "milestone",
    "preference",
    "goal",
    "pattern",
})

# -------------------------------------------------------
# LLM System Prompt
# -------------------------------------------------------
_EXTRACTION_SYSTEM_PROMPT = """\
You are a clinical memory extraction assistant.

You will be given a summary of a user wellness session.

Your task is to extract DURABLE, LONG-TERM semantic facts about the user.

Rules:
1. Only extract information that is EXPLICITLY supported by the episode.
2. Only extract information that is LIKELY to be USEFUL beyond this conversation.
3. Each fact must belong to ONE of these exact categories (use lowercase):
   - trigger          : Things that cause anxiety, stress, or negative emotions
   - coping_mechanism : Strategies that help the user manage difficult emotions
   - baseline         : Stable, long-term health or life facts (e.g. diagnoses, allergies)
   - symptom          : Recurring physical or psychological symptoms
   - milestone        : Positive achievements or progress the user has made
   - preference       : The user's stated likes, dislikes, or personal preferences
   - goal             : Something the user is actively working towards
   - pattern          : A recurring behavioural or emotional pattern
4. Do NOT infer information not explicitly stated.
5. Do NOT extract temporary or one-time facts (e.g. "user had a headache today").
6. Keep each fact concise (max 300 characters).
7. Return a JSON array of objects. Each object must have:
   - "content"  : string (the fact in third-person)
   - "category" : string (one of the categories above, lowercase)

IMPORTANT: If no durable facts can be extracted, return an EMPTY ARRAY: []
Do not force facts where none exist.

Output format (strict JSON only, no prose):
[
  {"content": "...", "category": "trigger"},
  {"content": "...", "category": "coping_mechanism"}
]
"""


# -------------------------------------------------------
# LLM Extraction
# -------------------------------------------------------

def _get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def _extract_facts_from_summary(session_summary: str) -> list:
    """
    Calls the LLM to extract durable semantic facts from a session summary.

    Returns a list of dicts: [{"content": str, "category": str}, ...]
    Returns [] if extraction fails or no durable facts exist.

    NEVER RAISES -- returns [] on any error so the pipeline can continue.
    """
    if not session_summary or not session_summary.strip():
        return []

    client = _get_openai_client()

    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=settings.OPENAI_CHAT_MODEL,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Extract durable semantic facts from the following "
                            f"session summary:\n\n{session_summary[:5000]}"
                        ),
                    },
                ],
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            # The LLM may return a bare list or {"facts": [...]}
            if isinstance(parsed, list):
                facts = parsed
            elif isinstance(parsed, dict):
                facts = (
                    parsed.get("facts")
                    or parsed.get("memories")
                    or parsed.get("items")
                    or []
                )
            else:
                facts = []

            # Validate: only allow known categories + non-empty content
            valid_facts = []
            for f in facts:
                if not isinstance(f, dict):
                    continue
                content = str(f.get("content", "")).strip()[:300]
                category = str(f.get("category", "")).strip().lower()
                if content and category in ALLOWED_CATEGORIES:
                    valid_facts.append({"content": content, "category": category})

            log.info(
                "Fact extraction complete",
                attempt=attempt + 1,
                total_facts=len(valid_facts),
                summary_preview=session_summary[:60],
            )
            return valid_facts

        except json.JSONDecodeError as e:
            log.warning("Fact extraction JSON parse error", attempt=attempt + 1, error=str(e))
        except Exception as e:
            log.error("Fact extraction LLM error", attempt=attempt + 1, error=str(e))

    log.error("Fact extraction failed after 2 attempts; returning empty list")
    return []


# -------------------------------------------------------
# Episode finalisation helpers
# -------------------------------------------------------

async def _mark_episode(episode_id: str, status: str) -> None:
    """
    Opens a fresh DB session and updates a single episode's status.
    Used after long external calls (LLM, embeddings) where we no longer
    hold any existing session or transaction open.
    """
    async with AsyncSessionLocal() as db:
        await db.execute(
            text("UPDATE episodes SET consolidation_status = :s WHERE id = :id"),
            {"s": status, "id": episode_id},
        )
        await db.commit()


# -------------------------------------------------------
# Core Batch Runner
# -------------------------------------------------------

async def run_batch(db: Optional[AsyncSession] = None) -> dict:
    """
    Top-level consolidation batch runner.

    Two-transaction design (Fix #3):
    ─────────────────────────────────
    Transaction 1 (short, minimal lock time):
        SELECT PENDING FOR UPDATE SKIP LOCKED → lock rows
        UPDATE status = PROCESSING
        COMMIT                    ← DB lock fully released here

    Long work (outside any lock):
        LLM call per episode
        OpenAI embedding API call

    Transaction 2 (per episode, Fix #5):
        BEGIN
          Upsert all facts into semantic_memories
          UPDATE episode status = CONSOLIDATED
        COMMIT  ← if any step fails: ROLLBACK, then mark FAILED

    Args:
        db : Optional AsyncSession. If None, a fresh session is created
             and closed by this function.

    Returns:
        dict with summary stats:
            processed    : Episodes attempted
            consolidated : Episodes successfully completed
            failed       : Episodes that errored
            created      : New memory rows inserted
            reinforced   : Existing memory rows reinforced
    """
    t_start = time.perf_counter()
    stats = {
        "processed": 0,
        "consolidated": 0,
        "failed": 0,
        "created": 0,
        "reinforced": 0,
    }

    log.info("Consolidation batch started", batch_size=settings.CONSOLIDATION_BATCH_SIZE)

    owns_session = db is None
    if owns_session:
        db = AsyncSessionLocal()

    try:
        # ── Transaction 1: Claim episodes atomically ──────────────────────────
        # FOR UPDATE SKIP LOCKED: Worker B will skip rows already locked by
        # Worker A. After this commit, rows show PROCESSING status so no
        # subsequent query will pick them up again.
        result = await db.execute(
            text("""
                SELECT id, user_id, session_summary
                FROM episodes
                WHERE consolidation_status = 'PENDING'
                ORDER BY timestamp ASC
                LIMIT :batch_size
                FOR UPDATE SKIP LOCKED
            """),
            {"batch_size": settings.CONSOLIDATION_BATCH_SIZE},
        )
        episodes = result.mappings().all()

        if not episodes:
            log.info("No pending episodes to consolidate")
            return stats

        episode_ids = [e["id"] for e in episodes]

        await db.execute(
            text("""
                UPDATE episodes
                SET consolidation_status = 'PROCESSING'
                WHERE id = ANY(:ids)
            """),
            {"ids": episode_ids},
        )
        await db.commit()
        # ── DB lock fully released here. LLM calls happen AFTER this point. ──

        log.info("Episodes claimed for processing", count=len(episodes))

        # ── Process each episode with its own independent transaction ─────────
        for episode in episodes:
            stats["processed"] += 1
            episode_id = episode["id"]
            user_id = episode["user_id"]
            session_summary = episode["session_summary"]

            try:
                # Step A: LLM extraction (outside any DB lock/transaction)
                facts = await _extract_facts_from_summary(session_summary)

                if not facts:
                    # LLM found nothing durable -- still counts as success
                    log.info(
                        "No durable facts extracted; marking consolidated",
                        episode_id=episode_id,
                        user_id=user_id,
                    )
                    await _mark_episode(episode_id, "CONSOLIDATED")
                    stats["consolidated"] += 1
                    continue

                # Step B: Batch-embed all facts in a single OpenAI API call
                contents = [f["content"] for f in facts]
                vectors = await embed_batch(contents)

                # Step C: Transaction 2 -- upsert facts + mark CONSOLIDATED
                # Fix #5: all DB writes for this episode are inside one
                # BEGIN/COMMIT. If ANY upsert fails, ROLLBACK ensures no
                # partial memories are written and the episode is marked FAILED.
                episode_created = 0
                episode_reinforced = 0

                async with AsyncSessionLocal() as inner_db:
                    async with inner_db.begin():  # auto-ROLLBACK on exception
                        for fact, vector in zip(facts, vectors):
                            result = await upsert_semantic_fact(
                                db=inner_db,
                                user_id=user_id,
                                category=fact["category"],
                                text_content=fact["content"],
                                embedding_vector=vector,
                            )
                            if result["action"] == "created":
                                episode_created += 1
                            else:
                                episode_reinforced += 1

                        # Phase 7.5: Also update the Temporal Knowledge Graph.
                        # Called inside the SAME transaction so graph writes
                        # and memory writes succeed or fail atomically.
                        await update_knowledge_graph(
                            db=inner_db,
                            user_id=user_id,
                            session_summary=session_summary,
                            episode_id=episode_id,
                        )

                        # Mark CONSOLIDATED inside the same transaction so
                        # the status change and memory writes are atomic.
                        await inner_db.execute(
                            text(
                                "UPDATE episodes SET consolidation_status = 'CONSOLIDATED' "
                                "WHERE id = :id"
                            ),
                            {"id": episode_id},
                        )
                    # inner_db.begin() context manager commits here

                stats["consolidated"] += 1
                stats["created"] += episode_created
                stats["reinforced"] += episode_reinforced

                log.info(
                    "Episode consolidated",
                    episode_id=episode_id,
                    user_id=user_id,
                    facts_total=len(facts),
                    created=episode_created,
                    reinforced=episode_reinforced,
                )

            except Exception as e:
                # Fix #5: If Transaction 2 raised (auto-rolled back by begin()),
                # no partial memories exist in the DB.
                # Mark this episode FAILED so it can be retried later.
                log.error(
                    "Episode consolidation failed; marking FAILED",
                    episode_id=episode_id,
                    user_id=user_id,
                    error=str(e),
                )
                try:
                    await _mark_episode(episode_id, "FAILED")
                except Exception as inner_e:
                    log.error(
                        "Could not mark episode as FAILED",
                        episode_id=episode_id,
                        error=str(inner_e),
                    )
                stats["failed"] += 1

    finally:
        if owns_session:
            await db.close()

    elapsed = round(time.perf_counter() - t_start, 3)
    log.info("Consolidation batch complete", elapsed_seconds=elapsed, **stats)
    return stats
