"""
============================================================
app/services/consolidation_service.py -- Phase 7 Batch Memory Consolidation
============================================================
PURPOSE:
    Processes daily Episodes and extracts durable, long-term
    semantic facts (triggers, coping mechanisms, baselines,
    symptoms, milestones, preferences, goals, patterns) into
    the `semantic_memories` table.

    This service is NEVER called per chat session. It is
    designed to be triggered by an external scheduler (cron,
    GitHub Actions, APScheduler, etc.) via:

        POST /api/v1/system/consolidate

ARCHITECTURE:
    1. Fetch PENDING episodes (batch of N, via FOR UPDATE SKIP LOCKED)
    2. Mark each as PROCESSING (atomic claim -- prevents double-run)
    3. Call LLM with strict JSON schema to extract durable facts
    4. For each fact: embed + upsert via memory_service.upsert_semantic_fact
    5. On success -> CONSOLIDATED | On any error -> FAILED (retry later)

WHY BATCH CONSOLIDATION (not per-session)?
    A single session message like "I am stressed about my exam"
    does not represent a durable long-term pattern. Batch processing
    allows the LLM to see multiple sessions worth of context and
    extract only facts that appear stable and meaningful over time.

CONCURRENT SAFETY:
    FOR UPDATE SKIP LOCKED ensures that even if multiple workers
    hit the endpoint simultaneously, each episode is processed
    exactly once. The PROCESSING state acts as a distributed lock.

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

You will be given a summary of a user's wellness session.

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

            # The LLM may return either a bare list or {"facts": [...]}
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

            # Validate and filter: only allow known categories + non-empty content
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
            log.warning(
                "Fact extraction JSON parse error",
                attempt=attempt + 1,
                error=str(e),
            )
        except Exception as e:
            log.error(
                "Fact extraction LLM error",
                attempt=attempt + 1,
                error=str(e),
            )

    log.error("Fact extraction failed after 2 attempts; returning empty list")
    return []


# -------------------------------------------------------
# Core Batch Runner
# -------------------------------------------------------

async def run_batch(db: Optional[AsyncSession] = None) -> dict:
    """
    Top-level consolidation batch runner.

    Processes up to settings.CONSOLIDATION_BATCH_SIZE PENDING
    episodes per call. Each episode is:
      1. Atomically claimed (PENDING -> PROCESSING) via FOR UPDATE SKIP LOCKED
      2. LLM-extracted for durable semantic facts
      3. Each fact is embedded and upserted via memory_service
      4. Marked CONSOLIDATED on success, FAILED on error

    Args:
        db : Optional AsyncSession. If None, a fresh session is created.

    Returns:
        dict with summary stats:
            processed    : Number of episodes attempted
            consolidated : Number successfully consolidated
            failed       : Number that failed
            created      : Total new memory rows created
            reinforced   : Total existing memory rows reinforced
    """
    t_start = time.perf_counter()
    stats = {
        "processed": 0,
        "consolidated": 0,
        "failed": 0,
        "created": 0,
        "reinforced": 0,
    }

    log.info(
        "Consolidation batch started",
        batch_size=settings.CONSOLIDATION_BATCH_SIZE,
    )

    owns_session = db is None
    if owns_session:
        db = AsyncSessionLocal()

    try:
        # Step 1: Atomically claim PENDING episodes
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

        # Mark all fetched episodes as PROCESSING
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

        log.info("Episodes claimed for processing", count=len(episodes))

        # Step 2: Process each episode individually
        for episode in episodes:
            stats["processed"] += 1
            episode_id = episode["id"]
            user_id = episode["user_id"]
            session_summary = episode["session_summary"]

            try:
                # 2a. Extract durable facts via LLM
                facts = await _extract_facts_from_summary(session_summary)

                if not facts:
                    log.info(
                        "No durable facts found; marking consolidated",
                        episode_id=episode_id,
                        user_id=user_id,
                    )
                    async with AsyncSessionLocal() as inner_db:
                        await inner_db.execute(
                            text(
                                "UPDATE episodes SET consolidation_status = 'CONSOLIDATED' "
                                "WHERE id = :id"
                            ),
                            {"id": episode_id},
                        )
                        await inner_db.commit()
                    stats["consolidated"] += 1
                    continue

                # 2b. Batch-embed all facts in one OpenAI call
                contents = [f["content"] for f in facts]
                vectors = await embed_batch(contents)

                # 2c. Upsert each fact (deduplicated) + mark CONSOLIDATED
                episode_created = 0
                episode_reinforced = 0

                async with AsyncSessionLocal() as inner_db:
                    async with inner_db.begin():
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

                        # Mark episode CONSOLIDATED inside same transaction
                        await inner_db.execute(
                            text(
                                "UPDATE episodes SET consolidation_status = 'CONSOLIDATED' "
                                "WHERE id = :id"
                            ),
                            {"id": episode_id},
                        )

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
                log.error(
                    "Episode consolidation failed; marking FAILED",
                    episode_id=episode_id,
                    user_id=user_id,
                    error=str(e),
                )
                try:
                    async with AsyncSessionLocal() as err_db:
                        await err_db.execute(
                            text(
                                "UPDATE episodes SET consolidation_status = 'FAILED' "
                                "WHERE id = :id"
                            ),
                            {"id": episode_id},
                        )
                        await err_db.commit()
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
