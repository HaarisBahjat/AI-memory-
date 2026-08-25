"""
============================================================
app/services/graph_service.py -- Phase 7.5 Temporal Graph Builder
============================================================
PURPOSE:
    Populates the Temporal Knowledge Graph during nightly consolidation.

    Called AFTER semantic fact extraction in consolidation_service.py.
    For each episode it:
        1. Calls the LLM to extract (entity, relation, entity, evidence) triples
        2. Resolves each entity to an existing node using pgvector similarity
           (cosine distance <= GRAPH_ENTITY_MERGE_DISTANCE = 0.10  =>  merge/reinforce)
        3. Inserts or reinforces edges with temporal bounds and weight

RELATION TYPES (strict enum, LLM is never allowed to invent new ones):
    TRIGGERS         source causes / precipitates target
    ALLEVIATES       source reduces / helps target
    WORSENS          source amplifies target
    FOLLOWED_BY      source temporally precedes target
    SUPERSEDES       source replaces an older coping strategy
    ASSOCIATED_WITH  co-occurrence / correlation
    PART_OF          target is a component of source
    REDUCED          source decreased / improved target

ENTITY TYPES (strict enum):
    TRIGGER, COPING_MECHANISM, SYMPTOM, PERSON,
    ACTIVITY, BASELINE, GOAL, EVENT

CONNECTED TO:
    Phase 7.5 -> app/services/consolidation_service.py  (caller)
    Phase 7.5 -> app/services/temporal_graph_engine.py  (reader / query side)
    Phase 7.5 -> schema.sql: knowledge_nodes, knowledge_edges
============================================================
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.services.embedding_service import embed_text, embed_batch

log = structlog.get_logger(__name__)
settings = get_settings()

# -------------------------------------------------------
# Strict enum constants
# -------------------------------------------------------

ALLOWED_ENTITY_TYPES = frozenset({
    "TRIGGER", "COPING_MECHANISM", "SYMPTOM",
    "PERSON", "ACTIVITY", "BASELINE", "GOAL", "EVENT",
})

ALLOWED_RELATION_TYPES = frozenset({
    "TRIGGERS", "ALLEVIATES", "WORSENS", "FOLLOWED_BY",
    "SUPERSEDES", "ASSOCIATED_WITH", "PART_OF", "REDUCED",
})

# -------------------------------------------------------
# LLM System Prompt — Triple Extraction
# -------------------------------------------------------

_TRIPLE_EXTRACTION_PROMPT = """\
You are a clinical knowledge graph extraction assistant.

You will receive a wellness session summary. Extract CAUSAL and TEMPORAL
relationships between named entities as directed triples.

Rules:
1. Each triple must be: (source_entity, relation, target_entity, evidence)
2. source_entity and target_entity must each have:
   - "name"        : short canonical name (2-5 words, Title Case)
   - "entity_type" : exactly one of:
       TRIGGER | COPING_MECHANISM | SYMPTOM | PERSON | ACTIVITY | BASELINE | GOAL | EVENT
3. relation must be exactly one of:
   TRIGGERS | ALLEVIATES | WORSENS | FOLLOWED_BY | SUPERSEDES | ASSOCIATED_WITH | PART_OF | REDUCED
4. evidence: a brief quoted phrase (max 120 chars) from the summary that supports this triple
5. weight: float 0.1–1.0 representing your confidence in this relationship

DO NOT invent entities not mentioned in the summary.
DO NOT extract trivial or one-time facts.
If there are no meaningful relationships, return: []

Output: strict JSON array, no prose.
[
  {
    "source": {"name": "Exam Stress", "entity_type": "TRIGGER"},
    "relation": "TRIGGERS",
    "target": {"name": "Insomnia", "entity_type": "SYMPTOM"},
    "evidence": "user reported not sleeping before exam week",
    "weight": 0.85
  }
]
"""


# -------------------------------------------------------
# LLM Triple Extraction
# -------------------------------------------------------

async def extract_triples(session_summary: str) -> list[dict]:
    """
    Calls GPT to extract temporal knowledge graph triples from a session summary.

    Returns a validated list of triple dicts. Returns [] on any failure.
    NEVER RAISES -- errors are logged and an empty list is returned so the
    consolidation pipeline can continue.

    Each returned triple has the shape:
        {
            "source": {"name": str, "entity_type": str},
            "relation": str,
            "target": {"name": str, "entity_type": str},
            "evidence": str,
            "weight": float (0.1 - 1.0)
        }
    """
    if not session_summary or not session_summary.strip():
        return []

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=settings.OPENAI_CHAT_MODEL,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _TRIPLE_EXTRACTION_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Extract relationship triples from this session summary:\n\n"
                            + session_summary[:5000]
                        ),
                    },
                ],
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            # Handle both a bare list and {"triples": [...]} wrapper
            if isinstance(parsed, list):
                triples_raw = parsed
            elif isinstance(parsed, dict):
                triples_raw = (
                    parsed.get("triples")
                    or parsed.get("relationships")
                    or parsed.get("items")
                    or []
                )
            else:
                triples_raw = []

            # Validate each triple
            valid = []
            for t in triples_raw:
                if not isinstance(t, dict):
                    continue
                src = t.get("source", {})
                tgt = t.get("target", {})
                rel = str(t.get("relation", "")).strip().upper()
                evidence = str(t.get("evidence", "")).strip()[:120]
                weight_raw = t.get("weight", 0.7)
                # Guard: LLM may return a non-numeric value (e.g. "high").
                # Use a safe cast with fallback to 0.7 instead of bare float().
                try:
                    weight = float(weight_raw)
                except (TypeError, ValueError):
                    weight = 0.7

                src_name = str(src.get("name", "")).strip()[:100]
                src_type = str(src.get("entity_type", "")).strip().upper()
                tgt_name = str(tgt.get("name", "")).strip()[:100]
                tgt_type = str(tgt.get("entity_type", "")).strip().upper()

                if (
                    src_name and tgt_name
                    and src_type in ALLOWED_ENTITY_TYPES
                    and tgt_type in ALLOWED_ENTITY_TYPES
                    and rel in ALLOWED_RELATION_TYPES
                    and 0.0 < weight <= 1.0
                ):
                    valid.append({
                        "source": {"name": src_name, "entity_type": src_type},
                        "relation": rel,
                        "target": {"name": tgt_name, "entity_type": tgt_type},
                        "evidence": evidence,
                        "weight": round(weight, 3),
                    })

            log.info(
                "Triple extraction complete",
                attempt=attempt + 1,
                valid_triples=len(valid),
            )
            return valid

        except json.JSONDecodeError as e:
            log.warning("Triple extraction JSON error", attempt=attempt + 1, error=str(e))
        except Exception as e:
            log.error("Triple extraction LLM error", attempt=attempt + 1, error=str(e))

    log.error("Triple extraction failed after 2 attempts; returning []")
    return []


# -------------------------------------------------------
# Entity Resolution (find or create a knowledge node)
# -------------------------------------------------------

async def resolve_node(
    db: AsyncSession,
    user_id: str,
    name: str,
    entity_type: str,
    embedding_vector: list[float],
) -> str:
    """
    Finds an existing knowledge node that is semantically close to `name`,
    or inserts a new one.

    Resolution logic:
        1. pgvector search: find the nearest node (same user, same entity_type)
        2. If cosine distance <= GRAPH_ENTITY_MERGE_DISTANCE (0.10) -> merge:
               - Increment mention_count
               - Update last_observed_at
               - Return existing node's id
        3. Else -> INSERT new node, return new id

    Args:
        db               : Async DB session (must be in an active transaction)
        user_id          : Owning user UUID
        name             : Canonical entity name
        entity_type      : One of ALLOWED_ENTITY_TYPES
        embedding_vector : 1536-dim float vector of the entity name

    Returns:
        node_id (str UUID)
    """
    merge_distance = settings.GRAPH_ENTITY_MERGE_DISTANCE
    vec_str = "[" + ",".join(str(v) for v in embedding_vector) + "]"

    # Search for nearest existing node in the same type bucket
    nearest = await db.execute(
        text("""
            SELECT id, name, (embedding <=> :emb::vector) AS cos_dist
            FROM knowledge_nodes
            WHERE user_id = :uid AND entity_type = :etype
            ORDER BY cos_dist ASC
            LIMIT 1
        """),
        {"emb": vec_str, "uid": user_id, "etype": entity_type},
    )
    row = nearest.mappings().first()

    if row is not None and row["cos_dist"] <= merge_distance:
        # Merge into existing node
        node_id = row["id"]
        await db.execute(
            text("""
                UPDATE knowledge_nodes
                SET
                    mention_count    = mention_count + 1,
                    last_observed_at = NOW()
                WHERE id = :nid AND user_id = :uid
            """),
            {"nid": node_id, "uid": user_id},
        )
        log.debug(
            "Entity resolved to existing node (merged)",
            node_id=node_id,
            existing_name=row["name"],
            new_name=name,
            cos_dist=round(row["cos_dist"], 4),
        )
        return node_id

    # Insert new node
    node_id = str(uuid.uuid4())
    await db.execute(
        text("""
            INSERT INTO knowledge_nodes
                (id, user_id, name, entity_type, embedding, description,
                 first_observed_at, last_observed_at, mention_count)
            VALUES
                (:id, :uid, :name, :etype, :emb::vector, NULL, NOW(), NOW(), 1)
        """),
        {
            "id": node_id,
            "uid": user_id,
            "name": name,
            "etype": entity_type,
            "emb": vec_str,
        },
    )
    log.info(
        "Entity node created",
        node_id=node_id,
        name=name,
        entity_type=entity_type,
    )
    return node_id


# -------------------------------------------------------
# Edge Upsert
# -------------------------------------------------------

async def upsert_edge(
    db: AsyncSession,
    user_id: str,
    source_node_id: str,
    target_node_id: str,
    relation_type: str,
    weight: float,
    evidence: str,
    episode_id: Optional[str] = None,
) -> str:
    """
    Inserts a new knowledge edge or reinforces an existing active one.

    "Active" means: valid_to IS NULL AND same user + same source + target + relation.

    Reinforcement:
        - Increments observation_count
        - Blends the edge weight: new_weight = (old_weight + new_weight) / 2
        - Updates evidence with the latest quote

    Args:
        db              : Async DB session (active transaction)
        user_id         : Owner UUID
        source_node_id  : UUID of source node
        target_node_id  : UUID of target node
        relation_type   : One of ALLOWED_RELATION_TYPES
        weight          : LLM confidence score 0.0–1.0
        evidence        : Supporting quote from session
        episode_id      : Episode UUID that produced this edge (nullable)

    Returns:
        edge_id (str UUID)
    """
    # Check for an existing active edge (same source → rel → target, valid_to IS NULL)
    existing = await db.execute(
        text("""
            SELECT id, weight, observation_count
            FROM knowledge_edges
            WHERE
                user_id        = :uid
                AND source_node_id = :src
                AND target_node_id = :tgt
                AND relation_type  = :rel
                AND valid_to       IS NULL
            LIMIT 1
        """),
        {
            "uid": user_id,
            "src": source_node_id,
            "tgt": target_node_id,
            "rel": relation_type,
        },
    )
    row = existing.mappings().first()

    if row is not None:
        # Reinforce: blend weight, update evidence, increment counter
        edge_id = row["id"]
        blended_weight = round((float(row["weight"]) + weight) / 2.0, 4)
        await db.execute(
            text("""
                UPDATE knowledge_edges
                SET
                    weight            = :w,
                    evidence          = :ev,
                    observation_count = observation_count + 1,
                    valid_from        = NOW()
                WHERE id = :eid AND user_id = :uid
            """),
            {"w": blended_weight, "ev": evidence, "eid": edge_id, "uid": user_id},
        )
        log.debug(
            "Edge reinforced",
            edge_id=edge_id,
            relation_type=relation_type,
            old_weight=row["weight"],
            new_weight=blended_weight,
        )
        return edge_id, "reinforced"

    # Insert new edge
    edge_id = str(uuid.uuid4())
    await db.execute(
        text("""
            INSERT INTO knowledge_edges
                (id, user_id, source_node_id, target_node_id, relation_type,
                 weight, valid_from, valid_to, episode_id, evidence, observation_count)
            VALUES
                (:id, :uid, :src, :tgt, :rel,
                 :w,  NOW(),   NULL,  :ep,  :ev,  1)
        """),
        {
            "id": edge_id,
            "uid": user_id,
            "src": source_node_id,
            "tgt": target_node_id,
            "rel": relation_type,
            "w": weight,
            "ep": episode_id,
            "ev": evidence,
        },
    )
    log.info(
        "Edge created",
        edge_id=edge_id,
        relation_type=relation_type,
        weight=weight,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
    )
    return edge_id, "created"


# -------------------------------------------------------
# Top-Level Graph Update (called by consolidation_service)
# -------------------------------------------------------

async def update_knowledge_graph(
    db: AsyncSession,
    user_id: str,
    session_summary: str,
    episode_id: Optional[str] = None,
) -> dict:
    """
    Full graph update pipeline for one episode.

    Steps:
        1. Extract (entity, relation, entity) triples via LLM
        2. Batch-embed all entity names in a single OpenAI call
        3. For each entity: resolve_node (merge or insert)
        4. For each triple: upsert_edge
        5. Return stats

    This function is called inside consolidation_service.run_batch()
    INSIDE the same Transaction 2 (inner_db.begin()) block so that
    graph updates and semantic fact upserts are atomically committed
    together. If any step raises, the entire episode's writes roll back.

    Args:
        db              : Async DB session (active transaction from consolidation_service)
        user_id         : Owner UUID
        session_summary : Raw episode summary text
        episode_id      : UUID of the source episode (nullable)

    Returns:
        dict: {triples_extracted, nodes_created, nodes_merged, edges_created, edges_reinforced}
    """
    stats = {
        "triples_extracted": 0,
        "nodes_created": 0,
        "nodes_merged": 0,
        "edges_created": 0,
        "edges_reinforced": 0,
    }

    # Step 1: LLM triple extraction
    triples = await extract_triples(session_summary)
    if not triples:
        log.info("No graph triples extracted", user_id=user_id, episode_id=episode_id)
        return stats

    stats["triples_extracted"] = len(triples)

    # Step 2: Collect all unique entity names and batch-embed them
    entity_map: dict[str, dict] = {}
    for t in triples:
        for side in ("source", "target"):
            ent = t[side]
            key = f"{ent['entity_type']}::{ent['name']}"
            if key not in entity_map:
                entity_map[key] = ent

    entity_keys = list(entity_map.keys())

    # Guard: if entity_map is empty (all triples invalid), skip embedding
    if not entity_keys:
        log.info("No valid entities to embed", user_id=user_id)
        return stats

    entity_names = [entity_map[k]["name"] for k in entity_keys]
    vectors = await embed_batch(entity_names)

    # Build a lookup: key -> embedding vector
    key_to_vector = {k: v for k, v in zip(entity_keys, vectors)}

    # Step 3: Resolve all nodes (merge or insert)
    key_to_node_id: dict[str, str] = {}
    for key in entity_keys:
        ent = entity_map[key]
        vector = key_to_vector[key]
        node_id = await resolve_node(
            db=db,
            user_id=user_id,
            name=ent["name"],
            entity_type=ent["entity_type"],
            embedding_vector=vector,
        )
        key_to_node_id[key] = node_id

    # Determine created vs merged (approximate: new nodes will have mention_count==1)
    # We track this via the resolve_node return path in the logs above.
    # For simplicity in stats, we count triples * 2 entities touched.

    # Step 4: Upsert all edges
    for triple in triples:
        src_key = f"{triple['source']['entity_type']}::{triple['source']['name']}"
        tgt_key = f"{triple['target']['entity_type']}::{triple['target']['name']}"
        src_id = key_to_node_id.get(src_key)
        tgt_id = key_to_node_id.get(tgt_key)

        if not src_id or not tgt_id:
            continue

        edge_id, action = await upsert_edge(
            db=db,
            user_id=user_id,
            source_node_id=src_id,
            target_node_id=tgt_id,
            relation_type=triple["relation"],
            weight=triple["weight"],
            evidence=triple["evidence"],
            episode_id=episode_id,
        )
        if action == "created":
            stats["edges_created"] += 1
        else:
            stats["edges_reinforced"] += 1

    log.info(
        "Knowledge graph updated for episode",
        user_id=user_id,
        episode_id=episode_id,
        **stats,
    )
    return stats
