"""
============================================================
app/services/temporal_graph_engine.py -- Phase 7.5 Temporal Graph Retrieval
============================================================
PURPOSE:
    The read-side of the Temporal Knowledge Graph.
    Called at chat time (in parallel with Layer 1/2/3 retrieval)
    to enrich the LLM prompt with multi-hop relational context.

PERFORMANCE TARGETS (verified by Phase 9 benchmark suite):
    Seed node search (pgvector HNSW)  : < 30 ms
    Graph traversal (Recursive CTE)   : < 50 ms
    Total graph retrieval             : < 100 ms

DESIGN DECISIONS:
    1. Seed search uses pgvector HNSW cosine similarity against
       knowledge_nodes.embedding (same OpenAI model as memory embeddings).
       Top K=3 nodes are returned (configurable via GRAPH_SEED_TOP_K).

    2. Traversal uses PostgreSQL Recursive CTEs (WITH RECURSIVE) which
       execute entirely inside the DB engine at C speed.
       - Depth limited by GRAPH_MAX_DEPTH (default 2)
       - Time-bounded: only edges with valid_from >= NOW() - INTERVAL
       - Weight-filtered: only edges with weight >= GRAPH_MIN_EDGE_WEIGHT
       - Path-capped: LIMIT GRAPH_MAX_PATHS to keep response bounded

    3. No external graph database needed (Neo4j, Memgraph, etc.).
       PostgreSQL handles this workload efficiently for personal-scale
       wellness graphs (< 10,000 nodes / user).

    4. Context formatting converts raw traversal paths into a structured
       natural-language section injected into the LLM system prompt.

CONNECTED TO:
    Phase 7.5 -> app/services/graph_service.py     (write side)
    Phase 7.5 -> app/services/retrieval_engine.py  (caller, parallel gather)
    Phase 7.5 -> schema.sql: knowledge_nodes, knowledge_edges
============================================================
"""

import time
from typing import Optional

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.services.embedding_service import embed_text

log = structlog.get_logger(__name__)
settings = get_settings()


# -------------------------------------------------------
# Step 1: Seed Node Search (< 30 ms target)
# -------------------------------------------------------

async def find_seed_nodes(
    db: AsyncSession,
    user_id: str,
    query_vector: list[float],
) -> list[dict]:
    """
    Finds the top-K knowledge nodes most semantically similar to the
    user's current query using pgvector HNSW cosine search.

    These nodes become the starting points (seeds) for the graph traversal.

    Performance:
        The HNSW index (m=16, ef_construction=64) on knowledge_nodes.embedding
        delivers sub-30ms ANN search at wellness-scale graph sizes.

    Args:
        db           : Async DB session
        user_id      : Owner UUID (user-partitioned search)
        query_vector : 1536-dim embedding of the user message

    Returns:
        List of node dicts with: id, name, entity_type, cos_dist
        Ordered by cosine distance ASC (most similar first).
    """
    t0 = time.perf_counter()

    top_k = settings.GRAPH_SEED_TOP_K
    vec_str = "[" + ",".join(str(v) for v in query_vector) + "]"

    result = await db.execute(
        text("""
            SELECT
                id,
                name,
                entity_type,
                mention_count,
                (embedding <=> :emb::vector) AS cos_dist
            FROM knowledge_nodes
            WHERE user_id = :uid
            ORDER BY cos_dist ASC
            LIMIT :top_k
        """),
        {"emb": vec_str, "uid": user_id, "top_k": top_k},
    )
    rows = result.mappings().all()
    seeds = [dict(r) for r in rows]

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    target_ms = settings.GRAPH_SEED_SEARCH_TARGET_MS

    if elapsed_ms > target_ms:
        log.warning(
            "Seed search exceeded target latency",
            elapsed_ms=elapsed_ms,
            target_ms=target_ms,
            user_id=user_id,
        )
    else:
        log.debug(
            "Seed nodes found",
            count=len(seeds),
            elapsed_ms=elapsed_ms,
            user_id=user_id,
        )

    return seeds


# -------------------------------------------------------
# Step 2: Temporal Graph Traversal (< 50 ms target)
# -------------------------------------------------------

async def traverse_temporal_subgraph(
    db: AsyncSession,
    user_id: str,
    seed_node_ids: list[str],
) -> list[dict]:
    """
    Executes a PostgreSQL Recursive CTE graph walk starting from seed nodes.

    Walk parameters (all from config, configurable via .env):
        GRAPH_MAX_DEPTH       = 2   (friends-of-friends)
        GRAPH_TIME_WINDOW_DAYS = 90  (only edges active within 90 days)
        GRAPH_MIN_EDGE_WEIGHT  = 0.3 (prune weak/noisy edges)
        GRAPH_MAX_PATHS        = 20  (cap result set)

    The CTE structure:
        1. Base case: all edges directly connected to any seed node
        2. Recursive case: walk outward one more hop (up to MAX_DEPTH)
        3. Filter: edge temporal validity, weight threshold, user ownership
        4. LIMIT MAX_PATHS at the query level to bound cost

    Returns:
        List of path dicts, each representing one traversed edge with:
            src_name, src_type, relation_type, tgt_name, tgt_type,
            weight, depth, evidence, valid_from, observation_count

        These are ordered by (depth ASC, weight DESC) so the most
        relevant direct relationships appear first.
    """
    if not seed_node_ids:
        return []

    t0 = time.perf_counter()

    max_depth = settings.GRAPH_MAX_DEPTH
    time_window = settings.GRAPH_TIME_WINDOW_DAYS
    min_weight = settings.GRAPH_MIN_EDGE_WEIGHT
    max_paths = settings.GRAPH_MAX_PATHS

    # Build a PostgreSQL array literal for the seed IDs
    # e.g. ARRAY['uuid1', 'uuid2']::uuid[]
    seed_array = "ARRAY[" + ",".join(f"'{sid}'" for sid in seed_node_ids) + "]::uuid[]"

    traversal_sql = text(f"""
        WITH RECURSIVE graph_walk AS (

            -- ── Base case: direct edges from any seed node ──────────────
            SELECT
                ke.id          AS edge_id,
                ke.source_node_id,
                ke.target_node_id,
                ke.relation_type,
                ke.weight,
                ke.evidence,
                ke.valid_from,
                ke.observation_count,
                src.name       AS src_name,
                src.entity_type AS src_type,
                tgt.name       AS tgt_name,
                tgt.entity_type AS tgt_type,
                1              AS depth,
                ARRAY[ke.source_node_id] AS visited_nodes
            FROM knowledge_edges ke
            JOIN knowledge_nodes src ON src.id = ke.source_node_id
            JOIN knowledge_nodes tgt ON tgt.id = ke.target_node_id
            WHERE
                ke.user_id   = :uid
                AND ke.source_node_id = ANY({seed_array})
                AND ke.valid_to IS NULL
                AND ke.valid_from >= NOW() - INTERVAL '{time_window} days'
                AND ke.weight >= :min_weight

            UNION ALL

            -- ── Recursive case: walk one more hop ───────────────────────
            SELECT
                ke.id,
                ke.source_node_id,
                ke.target_node_id,
                ke.relation_type,
                ke.weight,
                ke.evidence,
                ke.valid_from,
                ke.observation_count,
                src.name,
                src.entity_type,
                tgt.name,
                tgt.entity_type,
                gw.depth + 1,
                gw.visited_nodes || ke.source_node_id
            FROM knowledge_edges ke
            JOIN knowledge_nodes src ON src.id = ke.source_node_id
            JOIN knowledge_nodes tgt ON tgt.id = ke.target_node_id
            JOIN graph_walk gw       ON ke.source_node_id = gw.target_node_id
            WHERE
                ke.user_id   = :uid
                AND gw.depth < :max_depth
                AND ke.valid_to IS NULL
                AND ke.valid_from >= NOW() - INTERVAL '{time_window} days'
                AND ke.weight >= :min_weight
                -- Cycle prevention: do not revisit a node already in this path
                AND NOT (ke.source_node_id = ANY(gw.visited_nodes))
        )

        SELECT DISTINCT ON (src_name, relation_type, tgt_name)
            src_name, src_type,
            relation_type,
            tgt_name, tgt_type,
            weight, depth, evidence, valid_from, observation_count
        FROM graph_walk
        ORDER BY src_name, relation_type, tgt_name, depth ASC, weight DESC
        LIMIT :max_paths
    """)

    try:
        result = await db.execute(
            traversal_sql,
            {
                "uid": user_id,
                "max_depth": max_depth,
                "min_weight": min_weight,
                "max_paths": max_paths,
            },
        )
        rows = result.mappings().all()
        paths = [dict(r) for r in rows]

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        target_ms = settings.GRAPH_TRAVERSAL_TARGET_MS

        if elapsed_ms > target_ms:
            log.warning(
                "Graph traversal exceeded target latency",
                elapsed_ms=elapsed_ms,
                target_ms=target_ms,
                user_id=user_id,
                paths_found=len(paths),
            )
        else:
            log.debug(
                "Graph traversal complete",
                paths_found=len(paths),
                elapsed_ms=elapsed_ms,
                user_id=user_id,
            )

        return paths

    except Exception as e:
        log.error(
            "Graph traversal failed",
            user_id=user_id,
            error=str(e),
        )
        return []


# -------------------------------------------------------
# Step 3: Context Formatter
# -------------------------------------------------------

_RELATION_DESCRIPTIONS = {
    "TRIGGERS":         "has historically triggered",
    "ALLEVIATES":       "helps alleviate",
    "WORSENS":          "tends to worsen",
    "FOLLOWED_BY":      "is typically followed by",
    "SUPERSEDES":       "has replaced / superseded",
    "ASSOCIATED_WITH":  "is often associated with",
    "PART_OF":          "is part of",
    "REDUCED":          "has helped reduce",
}


def format_graph_context(paths: list[dict]) -> str:
    """
    Converts raw traversal paths into a structured natural-language
    section for injection into the LLM system prompt.

    Output example:
        [Causal History]
        - Exam Stress (TRIGGER) has historically triggered Insomnia (SYMPTOM)
          → Observed 3 times | Confidence: 0.82 | Depth: 1
          Evidence: "user reported not sleeping before exam week"
        - Mindfulness Walk (ACTIVITY) helps alleviate Insomnia (SYMPTOM)
          → Observed 2 times | Confidence: 0.75 | Depth: 1
          Evidence: "walking at night helped me calm down"

    Args:
        paths : List of path dicts from traverse_temporal_subgraph()

    Returns:
        Formatted string for LLM prompt injection.
        Returns empty string if paths is empty (so the caller can skip
        the section rather than inserting an empty block).
    """
    if not paths:
        return ""

    lines = ["[Causal & Temporal Knowledge Graph]"]
    # Sort: depth 1 first (direct), then by weight descending
    sorted_paths = sorted(paths, key=lambda p: (p.get("depth", 1), -p.get("weight", 0)))

    for p in sorted_paths:
        src = p.get("src_name", "?")
        src_type = p.get("src_type", "").replace("_", " ").title()
        rel = p.get("relation_type", "?")
        tgt = p.get("tgt_name", "?")
        tgt_type = p.get("tgt_type", "").replace("_", " ").title()
        weight = p.get("weight", 0)
        obs = p.get("observation_count", 1)
        evidence = p.get("evidence", "")
        depth = p.get("depth", 1)
        rel_text = _RELATION_DESCRIPTIONS.get(rel, rel.lower().replace("_", " "))

        line = (
            f"- {src} ({src_type}) {rel_text} {tgt} ({tgt_type})\n"
            f"  Observed: {obs}x | Confidence: {weight:.2f} | Depth: {depth}"
        )
        if evidence:
            line += f'\n  Evidence: "{evidence}"'
        lines.append(line)

    return "\n".join(lines)


# -------------------------------------------------------
# Top-Level Retrieval Entry Point (called by retrieval_engine.py)
# -------------------------------------------------------

async def retrieve_graph_context(
    db: AsyncSession,
    user_id: str,
    query_vector: list[float],
) -> str:
    """
    Full graph retrieval pipeline: seed search + traversal + formatting.

    This function is called by retrieval_engine.run_hybrid_rag_pipeline()
    IN PARALLEL with Layer 1/2/3 fetches via asyncio.gather().

    Total target: < 100 ms (seed < 30 ms + traversal < 50 ms + formatting < 5 ms)

    Args:
        db           : Async DB session
        user_id      : Owner UUID
        query_vector : 1536-dim query embedding

    Returns:
        Formatted context string (empty string if graph is empty or no paths found).
    """
    t_total = time.perf_counter()

    # Step 1: Seed search
    seeds = await find_seed_nodes(db, user_id, query_vector)
    if not seeds:
        log.debug("No seed nodes found; skipping graph traversal", user_id=user_id)
        return ""

    seed_ids = [s["id"] for s in seeds]
    log.info(
        "Graph seed nodes selected",
        seeds=[f"{s['name']} ({s['entity_type']})" for s in seeds],
        user_id=user_id,
    )

    # Step 2: Recursive CTE traversal
    paths = await traverse_temporal_subgraph(db, user_id, seed_ids)
    if not paths:
        log.debug("Graph traversal returned no paths", user_id=user_id)
        return ""

    # Step 3: Format into LLM prompt section
    context_str = format_graph_context(paths)

    total_ms = round((time.perf_counter() - t_total) * 1000, 2)
    target_ms = settings.GRAPH_TOTAL_TARGET_MS

    if total_ms > target_ms:
        log.warning(
            "Total graph retrieval exceeded target latency",
            total_ms=total_ms,
            target_ms=target_ms,
            user_id=user_id,
        )
    else:
        log.info(
            "Graph retrieval complete",
            total_ms=total_ms,
            paths_returned=len(paths),
            user_id=user_id,
        )

    return context_str
