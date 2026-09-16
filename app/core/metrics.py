"""
============================================================
app/core/metrics.py — Prometheus Custom Application Metrics
============================================================
PURPOSE:
    Defines all custom Prometheus metrics for the AI Wellness LMS.
    These metrics capture fine-grained pipeline latency, LLM token
    usage, and cost data for both production observability and the
    Phase 9 research benchmark.

DESIGN NOTES:
    - Only low-cardinality labels are used (model, operation,
      retrieval_strategy). user_id is deliberately EXCLUDED to
      prevent high-cardinality metric explosion.
    - Histograms use buckets tuned to expected latency distributions:
        Retrieval:  5ms – 2s
        LLM:        50ms – 30s
    - All metric names follow the Prometheus naming convention:
        <namespace>_<subsystem>_<unit>

CONNECTED TO:
    Phase 9.1 → main.py exposes /metrics endpoint
    Phase 9.1 → retrieval_engine.py records latency + token counters
    Phase 9.1 → cost_service.py records per-operation token counts
    Phase 9.2 → benchmark scripts read these metrics via /metrics
============================================================
"""

from prometheus_client import Counter, Histogram


# -------------------------------------------------------
# Latency Histograms
# Buckets cover: 5ms, 10ms, 25ms, 50ms, 100ms, 250ms,
#                500ms, 1s, 2.5s, 5s, 10s, 30s
# -------------------------------------------------------

# Full end-to-end retrieval pipeline (embed + all layers + filter)
RAG_RETRIEVAL_LATENCY = Histogram(
    name="rag_retrieval_latency_seconds",
    documentation=(
        "End-to-end latency of the hybrid RAG retrieval pipeline "
        "(embedding + Layer1/2/3 fetch + time-decay filter). "
        "Phase 9 benchmark target: p95 < 500ms."
    ),
    labelnames=["retrieval_strategy"],   # e.g. "hybrid", "vector_only", "graph_only"
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# pgvector HNSW approximate nearest-neighbor search (Layer 3)
VECTOR_SEARCH_LATENCY = Histogram(
    name="vector_search_latency_seconds",
    documentation=(
        "Latency of the pgvector HNSW cosine-similarity search "
        "against semantic_memories. Phase 9 target: p95 < 50ms."
    ),
    labelnames=["retrieval_strategy"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# Temporal Knowledge Graph seed search + recursive CTE traversal
GRAPH_TRAVERSAL_LATENCY = Histogram(
    name="graph_traversal_latency_seconds",
    documentation=(
        "Latency of the temporal knowledge graph retrieval "
        "(seed HNSW search + recursive CTE edge traversal). "
        "Phase 9 target: p95 < 100ms."
    ),
    labelnames=["retrieval_strategy"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# Temporal validity filter applied after retrieval
TEMPORAL_FILTER_LATENCY = Histogram(
    name="temporal_filter_latency_seconds",
    documentation=(
        "Latency of the time-decay scoring and threshold filtering "
        "step applied to raw semantic memory candidates."
    ),
    labelnames=["retrieval_strategy"],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

# OpenAI / Gemini LLM generation round-trip
LLM_GENERATION_LATENCY = Histogram(
    name="llm_generation_latency_seconds",
    documentation=(
        "Latency of the LLM chat completion call "
        "(from request dispatch to first byte of response received)."
    ),
    labelnames=["model", "operation"],
    # LLM calls are slower — shift buckets right
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0),
)


# -------------------------------------------------------
# Token Usage Counter
# -------------------------------------------------------

LLM_TOKENS_USED_TOTAL = Counter(
    name="llm_tokens_used_total",
    documentation=(
        "Cumulative LLM token consumption tracked by token type. "
        "Used to estimate API cost and enforce per-user token budgets. "
        "Labels: model (e.g. gemini-2.5-flash), operation (CHAT | "
        "CONSOLIDATION | EMBEDDING | GRAPH_EXTRACTION | EVALUATION), "
        "token_type (prompt | completion | total)."
    ),
    labelnames=["model", "operation", "token_type"],
)


# -------------------------------------------------------
# Request-level Counters
# -------------------------------------------------------

RAG_PIPELINE_REQUESTS_TOTAL = Counter(
    name="rag_pipeline_requests_total",
    documentation=(
        "Total number of hybrid RAG pipeline executions. "
        "Labelled by retrieval_strategy and outcome "
        "(success | error | budget_exceeded)."
    ),
    labelnames=["retrieval_strategy", "outcome"],
)

TOKEN_BUDGET_EXCEEDED_TOTAL = Counter(
    name="token_budget_exceeded_total",
    documentation=(
        "Number of LLM requests rejected because the user's "
        "cumulative token budget was exhausted."
    ),
    labelnames=["operation"],
)


# -------------------------------------------------------
# Consolidation Metrics
# -------------------------------------------------------

CONSOLIDATION_RUNS_TOTAL = Counter(
    name="consolidation_runs_total",
    documentation=(
        "Total nightly consolidation pipeline executions. "
        "Labels: status (success | failed | skipped)."
    ),
    labelnames=["status"],
)

CONSOLIDATION_MEMORIES_CREATED_TOTAL = Counter(
    name="consolidation_memories_created_total",
    documentation="Total new semantic memory records created by the consolidation pipeline.",
    labelnames=[],
)

CONSOLIDATION_MEMORIES_REINFORCED_TOTAL = Counter(
    name="consolidation_memories_reinforced_total",
    documentation=(
        "Total existing semantic memories reinforced (deduplication "
        "hit instead of insert) by the consolidation pipeline."
    ),
    labelnames=[],
)


# -------------------------------------------------------
# Safety / Triage Metrics
# -------------------------------------------------------

CRISIS_INTERCEPT_TOTAL = Counter(
    name="crisis_intercept_total",
    documentation=(
        "Number of chat requests intercepted by the clinical safety "
        "screener before reaching the LLM. Labels: crisis_type."
    ),
    labelnames=["crisis_type"],
)
