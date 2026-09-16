"""
============================================================
app/models/llm_usage_event.py — LLM Usage Events SQLAlchemy Model
============================================================
PURPOSE:
    Maps to the `llm_usage_events` table which records every
    LLM API call made by the system. Supports:
    - Per-user token budget enforcement (Phase 9.1)
    - Production cost monitoring (API spend tracking)
    - Research cost evaluation (Phase 9 benchmarking)

OPERATIONS:
    CHAT            → run_hybrid_rag_pipeline() LLM completion
    CONSOLIDATION   → nightly episode-to-semantic consolidation
    EMBEDDING       → text-embedding-3-small / gemini-embedding calls
    GRAPH_EXTRACTION→ LLM-powered knowledge graph entity/edge extraction
    EVALUATION      → LLM-as-judge scoring during benchmark runs

CONNECTED TO:
    Phase 9.1 → cost_service.py (record_usage, check_budget)
    Phase 9.1 → retrieval_engine.py (records CHAT events)
    Phase 9.2 → answer_benchmark.py (records EVALUATION events)
    schema.sql → CREATE TABLE llm_usage_events
============================================================
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, func, Text
from app.core.database import Base


class LLMUsageEvent(Base):
    """
    Records a single LLM API call with token counts and cost estimate.

    Fields:
        id               : UUID primary key
        user_id          : Owning user (not FK-enforced — allows system-level ops)
        operation        : CHAT | CONSOLIDATION | EMBEDDING |
                           GRAPH_EXTRACTION | EVALUATION
        model            : Model name used (e.g. gemini-2.5-flash)
        prompt_tokens    : Input tokens consumed
        completion_tokens: Output tokens consumed (0 for embeddings)
        total_tokens     : prompt_tokens + completion_tokens
        estimated_api_cost: Estimated USD cost (model-specific rate card)
        created_at       : Timestamp of the API call
        session_id       : Optional session identifier for correlation
        metadata_        : JSONB blob for extra context (e.g. memory_count,
                           retrieval_strategy, query_type)
    """
    __tablename__ = "llm_usage_events"

    id = Column(
        String,
        primary_key=True,
    )
    user_id = Column(
        String,
        nullable=False,
        index=True,
        comment="User who triggered this LLM call (not FK to allow system ops)",
    )
    operation = Column(
        String,
        nullable=False,
        comment="CHAT | CONSOLIDATION | EMBEDDING | GRAPH_EXTRACTION | EVALUATION",
    )
    model = Column(
        String,
        nullable=False,
        comment="Model name used for this call",
    )
    prompt_tokens = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Input (prompt) tokens consumed",
    )
    completion_tokens = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Output (completion) tokens consumed (0 for embeddings)",
    )
    total_tokens = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total tokens = prompt_tokens + completion_tokens",
    )
    estimated_api_cost = Column(
        Float,
        nullable=False,
        default=0.0,
        comment="Estimated USD cost for this API call based on model rate card",
    )
    session_id = Column(
        String,
        nullable=True,
        comment="Optional Redis session key for tracing CHAT calls",
    )
    extra_metadata = Column(
        Text,
        nullable=True,
        comment="JSON string: retrieval_strategy, memory_count, query_type, etc.",
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the LLM call was made",
    )
