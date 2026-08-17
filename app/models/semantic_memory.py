"""
============================================================
app/models/semantic_memory.py — Layer 3 Semantic Memory (pgvector)
============================================================
PURPOSE:
    Maps to the `semantic_memories` table in Supabase PostgreSQL.
    Each row stores a long-term compressed health fact (trigger,
    coping mechanism, baseline, symptom, or milestone) alongside
    its 1536-dimensional embedding vector.

    The HNSW index on `embedding` enables approximate nearest-
    neighbor cosine similarity search via:
        embedding <-> :query_vector
    which is what the retrieval engine uses in Phase 1.

TIME-DECAY SCORING:
    The `created_at` timestamp is used in the retrieval engine to
    compute how many days have passed since this memory was created:
        delta_days = (now - created_at).days
        S_adjusted = S_raw * exp(-0.005 * delta_days)

    In Phase 7, when a consolidation run finds a nearly-duplicate
    memory (cosine similarity > 0.88), instead of inserting a new
    row, it:
    1. Increments `reinforcement_count += 1`
    2. Resets `created_at` to NOW() (refreshing the decay clock)

CONNECTED TO:
    Phase 1  → Vector similarity search in retrieval_engine.py
    Phase 4  → Full CRUD operations added (upsert, delete)
    Phase 7  → Consolidation upserts new facts; dedup increments count
    Phase 8  → Cascade deleted on Right-to-Forget
============================================================
"""
from sqlalchemy import Column, String, Integer, DateTime, Text, func
from pgvector.sqlalchemy import Vector
from app.core.database import Base


class SemanticMemory(Base):
    """
    Layer 3 Semantic Memory — long-term compressed health fact.

    Fields:
        id                  : UUID primary key
        user_id             : Indexed for user-scoped vector queries
        category            : Fact type — trigger | baseline |
                              coping_mechanism | symptom | milestone
        text                : Human-readable fact text
                              (stored alongside vector for LLM injection)
        embedding           : 1536-dimensional float vector
                              (OpenAI text-embedding-3-small output)
        reinforcement_count : How many times this fact has been
                              re-confirmed by consolidation runs
        created_at          : Timestamp used in time-decay formula;
                              reset when reinforced (Phase 7)
    """
    __tablename__ = "semantic_memories"

    id = Column(String, primary_key=True, server_default="gen_random_uuid()")
    user_id = Column(String, nullable=False, index=True)
    category = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536))     # pgvector 1536-dim vector column
    reinforcement_count = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
