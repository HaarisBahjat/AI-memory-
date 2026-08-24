"""
============================================================
app/models/episode.py — Layer 2 Episodic Memory SQLAlchemy Model
============================================================
PURPOSE:
    Maps to the `episodes` table in Supabase PostgreSQL.
    Each row is a structured daily/session health summary
    produced after a chat session ends.

    The JSONB extracted_metrics field stores flexible health
    data that evolves across phases:
    - Phase 1: mood_score, primary_stressor, sleep_hours
    - Phase 10: adds biometrics{} nested object from wearables

CONNECTED TO:
    Phase 1  → Queried in Hybrid RAG (Layer 2 context fetch)
    Phase 5  → Inserted after each session terminates
    Phase 7  → Source for nightly LLM memory consolidation
    Phase 8  → Cascade deleted on Right-to-Forget
    Phase 10 → extracted_metrics.biometrics enriched with HRV/HR
============================================================
"""
from sqlalchemy import Column, String, DateTime, Text, func, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from app.core.database import Base


class Episode(Base):
    """
    Layer 2 Episodic Memory — structured daily health summary.

    Fields:
        id                : UUID primary key
        user_id           : Foreign key to users.user_id (indexed)
        timestamp         : When this episode was recorded
        session_summary   : Free-text narrative summary of the session
        extracted_metrics : JSONB health metrics blob:
                            {
                              "moodScore": 6,
                              "physicalSymptoms": ["headache"],
                              "primaryStressor": "exam anxiety",
                              "sleepHoursLogged": 6.5,
                              "anxietyLevel": 7,
                              "energyLevel": 4,
                              "biometrics": {}  ← Phase 10 fills this
                            }
        archived_at       : NULL = active; SET = cold storage (Phase 7)
        embedding         : 1536-dim vector of session_summary for semantic search (Phase 5)
    """
    __tablename__ = "episodes"

    id = Column(String, primary_key=True, server_default="gen_random_uuid()")
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    session_summary = Column(Text, nullable=False)
    extracted_metrics = Column(
        JSONB,
        nullable=False,
        default={
            "moodScore": None,
            "physicalSymptoms": [],
            "primaryStressor": None,
            "sleepHoursLogged": None,
            "anxietyLevel": None,
            "energyLevel": None,
            "biometrics": {},
        }
    )
    embedding = Column(Vector(1536), nullable=True)   # Phase 5: 1536-dim summary vector
    archived_at = Column(DateTime(timezone=True), nullable=True, default=None)
    # Phase 7: Batch consolidation status
    # PENDING → PROCESSING → CONSOLIDATED | FAILED
    # FOR UPDATE SKIP LOCKED is used to atomically claim rows without race conditions.
    consolidation_status = Column(String, nullable=False, default="PENDING", index=True)
