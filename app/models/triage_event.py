"""
============================================================
app/models/triage_event.py — Phase 6 Safety Triage Event ORM Model
============================================================
PURPOSE:
    Maps to the `triage_events` table in Supabase PostgreSQL.
    Each row is a persisted record of a clinical safety screener
    override that occurred during a chat session.

    Records are used for:
    - Regulatory / clinical audit compliance
    - Admin monitoring dashboard (GET /triage endpoints)
    - Nightly archival once older than SAFETY_TRIAGE_RETENTION_DAYS (Phase 7)
    - GDPR Right-to-Forget cascade delete (Phase 8)

CONNECTED TO:
    Phase 6  → Written by triage_service.evaluate_and_store()
    Phase 6  → Queried by app/api/v1/triage.py (admin endpoints)
    Phase 7  → Nightly archival sets archived_at
    Phase 8  → Cascade deleted on user deletion
============================================================
"""
from sqlalchemy import Column, String, DateTime, Float, Boolean, func, Text
from sqlalchemy.dialects.postgresql import JSONB
from app.core.database import Base


class TriageEvent(Base):
    """
    Phase 6 Safety Triage Event — persisted crisis interception record.

    Fields:
        id              : UUID primary key
        user_id         : Owning user (indexed for admin queries)
        session_id      : The Redis session key active at time of crisis
        created_at      : When the crisis was detected (server time)
        crisis_type     : One of: self_harm, eating_disorder, acute_medical
        severity        : "HIGH" — all triage events are currently HIGH severity
        triggered_by    : The regex pattern that matched (audit only, never shown to users)
        resources       : JSONB list of helpline resources shown to the user
        confidence      : Reserved for Phase 6 Layer B/C semantic classifier score
        session_hash    : SHA-256 of (user_id + session_id) for idempotent dedup
                          (only set if SAFETY_SESSION_HASH_ENABLED=True)
        alert_sent      : True if an external alert (email/Slack) was dispatched
        archived_at     : NULL = active; SET = archived (Phase 7 nightly job)
    """
    __tablename__ = "triage_events"

    id = Column(String, primary_key=True, server_default="gen_random_uuid()")
    user_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    crisis_type = Column(String, nullable=False)          # self_harm | eating_disorder | acute_medical
    severity = Column(String, nullable=False, default="HIGH")
    triggered_by = Column(Text, nullable=True)             # Internal audit field — never returned to client
    resources = Column(JSONB, nullable=True)               # Helpline list shown to user
    confidence = Column(Float, nullable=True)              # Reserved for semantic classifier (Phase 6 Layer B)
    session_hash = Column(String, nullable=True, index=True)  # Idempotent dedup hash (optional)
    alert_sent = Column(Boolean, nullable=False, default=False)
    archived_at = Column(DateTime(timezone=True), nullable=True, default=None)
