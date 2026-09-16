"""
============================================================
app/models/user.py — User Profile SQLAlchemy Model
============================================================
"""
from sqlalchemy import Column, String, DateTime, func, Boolean, Integer
from sqlalchemy.dialects.postgresql import JSONB
from app.core.database import Base


class User(Base):
    """
    Maps to the `users` table in Supabase PostgreSQL.

    Fields:
        user_id          : Unique user identifier (primary key)
        email            : Unique email address
        password_hash    : Bcrypt-hashed password (Phase 2)
        created_at       : Account creation timestamp
        is_admin         : Admin flag (Phase 6)
        baseline_profile : JSONB blob storing evolving health baseline.
                           Updated by Phase 7 consolidation pipeline
                           whenever a 'baseline' category memory is
                           extracted and confirmed.
        token_budget     : Maximum cumulative tokens allowed (0 = unlimited).
                           Set per user by admin for cost control.
                           Phase 9.1: enforced by cost_service.check_budget()
        tokens_used      : Running total of tokens consumed across all
                           LLM operations. Updated by cost_service.record_usage()
                           after every successful API call.
    """
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, index=True)
    email = Column(String, nullable=False, unique=True)
    password_hash = Column(String, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_admin = Column(Boolean, nullable=False, default=False)
    baseline_profile = Column(
        JSONB,
        nullable=False,
        default={
            "averageSleepHours": None,
            "knownTriggers": [],
            "effectiveCopingMechanisms": [],
            "dataRetentionDays": 365,
            "allowBiometrics": False,
        }
    )
    # Phase 9.1: Token budget enforcement
    # token_budget = 0 means unlimited (no cap enforced).
    # Values > 0 are checked before each LLM call in cost_service.py.
    token_budget = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Maximum cumulative token budget (0 = unlimited)",
    )
    tokens_used = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Cumulative tokens consumed across all LLM operations",
    )
