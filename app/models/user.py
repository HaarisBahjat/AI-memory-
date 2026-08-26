"""
============================================================
app/models/user.py — User Profile SQLAlchemy Model
============================================================
"""
from sqlalchemy import Column, String, DateTime, func, Boolean
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
        baseline_profile : JSONB blob storing evolving health
                           baseline (avg sleep, known triggers,
                           effective coping mechanisms).
                           Updated by Phase 7 consolidation pipeline
                           whenever a 'baseline' category memory
                           is extracted and confirmed.
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
