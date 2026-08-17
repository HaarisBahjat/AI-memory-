"""
============================================================
app/models/refresh_token.py — RefreshToken SQLAlchemy Model (Phase 2)
============================================================
PURPOSE:
    ORM representation of the `refresh_tokens` table.
    Maps Python class attributes to the DB columns defined
    in schema.sql Phase 2 section.

    This model is used by the auth service for:
    - Inserting a new hashed token on login
    - Querying by token_hash on refresh
    - Marking revoked=TRUE on logout
    - Cascade-deleting via the FK when the user is deleted

CONNECTED TO:
    Phase 2 → app/api/v1/auth.py (all token lifecycle ops)
    Phase 8 → ON DELETE CASCADE from users.user_id handles purge
============================================================
"""
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class RefreshToken(Base):
    """
    Maps to the `refresh_tokens` table in Supabase PostgreSQL.

    Fields:
        id          : UUID primary key (server-generated)
        user_id     : FK → users.user_id (ON DELETE CASCADE)
        token_hash  : SHA-256 of the raw refresh token (never store raw)
        expires_at  : Hard expiry — tokens older than this are invalid
        revoked     : Soft-revoke flag (set TRUE on logout)
        created_at  : Timestamp of token issuance
    """
    __tablename__ = "refresh_tokens"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id = Column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash = Column(String, nullable=False, unique=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, nullable=False, default=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
