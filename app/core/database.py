"""
============================================================
app/core/database.py — Async SQLAlchemy Engine (Supabase PostgreSQL)
============================================================
PURPOSE:
    Manages the async database connection pool to Supabase
    PostgreSQL. Provides:
    - A reusable async engine (pool) pointing at Supabase
    - An async session factory (AsyncSessionLocal) used by
      all service functions to query/mutate data
    - A FastAPI dependency (get_db) that yields a session
      per HTTP request and auto-commits or rolls back

HOW SUPABASE CONNECTION WORKS:
    Supabase gives you a PostgreSQL connection string:
    postgresql+asyncpg://postgres:[PASSWORD]@db.[REF].supabase.co:5432/postgres

    asyncpg is the fastest async PostgreSQL driver for Python.
    SQLAlchemy 2.0 wraps asyncpg to give us a clean ORM layer.

CONNECTED TO:
    Phase 1  → All model queries (users, episodes, semantic_memories)
    Phase 2  → User registration / auth DB writes
    Phase 5  → Episode insert after chat session ends
    Phase 7  → Nightly consolidation reads Layer 2 episodes
    Phase 8  → Cascading atomic DELETE across all tables
    Phase 10 → Biometric stream table reads/writes
============================================================
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from typing import AsyncGenerator
import structlog

from app.core.config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()


# -------------------------------------------------------
# Async Engine
# -------------------------------------------------------
# pool_size    → Number of persistent connections maintained.
#                Supabase free tier allows ~60 connections;
#                we use 5 to stay well within limits.
# max_overflow → Extra connections allowed during traffic spikes.
# pool_pre_ping→ Validates connections before use, preventing
#                "connection closed" errors after idle periods.
# echo         → Logs all SQL statements when DEBUG=True.
#                Disable in production (performance overhead).
# -------------------------------------------------------
engine = create_async_engine(
    settings.SUPABASE_DB_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    echo=settings.DEBUG,  # Set DEBUG=false in production
)


# -------------------------------------------------------
# Async Session Factory
# -------------------------------------------------------
# expire_on_commit=False → Prevents SQLAlchemy from expiring
# model instances after commit, which would cause lazy-load
# errors in async contexts (since there's no sync I/O fallback).
# -------------------------------------------------------
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# -------------------------------------------------------
# Declarative Base
# -------------------------------------------------------
# All SQLAlchemy models (User, Episode, SemanticMemory,
# BiometricsStream) inherit from this Base class.
# It gives them the __tablename__ and Column() machinery.
# -------------------------------------------------------
class Base(DeclarativeBase):
    """
    Shared declarative base for all SQLAlchemy ORM models.

    All model files in app/models/ inherit from this:
        class User(Base):
            __tablename__ = "users"
            ...
    """
    pass


# -------------------------------------------------------
# FastAPI Dependency: get_db()
# -------------------------------------------------------
# This is a FastAPI dependency injected into route handlers
# via Depends(get_db). It:
#   1. Opens a fresh async session from the connection pool
#   2. Yields it to the route handler
#   3. Auto-commits on success
#   4. Rolls back on any exception (data integrity protection)
#   5. Closes the session and returns the connection to the pool
#
# Usage in a route:
#   async def create_user(db: AsyncSession = Depends(get_db)):
#       db.add(user)
#       await db.commit()
# -------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a managed async database session.
    Auto-handles commit/rollback/close per request lifecycle.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            log.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()


# -------------------------------------------------------
# Health Check Utility (used by GET /api/v1/health)
# -------------------------------------------------------
async def ping_database() -> bool:
    """
    Executes a lightweight SELECT 1 to verify Supabase
    PostgreSQL connectivity. Used by the /health endpoint
    to report database status to monitoring systems.
    Returns True if reachable, False on error.
    """
    try:
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.error("Database health check failed", error=str(e))
        return False
