"""
============================================================
app/core/redis_client.py — Redis Connection Pool Manager
============================================================
PURPOSE:
    Manages the async Redis connection to Upstash Redis
    (or local Redis in development). Provides:
    - A global shared connection pool (created once at startup)
    - get_redis() FastAPI dependency for per-request access
    - ping_redis() health check for /api/v1/health endpoint

WHY REDIS FOR LAYER 1 SENSORY MEMORY?
    Redis stores the active conversational buffer for each user.
    Key format: sensory:{user_id}:session
    Value: JSON array of {role, content, timestamp} message objects

    The 30-minute sliding TTL means:
    - If the user is chatting, each message resets the timer
    - If they go idle for >30 minutes, the session auto-expires
    - Phase 7 hooks into the TTL expiry to trigger consolidation

CONNECTED TO:
    Phase 1  → Layer 1 sensory_service.py reads/writes session data
    Phase 5  → After each LLM response, message appended to session
    Phase 7  → Celery worker uses Redis as message broker
    Phase 8  → DEL sensory:{userId}:session on Right-to-Forget
============================================================
"""

import redis.asyncio as aioredis
from typing import AsyncGenerator
import structlog

from app.core.config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()

# -------------------------------------------------------
# Global Redis Connection Pool
# -------------------------------------------------------
# Created once at application startup (see main.py lifespan).
# decode_responses=True → automatically converts bytes to str,
# so we work with Python strings not bytes objects.
# max_connections=20    → Pool ceiling; prevents connection
# exhaustion under concurrent user loads.
# -------------------------------------------------------
_redis_pool: aioredis.Redis | None = None


async def init_redis() -> None:
    """
    Initialize the global Redis connection pool.
    Called once during FastAPI startup lifespan event.

    Example (main.py):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await init_redis()
            yield
            await close_redis()
    """
    global _redis_pool
    try:
        _redis_pool = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,  # Strings not bytes
            max_connections=20,
            socket_connect_timeout=5,   # Fail fast if Redis is down
            socket_timeout=5,
        )
        # Validate connection immediately
        await _redis_pool.ping()
        log.info("Redis connection established", url=settings.REDIS_URL)
    except Exception as e:
        log.error("Redis connection failed", error=str(e))
        raise


async def close_redis() -> None:
    """
    Gracefully close the Redis connection pool.
    Called during FastAPI shutdown lifespan event.
    Ensures all in-flight commands complete before closing.
    """
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        log.info("Redis connection pool closed")


def get_redis_client() -> aioredis.Redis:
    """
    Returns the global shared Redis client.
    Use this in service files (sensory_service.py) that
    aren't using FastAPI dependency injection.

    Raises RuntimeError if init_redis() hasn't been called yet.
    """
    if _redis_pool is None:
        raise RuntimeError(
            "Redis pool not initialized. "
            "Ensure init_redis() is called during app startup."
        )
    return _redis_pool


async def get_redis() -> AsyncGenerator[aioredis.Redis, None]:
    """
    FastAPI dependency that yields the shared Redis client.
    The connection is NOT closed per-request (it's shared pool).

    Usage in a route handler:
        async def chat(redis: aioredis.Redis = Depends(get_redis)):
            session = await redis.get(f"sensory:{user_id}:session")
    """
    yield get_redis_client()


# -------------------------------------------------------
# Health Check Utility
# -------------------------------------------------------
async def ping_redis() -> bool:
    """
    Sends a PING command to Redis and returns True if online.
    Used by GET /api/v1/health to report Redis status.
    """
    try:
        client = get_redis_client()
        result = await client.ping()
        return result is True
    except Exception as e:
        log.error("Redis health check failed", error=str(e))
        return False
