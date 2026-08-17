"""
============================================================
main.py — FastAPI Application Entry Point
============================================================
PURPOSE:
    The root of the FastAPI application. Handles:
    - Application lifecycle (startup/shutdown events)
    - Redis initialization and cleanup
    - Route registration
    - CORS, global middleware
    - Health check endpoint
    - Swagger UI configuration

HOW IT CONNECTS EVERYTHING:
    main.py is the assembly point where all Phase 1 components
    are wired together:
    - Imports and registers api_router (chat + user endpoints)
    - Starts Redis pool on startup (Layer 1 Sensory Memory)
    - Provides /api/v1/health for infrastructure monitoring
    - Phase 7: Celery worker will be initialized here
    - Phase 9: Prometheus metrics exporter registered here

RUNNING LOCALLY:
    1. Copy .env.example to .env and fill in Supabase + OpenAI keys
    2. Run: uvicorn main:app --reload --port 8000
    3. Open: http://localhost:8000/docs for Swagger UI
============================================================
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.core.config import get_settings
from app.core.database import ping_database
from app.core.redis_client import init_redis, close_redis, ping_redis
from app.api.v1.router import api_router
from app.schemas.chat import HealthResponse

log = structlog.get_logger(__name__)
settings = get_settings()


# -------------------------------------------------------
# Application Lifecycle Manager
# -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Code before `yield` runs at startup.
    Code after `yield` runs at shutdown.

    Startup:
    - Initializes Redis connection pool (Layer 1 Sensory Memory)
    - Phase 7: Will initialize Celery beat scheduler here
    - Phase 9: Will initialize Prometheus metrics here

    Shutdown:
    - Gracefully closes Redis pool
    - Drains in-flight requests before exit
    """
    log.info(
        "Starting AI Wellness LMS",
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
    )

    # Initialize Redis (Layer 1 Sensory Memory)
    await init_redis()
    log.info("Layer 1 Sensory Memory (Redis) initialized")

    yield  # Application runs here

    # Cleanup on shutdown
    await close_redis()
    log.info("AI Wellness LMS shutdown complete")


# -------------------------------------------------------
# FastAPI Application Instance
# -------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    description="""
## AI Wellness Longitudinal Memory System (LMS)

A stateful, token-efficient, privacy-preserving AI wellness agent 
capable of maintaining years of continuous context across three memory layers:

- **Layer 1**: Redis Sensory Memory (active session, 30-minute TTL)
- **Layer 2**: Supabase Episodic Memory (14-day health timeline)  
- **Layer 3**: Supabase pgvector Semantic Memory (multi-year compressed facts)

### Authentication (Phase 2)

All chat and user endpoints require a **Bearer token** in the `Authorization` header.

1. Register: `POST /api/v1/auth/register` — create account, receive tokens
2. Login: `POST /api/v1/auth/login` — authenticate, receive tokens  
3. Refresh: `POST /api/v1/auth/refresh` — rotate expiring access token
4. Logout: `POST /api/v1/auth/logout` — revoke refresh token

Click **Authorize** above and paste your `access_token` to test protected endpoints here.

### Key Endpoints
- `POST /api/v1/auth/register` — Create account
- `POST /api/v1/auth/login` — Login and receive JWT tokens
- `POST /api/v1/chat` — Send a wellness message (JWT required)
- `GET /api/v1/user/me` — View your profile (JWT required)
- `PUT /api/v1/user/me/baseline` — Update health baseline (JWT required)
- `DELETE /api/v1/user/me/memory` — GDPR Right-to-Forget (JWT required)
- `GET /api/v1/health` — Infrastructure health check
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# -------------------------------------------------------
# CORS Middleware
# -------------------------------------------------------
# In production (Phase 8), restrict origins to your actual
# frontend domain. allow_origins=["*"] is for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["https://your-app-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
# Health Check Endpoint
# -------------------------------------------------------
@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Infrastructure health check",
    description="Returns connectivity status of Supabase PostgreSQL and Redis.",
)
async def health_check():
    """
    Verifies connectivity to all critical infrastructure components.
    Used by:
    - Deployment platforms (Railway, Render) for readiness checks
    - Phase 9 Grafana dashboards for uptime monitoring
    - Automated health alerts
    """
    db_ok = await ping_database()
    redis_ok = await ping_redis()

    return HealthResponse(
        status="healthy" if (db_ok and redis_ok) else "degraded",
        version=settings.APP_VERSION,
        database=db_ok,
        redis=redis_ok,
        timestamp=datetime.now(timezone.utc),
    )


# -------------------------------------------------------
# Register All API Routes
# -------------------------------------------------------
app.include_router(api_router)


# -------------------------------------------------------
# Root Redirect
# -------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "AI Wellness LMS is running",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
