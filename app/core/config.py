"""
============================================================
app/core/config.py — Application Settings & Environment Manager
============================================================
PURPOSE:
    Single source of truth for all configuration values.
    Uses Pydantic BaseSettings to automatically:
    - Read values from the .env file
    - Validate types and required fields at startup
    - Expose them as typed Python attributes throughout the app

CONNECTED TO:
    Phase 1  → database.py reads SUPABASE_DB_URL
    Phase 1  → redis_client.py reads REDIS_URL, REDIS_SESSION_TTL
    Phase 1  → retrieval_engine.py reads DECAY_LAMBDA, SIMILARITY_THRESHOLD
    Phase 7  → consolidation pipeline reads CRON_CONSOLIDATION_SCHEDULE
    Phase 7  → dedup engine reads DEDUP_THRESHOLD
    Phase 9  → benchmarking reads DECAY_LAMBDA for precision tests
    Phase 10 → biometric ingestion reads OURA_API_KEY, FITBIT_CLIENT_ID
============================================================
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Pydantic BaseSettings class — all fields map 1:1 to .env variables.
    Types are enforced at startup (e.g., DECAY_LAMBDA must be a float).
    If a required field is missing from .env, the app fails fast with a
    clear ValidationError before accepting any traffic.
    """

    # -------------------------------------------------------
    # App Core
    # -------------------------------------------------------
    APP_NAME: str = "AI Wellness LMS"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    PORT: int = 8000

    # -------------------------------------------------------
    # Supabase PostgreSQL
    # SQLAlchemy async engine requires the postgresql+asyncpg:// scheme.
    # -------------------------------------------------------
    SUPABASE_DB_URL: str
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # -------------------------------------------------------
    # Redis — Layer 1 Sensory Memory
    # Also used as the Celery task queue broker in Phase 7.
    # -------------------------------------------------------
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_SESSION_TTL: int = 1800        # 30 minutes
    REDIS_MAX_MESSAGES: int = 10         # Rolling window buffer size

    # -------------------------------------------------------
    # OpenAI
    # -------------------------------------------------------
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_CHAT_MODEL_FALLBACK: str = ""   # If set, retried on 429 rate-limit errors
    OPENAI_CHAT_TEMPERATURE: float = 0.3

    # -------------------------------------------------------
    # Phase 5: Episode Synthesis
    # -------------------------------------------------------
    EPISODE_SYNTH_TRANSCRIPT_MAX_CHARS: int = 12000  # Truncate long sessions before LLM call
    EPISODE_SYNTH_MAX_RETRIES: int = 2               # LLM retry attempts before fallback

    # -------------------------------------------------------
    # Memory Retrieval Math
    # These drive the time-decay scoring formula:
    #   S_adjusted = S_raw * exp(-DECAY_LAMBDA * delta_days)
    # -------------------------------------------------------
    DECAY_LAMBDA: float = 0.005          # ~0.5% decay per day
    SIMILARITY_THRESHOLD: float = 0.65   # Minimum score to include in LLM context
    DEDUP_THRESHOLD: float = 0.88        # Legacy alias — prefer SEMANTIC_MEMORY_MAX_COSINE_DISTANCE
    # Phase 7 deduplication — pgvector <=> returns cosine DISTANCE (lower = more similar).
    # A cosine similarity of 0.88 corresponds to a cosine distance of 0.12.
    # So:  distance <= MAX_COSINE_DISTANCE  ↔  similarity >= 0.88
    # Configurable via .env without code change.
    SEMANTIC_MEMORY_MAX_COSINE_DISTANCE: float = 0.12  # 1 - 0.88
    CONSOLIDATION_BATCH_SIZE: int = 100  # Max episodes per consolidation run
    EPISODIC_ACTIVE_DAYS: int = 14       # Layer 2 retrieval window
    EPISODIC_ARCHIVE_DAYS: int = 90      # Phase 7: cold storage threshold

    # -------------------------------------------------------
    # Temporal GraphRAG (Phase 7.5 — Graph Engine)
    # -------------------------------------------------------
    # Entry-point vector search (how many entity nodes to seed traversal)
    GRAPH_SEED_TOP_K: int = 3             # top-k seed nodes via pgvector similarity
    # Recursive CTE traversal depth (1 = direct neighbours only, 2 = friends-of-friends)
    GRAPH_MAX_DEPTH: int = 2
    # Only traverse edges whose valid_from is within this window
    GRAPH_TIME_WINDOW_DAYS: int = 90      # matches episodic archive horizon
    # Max paths returned from the traversal before ranking/pruning
    GRAPH_MAX_PATHS: int = 20
    # Minimum edge weight to follow during traversal (prunes noisy/weak edges)
    GRAPH_MIN_EDGE_WEIGHT: float = 0.3
    # Minimum node-level similarity for entity resolution (merge near-duplicate entities)
    GRAPH_ENTITY_MERGE_DISTANCE: float = 0.10   # cosine distance <= 0.10 (sim >= 0.90)
    # Performance targets (ms) — used by Phase 9 benchmark suite for regression alerts
    GRAPH_SEED_SEARCH_TARGET_MS: int = 30
    GRAPH_TRAVERSAL_TARGET_MS: int = 50
    GRAPH_TOTAL_TARGET_MS: int = 100

    # -------------------------------------------------------
    # Security & Auth (Phase 2)
    # -------------------------------------------------------
    JWT_SECRET: str = "change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # -------------------------------------------------------
    # Clinical Safety
    # -------------------------------------------------------
    ENABLE_SAFETY_SCREENER: bool = True

    # -------------------------------------------------------
    # Phase 6: Safety Triage Persistence & Alerting
    # -------------------------------------------------------
    SAFETY_TRIAGE_RETENTION_DAYS: int = 365          # How long triage_events are kept before archival
    SAFETY_ALERT_CHANNEL: str = "NONE"               # "NONE" | "EMAIL" | "SLACK"
    SAFETY_ALERT_RATE_LIMIT_MIN: int = 15            # 1 alert per user per N minutes (Redis-backed)
    SAFETY_ALERT_REDIS_PREFIX: str = "safety:triage:alert"  # Redis key prefix for rate limiting
    SAFETY_SESSION_HASH_ENABLED: bool = False         # Enable session_hash dedup on triage_events
    # Email alert settings (used when SAFETY_ALERT_CHANNEL="EMAIL")
    SAFETY_ALERT_EMAIL_FROM: str = ""
    SAFETY_ALERT_EMAIL_TO: str = ""
    SAFETY_ALERT_SMTP_HOST: str = "smtp.gmail.com"
    SAFETY_ALERT_SMTP_PORT: int = 587
    SAFETY_ALERT_SMTP_USER: str = ""
    SAFETY_ALERT_SMTP_PASSWORD: str = ""
    # Slack alert settings (used when SAFETY_ALERT_CHANNEL="SLACK")
    SAFETY_ALERT_SLACK_WEBHOOK_URL: str = ""

    # -------------------------------------------------------
    # Celery / Background Jobs (Phase 7)
    # -------------------------------------------------------
    CRON_CONSOLIDATION_SCHEDULE: str = "0 2 * * *"
    CONSOLIDATION_EPISODE_WINDOW_DAYS: int = 7

    # -------------------------------------------------------
    # Phase 10 Wearables
    # -------------------------------------------------------
    OURA_API_KEY: str = ""
    FITBIT_CLIENT_ID: str = ""
    FITBIT_CLIENT_SECRET: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",         # Load from .env in project root
        env_file_encoding="utf-8",
        case_sensitive=True,     # SUPABASE_DB_URL != supabase_db_url
        extra="ignore",          # Silently ignore undeclared .env keys
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    Using @lru_cache means the .env file is parsed exactly once
    at application startup, not on every request. Subsequent calls
    return the same object from memory.

    Usage in any module:
        from app.core.config import get_settings
        settings = get_settings()
        print(settings.SUPABASE_DB_URL)
    """
    return Settings()
