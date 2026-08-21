-- ============================================================
-- AI Wellness LMS — Supabase Database Initialization Script
-- ============================================================
-- HOW TO USE:
--   1. Go to https://supabase.com → Your Project → SQL Editor
--   2. Paste and run this entire script
--   3. All tables, indexes, and extensions will be created
-- ============================================================
-- PHASE COVERAGE:
--   Phase 1 : semantic_memories, episodes, biometrics_stream, users
--   Phase 2 : refresh_tokens table (JWT stateless revocation)
--   Phase 4 : is_pinned column on semantic_memories (bypass time-decay)
--   Phase 7  : archived_at column on episodes (cold storage)
--   Phase 9  : consolidation_logs table
--   Phase 10 : biometrics_stream becomes the TimescaleDB-like table
-- ============================================================

-- -------------------------------------------------------
-- STEP 1: Enable pgvector Extension
-- Required for Layer 3 semantic memory vector storage.
-- This allows storing 1536-dimensional float arrays in a
-- column and running HNSW cosine similarity searches.
-- -------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

-- -------------------------------------------------------
-- STEP 2: Users Table (User Profiles & Baseline Data)
-- -------------------------------------------------------
-- Purpose:
--   Stores user identity and their evolving health baseline
--   profile (average sleep hours, known triggers, effective
--   coping mechanisms). The baseline_profile JSONB field is
--   updated by Phase 7 nightly consolidation when a new
--   "baseline" category memory is confirmed.
--
-- Connected to:
--   Phase 2  → User registration / auth endpoints
--   Phase 7  → Consolidation updates baseline_profile JSONB
--   Phase 8  → Cascading DELETE FROM users WHERE user_id = :id
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id          TEXT PRIMARY KEY,
    email            TEXT NOT NULL UNIQUE,
    password_hash    TEXT NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    baseline_profile JSONB DEFAULT '{
        "averageSleepHours": null,
        "knownTriggers": [],
        "effectiveCopingMechanisms": [],
        "dataRetentionDays": 365,
        "allowBiometrics": false
    }'::jsonb
);

-- -------------------------------------------------------
-- STEP 3: Phase 2 — Refresh Tokens Table (Stateless JWT Revocation)
-- -------------------------------------------------------
-- Purpose:
--   Stores SHA-256 hashed refresh tokens issued at login.
--   Allows server-side revocation on logout or suspicious activity,
--   without invalidating the user's account.
--
-- Security model:
--   - Raw refresh token is NEVER stored. Only the SHA-256 hash is persisted.
--   - On refresh, the client sends the raw token. We hash it and look up here.
--   - ON DELETE CASCADE from users ensures full purge on GDPR deletion.
--
-- Connected to:
--   Phase 2 → POST /api/v1/auth/login  (inserts row)
--   Phase 2 → POST /api/v1/auth/refresh (validates + rotates)
--   Phase 2 → POST /api/v1/auth/logout  (marks revoked=TRUE)
--   Phase 8 → ON DELETE CASCADE clears all tokens on user deletion
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token_hash   TEXT NOT NULL UNIQUE,   -- SHA-256 of the raw refresh token
    expires_at   TIMESTAMPTZ NOT NULL,
    revoked      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Fast lookup: given a user, find their active tokens (e.g. for bulk revoke on logout all)
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user
    ON refresh_tokens(user_id);

-- Fast revocation check: token_hash lookup is always the hot path
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_hash
    ON refresh_tokens(token_hash);

-- -------------------------------------------------------
-- STEP 5: Layer 2 — Episodes Table (Episodic Memory)
-- -------------------------------------------------------
-- Purpose:
--   Each row is a structured daily health summary produced
--   after a chat session ends. The extracted_metrics JSONB
--   field holds typed health data (mood score, sleep hours,
--   primary stressor, symptoms, biometrics from Phase 10).
--
-- Retrieval pattern:
--   SELECT * FROM episodes
--   WHERE user_id = :uid AND timestamp >= NOW() - INTERVAL '14 days'
--   ORDER BY timestamp DESC LIMIT 5;
--
-- Connected to:
--   Phase 1  → Retrieved in Hybrid RAG for context injection
--   Phase 5  → Real-time episode insert after each session end
--   Phase 7  → Source data for nightly LLM consolidation
--   Phase 8  → Cascading DELETE FROM episodes WHERE user_id = :id
--   Phase 10 → extracted_metrics enriched with wearable biometrics
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS episodes (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    timestamp        TIMESTAMPTZ DEFAULT NOW(),
    session_summary  TEXT NOT NULL,
    extracted_metrics JSONB NOT NULL DEFAULT '{
        "moodScore": null,
        "physicalSymptoms": [],
        "primaryStressor": null,
        "sleepHoursLogged": null,
        "anxietyLevel": null,
        "energyLevel": null,
        "biometrics": {}
    }'::jsonb,
    archived_at      TIMESTAMPTZ DEFAULT NULL
    -- archived_at NULL  = active (within 90 days)
    -- archived_at SET   = cold storage (Phase 7 archival policy)
);

-- Compound index: fast lookups for a user's recent episodes
CREATE INDEX IF NOT EXISTS idx_episodes_user_time
    ON episodes(user_id, timestamp DESC);

-- Partial index: only scan active (non-archived) episodes in retrieval
CREATE INDEX IF NOT EXISTS idx_episodes_active
    ON episodes(user_id, timestamp DESC)
    WHERE archived_at IS NULL;

-- -------------------------------------------------------
-- STEP 6: Layer 3 — Semantic Memories Table (pgvector)
-- -------------------------------------------------------
-- Purpose:
--   The long-term compressed health fact store. Each row is
--   a semantic insight (trigger, coping mechanism, baseline
--   change, recurring symptom, or milestone) stored as a
--   1536-dimensional vector alongside its source text and
--   metadata. HNSW index enables sub-millisecond approximate
--   nearest-neighbor cosine similarity search.
--
-- Time-decay scoring (Phase 1, Phase 5):
--   S_adjusted = S_raw * exp(-0.005 * delta_days)
--   Memories decaying below 0.65 are excluded from context.
--
-- Deduplication (Phase 7):
--   If a new vector has cosine similarity > 0.88 with an
--   existing memory, reinforcement_count is incremented
--   instead of inserting a duplicate vector.
--
-- Connected to:
--   Phase 1  → Vector search in Hybrid RAG retrieval
--   Phase 7  → Upsert of consolidation-extracted facts
--   Phase 8  → Cascading DELETE FROM semantic_memories WHERE user_id = :id
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS semantic_memories (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id              TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    category             TEXT NOT NULL CHECK (
                             category IN (
                                 'trigger',
                                 'baseline',
                                 'coping_mechanism',
                                 'symptom',
                                 'milestone'
                             )
                         ),
    text                 TEXT NOT NULL,
    embedding            vector(1536),    -- OpenAI text-embedding-3-small output
    reinforcement_count  INT DEFAULT 1,   -- Incremented on deduplication hit (Phase 7)
    is_pinned            BOOLEAN NOT NULL DEFAULT FALSE,  -- Phase 4: bypass time-decay when TRUE
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index: approximate nearest-neighbor cosine distance
-- m=16 (connectivity) and ef_construction=64 (build quality)
-- are good balanced defaults for our use case.
CREATE INDEX IF NOT EXISTS idx_semantic_memories_hnsw
    ON semantic_memories
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Standard B-tree index for fast user-scoped filtering
CREATE INDEX IF NOT EXISTS idx_semantic_memories_user
    ON semantic_memories(user_id);

-- Composite index: filter by user + category (Phase 7 dedup queries)
CREATE INDEX IF NOT EXISTS idx_semantic_memories_user_category
    ON semantic_memories(user_id, category);

-- Partial index: fast retrieval of pinned memories (Phase 4)
-- Enables the ORDER BY is_pinned DESC path to hit index without full scan
CREATE INDEX IF NOT EXISTS idx_semantic_memories_pinned
    ON semantic_memories(user_id, is_pinned)
    WHERE is_pinned = TRUE;

-- -------------------------------------------------------
-- PHASE 4 MIGRATION: Add is_pinned to existing deployments
-- -------------------------------------------------------
-- If you ran schema.sql before Phase 4, run this once in
-- Supabase SQL Editor to add the column to an existing table:
--
--   ALTER TABLE semantic_memories
--   ADD COLUMN IF NOT EXISTS is_pinned BOOLEAN NOT NULL DEFAULT FALSE;
--
--   CREATE INDEX IF NOT EXISTS idx_semantic_memories_pinned
--       ON semantic_memories(user_id, is_pinned)
--       WHERE is_pinned = TRUE;
--
-- Safe to run even if the column already exists (IF NOT EXISTS).
-- -------------------------------------------------------

-- -------------------------------------------------------
-- STEP 5: Layer 4 — Biometrics Stream Table
-- -------------------------------------------------------
-- Purpose:
--   High-volume time-series storage for wearable device data
--   (HRV, Resting HR, Sleep stages, Step counts, SpO2).
--   Each row is a single metric sample at a point in time.
--   Phase 10 turns this into a fully managed ingestion pipeline.
--   Phase 1 creates the schema so models can reference it.
--
-- Connected to:
--   Phase 10 → POST /api/v1/biometrics intake API
--   Phase 10 → Continuous SQL aggregate views (daily, hourly)
--   Phase 10 → Wearable enrichment of Layer 2 episodes
--   Phase 8  → Cascading DELETE FROM biometrics_stream WHERE user_id = :id
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS biometrics_stream (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id      TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    metric_type  TEXT NOT NULL,
    -- 'resting_hr', 'hrv_rmssd', 'sleep_deep_min',
    -- 'sleep_rem_min', 'steps', 'active_calories', 'spo2'
    value        DOUBLE PRECISION NOT NULL,
    device_id    TEXT,                    -- Source device identifier
    quality_flag SMALLINT DEFAULT 1       -- 1=clean, 0=noisy/artifact (Phase 10 filters)
);

-- Composite index: fastest access pattern for daily rollup queries
CREATE INDEX IF NOT EXISTS idx_biometrics_user_metric_time
    ON biometrics_stream(user_id, metric_type, time DESC);

-- -------------------------------------------------------
-- STEP 7: Layer 4 — Biometrics Stream Table
-- -------------------------------------------------------
-- Purpose:
--   Records each nightly consolidation pipeline run for
--   observability, debugging, and the research benchmark.
--
-- Connected to:
--   Phase 7  → Written after each nightly Celery cron run
--   Phase 9  → Queried for benchmark reports and Grafana dashboards
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS consolidation_logs (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           TEXT NOT NULL,
    ran_at            TIMESTAMPTZ DEFAULT NOW(),
    episodes_analyzed INT DEFAULT 0,
    insights_extracted INT DEFAULT 0,
    new_memories      INT DEFAULT 0,
    reinforced_memories INT DEFAULT 0,
    status            TEXT DEFAULT 'success'  -- 'success', 'failed', 'skipped'
);

-- -------------------------------------------------------
-- VERIFICATION: List all created tables
-- -------------------------------------------------------
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
