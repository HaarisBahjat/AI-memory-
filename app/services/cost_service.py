"""
============================================================
app/services/cost_service.py — Token Budget & LLM Cost Tracking
============================================================
PURPOSE:
    Provides two production-critical capabilities:

    1. TOKEN BUDGET ENFORCEMENT
       Before each LLM API call, estimates the token cost and
       checks against the user's cumulative budget. Requests
       that would exceed the budget are rejected.

    2. LLM USAGE EVENT RECORDING
       After each successful LLM call, persists a structured
       record to `llm_usage_events` for cost monitoring and
       Phase 9 research evaluation.

TOKEN BUDGET FLOW:
    estimate_token_count(prompt, max_completion)
             ↓
    check_budget(user_id, estimated_tokens, db)
             ↓
    ── If OK ──────────────────────────────────
    (LLM call happens in retrieval_engine.py)
             ↓
    record_usage(user_id, op, model, usage, db)
             ↓
    update users.tokens_used += actual_tokens

COST RATE CARDS:
    Rates are approximate USD per 1M tokens.
    They are used for estimation only — production billing
    should be read from provider dashboards.

CONNECTED TO:
    Phase 9.1 → app/models/user.py  (tokens_used, token_budget)
    Phase 9.1 → app/models/llm_usage_event.py (LLMUsageEvent)
    Phase 9.1 → app/services/retrieval_engine.py (CHAT events)
    Phase 9.1 → app/core/metrics.py (TOKEN_BUDGET_EXCEEDED_TOTAL)
    Phase 9.2 → benchmark scripts query llm_usage_events table
============================================================
"""

import json
import uuid
from typing import Optional

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.metrics import LLM_TOKENS_USED_TOTAL, TOKEN_BUDGET_EXCEEDED_TOTAL

log = structlog.get_logger(__name__)

# -------------------------------------------------------
# Model Cost Rate Cards (USD per 1M tokens, approximate)
# -------------------------------------------------------
# These are indicative estimates for research cost accounting.
# Update as pricing changes.
_COST_RATE_CARD: dict[str, dict[str, float]] = {
    # Gemini models (via Google AI Studio free tier or Vertex)
    "gemini-2.5-flash":             {"input": 0.075,  "output": 0.30},
    "gemini-2.5-pro":               {"input": 1.25,   "output": 5.00},
    "models/gemini-flash-lite-latest": {"input": 0.0,  "output": 0.0},
    "gemini-embedding-2":           {"input": 0.0,    "output": 0.0},
    # OpenAI models (fallback reference)
    "gpt-4o-mini":                  {"input": 0.15,   "output": 0.60},
    "gpt-4o":                       {"input": 2.50,   "output": 10.0},
    "text-embedding-3-small":       {"input": 0.02,   "output": 0.0},
    "text-embedding-3-large":       {"input": 0.13,   "output": 0.0},
}

# Default to free tier if model not found in rate card
_DEFAULT_RATE = {"input": 0.0, "output": 0.0}

# Default token budget per user if not set (set to 0 = unlimited)
DEFAULT_TOKEN_BUDGET = 0  # 0 means no budget limit


def estimate_token_count(
    prompt: str,
    max_completion_tokens: int = 500,
    chars_per_token: float = 4.0,
) -> int:
    """
    Fast heuristic estimate of total token count without calling tiktoken.

    Uses the common rule-of-thumb: 1 token ≈ 4 characters for English text.
    This keeps the budget check latency under 1ms.

    Args:
        prompt              : Full system + user prompt string
        max_completion_tokens: Expected max output tokens
        chars_per_token     : Conversion factor (default 4.0 chars/token)

    Returns:
        Estimated total tokens (prompt + completion)
    """
    prompt_tokens = int(len(prompt) / chars_per_token)
    return prompt_tokens + max_completion_tokens


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """
    Calculates estimated USD cost for a single LLM API call.

    Args:
        model            : Model name (must be in _COST_RATE_CARD)
        prompt_tokens    : Actual input tokens used
        completion_tokens: Actual output tokens used

    Returns:
        Estimated cost in USD (may be 0.0 for free-tier models)
    """
    rate = _COST_RATE_CARD.get(model, _DEFAULT_RATE)
    cost = (
        (prompt_tokens    / 1_000_000) * rate["input"] +
        (completion_tokens / 1_000_000) * rate["output"]
    )
    return round(cost, 8)


async def check_budget(
    user_id: str,
    estimated_tokens: int,
    db: AsyncSession,
    operation: str = "CHAT",
) -> tuple[bool, int]:
    """
    Checks whether a user has sufficient token budget for an estimated call.

    Fetches current (token_budget, tokens_used) from the users table.
    Budget of 0 means unlimited — the check always passes.

    Args:
        user_id         : User to check
        estimated_tokens: Token count estimate from estimate_token_count()
        db              : Async database session
        operation       : Operation type for metrics label

    Returns:
        (allowed: bool, remaining: int)
        allowed=True if budget is unlimited or sufficient.
        remaining=-1 means unlimited.
    """
    try:
        result = await db.execute(
            text("SELECT token_budget, tokens_used FROM users WHERE user_id = :uid"),
            {"uid": user_id},
        )
        row = result.mappings().first()
        if not row:
            # User not found — allow (graceful degradation)
            log.warning("Budget check: user not found", user_id=user_id)
            return True, -1

        budget: int = row["token_budget"] or 0
        used: int = row["tokens_used"] or 0

        # 0 means unlimited
        if budget == 0:
            return True, -1

        remaining = budget - used
        if estimated_tokens > remaining:
            log.warning(
                "Token budget exceeded",
                user_id=user_id,
                budget=budget,
                used=used,
                estimated=estimated_tokens,
                remaining=remaining,
                operation=operation,
            )
            TOKEN_BUDGET_EXCEEDED_TOTAL.labels(operation=operation).inc()
            return False, remaining

        return True, remaining

    except Exception as e:
        # Fail open — don't block requests on budget check errors
        log.error("Budget check failed", user_id=user_id, error=str(e))
        return True, -1


async def record_usage(
    user_id: str,
    operation: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    db: AsyncSession,
    session_id: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> None:
    """
    Records an LLM usage event and updates the user's cumulative token count.

    Steps:
        1. Insert row into llm_usage_events
        2. UPDATE users SET tokens_used += total_tokens
        3. Increment Prometheus token counters (low-cardinality labels only)

    Args:
        user_id          : Owning user
        operation        : CHAT | CONSOLIDATION | EMBEDDING |
                           GRAPH_EXTRACTION | EVALUATION
        model            : Model name
        prompt_tokens    : Input tokens from API response
        completion_tokens: Output tokens from API response
        db               : Async database session
        session_id       : Optional Redis session key
        extra_metadata   : Optional dict with retrieval_strategy, etc.
    """
    total_tokens = prompt_tokens + completion_tokens
    cost = calculate_cost(model, prompt_tokens, completion_tokens)
    event_id = str(uuid.uuid4())

    try:
        # 1. Insert usage event
        await db.execute(
            text("""
                INSERT INTO llm_usage_events
                    (id, user_id, operation, model, prompt_tokens,
                     completion_tokens, total_tokens, estimated_api_cost,
                     session_id, extra_metadata, created_at)
                VALUES
                    (:id, :user_id, :operation, :model, :prompt_tokens,
                     :completion_tokens, :total_tokens, :estimated_api_cost,
                     :session_id, :extra_metadata, NOW())
            """),
            {
                "id": event_id,
                "user_id": user_id,
                "operation": operation,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_api_cost": cost,
                "session_id": session_id,
                "extra_metadata": json.dumps(extra_metadata) if extra_metadata else None,
            },
        )

        # 2. Increment user's cumulative token usage
        await db.execute(
            text("""
                UPDATE users
                SET tokens_used = COALESCE(tokens_used, 0) + :total_tokens
                WHERE user_id = :uid
            """),
            {"total_tokens": total_tokens, "uid": user_id},
        )

        await db.commit()

        # 3. Prometheus counters (low-cardinality labels only — NO user_id)
        LLM_TOKENS_USED_TOTAL.labels(
            model=model, operation=operation, token_type="prompt"
        ).inc(prompt_tokens)
        LLM_TOKENS_USED_TOTAL.labels(
            model=model, operation=operation, token_type="completion"
        ).inc(completion_tokens)
        LLM_TOKENS_USED_TOTAL.labels(
            model=model, operation=operation, token_type="total"
        ).inc(total_tokens)

        log.info(
            "LLM usage recorded",
            event_id=event_id,
            user_id=user_id,
            operation=operation,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
        )

    except Exception as e:
        log.error(
            "Failed to record LLM usage event",
            user_id=user_id,
            operation=operation,
            error=str(e),
        )
        # Do not re-raise — usage recording must never crash the main request
