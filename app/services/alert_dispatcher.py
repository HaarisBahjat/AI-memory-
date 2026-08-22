"""
============================================================
app/services/alert_dispatcher.py — Phase 6 Crisis Alert Dispatcher
============================================================
PURPOSE:
    Sends an external notification when a triage event fires.

    Supported channels (controlled by SAFETY_ALERT_CHANNEL env var):
        "NONE"  → No-op (default, safe for dev/test environments)
        "EMAIL" → Sends via SMTP (configurable in settings)
        "SLACK" → POSTs to a Slack Incoming Webhook URL

DISTRIBUTED RATE-LIMITING (Redis-backed):
    To prevent alert spam (e.g. a user repeatedly sending crisis
    messages), each dispatch call first checks a Redis key:

        key = "{SAFETY_ALERT_REDIS_PREFIX}:{user_id}"

    - If the key EXISTS → skip sending (rate-limit active)
    - If the key DOES NOT EXIST → send alert + SET key with TTL
        TTL = SAFETY_ALERT_RATE_LIMIT_MIN * 60 seconds

    Because the key lives in Redis (not in-process memory), this
    works correctly across multiple backend instances. All workers
    share the same Redis state.

PRODUCTION NOTES:
    - All alert sends are wrapped in try/except — alert failures
      NEVER propagate to the caller or the chat response.
    - SMTP sends are done via asyncio.to_thread to avoid blocking
      the event loop (smtplib is synchronous).
    - Only the crisis_type and session_id are included in alerts —
      no raw message content is ever sent externally (privacy).

CONNECTED TO:
    Phase 6  → Called by app/services/triage_service.evaluate_and_store()
    Phase 6  → Reads settings from app/core/config.py (SAFETY_ALERT_*)
    Phase 6  → Uses Redis from app/core/redis_client.py
============================================================
"""
import asyncio
import smtplib
import structlog
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import httpx
import redis.asyncio as aioredis

from app.core.config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()


# -------------------------------------------------------
# Rate-Limit Check (Redis-backed, distributed)
# -------------------------------------------------------

async def _is_rate_limited(redis: aioredis.Redis, user_id: str) -> bool:
    """
    Returns True if an alert was already sent for this user within
    the configured rate-limit window (SAFETY_ALERT_RATE_LIMIT_MIN minutes).

    Uses Redis SETNX (set-if-not-exists) + EXPIRE atomically via SET NX EX.
    This is safe against race conditions in multi-process deployments.
    """
    key = f"{settings.SAFETY_ALERT_REDIS_PREFIX}:{user_id}"
    ttl_seconds = settings.SAFETY_ALERT_RATE_LIMIT_MIN * 60

    # SET key "" NX EX ttl  →  returns True if key was set (not rate-limited)
    #                         →  returns None if key already exists (rate-limited)
    result = await redis.set(key, "1", nx=True, ex=ttl_seconds)
    return result is None  # None means key already existed → rate-limited


# -------------------------------------------------------
# Channel Implementations
# -------------------------------------------------------

def _send_email_sync(
    user_id: str,
    session_id: str,
    crisis_type: str,
) -> None:
    """
    Synchronous SMTP email sender. Run via asyncio.to_thread to avoid
    blocking the async event loop.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[CRISIS ALERT] {crisis_type.upper()} detected — user {user_id[:8]}***"
    msg["From"] = settings.SAFETY_ALERT_EMAIL_FROM
    msg["To"] = settings.SAFETY_ALERT_EMAIL_TO

    body = (
        f"<h2>Safety Triage Alert</h2>"
        f"<p><b>Crisis Type:</b> {crisis_type}</p>"
        f"<p><b>Session ID:</b> {session_id}</p>"
        f"<p><b>User ID:</b> {user_id[:8]}*** (truncated for privacy)</p>"
        f"<p>Please follow your clinical response protocol.</p>"
        f"<hr><small>This alert was sent by the AI Wellness LMS safety screener.</small>"
    )
    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP(settings.SAFETY_ALERT_SMTP_HOST, settings.SAFETY_ALERT_SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(settings.SAFETY_ALERT_SMTP_USER, settings.SAFETY_ALERT_SMTP_PASSWORD)
        server.sendmail(
            settings.SAFETY_ALERT_EMAIL_FROM,
            settings.SAFETY_ALERT_EMAIL_TO,
            msg.as_string(),
        )


async def _send_slack_alert(
    user_id: str,
    session_id: str,
    crisis_type: str,
) -> None:
    """
    Sends a Slack message via Incoming Webhook.
    Uses httpx (async) to avoid blocking the event loop.
    """
    payload = {
        "text": (
            f":rotating_light: *Crisis Triage Alert* :rotating_light:\n"
            f"*Type:* `{crisis_type}`\n"
            f"*User:* `{user_id[:8]}***` (truncated for privacy)\n"
            f"*Session:* `{session_id}`\n"
            f"Please follow your clinical response protocol."
        )
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(settings.SAFETY_ALERT_SLACK_WEBHOOK_URL, json=payload)
        resp.raise_for_status()


# -------------------------------------------------------
# Public API
# -------------------------------------------------------

async def dispatch(
    redis: aioredis.Redis,
    user_id: str,
    session_id: str,
    crisis_type: str,
) -> bool:
    """
    Dispatch a crisis alert through the configured channel.

    Checks Redis rate-limit first. If the user already had an alert
    sent within the rate-limit window, skips silently and returns False.

    Args:
        redis       : Async Redis client (injected from request context)
        user_id     : The affected user's ID
        session_id  : The session identifier where the crisis was detected
        crisis_type : One of: self_harm, eating_disorder, acute_medical

    Returns:
        True  → alert was sent
        False → alert was skipped (rate-limited or channel=NONE)

    Raises:
        Never — all exceptions are caught and logged. Alert failures
        must never surface to the caller or affect the chat response.
    """
    channel = settings.SAFETY_ALERT_CHANNEL.upper()

    if channel == "NONE":
        log.debug("Alert channel is NONE — skipping dispatch", user_id=user_id)
        return False

    try:
        if await _is_rate_limited(redis, user_id):
            log.info(
                "Alert rate-limited — skipping",
                user_id=user_id,
                rate_limit_min=settings.SAFETY_ALERT_RATE_LIMIT_MIN,
            )
            return False
    except Exception as e:
        # Redis failure → degrade gracefully (don't block the crisis response)
        log.warning("Redis rate-limit check failed — proceeding with alert", error=str(e))

    try:
        if channel == "EMAIL":
            await asyncio.to_thread(_send_email_sync, user_id, session_id, crisis_type)
            log.info("Crisis email alert sent", user_id=user_id, crisis_type=crisis_type)
        elif channel == "SLACK":
            await _send_slack_alert(user_id, session_id, crisis_type)
            log.info("Crisis Slack alert sent", user_id=user_id, crisis_type=crisis_type)
        else:
            log.warning("Unknown alert channel — skipping", channel=channel)
            return False
        return True
    except Exception as e:
        log.error(
            "Crisis alert dispatch failed",
            channel=channel,
            user_id=user_id,
            crisis_type=crisis_type,
            error=str(e),
        )
        return False
