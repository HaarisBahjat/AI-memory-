"""
============================================================
app/core/security.py — JWT & Password Security Layer (Phase 2)
============================================================
PURPOSE:
    Single source of truth for all cryptographic operations:
    - Bcrypt password hashing/verification (timing-attack safe)
    - JWT access token creation and decoding (HS256 / RS256 ready)
    - Refresh token generation with SHA-256 fingerprinting

    All functions are synchronous and stateless — no I/O,
    no DB calls. That separation keeps this module testable
    in isolation without any running infrastructure.

SECURITY PROPERTIES:
    - Passwords: bcrypt with auto-generated salt (passlib[bcrypt])
    - Access tokens: HS256 JWT, 15-minute expiry (configurable)
    - Refresh tokens: 256-bit cryptographically random secret,
      stored only as SHA-256(token) in the DB — raw token is
      never persisted, mimicking how OAuth2 server flows work.
    - Token decode: raises HTTP 401 on expired, invalid signature,
      missing `sub` claim, or malformed input.

CONNECTED TO:
    Phase 2 → app/api/v1/auth.py (register, login, refresh, logout)
    Phase 2 → app/api/deps.py    (get_current_user decodes access token)
    Phase 8 → GDPR deletion verifies no tokens remain after purge
============================================================
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, status

from app.core.config import get_settings

settings = get_settings()

# -------------------------------------------------------
# Password Hashing (direct bcrypt — no passlib wrapper)
# -------------------------------------------------------
# We use the bcrypt library directly rather than via passlib
# because passlib 1.7.x has known compat issues with bcrypt 4.x+.
# Direct usage is simpler and equally secure.
# Work factor 12 is the current OWASP recommendation (2024).
_BCRYPT_ROUNDS = 12


def hash_password(password: str) -> str:
    """
    Hashes a plain-text password using bcrypt with a random salt.

    Work factor 12 means 2^12 = 4096 iterations — strong enough
    to resist GPU-based brute-force attacks as of 2024.

    Args:
        password: Plain-text password from registration form.

    Returns:
        Bcrypt-hashed password string safe for DB storage.
    """
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Constant-time bcrypt comparison.

    bcrypt.checkpw() uses a constant-time string comparison internally,
    making it safe against timing attacks.

    Args:
        plain_password : User-submitted raw password.
        hashed_password: Stored bcrypt hash from the DB.

    Returns:
        True if they match, False otherwise.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


# -------------------------------------------------------
# JWT Access Token
# -------------------------------------------------------

def create_access_token(data: dict[str, Any]) -> str:
    """
    Creates a signed JWT access token with a 15-minute expiry.

    The `sub` (subject) claim carries the user_id. All other
    data is also encoded in the payload but should be kept
    minimal (no sensitive fields — JWTs are only base64-encoded,
    not encrypted).

    Args:
        data: Dict with at minimum {"sub": user_id}.
              Additional claims (e.g. {"email": ...}) can be
              included but keep them small.

    Returns:
        Signed JWT string in compact serialization format.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode["exp"] = expire
    to_encode["iat"] = datetime.now(timezone.utc)
    to_encode["type"] = "access"

    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Decodes and validates a JWT access token.

    Raises HTTP 401 (not Python exceptions) so this function
    can be called directly from FastAPI dependencies without
    any additional error wrapping.

    Validation performed by python-jose:
    - Signature: verifies against JWT_SECRET
    - Expiry (`exp`): raises if token is past expiry
    - `type` claim: must be "access" (not "refresh")

    Args:
        token: Raw JWT string from the Authorization header.

    Returns:
        Decoded payload dict with at minimum {"sub": user_id}.

    Raises:
        HTTPException 401 on any validation failure.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError:
        raise credentials_exception

    # Ensure it's an access token, not a refresh token accidentally used here
    if payload.get("type") != "access":
        raise credentials_exception

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise credentials_exception

    return payload


# -------------------------------------------------------
# Refresh Token (256-bit random + SHA-256 fingerprint)
# -------------------------------------------------------

def generate_refresh_token() -> tuple[str, str]:
    """
    Generates a cryptographically random refresh token and
    returns both the raw token and its SHA-256 fingerprint.

    The RAW token is returned to the client (in the HTTP response).
    The HASH is stored in the `refresh_tokens` table.

    This pattern is analogous to how API keys are handled:
    the server never stores the raw secret, only a hash of it.
    Even if the DB is compromised, the attacker cannot use the
    stored hashes to impersonate users without brute-forcing
    the 256-bit token space.

    Returns:
        (raw_token, sha256_hash) — raw goes to client, hash to DB.
    """
    raw_token = secrets.token_urlsafe(32)  # 256 bits of entropy
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    return raw_token, token_hash


def hash_refresh_token(raw_token: str) -> str:
    """
    Computes the SHA-256 fingerprint of a raw refresh token.

    Used during the /refresh flow: the client sends the raw
    token, we hash it and look up the hash in the DB.

    Args:
        raw_token: The raw token string from the client.

    Returns:
        SHA-256 hex digest string for DB lookup.
    """
    return hashlib.sha256(raw_token.encode()).hexdigest()
