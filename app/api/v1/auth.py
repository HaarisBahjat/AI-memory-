"""
============================================================
app/api/v1/auth.py — Authentication Endpoints (Phase 2)
============================================================
PURPOSE:
    Provides the complete user authentication lifecycle:
    - POST /api/v1/auth/register → Create account + issue tokens
    - POST /api/v1/auth/login    → Verify credentials + issue tokens
    - POST /api/v1/auth/refresh  → Rotate access token using refresh token
    - POST /api/v1/auth/logout   → Revoke refresh token

    These endpoints are intentionally PUBLIC (no auth required).
    All other endpoints in the system require a valid Bearer token.

SECURITY NOTES:
    - Passwords are bcrypt-hashed before storage (never logged, never stored plain)
    - Refresh tokens are stored only as SHA-256 hashes in the DB
    - On login, any existing refresh tokens for the user are NOT revoked
      (allows multi-device login). POST /auth/logout revokes only the
      specific token sent in the request body.
    - Rate limiting is a Phase 9 concern (to be added via Redis-backed
      middleware). For now, Supabase connection limits act as a soft cap.

DATA FLOW:
    Register: ValidateRequest → CheckDuplicateEmail → HashPassword
              → InsertUser → GenerateTokens → StoreRefreshHash → Return

    Login:    ValidateRequest → FetchUserByEmail → VerifyPassword
              → GenerateTokens → StoreRefreshHash → Return

    Refresh:  HashIncomingToken → LookupHash → CheckRevoked+Expiry
              → GenerateNewAccessToken → RotateRefreshToken → Return

    Logout:   HashIncomingToken → LookupHash → SetRevoked=TRUE → Return

CONNECTED TO:
    Phase 2 → app/core/security.py  (all crypto ops)
    Phase 2 → app/models/refresh_token.py (DB token storage)
    Phase 8 → Purge flow deletes user row → CASCADE clears refresh_tokens
============================================================
"""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from app.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)

log = structlog.get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/auth", tags=["Authentication"])


# -------------------------------------------------------
# Helper: Build TokenResponse
# -------------------------------------------------------

def _build_token_response(user_id: str) -> tuple[TokenResponse, str, str]:
    """
    Generates a fresh access + refresh token pair for the given user.

    Kept as a helper so both /register and /login share identical
    token generation logic without code duplication.

    Returns:
        (TokenResponse, raw_refresh_token, refresh_token_hash)
        The raw_refresh_token goes to the client.
        The refresh_token_hash goes to the database.
    """
    access_token = create_access_token(data={"sub": user_id})
    raw_refresh, refresh_hash = generate_refresh_token()

    response = TokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
    return response, raw_refresh, refresh_hash


async def _store_refresh_token(
    db: AsyncSession,
    user_id: str,
    token_hash: str,
) -> None:
    """
    Inserts a new hashed refresh token into the `refresh_tokens` table.

    Args:
        db         : Active DB session (caller commits via get_db())
        user_id    : Owner of the token
        token_hash : SHA-256 fingerprint of the raw refresh token
    """
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )
    await db.execute(
        text("""
            INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
            VALUES (:user_id, :token_hash, :expires_at)
        """),
        {
            "user_id": user_id,
            "token_hash": token_hash,
            "expires_at": expires_at,
        },
    )


# -------------------------------------------------------
# POST /api/v1/auth/register
# -------------------------------------------------------

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    description=(
        "Creates a new user account with a bcrypt-hashed password and "
        "returns an access + refresh token pair. The access token expires "
        f"in 15 minutes. The refresh token expires in 7 days."
    ),
    responses={
        201: {"description": "Account created — tokens issued"},
        409: {"description": "Email already registered"},
        422: {"description": "Validation error (email format, password strength)"},
    },
)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Registration endpoint.

    Steps:
    1. Check email uniqueness (409 if duplicate)
    2. Generate a server-side UUID as user_id
    3. Bcrypt-hash the password
    4. Insert user row into `users` table
    5. Generate access + refresh tokens
    6. Store refresh token hash in `refresh_tokens` table
    7. Return TokenResponse
    """
    log.info("Registration attempt", email=request.email)

    # ── Step 1: Duplicate email check ──────────────────────────────
    existing = await db.execute(
        text("SELECT user_id FROM users WHERE email = :email"),
        {"email": request.email},
    )
    if existing.first() is not None:
        log.warning("Registration rejected — email already in use", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address already exists.",
        )

    # ── Step 2 + 3: Generate user_id + hash password ───────────────
    user_id = str(uuid4())
    password_hash = hash_password(request.password)

    # ── Step 4: Insert user ────────────────────────────────────────
    await db.execute(
        text("""
            INSERT INTO users (user_id, email, password_hash)
            VALUES (:user_id, :email, :password_hash)
        """),
        {
            "user_id": user_id,
            "email": request.email,
            "password_hash": password_hash,
        },
    )

    # ── Step 5 + 6: Generate tokens + store refresh hash ───────────
    token_response, _raw_refresh, refresh_hash = _build_token_response(user_id)
    await _store_refresh_token(db, user_id, refresh_hash)

    log.info("User registered successfully", user_id=user_id)
    return token_response


# -------------------------------------------------------
# POST /api/v1/auth/login
# -------------------------------------------------------

@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Login with email and password",
    description=(
        "Validates credentials and returns an access + refresh token pair. "
        "Returns a generic 401 on both 'email not found' and 'wrong password' "
        "to prevent user enumeration attacks."
    ),
    responses={
        200: {"description": "Login successful — tokens issued"},
        401: {"description": "Invalid credentials"},
        422: {"description": "Validation error"},
    },
)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Login endpoint.

    Anti-enumeration: both "email not found" and "wrong password" return
    the same HTTP 401 with the same message, preventing attackers from
    discovering which emails are registered.

    Steps:
    1. Fetch user by email (return generic 401 if not found)
    2. Bcrypt-verify the submitted password (return generic 401 if wrong)
    3. Generate access + refresh tokens
    4. Store refresh token hash in `refresh_tokens` table
    5. Return TokenResponse
    """
    log.info("Login attempt", email=request.email)

    # Use a shared, non-revealing error for both "not found" + "wrong password"
    auth_failed_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # ── Step 1: Fetch user by email ────────────────────────────────
    result = await db.execute(
        text("SELECT user_id, password_hash FROM users WHERE email = :email"),
        {"email": request.email},
    )
    user_row = result.mappings().first()
    if user_row is None:
        log.warning("Login failed — email not found", email=request.email)
        raise auth_failed_error

    # ── Step 2: Verify password ────────────────────────────────────
    if not verify_password(request.password, user_row["password_hash"]):
        log.warning("Login failed — wrong password", email=request.email)
        raise auth_failed_error

    user_id = user_row["user_id"]

    # ── Step 3 + 4: Generate tokens + store refresh hash ───────────
    token_response, _raw_refresh, refresh_hash = _build_token_response(user_id)
    await _store_refresh_token(db, user_id, refresh_hash)

    log.info("User logged in successfully", user_id=user_id)
    return token_response


# -------------------------------------------------------
# POST /api/v1/auth/refresh
# -------------------------------------------------------

@router.post(
    "/refresh",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Refresh an expiring access token",
    description=(
        "Accepts a valid, non-revoked refresh token and issues a new "
        "access token + rotated refresh token. The old refresh token is "
        "revoked (single-use rotation for replay attack prevention)."
    ),
    responses={
        200: {"description": "New tokens issued"},
        401: {"description": "Refresh token invalid, revoked, or expired"},
    },
)
async def refresh_token(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Token rotation endpoint.

    Implements 'Refresh Token Rotation' — each refresh cycle
    invalidates the old refresh token and issues a brand new one.
    This limits the damage window if a refresh token is stolen:
    the first use by the legitimate user will invalidate it.

    Steps:
    1. Hash the incoming raw refresh token
    2. Look up the hash in `refresh_tokens`
    3. Check: exists + not revoked + not expired
    4. Mark the old token as revoked (rotation)
    5. Generate new access + refresh token pair
    6. Store new refresh hash
    7. Return new TokenResponse
    """
    invalid_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Refresh token is invalid, expired, or has been revoked.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # ── Step 1: Hash incoming token ────────────────────────────────
    incoming_hash = hash_refresh_token(request.refresh_token)

    # ── Step 2: Look up in DB ──────────────────────────────────────
    result = await db.execute(
        text("""
            SELECT id, user_id, expires_at, revoked
            FROM refresh_tokens
            WHERE token_hash = :token_hash
        """),
        {"token_hash": incoming_hash},
    )
    token_row = result.mappings().first()

    if token_row is None:
        log.warning("Refresh failed — token hash not found")
        raise invalid_error

    # ── Step 3: Validate state ─────────────────────────────────────
    if token_row["revoked"]:
        log.warning("Refresh failed — token already revoked", user_id=token_row["user_id"])
        raise invalid_error

    expires_at = token_row["expires_at"]
    # Ensure timezone-aware comparison
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires_at:
        log.warning("Refresh failed — token expired", user_id=token_row["user_id"])
        raise invalid_error

    user_id = token_row["user_id"]

    # ── Step 4: Revoke old token (rotation) ────────────────────────
    await db.execute(
        text("UPDATE refresh_tokens SET revoked = TRUE WHERE id = :id"),
        {"id": str(token_row["id"])},
    )

    # ── Step 5 + 6: Issue new token pair ───────────────────────────
    token_response, _raw_refresh, new_refresh_hash = _build_token_response(user_id)
    await _store_refresh_token(db, user_id, new_refresh_hash)

    log.info("Token refreshed successfully", user_id=user_id)
    return token_response


# -------------------------------------------------------
# POST /api/v1/auth/logout
# -------------------------------------------------------

@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="Logout and revoke refresh token",
    description=(
        "Revokes the provided refresh token. The access token remains valid "
        "until its 15-minute TTL expires (stateless by design). "
        "Clients should discard both tokens immediately after this call."
    ),
    responses={
        200: {"description": "Logged out — refresh token revoked"},
        400: {"description": "Token already revoked or not found"},
    },
)
async def logout(
    request: LogoutRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Logout endpoint.

    Sets `revoked = TRUE` on the specific refresh token sent in the body.
    Only the matching token is revoked — other active sessions on other
    devices remain intact.

    To revoke ALL sessions (e.g., "logout from all devices"), call this
    endpoint once per active refresh token, or add a separate
    DELETE /api/v1/auth/sessions endpoint in a future phase.
    """
    incoming_hash = hash_refresh_token(request.refresh_token)

    result = await db.execute(
        text("""
            UPDATE refresh_tokens
            SET revoked = TRUE
            WHERE token_hash = :token_hash AND revoked = FALSE
            RETURNING user_id
        """),
        {"token_hash": incoming_hash},
    )
    updated = result.first()

    if updated is None:
        # Token not found or already revoked — either way, logout intent is satisfied
        log.info("Logout: token not found or already revoked")
    else:
        log.info("User logged out — refresh token revoked", user_id=updated[0])

    return {"message": "Successfully logged out."}
