"""
============================================================
app/api/deps.py — FastAPI Reusable Dependencies (Phase 2)
============================================================
PURPOSE:
    Centralizes all injectable FastAPI dependencies that are
    shared across multiple route modules.

    The primary dependency here is `get_current_user`, which
    guards every protected endpoint. By depending on it,
    a route is automatically:
        1. Requiring a valid Bearer token in the Authorization header
        2. Validating token signature + expiry
        3. Loading the user record from the DB
        4. Injecting the user object into the route handler

USAGE PATTERN (in any protected route):
    from app.api.deps import get_current_user
    from app.schemas.auth import CurrentUser

    @router.get("/me")
    async def my_route(
        current_user: CurrentUser = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ):
        # current_user.user_id is guaranteed to be a valid,
        # authenticated user at this point
        ...

CONNECTED TO:
    Phase 2 → app/api/v1/auth.py (not used — auth routes are public)
    Phase 2 → app/api/v1/user.py (/me endpoints)
    Phase 2 → app/api/v1/chat.py (guards the chat endpoint)
    Phase 4+ → All future memory CRUD endpoints
============================================================
"""
import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_access_token
from app.schemas.auth import CurrentUser

log = structlog.get_logger(__name__)

# -------------------------------------------------------
# OAuth2 Bearer Scheme
# -------------------------------------------------------
# FastAPI uses this to:
#   1. Add a "Authorize" button to /docs (Swagger UI)
#   2. Automatically extract the Bearer token from the
#      "Authorization: Bearer <token>" request header
#   3. Pass the raw token string to get_current_user
#
# tokenUrl="/api/v1/auth/login" tells Swagger where to obtain
# a token — this is ONLY used for the Swagger UI form, not
# for actual auth logic.
# -------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    """
    FastAPI dependency that authenticates every protected request.

    Flow:
    1. Extract Bearer token from Authorization header (via OAuth2PasswordBearer)
    2. Decode and validate the JWT (signature + expiry + type claim)
    3. Extract user_id from the `sub` claim
    4. Load the user from the database (ensures the account still exists)
    5. Return a CurrentUser object to the route handler

    Why check the DB even after validating the JWT?
        JWTs are stateless — a valid token doesn't mean the user
        still exists. If an account is deleted or suspended, the
        15-minute access token would otherwise still work. Loading
        from DB catches this edge case.

    Args:
        token: Raw JWT string (auto-extracted from Authorization header)
        db   : Async DB session (auto-injected)

    Returns:
        CurrentUser(user_id, email)

    Raises:
        HTTPException 401 — invalid/expired token
        HTTPException 401 — user not found in DB (account deleted)
    """
    # decode_access_token raises HTTPException 401 on any failure
    payload = decode_access_token(token)
    user_id: str = payload["sub"]

    # Verify the user still exists in the DB
    result = await db.execute(
        text("SELECT user_id, email, is_admin FROM users WHERE user_id = :uid"),
        {"uid": user_id},
    )
    user_row = result.mappings().first()

    if user_row is None:
        log.warning("Auth dependency: user not found after valid JWT", user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    log.debug("Auth dependency: user authenticated", user_id=user_id)
    return CurrentUser(user_id=user_row["user_id"], email=user_row["email"], is_admin=user_row["is_admin"])


async def require_admin(
    current_user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """
    FastAPI dependency that restricts an endpoint to admins only.
    """
    if not current_user.is_admin:
        log.warning("Auth dependency: non-admin attempted admin action", user_id=current_user.user_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Admin access required.",
        )
    return current_user
