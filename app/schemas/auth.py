"""
============================================================
app/schemas/auth.py — Pydantic Schemas for Auth API (Phase 2)
============================================================
PURPOSE:
    Defines all request and response shapes for the authentication
    endpoints. Pydantic v2 validates these at the FastAPI boundary
    before any business logic runs.

CONNECTED TO:
    Phase 2 → app/api/v1/auth.py (register, login, refresh, logout)
    Phase 2 → app/api/deps.py    (CurrentUser type returned by dependency)
    Phase 2 → app/api/v1/user.py (UserMeResponse, BaselineUpdateRequest)
============================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# -------------------------------------------------------
# Registration
# -------------------------------------------------------

class RegisterRequest(BaseModel):
    """
    Request body for POST /api/v1/auth/register.

    Validates email format via Pydantic's built-in EmailStr,
    and enforces minimum password strength before any DB write.
    """
    email: EmailStr = Field(
        ...,
        description="User's email address — must be unique across the system.",
        examples=["haaris@example.com"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Plain-text password. Min 8 characters. Bcrypt-hashed server-side.",
        examples=["Str0ng!Pass#2024"],
    )

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        """
        Enforce basic password complexity:
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        This is enforced here rather than in the route handler
        so validation error messages are consistent with other
        field validation errors (422 Unprocessable Entity).
        """
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit.")
        return v


# -------------------------------------------------------
# Login
# -------------------------------------------------------

class LoginRequest(BaseModel):
    """Request body for POST /api/v1/auth/login."""
    email: EmailStr = Field(..., description="Registered email address.")
    password: str = Field(..., description="Plain-text password.")


# -------------------------------------------------------
# Token Response (returned on login + register)
# -------------------------------------------------------

class TokenResponse(BaseModel):
    """
    Successful authentication response.

    `access_token`  → Short-lived (15 min). Sent as Bearer token in
                      Authorization header for every protected request.
    `refresh_token` → Long-lived (7 days). Used ONLY to obtain new
                      access tokens via POST /api/v1/auth/refresh.
                      Should be stored in HttpOnly cookie on web clients,
                      or secure storage on mobile clients.
    `expires_in`    → Access token TTL in seconds (for client-side timers).
    """
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(
        description="Access token lifetime in seconds.",
        default=900,  # 15 minutes — mirrors ACCESS_TOKEN_EXPIRE_MINUTES
    )


# -------------------------------------------------------
# Token Refresh
# -------------------------------------------------------

class RefreshRequest(BaseModel):
    """Request body for POST /api/v1/auth/refresh."""
    refresh_token: str = Field(
        ...,
        description="The raw refresh token received at login.",
    )


# -------------------------------------------------------
# Logout
# -------------------------------------------------------

class LogoutRequest(BaseModel):
    """Request body for POST /api/v1/auth/logout."""
    refresh_token: str = Field(
        ...,
        description="The refresh token to revoke.",
    )


# -------------------------------------------------------
# Current User (resolved by get_current_user dependency)
# -------------------------------------------------------

class CurrentUser(BaseModel):
    """
    Represents the authenticated user, resolved from the JWT.
    Injected by the `get_current_user` dependency into protected routes.
    """
    user_id: str
    email: str
    is_admin: bool

    model_config = {"from_attributes": True}


# -------------------------------------------------------
# Profile Responses & Updates
# -------------------------------------------------------

class UserMeResponse(BaseModel):
    """Response for GET /api/v1/user/me."""
    user_id: str
    email: str
    is_admin: bool
    created_at: datetime
    baseline_profile: dict[str, Any]

    model_config = {"from_attributes": True}


class BaselineUpdateRequest(BaseModel):
    """
    Request body for PUT /api/v1/user/me/baseline.

    All fields are Optional — the client sends only the fields
    they want to update. The server merges the patch into the
    existing JSONB using PostgreSQL's || operator.
    """
    averageSleepHours: Optional[float] = Field(
        default=None,
        ge=0,
        le=24,
        description="Average sleep hours per night.",
    )
    knownTriggers: Optional[list[str]] = Field(
        default=None,
        description="List of known anxiety/mood triggers.",
    )
    effectiveCopingMechanisms: Optional[list[str]] = Field(
        default=None,
        description="Coping strategies that have worked for this user.",
    )
    dataRetentionDays: Optional[int] = Field(
        default=None,
        ge=30,
        le=3650,
        description="How many days to retain episode history (GDPR preference).",
    )
    allowBiometrics: Optional[bool] = Field(
        default=None,
        description="Whether user consents to biometric data collection (Phase 10).",
    )
