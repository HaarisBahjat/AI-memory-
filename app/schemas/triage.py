"""
============================================================
app/schemas/triage.py — Phase 6 Safety Triage Pydantic Schemas
============================================================
PURPOSE:
    Defines request/response contracts for the triage_events
    data model. These schemas are used by:
    - The triage_service for database serialization
    - The GET /triage admin API endpoints
    - The alert_dispatcher for alert payload construction

CONNECTED TO:
    Phase 6  → app/services/triage_service.py
    Phase 6  → app/services/alert_dispatcher.py
    Phase 6  → app/api/v1/triage.py
============================================================
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


# -------------------------------------------------------
# Response Schemas
# -------------------------------------------------------

class TriageEventResponse(BaseModel):
    """
    Returned by GET /triage and GET /triage/{id}.

    Note: `triggered_by` is intentionally excluded — it contains
    the raw regex pattern and is internal audit data only.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    session_id: str
    created_at: datetime
    crisis_type: str
    severity: str
    resources: Optional[list] = None
    confidence: Optional[float] = None
    alert_sent: bool
    archived_at: Optional[datetime] = None


class TriageEventListResponse(BaseModel):
    """Paginated list wrapper for triage event queries."""
    items: List[TriageEventResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class TriageEventSoftDeleteResponse(BaseModel):
    """Returned after a soft-delete (archived_at is set)."""
    id: str
    archived_at: datetime
    message: str = "Triage event archived successfully."


# -------------------------------------------------------
# Internal Data Object — used between service layers
# -------------------------------------------------------

class TriageEventCreate(BaseModel):
    """
    Internal data object passed from triage_service to the DB insert.
    NOT exposed as an HTTP request body — triage events are only
    created automatically by the safety screener, never manually.
    """
    user_id: str
    session_id: str
    crisis_type: str
    severity: str = "HIGH"
    triggered_by: Optional[str] = None       # Audit only — stored in DB, not returned via API
    resources: Optional[list] = None
    confidence: Optional[float] = None
    session_hash: Optional[str] = None       # Populated only if SAFETY_SESSION_HASH_ENABLED=True
    alert_sent: bool = False


# -------------------------------------------------------
# Filter / Query Params
# -------------------------------------------------------

class TriageFilterParams(BaseModel):
    """
    Query parameters for GET /triage (admin list endpoint).
    Passed as query parameters, not a request body.
    """
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Results per page")
    user_id: Optional[str] = Field(default=None, description="Filter by specific user_id")
    crisis_type: Optional[str] = Field(
        default=None,
        description="Filter by crisis type: self_harm | eating_disorder | acute_medical",
    )
    active_only: bool = Field(
        default=True,
        description="If True, exclude archived (archived_at IS NOT NULL) events",
    )
