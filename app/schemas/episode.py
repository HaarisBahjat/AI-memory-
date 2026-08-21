"""
============================================================
app/schemas/episode.py -- Phase 5 Episode Pydantic Schemas
============================================================
PURPOSE:
    Strictly typed request/response contracts for the
    /api/v1/episodes endpoints.

CONNECTED TO:
    Phase 5 -> episode_service.py (creates episode rows)
    Phase 5 -> app/api/v1/episodes.py (API layer uses these)
    Phase 7 -> Consolidation reads extracted_metrics field
============================================================
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ExtractedMetrics(BaseModel):
    moodScore: Optional[float] = Field(None, ge=1, le=10)
    physicalSymptoms: list[str] = Field(default_factory=list)
    primaryStressor: Optional[str] = Field(None, max_length=300)
    sleepHoursLogged: Optional[float] = Field(None, ge=0, le=24)
    anxietyLevel: Optional[float] = Field(None, ge=1, le=10)
    energyLevel: Optional[float] = Field(None, ge=1, le=10)
    biometrics: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class EpisodeResponse(BaseModel):
    id: str = Field(..., description="Episode UUID")
    user_id: str
    timestamp: datetime
    session_summary: str
    extracted_metrics: ExtractedMetrics
    archived_at: Optional[datetime] = None
    similarity: Optional[float] = Field(None, ge=0.0, le=1.0)

    model_config = {"from_attributes": True}


class EpisodeListResponse(BaseModel):
    items: list[EpisodeResponse]
    total: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    per_page: int = Field(..., ge=1, le=100)
    active_only: bool = True


class EpisodeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(5, ge=1, le=20)
    active_only: bool = True
