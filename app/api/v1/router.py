"""
============================================================
app/api/v1/router.py — API v1 Route Aggregator
============================================================
"""
from fastapi import APIRouter

from app.api.v1 import admin, auth, chat, compliance, episodes, memories, session, system, triage, user

api_router = APIRouter(prefix="/api/v1")

# Phase 2: Public authentication endpoints
api_router.include_router(auth.router, tags=["Authentication"])

# Phase 1+: Core chat (JWT-protected)
api_router.include_router(chat.router, tags=["Chat"])

# Phase 1+: User profile & GDPR management (JWT-protected)
api_router.include_router(user.router, tags=["User Management"])

# Phase 3: Session lifecycle (JWT-protected)
api_router.include_router(session.router, tags=["Session Lifecycle"])

# Phase 4: Layer 3 Semantic Memory CRUD (JWT-protected)
api_router.include_router(memories.router, tags=["Semantic Memory"])

# Phase 5: Layer 2 Episode list + semantic search (JWT-protected)
api_router.include_router(episodes.router, tags=["Episodes"])

# Phase 6: Crisis triage event admin endpoints (JWT-protected)
api_router.include_router(triage.router, tags=["Safety Triage"])

# Phase 7: System admin endpoints — consolidation trigger + status (JWT-protected)
api_router.include_router(system.router, tags=["System Admin"])

# Phase 8: GDPR & Compliance (JWT-protected)
api_router.include_router(compliance.router, tags=["Compliance"])

# Phase 8: Admin User Management (JWT-protected)
api_router.include_router(admin.router, tags=["Admin"])
