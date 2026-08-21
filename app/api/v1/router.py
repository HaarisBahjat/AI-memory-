"""
============================================================
app/api/v1/router.py — API v1 Route Aggregator
============================================================
PURPOSE:
    Registers all v1 sub-routers under the /api/v1 prefix.

    Phase 1: chat, user
    Phase 2: auth (register / login / refresh / logout)
    Phase 3: session (session/active, session/end)
    Phase 4: memories (Layer 3 Semantic Memory CRUD)
    Phase 5: episodes (Layer 2 Episode list + semantic search)
    Phase 8: compliance (added here when implemented)

CONNECTED TO:
    main.py → app.include_router(api_router)
============================================================
"""
from fastapi import APIRouter

from app.api.v1 import auth, chat, episodes, memories, session, user

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
