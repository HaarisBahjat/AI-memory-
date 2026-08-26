"""
============================================================
app/api/v1/compliance.py — GDPR Compliance Endpoints
============================================================
"""
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis
from typing import Any

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.redis_client import get_redis
from app.core.security import verify_password
from app.schemas.auth import CurrentUser
from app.schemas.compliance import PurgeRequest

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/compliance", tags=["Compliance"])


@router.get("/export", summary="Export all user data (GDPR Right to Data Portability)")
async def export_data(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    uid = current_user.user_id

    # Fetch User Profile
    res_user = await db.execute(text("SELECT * FROM users WHERE user_id = :uid"), {"uid": uid})
    user = res_user.mappings().first()

    # Fetch Episodes
    res_episodes = await db.execute(text("SELECT * FROM episodes WHERE user_id = :uid ORDER BY timestamp DESC"), {"uid": uid})
    episodes = [dict(r) for r in res_episodes.mappings().all()]

    # Fetch Semantic Memories
    res_memories = await db.execute(text("SELECT * FROM semantic_memories WHERE user_id = :uid"), {"uid": uid})
    memories = [dict(r) for r in res_memories.mappings().all()]

    # Fetch Knowledge Nodes
    res_nodes = await db.execute(text("SELECT * FROM knowledge_nodes WHERE user_id = :uid"), {"uid": uid})
    nodes = [dict(r) for r in res_nodes.mappings().all()]

    # Fetch Knowledge Edges
    res_edges = await db.execute(text("SELECT * FROM knowledge_edges WHERE user_id = :uid"), {"uid": uid})
    edges = [dict(r) for r in res_edges.mappings().all()]

    # Fetch Triage Events
    res_triage = await db.execute(text("SELECT * FROM triage_events WHERE user_id = :uid"), {"uid": uid})
    triage = [dict(r) for r in res_triage.mappings().all()]

    # Convert UUIDs/datetimes/etc to strings for JSON serialization if needed, 
    # but FastAPI does a good job via jsonable_encoder automatically.
    # However, vector columns might fail to serialize. Let's exclude vectors if they crash, 
    # but we can try letting FastAPI handle it, or explicitly pop the 'embedding' column.
    
    for m in memories:
        m.pop("embedding", None)
    for n in nodes:
        n.pop("embedding", None)

    return {
        "user": dict(user) if user else None,
        "episodes": episodes,
        "semantic_memories": memories,
        "knowledge_graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "triage_events": triage,
    }


@router.delete("/purge", summary="Permanently delete user data (GDPR Right to be Forgotten)")
async def purge_data(
    body: PurgeRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
):
    uid = current_user.user_id

    # 1. Verify Password
    res_user = await db.execute(text("SELECT password_hash FROM users WHERE user_id = :uid"), {"uid": uid})
    row = res_user.fetchone()
    if not row or not verify_password(body.password, row[0]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password. Purge aborted.",
        )

    # 2. Hard Delete from Postgres in reverse dependency order
    tables = [
        "triage_events",
        "knowledge_edges",
        "knowledge_nodes",
        "semantic_memories",
        "episodes",
        "users",
    ]
    
    for table in tables:
        await db.execute(text(f"DELETE FROM {table} WHERE user_id = :uid"), {"uid": uid})
    
    await db.commit()

    # 3. Purge Redis session cache
    redis_key = f"session:{uid}"
    await redis.delete(redis_key)

    log.info("User data purged", user_id=uid)
    return {"status": "success", "message": "All data has been permanently deleted."}
