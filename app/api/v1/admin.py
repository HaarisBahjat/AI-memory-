"""
============================================================
app/api/v1/admin.py — Admin User Management API
============================================================
"""
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import math

from app.api.deps import require_admin
from app.core.database import get_db
from app.schemas.user import AdminUserResponse, PaginatedAdminUsersResponse, UpdateUserRoleRequest

log = structlog.get_logger(__name__)

# Protect the entire router
router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(require_admin)])


@router.get("/users", response_model=PaginatedAdminUsersResponse, summary="List all registered users (Admin)")
async def list_users(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size

    # Count total users
    count_res = await db.execute(text("SELECT COUNT(*) FROM users"))
    total = count_res.scalar() or 0

    # Fetch users (Safe fields only)
    res = await db.execute(
        text("""
            SELECT user_id, email, is_admin, created_at
            FROM users
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        {"limit": page_size, "offset": offset}
    )
    rows = res.mappings().all()

    items = [AdminUserResponse(**dict(row)) for row in rows]

    return PaginatedAdminUsersResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size
    )


@router.patch("/users/{user_id}/role", response_model=AdminUserResponse, summary="Update a user's admin role (Admin)")
async def update_user_role(
    user_id: str,
    body: UpdateUserRoleRequest,
    db: AsyncSession = Depends(get_db),
):
    # If demoting to non-admin, check if they are the last admin
    if body.is_admin is False:
        count_res = await db.execute(text("SELECT COUNT(*) FROM users WHERE is_admin = TRUE"))
        admin_count = count_res.scalar() or 0
        
        # We need to know if the target user is currently an admin
        target_res = await db.execute(text("SELECT is_admin FROM users WHERE user_id = :uid"), {"uid": user_id})
        target_row = target_res.fetchone()
        
        if not target_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
            
        if target_row[0] is True and admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove the last remaining admin."
            )

    # Update the user
    update_res = await db.execute(
        text("""
            UPDATE users
            SET is_admin = :is_admin
            WHERE user_id = :uid
            RETURNING user_id, email, is_admin, created_at
        """),
        {"uid": user_id, "is_admin": body.is_admin}
    )
    row = update_res.mappings().first()
    await db.commit()

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    log.info("User role updated", target_user_id=user_id, is_admin=body.is_admin)

    return AdminUserResponse(**dict(row))
