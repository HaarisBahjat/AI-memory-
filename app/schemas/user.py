from datetime import datetime
from pydantic import BaseModel, Field

class AdminUserResponse(BaseModel):
    id: str = Field(alias="user_id")
    email: str
    is_admin: bool
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}

class UpdateUserRoleRequest(BaseModel):
    is_admin: bool

class PaginatedAdminUsersResponse(BaseModel):
    items: list[AdminUserResponse]
    total: int
    page: int
    page_size: int
