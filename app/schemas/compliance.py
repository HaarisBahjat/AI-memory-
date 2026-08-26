from pydantic import BaseModel, Field

class PurgeRequest(BaseModel):
    """
    Requires the user to confirm their password before destructive deletion.
    """
    password: str = Field(..., description="The user's current password for verification.")
