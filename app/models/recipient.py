from pydantic import BaseModel, ConfigDict


class RecipientCreate(BaseModel):
    name: str | None = None
    email: str
    role: str | None = None


class Recipient(RecipientCreate):
    model_config = ConfigDict(from_attributes=True)
    id: str
