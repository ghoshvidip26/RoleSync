from pydantic import BaseModel, ConfigDict, EmailStr


class RecipientCreate(BaseModel):
    name: str | None = None
    email: EmailStr
    role: str | None = None


class Recipient(RecipientCreate):
    model_config = ConfigDict(from_attributes=True)
    id: str
