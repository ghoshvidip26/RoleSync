from pydantic import BaseModel, ConfigDict


class CompanyCreate(BaseModel):
    name: str
    website: str | None = None


class Company(CompanyCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
