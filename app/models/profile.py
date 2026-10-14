from pydantic import BaseModel, ConfigDict,EmailStr


class ProfileCreate(BaseModel):
    name: str
    # email: str
    email: EmailStr
    github: str | None = None
    portfolio: str | None = None
    linkedin: str | None = None
    resume: str | None = None
    summary: str | None = None
    skills: list[str] | None = None
    programming_languages: list[str] | None = None
    frameworks: list[str] | None = None
    databases_tools: list[str] | None = None
    experience: list[str] | None = None
    projects: list[str] | None = None
    achievements: list[str] | None = None
    education: list[str] | None = None
    target_roles: list[str] | None = None


class Profile(ProfileCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str


class ExtractedProfile(ProfileCreate):
    """Structured candidate profile returned for user review prior to persistence."""
    raw_resume_text: str | None = None

