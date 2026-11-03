"""Role recommendation: which of a set of known roles fit a candidate, and what each one is missing.

Everything here is word matching against the candidate's own resume, so a recommendation can never
credit a skill the resume does not show.
"""
from app.agents.review import _terms, _verdict

ROLES = {
  "roles": [
    {
      "role_id": "backend_engineer",
      "title": "Backend Engineer",
      "required_skills": [
        "Python",
        "FastAPI",
        "Node.js",
        "PostgreSQL",
        "REST APIs"
      ],
      "preferred_skills": [
        "Docker",
        "Redis",
        "Kafka",
        "Kubernetes"
      ]
    },
    {
      "role_id": "react_native_developer",
      "title": "React Native Developer",
      "required_skills": [
        "React Native",
        "JavaScript",
        "TypeScript",
        "REST APIs"
      ],
      "preferred_skills": [
        "Expo",
        "Firebase",
        "React Navigation"
      ]
    },
    {
      "role_id": "fullstack_developer",
      "title": "Full Stack Developer",
      "required_skills": [
        "React",
        "Node.js",
        "JavaScript",
        "PostgreSQL"
      ],
      "preferred_skills": [
        "Next.js",
        "Docker",
        "CI/CD",
        "Testing"
      ]
    }
  ]
}


PROFILE_FIELDS = ("skills", "programming_languages", "frameworks", "databases_tools", "experience", "projects")
REQUIRED_WEIGHT = 2   # a required skill counts twice as much as a preferred one in the score


def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(v for v in value if isinstance(v, str))
    return ""


def candidate_terms(profile: dict) -> set[str]:
    """The distinctive words in everything the candidate lists or describes."""
    text = "\n".join(_text(profile.get(field)) for field in PROFILE_FIELDS)
    return set(_terms(text.replace("/n", "\n")))


def _split_skills(skills: list[str], have: set[str]) -> tuple[list[str], list[str]]:
    """(matching, missing). A skill matches when every one of its words is on the resume.
    A skill with no distinctive words cannot be checked, so it is left out."""
    matching, missing = [], []
    for skill in skills:
        words = _terms(skill)
        if not words:
            continue
        (matching if set(words) <= have else missing).append(skill)
    return matching, missing


def score_role(role: dict, have: set[str]) -> dict:
    """How well one role fits, from the required and the preferred skills."""
    req_match, req_miss = _split_skills(role.get("required_skills", []), have)
    pref_match, pref_miss = _split_skills(role.get("preferred_skills", []), have)

    possible = REQUIRED_WEIGHT * (len(req_match) + len(req_miss)) + len(pref_match) + len(pref_miss)
    earned = REQUIRED_WEIGHT * len(req_match) + len(pref_match)
    # The label follows the required skills only, with the same thresholds as the job review.
    fit = _verdict(len(req_match), len(req_miss), 0) if (req_match or req_miss) else "weak_match"
    return {
        "role_id": role.get("role_id"),
        "title": role.get("title"),
        "fit": fit,
        "score": round(earned / possible, 2) if possible else 0.0,
        "matching": {"required": req_match, "preferred": pref_match},
        "gaps": {"required": req_miss, "preferred": pref_miss},
    }


def recommend_roles(profile: dict, roles: dict | None = None) -> list[dict]:
    """Every role scored against the profile, best fit first (equal scores keep the roles' order)."""
    have = candidate_terms(profile)
    scored = [score_role(role, have) for role in (roles or ROLES)["roles"]]
    return sorted(scored, key=lambda r: -r["score"])
