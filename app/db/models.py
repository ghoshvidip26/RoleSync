"""Table names and thin row-level helpers over the Supabase client."""
from datetime import datetime
from enum import Enum
import json
from typing import Any

from app.db.database import get_client

PROFILES = "profiles"
COMPANIES = "companies"
RECIPIENTS = "recipients"
EMAILS = "emails"
RESUME_DRAFTS = "resume_drafts"

KNOWN_COLUMNS: dict[str, set[str]] = {
    PROFILES: {"id", "name", "email", "github", "portfolio", "linkedin", "resume"},
    COMPANIES: {"id", "name", "website"},
    RECIPIENTS: {"id", "name", "email", "role"},
    EMAILS: {
        "id",
        "profile_id",
        "company_id",
        "recipient_id",
        "subject",
        "body",
        "status",
        "scheduled_at",
        "sent_at",
        "error",
        "resume_draft_id",
    },
}


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v.value if isinstance(v, Enum) else v.isoformat() if isinstance(v, datetime) else v
        for k, v in data.items()
    }


def _prepare_for_db(table: str, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Filter columns to what the table schema supports and preserve extra profile metadata."""
    cleaned = _clean(data)
    known = KNOWN_COLUMNS.get(table)
    if not known:
        return cleaned, {}

    db_data: dict[str, Any] = {}
    extra_data: dict[str, Any] = {}
    for k, v in cleaned.items():
        if k in known:
            db_data[k] = v
        else:
            extra_data[k] = v

    if table == PROFILES and extra_data:
        # If resume column is not already populated, serialize extra profile metadata into it
        if not db_data.get("resume"):
            db_data["resume"] = json.dumps(extra_data)

    return db_data, extra_data


def _unpack_row(table: str, row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None or table != PROFILES:
        return row
    result = dict(row)
    resume_val = result.get("resume")
    if isinstance(resume_val, str) and resume_val.strip().startswith("{"):
        try:
            extra = json.loads(resume_val)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if k not in result or result[k] is None:
                        result[k] = v
        except Exception:
            pass
    return result


def insert(table: str, data: dict[str, Any]) -> dict[str, Any]:
    db_data, extra = _prepare_for_db(table, data)
    inserted = get_client().table(table).insert(db_data).execute().data[0]
    result = _unpack_row(table, inserted)
    if extra and result:
        for k, v in extra.items():
            if k not in result or result[k] is None:
                result[k] = v
    return result


def fetch(table: str, id_: str) -> dict[str, Any] | None:
    rows = get_client().table(table).select("*").eq("id", id_).limit(1).execute().data
    return _unpack_row(table, rows[0]) if rows else None


def fetch_all(table: str, **filters: Any) -> list[dict[str, Any]]:
    query = get_client().table(table).select("*")
    cleaned_filters, _ = _prepare_for_db(table, filters)
    for column, value in cleaned_filters.items():
        query = query.eq(column, value)
    rows = query.execute().data
    return [_unpack_row(table, r) for r in rows]


def claim(table: str, id_: str, from_status: str, to_status: str, data: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Atomically move a row from `from_status` to `to_status`: one UPDATE ... WHERE id = ? AND status = ?.

    Returns the updated row, or None when the row was not in `from_status` (someone else got there first).
    This is what keeps two simultaneous approvals from both sending the same email.
    """
    db_data, _ = _prepare_for_db(table, {**(data or {}), "status": to_status})
    rows = get_client().table(table).update(db_data).eq("id", id_).eq("status", str(getattr(from_status, "value", from_status))).execute().data
    return _unpack_row(table, rows[0]) if rows else None


def update(table: str, id_: str, data: dict[str, Any]) -> dict[str, Any]:
    db_data, extra = _prepare_for_db(table, data)
    updated = get_client().table(table).update(db_data).eq("id", id_).execute().data[0]
    result = _unpack_row(table, updated)
    if extra and result:
        for k, v in extra.items():
            if k not in result or result[k] is None:
                result[k] = v
    return result

