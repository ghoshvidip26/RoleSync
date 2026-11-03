"""A safety net for the local language model: if it fails, Groq answers instead.

The local model (Ollama) stays the first choice, so nothing changes while it works. When a call fails
(Ollama down, model missing, generation aborted with "token repeat limit reached", ...) the same prompt is
sent to Groq's OpenAI-compatible API, using the `openai` package that is already installed. Whatever the
answer, the callers verify it exactly as before.

Settings (environment or .env):
  GROQ_API_KEY      enables the fallback; without it nothing is ever sent anywhere
  GROQ_CHAT_MODEL   default "openai/gpt-oss-120b" (a model that supports JSON-schema output)
  LLM_FALLBACK=off  switches the fallback off even when a key is present

Privacy: a fallback call sends the prompt (resume excerpts, job text) to Groq. Every use is logged.
Only chat goes to Groq. Embeddings stay on the local model (Groq has no embeddings endpoint).
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_TIMEOUT_SECONDS = 90
# A review makes several calls in a row, which can hit Groq's per-minute limit. The client waits as long as
# Groq's retry-after header says and tries again, instead of failing the whole review.
GROQ_RETRIES = 3

_ROLES = {"system": "system", "human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}


def groq_available() -> bool:
    """True when a key is set and the fallback has not been switched off."""
    if os.getenv("LLM_FALLBACK", "").strip().lower() in {"off", "0", "false", "no"}:
        return False
    return bool((os.getenv("GROQ_API_KEY") or "").strip())


def _to_chat_messages(messages: Any) -> list[dict[str, str]]:
    """("system", text) tuples, LangChain messages, or a plain string, as OpenAI-style chat messages."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, (tuple, list)) and len(m) == 2:
            role, content = m
        else:
            role, content = getattr(m, "type", "human"), getattr(m, "content", str(m))
        out.append({"role": _ROLES.get(str(role).lower(), "user"), "content": str(content)})
    return out


def _client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["GROQ_API_KEY"].strip(), base_url=GROQ_BASE_URL, timeout=GROQ_TIMEOUT_SECONDS, max_retries=GROQ_RETRIES)


def groq_invoke(messages: Any, schema: dict | None = None) -> SimpleNamespace:
    """Ask Groq. With `schema` (a JSON schema dict) the answer is JSON; the result has `.content` like a
    LangChain message. Raises if Groq is not configured or the call fails."""
    if not groq_available():
        raise RuntimeError("Groq fallback is not configured")
    model = os.getenv("GROQ_CHAT_MODEL", "").strip() or DEFAULT_GROQ_MODEL
    chat = _to_chat_messages(messages)
    client = _client()

    if schema is None:
        reply = client.chat.completions.create(model=model, messages=chat, temperature=0.3)
        return SimpleNamespace(content=reply.choices[0].message.content or "")

    name = str(schema.get("title") or "answer")[:60]
    try:
        reply = client.chat.completions.create(
            model=model, messages=chat, temperature=0.3,
            response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": False}},
        )
    except Exception as exc:
        if getattr(exc, "status_code", None) != 400:       # not "this model has no schema mode": a real failure
            raise
        # Some Groq models only offer plain JSON mode: describe the schema in the prompt instead.
        hint = {"role": "system", "content": "Reply with only one JSON object that matches this JSON schema:\n" + json.dumps(schema)}
        reply = client.chat.completions.create(model=model, messages=[hint, *chat], temperature=0.3,
                                               response_format={"type": "json_object"})
    return SimpleNamespace(content=reply.choices[0].message.content or "")


class _StructuredFallback:
    """`llm.with_structured_output(Schema)` that falls back to Groq and parses its JSON into `Schema`."""

    def __init__(self, primary: Any, schema: Any, owner: "FallbackChat"):
        self.primary, self.schema, self.owner = primary, schema, owner

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        try:
            return self.primary.invoke(messages, **kwargs)
        except Exception as exc:
            self.owner._note(exc)
            if not groq_available():
                raise
            try:
                content = groq_invoke(messages, self.schema.model_json_schema()).content
                return self.schema.model_validate_json(content)
            except Exception as groq_exc:
                logger.warning("Groq fallback failed too: %s", _brief(groq_exc))
                raise exc from None


class FallbackChat:
    """A local chat model that uses Groq when it fails. Offers the two calls this project makes:
    `invoke(messages, format=schema_dict)` and `with_structured_output(Schema).invoke(messages)`."""

    def __init__(self, primary: Any, label: str = "local model"):
        self.primary, self.label = primary, label

    def _note(self, exc: Exception) -> None:
        if groq_available():
            logger.warning("The %s failed (%s); answering with Groq instead.", self.label, _brief(exc))

    def invoke(self, messages: Any, format: dict | None = None, **kwargs: Any) -> Any:
        try:
            return self.primary.invoke(messages, **({"format": format} if format is not None else {}), **kwargs)
        except Exception as exc:
            self._note(exc)
            if not groq_available():
                raise
            try:
                return groq_invoke(messages, format)
            except Exception as groq_exc:
                logger.warning("Groq fallback failed too: %s", _brief(groq_exc))
                raise exc from None

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _StructuredFallback:
        return _StructuredFallback(self.primary.with_structured_output(schema, **kwargs), schema, self)

    def __getattr__(self, name: str) -> Any:        # anything else (model name, ...) comes from the local model
        return getattr(self.primary, name)


def _brief(exc: Exception) -> str:
    text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if key:
        text = text.replace(key, "***")           # a key must never reach a log
    return text[:200]
