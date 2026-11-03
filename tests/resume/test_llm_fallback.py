"""Local model first; Groq only when it fails. Groq is faked here: nothing in this file touches the network."""
import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from app.agents import resume_tailor, review
from app.services import llm_fallback as lf

KEY = "gsk_test_secret_1234567890"


class Answer(BaseModel):
    answer: str


class BadRequest(Exception):
    status_code = 400


class FakeGroq:
    """Stands in for the OpenAI client pointed at Groq. `script` items are replies (str) or exceptions."""

    def __init__(self, *script):
        self.script, self.calls = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=item))])


@pytest.fixture
def groq(monkeypatch):
    """Groq is configured and replaced by a fake whose script the test sets."""
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    holder = SimpleNamespace(client=FakeGroq('{"answer": "from groq"}'))
    monkeypatch.setattr(lf, "_client", lambda: holder.client)
    return holder


def local(result=None, error=None):
    m = MagicMock()
    m.invoke.side_effect = error if error else None
    if not error:
        m.invoke.return_value = result
    return m


# ------------------------------------------------------------------ when it is on

def test_off_without_a_key_and_when_switched_off(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert not lf.groq_available()
    monkeypatch.setenv("GROQ_API_KEY", "  ")
    assert not lf.groq_available()
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    assert lf.groq_available()
    for off in ("off", "OFF", "0", "false", "no"):
        monkeypatch.setenv("LLM_FALLBACK", off)
        assert not lf.groq_available()
    monkeypatch.setenv("LLM_FALLBACK", "on")
    assert lf.groq_available()


def test_without_a_key_the_local_error_is_raised_untouched(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    boom = RuntimeError("token repeat limit reached")
    with pytest.raises(RuntimeError, match="token repeat"):
        lf.FallbackChat(local(error=boom)).invoke([("human", "hi")], format={"title": "X"})
    primary = MagicMock()
    primary.with_structured_output.return_value.invoke.side_effect = boom
    with pytest.raises(RuntimeError, match="token repeat"):
        lf.FallbackChat(primary).with_structured_output(Answer).invoke([("human", "hi")])


# --------------------------------------------------------------- local model works

def test_a_working_local_model_never_triggers_groq(groq):
    primary = local(SimpleNamespace(content="local answer"))
    out = lf.FallbackChat(primary).invoke([("system", "s"), ("human", "h")], format={"title": "X"})
    assert out.content == "local answer" and groq.client.calls == []
    primary.invoke.assert_called_once_with([("system", "s"), ("human", "h")], format={"title": "X"})


def test_plain_calls_do_not_pass_a_format(groq):
    primary = local(SimpleNamespace(content="ok"))
    lf.FallbackChat(primary).invoke([("human", "h")])
    primary.invoke.assert_called_once_with([("human", "h")])


def test_structured_output_uses_the_local_model_first(groq):
    structured = MagicMock()
    structured.invoke.return_value = Answer(answer="local")
    primary = MagicMock()
    primary.with_structured_output.return_value = structured
    assert lf.FallbackChat(primary).with_structured_output(Answer).invoke([("human", "h")]) == Answer(answer="local")
    assert groq.client.calls == []


def test_other_attributes_come_from_the_local_model():
    primary = SimpleNamespace(model="qwen:latest")
    assert lf.FallbackChat(primary).model == "qwen:latest"


# ------------------------------------------------------------------ local model fails

def test_a_failed_local_call_is_answered_by_groq_with_json_schema(groq, caplog):
    caplog.set_level(logging.WARNING)
    groq.client = FakeGroq('{"requirements": ["a"]}')
    schema = {"title": "RequirementList", "type": "object"}
    out = lf.FallbackChat(local(error=RuntimeError("prediction aborted, token repeat limit reached"))).invoke(
        [("system", "be brief"), ("human", "list them"), ("assistant", "ok")], format=schema)
    assert out.content == '{"requirements": ["a"]}'
    (call,) = groq.client.calls
    assert call["model"] == lf.DEFAULT_GROQ_MODEL
    assert call["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "list them"},
                                {"role": "assistant", "content": "ok"}]
    assert call["response_format"] == {"type": "json_schema", "json_schema": {"name": "RequirementList", "schema": schema, "strict": False}}
    assert "answering with Groq" in caplog.text and "token repeat limit" in caplog.text


def test_the_model_can_be_chosen(groq, monkeypatch):
    monkeypatch.setenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
    lf.FallbackChat(local(error=RuntimeError("x"))).invoke([("human", "h")])
    assert groq.client.calls[0]["model"] == "openai/gpt-oss-20b"


def test_a_plain_failed_call_asks_groq_without_a_schema(groq):
    groq.client = FakeGroq("plain text")
    out = lf.FallbackChat(local(error=RuntimeError("down"))).invoke([("human", "h")])
    assert out.content == "plain text" and "response_format" not in groq.client.calls[0]


def test_a_model_without_schema_mode_gets_json_mode_and_the_schema_in_the_prompt(groq):
    groq.client = FakeGroq(BadRequest("response_format json_schema is not supported"), '{"answer": "x"}')
    schema = {"title": "Answer", "type": "object"}
    out = lf.FallbackChat(local(error=RuntimeError("down"))).invoke([("human", "h")], format=schema)
    assert out.content == '{"answer": "x"}'
    first, second = groq.client.calls
    assert first["response_format"]["type"] == "json_schema" and second["response_format"] == {"type": "json_object"}
    assert second["messages"][0]["role"] == "system" and json.dumps(schema) in second["messages"][0]["content"]


def test_a_real_groq_error_is_not_mistaken_for_a_missing_schema_mode(groq):
    err = RuntimeError("rate limited")
    err.status_code = 429
    groq.client = FakeGroq(err)
    original = RuntimeError("local down")
    with pytest.raises(RuntimeError, match="local down"):                  # the local error is what the caller sees
        lf.FallbackChat(local(error=original)).invoke([("human", "h")], format={"title": "X"})
    assert len(groq.client.calls) == 1


def test_structured_output_falls_back_and_parses_into_the_schema(groq):
    primary = MagicMock()
    primary.with_structured_output.return_value.invoke.side_effect = RuntimeError("model not found")
    groq.client = FakeGroq('{"answer": "from groq"}')
    out = lf.FallbackChat(primary).with_structured_output(Answer).invoke([("system", "s"), ("human", "h")])
    assert out == Answer(answer="from groq")
    assert groq.client.calls[0]["response_format"]["json_schema"]["name"] == "Answer"


def test_unparseable_groq_json_raises_the_original_local_error(groq):
    primary = MagicMock()
    primary.with_structured_output.return_value.invoke.side_effect = RuntimeError("model not found")
    groq.client = FakeGroq("not json at all")
    with pytest.raises(RuntimeError, match="model not found"):
        lf.FallbackChat(primary).with_structured_output(Answer).invoke([("human", "h")])


def test_the_key_never_appears_in_logs_or_errors(groq, caplog):
    caplog.set_level(logging.WARNING)
    groq.client = FakeGroq(RuntimeError(f"401 invalid api key {KEY}"))
    with pytest.raises(RuntimeError):
        lf.FallbackChat(local(error=RuntimeError("down"))).invoke([("human", "h")])
    assert KEY not in caplog.text and "***" in caplog.text


def test_langchain_message_objects_are_accepted(groq):
    from langchain_core.messages import HumanMessage, SystemMessage
    groq.client = FakeGroq("ok")
    lf.FallbackChat(local(error=RuntimeError("down"))).invoke([SystemMessage(content="s"), HumanMessage(content="h")])
    assert [m["role"] for m in groq.client.calls[0]["messages"]] == ["system", "user"]


def test_groq_invoke_refuses_when_not_configured(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        lf.groq_invoke([("human", "h")])


# ---------------------------------------------------- wired into the three call sites

def test_the_three_model_objects_use_the_fallback():
    from app.services import outreach_agent
    assert isinstance(review.review_agent, lf.FallbackChat)
    assert isinstance(resume_tailor.tailor_agent, lf.FallbackChat)
    assert isinstance(outreach_agent.get_llm(), lf.FallbackChat)
    assert review.review_agent.model and resume_tailor.tailor_agent.num_ctx == 4096           # local settings unchanged


class _Req(BaseModel):
    requirements: list[str]


def test_the_review_survives_a_local_failure_through_groq(groq, monkeypatch):
    groq.client = FakeGroq('{"requirements": ["React Native"]}')
    monkeypatch.setattr(review, "review_agent", lf.FallbackChat(local(error=RuntimeError("token repeat limit reached"))))
    assert review._generate(_Req, [("system", "s"), ("human", "h")]) == _Req(requirements=["React Native"])


def test_the_review_tries_groq_once_more_when_the_local_model_only_gives_unusable_json(groq, monkeypatch):
    groq.client = FakeGroq('{"requirements": ["from groq"]}')
    bad = local(SimpleNamespace(content="this is not json"))
    monkeypatch.setattr(review, "review_agent", lf.FallbackChat(bad))
    assert review._generate(_Req, [("system", "s"), ("human", "h")]) == _Req(requirements=["from groq"])
    assert bad.invoke.call_count == review.MAX_ATTEMPTS and len(groq.client.calls) == 1


def test_the_review_still_fails_clearly_when_both_models_fail(groq, monkeypatch):
    groq.client = FakeGroq('{"wrong": 1}')
    monkeypatch.setattr(review, "review_agent", lf.FallbackChat(local(SimpleNamespace(content="nope"))))
    with pytest.raises(review.ReviewError, match="invalid response"):
        review._generate(_Req, [("system", "s"), ("human", "h")])


def test_without_groq_the_review_behaves_exactly_as_before(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    bad = local(SimpleNamespace(content="nope"))
    monkeypatch.setattr(review, "review_agent", lf.FallbackChat(bad))
    with pytest.raises(review.ReviewError, match="after 2 attempts"):
        review._generate(_Req, [("system", "s"), ("human", "h")])
    assert bad.invoke.call_count == 2


def test_tailoring_text_from_groq_is_still_verified(groq, monkeypatch):
    facts = {"anchor_terms": {"react"}, "role": "React Native Developer", "title": "Dev", "company": "Acme", "skills": ["React"], "bullets": []}
    groq.client = FakeGroq(json.dumps({"headline": "Quantum blockchain guru", "summary": "I led forty engineers at Google on TensorFlow."}))
    monkeypatch.setattr(resume_tailor, "tailor_agent", lf.FallbackChat(local(error=RuntimeError("down"))))
    assert resume_tailor._llm_prose("React Native job", facts) is None          # unsupported claims are rejected


def test_the_real_client_points_at_groq_waits_out_rate_limits_and_never_logs_the_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", f"  {KEY}  ")
    client = lf._client()                                    # building it makes no network call
    assert str(client.base_url).rstrip("/") == lf.GROQ_BASE_URL and client.max_retries == lf.GROQ_RETRIES == 3
    assert client.api_key == KEY and client.timeout == lf.GROQ_TIMEOUT_SECONDS


def test_when_both_models_fail_the_error_says_so_and_why(groq):
    err = RuntimeError("rate limit reached for model; try again in 20s")
    err.status_code = 429
    groq.client = FakeGroq(err, err, err, err)
    with pytest.raises(RuntimeError) as shown:
        lf.FallbackChat(local(error=RuntimeError("prediction aborted, token repeat limit reached"))).invoke([("human", "h")], format={"title": "X"})
    text = str(shown.value)
    assert "token repeat limit reached" in text and "the Groq fallback failed too" in text and "rate limit reached" in text
    assert KEY not in text


def test_the_structured_path_reports_both_failures_too(groq):
    primary = MagicMock()
    primary.with_structured_output.return_value.invoke.side_effect = RuntimeError("model not found")
    groq.client = FakeGroq("not json")
    with pytest.raises(RuntimeError, match=r"model not found.*Groq fallback failed too"):
        lf.FallbackChat(primary).with_structured_output(Answer).invoke([("human", "h")])
