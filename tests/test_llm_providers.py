"""Provider selection and structured-output handling, tested offline with fake transports."""

from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm import client as llm
from app.llm.client import (
    AnthropicClient,
    LLMAuthError,
    LLMError,
    OpenAICompatibleClient,
    build_llm,
    resolve_provider,
)


class Out(BaseModel):
    answer: str


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch):
    """Keys in the developer's real environment must not leak into these tests."""
    for field in Settings.model_fields:
        if field.endswith(("_api_key", "_base_url")) or field in ("llm_provider", "llm_model"):
            monkeypatch.delenv(field.upper(), raising=False)
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _settings(**env) -> Settings:
    return Settings(_env_file=None, **env)


@pytest.mark.parametrize("env,expected", [
    ({}, "rule_based"),
    ({"openai_api_key": "sk-1"}, "openai"),
    ({"GEMINI_API_KEY": "g-1"}, "gemini"),
    ({"GOOGLE_API_KEY": "g-1"}, "gemini"),
    ({"groq_api_key": "x"}, "groq"),
    ({"anthropic_api_key": "a", "openai_api_key": "o"}, "anthropic"),  # AUTO_ORDER precedence
    ({"custom_llm_base_url": "http://localhost:1234/v1"}, "custom"),
    ({"llm_provider": "ollama"}, "ollama"),
    ({"llm_provider": "rule_based", "openai_api_key": "sk-1"}, "rule_based"),  # explicit choice wins
])
def test_auto_detects_provider_from_keys(monkeypatch, env, expected):
    aliases = {k: v for k, v in env.items() if k.isupper()}
    for k, v in aliases.items():
        monkeypatch.setenv(k, v)
    s = _settings(**{k: v for k, v in env.items() if not k.isupper()})
    assert resolve_provider(s) == expected


def test_build_llm_uses_provider_defaults_and_overrides():
    c = build_llm(_settings(groq_api_key="x"))
    assert (c.provider, c.model) == ("groq", "llama-3.3-70b-versatile")
    assert c._url == "https://api.groq.com/openai/v1/chat/completions"
    c = build_llm(_settings(openai_api_key="x", llm_model="gpt-4.1"))
    assert (c.provider, c.model) == ("openai", "gpt-4.1")
    assert build_llm(_settings()) is None
    with pytest.raises(LLMError):
        build_llm(_settings(llm_provider="anthropic"))  # explicit provider without its key
    with pytest.raises(LLMError):
        build_llm(_settings(llm_provider="custom", custom_llm_base_url="http://x/v1"))  # custom needs LLM_MODEL


def _reply(content: str, status: int = 200) -> httpx.Response:
    body = {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    return httpx.Response(status, json=body if status == 200 else {"error": "bad"},
                          request=httpx.Request("POST", "http://x/chat/completions"))


def test_openai_compatible_falls_back_to_json_mode(monkeypatch):
    sent = []

    def fake_post(url, json, headers, timeout):
        sent.append(json["response_format"]["type"])
        return _reply("", 400) if json["response_format"]["type"] == "json_schema" else _reply('{"answer": "42"}')

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    c = OpenAICompatibleClient("gemini", "m", "k", "http://x", 5)
    out, usage = c.structured([{"role": "user", "content": "q"}], Out)
    assert out.answer == "42" and usage.total == 5 and sent == ["json_schema", "json_object"]
    c.structured([{"role": "user", "content": "q"}], Out)
    assert sent[-1] == "json_object"  # remembers the server lacks schema support


def test_invalid_json_is_corrected_once_then_fails(monkeypatch):
    replies = iter([_reply('{"wrong": 1}'), _reply('```json\n{"answer": "fixed"}\n```')])
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: next(replies))
    out, _ = OpenAICompatibleClient("openai", "m", "k", "http://x", 5).structured([{"role": "user", "content": "q"}], Out)
    assert out.answer == "fixed"
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: _reply("not json"))
    with pytest.raises(LLMError, match="invalid structured output"):
        OpenAICompatibleClient("openai", "m", "k", "http://x", 5).structured([{"role": "user", "content": "q"}], Out)


def test_http_errors_become_llm_errors(monkeypatch):
    def boom(*a, **k):
        return httpx.Response(401, json={"error": "bad key"}, request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(llm.httpx, "post", boom)
    c = OpenAICompatibleClient("openai", "m", "k", "http://x", 5)
    c._schema_mode = False
    with pytest.raises(LLMAuthError, match="HTTP 401"):
        c.structured([{"role": "user", "content": "q"}], Out)


def _limited(retry_after: str) -> httpx.Response:
    return httpx.Response(429, headers={"retry-after": retry_after}, json={"error": "rate limit"},
                          request=httpx.Request("POST", "http://x"))


def test_short_rate_limits_are_waited_out(monkeypatch):
    replies = iter([_limited("0.01"), _limited("0.01"), _reply('{"answer": "after wait"}')])
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: next(replies))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    c = OpenAICompatibleClient("groq", "m", "k", "http://x", 5)
    c._schema_mode = False
    out, _ = c.structured([{"role": "user", "content": "q"}], Out)
    assert out.answer == "after wait"


def test_exhausted_quota_stops_calling_the_provider(settings, monkeypatch):
    from app.llm.client import LLMQuotaError
    from app.service import InsightFlowService

    calls = []

    def daily_limit(*a, **k):
        calls.append(1)
        return _limited("3600")  # e.g. the free-tier daily token quota is used up

    monkeypatch.setattr(llm.httpx, "post", daily_limit)
    c = OpenAICompatibleClient("groq", "m", "k", "http://x", 5)
    with pytest.raises(LLMQuotaError):
        c.structured([{"role": "user", "content": "q"}], Out)
    svc = InsightFlowService(settings, llm=OpenAICompatibleClient("groq", "m", "k", "http://x", 5))
    calls.clear()
    v = svc.start_investigation(svc.example_dataset().dataset_id, "Why did European revenue decrease in Q3?")
    assert v.status == "completed" and svc.deps.llm is None
    assert len(calls) == 1  # one failed call, then the circuit breaker kept the session offline


def test_rejected_key_trips_the_circuit_breaker(settings, monkeypatch):
    from app.service import InsightFlowService

    def unauthorized(*a, **k):
        return httpx.Response(401, json={"error": "bad key"}, request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(llm.httpx, "post", unauthorized)
    svc = InsightFlowService(settings, llm=OpenAICompatibleClient("openai", "m", "bad", "http://x", 5))
    v = svc.start_investigation(svc.example_dataset().dataset_id, "Why did European revenue decrease in Q3?")
    assert v.status == "completed" and svc.deps.llm is None  # finished on rule_based, provider switched off
    assert any("LLM failed" in e for e in v.errors)


class _FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.calls, self.stop_reason = [], stop_reason

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop_reason, parsed_output=Out(answer="claude"),
                               usage=SimpleNamespace(input_tokens=7, output_tokens=3))


def _anthropic(model: str, stop_reason="end_turn") -> tuple[AnthropicClient, _FakeMessages, _FakeMessages]:
    c = AnthropicClient(model, "key", 5)
    plain, beta = _FakeMessages(stop_reason), _FakeMessages(stop_reason)
    c._client = SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))
    return c, plain, beta


def test_anthropic_uses_parse_with_schema_and_system_split():
    c, plain, beta = _anthropic("claude-opus-5")
    out, usage = c.structured([{"role": "system", "content": "policy"}, {"role": "user", "content": "q"}], Out)
    assert out.answer == "claude" and usage.total == 10
    call = beta.calls[0]  # Opus 5 goes through the beta path with server-side refusal fallbacks
    assert call["output_format"] is Out and call["system"] == "policy"
    assert call["messages"] == [{"role": "user", "content": "q"}] and call["fallbacks"] == "default"
    c, plain, beta = _anthropic("claude-sonnet-5")
    c.structured([{"role": "user", "content": "q"}], Out)
    assert plain.calls and not beta.calls


def test_anthropic_refusal_triggers_fallback():
    c, _, _ = _anthropic("claude-sonnet-5", stop_reason="refusal")
    with pytest.raises(LLMError, match="refusal"):
        c.structured([{"role": "user", "content": "q"}], Out)
