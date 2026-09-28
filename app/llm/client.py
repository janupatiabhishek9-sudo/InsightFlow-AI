"""LLM clients behind one interface. Put a key in .env and it is picked up automatically.

Providers:
- anthropic         official `anthropic` SDK, `messages.parse()` with the Pydantic schema
- openai, gemini, groq, mistral, deepseek, openrouter, together, xai, custom
                    OpenAI-compatible chat-completions APIs (JSON-schema output, falling back to JSON mode)
- ollama            local server, JSON-schema constrained output
- rule_based        no model: every reasoning stage uses its deterministic implementation

LLM_PROVIDER=auto (the default) selects the first provider in AUTO_ORDER whose API key is set.
Every structured output is validated against the stage's Pydantic schema; one invalid reply is fed
back for correction, after which the stage falls back to rule_based (see graph/nodes/common.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMUnavailableError(LLMError):
    """The provider cannot serve this session any more; callers should stop calling it."""


class LLMAuthError(LLMUnavailableError):
    """The key was rejected (401/403): retrying is pointless until the configuration changes."""


class LLMQuotaError(LLMUnavailableError):
    """Rate limit with a long wait (e.g. a free-tier daily token quota is used up)."""


# Free tiers answer bursts with HTTP 429 + retry-after. Short waits are worth it; long ones are not.
MAX_RATE_LIMIT_WAIT_SECONDS = 20
MAX_RATE_LIMIT_RETRIES = 2


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    provider: str
    model: str

    def structured(self, messages: list[dict[str, str]], schema: type[T]) -> tuple[T, Usage]: ...


@dataclass(frozen=True)
class OpenAICompatible:
    base_url: str
    key_field: str
    default_model: str


# Hosted services that speak the OpenAI chat-completions protocol. Default models are only
# starting points; set LLM_MODEL to whatever your account offers.
OPENAI_COMPATIBLE: dict[str, OpenAICompatible] = {
    "openai": OpenAICompatible("", "openai_api_key", "gpt-4o-mini"),  # base URL from OPENAI_BASE_URL
    "gemini": OpenAICompatible("https://generativelanguage.googleapis.com/v1beta/openai", "gemini_api_key", "gemini-2.5-flash"),
    "groq": OpenAICompatible("https://api.groq.com/openai/v1", "groq_api_key", "llama-3.3-70b-versatile"),
    "mistral": OpenAICompatible("https://api.mistral.ai/v1", "mistral_api_key", "mistral-small-latest"),
    "deepseek": OpenAICompatible("https://api.deepseek.com/v1", "deepseek_api_key", "deepseek-chat"),
    "openrouter": OpenAICompatible("https://openrouter.ai/api/v1", "openrouter_api_key", "openai/gpt-4o-mini"),
    "together": OpenAICompatible("https://api.together.xyz/v1", "together_api_key", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "xai": OpenAICompatible("https://api.x.ai/v1", "xai_api_key", "grok-3-mini"),
    "custom": OpenAICompatible("", "custom_llm_api_key", ""),  # base URL from CUSTOM_LLM_BASE_URL
}
ANTHROPIC_DEFAULT_MODEL = "claude-opus-5"
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"
# Which key wins when several are set and LLM_PROVIDER=auto.
AUTO_ORDER = ["anthropic", "openai", "gemini", "groq", "mistral", "deepseek", "openrouter", "together", "xai", "custom"]


def _retry_after_seconds(r: httpx.Response) -> float:
    """Seconds to wait from a 429 response (retry-after header), defaulting to a short pause."""
    value = r.headers.get("retry-after", "")
    try:
        return max(float(value), 0.5)
    except ValueError:
        return 2.0


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class _JSONRetryClient:
    """Shared validate-and-correct loop for providers that return JSON text."""

    provider = "base"

    def __init__(self, model: str, timeout: float, max_attempts: int = 2):
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts

    def _call(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> tuple[str, Usage]:
        raise NotImplementedError

    def structured(self, messages: list[dict[str, str]], schema: type[T]) -> tuple[T, Usage]:
        usage = Usage()
        msgs = list(messages)
        last_error = ""
        for _ in range(self.max_attempts):
            try:
                text, u = self._call(msgs, schema)
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                error = LLMAuthError if code in (401, 403) else LLMError
                raise error(f"{self.provider} returned HTTP {code}: {e.response.text[:300]}") from e
            except httpx.HTTPError as e:
                raise LLMError(f"{self.provider} request failed: {e}") from e
            usage.prompt_tokens += u.prompt_tokens
            usage.completion_tokens += u.completion_tokens
            try:
                return schema.model_validate_json(_strip_fences(text)), usage
            except ValidationError as e:
                last_error = str(e)[:800]
                # Feed the validation error back once so the model can correct its output.
                msgs = msgs + [{"role": "assistant", "content": text[:4000]},
                               {"role": "user", "content": f"Your JSON did not match the schema: {last_error}. Reply with corrected JSON only."}]
        raise LLMError(f"invalid structured output after {self.max_attempts} attempts: {last_error}")


class OpenAICompatibleClient(_JSONRetryClient):
    """Chat completions with JSON-schema output; falls back to JSON mode for servers without schema support."""

    def __init__(self, provider: str, model: str, api_key: str, base_url: str, timeout: float):
        super().__init__(model, timeout)
        self.provider = provider
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._schema_mode = True

    def _call(self, messages, schema):
        if self._schema_mode:
            fmt = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
            r = self._post(self._body(messages, fmt))
            if r.status_code != 400:
                return self._parse(r)
            self._schema_mode = False  # this server rejects json_schema; the schema is already in the prompt
        return self._parse(self._post(self._body(messages, {"type": "json_object"})))

    def _post(self, body: dict) -> httpx.Response:
        """POST with rate-limit handling: wait out short 429s, give up quickly on long ones."""
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            r = httpx.post(self._url, json=body, headers=self._headers, timeout=self.timeout)
            if r.status_code != 429:
                return r
            wait = _retry_after_seconds(r)
            if wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                raise LLMQuotaError(f"{self.provider} rate limit/quota reached; retry after {wait:.0f}s. "
                                    "Continuing with the rule-based reasoner for this session.")
            if attempt < MAX_RATE_LIMIT_RETRIES:
                time.sleep(wait)
        raise LLMError(f"{self.provider} is still rate limiting after {MAX_RATE_LIMIT_RETRIES} retries")

    def _body(self, messages, fmt) -> dict:
        return {"model": self.model, "messages": messages, "temperature": 0, "response_format": fmt}

    @staticmethod
    def _parse(r: httpx.Response) -> tuple[str, Usage]:
        r.raise_for_status()
        data = r.json()
        u = data.get("usage") or {}
        content = data["choices"][0]["message"].get("content") or ""
        return content, Usage(prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0))


class OllamaClient(_JSONRetryClient):
    provider = "ollama"

    def __init__(self, model: str, base_url: str, timeout: float):
        super().__init__(model, timeout)
        self._url = base_url.rstrip("/") + "/api/chat"

    def _call(self, messages, schema):
        body = {"model": self.model, "messages": messages, "stream": False, "format": schema.model_json_schema(),
                "options": {"temperature": 0}}
        r = httpx.post(self._url, json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"], Usage(prompt_tokens=data.get("prompt_eval_count", 0),
                                                 completion_tokens=data.get("eval_count", 0))


class AnthropicClient:
    """Claude via the official SDK. `parse()` constrains the reply to the Pydantic schema and validates it."""

    provider = "anthropic"
    # Models that support server-side refusal fallbacks (see the Claude API docs).
    FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5")

    def __init__(self, model: str, api_key: str, timeout: float):
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)

    def structured(self, messages: list[dict[str, str]], schema: type[T]) -> tuple[T, Usage]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        kwargs = dict(model=self.model, max_tokens=16000, system=system, messages=chat, output_format=schema)
        try:
            if self.model.startswith(self.FALLBACK_MODELS):
                # If a safety classifier declines, the API re-runs the request on a fallback model.
                response = self._client.beta.messages.parse(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                response = self._client.messages.parse(**kwargs)
        except (self._anthropic.AuthenticationError, self._anthropic.PermissionDeniedError) as e:
            raise LLMAuthError(f"anthropic rejected the key (HTTP {e.status_code})") from e
        except self._anthropic.APIStatusError as e:
            raise LLMError(f"anthropic returned HTTP {e.status_code}: {str(e.message)[:300]}") from e
        except self._anthropic.APIConnectionError as e:
            raise LLMError(f"anthropic request failed: {e}") from e
        except ValidationError as e:
            raise LLMError(f"anthropic output did not match the schema: {str(e)[:300]}") from e
        except self._anthropic.AnthropicError as e:  # e.g. a schema the structured-output feature cannot express
            raise LLMError(f"anthropic client error: {str(e)[:300]}") from e
        usage = Usage(prompt_tokens=response.usage.input_tokens, completion_tokens=response.usage.output_tokens)
        if response.stop_reason == "refusal":
            raise LLMError("anthropic declined the request (refusal)")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise LLMError(f"anthropic returned no parsable output (stop_reason={response.stop_reason})")
        return response.parsed_output, usage


def _secret(settings: Settings, field: str) -> str:
    value = getattr(settings, field, None)
    return value.get_secret_value().strip() if value is not None else ""


def resolve_provider(settings: Settings) -> str:
    """The provider that will actually be used: explicit LLM_PROVIDER, or the first configured key."""
    if settings.llm_provider != "auto":
        return settings.llm_provider
    for name in AUTO_ORDER:
        if name == "anthropic" and _secret(settings, "anthropic_api_key"):
            return name
        if name == "custom" and settings.custom_llm_base_url:
            return name
        if name in OPENAI_COMPATIBLE and name != "custom" and _secret(settings, OPENAI_COMPATIBLE[name].key_field):
            return name
    return "rule_based"


def build_llm(settings: Settings) -> LLMClient | None:
    provider = resolve_provider(settings)
    model = settings.llm_model.strip()
    timeout = settings.llm_timeout_seconds
    if provider == "rule_based":
        return None
    if provider == "anthropic":
        key = _secret(settings, "anthropic_api_key")
        if not key:
            raise LLMError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        return AnthropicClient(model or ANTHROPIC_DEFAULT_MODEL, key, timeout)
    if provider == "ollama":
        return OllamaClient(model or OLLAMA_DEFAULT_MODEL, settings.ollama_base_url, timeout)
    spec = OPENAI_COMPATIBLE[provider]
    key = _secret(settings, spec.key_field)
    base_url = {"openai": settings.openai_base_url, "custom": settings.custom_llm_base_url}.get(provider, spec.base_url)
    if provider == "custom":
        if not base_url or not model:
            raise LLMError("LLM_PROVIDER=custom needs CUSTOM_LLM_BASE_URL and LLM_MODEL")
    elif not key:
        raise LLMError(f"LLM_PROVIDER={provider} but {spec.key_field.upper()} is not set")
    return OpenAICompatibleClient(provider, model or spec.default_model, key, base_url, timeout)


def describe(llm: LLMClient | None) -> tuple[str, str]:
    return ("rule_based", "deterministic-v1") if llm is None else (llm.provider, llm.model)


def ping(llm: LLMClient) -> str:
    """Tiny round trip used by `python -m app.llm.check` to verify a key and model work."""

    class Pong(BaseModel):
        ok: bool
        message: str

    out, usage = llm.structured(
        [{"role": "system", "content": "Reply with JSON only."},
         {"role": "user", "content": 'Return {"ok": true, "message": "pong"}. Schema: ' + json.dumps(Pong.model_json_schema())}],
        Pong,
    )
    return f"{llm.provider}/{llm.model}: ok={out.ok} message={out.message!r} tokens={usage.total}"
