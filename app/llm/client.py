"""LLM clients behind one interface. Swap providers through configuration (LLM_PROVIDER, LLM_MODEL).

`rule_based` means no model: every reasoning stage uses its deterministic implementation.
Plain httpx is used for both providers to avoid heavy SDK dependencies.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


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


class _BaseHTTPClient:
    provider = "base"

    def __init__(self, model: str, timeout: float, max_attempts: int = 2):
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts

    def _call(self, messages: list[dict[str, str]], schema: dict) -> tuple[str, Usage]:
        raise NotImplementedError

    def structured(self, messages: list[dict[str, str]], schema: type[T]) -> tuple[T, Usage]:
        usage = Usage()
        msgs = list(messages)
        last_error = ""
        for _ in range(self.max_attempts):
            try:
                text, u = self._call(msgs, schema.model_json_schema())
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


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class OpenAICompatibleClient(_BaseHTTPClient):
    provider = "openai"

    def __init__(self, model: str, api_key: str, base_url: str, timeout: float):
        super().__init__(model, timeout)
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._url = base_url.rstrip("/") + "/chat/completions"

    def _call(self, messages, schema):
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "response_format": {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}}
        r = httpx.post(self._url, json=body, headers=self._headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        u = data.get("usage") or {}
        return data["choices"][0]["message"]["content"], Usage(prompt_tokens=u.get("prompt_tokens", 0),
                                                              completion_tokens=u.get("completion_tokens", 0))


class OllamaClient(_BaseHTTPClient):
    provider = "ollama"

    def __init__(self, model: str, base_url: str, timeout: float):
        super().__init__(model, timeout)
        self._url = base_url.rstrip("/") + "/api/chat"

    def _call(self, messages, schema):
        body = {"model": self.model, "messages": messages, "stream": False, "format": schema,
                "options": {"temperature": 0}}
        r = httpx.post(self._url, json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"], Usage(prompt_tokens=data.get("prompt_eval_count", 0),
                                                 completion_tokens=data.get("eval_count", 0))


def build_llm(settings: Settings) -> LLMClient | None:
    if settings.llm_provider == "rule_based":
        return None
    if settings.llm_provider == "openai":
        if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
            raise LLMError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAICompatibleClient(settings.llm_model, settings.openai_api_key.get_secret_value(),
                                      settings.openai_base_url, settings.llm_timeout_seconds)
    if settings.llm_provider == "ollama":
        return OllamaClient(settings.llm_model, settings.ollama_base_url, settings.llm_timeout_seconds)
    raise LLMError(f"unknown LLM_PROVIDER {settings.llm_provider}")


def describe(llm: LLMClient | None) -> tuple[str, str]:
    return ("rule_based", "deterministic-v1") if llm is None else (llm.provider, llm.model)

