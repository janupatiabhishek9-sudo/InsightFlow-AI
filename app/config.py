"""Application configuration loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AccessLevel = Literal["public", "internal", "restricted"]
LLMProvider = Literal["auto", "rule_based", "anthropic", "openai", "gemini", "ollama", "groq", "mistral",
                      "deepseek", "openrouter", "together", "xai", "custom"]


class Settings(BaseSettings):
    """All tunables live here so nothing is hard-coded in the modules."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # LLM. "auto" picks the first provider whose key is set (see app/llm/client.py); no key -> rule_based.
    llm_provider: LLMProvider = "auto"
    llm_model: str = Field("", description="Empty = the provider's default model")
    llm_timeout_seconds: float = 60
    token_budget: int = 60_000
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = Field(None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    groq_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    together_api_key: SecretStr | None = None
    xai_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    # Any other OpenAI-compatible server (vLLM, LM Studio, LiteLLM proxy, Azure OpenAI v1, ...)
    custom_llm_base_url: str = ""
    custom_llm_api_key: SecretStr | None = None

    # RAG
    embedding_model: str = "hashing"
    vector_db_path: Path = Path("data/processed/vector_store")
    knowledge_dir: Path = Path("knowledge")
    default_user_clearance: AccessLevel = "internal"

    # Data
    data_dir: Path = Path("data")
    # SQLite file for workflow checkpoints (paused approvals survive restarts); "memory" = not persisted
    checkpoint_db: str = "data/processed/checkpoints.sqlite"
    max_upload_mb: int = 100

    # Agent loop control
    max_tool_calls: int = Field(12, ge=1)
    max_retries: int = Field(2, ge=0)
    max_plan_steps: int = Field(8, ge=1)
    max_execution_time: float = Field(120, gt=0)
    query_timeout: float = Field(15, gt=0)
    max_result_rows: int = Field(500, ge=1)
    sandbox_timeout: float = Field(20, gt=0)
    sandbox_memory_mb: int = Field(1024, ge=128)

    # UI / API
    ui_backend: Literal["local", "http"] = "local"
    api_url: str = "http://localhost:8000"
    api_key: SecretStr | None = Field(None, description="Key the Streamlit UI sends when UI_BACKEND=http")
    api_keys: SecretStr | None = Field(None, description="name:key:role:clearance,... - empty disables API auth")

    # Observability
    log_level: str = "INFO"
    enable_langsmith: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "insightflow-ai"

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.data_dir) / "raw"

    @property
    def trace_dir(self) -> Path:
        return self.resolve(self.data_dir) / "traces"

    @property
    def sandbox_dir(self) -> Path:
        return self.resolve(self.data_dir) / "sandbox"


@lru_cache
def get_settings() -> Settings:
    return Settings()
