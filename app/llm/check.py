"""Check which LLM provider .env selects and make one tiny test call.

Usage:  python -m app.llm.check
"""

from __future__ import annotations

from app.config import get_settings
from app.llm.client import LLMError, build_llm, ping, resolve_provider


def main() -> None:
    settings = get_settings()
    provider = resolve_provider(settings)
    print(f"LLM_PROVIDER={settings.llm_provider} -> using: {provider}")
    try:
        llm = build_llm(settings)
    except LLMError as e:
        print(f"Configuration problem: {e}")
        raise SystemExit(1) from e
    if llm is None:
        print("No API key found; running offline with the deterministic rule-based reasoner.")
        return
    try:
        print(ping(llm))
    except LLMError as e:
        print(f"Call failed: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
