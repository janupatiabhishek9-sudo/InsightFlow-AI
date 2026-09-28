"""Input guardrail: prompt-injection detection and explicit trust boundaries.

Heuristic detection is only one layer. The structural defenses matter more: untrusted text
(dataset cells, retrieved documents) is always wrapped and labelled as data, and no text can
grant permissions, because authority lives in deterministic policy code.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "override_instructions": r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|all|system)\b.{0,20}\b(instructions?|rules|prompts?|polic(y|ies))",
        "role_hijack": r"\byou are now\b|\bact as (an? )?(admin|root|system|developer)\b|\bdeveloper mode\b|\bjailbreak\b",
        "prompt_extraction": r"\b(reveal|print|show|repeat)\b.{0,20}\b(system|hidden|developer) (prompt|instructions?)",
        "policy_override": r"\b(override|bypass|disable)\b.{0,20}\b(guardrails?|polic(y|ies)|safety|security|restrictions?)",
        "exfiltration": r"\b(send|upload|post|exfiltrate|transmit|email|forward)\b.{0,40}\b(to|into)\b.{0,20}(https?://|external|server|webhook|\S+@\S+)",
        # No leading \b: names like OPENAI_API_KEY put a word character right before "API".
        "secret_access": r"(api[_ -]?keys?|passwords?|secrets?|credentials?|access[_ ]tokens?)\b|os\.environ|getenv\(|\.env\b|environment variables?",
        "code_execution": r"\bsubprocess\b|os\.system|\brm\s+-rf\b|\b(curl|wget)\s+\S+|__import__|\beval\(|\bexec\(",
    }.items()
}


class GuardVerdict(BaseModel):
    allowed: bool
    flags: list[str] = []
    reason: str = ""


def scan_text(text: str) -> list[str]:
    """Return the names of injection patterns found in `text`."""
    return [name for name, pattern in INJECTION_PATTERNS.items() if pattern.search(text or "")]


def check_user_question(question: str, max_length: int = 1000) -> GuardVerdict:
    q = (question or "").strip()
    if not q:
        return GuardVerdict(allowed=False, reason="empty question")
    if len(q) > max_length:
        return GuardVerdict(allowed=False, reason=f"question longer than {max_length} characters")
    flags = scan_text(q)
    if flags:
        return GuardVerdict(
            allowed=False,
            flags=flags,
            reason="request looks like an attempt to override policy, access secrets or exfiltrate data",
        )
    return GuardVerdict(allowed=True)


def wrap_untrusted(source: str, text: str) -> str:
    """Wrap untrusted content so prompts can clearly separate data from instructions."""
    safe = text.replace("</untrusted_data>", "</untrusted_data_>")
    return f'<untrusted_data source="{source}">\n{safe}\n</untrusted_data>'
