"""Reasoning stages. Each has a deterministic implementation (used offline and as a fallback)
and an LLM implementation behind a versioned prompt contract; both produce the same Pydantic schema."""
