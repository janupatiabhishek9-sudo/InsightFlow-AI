"""Output guardrail: verify claims against evidence before anything reaches the report.

- A FACT or INTERPRETATION must cite existing evidence; otherwise it is downgraded to HYPOTHESIS.
- Every number in a FACT or INTERPRETATION must match a value in its cited evidence (within display
  rounding); otherwise it is downgraded, because the number was not produced by a deterministic tool.
- Business-context statements must cite retrieved documents; otherwise they are removed
  (a hallucinated definition is worse than none).
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.domain.evidence import Claim, Evidence

# Optional K/M/B/% suffix; a number otherwise glued to letters ("27in") is part of a name, not a measurement.
_NUM = re.compile(r"(?<![\w.])(-?\$?-?\d[\d,]*(?:\.\d+)?)(?:\s?([KkMmBb](?![a-zA-Z])|%))?(?![\w])")
_PERIOD_TOKENS = re.compile(
    r"\b(Q[1-4]|H[12])\s+\d{4}\b|\b(19|20)\d{2}-\d{2}(-\d{2})?\b|\b(January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{4}\b|\b(19|20)\d{2}\b(?!\s*%)|\bE\d+\b|\bstep \d+\b|\btop \d+\b"
)
SCALES = {"k": 1e3, "m": 1e6, "b": 1e9}


class GuardedClaims(BaseModel):
    claims: list[Claim]
    downgraded: int = 0
    removed: int = 0
    issues: list[str] = []


def extract_numbers(text: str) -> list[tuple[float, float, bool]]:
    """Return (value, tolerance, is_percent) for each number that should be verifiable."""
    cleaned = _PERIOD_TOKENS.sub(" ", text)
    out = []
    for m in _NUM.finditer(cleaned):
        raw, suffix = m.group(1), (m.group(2) or "").lower()
        is_money = "$" in raw
        digits = raw.replace("$", "").replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        scale = SCALES.get(suffix, 1.0)
        if not is_money and not suffix and abs(value) < 10 and decimals == 0:
            continue  # small counts like "3 products" are structural, not measurements
        tol = 0.5 * 10 ** (-decimals) * scale + 1e-9
        out.append((value * scale, tol, suffix == "%"))
    return out


def _supported(number: tuple[float, float, bool], values: list[float]) -> bool:
    x, tol, _ = number
    return any(abs(abs(v) - abs(x)) <= tol for v in values)


def guard_claims(claims: list[Claim], evidence: list[Evidence]) -> GuardedClaims:
    by_id = {e.id: e for e in evidence}
    out: list[Claim] = []
    downgraded = removed = 0
    issues: list[str] = []
    for claim in claims:
        c = claim.model_copy()
        cited = [by_id[i] for i in c.evidence_ids if i in by_id]
        if c.section == "context":
            if not any(e.evidence_type == "rag" for e in cited):
                removed += 1
                issues.append(f"removed uncited business-context statement: {c.text[:80]}")
                continue
            out.append(c)
            continue
        if c.kind == "hypothesis":  # already labelled as unconfirmed
            out.append(c)
            continue
        # Facts and interpretations must both rest on evidence and use only real numbers.
        if not cited:
            c.kind, c.note = "hypothesis", "Downgraded: no supporting evidence was cited."
            downgraded += 1
            issues.append(f"{claim.kind} without evidence downgraded: {c.text[:80]}")
            out.append(c)
            continue
        # A number is supported only if it was computed, or appears literally in a cited document.
        # (Citing a document must not launder a made-up figure.)
        support = [v for e in cited if e.evidence_type in ("computed", "derived") for v in e.values.values()]
        support += [n[0] for e in cited if e.evidence_type == "rag" for n in extract_numbers(str(e.result))]
        unsupported = [n for n in extract_numbers(c.text) if not _supported(n, support)]
        if unsupported:
            c.kind = "hypothesis"
            c.note = f"Downgraded: {len(unsupported)} number(s) not found in the cited evidence."
            downgraded += 1
            issues.append(f"unsupported number(s) {[round(n[0], 4) for n in unsupported]} in: {c.text[:80]}")
        out.append(c)
    return GuardedClaims(claims=out, downgraded=downgraded, removed=removed, issues=issues)
