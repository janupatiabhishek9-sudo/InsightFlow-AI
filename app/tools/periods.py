"""Deterministic period parsing and arithmetic (quarters, months, halves, years)."""

from __future__ import annotations

import re
from datetime import date

from app.domain.question import Period

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
ORDINAL_Q = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, 1)


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def label_for(start: date, end: date) -> str:
    n = _months_between(start, end)
    if n == 3 and start.month in (1, 4, 7, 10):
        return f"Q{(start.month - 1) // 3 + 1} {start.year}"
    if n == 1:
        return start.strftime("%B %Y")
    if n == 6 and start.month in (1, 7):
        return f"H{1 if start.month == 1 else 2} {start.year}"
    if n == 12 and start.month == 1:
        return str(start.year)
    return f"{start.isoformat()} to {end.isoformat()}"


def make_period(start: date, end: date) -> Period:
    return Period(label=label_for(start, end), start=start, end=end)


def quarter(year: int, q: int) -> Period:
    start = date(year, 3 * (q - 1) + 1, 1)
    return make_period(start, _add_months(start, 3))


def shift(period: Period, months: int) -> Period:
    return make_period(_add_months(period.start, months), _add_months(period.end, months))


def previous_period(period: Period) -> Period:
    return shift(period, -_months_between(period.start, period.end))


def same_period_last_year(period: Period) -> Period:
    return shift(period, -12)


def latest_complete_quarter(data_max: date) -> Period:
    q_start = date(data_max.year, 3 * ((data_max.month - 1) // 3) + 1, 1)
    candidate = make_period(q_start, _add_months(q_start, 3))
    # Only use it if the data covers the quarter's last day.
    return candidate if data_max >= date.fromordinal(candidate.end.toordinal() - 1) else previous_period(candidate)


def _year(token: str | None) -> int | None:
    if not token:
        return None
    token = token.strip("'")
    return int(token) if len(token) == 4 else 2000 + int(token)


def parse_periods(text: str, data_min: date, data_max: date) -> tuple[list[Period], list[str]]:
    """Find period mentions in order of appearance. Returns (periods, assumptions)."""
    found: list[tuple[int, str, int, int | None]] = []  # (position, unit, number, year)
    t = text.lower()
    for m in re.finditer(r"\b(\d{4})\s*[-/ ]?\s*q([1-4])\b", t):
        found.append((m.start(), "q", int(m.group(2)), int(m.group(1))))
    for m in re.finditer(r"(?<![\d-])\bq([1-4])\b(?:\s*(?:of\s*)?[-' ]?\s*(\d{4}|'\d{2}))?", t):
        if not any(abs(m.start() - f[0]) < 8 for f in found):
            found.append((m.start(), "q", int(m.group(1)), _year(m.group(2))))
    for m in re.finditer(r"\b(first|1st|second|2nd|third|3rd|fourth|4th) quarter(?:\s+(?:of\s+)?(\d{4}))?", t):
        found.append((m.start(), "q", ORDINAL_Q[m.group(1)], _year(m.group(2))))
    for m in re.finditer(r"\bh([12])\b(?:\s*(\d{4}))?", t):
        found.append((m.start(), "h", int(m.group(1)), _year(m.group(2))))
    month_names = "|".join(sorted(MONTHS, key=len, reverse=True))
    for m in re.finditer(rf"\b({month_names})\b\.?(?:\s+(\d{{4}}))?", t):
        if m.group(1) == "may" and not m.group(2):
            continue  # "may" is usually the verb
        found.append((m.start(), "m", MONTHS[m.group(1)], _year(m.group(2))))
    covered = [f[0] for f in found]
    for m in re.finditer(r"\b(?:in|for|during|fy)\s*(\d{4})\b", t):
        if not any(abs(m.start() - c) < 20 for c in covered):
            found.append((m.start(), "y", 0, int(m.group(1))))

    periods: list[Period] = []
    assumptions: list[str] = []
    for _, unit, num, year in sorted(found):
        explicit = year is not None
        if year is None:
            year = _infer_year(unit, num, data_min, data_max)
        if unit == "q":
            p = quarter(year, num)
        elif unit == "h":
            start = date(year, 1 if num == 1 else 7, 1)
            p = make_period(start, _add_months(start, 6))
        elif unit == "m":
            start = date(year, num, 1)
            p = make_period(start, _add_months(start, 1))
        else:
            p = make_period(date(year, 1, 1), date(year + 1, 1, 1))
        if not explicit:
            assumptions.append(f"No year given; interpreted as {p.label} (latest occurrence in the data).")
        if p not in periods:
            periods.append(p)
    return periods, assumptions


def _infer_year(unit: str, num: int, data_min: date, data_max: date) -> int:
    """Latest year in which the period is fully covered by the data."""
    for year in range(data_max.year, data_min.year - 1, -1):
        if unit == "q":
            p = quarter(year, num)
        elif unit == "h":
            start = date(year, 1 if num == 1 else 7, 1)
            p = make_period(start, _add_months(start, 6))
        else:
            start = date(year, num, 1)
            p = make_period(start, _add_months(start, 1))
        if p.start >= data_min and date.fromordinal(p.end.toordinal() - 1) <= data_max:
            return year
    return data_max.year
