"""Question understanding: deterministic parser + grounding against the dataset.

`understand_rule_based` parses the question directly. `ground` takes any draft (from the parser
or from an LLM) and checks every metric, dimension, filter value and period against the real
data, turning anything unresolvable into an explicit ambiguity instead of a guess.
"""

from __future__ import annotations

import re
from typing import Callable

from app.domain.question import Entity, Intent, ModificationRequest, Period, QuestionUnderstanding
from app.reasoning.catalog import DataCatalog
from app.tools.duckdb_engine import TABLE, quote
from app.tools.periods import latest_complete_quarter, parse_periods, previous_period, same_period_last_year

# (pattern, metric). Order matters: the first metric mentioned wins.
METRIC_PATTERNS = [
    (r"\b(revenue|sales|turnover)\b", "revenue"),
    (r"\b(profits?|margins?|earnings)\b", "profit"),
    (r"\b(costs?|cogs)\b", "cost"),
    (r"\b(orders?|order count)\b", "orders"),
    (r"\b(units|quantity|quantities|volumes?)\b", "quantity"),
    (r"\b(discounts?|discounting)\b", "discount"),
    (r"\b(prices?|pricing)\b", "unit_price"),
]
DRIVER_METRICS = {"orders", "quantity", "discount", "unit_price"}
DIMENSION_PATTERNS = [
    (r"\bproduct categor(y|ies)\b|\bcategor(y|ies)\b", "product_category"),
    (r"\bproducts?\b", "product"),
    (r"\bcountr(y|ies)\b|\bmarkets?\b", "country"),
    (r"\bregions?\b", "region"),
    (r"\b(customer )?segments?\b|\bcustomer types?\b", "customer_segment"),
]
DEMONYMS = {
    "european": "Europe", "german": "Germany", "french": "France", "dutch": "Netherlands",
    "spanish": "Spain", "italian": "Italy", "british": "United Kingdom", "uk": "United Kingdom",
    "american": "United States", "us": "United States", "usa": "United States", "canadian": "Canada",
    "mexican": "Mexico", "japanese": "Japan", "australian": "Australia", "indian": "India",
    "asian": "APAC", "asia": "APAC", "north american": "North America",
}
CHANGE_WORDS = r"\b(why|compar\w*|versus|vs|declin\w*|decreas\w*|drop\w*|fall|fell|down|reduc\w*|increas\w*|grow\w*|growth|rise|rose|chang\w*|contribut\w*|driv\w*|caus\w*|affect\w*|impact\w*|explain)\b"
DECREASE_WORDS = r"\b(declin\w*|decreas\w*|drop\w*|fall|fell|down|reduc\w*|lower|loss|shrink\w*)\b"
INCREASE_WORDS = r"\b(increas\w*|grow\w*|growth|rise|rose|gain\w*|improv\w*)\b"
TREND_WORDS = r"\b(trend\w*|over time|monthly|month by month|evolv\w*|evolution)\b"
RANK_WORDS = r"\b(which|top|largest|highest|lowest|best|worst|most|least|rank\w*)\b"
SUMMARY_WORDS = r"\b(what (was|is|were)|how much|how many|total)\b"
MODIFY_WORDS = r"\b(delete|remove|drop|update|overwrite|truncate|dedupe|deduplicate|clean up|purge)\b"
DATA_OBJECTS = r"\b(rows?|records?|tables?|data|dataset|duplicates?|values?|entries)\b"
YOY_WORDS = r"\b(yoy|year[- ]over[- ]year|last year|previous year|prior year|same (quarter|period|month) last year)\b"
NEXT_WORDS = r"\b(additional|further|next|else)\b.{0,20}\b(analys[ie]s|analy[sz]e|investigat\w*|look at)\b"
STOP_CAPS = {"why", "which", "what", "how", "was", "did", "the", "q1", "q2", "q3", "q4", "h1", "h2", "and",
             "is", "are", "were", "show", "compare", "give", "list", "i", "we", "delete", "remove", "yoy", "vs"}

# (period, comparison) -> metric change; lets the parser find "the decline" when no period is named.
ChangeProbe = Callable[[Period, Period], float | None]


def _match_values(question: str, catalog: DataCatalog) -> tuple[list[Entity], list[str]]:
    q = question.lower()
    entities: list[Entity] = []
    ambiguities: list[str] = []
    for col, values in catalog.values.items():
        exact = {v.strip().lower(): v for v in values if v == v.strip()}
        for key, value in exact.items():
            if len(key) >= 2 and re.search(rf"(?<![\w-]){re.escape(key)}(?![\w-])", q):
                entities.append(Entity(text=key, column=col, value=value))
        for word, target in DEMONYMS.items():
            if target.lower() in exact and re.search(rf"\b{re.escape(word)}\b", q):
                if not any(e.value == exact[target.lower()] for e in entities):
                    entities.append(Entity(text=word, column=col, value=exact[target.lower()]))
    # Longer matches win ("North America" over "America" style overlaps).
    entities.sort(key=lambda e: -len(e.text))
    kept: list[Entity] = []
    for e in entities:
        if not any(e.text in k.text and e.text != k.text for k in kept):
            kept.append(e)
    by_value: dict[str, set[str]] = {}
    for e in kept:
        by_value.setdefault(e.value.lower(), set()).add(e.column)
    for value, cols in by_value.items():
        if len(cols) > 1:
            ambiguities.append(f"'{value}' matches several columns ({', '.join(sorted(cols))}); which one do you mean?")
    return kept, ambiguities


def grounded_filters(question: str, catalog: DataCatalog) -> dict[str, list[str]]:
    """Column -> dataset values mentioned in the question."""
    filters: dict[str, list[str]] = {}
    for e in _match_values(question, catalog)[0]:
        if e.value not in filters.setdefault(e.column, []):
            filters[e.column].append(e.value)
    return filters


def _unknown_names(question: str, entities: list[Entity], catalog: DataCatalog) -> list[str]:
    """Capitalised words that look like entity names but match nothing in the data."""
    known = {e.text.lower() for e in entities} | {w for e in entities for w in e.value.lower().split()}
    known |= {v.lower() for vals in catalog.values.values() for v in vals for v in v.split()}
    months = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
              "october", "november", "december"}
    out = []
    for i, token in enumerate(re.findall(r"\b[A-Z][a-zA-Z]+\b", question)):
        low = token.lower()
        if i == 0 and question.strip().startswith(token):
            continue
        if low in STOP_CAPS or low in known or low in months or low in DEMONYMS or low in {"europe", "enterprise"}:
            continue
        out.append(token)
    return out


def _modification(question: str, catalog: DataCatalog) -> ModificationRequest | None:
    q = question.lower()
    if not (re.search(MODIFY_WORDS, q) and re.search(DATA_OBJECTS, q)):
        return None
    if re.search(r"\bdrop\b.{0,20}\btable\b|\btruncate\b", q):
        return ModificationRequest(operation="delete", description="Drop the dataset table", sql=f"DROP TABLE {TABLE}")
    for metric in catalog.schema_.by_kind("numeric"):
        if re.search(rf"\bnegative {metric.replace('_', ' ')}\b", q):
            return ModificationRequest(operation="delete", description=f"Delete rows with negative {metric}",
                                       sql=f"DELETE FROM {TABLE} WHERE {quote(metric)} < 0")
    for col in catalog.schema_.names:
        if re.search(rf"\b(missing|null|empty|blank) {col.replace('_', ' ')}\b", q):
            return ModificationRequest(operation="delete", description=f"Delete rows with missing {col}",
                                       sql=f"DELETE FROM {TABLE} WHERE {quote(col)} IS NULL")
    if re.search(r"\bduplicat\w*|dedupe\b", q):
        cols = ", ".join(quote(c) for c in catalog.schema_.names)
        return ModificationRequest(operation="delete", description="Delete fully duplicated rows (keep the first copy)",
                                   sql=f"DELETE FROM {TABLE} WHERE rowid NOT IN (SELECT min(rowid) FROM {TABLE} GROUP BY {cols})")
    return ModificationRequest(operation="delete", description="Unrecognised modification", sql="")


def understand_rule_based(question: str, catalog: DataCatalog, probe: ChangeProbe | None = None) -> QuestionUnderstanding:
    q = question.lower()
    entities, ambiguities = _match_values(question, catalog)
    filters = grounded_filters(question, catalog)

    modification = _modification(question, catalog)
    if modification is not None:
        u = QuestionUnderstanding(intent=Intent.DATA_MODIFICATION, entities=entities, filters=filters,
                                  modification=modification, requested_output=["modified_working_copy"])
        if not modification.sql:
            u.ambiguities.append("I could not translate this modification into a precise operation. "
                                 "Please state which rows to change (e.g. 'delete rows with negative revenue').")
        return u

    mentioned = [(m.start(), metric) for pattern, metric in METRIC_PATTERNS for m in re.finditer(pattern, q)
                 if catalog.has_metric(metric)]
    mentioned_metrics = [m for _, m in sorted(mentioned)]
    assumptions: list[str] = []
    requested: list[str] = []
    causal = re.search(r"\b(caused by|driven by|due to|because of|explained by)\b", q)
    if causal and set(mentioned_metrics) & DRIVER_METRICS and "revenue" not in mentioned_metrics:
        metric = "revenue" if catalog.has_metric("revenue") else mentioned_metrics[0]
        requested.append("drivers")
    elif mentioned_metrics:
        metric = mentioned_metrics[0]
    elif catalog.has_metric("revenue"):
        metric = "revenue"
        assumptions.append("No metric named; using revenue, the primary KPI (company_kpis.md).")
    else:
        metric = None
        ambiguities.append("Which metric should be analysed?")

    dimensions: list[str] = []
    scan = q
    for pattern, dim in DIMENSION_PATTERNS:
        if catalog.has_dimension(dim) and re.search(pattern, scan):
            dimensions.append(dim)
            scan = re.sub(pattern, " ", scan)
    if re.search(r"\bby (\w+)", q):
        for col in catalog.schema_.by_kind("categorical"):
            if re.search(rf"\bby {col.replace('_', ' ')}\b", q) and col not in dimensions:
                dimensions.append(col)

    direction = "decrease" if re.search(DECREASE_WORDS, q) else "increase" if re.search(INCREASE_WORDS, q) else "any"
    if re.search(TREND_WORDS, q):
        intent = Intent.TREND
    elif re.search(CHANGE_WORDS, q) or re.search(NEXT_WORDS, q):
        intent = Intent.CHANGE_ANALYSIS
    elif re.search(RANK_WORDS, q) and dimensions:
        intent = Intent.RANKING
    elif re.search(SUMMARY_WORDS, q) or mentioned_metrics:
        intent = Intent.SUMMARY
    else:
        intent = Intent.SUMMARY
        ambiguities.append("I could not tell what analysis you want. Try e.g. 'Why did revenue decline in Q3?'")

    if re.search(r"^\s*(why|how come)\b.{0,15}\b(it|this|that)\b", q) and not mentioned_metrics and not entities:
        ambiguities.append("What does 'it' refer to? Please name the metric and the market or product.")
    unknown = _unknown_names(question, entities, catalog)
    if unknown:
        ambiguities.append(f"{', '.join(unknown)} does not match any value in the dataset. Which value do you mean?")

    periods, period_notes = ([], []) if catalog.date_min is None else parse_periods(question, catalog.date_min, catalog.date_max)
    assumptions += period_notes
    time_period = comparison = None
    comparison_type = "none"
    if len(periods) >= 2:
        time_period, comparison, comparison_type = periods[0], periods[1], "explicit"
    elif len(periods) == 1:
        time_period = periods[0]
        if re.search(YOY_WORDS, q):
            comparison, comparison_type = same_period_last_year(time_period), "same_period_last_year"
        elif intent == Intent.CHANGE_ANALYSIS:
            comparison, comparison_type = previous_period(time_period), "previous_period"
            assumptions.append(f"Compared with the previous period ({comparison.label}), the default in company_kpis.md.")
    elif intent == Intent.CHANGE_ANALYSIS and catalog.date_max is not None:
        time_period, comparison = _default_change_period(catalog, direction, probe)
        comparison_type = "previous_period"
        why = f" with a {direction} in {metric}" if direction != "any" and probe else ""
        assumptions.append(f"No period named; using the most recent complete quarter{why}: "
                           f"{time_period.label} vs {comparison.label}.")
    elif intent in (Intent.SUMMARY, Intent.RANKING):
        assumptions.append("No period named; using all available data.")

    if intent == Intent.CHANGE_ANALYSIS:
        requested.insert(0, "explanation")
        # Geographic drill-down one level below the filter; none when a single country is already selected.
        geo = None if "country" in filters else "country" if "region" in filters else "region"
        if not dimensions:
            dimensions = [d for d in (geo, "product") if d and catalog.has_dimension(d)]
            assumptions.append(f"No breakdown requested; analysing contributions by {' and '.join(dimensions)}.")
        elif re.search(r"\bwhy\b", q) and geo and geo not in dimensions and catalog.has_dimension(geo):
            # "Why" questions need to know *where* the change happened, not only what was asked for.
            dimensions.insert(0, geo)
        if metric == "revenue" and "drivers" not in requested:
            requested.append("drivers")
        if re.search(NEXT_WORDS, q):
            requested.append("next_analyses")
    requested += [f"contributing_{d}" for d in dimensions]
    if intent == Intent.RANKING and re.search(r"\b(lowest|worst|least|smallest|bottom)\b", q):
        requested.append("lowest")

    required_context = [f"definition of {metric}"] if metric else []
    if time_period:
        required_context.append("fiscal calendar")
    if "customer_segment" in dimensions or "customer_segment" in filters:
        required_context.append("customer segment definitions")

    return QuestionUnderstanding(
        intent=intent, direction=direction, metric=metric, dimensions=dimensions, filters=filters,
        entities=entities, time_period=time_period, comparison_period=comparison, comparison_type=comparison_type,
        requested_output=list(dict.fromkeys(requested)), ambiguities=ambiguities, assumptions=assumptions,
        required_context=required_context,
    )


def _default_change_period(catalog: DataCatalog, direction: str, probe: ChangeProbe | None) -> tuple[Period, Period]:
    """Latest complete quarter; if a direction was asked for, the latest quarter moving that way."""
    current = latest_complete_quarter(catalog.date_max)
    if probe is not None and direction in ("decrease", "increase"):
        candidate = current
        for _ in range(8):
            prev = previous_period(candidate)
            if catalog.date_min and prev.start < catalog.date_min:
                break
            change = probe(candidate, prev)
            if change is not None and ((change < 0) == (direction == "decrease")) and change != 0:
                return candidate, prev
            candidate = prev
    return current, previous_period(current)


def ground(draft: QuestionUnderstanding, catalog: DataCatalog) -> QuestionUnderstanding:
    """Validate a draft (typically from an LLM) against the data. Never invents values."""
    u = draft.model_copy(deep=True)
    if u.metric and not catalog.has_metric(u.metric):
        u.ambiguities.append(f"Metric '{u.metric}' does not exist in the dataset (available: {', '.join(catalog.metrics)}).")
        u.metric = None
    bad_dims = [d for d in u.dimensions if not catalog.has_dimension(d)]
    if bad_dims:
        u.assumptions.append(f"Ignored unknown breakdown dimensions: {bad_dims}")
        u.dimensions = [d for d in u.dimensions if d not in bad_dims]
    clean: dict[str, list[str]] = {}
    for col, values in u.filters.items():
        allowed = {v.lower(): v for v in catalog.values.get(col, [])}
        if not allowed:
            u.ambiguities.append(f"Filter column '{col}' is not a categorical column in the dataset.")
            continue
        for v in values:
            if v.lower() in allowed:
                clean.setdefault(col, []).append(allowed[v.lower()])
            else:
                u.ambiguities.append(f"'{v}' is not a value of {col}.")
    u.filters = clean
    if u.intent != Intent.DATA_MODIFICATION and u.metric is None and not any("metric" in a.lower() for a in u.ambiguities):
        u.ambiguities.append("Which metric should be analysed?")
    return u
