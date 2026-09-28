"""Evidence-first report generation.

`draft_claims_rule_based` turns validated results into labelled claims (FACT / INTERPRETATION /
HYPOTHESIS), each citing evidence ids. An LLM draft uses the same Claim schema. Whatever produced
the draft, `render_markdown` only renders claims that passed the output guard.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.data import DataQualityReport
from app.domain.evidence import Claim, Evidence
from app.domain.governance import ResultValidation, StepResult
from app.domain.plan import InvestigationPlan
from app.domain.question import Intent, QuestionUnderstanding
from app.domain.report import ContextCitation, Report

MONEY_METRICS = {"revenue", "profit", "cost", "unit_price"}
SECTION_TITLES = [
    ("executive", "Executive Finding"),
    ("evidence", "Evidence"),
    ("contributors", "Main Contributors"),
    ("detail", "Detailed Analysis"),
    ("context", "Business Context"),
    ("interpretation", "Interpretation"),
    ("hypothesis", "Hypotheses"),
]


class ReportDraft(BaseModel):
    title: str
    executive_finding: str
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_analyses: list[str] = Field(default_factory=list)


def fmt(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    if metric in MONEY_METRICS:
        return f"{'-' if value < 0 else ''}${abs(value):,.0f}"
    if metric == "discount":
        return f"{value * 100:.1f}%"
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{abs(value):.1f}%"


def _label(dim: str) -> str:
    return dim.replace("_", " ")


def _scope(u: QuestionUnderstanding) -> str:
    vals = [v for vs in u.filters.values() for v in vs]
    return " / ".join(vals) if vals else "Total"


def draft_claims_rule_based(
    u: QuestionUnderstanding,
    plan: InvestigationPlan,
    results: dict[int, StepResult],
    evidence: list[Evidence],
    quality: DataQualityReport,
    validation: ResultValidation,
    extra_limitations: list[str],
) -> ReportDraft:
    claims: list[Claim] = []
    by_step = {s.step: s for s in plan.steps}
    metric = u.metric or "revenue"
    ok = {n: r for n, r in results.items() if r.status == "ok"}

    def add(text: str, kind: str, section: str, ev: list[str | None], conf: float = 1.0) -> None:
        claims.append(Claim(text=text, kind=kind, section=section, evidence_ids=[e for e in ev if e], confidence=conf))

    # ---- business context from retrieved definitions ----
    for r in ok.values():
        for eid in r.summary.get("evidence_ids", [])[:2]:
            e = next(x for x in evidence if x.id == eid)
            first = str(e.result).strip().splitlines()[0]
            add(f"{e.description}: {first}", "fact", "context", [eid])

    executive = "No conclusive finding could be computed."
    title = f"Investigation: {plan.objective}"

    if u.intent == Intent.DATA_MODIFICATION:
        r = next((r for r in ok.values() if r.tool == "execute_write_sql"), None)
        if r:
            s = r.summary
            executive = (f"Applied '{u.modification.description}' to this investigation's working copy: "
                         f"{s['affected_rows']:,.0f} rows affected ({s['rows_before']:,.0f} -> {s['rows_after']:,.0f} rows). "
                         f"The source file was not changed.")
            add(executive, "fact", "executive", [r.evidence_id])
        return ReportDraft(title=title, executive_finding=executive, claims=claims,
                           limitations=["Changes apply only to this investigation's in-memory working copy."])

    comparison = next((r for r in ok.values() if r.kind == "period_comparison"), None)
    if comparison and comparison.rows and comparison.rows[0]["current_value"] is not None:
        c = comparison.rows[0]
        change = c["abs_change"] or 0
        verb = "decreased" if change < 0 else "increased" if change > 0 else "did not change"
        executive = (f"{_scope(u)} {metric} {verb} by {pct(c['pct_change'])} from {fmt(metric, c['comparison_value'])} "
                     f"in {c['comparison_period']} to {fmt(metric, c['current_value'])} in {c['period']} "
                     f"(change: {fmt(metric, change)}).")
        add(executive, "fact", "executive", [comparison.evidence_id])
        add(f"{metric.capitalize()} in {c['comparison_period']}: {fmt(metric, c['comparison_value'])}; in {c['period']}: "
            f"{fmt(metric, c['current_value'])}; change: {fmt(metric, change)} ({'-' if change < 0 else '+'}{pct(c['pct_change'])}). "
            f"Source: {next((e.source for e in evidence if e.id == comparison.evidence_id), 'n/a')}.",
            "fact", "evidence", [comparison.evidence_id])
        add(f"Rows analysed: {c['current_rows']:,} in {c['period']} and {c['comparison_rows']:,} in {c['comparison_period']}.",
            "fact", "evidence", [comparison.evidence_id])
        if u.direction == "decrease" and change > 0 or u.direction == "increase" and change < 0:
            add(f"The question assumes a {'decline' if u.direction == 'decrease' else 'rise'}, but the data shows {metric} {verb}.",
                "fact", "executive", [comparison.evidence_id])

        # ---- contributors per breakdown ----
        for n, r in sorted(ok.items()):
            if r.kind != "dimension_breakdown" or by_step[n].analysis.comparison_period is None:
                continue
            dim = by_step[n].analysis.dimension
            same_sign = [x for x in r.rows if (x["abs_change"] or 0) * change > 0] or r.rows
            top = sorted(same_sign, key=lambda x: -abs(x["abs_change"] or 0))[:3]
            for i, x in enumerate(top, 1):
                share = f", {x['share_of_total_change']:.1f}% of the total change" if x.get("share_of_total_change") is not None else ""
                add(f"{_label(dim).capitalize()} #{i}: {x['dim_value']} changed by {fmt(metric, x['abs_change'])} "
                    f"({pct(x['pct_change'])} vs {c['comparison_period']}){share}.", "fact", "contributors", [r.evidence_id])
            lead = top[0] if top else None
            if lead and lead.get("share_of_total_change") is not None:
                s = lead["share_of_total_change"]
                shape = "concentrated" if s >= 40 else "moderately concentrated" if s >= 25 else "broad-based"
                add(f"By {_label(dim)}, the change is {shape}: the largest contributor, {lead['dim_value']}, accounts for "
                    f"{s:.1f}% of it.", "interpretation", "interpretation", [r.evidence_id], 0.8)
            if dim == "customer_segment" and lead and lead["dim_value"] == "Enterprise" and change < 0:
                add("Reduced Enterprise purchasing (for example budget freezes or delayed procurement cycles) may explain "
                    "part of the decline. This needs customer-level or CRM data to confirm.", "hypothesis", "hypothesis",
                    [r.evidence_id], 0.4)

        # ---- driver bridge ----
        drv = next((r for r in ok.values() if r.kind == "driver_decomposition"), None)
        if drv:
            s = drv.summary
            add(f"Orders went from {s['orders_comparison']:,.0f} to {s['orders_current']:,.0f}, an effect of "
                f"{fmt('revenue', s['orders_effect'])}.", "fact", "detail", [drv.evidence_id])
            add(f"Units per order went from {s['units_per_order_comparison']:.2f} to {s['units_per_order_current']:.2f}, "
                f"an effect of {fmt('revenue', s['basket_size_effect'])}.", "fact", "detail", [drv.evidence_id])
            add(f"Average list price per unit went from {fmt('revenue', s['avg_list_price_comparison'])} to "
                f"{fmt('revenue', s['avg_list_price_current'])}, an effect of {fmt('revenue', s['list_price_effect'])}.",
                "fact", "detail", [drv.evidence_id])
            add(f"Effective discount went from {s['effective_discount_pct_comparison']:.1f}% to "
                f"{s['effective_discount_pct_current']:.1f}%, an effect of {fmt('revenue', s['discount_effect'])}.",
                "fact", "detail", [drv.evidence_id])
            effects = {"fewer or more orders": s["orders_effect"], "basket size": s["basket_size_effect"],
                       "list prices": s["list_price_effect"], "discounting": s["discount_effect"]}
            main = max(effects, key=lambda k: abs(effects[k]))
            add(f"The largest driver of the change is {main}.", "interpretation", "interpretation", [drv.evidence_id], 0.85)
            if s["discount_effect"] < 0 and s["effective_discount_pct_current"] > s["effective_discount_pct_comparison"] + 1:
                add("Heavier discounting may reflect a promotion or competitive price pressure. Check promotion and pricing "
                    "records for the period.", "hypothesis", "hypothesis", [drv.evidence_id], 0.4)
            if s["orders_effect"] < 0 and abs(s["orders_effect"]) >= abs(s["total_change"]) * 0.3:
                add("Lower order volume may reflect weaker demand or lost accounts in the main contributing markets. "
                    "This dataset has no customer identifier, so it cannot be verified here.", "hypothesis", "hypothesis",
                    [drv.evidence_id], 0.35)

        stat = next((r for r in ok.values() if r.tool == "run_statistical_test"), None)
        if stat:
            s = stat.summary
            verdict = "statistically significant" if s["significant_at_5pct"] else "not statistically significant"
            sig = f"the difference is {verdict} (Mann-Whitney test" + (
                f", p = {s['p_value']:.4f})" if s["p_value"] >= 0.001 else ", p below one in a thousand)")
            add(f"Mean {metric} per order line went from {fmt(metric, s['mean_a'])} ({s['period_a']}) to "
                f"{fmt(metric, s['mean_b'])} ({s['period_b']}); {sig}.", "fact", "detail", [stat.evidence_id])

    # ---- other analysis kinds ----
    summary = next((r for r in ok.values() if r.kind == "metric_summary"), None)
    if summary and summary.rows and not comparison:
        v = summary.rows[0]
        when = f" in {u.time_period.label}" if u.time_period else " across all available data"
        executive = f"{_scope(u)} {metric}{when} was {fmt(metric, v['value'])} ({v['row_count']:,} rows)."
        add(executive, "fact", "executive", [summary.evidence_id])
    trend = next((r for r in ok.values() if r.kind == "time_trend"), None)
    if trend and trend.rows:
        first, last = trend.rows[0], trend.rows[-1]
        peak = max(trend.rows, key=lambda x: x["value"] or 0)
        executive = (f"{_scope(u)} monthly {metric} went from {fmt(metric, first['value'])} in {first['month'][:7]} to "
                     f"{fmt(metric, last['value'])} in {last['month'][:7]}; the peak was {fmt(metric, peak['value'])} in {peak['month'][:7]}.")
        add(executive, "fact", "executive", [trend.evidence_id])
    if u.intent == Intent.RANKING:
        for n, r in sorted(ok.items()):
            if r.kind == "dimension_breakdown" and r.rows:
                dim = by_step[n].analysis.dimension
                for i, x in enumerate(r.rows[:3], 1):
                    share = f" ({x['share_pct']:.1f}% of total)" if x.get("share_pct") is not None else ""
                    add(f"{_label(dim).capitalize()} #{i}: {x['dim_value']} with {fmt(metric, x['value'])}{share}.",
                        "fact", "contributors", [r.evidence_id])
                top = r.rows[0]
                if executive.startswith("No conclusive"):
                    executive = f"{top['dim_value']} ranks first by {metric} with {fmt(metric, top['value'])}."
                    add(executive, "fact", "executive", [r.evidence_id])

    limitations = _limitations(quality, validation, extra_limitations, results)
    return ReportDraft(title=title, executive_finding=executive, claims=claims, limitations=limitations,
                       next_analyses=_next_analyses(u, plan))


def _limitations(quality: DataQualityReport, validation: ResultValidation, extra: list[str],
                 results: dict[int, StepResult]) -> list[str]:
    out = [f"Data quality: {w}." for w in quality.warnings[:6]]
    out += [f"Validation: {c.name} - {c.detail}" for c in validation.checks if not c.passed and c.name != "llm_review"]
    out += [f"AI reviewer note (not verified): {c.detail}" for c in validation.checks if c.name == "llm_review"]
    out += [f"Step {r.step} ({r.tool}) did not complete: {r.error}" for r in results.values() if r.status != "ok"]
    out += extra
    out.append("Descriptive data shows where a change happened, not why; causal statements are labelled as hypotheses.")
    return list(dict.fromkeys(out))


def _next_analyses(u: QuestionUnderstanding, plan: InvestigationPlan) -> list[str]:
    done = {s.analysis.dimension for s in plan.steps if s.analysis and s.analysis.dimension}
    ideas = []
    for dim, idea in (("customer_segment", "Break the change down by customer segment."),
                      ("product", "Break the change down by product."),
                      ("country", "Break the change down by country.")):
        if dim not in done:
            ideas.append(idea)
    ideas += ["Drill into products within the largest contributing market.",
              "Compare with the same period last year to separate seasonality from a real decline.",
              "Analyse discount levels by product to find where discounting increased.",
              "Add customer-level data (new versus returning customers, churn) to test the demand hypotheses."]
    return ideas[:5]


def render_markdown(report: Report, evidence: list[Evidence]) -> str:
    lines = [f"# {report.title}", ""]
    labels = {"fact": "FACT", "interpretation": "INTERPRETATION", "hypothesis": "HYPOTHESIS"}
    for key, heading in SECTION_TITLES:
        section = [c for c in report.claims if c.section == key]
        if key == "executive":
            lines += [f"## {heading}", "", report.executive_finding, ""]
            section = [c for c in section if c.text != report.executive_finding]
        elif not section:
            continue
        else:
            lines += [f"## {heading}", ""]
        for c in section:
            refs = f" [{', '.join(c.evidence_ids)}]" if c.evidence_ids else ""
            note = f" _({c.note})_" if c.note else ""
            lines.append(f"- **{labels[c.kind]}:** {c.text}{refs}{note}")
        lines.append("")
    for heading, items in (("Assumptions", report.assumptions), ("Limitations", report.limitations),
                           ("Recommended Next Analyses", report.next_analyses)):
        if items:
            lines += [f"## {heading}", ""] + [f"- {i}" for i in items] + [""]
    if evidence:
        lines += ["## Evidence Appendix", ""]
        for e in evidence:
            lines.append(f"- **{e.id}** ({e.evidence_type}) {e.description} - source: `{e.source}`")
    return "\n".join(lines).strip() + "\n"


def citations(evidence: list[Evidence]) -> list[ContextCitation]:
    return [ContextCitation(source=e.source, section=e.description, excerpt=str(e.result)[:300])
            for e in evidence if e.evidence_type == "rag"]
