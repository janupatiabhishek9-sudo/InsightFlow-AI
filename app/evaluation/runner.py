"""Run the golden set through the full system and score every trajectory.

Usage:
  python -m app.evaluation.runner                         # run and save results
  python -m app.evaluation.runner --save-baseline         # also store as evaluations/baseline.json
  python -m app.evaluation.runner --baseline evaluations/baseline.json   # exit 1 on regression
  python -m app.evaluation.runner --only chg-01 sec-01    # subset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from app.config import PROJECT_ROOT
from app.evaluation import evaluators as ev
from app.evaluation.datasets import DEFAULT_GOLDEN, GoldenCase, load_cases
from app.evaluation.oracle import Oracle
from app.service import InsightFlowService

LOWER_IS_BETTER = {"unsupported_claim_rate"}
REGRESSION_TOLERANCE = 0.02


def evaluate_case(svc: InsightFlowService, case: GoldenCase, dataset_id: str, oracle: Oracle) -> dict:
    started = time.perf_counter()
    view = svc.start_investigation(dataset_id, case.question)
    first_status = view.status
    if first_status == "awaiting_approval" and case.hitl_decision:
        view = svc.decide(view.investigation_id, case.hitl_decision == "approve", "evaluator", "golden-set run")
    scores = {
        "status_match": ev.status_match(case, first_status),
        "question_understanding": ev.question_understanding(case, view),
        "planning": ev.planning(case, view),
        "tool_selection": ev.tool_selection(case, view),
        "sql_accuracy": ev.sql_accuracy(case, view),
        "numerical_accuracy": ev.numerical_accuracy(case, view, oracle),
        "answer_accuracy": ev.answer_accuracy(case, view, oracle),
        "rag_retrieval": ev.rag_retrieval(case, view),
        "evidence_accuracy": ev.evidence_accuracy(case, view),
        "safety": ev.safety(case, view, first_status),
        "hitl_behavior": ev.hitl_behavior(case, view, first_status),
        "report_quality": ev.report_quality(case, view),
        "unsupported_claim_rate": ev.unsupported_claim_rate(view),
        "efficiency": ev.efficiency(view, svc.settings.max_tool_calls),
    }
    passed = all(v == 1.0 for k, v in scores.items() if v is not None and k not in LOWER_IS_BETTER) \
        and not (scores["unsupported_claim_rate"] or 0) > 0
    return {"id": case.id, "category": case.category, "question": case.question, "status": view.status,
            "first_status": first_status, "investigation_id": view.investigation_id, "passed": passed,
            "scores": scores, "latency_s": round(time.perf_counter() - started, 3)}


def aggregate(results: list[dict]) -> dict[str, float]:
    keys = sorted({k for r in results for k in r["scores"]})
    summary = {}
    for k in keys:
        vals = [r["scores"][k] for r in results if r["scores"][k] is not None]
        if vals:
            summary[k] = round(sum(vals) / len(vals), 4)
    summary["case_pass_rate"] = round(sum(r["passed"] for r in results) / len(results), 4) if results else 0.0
    summary["mean_latency_s"] = round(sum(r["latency_s"] for r in results) / len(results), 3) if results else 0.0
    return summary


def compare(summary: dict, baseline: dict) -> list[str]:
    regressions = []
    for k, base in baseline.items():
        if k == "mean_latency_s" or k not in summary:
            continue
        cur = summary[k]
        worse = cur - base > REGRESSION_TOLERANCE if k in LOWER_IS_BETTER else base - cur > REGRESSION_TOLERANCE
        if worse:
            regressions.append(f"{k}: {base} -> {cur}")
    return regressions


def run(cases: list[GoldenCase], svc: InsightFlowService | None = None) -> tuple[list[dict], dict]:
    svc = svc or InsightFlowService()
    dataset = svc.example_dataset()
    oracle = Oracle(Path(dataset.path))
    results = [evaluate_case(svc, c, dataset.dataset_id, oracle) for c in cases]
    return results, aggregate(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="InsightFlow golden-set evaluation")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--baseline", type=Path, help="fail (exit 1) if any metric regresses versus this file")
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.golden)
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    svc = InsightFlowService()
    results, summary = run(cases, svc)

    out_dir = PROJECT_ROOT / "evaluations" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {"summary": summary, "model": svc.settings.llm_provider, "results": results}
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for r in results:
        failed = [k for k, v in r["scores"].items() if v is not None and ((v > 0) if k in LOWER_IS_BETTER else v < 1)]
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:<8} {r['status']:<20} {', '.join(failed)}")
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k:<24} {v}")
    print(f"\nResults written to {out_dir / f'eval-{stamp}.json'}")

    if args.save_baseline:
        (PROJECT_ROOT / "evaluations" / "baseline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Baseline saved to evaluations/baseline.json")
    if args.baseline:
        regressions = compare(summary, json.loads(args.baseline.read_text(encoding="utf-8")))
        if regressions:
            print("\nREGRESSIONS:\n  " + "\n  ".join(regressions))
            sys.exit(1)
        print("\nNo regressions versus baseline.")


if __name__ == "__main__":
    main()
