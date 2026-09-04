"""
Phase 10: Golden End-to-End Test Runner & Report Generator CLI.

Executes all queries from the golden manifest, applies semantic validators,
displays a structured summary table, and exports reports/golden_suite_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_backend_dir = Path(__file__).resolve().parent.parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.api.routes_query import process_query
from app.evaluation.paths import resolve_repo_path
from app.golden.manifest import GoldenQuery, load_golden_manifest
from app.golden.validators import GoldenValidationResult, validate_golden_result
from app.schemas.query import QueryRequest


def run_golden_suite(
    queries: Optional[List[GoldenQuery]] = None,
    save_report: bool = True,
) -> Tuple[List[GoldenValidationResult], Dict[str, Any]]:
    """
    Executes all golden queries end-to-end through process_query
    and collects structured validation results.
    """
    if queries is None:
        queries = load_golden_manifest()

    results: List[GoldenValidationResult] = []
    total_start = time.time()

    for idx, gq in enumerate(queries, 1):
        print(f"[{idx}/{len(queries)}] Executing {gq.id}: '{gq.name}'...")
        req = QueryRequest(query=gq.query)

        try:
            t0 = time.time()
            res = process_query(req)
            elapsed = round(time.time() - t0, 2)
            val_res = validate_golden_result(res, gq)
            status_str = "PASS" if val_res.passed else "FAIL"
            print(f"  -> {status_str} ({elapsed}s)")
            if not val_res.passed:
                for err in val_res.errors:
                    print(f"     [ERROR] {err}")
        except Exception as exc:
            val_res = GoldenValidationResult(
                query_id=gq.id,
                passed=False,
                errors=[f"Pipeline execution crashed: {exc}"],
                intent=gq.expected_intent,
                temporal_mode=gq.expected_temporal_mode,
            )
            print(f"  -> FAIL (CRASH): {exc}")

        results.append(val_res)

    total_time = round(time.time() - total_start, 2)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    summary_data = {
        "title": "SATQUERY AI: PHASE 10 GOLDEN SUITE REPORT",
        "total_queries": len(results),
        "passed_queries": passed_count,
        "failed_queries": failed_count,
        "pass_rate": round(passed_count / len(results) * 100.0, 1) if results else 0.0,
        "runtime_seconds": total_time,
        "results": [
            {
                "query_id": r.query_id,
                "passed": r.passed,
                "intent": r.intent,
                "temporal_mode": r.temporal_mode,
                "observation_count": r.observation_count,
                "evidence_state": r.evidence_state,
                "spatial_support": r.spatial_support,
                "temporal_support": r.temporal_support,
                "observation_reliability": r.observation_reliability,
                "interpretation_support": r.interpretation_support,
                "conclusion": r.conclusion,
                "checks_passed": len(r.checks_passed),
                "errors": r.errors,
                "warnings": r.warnings,
            }
            for r in results
        ],
    }

    if save_report:
        reports_dir = resolve_repo_path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / "golden_suite_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n[Golden Suite] Report saved to: {report_file}")

    return results, summary_data


def generate_golden_report(results: List[GoldenValidationResult]) -> str:
    """Formats a concise, human-readable terminal table."""
    lines: List[str] = [
        "",
        "=" * 120,
        "PHASE 10: SATQUERY AI GOLDEN VALIDATION SUITE RESULTS",
        "=" * 120,
        f"{'Query ID':<26} {'Intent':<16} {'Temporal Mode':<16} {'Obs':<5} {'Evidence':<10} {'Spatial':<10} {'Temporal':<10} {'Reliability':<12} {'Result':<6}",
        "-" * 120,
    ]

    for r in results:
        status_str = "PASS" if r.passed else "FAIL"
        lines.append(
            f"{r.query_id:<26} "
            f"{r.intent[:15]:<16} "
            f"{r.temporal_mode[:15]:<16} "
            f"{r.observation_count:<5} "
            f"{str(r.evidence_state)[:9]:<10} "
            f"{str(r.spatial_support)[:9]:<10} "
            f"{str(r.temporal_support)[:9]:<10} "
            f"{str(r.observation_reliability)[:11]:<12} "
            f"{status_str:<6}"
        )

    lines.append("=" * 120)
    passed_count = sum(1 for r in results if r.passed)
    lines.append(f"TOTAL: {len(results)} | PASSED: {passed_count} | FAILED: {len(results) - passed_count}")
    lines.append("=" * 120)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI Phase 10 Golden Suite Runner")
    parser.add_argument("--no-save", action="store_true", help="Do not save reports/golden_suite_report.json")
    args = parser.parse_args()

    results, _ = run_golden_suite(save_report=not args.no_save)
    print(generate_golden_report(results))

    failed = any(not r.passed for r in results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
