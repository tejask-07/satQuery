"""
Phase 9: Benchmark & Evaluation API Router.

Strictly isolated from /api/query/.
Exposes developer/research inspection endpoints for benchmark status,
latest evaluation metrics, ablation results, and dataset metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.evaluation.paths import get_manifest_path, get_reports_dir

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

REPORTS_DIR = get_reports_dir()
MANIFEST_PATH = get_manifest_path()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading {path.name}: {str(e)}")


@router.get("/summary")
def get_benchmark_summary() -> Dict[str, Any]:
    """
    Returns high-level summary of benchmark dataset and latest evaluation results.
    """
    summary: Dict[str, Any] = {
        "benchmark_ready": MANIFEST_PATH.exists(),
        "has_reports": (REPORTS_DIR / "overall_metrics.json").exists(),
    }

    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                m = json.load(f)
            summary["dataset_version"] = m.get("benchmark_version", "1.0.0")
            summary["total_scenes"] = m.get("total_candidate_examples", m.get("total_examples", 0))
            summary["split_counts"] = m.get("split_counts", {})
            summary["benchmark_status"] = m.get("status", "pending_reference_labels")
            summary["stage"] = m.get("stage", "STAGE_A_INFRASTRUCTURE")
        except Exception:
            pass

    if (REPORTS_DIR / "overall_metrics.json").exists():
        try:
            with open(REPORTS_DIR / "overall_metrics.json", "r", encoding="utf-8") as f:
                om = json.load(f)
            summary["last_evaluated"] = om.get("benchmark_date")
            summary["benchmark_status"] = om.get("benchmark_status", summary.get("benchmark_status"))
            summary["status_message"] = om.get("status_message")
            summary["ml_status"] = om.get("ml_status", "DEFERRED")
            summary["materialized_labeled_scenes"] = om.get("materialized_labeled_scenes", 0)
            summary["baselines"] = om.get("baselines_comparison")
        except Exception:
            pass

    return summary


def _read_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading {path.name}: {str(e)}")


@router.get("/reports")
def get_benchmark_reports() -> Dict[str, Any]:
    """
    Returns full structured reports from the latest benchmark run.
    """
    return {
        "overall": _read_json_safe(REPORTS_DIR / "overall_metrics.json"),
        "per_class": _read_json_safe(REPORTS_DIR / "per_class_metrics.json"),
        "per_scene": _read_json_safe(REPORTS_DIR / "per_scene_metrics.json"),
        "ablation": _read_json_safe(REPORTS_DIR / "ablation_study.json"),
    }


@router.get("/manifest")
def get_benchmark_manifest() -> Dict[str, Any]:
    """
    Returns the benchmark dataset manifest.
    """
    return _read_json(MANIFEST_PATH)
