"""
Phase 9: Comprehensive Benchmark Runner & Report Generator CLI.

Adheres strictly to the scientific guidelines and Phase 9C requirements:
1. Validates dataset integrity and split isolation.
2. If real reference labels are pending, reports:
   "Benchmark infrastructure is complete; numerical evaluation is pending validated reference labels."
   and sets ML STATUS = DEFERRED.
3. When validated reference labels exist:
   - Evaluates ONLY examples with status "validated".
   - Does NOT evaluate pending_reference_label examples.
   - For single-scene / TRAIN-only validated examples:
     - Labels all results: "PRELIMINARY SINGLE-SCENE BENCHMARK"
     - Discloses ESA WorldCover algorithm/version effects vs physical land-cover change.
     - Preserves label_type = "derived_reference".
     - Keeps ML STATUS = DEFERRED.
     - Final status: PHASE_9C_INITIAL_BENCHMARK_COMPLETE.
4. Generates structured reports:
   - overall_metrics.json
   - per_class_metrics.json
   - per_scene_metrics.json
   - region_metrics.json
   - confusion_matrix.json
   - error_analysis.json
   - threshold_sensitivity.json
   - BENCHMARK_REPORT.md
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

import cv2
import numpy as np
import rasterio

from app.evaluation.baselines import (
    explore_threshold_sensitivity,
    predict_index_threshold,
    predict_no_change,
    run_deterministic_satquery,
)
from app.evaluation.error_analysis import analyze_scene_errors
from app.evaluation.metrics import (
    calculate_balanced_accuracy,
    calculate_confusion_matrix,
    calculate_macro_metrics,
    calculate_overall_accuracy,
    calculate_per_class_metrics,
    match_regions,
)
from app.evaluation.paths import (
    get_manifest_path,
    get_reports_dir,
    resolve_repo_path,
)
from app.evaluation.validator import validate_benchmark_manifest


CLASS_NAMES = {
    0: "no_change",
    1: "urban_expansion",
    2: "urban_reduction",
    3: "vegetation_loss",
    4: "vegetation_gain",
    5: "water_loss",
    6: "water_gain",
    7: "ambiguous",
    8: "invalid",
}


def load_bands(ex: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """Loads before/after multi-spectral bands and ground-truth raster."""
    gt_path = resolve_repo_path(ex["ground_truth_path"])
    with rasterio.open(gt_path) as src:
        gt = src.read(1)
        meta = src.meta.copy()

    b_p = ex["image_before_path"]
    a_p = ex["image_after_path"]

    def read_b(p: str) -> np.ndarray:
        full_p = resolve_repo_path(p)
        with rasterio.open(full_p) as s:
            d = s.read(1).astype(np.float32)
            return d / 10000.0 if np.nanmax(d) > 2.0 else d

    bands_b = {k: read_b(b_p[k]) for k in ["red", "green", "nir", "swir"]}
    bands_a = {k: read_b(a_p[k]) for k in ["red", "green", "nir", "swir"]}

    return bands_b, bands_a, gt, meta


def _write_stage_a_report(
    reports_path: Path,
    overall_metrics: Dict[str, Any],
    manifest_data: Dict[str, Any],
    region_iou_threshold: float,
) -> None:
    """Write Stage A BENCHMARK_REPORT.md when 0 reference labels are validated."""
    report_md = f"""# SatQuery AI: Phase 9 Scientific Benchmark Report

**Status**: INFRASTRUCTURE COMPLETE — BENCHMARKING PENDING DATASET
**Evaluation Date**: {overall_metrics['benchmark_date']}
**Dataset Version**: {manifest_data.get('benchmark_version', '1.0.0')}
**Evaluation Stage**: STAGE A (Infrastructure Verified; Pending Independent Reference Labels)

---

## 1. Scientific Status Statement

Benchmark infrastructure is complete; numerical evaluation is pending validated reference labels.

In accordance with Phase 9 scientific safeguards:
1. No synthetic or self-derived rasters are reported as empirical ground truth.
2. Numerical precision, recall, F1, and IoU metrics are NOT fabricated.
3. Machine Learning training is DEFERRED until an independently sourced, validated reference dataset is ingested.

---

## 2. OSCD Dataset Rejection

OSCD (Onera Satellite Change Detection) was assessed and REJECTED.
Reason: OSCD covers 24 cities (Abu Dhabi, Aguas Claras, Beihai, Beirut, Bercy, Bordeaux,
Brasilia, Chongqing, Cupertino, Dubai, Hong Kong, Las Vegas, Milano, Montpellier, Mumbai,
Nantes, Norcia, Paris, Pisa, Rennes, Rio, Saclay E/W, Valencia).
None of our AOIs (Vienna T33UXP, Mumbai T43QCA, Queensland T56HLH) appear in that list.

## 3. Benchmark Architecture Verified

- Manifest Schema and Validator: Verified clean (0 errors, split leakage detection active).
- Metric Engine: Deterministic confusion matrix, macro metrics, balanced accuracy tested.
- Region Matching Engine: Bipartite greedy matching at IoU >= {region_iou_threshold} with MMU filtering.
- Error Analysis Taxonomy: 8 physical diagnostic failure categories implemented.
- Production Isolation: /api/query remains completely isolated from benchmark assets.
"""
    with open(reports_path / "BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_md)


def _run_preliminary_benchmark(
    evaluated_examples: List[Dict[str, Any]],
    all_examples: List[Dict[str, Any]],
    materialized_examples: List[Dict[str, Any]],
    pending_examples: List[Dict[str, Any]],
    manifest_data: Dict[str, Any],
    reports_path: Path,
    random_seed: int,
    enable_ml: bool,
    region_iou_threshold: float,
) -> Dict[str, Any]:
    """
    Executes Phase 9C deterministic numerical benchmarking on validated reference labels.
    Clearly labeled as PRELIMINARY SINGLE-SCENE BENCHMARK when sample size is limited.
    """
    t_start = time.time()
    print(f"\n  Evaluating {len(evaluated_examples)} validated example(s)...")

    cm_all = np.zeros((8, 8), dtype=np.int64)
    total_eval_px = 0
    total_ign_px = 0
    per_scene_results: List[Dict[str, Any]] = []
    all_region_metrics: List[Dict[str, Any]] = []
    all_error_analyses: List[Dict[str, Any]] = []

    for ex in evaluated_examples:
        t0 = time.time()
        bands_b, bands_a, gt, meta = load_bands(ex)
        invalid = (gt == 8)

        # Run deterministic SatQuery pipeline (Phases 5A-8)
        pred, raw_pred, diag = run_deterministic_satquery(
            bands_b["red"], bands_b["green"], bands_b["nir"], bands_b["swir"],
            bands_a["red"], bands_a["green"], bands_a["nir"], bands_a["swir"],
            invalid_mask=invalid,
            candidate_threshold=0.45,
            mmu_pixels=4,
        )

        # Confusion matrix excluding class 8 invalid pixels
        cm, ev_px, ig_px = calculate_confusion_matrix(gt, pred, num_classes=8)
        cm_all += cm
        total_eval_px += ev_px
        total_ign_px += ig_px

        per_cls = calculate_per_class_metrics(cm, CLASS_NAMES)
        macro = calculate_macro_metrics(per_cls)

        # Region-level spatial matching
        gt_bin = (gt > 0) & (gt < 8)
        pred_bin = (pred > 0) & (pred < 8)
        reg_met = match_regions(gt_bin, pred_bin, iou_threshold=region_iou_threshold, min_area_pixels=4)
        all_region_metrics.append({"example_id": ex["example_id"], **reg_met})

        # No-change baseline comparison
        nc_pred = predict_no_change(gt.shape)
        nc_cm, _, _ = calculate_confusion_matrix(gt, nc_pred, num_classes=8, valid_mask=~invalid)
        nc_per_cls = calculate_per_class_metrics(nc_cm, CLASS_NAMES)
        nc_macro = calculate_macro_metrics(nc_per_cls)

        # Diagnostic error analysis
        eps = 1e-6
        ndvi_b = (bands_b["nir"] - bands_b["red"]) / (bands_b["nir"] + bands_b["red"] + eps)
        ndvi_a = (bands_a["nir"] - bands_a["red"]) / (bands_a["nir"] + bands_a["red"] + eps)
        ndwi_b = (bands_b["green"] - bands_b["nir"]) / (bands_b["green"] + bands_b["nir"] + eps)
        ndwi_a = (bands_a["green"] - bands_a["nir"]) / (bands_a["green"] + bands_a["nir"] + eps)
        ndbi_b = (bands_b["swir"] - bands_b["nir"]) / (bands_b["swir"] + bands_b["nir"] + eps)
        ndbi_a = (bands_a["swir"] - bands_a["nir"]) / (bands_a["swir"] + bands_a["nir"] + eps)

        scene_err = analyze_scene_errors(
            y_true=gt,
            y_pred=pred,
            ndvi_before=ndvi_b,
            ndvi_after=ndvi_a,
            ndwi_before=ndwi_b,
            ndwi_after=ndwi_a,
            ndbi_before=ndbi_b,
            ndbi_after=ndbi_a,
            raw_pred_before_mmu=raw_pred,
            valid_mask=~invalid,
        )
        all_error_analyses.append({"example_id": ex["example_id"], **scene_err})

        elapsed = round(time.time() - t0, 3)
        per_scene_results.append({
            "example_id": ex["example_id"],
            "split": ex.get("split"),
            "region_id": ex.get("region_id"),
            "target_class": ex.get("target_class"),
            "label_type": ex.get("label_type", "derived_reference"),
            "evaluated_pixels": ev_px,
            "ignored_pixels": ig_px,
            "macro_precision": macro["macro_precision"],
            "macro_recall": macro["macro_recall"],
            "macro_f1": macro["macro_f1"],
            "macro_iou": macro["macro_iou"],
            "no_change_baseline_f1": nc_macro["macro_f1"],
            "satquery_vs_no_change_improvement": round(
                macro["macro_f1"] - nc_macro["macro_f1"], 4
            ),
            "region_precision": reg_met.get("region_precision", 0.0),
            "region_recall": reg_met.get("region_recall", 0.0),
            "region_f1": reg_met.get("region_f1", 0.0),
            "overlapping_gt_regions": reg_met.get("overlapping_gt_regions", 0),
            "detection_rate": reg_met.get("detection_rate", 0.0),
            "mean_centroid_error_px": reg_met.get("centroid_distance_mean_px", 0.0),
            "mean_area_error_m2": reg_met.get("area_absolute_error_mean_m2", 0.0),
            "runtime_sec": elapsed,
        })
        print(f"  [{ex['example_id']}] macro_F1={macro['macro_f1']:.4f}, "
              f"eval_px={ev_px}, ign_px={ig_px}, region_F1={reg_met['region_f1']:.4f}, "
              f"det_rate={reg_met.get('detection_rate', 0.0)*100:.1f}%")

    det_per_class = calculate_per_class_metrics(cm_all, CLASS_NAMES)
    det_macro = calculate_macro_metrics(det_per_class)
    det_oa = calculate_overall_accuracy(cm_all)
    det_ba = calculate_balanced_accuracy(cm_all)

    is_single_scene = len(evaluated_examples) == 1
    distinct_regions = len(set(e.get("region_id") for e in evaluated_examples))
    distinct_countries = sorted(set(e.get("country", "") for e in evaluated_examples if e.get("country")))

    if is_single_scene:
        benchmark_title = "PRELIMINARY SINGLE-SCENE BENCHMARK"
        benchmark_status = "PHASE_9C_INITIAL_BENCHMARK_COMPLETE"
        sample_warning = (
            "PRELIMINARY SINGLE-SCENE BENCHMARK: Sample size (N=1 scene pair) is insufficient "
            "for broad performance generalization. Do NOT claim global SatQuery accuracy or generalized F1. "
            "Results reflect agreement with a single ESA WorldCover derived reference raster."
        )
        status_msg = (
            "PRELIMINARY SINGLE-SCENE BENCHMARK complete. "
            f"{len(evaluated_examples)} validated example evaluated against real ESA WorldCover derived reference raster. "
            "Phase 9 full benchmark requires multi-scene geographically diverse test set."
        )
    else:
        benchmark_title = "VALIDATED MULTI-SCENE BENCHMARK"
        benchmark_status = "PHASE_9C_MULTI_SCENE_BENCHMARK_COMPLETE"
        sample_warning = (
            f"VALIDATED MULTI-SCENE BENCHMARK: Evaluated N={len(evaluated_examples)} scene pairs across "
            f"{len(distinct_countries)} countries ({', '.join(distinct_countries)}) against real ESA WorldCover derived reference rasters. "
            "Reference masks are designated strictly as derived_reference, not absolute ground truth."
        )
        status_msg = (
            f"VALIDATED MULTI-SCENE BENCHMARK complete: {len(evaluated_examples)} validated examples evaluated across "
            f"{distinct_regions} distinct regions against real ESA WorldCover derived reference rasters."
        )

    print(f"\n  [NOTE] {sample_warning}")

    worldcover_disclosure = (
        "ESA WorldCover 2020 (v100) and 2021 (v200) use different classification algorithms. "
        "Observed differences may include algorithm/version effects in addition to physical land-cover change. "
        "Reference masks are designated strictly as derived_reference, not absolute ground truth."
    )

    # 1. confusion_matrix.json
    cm_serializable = cm_all.tolist()
    with open(reports_path / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_title": benchmark_title,
            "confusion_matrix": cm_serializable,
            "class_names": CLASS_NAMES,
            "total_evaluated_pixels": total_eval_px,
            "total_ignored_pixels": total_ign_px,
            "note": "Rows=true class, Cols=predicted class. Class 8 (invalid/nodata) excluded from evaluation.",
        }, f, indent=2)

    # 2. per_class_metrics.json
    with open(reports_path / "per_class_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_title": benchmark_title,
            "per_class": det_per_class,
            "macro_metrics": det_macro,
        }, f, indent=2)

    # 3. per_scene_metrics.json
    with open(reports_path / "per_scene_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_title": benchmark_title,
            "per_scene_results": per_scene_results,
        }, f, indent=2)

    # 4. region_metrics.json
    with open(reports_path / "region_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_title": benchmark_title,
            "iou_threshold_used": region_iou_threshold,
            "min_mmu_pixels": 4,
            "region_metrics": all_region_metrics,
        }, f, indent=2)

    # 5. error_analysis.json
    with open(reports_path / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_title": benchmark_title,
            "scenes": all_error_analyses,
            "disclaimer": worldcover_disclosure,
        }, f, indent=2)

    # 6. threshold_sensitivity.json
    val_examples = [e for e in evaluated_examples if e.get("split") == "VALIDATION"]
    threshold_sensitivity_res = None
    if val_examples:
        try:
            threshold_sensitivity_res = explore_threshold_sensitivity(val_examples)
        except Exception as exc:
            print(f"  [WARNING] Threshold sensitivity analysis error: {exc}")

    with open(reports_path / "threshold_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_title": benchmark_title,
            "threshold_sensitivity": threshold_sensitivity_res,
            "val_scenes_evaluated": len(val_examples),
            "note": f"Threshold sensitivity evaluated on {len(val_examples)} VALIDATION scene(s). Production thresholds left unaltered at candidate_threshold=0.45, MMU=4." if val_examples else "Validation-only threshold analysis deferred: zero VALIDATION examples materialized.",
            "policy": "No threshold tuning on TEST split; production thresholds left unaltered.",
        }, f, indent=2)

    # 7. overall_metrics.json
    nc_f1_avg = round(float(np.mean([sc["no_change_baseline_f1"] for sc in per_scene_results])), 4) if per_scene_results else 0.0
    overall_metrics: Dict[str, Any] = {
        "benchmark_title": benchmark_title,
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_version": manifest_data.get("benchmark_version", "1.1.0"),
        "benchmark_status": benchmark_status,
        "stage": "STAGE_B_PARTIAL_NUMERICAL_EVALUATION",
        "status_message": status_msg,
        "ml_status": "DEFERRED",
        "candidate_examples": len(all_examples),
        "total_candidate_scenes": len(all_examples),
        "materialized_examples": len(materialized_examples),
        "materialized_labeled_scenes": len(materialized_examples),
        "validated_examples": len(evaluated_examples),
        "evaluated_examples": len(evaluated_examples),
        "evaluated_pixel_count": total_eval_px,
        "ignored_pixel_count": total_ign_px,
        "splits_evaluated": sorted(list({e.get("split") for e in evaluated_examples if e.get("split")})),
        "splits_pending": sorted(list({e.get("split") for e in pending_examples if e.get("split")})),
        "disclaimer": sample_warning,
        "reference_label_caveat": worldcover_disclosure,
        "deterministic_satquery": {
            "overall_accuracy": det_oa,
            "balanced_accuracy": det_ba,
            **det_macro,
            "label_type": "derived_reference",
        },
        "baselines_comparison": {
            "no_change_f1": nc_f1_avg,
            "deterministic_satquery_f1": det_macro["macro_f1"],
        },
        "sample_size_warning": sample_warning,
        "oscd_rejection": manifest_data.get("oscd_rejection_record", {}),
    }

    with open(reports_path / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    # 8. BENCHMARK_REPORT.md
    dm = overall_metrics["deterministic_satquery"]
    report_lines = [
        "# SatQuery AI: Phase 9 Scientific Benchmark Report",
        "",
        f"**Benchmark Scope**: {benchmark_title}  ",
        f"**Benchmark Status**: {benchmark_status}  ",
        f"**Evaluation Date**: {overall_metrics['benchmark_date']}  ",
        f"**Dataset Version**: {manifest_data.get('benchmark_version', '1.1.0')}  ",
        f"**ML Status**: DEFERRED (No ML training conducted or claimed)  ",
        "",
        "---",
        "",
        "## 1. Scientific Disclosures & Benchmark Scope",
        "",
        "> [!IMPORTANT]",
        f"> **{sample_warning}**",
        "",
        f"- **Reference Label Qualification**: {worldcover_disclosure}",
        f"- **Status Distinction**: This milestone represents `{benchmark_status}`.",
        "  Full comprehensive benchmark requires post-2021 independent reference data across all splits.",
        "- **Locked Test Split**: Test-set scenes remain locked with zero threshold tuning applied.",
        "- **Production Integrity**: Production thresholds and `/api/query` algorithms remain completely unaltered.",
        "",
        "---",
        "",
        "## 2. Sample Accounting",
        "",
        "| Metric | Count | Description |",
        "|:---|:---:|:---|",
        f"| **Candidate Examples** | {overall_metrics['candidate_examples']} | Total multi-temporal scene pairs across all splits in manifest |",
        f"| **Materialized Examples** | {overall_metrics['materialized_examples']} | Examples with ingested reference rasters on disk |",
        f"| **Validated Examples** | {overall_metrics['validated_examples']} | Materialized examples passing strict integrity & class checks |",
        f"| **Evaluated Examples** | {overall_metrics['evaluated_examples']} | Examples evaluated by the deterministic benchmark runner |",
        f"| **Evaluated Pixel Count** | {overall_metrics['evaluated_pixel_count']:,} | Total non-invalid pixels evaluated |",
        f"| **Ignored Pixel Count** | {overall_metrics['ignored_pixel_count']:,} | Cloud, shadow, and nodata pixels excluded from evaluation |",
        "",
        "---",
        "",
        "## 3. Pixel-Level Metrics — Deterministic SatQuery",
        "",
        "| Metric | Value |",
        "|:---|:---:|",
        f"| **Overall Accuracy** | {dm['overall_accuracy']:.4f} |",
        f"| **Balanced Accuracy** | {dm['balanced_accuracy']:.4f} |",
        f"| **Macro Precision** | {dm['macro_precision']:.4f} |",
        f"| **Macro Recall** | {dm['macro_recall']:.4f} |",
        f"| **Macro F1** | {dm['macro_f1']:.4f} |",
        f"| **Macro IoU** | {dm['macro_iou']:.4f} |",
        "",
        "---",
        "",
        "## 4. Region-Level Spatial Matching",
        "",
        f"Region matching performed at $\\text{{IoU}} \\ge {region_iou_threshold:.2f}$ with $\\text{{MMU}} = 4$ pixels ($400\\,\\text{{m}}^2$).",
        "",
        "| Example ID | Target Class | Region Precision | Region Recall | Region F1 | Detection Rate | Mean Centroid Error | Mean Area Error |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for sc in per_scene_results:
        report_lines.append(
            f"| {sc['example_id']} | {sc['target_class']} | "
            f"{sc['region_precision']:.4f} | {sc['region_recall']:.4f} | {sc['region_f1']:.4f} | "
            f"{sc.get('detection_rate', 0.0) * 100:.1f}% | "
            f"{sc['mean_centroid_error_px']:.2f} px | {sc.get('mean_area_error_m2', 0.0):.1f} m² |"
        )

    # Aggregate error analysis
    agg_cat_counts: Dict[str, int] = {}
    tot_diag_pixels = 0
    for ea in all_error_analyses:
        for cat, cnt in ea.get("category_counts", {}).items():
            agg_cat_counts[cat] = agg_cat_counts.get(cat, 0) + cnt
            tot_diag_pixels += cnt

    report_lines += [
        "",
        "---",
        "",
        "## 5. Diagnostic Error Analysis",
        "",
        f"Primary Failure Mode: **{all_error_analyses[0]['primary_failure_mode'] if all_error_analyses else 'NONE'}**",
        "",
        "| Diagnostic Category | Pixel Count | Percentage |",
        "|:---|:---:|:---:|",
    ]
    if agg_cat_counts:
        for cat, cnt in agg_cat_counts.items():
            pct = (cnt / tot_diag_pixels * 100.0) if tot_diag_pixels > 0 else 0.0
            report_lines.append(f"| `{cat}` | {cnt:,} | {pct:.1f}% |")

    report_lines += [
        "",
        "---",
        "",
        "## 6. Baseline Comparison",
        "",
        "| Pipeline | Macro F1 | Note |",
        "|:---|:---:|:---|",
        f"| **No-Change Baseline** | {overall_metrics['baselines_comparison']['no_change_f1']:.4f} | Predicts zero change across all pixels |",
        f"| **Deterministic SatQuery** | {overall_metrics['baselines_comparison']['deterministic_satquery_f1']:.4f} | Full production multi-index pipeline (Phases 5A–8) |",
        "",
        "---",
        "",
        "## 7. Next Steps for Full Phase 9 Benchmark",
        "",
        "1. Materialize additional post-2021 geographic reference pairs across Queensland and other regions once reference rasters are verified.",
        "2. Keep locked TEST split evaluated only once reference labels are independently validated.",
        "3. Machine Learning remains DEFERRED until a large multi-scene reference dataset is established.",
    ]

    report_md = "\n".join(report_lines)
    with open(reports_path / "BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n[Done] Benchmark reports written to: {reports_path}")
    return overall_metrics


def run_benchmark(
    manifest_path: Optional[str | Path] = None,
    report_dir: Optional[str | Path] = None,
    random_seed: int = 42,
    enable_ml: bool = False,
    region_iou_threshold: float = 0.30,
) -> Dict[str, Any]:
    """
    Main benchmark execution entrypoint.
    Resolves manifest and reports directory canonically.
    """
    manifest_file = get_manifest_path() if manifest_path is None else resolve_repo_path(manifest_path)
    reports_path = get_reports_dir() if report_dir is None else resolve_repo_path(report_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("SATQUERY AI: PHASE 9 BENCHMARK EVALUATION")
    print("==================================================")

    # 1. Validation
    print("\n[Step 1/4] Validating benchmark manifest and integrity...")
    val_res = validate_benchmark_manifest(manifest_file)
    if not val_res.is_valid:
        raise ValueError(f"Manifest validation failed with {val_res.error_count} errors: {val_res.issues}")

    print(f"  Manifest valid: {val_res.total_examples} candidate scenes.")
    print(f"  Materialized scenes: {val_res.materialized_count} | Validated scenes: {val_res.validated_count} | Pending: {val_res.pending_count}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    all_examples = manifest_data.get("candidate_examples") or manifest_data.get("examples", [])

    materialized_examples = [
        e for e in all_examples
        if e.get("ground_truth_path") and e.get("status") in ("materialized", "validated")
    ]

    # Rule 4: Execute deterministic benchmark ONLY on examples whose manifest status is "validated".
    # Rule 5: Do not evaluate pending_reference_label examples.
    validated_examples = [
        e for e in all_examples
        if e.get("status") == "validated" and e.get("ground_truth_path")
    ]

    pending_examples = [
        e for e in all_examples
        if e.get("status") == "pending_reference_label" or not e.get("ground_truth_path")
    ]

    # STAGE A BRANCH: zero validated reference labels
    if len(validated_examples) == 0:
        print("\n" + "=" * 60)
        print("STAGE A INFRASTRUCTURE VERIFICATION COMPLETE")
        print("STATUS: Benchmark infrastructure is complete; numerical evaluation is pending validated reference labels.")
        print("ML STATUS: DEFERRED (No validated reference-labeled dataset is materialized)")
        print("=" * 60)

        overall_metrics = {
            "benchmark_title": "STAGE A BENCHMARK INFRASTRUCTURE",
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": "STAGE_A_INFRASTRUCTURE_COMPLETE",
            "benchmark_status": "INFRASTRUCTURE_COMPLETE_PENDING_DATASET",
            "status_message": "Benchmark infrastructure is complete; numerical evaluation is pending validated reference labels.",
            "ml_status": "DEFERRED",
            "candidate_examples": len(all_examples),
            "materialized_examples": len(materialized_examples),
            "validated_examples": 0,
            "evaluated_examples": 0,
            "total_candidate_scenes": len(all_examples),
            "materialized_labeled_scenes": len(materialized_examples),
            "pending_reference_scenes": len(pending_examples),
            "discovered_datasets": manifest_data.get("discovered_independent_datasets", []),
            "deterministic_satquery": None,
            "baselines_comparison": None,
            "sample_size_warning": "Zero validated reference labels. No numerical metrics computed.",
            "note": "Numerical metrics are deliberately not fabricated in accordance with scientific safeguards.",
        }

        with open(reports_path / "overall_metrics.json", "w", encoding="utf-8") as f:
            json.dump(overall_metrics, f, indent=2)

        _write_stage_a_report(reports_path, overall_metrics, manifest_data, region_iou_threshold)
        print(f"\nReport generated: {reports_path / 'BENCHMARK_REPORT.md'}")
        return overall_metrics

    # PRELIMINARY BENCHMARK BRANCH: Validated reference labels exist
    print("\n" + "=" * 60)
    print(f"PRELIMINARY BENCHMARK: {len(validated_examples)} validated example(s) evaluated.")
    print("Executing deterministic SatQuery evaluation against real reference raster.")
    print("=" * 60)

    return _run_preliminary_benchmark(
        evaluated_examples=validated_examples,
        all_examples=all_examples,
        materialized_examples=materialized_examples,
        pending_examples=pending_examples,
        manifest_data=manifest_data,
        reports_path=reports_path,
        random_seed=random_seed,
        enable_ml=enable_ml,
        region_iou_threshold=region_iou_threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SatQuery Phase 9 Benchmark Suite")
    parser.add_argument("--manifest", default=None, help="Path to manifest JSON")
    parser.add_argument("--report-dir", default=None, help="Directory to store reports")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--region-iou", type=float, default=0.30, help="Configurable region matching IoU threshold")
    parser.add_argument("--enable-ml", action="store_true", help="Enable ML training (deferred by default)")

    args = parser.parse_args()
    run_benchmark(
        manifest_path=args.manifest,
        report_dir=args.report_dir,
        random_seed=args.seed,
        enable_ml=args.enable_ml,
        region_iou_threshold=args.region_iou,
    )
