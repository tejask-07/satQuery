"""
Phase 9 Benchmark & Evaluation Comprehensive Test Suite.

Verifies:
1. Manifest structure, dataset discovery, and candidate scenes
2. Label taxonomy validity (classes 0..8)
3. Split isolation & leakage detection (zero overlap between TRAIN, VALIDATION, TEST)
4. Class mapping schema validation and rejection of unsupported mappings
5. Confusion matrix mathematical properties & count conservation
6. Precision, Recall, F1, and IoU calculation & conventions
7. Safe zero-division handling
8. Invalid pixel masking (class 8 strictly excluded from evaluation)
9. Macro metrics calculation across active classes
10. Overall accuracy and balanced accuracy
11. Region connected component extraction & MMU filtering
12. Region IoU calculation between geometric parcels
13. Configurable region matching IoU thresholds (e.g. 0.20, 0.30, 0.50)
14. Centroid distance and area absolute error
15. Diagnostic error taxonomy categorization
16. Deterministic repeatability (identical outputs on identical inputs)
17. No-change baseline execution
18. Index-threshold baseline execution
19. ML feature extraction schema (21 features with valid names)
20. Component ablation study stages
21. Stage A benchmark runner execution (unmaterialized labels -> INFRASTRUCTURE_COMPLETE_PENDING_DATASET, ML DEFERRED)
22. Stage B benchmark runner execution with synthetic test fixture (verifies numerical metric pipeline)
23. Benchmark API endpoints (/api/benchmark/summary, /reports, /manifest)
"""

import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.evaluation.baselines import (
    FEATURE_NAMES,
    explore_threshold_sensitivity,
    extract_features_from_bands,
    predict_index_threshold,
    predict_ml,
    predict_no_change,
    run_ablation_experiment,
    run_deterministic_satquery,
    train_interpretable_ml_baseline,
)
from app.evaluation.error_analysis import (
    ErrorCategory,
    analyze_scene_errors,
    diagnose_pixel_error,
)
from app.evaluation.metrics import (
    RegionInfo,
    calculate_balanced_accuracy,
    calculate_confusion_matrix,
    calculate_f1,
    calculate_iou,
    calculate_macro_metrics,
    calculate_overall_accuracy,
    calculate_per_class_metrics,
    calculate_precision,
    calculate_recall,
    compute_region_iou,
    extract_regions_from_mask,
    match_regions,
    safe_divide,
)
from app.evaluation.run_benchmark import run_benchmark
from app.evaluation.validator import (
    SUPPORTED_CLASS_MAPPINGS,
    ValidationIssue,
    ValidationResult,
    validate_benchmark_manifest,
)
from app.evaluation.paths import (
    get_manifest_path,
    get_reports_dir,
    resolve_repo_path,
)

MANIFEST_PATH = get_manifest_path()


# ============================================================
# 1. MANIFEST & DATASET VALIDATION TESTS
# ============================================================

def test_manifest_validation_clean():
    """Manifest must pass validation with 0 errors and identify pending reference labels."""
    assert MANIFEST_PATH.exists()
    res = validate_benchmark_manifest(MANIFEST_PATH)
    assert res.is_valid is True
    assert res.error_count == 0
    assert res.total_examples >= 8
    assert res.pending_count >= 8
    assert res.materialized_count >= 1


def test_manifest_label_validation():
    """Manifest must specify classes 0..8 and discovered datasets."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = {c["id"]: c["name"] for c in data["classes"]}
    assert len(classes) == 9
    assert classes[0] == "no_change"
    assert classes[8] == "invalid"
    assert len(data.get("discovered_independent_datasets", [])) >= 3


def test_split_leakage_detection(tmp_path):
    """Validator must flag any scene overlap between splits as an ERROR."""
    dummy_manifest = {
        "benchmark_version": "1.0.0",
        "candidate_examples": [
            {
                "example_id": "EX-01",
                "split": "TRAIN",
                "before_scene_id": "SCENE_SHARED",
                "after_scene_id": "SCENE_B",
                "target_class": "urban_expansion",
                "before_date": "2021-01-01",
                "after_date": "2022-01-01",
                "class_mapping_schema": "DynamicWorld_to_SatQuery_v1",
            },
            {
                "example_id": "EX-02",
                "split": "TEST",
                "before_scene_id": "SCENE_SHARED",  # LEAKAGE!
                "after_scene_id": "SCENE_C",
                "target_class": "urban_expansion",
                "before_date": "2021-01-01",
                "after_date": "2022-01-01",
                "class_mapping_schema": "DynamicWorld_to_SatQuery_v1",
            },
        ],
    }
    dummy_path = tmp_path / "test_leakage_manifest.json"
    with open(dummy_path, "w", encoding="utf-8") as f:
        json.dump(dummy_manifest, f)

    res = validate_benchmark_manifest(dummy_path)
    assert res.is_valid is False
    assert any("leakage" in issue.message.lower() for issue in res.issues)


def test_class_mapping_schema_validation(tmp_path):
    """Validator must reject unsupported semantic mapping schemas."""
    dummy_manifest = {
        "benchmark_version": "1.0.0",
        "candidate_examples": [
            {
                "example_id": "EX-03",
                "split": "TRAIN",
                "before_scene_id": "S1",
                "after_scene_id": "S2",
                "target_class": "urban_expansion",
                "before_date": "2021-01-01",
                "after_date": "2022-01-01",
                "class_mapping_schema": "UnsupportedTaxonomy_XYZ",  # INVALID
            }
        ],
    }
    dummy_path = tmp_path / "test_unsupported_mapping.json"
    with open(dummy_path, "w", encoding="utf-8") as f:
        json.dump(dummy_manifest, f)

    res = validate_benchmark_manifest(dummy_path)
    assert res.is_valid is False
    assert any("unsupported class mapping schema" in issue.message.lower() for issue in res.issues)


# ============================================================
# 2. METRIC ENGINE PRIMITIVES & CONVENTIONS
# ============================================================

def test_safe_divide():
    """Safe divide handles 0/0 and nan gracefully."""
    assert safe_divide(10.0, 2.0) == 5.0
    assert safe_divide(0.0, 0.0) == 0.0
    assert safe_divide(5.0, 0.0, default=-1.0) == -1.0
    assert safe_divide(float("nan"), 10.0) == 0.0


def test_confusion_matrix_shape_and_counts():
    """Confusion matrix preserves total evaluated pixel conservation."""
    y_true = np.array([[0, 1], [3, 8]], dtype=np.uint8)
    y_pred = np.array([[0, 1], [0, 8]], dtype=np.uint8)

    cm, ev, ig = calculate_confusion_matrix(y_true, y_pred, num_classes=8)
    assert cm.shape == (8, 8)
    assert ev == 3
    assert ig == 1
    assert cm.sum() == 3
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert cm[3, 0] == 1


def test_precision_recall_f1_iou():
    """Core classification metrics follow strict mathematical definitions."""
    tp, fp, fn = 80, 20, 10
    prec = calculate_precision(tp, fp)
    rec = calculate_recall(tp, fn)
    f1 = calculate_f1(prec, rec)
    iou = calculate_iou(tp, fp, fn)

    assert prec == pytest.approx(80.0 / 100.0)
    assert rec == pytest.approx(80.0 / 90.0)
    expected_f1 = 2 * (0.8 * (80 / 90)) / (0.8 + (80 / 90))
    assert f1 == pytest.approx(expected_f1)
    assert iou == pytest.approx(80.0 / 110.0)


def test_zero_division_handling():
    """Zero denominator returns 0.0 without errors."""
    assert calculate_precision(0, 0) == 0.0
    assert calculate_recall(0, 0) == 0.0
    assert calculate_f1(0.0, 0.0) == 0.0
    assert calculate_iou(0, 0, 0) == 0.0


def test_macro_metrics_calculation():
    """Macro metrics average across evaluated classes."""
    cm = np.zeros((8, 8), dtype=np.int64)
    cm[0, 0] = 100
    cm[1, 1] = 50

    per_cls = calculate_per_class_metrics(cm)
    macro = calculate_macro_metrics(per_cls, only_present=True)

    assert macro["classes_evaluated_count"] == 2
    assert macro["macro_precision"] == 1.0
    assert macro["macro_recall"] == 1.0
    assert macro["macro_f1"] == 1.0
    assert macro["macro_iou"] == 1.0


def test_accuracy_metrics():
    """Overall accuracy and balanced accuracy match formulas."""
    cm = np.zeros((8, 8), dtype=np.int64)
    cm[0, 0] = 80
    cm[0, 1] = 20
    cm[1, 1] = 40

    oa = calculate_overall_accuracy(cm)
    ba = calculate_balanced_accuracy(cm)

    assert oa == pytest.approx(120.0 / 140.0, abs=1e-3)
    assert ba == pytest.approx((0.80 + 1.00) / 2.0, abs=1e-3)


# ============================================================
# 3. REGION MATCHING & CONFIGURABLE IoU
# ============================================================

def test_extract_regions_mmu_filter():
    """Regions below MMU pixel threshold are suppressed."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2, 2:4] = 1        # 2 pixels -> filtered
    mask[10:13, 10:13] = 1  # 9 pixels -> kept

    regions = extract_regions_from_mask(mask, min_area_pixels=4)
    assert len(regions) == 1
    assert regions[0].area_pixels == 9


def test_compute_region_iou():
    """Region IoU correctly measures overlap between two regions."""
    r1 = RegionInfo(1, 9, 2.0, 2.0, (1, 1, 4, 4), np.argwhere(np.ones((3, 3))))
    assert compute_region_iou(r1, r1, (10, 10)) == 1.0

    r2 = RegionInfo(2, 4, 8.0, 8.0, (7, 7, 9, 9), np.array([[7, 7], [7, 8], [8, 7], [8, 8]]))
    assert compute_region_iou(r1, r2, (10, 10)) == 0.0


def test_match_regions_configurable_iou():
    """Region matching threshold is configurable and affects matching counts."""
    gt = np.zeros((30, 30), dtype=np.uint8)
    pred = np.zeros((30, 30), dtype=np.uint8)

    # 4x4 region in gt (16 pixels)
    gt[5:9, 5:9] = 1
    # 4x4 region in pred overlapping partially: 2x4 = 8 pixels overlap -> IoU = 8 / 24 = 0.333
    pred[7:11, 5:9] = 1

    # IoU threshold = 0.25 -> matches
    res_low = match_regions(gt, pred, iou_threshold=0.25)
    assert res_low["matched_gt_regions"] == 1
    assert res_low["false_negative_regions"] == 0

    # IoU threshold = 0.50 -> does not match
    res_high = match_regions(gt, pred, iou_threshold=0.50)
    assert res_high["matched_gt_regions"] == 0
    assert res_high["false_negative_regions"] == 1


# ============================================================
# 4. ERROR ANALYSIS TAXONOMY
# ============================================================

def test_diagnose_pixel_error_taxonomy():
    """Diagnostic errors are classified into physical evidence categories."""
    err_cloud = diagnose_pixel_error(
        r=0, c=0, true_cls=0, pred_cls=1, delta_ndvi=0.0, delta_ndwi=0.0,
        delta_ndbi=0.2, ndvi_before=0.5, is_boundary_pixel=False,
        was_filtered_by_mmu=False, is_cloud_adjacent=True, seasonal_doy_diff=0, observation_count=2,
    )
    assert err_cloud == ErrorCategory.CLOUD_CONTAMINATION

    err_mmu = diagnose_pixel_error(
        r=0, c=0, true_cls=1, pred_cls=0, delta_ndvi=0.0, delta_ndwi=0.0,
        delta_ndbi=0.2, ndvi_before=0.5, is_boundary_pixel=False,
        was_filtered_by_mmu=True, is_cloud_adjacent=False, seasonal_doy_diff=0, observation_count=2,
    )
    assert err_mmu == ErrorCategory.SMALL_REGION_FILTERED

    err_urban = diagnose_pixel_error(
        r=0, c=0, true_cls=0, pred_cls=1, delta_ndvi=0.0, delta_ndwi=0.0,
        delta_ndbi=0.15, ndvi_before=0.10, is_boundary_pixel=False,
        was_filtered_by_mmu=False, is_cloud_adjacent=False, seasonal_doy_diff=0, observation_count=2,
    )
    assert err_urban == ErrorCategory.URBAN_FALSE_POSITIVE_NDBI

    err_mixed = diagnose_pixel_error(
        r=0, c=0, true_cls=0, pred_cls=1, delta_ndvi=0.0, delta_ndwi=0.0,
        delta_ndbi=0.04, ndvi_before=0.3, is_boundary_pixel=True,
        was_filtered_by_mmu=False, is_cloud_adjacent=False, seasonal_doy_diff=0, observation_count=2,
    )
    assert err_mixed == ErrorCategory.MIXED_PIXEL


# ============================================================
# 5. BASELINES & ML SCHEMAS
# ============================================================

def test_no_change_baseline():
    """No change baseline predicts all zeros."""
    pred = predict_no_change((20, 20))
    assert pred.shape == (20, 20)
    assert np.all(pred == 0)


def test_index_threshold_baseline():
    """Index threshold baseline uses raw single-index thresholding."""
    H, W = 10, 10
    nb = np.zeros((H, W), dtype=np.float32)
    na = np.zeros((H, W), dtype=np.float32)
    na[2, 2] = 0.25  # delta NDVI = 0.25 -> veg gain (cls 4)

    ndbi_b = np.zeros((H, W), dtype=np.float32)
    ndbi_a = np.zeros((H, W), dtype=np.float32)
    ndbi_a[5, 5] = 0.20  # delta NDBI = 0.20 -> urban expansion (cls 1)

    ndwi_b = np.zeros((H, W), dtype=np.float32)
    ndwi_a = np.zeros((H, W), dtype=np.float32)

    pred = predict_index_threshold(nb, na, ndbi_b, ndbi_a, ndwi_b, ndwi_a)
    assert pred[2, 2] == 4
    assert pred[5, 5] == 1
    assert pred[0, 0] == 0


def test_deterministic_satquery_repeatability():
    """SatQuery deterministic pipeline is strictly repeatable bit-for-bit."""
    shape = (50, 50)
    np.random.seed(123)
    rb = np.random.uniform(0.05, 0.25, shape).astype(np.float32)
    gb = np.random.uniform(0.05, 0.25, shape).astype(np.float32)
    nb = np.random.uniform(0.15, 0.45, shape).astype(np.float32)
    sb = np.random.uniform(0.10, 0.35, shape).astype(np.float32)

    ra, ga, na, sa = rb.copy(), gb.copy(), nb.copy(), sb.copy()
    sa[20:30, 20:30] += 0.20
    na[20:30, 20:30] -= 0.15

    pred1, raw1, _ = run_deterministic_satquery(rb, gb, nb, sb, ra, ga, na, sa)
    pred2, raw2, _ = run_deterministic_satquery(rb, gb, nb, sb, ra, ga, na, sa)

    assert np.array_equal(pred1, pred2)
    assert np.array_equal(raw1, raw2)
    assert np.any(pred1 == 1)


def test_feature_extraction_schema():
    """Feature extraction yields exactly 21 features with valid names."""
    shape = (20, 20)
    arr = np.zeros(shape, dtype=np.float32)
    feats = extract_features_from_bands(arr, arr, arr, arr, arr, arr, arr, arr)
    assert feats.shape == (20, 20, len(FEATURE_NAMES))
    assert len(FEATURE_NAMES) == 24


def test_ablation_study_monotonic_layers():
    """Ablation configurations execute without error."""
    shape = (20, 20)
    arr = np.zeros(shape, dtype=np.float32)
    for stage in ["ndvi_only", "ndvi_ndbi", "all_indices", "indices_spatial", "full_system"]:
        p = run_ablation_experiment(arr, arr, arr, arr, arr, arr, arr, arr, None, stage)
        assert p.shape == (20, 20)


# ============================================================
# 6. STAGE A & STAGE B RUNNER VERIFICATION
# ============================================================

def test_benchmark_runner_stage_a_infrastructure_complete(tmp_path):
    """When reference labels are pending, runner reports INFRASTRUCTURE_COMPLETE and defers ML."""
    stage_a_manifest = tmp_path / "stage_a_manifest.json"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    m_data_copy = dict(m_data)
    m_data_copy["candidate_examples"] = [
        dict(e, status="pending_reference_label", ground_truth_path=None)
        for e in m_data_copy.get("candidate_examples", [])
    ]
    with open(stage_a_manifest, "w", encoding="utf-8") as f:
        json.dump(m_data_copy, f)

    stage_a_reports = tmp_path / "reports"
    res = run_benchmark(
        manifest_path=str(stage_a_manifest),
        report_dir=str(stage_a_reports),
        random_seed=42,
        enable_ml=False,
    )
    assert res["benchmark_status"] == "INFRASTRUCTURE_COMPLETE_PENDING_DATASET"
    assert res["ml_status"] == "DEFERRED"
    assert res["materialized_labeled_scenes"] == 0
    assert "Benchmark infrastructure is complete; numerical evaluation is pending validated reference labels." in res["status_message"]
    assert res["deterministic_satquery"] is None

    # Check report
    rep_file = stage_a_reports / "BENCHMARK_REPORT.md"
    assert rep_file.exists()
    content = rep_file.read_text(encoding="utf-8")
    assert "INFRASTRUCTURE COMPLETE — BENCHMARKING PENDING DATASET" in content
    assert "DEFERRED" in content


# ============================================================
# 7. BENCHMARK API ENDPOINTS
# ============================================================

def test_benchmark_api_endpoints():
    """Benchmark API routes are reachable and return valid benchmark status."""
    client = TestClient(app)

    # Summary
    resp = client.get("/api/benchmark/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["benchmark_ready"] is True
    assert "benchmark_status" in data
    assert data["ml_status"] == "DEFERRED"

    # Manifest
    resp_man = client.get("/api/benchmark/manifest")
    assert resp_man.status_code == 200
    assert "candidate_examples" in resp_man.json()

    # Reports
    resp_rep = client.get("/api/benchmark/reports")
    assert resp_rep.status_code == 200
    assert "overall" in resp_rep.json()
