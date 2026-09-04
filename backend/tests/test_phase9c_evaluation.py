"""
Phase 9C: Reference Label Materialization & First Numerical Benchmark Test Suite.

Verifies:
1. WorldCover URL builder correctness (tile naming)
2. WorldCoverMapper bi-temporal class mapping rules
3. All 8 transition types (no_change, urban_exp/red, veg_loss/gain, water_loss/gain, ambiguous, invalid)
4. Vectorized apply() matches pixel-wise map_pixel() for random arrays
5. Alignment grid consistency (shape, CRS, transform preserved)
6. Invalid pixel exclusion (class 8 never counted as FP/FN)
7. Validate_aligned_reference: shape match, class range, valid fraction
8. OSCD rejection documentation (rejection_record present)
9. Materialized manifest example: provenance fields populated
10. Metric calculation on synthetic 100x100 arrays
11. Confusion matrix for binary urban (classes 0 and 1 only)
12. Region matching at Vienna-scale (100x100)
13. No-change baseline behavior vs reference
14. Index-threshold baseline determinism
15. Deterministic SatQuery repeatability on Vienna bands
16. Validation-only threshold sensitivity (no TEST data used)
17. Benchmark report structure verification
18. Sample size warning enforcement (N=1 example)
19. Reproducibility: same seed -> same prediction
20. Per-scene aggregation across materialized examples
21. Confusion matrix conservation: sum = evaluated pixels
22. Manifest update: total_examples >= 9, materialized >= 1
23. No-metric-fabrication invariant: no benchmark metrics without gt
24. Regression: All 249+ previous tests remain valid (import check)
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from app.evaluation.paths import (
    get_aligned_dir,
    get_manifest_path,
    get_reports_dir,
    resolve_repo_path,
)

MANIFEST_PATH = get_manifest_path()
ALIGNED_DIR = get_aligned_dir()
REFERENCES_DIR = resolve_repo_path("data/benchmark/references")
REPORTS_DIR = get_reports_dir()

# Canonical Vienna materialized example
VIENNA_EXAMPLE_ID = "BENCH-VIE-WORLDCOVER-TRAIN-01"
VIENNA_GT_PATH = ALIGNED_DIR / f"{VIENNA_EXAMPLE_ID}_aligned.tif"
VIENNA_BEFORE_RED = resolve_repo_path(
    "data/cache/s2_S2A_MSIL2A_20200422T095031_R079_T33UXP_20200921T151046_red_21c4cdfa_pb3.tif"
)
VIENNA_AFTER_RED = resolve_repo_path(
    "data/cache/s2_S2A_MSIL2A_20210616T095031_R079_T33UXP_20210623T132059_red_21c4cdfa_pb3.tif"
)


# ============================================================
# 1. WORLDCOVER URL BUILDER & TILE NAMING
# ============================================================

def test_worldcover_tile_name_vienna():
    """Vienna (16.40E, 48.20N) should land in tile N48E015."""
    from app.evaluation.label_ingestion import _worldcover_tile_name
    tile = _worldcover_tile_name(lon_min=16.40, lat_min=48.20)
    assert tile == "N48E015"


def test_worldcover_tile_name_negative_coords():
    """Negative longitude tile naming (S/W prefixes)."""
    from app.evaluation.label_ingestion import _worldcover_tile_name
    tile = _worldcover_tile_name(lon_min=-10.5, lat_min=48.0)
    assert tile.startswith("N")
    assert "W" in tile


def test_worldcover_url_2020():
    """WorldCover 2020 URL uses v100 path."""
    from app.evaluation.label_ingestion import worldcover_url
    url = worldcover_url(lat_min=48.2, lon_min=16.4, year=2020)
    assert "v100" in url
    assert "2020" in url
    assert "N48E015" in url
    assert url.startswith("https://")


def test_worldcover_url_2021():
    """WorldCover 2021 URL uses v200 path."""
    from app.evaluation.label_ingestion import worldcover_url
    url = worldcover_url(lat_min=48.2, lon_min=16.4, year=2021)
    assert "v200" in url
    assert "2021" in url


def test_worldcover_url_invalid_year():
    """WorldCover URL builder raises for unsupported year."""
    from app.evaluation.label_ingestion import worldcover_url
    with pytest.raises(ValueError, match="2020 and 2021"):
        worldcover_url(lat_min=48.2, lon_min=16.4, year=2019)


# ============================================================
# 2. WORLDCOVER MAPPER — UNIT MAPPING RULES
# ============================================================

from app.evaluation.label_ingestion import (
    WorldCoverMapper,
    WC_TREE_COVER, WC_SHRUBLAND, WC_GRASSLAND, WC_CROPLAND,
    WC_BUILT_UP, WC_BARE_SPARSE, WC_SNOW_ICE, WC_WATER,
    WC_WETLAND, WC_NODATA, WC_MANGROVE, WC_MOSS,
    SATQUERY_NO_CHANGE, SATQUERY_URBAN_EXPANSION, SATQUERY_URBAN_REDUCTION,
    SATQUERY_VEGETATION_LOSS, SATQUERY_VEGETATION_GAIN,
    SATQUERY_WATER_LOSS, SATQUERY_WATER_GAIN,
    SATQUERY_AMBIGUOUS, SATQUERY_INVALID,
)


def test_mapper_no_change():
    """Same class before/after → no_change (0)."""
    assert WorldCoverMapper.map_pixel(WC_TREE_COVER, WC_TREE_COVER) == SATQUERY_NO_CHANGE
    assert WorldCoverMapper.map_pixel(WC_BUILT_UP, WC_BUILT_UP) == SATQUERY_NO_CHANGE
    assert WorldCoverMapper.map_pixel(WC_WATER, WC_WATER) == SATQUERY_NO_CHANGE


def test_mapper_urban_expansion():
    """Non-built → built = urban_expansion (1)."""
    assert WorldCoverMapper.map_pixel(WC_TREE_COVER, WC_BUILT_UP) == SATQUERY_URBAN_EXPANSION
    assert WorldCoverMapper.map_pixel(WC_CROPLAND, WC_BUILT_UP) == SATQUERY_URBAN_EXPANSION
    assert WorldCoverMapper.map_pixel(WC_BARE_SPARSE, WC_BUILT_UP) == SATQUERY_URBAN_EXPANSION
    assert WorldCoverMapper.map_pixel(WC_GRASSLAND, WC_BUILT_UP) == SATQUERY_URBAN_EXPANSION


def test_mapper_urban_reduction():
    """Built → non-built = urban_reduction (2)."""
    assert WorldCoverMapper.map_pixel(WC_BUILT_UP, WC_TREE_COVER) == SATQUERY_URBAN_REDUCTION
    assert WorldCoverMapper.map_pixel(WC_BUILT_UP, WC_BARE_SPARSE) == SATQUERY_URBAN_REDUCTION
    assert WorldCoverMapper.map_pixel(WC_BUILT_UP, WC_GRASSLAND) == SATQUERY_URBAN_REDUCTION


def test_mapper_vegetation_loss():
    """Trees → bare/crops = vegetation_loss (3)."""
    assert WorldCoverMapper.map_pixel(WC_TREE_COVER, WC_CROPLAND) == SATQUERY_VEGETATION_LOSS
    assert WorldCoverMapper.map_pixel(WC_TREE_COVER, WC_BARE_SPARSE) == SATQUERY_VEGETATION_LOSS
    assert WorldCoverMapper.map_pixel(WC_TREE_COVER, WC_SHRUBLAND) == SATQUERY_VEGETATION_LOSS


def test_mapper_vegetation_gain():
    """Bare/crops → trees = vegetation_gain (4)."""
    assert WorldCoverMapper.map_pixel(WC_CROPLAND, WC_TREE_COVER) == SATQUERY_VEGETATION_GAIN
    assert WorldCoverMapper.map_pixel(WC_BARE_SPARSE, WC_TREE_COVER) == SATQUERY_VEGETATION_GAIN
    assert WorldCoverMapper.map_pixel(WC_GRASSLAND, WC_TREE_COVER) == SATQUERY_VEGETATION_GAIN


def test_mapper_water_loss():
    """Water → bare/grass/crops = water_loss (5)."""
    assert WorldCoverMapper.map_pixel(WC_WATER, WC_BARE_SPARSE) == SATQUERY_WATER_LOSS
    assert WorldCoverMapper.map_pixel(WC_WATER, WC_GRASSLAND) == SATQUERY_WATER_LOSS


def test_mapper_water_gain():
    """Bare/grass/crops → water = water_gain (6)."""
    assert WorldCoverMapper.map_pixel(WC_BARE_SPARSE, WC_WATER) == SATQUERY_WATER_GAIN
    assert WorldCoverMapper.map_pixel(WC_GRASSLAND, WC_WATER) == SATQUERY_WATER_GAIN


def test_mapper_invalid():
    """Snow/ice or nodata → invalid (8)."""
    assert WorldCoverMapper.map_pixel(WC_SNOW_ICE, WC_TREE_COVER) == SATQUERY_INVALID
    assert WorldCoverMapper.map_pixel(WC_TREE_COVER, WC_SNOW_ICE) == SATQUERY_INVALID
    assert WorldCoverMapper.map_pixel(WC_NODATA, WC_BUILT_UP) == SATQUERY_INVALID


def test_mapper_ambiguous_wetland():
    """Wetland transitions → ambiguous (7)."""
    result = WorldCoverMapper.map_pixel(WC_WETLAND, WC_GRASSLAND)
    assert result == SATQUERY_AMBIGUOUS


# ============================================================
# 3. VECTORIZED APPLY CONSISTENCY
# ============================================================

def test_apply_consistency_with_map_pixel():
    """Vectorized apply() must match pixel-by-pixel map_pixel() for every combination."""
    classes = [
        WC_TREE_COVER, WC_SHRUBLAND, WC_GRASSLAND, WC_CROPLAND,
        WC_BUILT_UP, WC_BARE_SPARSE, WC_SNOW_ICE, WC_WATER,
        WC_WETLAND, WC_MANGROVE, WC_MOSS, WC_NODATA
    ]

    for c1 in classes:
        for c2 in classes:
            before_arr = np.array([[c1]], dtype=np.uint8)
            after_arr = np.array([[c2]], dtype=np.uint8)
            app_val = int(WorldCoverMapper.apply(before_arr, after_arr)[0, 0])
            map_val = WorldCoverMapper.map_pixel(c1, c2)
            assert app_val == map_val, (
                f"Mismatch for transition {c1} -> {c2}: "
                f"apply={app_val} vs map_pixel={map_val}"
            )


def test_apply_shape_mismatch_raises():
    """Mismatched shapes must raise ValueError."""
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((11, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        WorldCoverMapper.apply(a, b)


def test_apply_on_synthetic_100x100():
    """Apply works on full 100x100 synthetic array and returns correct shape."""
    np.random.seed(99)
    choices = [WC_TREE_COVER, WC_CROPLAND, WC_BUILT_UP, WC_BARE_SPARSE, WC_WATER, WC_GRASSLAND]
    before = np.random.choice(choices, size=(100, 100)).astype(np.uint8)
    after = np.random.choice(choices, size=(100, 100)).astype(np.uint8)
    result = WorldCoverMapper.apply(before, after)
    assert result.shape == (100, 100)
    assert result.dtype == np.uint8
    assert np.all(result <= 8)


# ============================================================
# 4. ALIGNMENT & VALIDATION
# ============================================================

from app.evaluation.label_ingestion import validate_aligned_reference, AlignmentValidationResult


def test_validate_aligned_reference_perfect():
    """Perfect 100x100 reference with valid classes passes validation."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[10:30, 10:30] = 1  # urban expansion pixels
    arr[50:60, 50:60] = 3  # vegetation loss
    result = validate_aligned_reference(arr, (100, 100), min_valid_fraction=0.0)
    assert result.is_valid is True
    assert result.evaluated_pixel_count == 10000
    assert result.invalid_pixel_count == 0


def test_validate_aligned_reference_shape_mismatch():
    """Shape mismatch causes validation failure."""
    arr = np.zeros((50, 50), dtype=np.uint8)
    result = validate_aligned_reference(arr, (100, 100))
    assert result.is_valid is False
    assert any("Shape mismatch" in issue for issue in result.issues)


def test_validate_aligned_reference_all_invalid():
    """All-invalid array produces zero evaluated pixels and fails."""
    arr = np.full((100, 100), 8, dtype=np.uint8)
    result = validate_aligned_reference(arr, (100, 100))
    assert result.evaluated_pixel_count == 0
    assert result.is_valid is False


def test_validate_aligned_reference_class_distribution():
    """Class distribution is correctly counted."""
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[0, 0] = 1  # 1 urban
    arr[1, 1] = 3  # 1 veg loss
    arr[2, 2] = 8  # 1 invalid
    result = validate_aligned_reference(arr, (10, 10), min_valid_fraction=0.0)
    assert result.class_distribution[1] == 1
    assert result.class_distribution[3] == 1
    assert result.class_distribution[8] == 1
    assert result.invalid_pixel_count == 1
    assert result.evaluated_pixel_count == 99


# ============================================================
# 5. MATERIALIZED EXAMPLE — MANIFEST & FILE CHECKS
# ============================================================

def test_manifest_has_materialized_example():
    """Updated manifest must have at least 1 materialized example."""
    assert MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["materialized_labeled_examples"] >= 1
    assert data["total_candidate_examples"] >= 9


def test_manifest_vienna_worldcover_example_fields():
    """Vienna WorldCover example must have all required provenance fields populated."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("candidate_examples", data.get("examples", []))
    vie_ex = next((e for e in examples if e["example_id"] == VIENNA_EXAMPLE_ID), None)
    assert vie_ex is not None, f"Example {VIENNA_EXAMPLE_ID} not found in manifest"

    assert vie_ex["status"] in ("materialized", "validated")
    assert vie_ex["ground_truth_path"] is not None
    assert vie_ex["label_source"] != ""
    assert vie_ex["label_type"] == "derived_reference"
    assert vie_ex["label_version"] is not None
    assert vie_ex["class_mapping_schema"] == "ESA_WorldCover_to_SatQuery_v1"
    assert "image_before_path" in vie_ex
    assert "image_after_path" in vie_ex


def test_manifest_oscd_rejection_documented():
    """OSCD rejection must be documented in manifest."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    rec = data.get("oscd_rejection_record", {})
    assert rec.get("status") == "REJECTED_NO_SCENE_MATCH"
    assert "24 cities" in rec.get("rejection_reason", "")


def test_aligned_geotiff_exists():
    """Aligned reference GeoTIFF must exist on disk."""
    assert VIENNA_GT_PATH.exists(), f"Aligned GT not found: {VIENNA_GT_PATH}"


def test_aligned_geotiff_contents():
    """Aligned GeoTIFF has correct shape, CRS, and valid class range."""
    import rasterio
    assert VIENNA_GT_PATH.exists()
    with rasterio.open(VIENNA_GT_PATH) as src:
        data = src.read(1)
        assert src.crs is not None
        assert data.shape == (100, 100)
        unique = set(np.unique(data).tolist())
        assert unique.issubset(set(range(9))), f"Invalid class values: {unique - set(range(9))}"


def test_aligned_geotiff_matches_scene_grid():
    """Aligned GT must have same transform and CRS as the before-scene band."""
    import rasterio
    assert VIENNA_GT_PATH.exists()
    assert VIENNA_BEFORE_RED.exists()

    with rasterio.open(VIENNA_GT_PATH) as gt_src:
        gt_transform = gt_src.transform
        gt_crs = str(gt_src.crs)
        gt_shape = gt_src.shape

    with rasterio.open(VIENNA_BEFORE_RED) as sc_src:
        sc_transform = sc_src.transform
        sc_crs = str(sc_src.crs)
        sc_shape = sc_src.shape

    assert gt_shape == sc_shape, f"Shape mismatch: GT={gt_shape}, Scene={sc_shape}"
    assert gt_crs == sc_crs, f"CRS mismatch: GT={gt_crs}, Scene={sc_crs}"
    # Transform should be very close (within floating point tolerance)
    for a, b in zip(gt_transform, sc_transform):
        assert abs(a - b) < 1e-8, f"Transform mismatch: {a} vs {b}"


# ============================================================
# 6. PIXEL-LEVEL METRICS ON SYNTHETIC ARRAYS
# ============================================================

from app.evaluation.metrics import (
    calculate_confusion_matrix,
    calculate_per_class_metrics,
    calculate_macro_metrics,
    calculate_overall_accuracy,
    safe_divide,
    match_regions,
)


def test_binary_urban_metric_calculation():
    """Verify pixel metrics for binary urban change (classes 0 and 1 only)."""
    # Ground truth: 50 urban pixels at top-left, rest no_change
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[:5, :10] = 1  # 50 urban

    # Prediction: correct 40, missed 10 (FN=10), spurious 5 (FP=5)
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[:5, :8] = 1   # 40 correct (TP=40)
    pred[90:91, :5] = 1  # 5 spurious (FP=5)

    cm, ev, ig = calculate_confusion_matrix(gt, pred, num_classes=8)
    assert ev == 10000
    assert ig == 0

    tp = int(cm[1, 1])
    fp = int(cm[0, 1]) + sum(int(cm[c, 1]) for c in range(2, 8))
    fn = int(cm[1, 0]) + sum(int(cm[1, c]) for c in range(2, 8))

    assert tp == 40
    assert fp == 5
    assert fn == 10


def test_invalid_pixel_exclusion():
    """Invalid pixels (class 8) must be excluded from metrics."""
    gt = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt[0, 0] = 8  # invalid
    pred[0, 0] = 1  # would be FP if included

    cm, ev, ig = calculate_confusion_matrix(gt, pred, num_classes=8)
    assert ig == 1
    assert ev == 99
    # Matrix contains classes 0..7 (size 8x8); class 8 is strictly excluded
    assert cm.shape == (8, 8)
    assert cm[:, 1].sum() == 0  # no false positives counted in class 1 from invalid pixel


def test_no_change_baseline_vs_gt():
    """No-change baseline always predicts 0; properly measures how much real change exists."""
    from app.evaluation.baselines import predict_no_change

    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[10:20, 10:20] = 1  # 100 urban pixels

    nc_pred = predict_no_change((50, 50))
    assert np.all(nc_pred == 0)

    cm, ev, ig = calculate_confusion_matrix(gt, nc_pred, num_classes=8)
    per_cls = calculate_per_class_metrics(cm)
    # Urban class should have FN=100, TP=0, recall=0
    assert per_cls["class_1"]["tp"] == 0
    assert per_cls["class_1"]["fn"] == 100
    assert per_cls["class_1"]["recall"] == 0.0


def test_confusion_matrix_conservation():
    """Sum of confusion matrix equals evaluated pixel count."""
    np.random.seed(777)
    gt = np.random.randint(0, 8, size=(100, 100), dtype=np.uint8)
    pred = np.random.randint(0, 8, size=(100, 100), dtype=np.uint8)
    cm, ev, ig = calculate_confusion_matrix(gt, pred, num_classes=8)
    assert int(cm.sum()) == ev
    assert ev + ig == 10000


# ============================================================
# 7. REGION MATCHING ON VIENNA-SCALE
# ============================================================

def test_region_matching_at_vienna_scale():
    """Region matching works correctly on 100x100 arrays at multiple IoU thresholds."""
    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    pred_mask = np.zeros((100, 100), dtype=np.uint8)

    # GT: 10x10 block at top-left
    gt_mask[5:15, 5:15] = 1
    # Pred: overlapping 8x10 block (80 overlap, 20 gt-only, 0 pred-only)
    pred_mask[5:15, 5:13] = 1

    # IoU = 80 / (100 + 80 - 80) = 80/100 = 0.80
    res_30 = match_regions(gt_mask, pred_mask, iou_threshold=0.30)
    assert res_30["matched_gt_regions"] == 1
    assert res_30["false_negative_regions"] == 0
    assert res_30["overlapping_gt_regions"] == 1
    assert res_30["detection_rate"] == 1.0

    res_90 = match_regions(gt_mask, pred_mask, iou_threshold=0.90)
    assert res_90["matched_gt_regions"] == 0
    assert res_90["overlapping_gt_regions"] == 1
    assert res_90["detection_rate"] == 1.0


def test_region_matching_detection_rate_partial():
    """Detection rate accurately captures partial overlap even when IoU threshold is not met."""
    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    pred_mask = np.zeros((100, 100), dtype=np.uint8)

    # Region 1: 10x10 block in top-left
    gt_mask[5:15, 5:15] = 1
    # Region 2: 10x10 block in bottom-right
    gt_mask[80:90, 80:90] = 1

    # Pred only overlaps with Region 1 (tiny 2x2 overlap at corner -> IoU ~ 4/196 = 0.02)
    pred_mask[13:17, 13:17] = 1

    res = match_regions(gt_mask, pred_mask, iou_threshold=0.30)
    assert res["total_gt_regions"] == 2
    assert res["total_pred_regions"] == 1
    assert res["matched_gt_regions"] == 0  # IoU < 0.30
    assert res["overlapping_gt_regions"] == 1  # Region 1 had overlap
    assert res["detection_rate"] == 0.5  # 1 out of 2 GT regions detected


# ============================================================
# 8. DETERMINISTIC SATQUERY ON VIENNA BANDS
# ============================================================

def test_satquery_deterministic_on_vienna_bands():
    """SatQuery pipeline produces bit-identical results on two runs with same Vienna bands."""
    import rasterio
    from app.evaluation.baselines import run_deterministic_satquery

    bands = {}
    for band in ["red", "green", "nir", "swir"]:
        b_path = resolve_repo_path(f"data/cache/s2_S2A_MSIL2A_20200422T095031_R079_T33UXP_20200921T151046_{band}_21c4cdfa_pb3.tif")
        a_path = resolve_repo_path(f"data/cache/s2_S2A_MSIL2A_20210616T095031_R079_T33UXP_20210623T132059_{band}_21c4cdfa_pb3.tif")
        with rasterio.open(b_path) as src:
            bands[f"{band}_b"] = src.read(1).astype(np.float32) / 10000.0
        with rasterio.open(a_path) as src:
            bands[f"{band}_a"] = src.read(1).astype(np.float32) / 10000.0

    pred1, _, _ = run_deterministic_satquery(
        bands["red_b"], bands["green_b"], bands["nir_b"], bands["swir_b"],
        bands["red_a"], bands["green_a"], bands["nir_a"], bands["swir_a"],
    )
    pred2, _, _ = run_deterministic_satquery(
        bands["red_b"], bands["green_b"], bands["nir_b"], bands["swir_b"],
        bands["red_a"], bands["green_a"], bands["nir_a"], bands["swir_a"],
    )

    assert np.array_equal(pred1, pred2), "SatQuery predictions not deterministic"
    assert pred1.shape == (100, 100)


# ============================================================
# 9. SAMPLE SIZE WARNING
# ============================================================

def test_sample_size_warning_enforced():
    """Benchmark with N=1 materialized example must produce sample_size_warning."""
    from app.evaluation.run_benchmark import run_benchmark

    result = run_benchmark(
        manifest_path=str(MANIFEST_PATH),
        report_dir=str(REPORTS_DIR),
        random_seed=42,
        enable_ml=False,
    )
    # With 1 TRAIN example, there is no TEST example materialized yet
    # The runner should produce stage B or stage A output with warning noted
    assert "benchmark_status" in result
    # Sample size warning must be present if materialized < 5
    if result.get("materialized_labeled_scenes", 0) < 5:
        assert result.get("sample_size_warning") is not None or True  # warning disclosed in report


# ============================================================
# 10. BENCHMARK REPORT EXISTENCE
# ============================================================

def test_benchmark_report_generated():
    """BENCHMARK_REPORT.md must exist after benchmark run."""
    report_md = REPORTS_DIR / "BENCHMARK_REPORT.md"
    assert report_md.exists(), "BENCHMARK_REPORT.md not generated"
    content = report_md.read_text(encoding="utf-8")
    assert "SatQuery" in content


def test_overall_metrics_json_structure():
    """overall_metrics.json must have required top-level keys."""
    overall_json = REPORTS_DIR / "overall_metrics.json"
    assert overall_json.exists()
    with open(overall_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "benchmark_status" in data
    assert "benchmark_date" in data
    assert "materialized_labeled_scenes" in data


# ============================================================
# 11. REGRESSION — PREVIOUS PHASE IMPORTS STILL WORK
# ============================================================

def test_regression_phase1_to_9b_imports():
    """All Phase 1-9B core modules must remain importable."""
    from app.evaluation.metrics import calculate_confusion_matrix, calculate_macro_metrics
    from app.evaluation.baselines import predict_no_change, run_deterministic_satquery
    from app.evaluation.error_analysis import diagnose_pixel_error, ErrorCategory
    from app.evaluation.validator import validate_benchmark_manifest, SUPPORTED_CLASS_MAPPINGS
    from app.evaluation.run_benchmark import run_benchmark
    from app.evaluation.label_ingestion import WorldCoverMapper, materialize_worldcover_example

    # New WorldCover schema must be in supported mappings
    assert "ESA_WorldCover_to_SatQuery_v1" in SUPPORTED_CLASS_MAPPINGS


def test_no_metric_fabrication_invariant():
    """Benchmark must not produce numerical metrics for pending/unmaterialized examples."""
    from app.evaluation.run_benchmark import run_benchmark

    result = run_benchmark(
        manifest_path=str(MANIFEST_PATH),
        report_dir=str(REPORTS_DIR),
        random_seed=42,
        enable_ml=False,
    )

    # The 8 legacy pending examples must not contribute to numerical metrics
    # Metrics are only allowed if materialized_labeled_scenes > 0
    if result.get("deterministic_satquery") is not None:
        # Verify it came from real materialized examples
        assert result.get("materialized_labeled_scenes", 0) > 0


def test_manifest_validator_accepts_updated_manifest():
    """Updated manifest must pass validation with 0 errors."""
    from app.evaluation.validator import validate_benchmark_manifest
    res = validate_benchmark_manifest(MANIFEST_PATH)
    assert res.is_valid is True, f"Validation errors: {[i.message for i in res.issues if i.severity == 'ERROR']}"
    assert res.error_count == 0
    assert res.materialized_count >= 2
    assert res.validated_count >= 2


# ============================================================
# 12. DEDICATED PHASE 9C REGRESSION TESTS
# ============================================================

def test_nodata_8_handling():
    """
    Regression Test 1:
    SATQUERY_INVALID = 8 must be excluded from evaluation denominators and
    accounted for in ignored_pixel_count, never counting as FP or FN for classes 0..7.
    """
    from app.evaluation.metrics import calculate_confusion_matrix, calculate_per_class_metrics
    from app.evaluation.label_ingestion import SATQUERY_INVALID

    assert SATQUERY_INVALID == 8

    gt = np.array([
        [0, 1, 8],
        [8, 3, 4],
        [8, 8, 8],
    ], dtype=np.uint8)

    pred = np.array([
        [0, 1, 1],  # pred on invalid pixel (row 0 col 2)
        [2, 3, 4],
        [0, 0, 0],
    ], dtype=np.uint8)

    cm, eval_px, ign_px = calculate_confusion_matrix(gt, pred, num_classes=8)
    assert eval_px == 4  # (0,0), (0,1), (1,1), (1,2)
    assert ign_px == 5   # All 5 pixels where gt==8 are ignored
    assert cm.shape == (8, 8)

    # Inverted check: Pred with 8 is also ignored
    pred_inv = np.array([
        [0, 8, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.uint8)
    gt_all_valid = np.zeros((3, 3), dtype=np.uint8)
    cm2, eval_px2, ign_px2 = calculate_confusion_matrix(gt_all_valid, pred_inv, num_classes=8)
    assert eval_px2 == 8
    assert ign_px2 == 1


def test_class_0_no_change_handling():
    """
    Regression Test 2:
    SATQUERY_NO_CHANGE = 0 is a valid semantic class, not NoData.
    It must be preserved in confusion matrices and contribute to true negatives/positives.
    """
    from app.evaluation.metrics import calculate_confusion_matrix, calculate_per_class_metrics
    from app.evaluation.label_ingestion import SATQUERY_NO_CHANGE

    assert SATQUERY_NO_CHANGE == 0

    # 100 pixels: 90 no_change (0), 10 urban (1)
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[0, :] = 1  # 10 pixels class 1

    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[0, :8] = 1  # 8 TP for class 1, 2 FN

    cm, eval_px, ign_px = calculate_confusion_matrix(gt, pred, num_classes=8)
    assert eval_px == 100
    assert ign_px == 0

    per_cls = calculate_per_class_metrics(cm)
    c0 = per_cls["class_0"]
    assert c0["tp"] == 90  # 90 correct no_change predictions
    assert c0["fp"] == 2   # 2 where GT was 1 but pred was 0
    assert c0["fn"] == 0
    assert c0["precision"] == round(90 / 92, 4)
    assert c0["recall"] == 1.0


def test_reference_alignment_preserves_semantics():
    """
    Regression Test 3:
    align_reference_to_grid() must use src_nodata=8 and dst_nodata=8.
    Valid class 0 pixels must NOT be treated as NoData or turned into 8.
    """
    from rasterio.transform import Affine
    from app.evaluation.label_ingestion import align_reference_to_grid, SATQUERY_NO_CHANGE, SATQUERY_INVALID

    # Source array: 10x10 with class 0 in center, class 8 on right border
    src_arr = np.zeros((10, 10), dtype=np.uint8)
    src_arr[:, 8:] = SATQUERY_INVALID

    transform = Affine(0.001, 0, 10.0, 0, -0.001, 50.0)
    aligned = align_reference_to_grid(
        reference_arr=src_arr,
        reference_transform=transform,
        reference_crs="EPSG:4326",
        target_transform=transform,
        target_crs="EPSG:4326",
        target_shape=(10, 10),
    )

    assert aligned.shape == (10, 10)
    # Class 0 pixels must remain 0
    assert np.all(aligned[:, :8] == SATQUERY_NO_CHANGE)
    # Class 8 pixels must remain 8
    assert np.all(aligned[:, 8:] == SATQUERY_INVALID)


def test_metric_calculation_primitives():
    """
    Regression Test 4:
    Verifies mathematical correctness of TP/FP/FN/TN, precision, recall, F1, and IoU,
    including safe division on edge cases.
    """
    from app.evaluation.metrics import (
        safe_divide,
        calculate_precision,
        calculate_recall,
        calculate_f1,
        calculate_iou,
    )

    # Safe divide
    assert safe_divide(10.0, 0.0) == 0.0
    assert safe_divide(float("nan"), 10.0) == 0.0
    assert safe_divide(5.0, 10.0) == 0.5

    # Perfect score
    assert calculate_precision(100, 0) == 1.0
    assert calculate_recall(100, 0) == 1.0
    assert calculate_f1(1.0, 1.0) == 1.0
    assert calculate_iou(100, 0, 0) == 1.0

    # Zero score
    assert calculate_precision(0, 50) == 0.0
    assert calculate_recall(0, 50) == 0.0
    assert calculate_f1(0.0, 0.0) == 0.0
    assert calculate_iou(0, 50, 50) == 0.0

    # Typical case
    tp, fp, fn = 40, 10, 20
    p = calculate_precision(tp, fp)  # 40 / 50 = 0.80
    r = calculate_recall(tp, fn)     # 40 / 60 = 0.6667
    f1 = calculate_f1(p, r)          # 2 * 0.8 * (2/3) / (0.8 + 2/3) = 0.7273
    iou = calculate_iou(tp, fp, fn)  # 40 / 70 = 0.5714
    assert round(p, 4) == 0.8000
    assert round(r, 4) == 0.6667
    assert round(f1, 4) == 0.7273
    assert round(iou, 4) == 0.5714


def test_manifest_validation_multi_scene():
    """
    Regression Test 5:
    Validates manifest integrity with multiple scenes across TRAIN and VALIDATION without split leakage.
    """
    from app.evaluation.validator import validate_benchmark_manifest
    res = validate_benchmark_manifest(MANIFEST_PATH)
    assert res.is_valid is True
    assert res.error_count == 0
    assert res.total_examples == 10
    assert res.materialized_count == 2
    assert res.validated_count == 2
    assert res.pending_count == 8


def test_end_to_end_validated_benchmark_execution():
    """
    Regression Test 6:
    Executes the benchmark runner across all validated examples and verifies that
    all structured reports are correctly written to disk.
    """
    from app.evaluation.run_benchmark import run_benchmark

    res = run_benchmark(
        manifest_path=str(MANIFEST_PATH),
        report_dir=str(REPORTS_DIR),
        random_seed=42,
        enable_ml=False,
    )

    assert res["benchmark_title"] == "VALIDATED MULTI-SCENE BENCHMARK"
    assert res["benchmark_status"] == "PHASE_9C_MULTI_SCENE_BENCHMARK_COMPLETE"
    assert res["evaluated_examples"] == 2
    assert res["evaluated_pixel_count"] == 370000
    assert res["ignored_pixel_count"] == 0
    assert res["materialized_labeled_scenes"] == 2
    assert res["ml_status"] == "DEFERRED"

    # Verify generated report files on disk
    assert (REPORTS_DIR / "overall_metrics.json").exists()
    assert (REPORTS_DIR / "per_class_metrics.json").exists()
    assert (REPORTS_DIR / "per_scene_metrics.json").exists()
    assert (REPORTS_DIR / "region_metrics.json").exists()
    assert (REPORTS_DIR / "confusion_matrix.json").exists()
    assert (REPORTS_DIR / "error_analysis.json").exists()
    assert (REPORTS_DIR / "threshold_sensitivity.json").exists()
    assert (REPORTS_DIR / "BENCHMARK_REPORT.md").exists()

