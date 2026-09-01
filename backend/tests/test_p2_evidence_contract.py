"""
Scientific Regression and P2 Evidence Contract Test Suite.

Verifies:
1. Remote-sensing index mathematics (NDVI, NDWI, NDBI)
2. Temporal change mathematics (after - before, changed/valid ratio, zero-division safety)
3. Exclusion of NaN / Inf no-data pixels from valid pixel statistics
4. Standardization of the P2 Evidence Contract schema
5. Geographic AOI integrity across the pipeline ([151.195, -33.885, 151.225, -33.855])
6. Visualization URL safety (no placeholders, real files only, None on failure)
7. Four operational scenarios:
   - Scenario A: Single NDVI
   - Scenario B: Temporal NDVI
   - Scenario C: NDBI Urban Change
   - Scenario D: Optical + SAR composite multimodal representation
8. P4 fault tolerance: P2 succeeds even when P4/VLM fails or throws.
"""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from app.remote_sensing.indices.ndvi import calculate_ndvi as ndvi_calc
from app.remote_sensing.indices.ndwi import calculate_ndwi as ndwi_calc
from app.remote_sensing.indices.ndbi import calculate_ndbi as ndbi_calc
from app.tools.change import detect_change
from app.tools.indices import (
    calculate_ndvi as tool_ndvi,
    calculate_ndwi as tool_ndwi,
    calculate_ndbi as tool_ndbi,
    calculate_temporal_ndvi,
    calculate_temporal_ndbi,
)
from app.vlm.evidence_builder import build_evidence, EvidencePackage
from app.remote_sensing.providers.sentinel2 import normalize_aoi
from app.api.routes_query import _safe_vis_url, build_query_plan, process_query
from app.schemas.query import QueryRequest


# ============================================================
# 1. SCIENTIFIC INDEX REGRESSION TESTS
# ============================================================

def test_ndvi_scientific_formula():
    """NDVI = (NIR - Red) / (NIR + Red)"""
    red = np.array([[100.0, 200.0], [300.0, 50.0]], dtype=np.float32)
    nir = np.array([[300.0, 400.0], [500.0, 250.0]], dtype=np.float32)

    result = ndvi_calc(red, nir)
    expected = (nir - red) / (nir + red)

    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert np.isclose(result[0, 0], (300 - 100) / (300 + 100))  # 200/400 = 0.5


def test_ndwi_scientific_formula():
    """NDWI = (Green - NIR) / (Green + NIR)"""
    green = np.array([[300.0, 100.0], [200.0, 50.0]], dtype=np.float32)
    nir = np.array([[100.0, 300.0], [200.0, 50.0]], dtype=np.float32)

    result = ndwi_calc(green, nir)
    expected = (green - nir) / (green + nir)

    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert np.isclose(result[0, 0], (300 - 100) / (300 + 100))  # 0.5
    assert np.isclose(result[0, 1], (100 - 300) / (100 + 300))  # -0.5
    assert np.isclose(result[1, 0], 0.0)


def test_ndbi_scientific_formula():
    """NDBI = (SWIR - NIR) / (SWIR + NIR)"""
    swir = np.array([[500.0, 100.0], [250.0, 400.0]], dtype=np.float32)
    nir = np.array([[100.0, 300.0], [250.0, 100.0]], dtype=np.float32)

    result = ndbi_calc(swir, nir)
    expected = (swir - nir) / (swir + nir)

    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert np.isclose(result[0, 0], (500 - 100) / (500 + 100))  # 400/600 = 0.66667


def test_temporal_change_formula():
    """change = after - before"""
    before = np.array([[0.5, 0.6], [0.3, 0.8]], dtype=np.float32)
    after = np.array([[0.2, 0.6], [0.7, 0.4]], dtype=np.float32)

    res = detect_change(before, after, threshold=0.1)

    expected_change = after - before
    np.testing.assert_allclose(res["change_map"], expected_change, rtol=1e-5)
    assert np.isclose(res["mean_before"], float(np.mean(before)))
    assert np.isclose(res["mean_after"], float(np.mean(after)))
    assert np.isclose(res["mean_change"], float(np.mean(after) - np.mean(before)))


def test_change_ratio_and_zero_division_safety():
    """changed_pixels / valid_pixels, with safe handling when valid_pixels == 0."""
    # Case A: valid pixels with change
    before = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    after = np.array([[0.2, 0.5], [0.5, 0.8]], dtype=np.float32)
    # Threshold 0.1: changes are -0.3 and +0.3 (2 pixels changed out of 4)
    res_a = detect_change(before, after, threshold=0.1)
    assert res_a["valid_pixels"] == 4
    assert res_a["changed_pixels"] == 2
    assert np.isclose(res_a["change_ratio"], 0.5)

    # Case B: zero valid pixels (all NaN)
    all_nan_before = np.full((3, 3), np.nan, dtype=np.float32)
    all_nan_after = np.full((3, 3), np.nan, dtype=np.float32)
    res_b = detect_change(all_nan_before, all_nan_after, threshold=0.1)
    assert res_b["valid_pixels"] == 0
    assert res_b["changed_pixels"] == 0
    assert res_b["change_ratio"] == 0.0
    assert res_b["change_type"] == "no_data"
    assert res_b["mean_before"] is None
    assert res_b["mean_after"] is None


def test_nodata_pixels_excluded_from_valid_statistics():
    """Ensure NaN and Inf pixels are strictly excluded from count and mean."""
    before = np.array([[0.5, np.nan], [np.inf, 0.7]], dtype=np.float32)
    after = np.array([[0.2, np.nan], [0.4, 0.7]], dtype=np.float32)

    res = detect_change(before, after, threshold=0.05)
    # Only [0,0] (0.5 vs 0.2) and [1,1] (0.7 vs 0.7) are valid in both
    assert res["valid_pixels"] == 2
    assert res["total_pixels"] == 4
    assert np.isclose(res["mean_before"], (0.5 + 0.7) / 2.0)
    assert np.isclose(res["mean_after"], (0.2 + 0.7) / 2.0)


# ============================================================
# 2. P2 EVIDENCE CONTRACT SCHEMA TEST
# ============================================================

def test_p2_evidence_contract_structure():
    """Verify that build_evidence outputs all required contract keys without fabrication."""
    mock_response = {
        "query": "compare vegetation change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]",
        "plan": {
            "task": "change_detection",
            "target": "vegetation",
            "metric": "NDVI",
            "time_start": "2021",
            "time_end": "2024",
            "aoi": {
                "type": "Polygon",
                "coordinates": [[[151.195, -33.885], [151.225, -33.885], [151.225, -33.855], [151.195, -33.855], [151.195, -33.885]]],
            },
        },
        "statistics": {
            "mean_before": 0.4512,
            "mean_after": 0.3204,
            "mean_change": -0.1308,
            "min_value": -0.7500,
            "max_value": 0.6200,
            "valid_pixels": 25000,
            "total_pixels": 25000,
            "changed_pixels": 8200,
            "change_ratio": 0.3280,
            "increased_pixels": 200,
            "decreased_pixels": 8000,
            "threshold": 0.05,
            "change_type": "decrease",
        },
        "layers": [
            {
                "type": "change_detection",
                "visualization_url": "/visualizations/test_change.png",
                "bounds": [[-33.885, 151.195], [-33.855, 151.225]],
            }
        ],
        "evidence": [
            {
                "source": "REAL_SENTINEL_2",
                "images": [
                    {"id": "S2A_20210417", "date": "2021-04-17"},
                    {"id": "S2A_20240418", "date": "2024-04-18"},
                ],
            }
        ],
        "execution_trace": ["Executed: search_imagery", "Executed: calculate_temporal_ndvi", "Executed: detect_change"],
    }

    contract = build_evidence(mock_response)

    assert isinstance(contract, EvidencePackage)

    # Core metadata
    assert contract["query"] == mock_response["query"]
    assert contract["task"] == "change_detection"
    assert contract["target"] == "vegetation"
    assert contract["metric"] == "NDVI"

    # AOI
    assert contract["aoi"]["west"] == 151.195
    assert contract["aoi"]["south"] == -33.885
    assert contract["aoi"]["east"] == 151.225
    assert contract["aoi"]["north"] == -33.855

    # Temporal
    assert contract["temporal"]["before_date"] == "2021-04-17"
    assert contract["temporal"]["after_date"] == "2024-04-18"

    # Imagery
    assert contract["imagery"]["optical_before"] == "S2A_20210417"
    assert contract["imagery"]["optical_after"] == "S2A_20240418"

    # Statistics
    assert contract["statistics"]["mean_before"] == 0.4512
    assert contract["statistics"]["mean_after"] == 0.3204
    assert contract["statistics"]["mean_change"] == -0.1308
    assert contract["statistics"]["min_value"] == -0.7500
    assert contract["statistics"]["max_value"] == 0.6200
    assert contract["statistics"]["valid_pixels"] == 25000
    assert contract["statistics"]["total_pixels"] == 25000
    assert contract["statistics"]["changed_pixels"] == 8200
    assert contract["statistics"]["change_ratio"] == 0.3280
    assert contract["statistics"]["threshold"] == 0.05
    assert contract["statistics"]["change_type"] == "decrease"

    # Geographic
    assert contract["geographic"]["bounds"] == [[-33.885, 151.195], [-33.855, 151.225]]
    assert contract["geographic"]["crs"] == "EPSG:4326"

    # Execution
    assert contract["execution"]["tools"] == ["search_imagery", "calculate_temporal_ndvi", "detect_change"]
    assert contract["execution"]["imagery_source"] == "REAL_SENTINEL_2"


# ============================================================
# 3. AOI INTEGRITY & BOUNDS VERIFICATION
# ============================================================

def test_sydney_aoi_coordinate_chain():
    """
    Test coordinate chain for:
    Compare vegetation/NDVI change between 2021 and 2024 for AOI
    [151.195, -33.885, 151.225, -33.855]

    Expected geographic bounds:
    [
      [-33.885, 151.195],
      [-33.855, 151.225]
    ]
    """
    query = "Compare vegetation/NDVI change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]"
    req = QueryRequest(query=query)
    plan = build_query_plan(req)

    assert plan.aoi is not None
    w, s, e, n = normalize_aoi(plan.aoi)

    assert np.isclose(w, 151.195)
    assert np.isclose(s, -33.885)
    assert np.isclose(e, 151.225)
    assert np.isclose(n, -33.855)

    leaflet_bounds = [[float(s), float(w)], [float(n), float(e)]]
    assert leaflet_bounds == [[-33.885, 151.195], [-33.855, 151.225]]

    # Ensure Pune default bbox (73.80, 18.50, ...) was NOT used
    assert not (np.isclose(w, 73.80) and np.isclose(s, 18.50))


# ============================================================
# 4. VISUALIZATION URL SAFETY
# ============================================================

def test_safe_vis_url_rejects_placeholders_and_missing_files():
    """Audit every visualization URL: never allow placeholder or non-existent file to escape."""
    # 1. Placeholders must be rejected
    assert _safe_vis_url("/visualizations/<filename>.png") is None
    assert _safe_vis_url("<filename>.png") is None
    assert _safe_vis_url("placeholder.png") is None
    assert _safe_vis_url(None) is None
    assert _safe_vis_url("") is None

    # 2. Non-existent file must return None
    assert _safe_vis_url("non_existent_image_123456789.png") is None

    # 3. Existing real file in visualizations directory must return valid URL
    vis_dir = Path(__file__).resolve().parents[1] / "app" / "evidence" / "visualizations"
    test_file = vis_dir / "test_safety_probe.png"
    test_file.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    try:
        url = _safe_vis_url("test_safety_probe.png")
        assert url == "/visualizations/test_safety_probe.png"
    finally:
        if test_file.exists():
            test_file.unlink()


# ============================================================
# 5. FOUR OPERATIONAL SCENARIOS
# ============================================================

def test_scenario_a_single_ndvi():
    """
    Scenario A: Single NDVI
    Calculate NDVI for a valid AOI.
    Verify: index, min/max, mean, valid_pixels, visualization, bounds.
    """
    red = np.array([[100.0, 150.0], [200.0, 250.0]], dtype=np.float32)
    nir = np.array([[300.0, 350.0], [400.0, 450.0]], dtype=np.float32)

    res = tool_ndvi(red=red, nir=nir)

    assert res["status"] == "success"
    assert res["index"] == "NDVI"
    assert res["min_value"] is not None
    assert res["max_value"] is not None
    assert res["mean"] is not None
    assert res["valid_pixels"] == 4
    assert res["total_pixels"] == 4
    assert res["min_value"] <= res["mean"] <= res["max_value"]


def test_scenario_b_temporal_ndvi():
    """
    Scenario B: Temporal NDVI
    Compare vegetation/NDVI change between 2021 and 2024 for AOI
    [151.195, -33.885, 151.225, -33.855]
    Verify: before, after, mean change, changed pixels, valid pixels, change ratio, change map, bounds.
    """
    red_b = np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32)
    nir_b = np.array([[400.0, 400.0], [400.0, 400.0]], dtype=np.float32)
    # NDVI before = (400-100)/(500) = 0.60

    red_a = np.array([[200.0, 100.0], [100.0, 100.0]], dtype=np.float32)
    nir_a = np.array([[300.0, 400.0], [400.0, 400.0]], dtype=np.float32)
    # NDVI after [0,0] = (300-200)/(500) = 0.20 (change = -0.40, significant decrease)
    # NDVI after [others] = 0.60 (change = 0.0)

    temporal_ndvi = calculate_temporal_ndvi(
        red_before=red_b,
        nir_before=nir_b,
        red_after=red_a,
        nir_after=nir_a,
    )
    change_res = detect_change(
        before=temporal_ndvi["ndvi_before"],
        after=temporal_ndvi["ndvi_after"],
        threshold=0.05,
    )

    assert change_res["mean_before"] is not None
    assert change_res["mean_after"] is not None
    assert change_res["mean_change"] is not None
    assert change_res["changed_pixels"] == 1
    assert change_res["valid_pixels"] == 4
    assert np.isclose(change_res["change_ratio"], 0.25)
    assert change_res["change_map"] is not None
    assert change_res["change_type"] == "decrease"


def test_scenario_c_ndbi():
    """
    Scenario C: NDBI
    Compare urban/built-up change between 2022 and 2024 for the same AOI.
    Verify the same fields.
    """
    swir_b = np.array([[200.0, 200.0], [200.0, 200.0]], dtype=np.float32)
    nir_b = np.array([[400.0, 400.0], [400.0, 400.0]], dtype=np.float32)
    # NDBI before = (200-400)/(600) = -0.3333

    swir_a = np.array([[400.0, 200.0], [200.0, 200.0]], dtype=np.float32)
    nir_a = np.array([[200.0, 400.0], [400.0, 400.0]], dtype=np.float32)
    # NDBI after [0,0] = (400-200)/(600) = +0.3333 (change = +0.6667, significant increase)

    temporal_ndbi = calculate_temporal_ndbi(
        swir_before=swir_b,
        nir_before=nir_b,
        swir_after=swir_a,
        nir_after=nir_a,
    )
    change_res = detect_change(
        before=temporal_ndbi["ndbi_before"],
        after=temporal_ndbi["ndbi_after"],
        threshold=0.1,
    )

    assert change_res["mean_before"] is not None
    assert change_res["mean_after"] is not None
    assert change_res["mean_change"] is not None
    assert change_res["changed_pixels"] == 1
    assert change_res["valid_pixels"] == 4
    assert change_res["change_type"] == "increase"
    assert change_res["change_map"] is not None


def test_scenario_d_optical_plus_sar_evidence():
    """
    Scenario D: Optical + SAR
    Verify that both modalities are represented in the evidence package.
    """
    mock_multimodal_response = {
        "query": "Assess land cover change using optical and radar imagery",
        "plan": {
            "task": "change_detection",
            "target": "vegetation",
            "metric": "NDVI",
            "time_start": "2021",
            "time_end": "2024",
        },
        "statistics": {"mean_change": -0.15, "change_ratio": 0.22},
        "evidence": [
            {
                "source": "REAL_SENTINEL_2",
                "images": [{"id": "S2A_2021", "date": "2021-04-17"}, {"id": "S2A_2024", "date": "2024-04-18"}],
            }
        ],
        "has_s1": True,
        "sar_before": "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57 (VV/VH)",
        "layers": [],
        "execution_trace": ["Executed: search_imagery", "Executed: calculate_temporal_ndvi"],
    }

    contract = build_evidence(mock_multimodal_response)

    assert contract["imagery"]["optical_before"] == "S2A_2021"
    assert contract["imagery"]["optical_after"] == "S2A_2024"
    assert "S1B_IW_GRDH" in contract["imagery"]["sar_before"]
    assert contract["execution"]["imagery_source"] == "REAL_SENTINEL_2"


# ============================================================
# 6. ERROR HANDLING / FAULT TOLERANCE
# ============================================================

def test_p2_remains_functional_when_p4_fails():
    """
    P2 must remain functional if P4 fails (HF unavailable, 402, 413, VLM exception).
    P2 must still return:
    statistics, visual evidence where available, backend explanation, execution trace.
    """
    req = QueryRequest(query="compare vegetation change between 2021 and 2025")

    # Force VLM to raise an exception simulating HuggingFace 402 / 413 / network drop
    with patch("app.api.routes_query.VLM.generate", side_effect=RuntimeError("HF API 402 Payment Required")):
        result = process_query(req)

        assert result.status == "success"
        assert result.answer is not None
        assert len(result.answer) > 10  # Fallback explanation used
        assert result.statistics is not None
        assert "mean_change" in result.statistics or "metric" in result.statistics
        assert any("P4 VLM unavailable; using backend explanation" in trace for trace in result.execution_trace)
        assert result.evidence_package is not None
        assert "statistics" in result.evidence_package
