"""
Test Suite for Step 14: Automatic Optical-SAR Acquisition Integration into Agent + API.

Covers all 15 test specifications:
1. Test 1: Automatic mode without explicit image references invokes find_optical_sar_pair.
2. Test 2: Automatic pairing result paths reach run_optical_sar_analysis.
3. Test 3: Explicit optical + SAR inputs bypass automatic acquisition.
4. Test 4: Missing AOI fails cleanly (HTTP 400 "Optical-SAR automatic acquisition requires an AOI.").
5. Test 5: Missing date/time fails cleanly (HTTP 400 "Optical-SAR automatic acquisition requires a target date or time range.").
6. Test 6: No compatible pair fails cleanly (HTTP 404).
7. Test 7: Pair metadata survives into API response (metadata, statistics, evidence).
8. Test 8: Execution trace contains acquisition/pairing steps.
9. Test 9: Normal NDVI does not invoke Sentinel-1 acquisition.
10. Test 10: Normal change detection does not invoke Sentinel-1 acquisition.
11. Test 11: Existing upload/reference Optical-SAR test still passes.
12. Test 12: VLM failure returns controlled fallback/error behavior.
13. Test 13: Automatic pair result is compatible with validate_optical_sar_pair.
14. Test 14: Automatic pair result is compatible with align_optical_sar_pair.
15. Test 15: No BigEarthNet fallback occurs.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from fastapi.testclient import TestClient

from app.main import app
from app.remote_sensing.multimodal.pairing import PairingErrorType
from app.remote_sensing.multimodal.optical_sar import validate_optical_sar_pair, align_optical_sar_pair


# ============================================================
# FIXTURES & HELPERS
# ============================================================

def _create_mock_geotiff(path: Path, count: int = 1, width: int = 20, height: int = 20, value: int = 100):
    transform = from_bounds(13.0, 48.0, 13.02, 48.02, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype="uint16",
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=0,
    ) as dst:
        for b in range(1, count + 1):
            data = np.full((height, width), value * b, dtype=np.uint16)
            dst.write(data, b)
    return str(path.resolve())


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def mock_pair_paths(tmp_path):
    opt_path = _create_mock_geotiff(tmp_path / "mock_s2.tif", count=3, value=150)
    sar_vv = _create_mock_geotiff(tmp_path / "mock_s1_vv.tif", count=1, value=250)
    sar_vh = _create_mock_geotiff(tmp_path / "mock_s1_vh.tif", count=1, value=50)
    return {
        "status": "REAL_SUCCESS",
        "pair_found": True,
        "temporal_delta_days": 0.28,
        "spatial_overlap": {
            "has_overlap": True,
            "intersection_bbox": [13.0, 48.0, 13.02, 48.02],
            "aoi_coverage_percent": 100.0,
        },
        "selection_reason": "Selected Optical S2A_mock and SAR S1B_mock with score 0.95",
        "optical": {
            "item_id": "S2A_MSIL2A_mock",
            "acquisition_datetime": "2021-06-27T10:00:00Z",
            "sensor": "Sentinel-2A",
            "product": "Level-2A",
            "cloud_cover": 1.2,
            "coverage_percentage": 100.0,
            "crs": "EPSG:4326",
            "bounds": [13.0, 48.0, 13.02, 48.02],
            "path": opt_path,
        },
        "sar": {
            "item_id": "S1B_IW_GRDH_mock",
            "acquisition_datetime": "2021-06-27T16:00:00Z",
            "sensor": "Sentinel-1B",
            "product": "GRD",
            "mode": "IW",
            "orbit_direction": "ascending",
            "polarizations": ["VV", "VH"],
            "coverage_percentage": 100.0,
            "crs": "EPSG:4326",
            "bounds": [13.0, 48.0, 13.02, 48.02],
            "path": sar_vv,
            "vv": sar_vv,
            "vh": sar_vh,
        },
        "errors": [],
    }


# ============================================================
# TESTS
# ============================================================

def test_1_auto_mode_invokes_find_optical_sar_pair(test_client, mock_pair_paths):
    """
    Test 1: Automatic mode without explicit image references invokes find_optical_sar_pair.
    """
    with patch("app.agent.executor.find_optical_sar_pair", return_value=mock_pair_paths) as mock_matcher:
        with patch("app.vlm.model.VLM.generate", return_value="Built-up structures identified."):
            payload = {
                "query": "Use the optical and SAR images together to identify built-up areas.",
                "aoi": {
                    "type": "Polygon",
                    "coordinates": [[[13.0, 48.0], [13.02, 48.0], [13.02, 48.02], [13.0, 48.02], [13.0, 48.0]]],
                },
                "time_start": "2021-06-25",
                "time_end": "2021-06-28",
            }
            res = test_client.post("/api/query", json=payload)
            assert res.status_code == 200
            assert mock_matcher.called is True


def test_2_auto_pairing_paths_reach_vlm(test_client, mock_pair_paths):
    """
    Test 2: Automatic pairing result paths reach run_optical_sar_analysis and VLM specialist.
    """
    with patch("app.agent.executor.find_optical_sar_pair", return_value=mock_pair_paths):
        with patch("app.vlm.optical_sar.answer_optical_sar_question") as mock_answer:
            mock_answer.return_value = {
                "success": True,
                "answer": "Grounded multimodal analysis answer.",
                "modalities": ["optical", "sar_vv", "sar_vh"],
                "metadata": {"grid": {"width": 20, "height": 20}},
                "evidence_used": True,
                "visuals": {},
                "fallback": False,
            }

            payload = {
                "query": "Use the optical and SAR images together to identify built-up areas.",
                "aoi": [13.0, 48.0, 13.02, 48.02],
                "time_start": "2021-06-25",
                "time_end": "2021-06-28",
            }
            res = test_client.post("/api/query", json=payload)
            assert res.status_code == 200
            assert mock_answer.called is True
            data = res.json()
            assert data["status"] == "success"
            assert data["answer"] == "Grounded multimodal analysis answer."


def test_3_explicit_inputs_bypass_automatic_acquisition(test_client, tmp_path):
    """
    Test 3: Explicit optical + SAR inputs bypass automatic acquisition.
    """
    opt_file = tmp_path / "explicit_opt.tif"
    sar_file = tmp_path / "explicit_sar.tif"
    _create_mock_geotiff(opt_file, count=3)
    _create_mock_geotiff(sar_file, count=2)

    with patch("app.agent.executor.find_optical_sar_pair") as mock_matcher:
        with patch("app.vlm.model.VLM.generate", return_value="Explicit analysis complete."):
            with patch("app.remote_sensing.multimodal.ingestion.resolve_image_reference") as mock_resolve:
                mock_resolve.side_effect = lambda ref: opt_file if ref == "opt_123" else sar_file

                payload = {
                    "query": "Use the optical and SAR images together to identify built-up areas.",
                    "optical_image_id": "opt_123",
                    "sar_image_id": "sar_123",
                    "aoi": [13.0, 48.0, 13.02, 48.02],
                }
                res = test_client.post("/api/query", json=payload)
                assert res.status_code == 200
                assert mock_matcher.called is False


def test_4_missing_aoi_fails_cleanly(test_client):
    """
    Test 4: Missing AOI fails cleanly with HTTP 400 "Optical-SAR automatic acquisition requires an AOI.".
    """
    payload = {
        "query": "Use the optical and SAR images together to identify built-up areas.",
        "time_start": "2021-06-25",
        "time_end": "2021-06-28",
    }
    res = test_client.post("/api/query", json=payload)
    assert res.status_code == 400
    assert "Optical-SAR automatic acquisition requires an AOI." in res.json()["detail"]


def test_5_missing_date_fails_cleanly(test_client):
    """
    Test 5: Missing date/time fails cleanly with HTTP 400 "Optical-SAR automatic acquisition requires a target date or time range.".
    """
    payload = {
        "query": "Use the optical and SAR images together to identify built-up areas.",
        "aoi": [13.0, 48.0, 13.02, 48.02],
    }
    res = test_client.post("/api/query", json=payload)
    assert res.status_code == 400
    assert "Optical-SAR automatic acquisition requires a target date or time range." in res.json()["detail"]


def test_6_no_compatible_pair_fails_cleanly(test_client):
    """
    Test 6: No compatible pair fails cleanly (HTTP 404).
    """
    failure_result = {
        "status": "REAL_FAILURE",
        "pair_found": False,
        "error_type": PairingErrorType.NO_TEMPORALLY_COMPATIBLE_PAIR,
        "error": "No compatible Optical-SAR pair found within 3.0 days.",
        "details": {},
        "errors": ["No compatible pair found."],
    }

    with patch("app.agent.executor.find_optical_sar_pair", return_value=failure_result):
        payload = {
            "query": "Use the optical and SAR images together to identify built-up areas.",
            "aoi": [13.0, 48.0, 13.02, 48.02],
            "time_start": "2021-06-25",
            "time_end": "2021-06-28",
        }
        res = test_client.post("/api/query", json=payload)
        assert res.status_code == 404
        assert "No compatible Optical-SAR pair found" in res.json()["detail"]


def test_7_pair_metadata_survives_into_api_response(test_client, mock_pair_paths):
    """
    Test 7: Pair selection metadata survives into API response (statistics, evidence).
    """
    with patch("app.agent.executor.find_optical_sar_pair", return_value=mock_pair_paths):
        with patch("app.vlm.model.VLM.generate", return_value="Urban structures detected."):
            payload = {
                "query": "Use the optical and SAR images together to identify built-up areas.",
                "aoi": [13.0, 48.0, 13.02, 48.02],
                "time_start": "2021-06-25",
                "time_end": "2021-06-28",
            }
            res = test_client.post("/api/query", json=payload)
            assert res.status_code == 200
            data = res.json()

            stats = data.get("statistics", {})
            pair_meta = stats.get("optical_sar_pair", {})
            assert pair_meta.get("source") == "automatic"
            assert pair_meta.get("optical_item_id") == "S2A_MSIL2A_mock"
            assert pair_meta.get("sar_item_id") == "S1B_IW_GRDH_mock"
            assert pair_meta.get("temporal_delta_days") == pytest.approx(0.28, abs=0.01)
            assert "VV" in pair_meta.get("polarizations", [])


def test_8_execution_trace_contains_acquisition_steps(test_client, mock_pair_paths):
    """
    Test 8: Execution trace contains acquisition/pairing steps.
    """
    with patch("app.agent.executor.find_optical_sar_pair", return_value=mock_pair_paths):
        with patch("app.vlm.model.VLM.generate", return_value="Analysis complete."):
            payload = {
                "query": "Use the optical and SAR images together to identify built-up areas.",
                "aoi": [13.0, 48.0, 13.02, 48.02],
                "time_start": "2021-06-25",
                "time_end": "2021-06-28",
            }
            res = test_client.post("/api/query", json=payload)
            assert res.status_code == 200
            data = res.json()
            trace = data.get("execution_trace", [])
            trace_text = " ".join(trace).lower()

            assert "automatic optical-sar mode triggered" in trace_text
            assert "sentinel-2" in trace_text
            assert "sentinel-1" in trace_text
            assert "pair selected" in trace_text


def test_9_ndvi_does_not_invoke_sentinel1(test_client):
    """
    Test 9: Normal NDVI query does not invoke Sentinel-1 acquisition.
    """
    with patch("app.agent.executor.find_optical_sar_pair") as mock_matcher:
        with patch("app.api.routes_query.execute_plan", return_value={"calculate_ndvi": {"mean": 0.45}}):
            payload = {
                "query": "Calculate NDVI for Pune.",
                "aoi": [73.80, 18.50, 73.86, 18.56],
            }
            res = test_client.post("/api/query", json=payload)
            # find_optical_sar_pair must never be called for NDVI queries
            assert mock_matcher.called is False


def test_10_change_detection_does_not_invoke_sentinel1(test_client):
    """
    Test 10: Normal change detection does not invoke Sentinel-1 acquisition.
    """
    with patch("app.agent.executor.find_optical_sar_pair") as mock_matcher:
        with patch("app.api.routes_query.execute_plan", return_value={"detect_change": {"change_ratio": 0.12}}):
            payload = {
                "query": "What changed between 2021 and 2025?",
                "aoi": [73.80, 18.50, 73.86, 18.56],
            }
            res = test_client.post("/api/query", json=payload)
            # find_optical_sar_pair must never be called for change detection queries
            assert mock_matcher.called is False


def test_11_existing_upload_reference_test_still_passes(test_client, tmp_path):
    """
    Test 11: Existing upload/reference Optical–SAR test still passes.
    """
    opt_file = tmp_path / "legacy_opt.tif"
    sar_file = tmp_path / "legacy_sar.tif"
    _create_mock_geotiff(opt_file, count=3)
    _create_mock_geotiff(sar_file, count=2)

    with patch("app.remote_sensing.multimodal.ingestion.resolve_image_reference") as mock_resolve:
        mock_resolve.side_effect = lambda ref: opt_file if ref == "opt_ref" else sar_file
        with patch("app.vlm.model.VLM.generate", return_value="Legacy mode works."):
            payload = {
                "query": "Analyze the optical and SAR images together.",
                "optical_image_id": "opt_ref",
                "sar_image_id": "sar_ref",
            }
            res = test_client.post("/api/query", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "success"
            assert data["statistics"]["optical_sar_pair"]["source"] == "user_supplied"


def test_12_vlm_failure_returns_controlled_fallback(test_client, mock_pair_paths):
    """
    Test 12: VLM failure still returns controlled fallback/error behavior without 500 crash.
    """
    with patch("app.agent.executor.find_optical_sar_pair", return_value=mock_pair_paths):
        with patch("app.vlm.model.VLM.generate", side_effect=RuntimeError("VLM connection timed out")):
            payload = {
                "query": "Use the optical and SAR images together to identify built-up areas.",
                "aoi": [13.0, 48.0, 13.02, 48.02],
                "time_start": "2021-06-25",
                "time_end": "2021-06-28",
            }
            res = test_client.post("/api/query", json=payload)
            # Handled gracefully via specialist fallback
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "success"
            assert "fallback" in data["answer"].lower() or "built-up" in data["answer"].lower()


def test_13_auto_pair_result_validates(mock_pair_paths):
    """
    Test 13: Automatic pair result is compatible with validate_optical_sar_pair.
    """
    val = validate_optical_sar_pair(
        mock_pair_paths["optical"]["path"],
        mock_pair_paths["sar"]["path"],
    )
    assert val["valid"] is True
    assert val["compatibility"]["spatial_overlap"] is True


def test_14_auto_pair_result_aligns(mock_pair_paths):
    """
    Test 14: Automatic pair result is compatible with align_optical_sar_pair.
    """
    align = align_optical_sar_pair(
        mock_pair_paths["optical"]["path"],
        mock_pair_paths["sar"]["path"],
    )
    assert align["success"] is True
    assert align["alignment"]["target_width"] == 20
    assert align["alignment"]["target_height"] == 20
    assert align["valid_mask"] is not None


def test_15_no_bigearthnet_fallback_occurs(test_client, mock_pair_paths):
    """
    Test 15: No BigEarthNet fallback occurs during automatic Optical-SAR execution.
    """
    with patch("app.agent.executor.find_optical_sar_pair", return_value=mock_pair_paths):
        with patch("app.vlm.model.VLM.generate", return_value="Independent multimodal analysis without benchmark dataset."):
            payload = {
                "query": "Use the optical and SAR images together to identify built-up areas.",
                "aoi": [13.0, 48.0, 13.02, 48.02],
                "time_start": "2021-06-25",
                "time_end": "2021-06-28",
            }
            res = test_client.post("/api/query", json=payload)
            assert res.status_code == 200
            data = res.json()
            answer = data.get("answer", "")
            assert "bigearthnet" not in answer.lower()
            stats = data.get("statistics", {})
            pair_info = stats.get("optical_sar_pair", {})
            assert "bigearthnet" not in str(pair_info.get("optical_item_id", "")).lower()
            assert "bigearthnet" not in str(pair_info.get("sar_item_id", "")).lower()
            assert "33uup" not in str(pair_info.get("optical_item_id", "")).lower()
            assert "33uup" not in str(pair_info.get("sar_item_id", "")).lower()
            assert pair_info.get("source") == "automatic"
