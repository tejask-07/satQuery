"""
Comprehensive Test Suite for Optical-SAR API Output Verification.

Tests all 9 specifications required:
1. Test 1: Live VLM path selection (mocked VLM, verifies /api/query uses live VLM, VLM.generate is reached).
2. Test 2: VLM failure fallback (verifies deterministic fallback when inference service fails).
3. Test 3: Dual-pol layer response (valid VV + VH returns optical, s1_vv, s1_vh, s1_composite).
4. Test 4: VV-only response (only optical and s1_vv appear; no VH/composite fabrication).
5. Test 5: Modality metadata (statistics.modalities matches actual layers).
6. Test 6: No filesystem leakage (complete JSON serialized has zero local machine paths).
7. Test 7: Provenance preserved (item_ids, timestamps, delta, polarizations, selection_reason).
8. Test 8: Normal NDVI query unaffected.
9. Test 9: Normal change detection query unaffected.
"""

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from fastapi.testclient import TestClient

from app.main import app


# ============================================================
# FIXTURES & HELPERS
# ============================================================

def _create_mock_raster(path: Path, count: int = 1, width: int = 20, height: int = 20, value: int = 100):
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
def client():
    return TestClient(app)


@pytest.fixture
def dual_pol_pair(tmp_path):
    opt_path = _create_mock_raster(tmp_path / "test_s2_rgb.tif", count=3, value=150)
    sar_vv = _create_mock_raster(tmp_path / "test_s1_vv.tif", count=1, value=250)
    sar_vh = _create_mock_raster(tmp_path / "test_s1_vh.tif", count=1, value=60)
    return {
        "status": "REAL_SUCCESS",
        "pair_found": True,
        "temporal_delta_days": 0.287,
        "spatial_overlap": {
            "has_overlap": True,
            "intersection_bbox": [13.0, 48.0, 13.02, 48.02],
            "aoi_coverage_percent": 100.0,
        },
        "selection_reason": "Selected Optical S2B_MSIL2A_mock (34.3% cloud) and SAR S1B_IW_mock (temporal delta: 0.29 days, polarizations: ['VV', 'VH']) with score 0.898.",
        "optical": {
            "item_id": "S2B_MSIL2A_test_item",
            "acquisition_datetime": "2021-06-27T10:05:59Z",
            "sensor": "Sentinel-2B",
            "product": "Level-2A",
            "cloud_cover": 34.26,
            "coverage_percentage": 100.0,
            "crs": "EPSG:4326",
            "bounds": [13.0, 48.0, 13.02, 48.02],
            "path": opt_path,
        },
        "sar": {
            "item_id": "S1B_IW_GRDH_test_item",
            "acquisition_datetime": "2021-06-27T16:58:47Z",
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


@pytest.fixture
def vv_only_pair(tmp_path):
    opt_path = _create_mock_raster(tmp_path / "vv_only_s2_rgb.tif", count=3, value=150)
    # File without companion _vh.tif
    sar_vv = _create_mock_raster(tmp_path / "vv_only_radar.tif", count=1, value=250)
    return {
        "status": "REAL_SUCCESS",
        "pair_found": True,
        "temporal_delta_days": 0.45,
        "spatial_overlap": {
            "has_overlap": True,
            "intersection_bbox": [13.0, 48.0, 13.02, 48.02],
            "aoi_coverage_percent": 100.0,
        },
        "selection_reason": "Selected Optical S2A_mock and SAR S1A_mock VV-only with score 0.85.",
        "optical": {
            "item_id": "S2A_MSIL2A_vv_only",
            "acquisition_datetime": "2021-06-27T10:00:00Z",
            "sensor": "Sentinel-2A",
            "product": "Level-2A",
            "cloud_cover": 5.0,
            "coverage_percentage": 100.0,
            "crs": "EPSG:4326",
            "bounds": [13.0, 48.0, 13.02, 48.02],
            "path": opt_path,
        },
        "sar": {
            "item_id": "S1A_IW_GRDH_vv_only",
            "acquisition_datetime": "2021-06-27T18:00:00Z",
            "sensor": "Sentinel-1A",
            "product": "GRD",
            "mode": "IW",
            "orbit_direction": "descending",
            "polarizations": ["VV"],
            "coverage_percentage": 100.0,
            "crs": "EPSG:4326",
            "bounds": [13.0, 48.0, 13.02, 48.02],
            "path": sar_vv,
            "vv": sar_vv,
            "vh": None,
        },
        "errors": [],
    }


# ============================================================
# TEST CASES
# ============================================================

def test_1_live_vlm_path_selection(client, dual_pol_pair):
    """
    Test 1: Mock a successful VLM and verify /api/query uses it instead of fallback.
    Verify the specialist's VLM.generate() is reached without calling real Hugging Face.
    """
    mock_answer = "Likely built-up structures identified through high optical red reflectance and strong SAR double-bounce backscatter."
    with patch("app.agent.executor.find_optical_sar_pair", return_value=dual_pol_pair):
        with patch("app.vlm.model.VLM.generate", return_value=mock_answer) as mock_vlm_gen:
            with patch.dict("os.environ", {"HF_TOKEN": "mock_test_token"}):
                res = client.post(
                    "/api/query",
                    json={
                        "query": "Use the optical and SAR images together to identify likely built-up areas.",
                        "aoi": [13.0, 48.0, 13.02, 48.02],
                        "time_start": "2021-06-25",
                        "time_end": "2021-06-28",
                    },
                )
                assert res.status_code == 200, res.text
                data = res.json()
                assert data["answer"] == mock_answer
                assert "VLM inference service was unavailable" not in data["answer"]
                assert mock_vlm_gen.called


def test_2_vlm_failure_fallback(client, dual_pol_pair):
    """
    Test 2: Mock VLM failure and verify deterministic fallback still works gracefully.
    """
    with patch("app.agent.executor.find_optical_sar_pair", return_value=dual_pol_pair):
        with patch("app.vlm.model.VLM.generate", side_effect=RuntimeError("Simulated remote 503 error")):
            with patch.dict("os.environ", {"HF_TOKEN": "mock_test_token"}):
                res = client.post(
                    "/api/query",
                    json={
                        "query": "Use the optical and SAR images together to identify likely built-up areas.",
                        "aoi": [13.0, 48.0, 13.02, 48.02],
                        "time_start": "2021-06-25",
                        "time_end": "2021-06-28",
                    },
                )
                assert res.status_code == 200, res.text
                data = res.json()
                assert data["answer"] is not None
                assert "VLM" in data["answer"]
                assert "deterministic multimodal analysis summary" in data["answer"]


def test_3_dual_pol_layer_response(client, dual_pol_pair):
    """
    Test 3: Mock/fixture a valid VV + VH pair and verify optical, s1_vv, s1_vh, and s1_composite are returned.
    """
    with patch("app.agent.executor.find_optical_sar_pair", return_value=dual_pol_pair):
        with patch("app.vlm.model.VLM.generate", return_value="Dual-pol multimodal analysis complete."):
            with patch.dict("os.environ", {"HF_TOKEN": "mock_test_token"}):
                res = client.post(
                    "/api/query",
                    json={
                        "query": "Analyze optical and SAR data together for vegetation and urban structures.",
                        "aoi": [13.0, 48.0, 13.02, 48.02],
                        "time_start": "2021-06-25",
                        "time_end": "2021-06-28",
                    },
                )
                assert res.status_code == 200
                data = res.json()
                layer_types = [l.get("type") for l in data.get("layers", [])]
                assert "optical_rgb" in layer_types
                assert "sar_vv" in layer_types
                assert "sar_vh" in layer_types
                assert "sar_composite" in layer_types

                images = data.get("images", {})
                assert "optical" in images
                assert "s1_vv" in images
                assert "s1_vh" in images
                assert "s1_composite" in images


def test_4_vv_only_response(client, vv_only_pair):
    """
    Test 4: Verify only VV-related layers appear when only VV polarization exists (no fabrication of VH or composite).
    """
    with patch("app.agent.executor.find_optical_sar_pair", return_value=vv_only_pair):
        with patch("app.vlm.model.VLM.generate", return_value="VV-only analysis complete."):
            with patch.dict("os.environ", {"HF_TOKEN": "mock_test_token"}):
                res = client.post(
                    "/api/query",
                    json={
                        "query": "Analyze optical and SAR data together.",
                        "aoi": [13.0, 48.0, 13.02, 48.02],
                        "time_start": "2021-06-25",
                        "time_end": "2021-06-28",
                    },
                )
                assert res.status_code == 200
                data = res.json()
                layer_types = [l.get("type") for l in data.get("layers", [])]
                assert "optical_rgb" in layer_types
                assert "sar_vv" in layer_types
                assert "sar_vh" not in layer_types
                assert "sar_composite" not in layer_types

                images = data.get("images", {})
                assert "optical" in images
                assert "s1_vv" in images
                assert "s1_vh" not in images
                assert "s1_composite" not in images


def test_5_modality_metadata(client, dual_pol_pair, vv_only_pair):
    """
    Test 5: Verify statistics.modalities accurately matches the actual returned layers for dual-pol and VV-only.
    """
    # Dual-pol
    with patch("app.agent.executor.find_optical_sar_pair", return_value=dual_pol_pair):
        with patch("app.vlm.model.VLM.generate", return_value="Dual-pol test"):
            with patch.dict("os.environ", {"HF_TOKEN": "mock_test_token"}):
                res_dp = client.post(
                    "/api/query",
                    json={
                        "query": "Optical and SAR multimodal test.",
                        "aoi": [13.0, 48.0, 13.02, 48.02],
                        "time_start": "2021-06-25",
                        "time_end": "2021-06-28",
                    },
                )
                dp_mods = res_dp.json()["statistics"]["modalities"]
                assert dp_mods == ["optical", "sar_vv", "sar_vh", "sar_composite"]

    # VV-only
    with patch("app.agent.executor.find_optical_sar_pair", return_value=vv_only_pair):
        with patch("app.vlm.model.VLM.generate", return_value="VV-only test"):
            with patch.dict("os.environ", {"HF_TOKEN": "mock_test_token"}):
                res_vv = client.post(
                    "/api/query",
                    json={
                        "query": "Optical and SAR multimodal test.",
                        "aoi": [13.0, 48.0, 13.02, 48.02],
                        "time_start": "2021-06-25",
                        "time_end": "2021-06-28",
                    },
                )
                vv_mods = res_vv.json()["statistics"]["modalities"]
                assert vv_mods == ["optical", "sar_vv"]


def test_6_no_filesystem_leakage(client, dual_pol_pair):
    """
    Test 6: Serialize the complete API response and assert it does NOT contain raw machine-specific internal paths.
    Uses generic cross-platform path-leak detection.
    """
    with patch("app.agent.executor.find_optical_sar_pair", return_value=dual_pol_pair):
        with patch("app.vlm.model.VLM.generate", return_value="Leakage test answer."):
            with patch.dict("os.environ", {"HF_TOKEN": "mock_test_token"}):
                res = client.post(
                    "/api/query",
                    json={
                        "query": "Use the optical and SAR images together to identify likely built-up areas.",
                        "aoi": [13.0, 48.0, 13.02, 48.02],
                        "time_start": "2021-06-25",
                        "time_end": "2021-06-28",
                    },
                )
                assert res.status_code == 200
                raw_json = json.dumps(res.json())

                # Generic path leak regexes
                # Windows drive letter paths: C:\ or C:/ or D:\
                win_drive_match = re.search(r'[A-Za-z]:[\\/]', raw_json)
                assert win_drive_match is None, f"Detected Windows filesystem path in response: {win_drive_match.group()}"

                # Unix server directories: /home/, /Users/, /etc/, /tmp/
                unix_match = re.search(r'/(?:home|Users|etc|tmp|var|private|root)/', raw_json)
                assert unix_match is None, f"Detected Unix filesystem path in response: {unix_match.group()}"

                # Cache directory strings
                assert "data/cache" not in raw_json
                assert "data\\cache" not in raw_json
                assert "backend/data" not in raw_json
                assert "backend\\data" not in raw_json

                # Raster GeoTIFF file references (.tif on filesystem)
                assert ".tif" not in raw_json


def test_7_provenance_preserved(client, dual_pol_pair):
    """
    Test 7: Verify provenance fields remain available in statistics.optical_sar_pair.
    """
    with patch("app.agent.executor.find_optical_sar_pair", return_value=dual_pol_pair):
        with patch("app.vlm.model.VLM.generate", return_value="Provenance test answer."):
            with patch.dict("os.environ", {"HF_TOKEN": "mock_test_token"}):
                res = client.post(
                    "/api/query",
                    json={
                        "query": "Use the optical and SAR images together.",
                        "aoi": [13.0, 48.0, 13.02, 48.02],
                        "time_start": "2021-06-25",
                        "time_end": "2021-06-28",
                    },
                )
                assert res.status_code == 200
                pair_meta = res.json()["statistics"]["optical_sar_pair"]

                assert pair_meta["optical_item_id"] == "S2B_MSIL2A_test_item"
                assert pair_meta["sar_item_id"] == "S1B_IW_GRDH_test_item"
                assert pair_meta["optical_acquisition_datetime"] == "2021-06-27T10:05:59Z"
                assert pair_meta["sar_acquisition_datetime"] == "2021-06-27T16:58:47Z"
                assert pair_meta["temporal_delta_days"] == 0.287
                assert pair_meta["polarizations"] == ["VV", "VH"]
                assert "Selected Optical" in pair_meta["selection_reason"]
                assert pair_meta["optical"]["item_id"] == "S2B_MSIL2A_test_item"
                assert pair_meta["sar"]["item_id"] == "S1B_IW_GRDH_test_item"
                assert pair_meta["sar"]["mode"] == "IW"


def test_8_normal_ndvi_unaffected(client):
    """
    Test 8: Verify normal NDVI queries remain unaffected and do not invoke Optical-SAR logic.
    """
    mock_ndvi_res = {
        "status": "success",
        "index": "NDVI",
        "mean": 0.45,
        "min_value": 0.1,
        "max_value": 0.8,
        "valid_pixels": 1000,
        "total_pixels": 1000,
    }
    with patch("app.api.routes_query.execute_plan", return_value={"calculate_ndvi": mock_ndvi_res}):
        res = client.post(
            "/api/query",
            json={
                "query": "Calculate the average NDVI for AOI [13.0, 48.0, 13.02, 48.02] on 2021-06-25",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["plan"]["task"] != "optical_sar_analysis"
        assert data["statistics"].get("metric") == "NDVI"
        assert data["statistics"].get("mean") == 0.45


def test_9_normal_change_detection_unaffected(client):
    """
    Test 9: Verify normal change detection queries remain unaffected and do not invoke Optical-SAR logic.
    """
    mock_change_res = {
        "status": "success",
        "metric": "NDVI",
        "mean_before": 0.6,
        "mean_after": 0.3,
        "mean_change": -0.3,
        "change_ratio": 0.25,
        "regions_detected": 2,
        "changed_pixels": 250,
        "valid_pixels": 1000,
        "total_pixels": 1000,
    }
    with patch("app.api.routes_query.execute_plan", return_value={"detect_change": mock_change_res}):
        res = client.post(
            "/api/query",
            json={
                "query": "Detect vegetation change between 2020-01-01 and 2021-01-01 for AOI [13.0, 48.0, 13.02, 48.02]",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["plan"]["task"] != "optical_sar_analysis"
        assert data["statistics"].get("metric") == "NDVI"
        assert data["statistics"].get("mean_change") == -0.3
