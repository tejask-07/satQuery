"""
Focused Test Suite for Optical-SAR Confidence Handling.

Verifies:
1. Model confidence is propagated when available (RS-VLM or specialist result).
2. Missing confidence returns None (no invented numbers).
3. Deterministic evidence/fusion confidence is used if available when model confidence is missing.
4. No hardcoded 0.9 remains anywhere in the Optical-SAR response path.
5. End-to-end API response propagates confidence correctly.
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
from app.schemas.analysis import AnalysisResult
from app.schemas.query import QueryPlan
from app.api.routes_query import (
    _build_optical_sar_api_response,
    _extract_optical_sar_confidence,
)


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
def test_client():
    return TestClient(app)


@pytest.fixture
def dual_pol_pair(tmp_path):
    opt_path = _create_mock_raster(tmp_path / "s2_rgb.tif", count=3, value=150)
    sar_vv = _create_mock_raster(tmp_path / "s1_vv.tif", count=1, value=250)
    sar_vh = _create_mock_raster(tmp_path / "s1_vh.tif", count=1, value=60)
    return {
        "status": "REAL_SUCCESS",
        "pair_found": True,
        "temporal_delta_days": 0.287,
        "spatial_overlap": {
            "has_overlap": True,
            "intersection_bbox": [13.0, 48.0, 13.02, 48.02],
            "aoi_coverage_percent": 100.0,
        },
        "selection_reason": "Selected Optical S2B and SAR S1B with score 0.898.",
        "optical": {
            "item_id": "S2B_MSIL2A_item",
            "acquisition_datetime": "2021-06-27T10:05:59Z",
            "sensor": "Sentinel-2B",
            "product": "Level-2A",
            "cloud_cover": 12.0,
            "coverage_percentage": 100.0,
            "crs": "EPSG:4326",
            "bounds": [13.0, 48.0, 13.02, 48.02],
            "path": opt_path,
        },
        "sar": {
            "item_id": "S1B_IW_GRDH_item",
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
def mock_plan():
    return QueryPlan(
        task="optical_sar_analysis",
        target="urban",
        metric=None,
        intent="optical_sar_analysis",
        outputs=["explanation", "visualization"],
    )


# ============================================================
# 1. UNIT TESTS: _extract_optical_sar_confidence
# ============================================================

def test_extract_confidence_from_direct_field():
    """Model confidence in sar_res['confidence'] is extracted directly."""
    sar_res = {"answer": "Ground test", "confidence": 0.85}
    assert _extract_optical_sar_confidence(sar_res) == 0.85


def test_extract_confidence_from_nested_vlm_res():
    """Confidence nested in sar_res['vlm_res']['confidence'] is extracted."""
    sar_res = {
        "answer": "Ground test",
        "vlm_res": {"confidence": 0.92, "model": "rs-vlm-custom"},
    }
    assert _extract_optical_sar_confidence(sar_res) == 0.92


def test_extract_confidence_from_model_dict():
    """Confidence nested in sar_res['model']['confidence'] is extracted."""
    sar_res = {
        "answer": "Ground test",
        "model": {"name": "test-vlm", "confidence": 0.78},
    }
    assert _extract_optical_sar_confidence(sar_res) == 0.78


def test_extract_confidence_prefers_model_over_deterministic():
    """Model confidence takes precedence over deterministic evidence confidence."""
    sar_res = {
        "answer": "Ground test",
        "confidence": 0.88,
        "fusion_result": {"confidence": 0.65},
    }
    assert _extract_optical_sar_confidence(sar_res) == 0.88


def test_extract_confidence_falls_back_to_deterministic_evidence():
    """Deterministic evidence confidence is used when model confidence is absent."""
    sar_res = {
        "answer": "Ground test",
        "confidence": None,
        "evidence": {"confidence": 0.74},
    }
    assert _extract_optical_sar_confidence(sar_res) == 0.74


def test_extract_confidence_from_evidence_list():
    """Deterministic confidence inside evidence list is extracted."""
    sar_res = {
        "answer": "Ground test",
        "evidence": [
            {"source": "optical_sar", "confidence": 0.71},
        ],
    }
    assert _extract_optical_sar_confidence(sar_res) == 0.71


def test_extract_confidence_from_fusion_result():
    """Deterministic confidence from fusion_result is extracted."""
    sar_res = {
        "answer": "Ground test",
        "fusion_result": {"confidence": 0.68},
    }
    assert _extract_optical_sar_confidence(sar_res) == 0.68


def test_extract_confidence_returns_none_when_no_source():
    """When no model or deterministic confidence exists, returns None (never hardcoded 0.9)."""
    sar_res = {
        "answer": "Ground test",
        "modalities": ["optical", "sar_vv"],
        "metadata": {},
    }
    assert _extract_optical_sar_confidence(sar_res) is None
    assert _extract_optical_sar_confidence(sar_res) != 0.9


def test_extract_confidence_handles_invalid_ranges():
    """Out-of-range or non-numeric values are safely ignored and return None."""
    assert _extract_optical_sar_confidence({"confidence": 1.5}) is None
    assert _extract_optical_sar_confidence({"confidence": -0.2}) is None
    assert _extract_optical_sar_confidence({"confidence": "not_a_number"}) is None
    assert _extract_optical_sar_confidence(None) is None


# ============================================================
# 2. UNIT TESTS: _build_optical_sar_api_response
# ============================================================

def test_build_response_propagates_model_confidence(mock_plan):
    """_build_optical_sar_api_response sets AnalysisResult.confidence from model result."""
    sar_res = {
        "answer": "Test answer with high certainty",
        "confidence": 0.87,
        "modalities": ["optical", "sar_vv"],
        "visuals": {},
    }
    res = _build_optical_sar_api_response(
        plan=mock_plan,
        query="Analyze optical and SAR",
        sar_res=sar_res,
    )
    assert isinstance(res, AnalysisResult)
    assert res.confidence == 0.87


def test_build_response_missing_confidence_returns_none(mock_plan):
    """_build_optical_sar_api_response sets AnalysisResult.confidence = None when missing."""
    sar_res = {
        "answer": "Test answer without confidence score",
        "modalities": ["optical", "sar_vv"],
        "visuals": {},
    }
    res = _build_optical_sar_api_response(
        plan=mock_plan,
        query="Analyze optical and SAR",
        sar_res=sar_res,
    )
    assert isinstance(res, AnalysisResult)
    assert res.confidence is None
    # Verify explicitly that the old hardcoded 0.9 is NOT present
    assert res.confidence != 0.9


def test_build_response_no_hardcoded_0_9(mock_plan):
    """Ensure hardcoded 0.9 is completely eliminated across empty and mock results."""
    empty_sar_res = {}
    res = _build_optical_sar_api_response(
        plan=mock_plan,
        query="Analyze optical and SAR",
        sar_res=empty_sar_res,
    )
    assert res.confidence is None
    assert res.confidence != 0.9


# ============================================================
# 3. SPECIALIST & API INTEGRATION TESTS
# ============================================================

def test_api_optical_sar_propagates_model_confidence(test_client, dual_pol_pair):
    """End-to-end /api/query propagates model confidence when returned by specialist."""
    mock_optical_sar_result = {
        "success": True,
        "answer": "Multimodal analysis with calibrated confidence.",
        "confidence": 0.84,
        "modalities": ["optical", "sar_vv", "sar_vh", "sar_composite"],
        "metadata": {},
        "evidence_used": False,
        "visuals": {},
        "fallback": False,
        "model": "satquery-rs-vlm-v1",
        "status": "active",
        "adapter_loaded": True,
    }
    with patch("app.agent.executor.find_optical_sar_pair", return_value=dual_pol_pair):
        with patch("app.api.routes_query.execute_plan", return_value={"optical_sar_analysis": mock_optical_sar_result}):
            res = test_client.post(
                "/api/query",
                json={
                    "query": "Use optical and SAR images together to identify built-up areas.",
                    "aoi": [13.0, 48.0, 13.02, 48.02],
                    "time_start": "2021-06-25",
                    "time_end": "2021-06-28",
                },
            )
            assert res.status_code == 200, res.text
            data = res.json()
            assert data["status"] == "success"
            assert data["confidence"] == 0.84


def test_api_optical_sar_returns_none_confidence_by_default(test_client, dual_pol_pair):
    """End-to-end /api/query returns confidence=None when model provides none (no hardcoded 0.9)."""
    with patch("app.agent.executor.find_optical_sar_pair", return_value=dual_pol_pair):
        with patch("app.vlm.model.VLM.generate", return_value="Standard mock VLM answer."):
            with patch.dict("os.environ", {"HF_TOKEN": "mock_test_token"}):
                res = test_client.post(
                    "/api/query",
                    json={
                        "query": "Use optical and SAR images together to identify built-up areas.",
                        "aoi": [13.0, 48.0, 13.02, 48.02],
                        "time_start": "2021-06-25",
                        "time_end": "2021-06-28",
                    },
                )
                assert res.status_code == 200, res.text
                data = res.json()
                assert data["status"] == "success"
                assert data["confidence"] is None
                assert data["confidence"] != 0.9
