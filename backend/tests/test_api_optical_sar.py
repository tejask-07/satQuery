"""
Tests for Step 7: Real Optical–SAR Input Ingestion and API Integration.

Validates:
1. Valid Optical + SAR request via /api/query
2. Optical-only rejected with clean HTTP 400 error
3. SAR-only rejected with clean HTTP 400 error
4. Invalid image reference rejected cleanly
5. Normal NDVI request unaffected
6. Normal NDWI request unaffected
7. Normal temporal change request unaffected
8. Optical + SAR with explicit scientific index routes to index (NDVI priority)
9. Invalid non-raster upload rejected
10. Security: arbitrary filesystem path injection blocked
11. Multipart direct upload via /api/upload/optical-sar
12. Backwards-compatible image_ids resolution
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi.testclient import TestClient
from rasterio.crs import CRS

from app.main import app
from app.remote_sensing.multimodal.ingestion import (
    store_uploaded_raster,
    resolve_image_reference,
    resolve_optical_sar_references,
    UPLOAD_DIR,
)


# ============================================================
# FIXTURES & TEST GEOTIFF GENERATORS
# ============================================================

def _generate_geotiff_bytes(
    count: int = 3,
    width: int = 64,
    height: int = 64,
    dtype: str = "uint16",
    crs_epsg: int = 32633,
) -> bytes:
    """Generate in-memory GeoTIFF bytes."""
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0)
    crs = CRS.from_epsg(crs_epsg)

    bio = io.BytesIO()
    with rasterio.open(
        bio,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for b in range(1, count + 1):
            arr = (np.random.rand(height, width) * 2000 + 500).astype(dtype)
            dst.write(arr, b)

    return bio.getvalue()


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def sample_optical_bytes():
    return _generate_geotiff_bytes(count=3, width=64, height=64, dtype="uint16")


@pytest.fixture
def sample_sar_bytes():
    return _generate_geotiff_bytes(count=2, width=64, height=64, dtype="float32")


@pytest.fixture
def uploaded_pair(sample_optical_bytes, sample_sar_bytes):
    opt_id = store_uploaded_raster(sample_optical_bytes, "test_optical.tif", modality_hint="optical")
    sar_id = store_uploaded_raster(sample_sar_bytes, "test_sar.tif", modality_hint="sar")
    return opt_id, sar_id


# ============================================================
# TESTS
# ============================================================

def test_1_valid_optical_sar_request_via_query_api(test_client, uploaded_pair):
    """Test 1: Valid Optical + SAR request via /api/query."""
    opt_id, sar_id = uploaded_pair

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = "Identified dense urban structures via double-bounce radar and high reflectance."

    with patch("app.vlm.model.VLM", return_value=mock_vlm):
        response = test_client.post(
            "/api/query",
            json={
                "query": "Use the optical and SAR images together to identify built-up areas.",
                "optical_image_id": opt_id,
                "sar_image_id": sar_id,
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["status"] == "success"
    assert data["plan"]["task"] == "optical_sar_analysis"
    assert "urban structures" in data["answer"].lower()
    assert len(data["layers"]) >= 2
    assert data["visualization_url"] is not None
    assert data["bounds"] is not None
    assert "optical" in data["images"]
    assert "s1_vv" in data["images"]


def test_2_optical_only_rejected(test_client, uploaded_pair):
    """Test 2: Optical only input rejected with clean 400 validation error."""
    opt_id, _ = uploaded_pair

    response = test_client.post(
        "/api/query",
        json={
            "query": "Use the optical and SAR images together to identify built-up areas.",
            "optical_image_id": opt_id,
        },
    )

    assert response.status_code == 400
    assert "SAR input is required" in response.json()["detail"]


def test_3_sar_only_rejected(test_client, uploaded_pair):
    """Test 3: SAR only input rejected with clean 400 validation error."""
    _, sar_id = uploaded_pair

    response = test_client.post(
        "/api/query",
        json={
            "query": "Use the optical and SAR images together to identify built-up areas.",
            "sar_image_id": sar_id,
        },
    )

    assert response.status_code == 400
    assert "Optical input is required" in response.json()["detail"]


def test_4_invalid_image_reference_rejected(test_client, uploaded_pair):
    """Test 4: Non-existent image reference rejected with clean 400."""
    _, sar_id = uploaded_pair

    response = test_client.post(
        "/api/query",
        json={
            "query": "Use the optical and SAR images together to identify built-up areas.",
            "optical_image_id": "non_existent_image_12345",
            "sar_image_id": sar_id,
        },
    )

    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_5_normal_ndvi_request_unaffected(test_client):
    """Test 5: Normal NDVI request still works and does NOT invoke Optical-SAR."""
    sample_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    mock_search = MagicMock(return_value={
        "status": "REAL_SUCCESS",
        "images": [
            {
                "bands": {
                    "red": str(sample_dir / "before_red.tif"),
                    "nir": str(sample_dir / "before_nir.tif"),
                }
            }
        ],
    })

    with patch("app.agent.executor.get_tool") as mock_get_tool:
        # Default mock tools
        calc_mock = MagicMock(return_value={
            "index": "NDVI",
            "mean": 0.45,
            "min_value": 0.1,
            "max_value": 0.8,
            "valid_pixels": 1000,
            "total_pixels": 1000,
            "data": np.zeros((10, 10), dtype=np.float32),
        })

        def side_effect(tool_name):
            if tool_name == "search_imagery":
                return mock_search
            return calc_mock

        mock_get_tool.side_effect = side_effect

        response = test_client.post(
            "/api/query",
            json={"query": "Calculate NDVI for this area."},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["plan"]["task"] == "vegetation_index"


def test_6_normal_ndwi_request_unaffected(test_client):
    """Test 6: Normal NDWI query is routed to water_index."""
    from app.agent.parser import parse_query
    from app.schemas.query import QueryRequest

    plan = parse_query(QueryRequest(query="Calculate NDWI."))
    assert plan.task == "water_index"


def test_7_normal_change_request_unaffected(test_client):
    """Test 7: Normal temporal change query is routed to change_detection."""
    from app.agent.parser import parse_query
    from app.schemas.query import QueryRequest

    plan = parse_query(QueryRequest(query="What changed between these two dates?"))
    assert plan.task in ("change_detection", "general_change_detection")



def test_8_optical_sar_with_explicit_scientific_index(test_client):
    """Test 8: Query with both words but asking for NDVI routes to NDVI."""
    from app.agent.parser import parse_query
    from app.schemas.query import QueryRequest

    plan = parse_query(QueryRequest(query="Calculate NDVI using the optical and SAR images."))
    assert plan.task == "vegetation_index"


def test_9_invalid_non_raster_upload_rejected(test_client):
    """Test 9: Uploading a non-raster file is rejected with 400."""
    fake_txt = b"This is not a GeoTIFF raster file."

    response = test_client.post(
        "/api/upload/image",
        files={"file": ("notes.txt", fake_txt, "text/plain")},
    )

    assert response.status_code == 400
    assert "not a valid geotiff" in response.json()["detail"].lower()


def test_10_security_arbitrary_path_injection_blocked(test_client, uploaded_pair):
    """Test 10: Security check rejects path traversal or paths outside allowed directories."""
    _, sar_id = uploaded_pair

    # Traversal attempt
    response = test_client.post(
        "/api/query",
        json={
            "query": "Use the optical and SAR images together to identify built-up areas.",
            "optical_image_id": "../../../../etc/passwd",
            "sar_image_id": sar_id,
        },
    )
    assert response.status_code == 400
    assert "security error" in response.json()["detail"].lower()

    # Absolute unauthorized path attempt
    response_abs = test_client.post(
        "/api/query",
        json={
            "query": "Use the optical and SAR images together to identify built-up areas.",
            "optical_image_id": "C:\\Windows\\System32\\cmd.exe",
            "sar_image_id": sar_id,
        },
    )
    assert response_abs.status_code == 400
    assert "security error" in response_abs.json()["detail"].lower()


def test_11_multipart_direct_upload_optical_sar(test_client, sample_optical_bytes, sample_sar_bytes):
    """Test 11: Direct multipart upload endpoint /api/upload/optical-sar."""
    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = "Multimodal analysis successful: observed water bodies with low backscatter."

    with patch("app.vlm.model.VLM", return_value=mock_vlm):
        response = test_client.post(
            "/api/upload/optical-sar",
            files={
                "optical_image": ("optical_scene.tif", sample_optical_bytes, "image/tiff"),
                "sar_image": ("sar_scene.tif", sample_sar_bytes, "image/tiff"),
            },
            data={"query": "Use the optical and SAR images together to identify water-covered regions."},
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert data["plan"]["task"] == "optical_sar_analysis"
    assert "water bodies" in data["answer"].lower()
    assert len(data["layers"]) >= 2


def test_12_positional_image_ids_fallback(test_client, uploaded_pair):
    """Test 12: image_ids list fallback in QueryRequest works cleanly."""
    opt_id, sar_id = uploaded_pair

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = "Multimodal reasoning completed via image_ids."

    with patch("app.vlm.model.VLM", return_value=mock_vlm):
        response = test_client.post(
            "/api/query",
            json={
                "query": "Analyze the optical and SAR images together and describe the major land-cover patterns.",
                "image_ids": [opt_id, sar_id],
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert data["plan"]["task"] == "optical_sar_analysis"
    assert "multimodal reasoning completed" in data["answer"].lower()
