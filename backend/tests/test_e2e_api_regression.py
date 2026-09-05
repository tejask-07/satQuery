"""
End-to-End API Regression Test Suite (Step 19).

Validates the complete pipeline:
API request -> parser -> planner -> registry -> executor -> specialist/tool -> AnalysisResult -> API response

Covers the 7 operational pathways:
1. Single-Image Optical VQA
2. Single-Image SAR VQA
3. Optical-SAR Multimodal Analysis
4. Scientific Index (NDVI / NDWI / NDBI)
5. Temporal Change Detection
6. Automatic Optical + SAR Acquisition
7. Uploaded Image Path Resolution

Also validates:
- Routing collision matrix (A through F)
- Canonical AnalysisResult contract consistency
- Zero filesystem path or secret leakage (recursive inspection)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi.testclient import TestClient
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from app.main import app
from app.schemas.query import QueryRequest, QueryPlan
from app.agent.parser import parse_query
from app.remote_sensing.multimodal.ingestion import store_uploaded_raster


# ============================================================
# RECURSIVE SECURITY SCANNER
# ============================================================

FORBIDDEN_SUBSTRINGS = [
    "C:\\Users\\",
    "C:/Users/",
    "/home/",
    "/mnt/data",
    "data/uploads",
    "data\\uploads",
    "backend/data/cache",
    "backend\\data\\cache",
    "HF_TOKEN",
    "hf_",
    "api_key",
    "secret_key",
]


def assert_no_sensitive_leakage(obj: Any, path: str = ""):
    """
    Recursively walk any dictionary, list, or primitive and verify that
    no server paths or secret credentials appear in keys or string values.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            for forbidden in FORBIDDEN_SUBSTRINGS:
                assert forbidden not in str(k), f"Security violation: found '{forbidden}' in key '{path}.{k}'"
            assert_no_sensitive_leakage(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            assert_no_sensitive_leakage(item, f"{path}[{idx}]")
    elif isinstance(obj, str):
        # Ignore common safe substrings
        for forbidden in FORBIDDEN_SUBSTRINGS:
            assert forbidden not in obj, f"Security violation: found '{forbidden}' in value at '{path}': '{obj}'"


# ============================================================
# FIXTURES & TEST GEOTIFF GENERATORS
# ============================================================

def _generate_geotiff_bytes(
    count: int = 3,
    width: int = 40,
    height: int = 40,
    dtype: str = "uint16",
    crs_epsg: int = 32633,
) -> bytes:
    """Generate lightweight in-memory GeoTIFF bytes."""
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
def sample_optical_id():
    b = _generate_geotiff_bytes(count=3, dtype="uint16")
    return store_uploaded_raster(b, "test_e2e_optical.tif", modality_hint="optical")


@pytest.fixture
def sample_sar_id():
    b = _generate_geotiff_bytes(count=2, dtype="float32")
    return store_uploaded_raster(b, "test_e2e_sar.tif", modality_hint="sar")


# ============================================================
# PATHWAY 1: SINGLE-IMAGE OPTICAL VQA
# ============================================================

def test_pathway_1_single_image_optical_vqa(test_client, sample_optical_id):
    """
    Verify complete pipeline for single-image optical VQA:
    - task = single_image_vqa
    - correct image resolved
    - VQA specialist executed
    - answer returned
    - evidence returned with single_image_observation
    - confidence is None
    - execution trace has 6 lifecycle milestones
    - zero server path or secret leakage
    """
    mock_answer = "A dense agricultural landscape with active crop cultivation."
    mock_res = {
        "task": "single_image_vqa",
        "question": "What type of landscape is visible in this image?",
        "answer": mock_answer,
        "modality": "optical",
        "confidence": None,
    }

    with patch("app.vlm.vqa.run_vqa", return_value=mock_res) as mock_run:
        response = test_client.post(
            "/api/query",
            json={
                "query": "What type of landscape is visible in this image?",
                "optical_image_id": sample_optical_id,
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["status"] == "success"
    assert data["plan"]["task"] == "single_image_vqa"
    assert data["answer"] == mock_answer
    assert data["confidence"] is None

    # Statistics validation
    assert data["statistics"]["task"] == "single_image_vqa"
    assert data["statistics"]["modality"] == "optical"
    assert data["statistics"]["confidence"] is None

    # Evidence contract validation
    assert len(data["evidence"]) == 1
    ev = data["evidence"][0]
    assert ev["source"] == "single_image_observation"
    assert ev["modality"] == "optical"
    assert ev["evidence_used"] is False

    # Execution trace lifecycle validation
    trace = data["execution_trace"]
    assert len(trace) == 6
    assert trace[0] == "Natural-language query received for visual inspection"
    assert trace[1] == "Natural-language query classified as single-image VQA"
    assert trace[2] == "Execution plan created for single-image VQA"
    assert trace[3] == "Single image resolved (modality: optical)"
    assert trace[4] == "VQA specialist executed with grounded prompt"
    assert trace[5] == "VQA response formatted"

    # Security scan
    assert_no_sensitive_leakage(data)


# ============================================================
# PATHWAY 2: SINGLE-IMAGE SAR VQA
# ============================================================

def test_pathway_2_single_image_sar_vqa(test_client, sample_sar_id):
    """
    Verify complete pipeline for single-image SAR VQA:
    - task = single_image_vqa
    - modality = sar
    - SAR semantics preserved
    - confidence is None
    - evidence contains modality sar
    - zero server path leakage
    """
    mock_answer = "High radar backscatter indicates double-bounce reflections from built structures."
    mock_res = {
        "task": "single_image_vqa",
        "question": "What does the SAR image suggest about the observed surface?",
        "answer": mock_answer,
        "modality": "sar",
        "confidence": None,
    }

    with patch("app.vlm.vqa.run_vqa", return_value=mock_res):
        response = test_client.post(
            "/api/query",
            json={
                "query": "What does the SAR image suggest about the observed surface?",
                "sar_image_id": sample_sar_id,
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["status"] == "success"
    assert data["plan"]["task"] == "single_image_vqa"
    assert data["answer"] == mock_answer
    assert data["confidence"] is None
    assert data["statistics"]["modality"] == "sar"

    # Evidence and trace
    assert data["evidence"][0]["source"] == "single_image_observation"
    assert data["evidence"][0]["modality"] == "sar"
    assert data["evidence"][0]["evidence_used"] is False
    assert data["execution_trace"][3] == "Single image resolved (modality: sar)"

    # Security scan
    assert_no_sensitive_leakage(data)


# ============================================================
# PATHWAY 3: OPTICAL-SAR MULTIMODAL
# ============================================================

def test_pathway_3_optical_sar_multimodal(test_client, sample_optical_id, sample_sar_id):
    """
    Verify complete pipeline for Optical-SAR multimodal analysis:
    - task = optical_sar_analysis
    - Optical-SAR specialist is selected
    - optical and SAR layers exist
    - modalities remain distinct
    - no VQA specialist hijacking
    - zero path leakage
    """
    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = "Optical reveals green vegetation, while SAR VV backscatter identifies metallic bridge pylons."

    with patch("app.vlm.model.VLM", return_value=mock_vlm), patch("app.vlm.vqa.run_vqa") as mock_vqa:
        response = test_client.post(
            "/api/query",
            json={
                "query": "Compare the optical and SAR images and explain what each modality contributes.",
                "optical_image_id": sample_optical_id,
                "sar_image_id": sample_sar_id,
            },
        )

        # Ensure VQA specialist was NOT hijacked
        assert not mock_vqa.called

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["status"] == "success"
    assert data["plan"]["task"] == "optical_sar_analysis"
    assert "optical reveals green vegetation" in data["answer"].lower()
    assert data["confidence"] == 0.9
    assert len(data["layers"]) >= 2
    assert "optical" in data["images"]
    assert "s1_vv" in data["images"]

    # Modalities distinct in evidence
    assert data["evidence"][0]["optical_sar_pair"]["source"] in ("user_supplied", "manual")

    # Security scan
    assert_no_sensitive_leakage(data)


# ============================================================
# PATHWAY 4: SCIENTIFIC INDEX
# ============================================================

def test_pathway_4_scientific_index(test_client):
    """
    Verify complete pipeline for scientific index calculation:
    - query: 'What is the NDVI of this image?'
    - task = vegetation_index (NOT VQA)
    - NDVI calculation is executed
    - numerical statistics preserved
    - zero path leakage
    """
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

    with patch("app.agent.executor.get_tool") as mock_get_tool, patch("app.vlm.vqa.run_vqa") as mock_vqa:
        calc_mock = MagicMock(return_value={
            "index": "NDVI",
            "mean": 0.52,
            "min_value": 0.05,
            "max_value": 0.88,
            "valid_pixels": 2500,
            "total_pixels": 2500,
            "data": np.zeros((10, 10), dtype=np.float32),
        })

        def side_effect(tool_name):
            if tool_name == "search_imagery":
                return mock_search
            return calc_mock

        mock_get_tool.side_effect = side_effect

        response = test_client.post(
            "/api/query",
            json={
                "query": "What is the NDVI of this image?",
                "aoi": [73.80, 18.50, 73.86, 18.56],
            },
        )

        assert not mock_vqa.called

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["status"] == "success"
    assert data["plan"]["task"] == "vegetation_index"
    assert data["plan"]["metric"] == "ndvi"
    assert data["statistics"]["mean"] == 0.52
    assert data["statistics"]["valid_pixels"] == 2500

    # Security scan
    assert_no_sensitive_leakage(data)


# ============================================================
# PATHWAY 5: TEMPORAL CHANGE
# ============================================================

def test_pathway_5_temporal_change(test_client):
    """
    Verify complete pipeline for temporal change detection:
    - query: 'What changed between these two images?'
    - task in (change_detection, general_change_detection)
    - no generic VQA hijacking
    - zero path leakage
    """
    sample_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    mock_search = MagicMock(return_value={
        "status": "REAL_SUCCESS",
        "images": [
            {
                "bands": {
                    "red": str(sample_dir / "before_red.tif"),
                    "nir": str(sample_dir / "before_nir.tif"),
                    "green": str(sample_dir / "before_red.tif"),
                    "swir": str(sample_dir / "before_nir.tif"),
                    "swir1": str(sample_dir / "before_nir.tif"),
                    "swir2": str(sample_dir / "before_nir.tif"),
                }
            },
            {
                "bands": {
                    "red": str(sample_dir / "after_red.tif"),
                    "nir": str(sample_dir / "after_nir.tif"),
                    "green": str(sample_dir / "after_red.tif"),
                    "swir": str(sample_dir / "after_nir.tif"),
                    "swir1": str(sample_dir / "after_nir.tif"),
                    "swir2": str(sample_dir / "after_nir.tif"),
                }
            },
        ],
    })

    with patch("app.agent.executor.get_tool") as mock_get_tool, patch("app.vlm.vqa.run_vqa") as mock_vqa:
        mock_arr = np.zeros((10, 10), dtype=np.float32)
        temporal_idx_mock = MagicMock(return_value={
            "index": "NDVI",
            "ndvi_before": mock_arr,
            "ndvi_after": mock_arr,
            "mean_ndvi_before": 0.60,
            "mean_ndvi_after": 0.40,
            "mean_ndvi_change": -0.20,
            "valid_mask": np.ones((10, 10), dtype=bool),
            "valid_pixels": 100,
            "total_pixels": 100,
        })
        change_mock = MagicMock(return_value={
            "index": "NDVI",
            "mean_before": 0.60,
            "mean_after": 0.40,
            "mean_change": -0.20,
            "change_ratio": 0.15,
            "regions_detected": 3,
            "changed_pixels": 450,
            "valid_pixels": 3000,
            "change_map": np.zeros((10, 10), dtype=np.float32),
        })

        def side_effect(tool_name):
            if tool_name == "search_imagery":
                return mock_search
            if "temporal" in tool_name:
                return temporal_idx_mock
            return change_mock

        mock_get_tool.side_effect = side_effect

        response = test_client.post(
            "/api/query",
            json={
                "query": "What changed between these two images?",
                "aoi": [16.40, 48.20, 16.41, 48.21],
                "time_start": "2021-01-01",
                "time_end": "2025-01-01",
            },
        )

        assert not mock_vqa.called

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["status"] == "success"
    assert data["plan"]["task"] in ("change_detection", "general_change_detection")
    assert_no_sensitive_leakage(data)


# ============================================================
# PATHWAY 6: AUTOMATIC OPTICAL + SAR ACQUISITION
# ============================================================

def test_pathway_6_automatic_optical_sar_acquisition(test_client, tmp_path):
    """
    Verify complete pipeline for automatic Optical-SAR acquisition:
    - query: 'Compare optical and SAR for urban areas' with AOI and time
    - triggers automatic Sentinel-2 + Sentinel-1 acquisition/pairing
    - pair selection, co-registration, and specialist output verified
    - zero path leakage
    """
    # Create mock rasters on disk
    def _create_raster(p, count=1, val=100):
        transform = from_bounds(13.0, 48.0, 13.02, 48.02, 20, 20)
        with rasterio.open(
            p, "w", driver="GTiff", height=20, width=20, count=count,
            dtype="uint16", crs=CRS.from_epsg(4326), transform=transform, nodata=0
        ) as dst:
            for b in range(1, count + 1):
                dst.write(np.full((20, 20), val * b, dtype=np.uint16), b)
        return str(p.resolve())

    opt_path = _create_raster(tmp_path / "auto_opt.tif", count=3, val=120)
    sar_vv = _create_raster(tmp_path / "auto_vv.tif", count=1, val=220)
    sar_vh = _create_raster(tmp_path / "auto_vh.tif", count=1, val=40)

    mock_pairing_result = {
        "status": "REAL_SUCCESS",
        "pair_found": True,
        "temporal_delta_days": 0.42,
        "spatial_overlap": {
            "has_overlap": True,
            "intersection_bbox": [13.0, 48.0, 13.02, 48.02],
            "aoi_coverage_percent": 100.0,
        },
        "selection_reason": "Selected Optical S2A and SAR S1B with optimal temporal delta 0.42 days",
        "optical": {
            "item_id": "S2A_auto_mock",
            "acquisition_datetime": "2021-06-25T10:00:00Z",
            "sensor": "Sentinel-2A",
            "product": "Level-2A",
            "cloud_cover": 0.5,
            "coverage_percentage": 100.0,
            "crs": "EPSG:4326",
            "bounds": [13.0, 48.0, 13.02, 48.02],
            "path": opt_path,
        },
        "sar": {
            "item_id": "S1B_auto_mock",
            "acquisition_datetime": "2021-06-25T20:00:00Z",
            "sensor": "Sentinel-1B",
            "product": "GRD",
            "mode": "IW",
            "orbit_direction": "ascending",
            "polarizations": ["VV", "VH"],
            "coverage_percentage": 100.0,
            "crs": "EPSG:4326",
            "bounds": [13.0, 48.0, 13.02, 48.02],
            "path": sar_vv,
            "polarization_paths": {"VV": sar_vv, "VH": sar_vh},
        },
    }

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = "Co-registered automatic acquisition confirms urban development."

    with (
        patch("app.agent.executor.find_optical_sar_pair", return_value=mock_pairing_result),
        patch("app.vlm.model.VLM", return_value=mock_vlm),
    ):
        response = test_client.post(
            "/api/query",
            json={
                "query": "Compare optical and SAR for urban areas",
                "aoi": [13.0, 48.0, 13.02, 48.02],
                "time_start": "2021-06-01",
                "time_end": "2021-06-30",
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["status"] == "success"
    assert data["plan"]["task"] == "optical_sar_analysis"
    assert "urban development" in data["answer"].lower()

    # Pair metadata validation
    pair_info = data["evidence"][0]["optical_sar_pair"]
    assert pair_info["source"] == "automatic"
    assert pair_info["temporal_delta_days"] == 0.42

    # Trace validation
    trace_text = " ".join(data["execution_trace"])
    assert "Automatic Optical-SAR mode triggered" in trace_text
    assert "0.42 days" in trace_text

    # Security scan
    assert_no_sensitive_leakage(data)


# ============================================================
# PATHWAY 7: UPLOADED IMAGE PATH
# ============================================================

def test_pathway_7_uploaded_image_path(test_client):
    """
    Verify complete upload and reference workflow:
    1. Upload image via /api/upload/image
    2. Query with returned image_id
    3. Verify VQA execution and result
    4. Verify actual server path is NOT exposed
    """
    raw_geotiff = _generate_geotiff_bytes(count=3, width=32, height=32, dtype="uint16")

    upload_resp = test_client.post(
        "/api/upload/image",
        files={"file": ("e2e_survey.tif", raw_geotiff, "image/tiff")},
        data={"modality": "optical"},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    uploaded_id = upload_resp.json()["image_id"]
    assert uploaded_id is not None

    mock_res = {
        "task": "single_image_vqa",
        "question": "What is visible in this image?",
        "answer": "A coastal harbor area with boats and docks.",
        "modality": "optical",
        "confidence": None,
    }

    with patch("app.vlm.vqa.run_vqa", return_value=mock_res):
        response = test_client.post(
            "/api/query",
            json={
                "query": "What is visible in this image?",
                "image_ids": [uploaded_id],
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["status"] == "success"
    assert data["plan"]["task"] == "single_image_vqa"
    assert data["answer"] == "A coastal harbor area with boats and docks."
    assert data["confidence"] is None
    assert data["evidence"][0]["source"] == "single_image_observation"

    # Security scan: ensure upload store filesystem path is not exposed
    assert_no_sensitive_leakage(data)


# ============================================================
# ROUTING COLLISION MATRIX TESTS
# ============================================================

@pytest.mark.parametrize(
    "query,expected_task,expected_metric,expected_modality",
    [
        # Collision A: 'What is the NDVI?' -> scientific, NOT VQA
        ("What is the NDVI?", "vegetation_index", "ndvi", "optical"),
        ("What is the NDVI of this image?", "vegetation_index", "ndvi", "optical"),
        # Collision B: 'What changed between these images?' -> temporal/change, NOT VQA
        ("What changed between these images?", "general_change_detection", None, "multispectral"),
        ("What changed between the two images?", "general_change_detection", None, "multispectral"),
        # Collision C: 'Compare optical and SAR.' -> optical_sar_analysis, NOT VQA
        ("Compare optical and SAR.", "optical_sar_analysis", None, "optical"),
        ("Compare the optical and SAR images and explain what each modality contributes.", "optical_sar_analysis", None, "optical"),
        # Collision D: 'What do you see in this image?' -> single_image_vqa
        ("What do you see in this image?", "single_image_vqa", None, "unknown"),
        ("What type of landscape is visible in this image?", "single_image_vqa", None, "unknown"),
        # Collision E: 'What does the SAR image show?' -> single_image_vqa with SAR modality
        ("What does the SAR image show?", "single_image_vqa", None, "sar"),
        ("What does the SAR image suggest about the observed surface?", "single_image_vqa", None, "sar"),
        # Collision F: 'Calculate vegetation index and explain what you see.' -> scientific
        ("Calculate vegetation index and explain what you see.", "vegetation_index", "ndvi", "optical"),
    ],
)
def test_routing_collision_matrix(query: str, expected_task: str, expected_metric: str | None, expected_modality: str):
    """
    Verify unambiguous task parsing and prevent cross-task collision/hijacking.
    """
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task == expected_task, f"Failed routing for '{query}': got {plan.task}, expected {expected_task}"
    if expected_metric:
        assert plan.metric == expected_metric
    if expected_modality:
        assert expected_modality in plan.modalities


# ============================================================
# RESULT CONTRACT INTEGRITY VALIDATION
# ============================================================

def test_result_contract_vqa_fields(test_client, sample_optical_id):
    """Verify VQA AnalysisResult strictly exposes canonical fields."""
    mock_res = {
        "task": "single_image_vqa",
        "question": "What is visible?",
        "answer": "Test answer.",
        "modality": "optical",
        "confidence": None,
    }

    with patch("app.vlm.vqa.run_vqa", return_value=mock_res):
        res = test_client.post(
            "/api/query",
            json={"query": "What is visible?", "optical_image_id": sample_optical_id},
        )

    data = res.json()
    # Required canonical fields
    assert "status" in data
    assert "answer" in data
    assert "confidence" in data
    assert "plan" in data
    assert "statistics" in data
    assert "evidence" in data
    assert "execution_trace" in data

    # VQA contract requirements
    assert data["confidence"] is None
    assert isinstance(data["evidence"], list)
    assert len(data["evidence"]) >= 1
    assert data["evidence"][0]["source"] == "single_image_observation"


def test_result_contract_optical_sar_fields(test_client, sample_optical_id, sample_sar_id):
    """Verify Optical-SAR AnalysisResult exposes canonical fields."""
    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = "Test Optical-SAR answer."

    with patch("app.vlm.model.VLM", return_value=mock_vlm):
        res = test_client.post(
            "/api/query",
            json={
                "query": "Compare optical and SAR.",
                "optical_image_id": sample_optical_id,
                "sar_image_id": sample_sar_id,
            },
        )

    data = res.json()
    assert data["status"] == "success"
    assert data["plan"]["task"] == "optical_sar_analysis"
    assert data["confidence"] == 0.9
    assert "layers" in data
    assert "images" in data
    assert "visualization_url" in data
    assert "bounds" in data
    assert "optical" in data["images"]
    assert "s1_vv" in data["images"]
