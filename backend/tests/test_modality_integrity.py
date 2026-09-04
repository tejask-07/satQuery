"""
Tests for Step 8: Optical–SAR End-to-End Validation + Modality Integrity.

Covers:
- Step 8D: Modality Integrity (Tests 1-8: Optical+SAR, Optical+Optical, SAR+SAR, Unknown+SAR, Optical+Unknown,
            Explicit Metadata, Single Uploads, Temporal Change).
- Step 8E: Mocked Multimodal End-to-End Execution via FastAPI TestClient.
- Step 8F: Exact Multi-Sensor Payload Verification (both modalities reach VLM).
- Step 8G: Multimodal Reasoning Behavior Contract for Queries A, B, C, D.
- Step 8H: Anti-Hallucination & Uncertainty Prompt Contracts.
- Step 8I: False-Color Optical Contract.
- Step 8J: SAR Physical-Units Safety (No log10/dB conversion).
- Step 8K: Legacy BigEarthNet Isolation.
- Step 8L & 8M: Response Schema Structure, Error Handling, and VLM Fallback.
"""

from __future__ import annotations

import io
import inspect
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
    determine_raster_modality,
    UPLOAD_DIR,
)
from app.vlm.optical_sar import (
    build_optical_sar_prompt,
    run_optical_sar_analysis,
    answer_optical_sar_question,
)
import app.remote_sensing.multimodal.optical_sar as rs_optical_sar
import app.vlm.optical_sar as vlm_optical_sar


# ============================================================
# TEST HELPERS
# ============================================================

def _create_geotiff_bytes(
    count: int = 3,
    width: int = 64,
    height: int = 64,
    dtype: str = "uint16",
    crs_epsg: int = 32633,
    descriptions: list[str] | None = None,
    tags: dict | None = None,
) -> bytes:
    """Generate in-memory GeoTIFF bytes with customizable band names and tags."""
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
        if tags:
            dst.update_tags(**tags)
        for b in range(1, count + 1):
            arr = (np.random.rand(height, width) * 2000 + 500).astype(dtype)
            dst.write(arr, b)
            if descriptions and (b - 1) < len(descriptions):
                dst.set_band_description(b, descriptions[b - 1])

    return bio.getvalue()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def optical_rgb_bytes():
    return _create_geotiff_bytes(
        count=3,
        descriptions=["Red", "Green", "Blue"],
        dtype="uint16",
    )


@pytest.fixture
def sar_vv_vh_bytes():
    return _create_geotiff_bytes(
        count=2,
        descriptions=["VV", "VH"],
        dtype="float32",
    )


@pytest.fixture
def sar_vv_only_bytes():
    return _create_geotiff_bytes(
        count=1,
        descriptions=["VV"],
        dtype="float32",
    )


@pytest.fixture
def ambiguous_raster_bytes():
    # 1 band with no descriptions or tags
    return _create_geotiff_bytes(
        count=1,
        descriptions=None,
        dtype="float32",
    )


# ============================================================
# STEP 8D: MODALITY INTEGRITY TESTS (TESTS 1-8)
# ============================================================

def test_8d_1_valid_optical_and_sar_pair_accepted(optical_rgb_bytes, sar_vv_vh_bytes):
    """Test 1: Valid Optical + SAR pair is accepted and resolved."""
    opt_id = store_uploaded_raster(optical_rgb_bytes, "opt_scene.tif", modality_hint="optical")
    sar_id = store_uploaded_raster(sar_vv_vh_bytes, "sar_scene.tif", modality_hint="sar")

    p_opt, p_sar = resolve_optical_sar_references(optical_ref=opt_id, sar_ref=sar_id)
    assert p_opt.exists()
    assert p_sar.exists()

    mod_opt, _ = determine_raster_modality(p_opt)
    mod_sar, _ = determine_raster_modality(p_sar)
    assert mod_opt == "optical"
    assert mod_sar == "sar"


def test_8d_2_optical_plus_optical_rejected_cleanly(optical_rgb_bytes):
    """Test 2: Optical + Optical pair is rejected with explicit error message."""
    opt1_id = store_uploaded_raster(optical_rgb_bytes, "opt1.tif", modality_hint="optical")
    opt2_id = store_uploaded_raster(optical_rgb_bytes, "opt2.tif", modality_hint="optical")

    with pytest.raises(ValueError) as exc:
        resolve_optical_sar_references(optical_ref=opt1_id, sar_ref=opt2_id)
    assert "Both inputs were identified as Optical imagery" in str(exc.value)


def test_8d_3_sar_plus_sar_rejected_cleanly(sar_vv_vh_bytes):
    """Test 3: SAR + SAR pair is rejected with explicit error message."""
    sar1_id = store_uploaded_raster(sar_vv_vh_bytes, "sar1.tif", modality_hint="sar")
    sar2_id = store_uploaded_raster(sar_vv_vh_bytes, "sar2.tif", modality_hint="sar")

    with pytest.raises(ValueError) as exc:
        resolve_optical_sar_references(optical_ref=sar1_id, sar_ref=sar2_id)
    assert "Both inputs were identified as SAR radar imagery" in str(exc.value)


def test_8d_4_unknown_plus_sar_rejected_unless_declared(ambiguous_raster_bytes, sar_vv_vh_bytes):
    """Test 4: Unknown + SAR rejected unless explicit modality identifies the first as optical."""
    # Stored without modality hint
    unk_id = store_uploaded_raster(ambiguous_raster_bytes, "unlabeled_data.tif")
    sar_id = store_uploaded_raster(sar_vv_vh_bytes, "sar_scene.tif", modality_hint="sar")

    with pytest.raises(ValueError) as exc:
        resolve_optical_sar_references(optical_ref=unk_id, sar_ref=sar_id)
    assert "could not be verified as an optical raster" in str(exc.value)

    # When explicitly declared as optical on ingestion, it is accepted
    opt_declared_id = store_uploaded_raster(ambiguous_raster_bytes, "declared_opt.tif", modality_hint="optical")
    p_opt, p_sar = resolve_optical_sar_references(optical_ref=opt_declared_id, sar_ref=sar_id)
    assert p_opt.exists()
    assert p_sar.exists()


def test_8d_5_optical_plus_unknown_rejected_unless_declared(optical_rgb_bytes, ambiguous_raster_bytes):
    """Test 5: Optical + Unknown rejected unless explicit modality identifies the second as SAR."""
    opt_id = store_uploaded_raster(optical_rgb_bytes, "opt_scene.tif", modality_hint="optical")
    unk_id = store_uploaded_raster(ambiguous_raster_bytes, "unlabeled_data2.tif")

    with pytest.raises(ValueError) as exc:
        resolve_optical_sar_references(optical_ref=opt_id, sar_ref=unk_id)
    assert "could not be verified as a SAR raster" in str(exc.value)

    # When explicitly declared as sar on ingestion, it is accepted
    sar_declared_id = store_uploaded_raster(ambiguous_raster_bytes, "declared_sar.tif", modality_hint="sar")
    p_opt, p_sar = resolve_optical_sar_references(optical_ref=opt_id, sar_ref=sar_declared_id)
    assert p_opt.exists()
    assert p_sar.exists()



def test_8d_6_explicit_modality_metadata_succeeds(ambiguous_raster_bytes):
    """Test 6: Ambiguous rasters succeed when trusted ingestion metadata declares modality."""
    opt_id = store_uploaded_raster(ambiguous_raster_bytes, "custom_opt.tif", modality_hint="optical")
    sar_id = store_uploaded_raster(ambiguous_raster_bytes, "custom_sar.tif", modality_hint="sar")

    p_opt, p_sar = resolve_optical_sar_references(optical_ref=opt_id, sar_ref=sar_id)
    assert p_opt.exists()
    assert p_sar.exists()


def test_8d_7_existing_single_image_uploads_remain_valid(client, optical_rgb_bytes):
    """Test 7: Ordinary single-image uploads remain valid for existing workflows."""
    response = client.post(
        "/api/upload/image",
        files={"file": ("single_scene.tif", optical_rgb_bytes, "image/tiff")},
        data={"modality": "optical"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["image_id"] is not None


def test_8d_8_existing_temporal_change_workflow_unaffected(client, optical_rgb_bytes):
    """Test 8: Two optical temporal images remain valid for change detection and are NOT treated as Optical-SAR."""
    response = client.post(
        "/api/upload/analyze",
        files={
            "before_image": ("before.tif", optical_rgb_bytes, "image/tiff"),
            "after_image": ("after.tif", optical_rgb_bytes, "image/tiff"),
        },
        data={"query": "Show vegetation change between these two dates."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "change" in data["plan"]["task"]
    assert data["plan"]["task"] != "optical_sar_analysis"


# ============================================================
# STEP 8E & 8F: MOCKED END-TO-END EXECUTION & PAYLOAD VERIFICATION
# ============================================================

def test_8e_and_8f_vlm_receives_both_modalities_end_to_end(client, optical_rgb_bytes, sar_vv_vh_bytes):
    """
    Step 8E & 8F:
    Executes real path: /api/query -> image resolution -> parser -> planner -> executor
    -> Optical-SAR runner -> alignment -> visual builder -> VLM.
    Verifies that BOTH modalities reach VLM.generate with correct signatures.
    """
    opt_id = store_uploaded_raster(optical_rgb_bytes, "e2e_opt.tif", modality_hint="optical")
    sar_id = store_uploaded_raster(sar_vv_vh_bytes, "e2e_sar.tif", modality_hint="sar")

    mock_vlm = MagicMock()
    captured_kwargs = {}

    def capture_generate(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return "Co-registered synthesis: Urban area confirmed by double-bounce SAR."

    mock_vlm.generate.side_effect = capture_generate

    with patch("app.vlm.model.VLM", return_value=mock_vlm):
        response = client.post(
            "/api/query",
            json={
                "query": "Use the optical and SAR images together to identify built-up areas.",
                "optical_image_id": opt_id,
                "sar_image_id": sar_id,
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()

    # Verify response structure
    assert data["status"] == "success"
    assert data["plan"]["task"] == "optical_sar_analysis"
    assert "urban area confirmed" in data["answer"].lower()
    assert data["visualization_url"] is not None

    # Verify VLM received both modalities
    assert "image" in captured_kwargs, "Primary optical image must be passed as 'image'"
    assert "images" in captured_kwargs, "SAR images must be passed as 'images' dictionary"

    sar_images = captured_kwargs["images"]
    assert "s1_vv" in sar_images, "VV band must be passed to VLM"
    assert "s1_vh" in sar_images, "VH band must be passed to VLM"
    assert "s1_composite" in sar_images, "SAR composite must be passed to VLM"

    # Verify PIL Image types
    from PIL import Image
    assert isinstance(captured_kwargs["image"], Image.Image)
    assert isinstance(sar_images["s1_vv"], Image.Image)
    assert isinstance(sar_images["s1_vh"], Image.Image)
    assert isinstance(sar_images["s1_composite"], Image.Image)

    # Verify user question is forwarded
    assert "built-up areas" in captured_kwargs["question"]


def test_8f_single_band_sar_does_not_fabricate_vh(optical_rgb_bytes, sar_vv_only_bytes):
    """Step 8F: Verify VV-only SAR passes VV and does NOT fabricate VH."""
    opt_id = store_uploaded_raster(optical_rgb_bytes, "vv_opt.tif", modality_hint="optical")
    sar_id = store_uploaded_raster(sar_vv_only_bytes, "vv_only_sar.tif", modality_hint="sar")

    opt_p = resolve_image_reference(opt_id)
    sar_p = resolve_image_reference(sar_id)

    mock_vlm = MagicMock()
    captured_kwargs = {}

    def capture_generate(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return "Single-band VV interpretation complete."

    mock_vlm.generate.side_effect = capture_generate

    res = run_optical_sar_analysis(
        optical_path=str(opt_p),
        sar_path=str(sar_p),
        question="Describe SAR backscatter.",
        vlm=mock_vlm,
    )

    assert res["success"] is True
    sar_images = captured_kwargs["images"]
    assert "s1_vv" in sar_images
    assert "s1_vh" not in sar_images, "VH must not be fabricated when missing from input"


# ============================================================
# STEP 8G: BEHAVIORAL CONTRACT TESTS FOR QUERIES A-D
# ============================================================

def test_8g_query_a_built_up_areas_contract(optical_rgb_bytes, sar_vv_vh_bytes):
    """Query A: 'Use the optical and SAR images together to identify built-up areas.'"""
    opt_p = resolve_image_reference(store_uploaded_raster(optical_rgb_bytes, "qa_opt.tif", modality_hint="optical"))
    sar_p = resolve_image_reference(store_uploaded_raster(sar_vv_vh_bytes, "qa_sar.tif", modality_hint="sar"))

    mock_vlm = MagicMock()
    captured_kwargs = {}
    mock_vlm.generate.side_effect = lambda **kw: captured_kwargs.update(kw) or "Answer A"

    run_optical_sar_analysis(
        optical_path=str(opt_p),
        sar_path=str(sar_p),
        question="Use the optical and SAR images together to identify built-up areas.",
        vlm=mock_vlm,
    )

    prompt = captured_kwargs["question"]
    assert "double-bounce backscatter" in prompt
    assert "Optical provides surface solar reflectance" in prompt
    assert "SAR provides microwave backscatter" in prompt


def test_8g_query_b_water_covered_regions_contract(optical_rgb_bytes, sar_vv_vh_bytes):
    """Query B: 'Use optical and SAR imagery to identify water-covered regions.'"""
    opt_p = resolve_image_reference(store_uploaded_raster(optical_rgb_bytes, "qb_opt.tif", modality_hint="optical"))
    sar_p = resolve_image_reference(store_uploaded_raster(sar_vv_vh_bytes, "qb_sar.tif", modality_hint="sar"))

    mock_vlm = MagicMock()
    captured_kwargs = {}
    mock_vlm.generate.side_effect = lambda **kw: captured_kwargs.update(kw) or "Answer B"

    run_optical_sar_analysis(
        optical_path=str(opt_p),
        sar_path=str(sar_p),
        question="Use optical and SAR imagery to identify water-covered regions.",
        vlm=mock_vlm,
    )

    prompt = captured_kwargs["question"]
    assert "Calm water bodies typically act as specular reflectors" in prompt
    assert "dark in VV/VH" in prompt


def test_8g_query_c_sar_complementary_information_contract():
    """Query C: 'What information does SAR provide that is less apparent in the optical image?'"""
    prompt = build_optical_sar_prompt(
        question="What information does SAR provide that is less apparent in the optical image?",
        optical_metadata={"is_false_color": False, "description": "RGB"},
        available_sar_modalities=["sar_vv", "sar_vh", "sar_composite"],
    )
    assert "Complementary Evidence:" in prompt
    assert "surface roughness, geometric structure, and moisture" in prompt


def test_8g_query_d_compare_vv_and_vh_evidence():
    """Query D: 'Compare the evidence from VV and VH for likely urban regions.'"""
    prompt = build_optical_sar_prompt(
        question="Compare the evidence from VV and VH for likely urban regions.",
        optical_metadata={"is_false_color": False, "description": "RGB"},
        available_sar_modalities=["sar_vv", "sar_vh", "sar_composite"],
    )
    assert "VV co-polarization" in prompt
    assert "VH cross-polarization" in prompt
    assert "VV/VH dual-polarization composite" in prompt


# ============================================================
# STEP 8H: ANTI-HALLUCINATION CONTRACT
# ============================================================

def test_8h_anti_hallucination_safeguards():
    """Step 8H: Verify prompt forbids invented dB, percentages, areas, pixel counts, and color claims."""
    prompt = build_optical_sar_prompt(
        question="What are the land covers?",
        optical_metadata={"is_false_color": False, "description": "RGB"},
        available_sar_modalities=["sar_vv", "sar_vh"],
    )

    # Specific anti-hallucination rules
    assert "Do not describe radar channels as ordinary visible light colors" in prompt
    assert "Do NOT invent numerical backscatter measurements" in prompt
    assert "exact percentages" in prompt
    assert "areas" in prompt
    assert "pixel counts" in prompt
    assert "dB values" in prompt
    assert "distinguish direct visual observations from inferred semantic land-cover interpretations" in prompt
    assert "clearly state the uncertainty" in prompt


# ============================================================
# STEP 8I: FALSE-COLOR OPTICAL CONTRACT
# ============================================================

def test_8i_false_color_optical_contract():
    """Step 8I: Verify false-color optical input is explicitly identified to prevent true-color confusion."""
    # False-color prompt
    fc_prompt = build_optical_sar_prompt(
        question="Analyze land cover.",
        optical_metadata={"is_false_color": True, "description": "NIR false-color", "bands_used": ["NIR", "Red", "Green"]},
        available_sar_modalities=["sar_vv", "sar_vh"],
    )
    assert "False-color composite" in fc_prompt
    assert "NOTE: This optical image is a false-color representation, not true visible light." in fc_prompt

    # True-color prompt
    tc_prompt = build_optical_sar_prompt(
        question="Analyze land cover.",
        optical_metadata={"is_false_color": False, "description": "True-color RGB"},
        available_sar_modalities=["sar_vv"],
    )
    assert "True-color RGB" in tc_prompt
    assert "NOTE: This optical image represents true-color visible reflectance." in tc_prompt


# ============================================================
# STEP 8J: SAR PHYSICAL-UNITS SAFETY (NO LOG10/DB CONVERSION)
# ============================================================

def test_8j_sar_physical_units_safety():
    """Step 8J: Verify zero uncalibrated dB / log10 conversion exists in multimodal pipeline."""
    rs_source = inspect.getsource(rs_optical_sar)
    vlm_source = inspect.getsource(vlm_optical_sar)

    assert "log10" not in rs_source, "No log10 conversion allowed in remote_sensing.multimodal.optical_sar"
    assert "10 * log" not in rs_source
    assert "log10" not in vlm_source, "No log10 conversion allowed in vlm.optical_sar"


# ============================================================
# STEP 8K: LEGACY BIGEARTHNET ISOLATION
# ============================================================

def test_8k_legacy_bigearthnet_isolation(optical_rgb_bytes, sar_vv_vh_bytes):
    """Step 8K: Verify optical_sar_analysis never calls legacy BigEarthNet functions."""
    opt_p = resolve_image_reference(store_uploaded_raster(optical_rgb_bytes, "k_opt.tif", modality_hint="optical"))
    sar_p = resolve_image_reference(store_uploaded_raster(sar_vv_vh_bytes, "k_sar.tif", modality_hint="sar"))

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = "Isolated analysis complete."

    with patch("app.vlm.bigearthnet.remote_s1.load_s1_bands") as mock_legacy_load, \
         patch("app.vlm.bigearthnet.s1_p4.build_s1_visualization") as mock_legacy_vis:

        res = run_optical_sar_analysis(
            optical_path=str(opt_p),
            sar_path=str(sar_p),
            question="Analyze area.",
            vlm=mock_vlm,
        )

        assert res["success"] is True
        mock_legacy_load.assert_not_called()
        mock_legacy_vis.assert_not_called()


# ============================================================
# STEP 8L & 8M: RESPONSE SCHEMA & FALLBACK HANDLING
# ============================================================

def test_8l_response_structure_completeness(client, optical_rgb_bytes, sar_vv_vh_bytes):
    """Step 8L: Verify full response schema structure and output fields."""
    opt_id = store_uploaded_raster(optical_rgb_bytes, "resp_opt.tif", modality_hint="optical")
    sar_id = store_uploaded_raster(sar_vv_vh_bytes, "resp_sar.tif", modality_hint="sar")

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = "Detailed response with all layers."

    with patch("app.vlm.model.VLM", return_value=mock_vlm):
        response = client.post(
            "/api/query",
            json={
                "query": "Use the optical and SAR images together to identify built-up areas.",
                "optical_image_id": opt_id,
                "sar_image_id": sar_id,
            },
        )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert body["plan"]["task"] == "optical_sar_analysis"
    assert body["answer"] is not None
    assert isinstance(body["layers"], list)
    assert len(body["layers"]) >= 3  # Optical, VV, VH
    assert body["visualization_url"] is not None
    assert body["bounds"] is not None
    assert isinstance(body["execution_trace"], list)
    assert len(body["execution_trace"]) >= 4
    assert body["statistics"]["modalities"] is not None


def test_8m_vlm_failure_fallback_does_not_crash(client, optical_rgb_bytes, sar_vv_vh_bytes):
    """Step 8M: When VLM fails, verify clean fallback without crashing FastAPI."""
    opt_id = store_uploaded_raster(optical_rgb_bytes, "fb_opt.tif", modality_hint="optical")
    sar_id = store_uploaded_raster(sar_vv_vh_bytes, "fb_sar.tif", modality_hint="sar")

    mock_vlm = MagicMock()
    mock_vlm.generate.side_effect = RuntimeError("HuggingFace API timeout")

    with patch("app.vlm.model.VLM", return_value=mock_vlm):
        response = client.post(
            "/api/query",
            json={
                "query": "Use the optical and SAR images together to identify built-up areas.",
                "optical_image_id": opt_id,
                "sar_image_id": sar_id,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "deterministic multimodal analysis summary" in body["answer"].lower()
    assert body["visualization_url"] is not None

