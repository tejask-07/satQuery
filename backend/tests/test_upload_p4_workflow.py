"""
Comprehensive Test Suite for P4 Step 3: User Upload Analysis -> P4 VLM Integration.

Verifies:
  TEST 1: Two multispectral GeoTIFFs -> NDVI change detection + P4 VLM grounding
  TEST 2: Two multispectral GeoTIFFs -> NDWI change detection + P4 VLM grounding
  TEST 3: Two multispectral GeoTIFFs -> NDBI change detection + P4 VLM grounding
  TEST 4: Two RGB images -> No fake NDVI, spectral notice, visual change detection
  TEST 5: One invalid/missing upload -> HTTP 400, clear error, no VLM hallucination
"""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_bounds

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agent.upload_processor import (
    inspect_image,
    parse_upload_query,
    process_upload_analysis,
)
from app.main import app
from app.vlm.model import VLM


def create_multispectral_geotiff_bytes(
    red_val: float,
    green_val: float,
    blue_val: float,
    nir_val: float,
    swir_val: float,
    width: int = 50,
    height: int = 50,
) -> bytes:
    """Create in-memory 5-band GeoTIFF bytes (Blue, Green, Red, NIR, SWIR)."""
    data = np.zeros((5, height, width), dtype=np.float32)
    data[0, :, :] = blue_val
    data[1, :, :] = green_val
    data[2, :, :] = red_val
    data[3, :, :] = nir_val
    data[4, :, :] = swir_val

    transform = from_bounds(151.195, -33.885, 151.225, -33.855, width, height)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=5,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dataset:
            dataset.write(data)
            dataset.set_band_description(1, "Blue")
            dataset.set_band_description(2, "Green")
            dataset.set_band_description(3, "Red")
            dataset.set_band_description(4, "NIR")
            dataset.set_band_description(5, "SWIR")
        return memfile.read()


def create_rgb_image_bytes(
    color: tuple = (100, 150, 200), width: int = 50, height: int = 50
) -> bytes:
    """Create in-memory standard 3-channel visible RGB PNG bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_vlm_create():
    """Mock the external Qwen completions call while validating prompt contents."""
    captured = {"prompt": None, "images": None}

    def fake_create(model, messages, max_tokens):
        captured["prompt"] = messages[0]["content"][0]["text"]
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=(
                        "Analysis of the uploaded scenes confirms significant environmental change. "
                        "P2 numerical measurements show the authoritative transition across the monitored area. "
                        "No spectral indices were hallucinated."
                    )
                )
            )
        ]
        return mock_resp

    return fake_create, captured


# ============================================================
# TEST 1: Multispectral NDVI
# ============================================================
def test_upload_multispectral_ndvi(monkeypatch):
    """
    TEST 1:
    Two multispectral images, Query: 'Show vegetation change'
    Verify:
    - P2 receives both images
    - NDVI is calculated
    - before/after NDVI statistics returned
    - change map generated
    - P4 receives images + change map + evidence
    - Grounded answer returned
    """
    # Before: High vegetation (Red=0.1, NIR=0.8 -> NDVI = 0.777)
    # After: Decreased vegetation (Red=0.3, NIR=0.4 -> NDVI = 0.142)
    before_bytes = create_multispectral_geotiff_bytes(
        red_val=0.1, green_val=0.2, blue_val=0.1, nir_val=0.8, swir_val=0.2
    )
    after_bytes = create_multispectral_geotiff_bytes(
        red_val=0.3, green_val=0.2, blue_val=0.1, nir_val=0.4, swir_val=0.2
    )

    captured_prompt = None

    def fake_generate(self, image=None, question="", evidence=None, images=None):
        nonlocal captured_prompt
        assert "before" in images
        assert "after" in images
        assert "change_map" in images
        assert "NDVI" in str(evidence)
        return (
            "Vegetation experienced a notable decrease between the two uploaded scenes. "
            "Mean NDVI decreased significantly from 0.7778 to 0.1429 (net change: -0.6349)."
        )

    monkeypatch.setattr(VLM, "generate", fake_generate)

    result = process_upload_analysis(
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        before_name="scene_2021.tif",
        after_name="scene_2024.tif",
        query="Show vegetation change",
    )

    assert result.status == "success"
    assert result.plan["task"] == "vegetation_change"
    assert result.plan["metric"] == "NDVI"
    assert result.statistics["metric"] == "NDVI"
    assert result.statistics["mean_before"] > 0.7
    assert result.statistics["mean_after"] < 0.2
    assert result.statistics["mean_change"] < 0.0
    assert result.statistics["change_type"] == "decrease"
    assert result.visualization_url is not None
    assert result.visualization_url.startswith("/visualizations/")
    assert result.classified_visualization_url is not None
    assert "NDVI" in result.answer
    assert result.images["before"] is not None
    assert result.images["after"] is not None
    assert result.images["change_map"] is not None
    print("\n[TEST 1 PASSED] Multispectral NDVI change detection and P4 handoff verified.")


# ============================================================
# TEST 2: Multispectral NDWI
# ============================================================
def test_upload_multispectral_ndwi(monkeypatch):
    """
    TEST 2:
    Two multispectral images, Query: 'Show water change'
    Verify:
    - NDWI calculation
    - change statistics
    - change visualization
    - P4 response
    """
    # NDWI = (Green - NIR) / (Green + NIR)
    # Before: High water (Green=0.7, NIR=0.1 -> NDWI = 0.6 / 0.8 = 0.75)
    # After: Low water (Green=0.2, NIR=0.6 -> NDWI = -0.4 / 0.8 = -0.5)
    before_bytes = create_multispectral_geotiff_bytes(
        red_val=0.1, green_val=0.7, blue_val=0.3, nir_val=0.1, swir_val=0.1
    )
    after_bytes = create_multispectral_geotiff_bytes(
        red_val=0.1, green_val=0.2, blue_val=0.2, nir_val=0.6, swir_val=0.2
    )

    def fake_generate(self, image=None, question="", evidence=None, images=None):
        assert "NDWI" in str(evidence)
        return "Water bodies receded between the two dates. Mean NDWI dropped from 0.7500 to -0.5000."

    monkeypatch.setattr(VLM, "generate", fake_generate)

    result = process_upload_analysis(
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        before_name="lake_before.tif",
        after_name="lake_after.tif",
        query="Show water change",
    )

    assert result.status == "success"
    assert result.plan["metric"] == "NDWI"
    assert result.statistics["metric"] == "NDWI"
    assert result.statistics["mean_before"] > 0.7
    assert result.statistics["mean_after"] < 0.0
    assert result.statistics["change_type"] == "decrease"
    assert result.visualization_url is not None
    assert "NDWI" in result.answer
    print("\n[TEST 2 PASSED] Multispectral NDWI change detection and P4 handoff verified.")


# ============================================================
# TEST 3: Multispectral NDBI
# ============================================================
def test_upload_multispectral_ndbi(monkeypatch):
    """
    TEST 3:
    Two multispectral images, Query: 'Show urban change'
    Verify:
    - NDBI calculation
    - change statistics
    - change visualization
    - P4 response
    """
    # NDBI = (SWIR - NIR) / (SWIR + NIR)
    # Before: Low urban / high veg (SWIR=0.2, NIR=0.6 -> NDBI = -0.4 / 0.8 = -0.5)
    # After: High urban built-up (SWIR=0.7, NIR=0.3 -> NDBI = 0.4 / 1.0 = +0.4)
    before_bytes = create_multispectral_geotiff_bytes(
        red_val=0.2, green_val=0.3, blue_val=0.1, nir_val=0.6, swir_val=0.2
    )
    after_bytes = create_multispectral_geotiff_bytes(
        red_val=0.4, green_val=0.2, blue_val=0.2, nir_val=0.3, swir_val=0.7
    )

    def fake_generate(self, image=None, question="", evidence=None, images=None):
        assert "NDBI" in str(evidence)
        return "Urban built-up area expanded noticeably. Mean NDBI increased from -0.5000 to +0.4000."

    monkeypatch.setattr(VLM, "generate", fake_generate)

    result = process_upload_analysis(
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        before_name="city_2020.tif",
        after_name="city_2024.tif",
        query="Show urban change",
    )

    assert result.status == "success"
    assert result.plan["metric"] == "NDBI"
    assert result.statistics["metric"] == "NDBI"
    assert result.statistics["mean_before"] < 0.0
    assert result.statistics["mean_after"] > 0.3
    assert result.statistics["change_type"] == "increase"
    assert result.visualization_url is not None
    assert "NDBI" in result.answer
    print("\n[TEST 3 PASSED] Multispectral NDBI change detection and P4 handoff verified.")


# ============================================================
# TEST 4: RGB Images (No Fake NDVI)
# ============================================================
def test_upload_rgb_no_fake_ndvi(monkeypatch):
    """
    TEST 4:
    Two RGB images, Query: 'Show vegetation change'
    Verify:
    - system does NOT invent NDVI
    - system explains that quantitative NDVI requires suitable spectral bands
    - visual analysis can still be performed if supported
    - P4 does not claim a fake NDVI value
    """
    before_bytes = create_rgb_image_bytes(color=(30, 150, 40))
    after_bytes = create_rgb_image_bytes(color=(160, 120, 50))

    captured_evidence = None

    def fake_generate(self, image=None, question="", evidence=None, images=None):
        nonlocal captured_evidence
        captured_evidence = str(evidence)
        # Verify that VLM prompt contains the spectral warning
        assert "Near-Infrared (NIR)" in captured_evidence
        assert "CRITICAL: Do NOT invent or claim quantitative NDVI" in captured_evidence
        return (
            "The uploaded images are standard 3-band visible RGB images without Near-Infrared (NIR). "
            "Therefore, quantitative NDVI cannot be computed without fabricating data. "
            "However, visual comparison shows a color transition from green to brown tones."
        )

    monkeypatch.setattr(VLM, "generate", fake_generate)

    result = process_upload_analysis(
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        before_name="photo_before.png",
        after_name="photo_after.png",
        query="Show vegetation change",
    )

    assert result.status == "success"
    # Must NOT claim NDVI as metric
    assert result.statistics["metric"] != "NDVI"
    assert result.plan["metric"] != "NDVI"
    assert result.plan["spectral_capability"] == "rgb_only"
    assert "spectral_warning" in result.statistics
    assert "Near-Infrared (NIR)" in result.statistics["spectral_warning"]
    assert result.visualization_url is not None
    assert "cannot be computed without fabricating" in result.answer or "RGB" in result.answer
    print("\n[TEST 4 PASSED] Standard RGB upload correctly rejects fake NDVI and conducts visual analysis.")


# ============================================================
# TEST 5: Invalid / Missing Upload
# ============================================================
def test_upload_invalid_missing():
    """
    TEST 5:
    One invalid/missing upload
    Verify:
    - clear error
    - no VLM hallucination
    - API does not crash unexpectedly
    """
    client = TestClient(app)

    # Missing after image
    resp = client.post(
        "/api/upload/analyze",
        data={"query": "Show vegetation change"},
        files={"before_image": ("before.png", b"fake_png_data", "image/png")},
    )
    assert resp.status_code == 400
    assert "required" in resp.json()["detail"].lower()

    # Empty file content
    resp2 = client.post(
        "/api/upload/analyze",
        data={"query": "Show vegetation change"},
        files={
            "before_image": ("before.png", b"", "image/png"),
            "after_image": ("after.png", b"", "image/png"),
        },
    )
    assert resp2.status_code == 400
    assert "empty" in resp2.json()["detail"].lower()

    # Corrupted / invalid image bytes
    resp3 = client.post(
        "/api/upload/analyze",
        data={"query": "Show vegetation change"},
        files={
            "before_image": ("before.png", b"not_an_image", "image/png"),
            "after_image": ("after.png", b"also_not_an_image", "image/png"),
        },
    )
    assert resp3.status_code == 400
    assert "could not open image" in resp3.json()["detail"].lower()

    print("\n[TEST 5 PASSED] Invalid/missing uploads cleanly rejected with HTTP 400 without crashing.")


# ============================================================
# TEST 6: Fast-API Endpoint Multipart Integration
# ============================================================
def test_upload_api_endpoint_multipart(monkeypatch):
    """Verify full FastAPI /api/upload/analyze endpoint with multipart payload."""
    before_bytes = create_rgb_image_bytes(color=(50, 100, 150))
    after_bytes = create_rgb_image_bytes(color=(60, 110, 160))

    def fake_generate(self, image=None, question="", evidence=None, images=None):
        return "Visual difference analysis completed via HTTP endpoint."

    monkeypatch.setattr(VLM, "generate", fake_generate)

    client = TestClient(app)
    resp = client.post(
        "/api/upload/analyze",
        data={"query": "Compare visual changes"},
        files={
            "before_image": ("before.png", before_bytes, "image/png"),
            "after_image": ("after.png", after_bytes, "image/png"),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["visualization_url"] is not None
    assert "before" in data["images"]
    assert "after" in data["images"]
    print("\n[TEST 6 PASSED] Full FastAPI multipart upload endpoint verified.")


# ============================================================
# TEST 7: Full P4 VLM + BigEarthNet Retrieval for Uploads
# ============================================================
def test_upload_full_p4_retrieval(monkeypatch):
    """
    Verify upload flow end-to-end through real VLM.generate:
    - User query triggers BigEarthNet retrieval.
    - Examples are injected into Qwen prompt with demonstration rules.
    - Authoritative P2 statistics and images are delivered.
    """
    before_bytes = create_multispectral_geotiff_bytes(
        red_val=0.15, green_val=0.2, blue_val=0.1, nir_val=0.75, swir_val=0.2
    )
    after_bytes = create_multispectral_geotiff_bytes(
        red_val=0.25, green_val=0.2, blue_val=0.1, nir_val=0.45, swir_val=0.2
    )

    captured_prompt = None

    def fake_create(model, messages, max_tokens):
        nonlocal captured_prompt
        captured_prompt = messages[0]["content"][0]["text"]
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=(
                        "Multimodal analysis of the uploaded Sentinel-style multispectral scenes "
                        "confirms a significant reduction in NDVI from 0.6667 to 0.2857. "
                        "The change map accurately isolates canopy loss."
                    )
                )
            )
        ]
        return mock_resp

    monkeypatch.setenv("HF_TOKEN", "hf_test_dummy_token")

    # Intercept VLM.__init__ to avoid network client errors, and mock completions.create
    orig_init = VLM.__init__

    def fake_init(self):
        orig_init(self)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create
        self.client = mock_client

    monkeypatch.setattr(VLM, "__init__", fake_init)

    result = process_upload_analysis(
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        before_name="multispectral_before.tif",
        after_name="multispectral_after.tif",
        query="Show vegetation change",
    )

    assert result.status == "success"
    assert result.plan["metric"] == "NDVI"
    assert captured_prompt is not None
    assert "BIGEARTHNET REMOTE-SENSING EXAMPLES" in captured_prompt
    assert "Treat backend numerical measurements as authoritative." in captured_prompt
    assert "Never copy or use BigEarthNet example values" in captured_prompt
    assert "NDVI" in captured_prompt
    assert "reduction in NDVI" in result.answer
    print("\n[TEST 7 PASSED] Upload -> P2 -> BigEarthNet Retrieval -> Qwen full prompt verified.")
