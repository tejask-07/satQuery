"""
Comprehensive test suite for SatQuery backend input validation.

Covers:
1. Single Image Validation (optical, SAR, missing, unsupported types, corrupt files)
2. Temporal Change Validation (valid pair, missing before/after, wrong count, mismatched dimensions, spatial overlap, corrupt inputs)
3. Optical-SAR Multimodal Validation (valid pair, optical+optical, SAR+SAR, only optical/SAR, corrupt, missing metadata)
4. GeoTIFF Specific Validation (valid, corrupt, invalid dimensions, 0 bands, missing CRS/transform)
5. Query / Input Compatibility Validation (query-intent vs uploaded input mismatch)
6. Error Behavior & Schema Integrity (clean HTTP 400 responses, no 500s, no raw stack traces)
"""

import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.query import QueryRequest
from app.schemas.validation import ValidationErrorCode, ValidationResult
from app.remote_sensing.validation import (
    validate_file_format,
    validate_image_bytes,
    validate_temporal_pair,
    validate_optical_sar_pair,
    validate_optical_sar_request,
    validate_query_input_compatibility,
)
from app.agent.upload_processor import inspect_image, process_upload_analysis


# ============================================================
# SYNTHETIC RASTER FIXTURES
# ============================================================

def make_geotiff_bytes(
    width: int = 64,
    height: int = 64,
    count: int = 4,
    crs: str = "EPSG:32643",
    bounds: tuple = (500000.0, 3000000.0, 501000.0, 3001000.0),
    descriptions: tuple = ("Red", "Green", "Blue", "NIR"),
    dtype: str = "uint16",
) -> bytes:
    """Generate in-memory valid GeoTIFF bytes."""
    transform = from_bounds(*bounds, width, height)
    crs_obj = CRS.from_user_input(crs) if crs else None
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs_obj,
        transform=transform,
    ) as dst:
        for i in range(1, count + 1):
            data = np.full((height, width), fill_value=i * 200, dtype=dtype)
            dst.write(data, i)
            if i <= len(descriptions) and descriptions[i - 1]:
                dst.set_band_description(i, descriptions[i - 1])
    return memfile.read()


def make_sar_geotiff_bytes(
    width: int = 64,
    height: int = 64,
    crs: str = "EPSG:32643",
    bounds: tuple = (500000.0, 3000000.0, 501000.0, 3001000.0),
) -> bytes:
    """Generate in-memory valid Sentinel-1 SAR GeoTIFF bytes with VV/VH bands."""
    transform = from_bounds(*bounds, width, height)
    crs_obj = CRS.from_user_input(crs)
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="float32",
        crs=crs_obj,
        transform=transform,
    ) as dst:
        dst.write(np.random.uniform(0.01, 0.5, (height, width)).astype("float32"), 1)
        dst.write(np.random.uniform(0.005, 0.2, (height, width)).astype("float32"), 2)
        dst.set_band_description(1, "VV")
        dst.set_band_description(2, "VH")
        dst.update_tags(POLARIZATION="VV,VH", SENSOR="SENTINEL-1")
    return memfile.read()


def make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Generate in-memory valid RGB PNG bytes."""
    img = Image.new("RGB", (width, height), color=(120, 180, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Generate in-memory valid RGB JPEG bytes."""
    img = Image.new("RGB", (width, height), color=(150, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================
# 1. SINGLE IMAGE VALIDATION TESTS
# ============================================================

class TestSingleImageValidation:
    """Test validation of single-image uploads and ingestion."""

    def test_valid_optical_geotiff(self):
        buf = make_geotiff_bytes()
        res_fmt = validate_file_format("scene.tif", buf)
        assert res_fmt.valid is True

        res_int = validate_image_bytes(buf, "scene.tif")
        assert res_int.valid is True
        assert res_int.details["width"] == 64
        assert res_int.details["count"] == 4

    def test_valid_sar_geotiff(self):
        buf = make_sar_geotiff_bytes()
        res_fmt = validate_file_format("sar_scene.tif", buf)
        assert res_fmt.valid is True

        res_int = validate_image_bytes(buf, "sar_scene.tif")
        assert res_int.valid is True
        assert res_int.details["count"] == 2

    def test_valid_png_and_jpeg(self):
        png_buf = make_png_bytes()
        res_png = validate_file_format("image.png", png_buf)
        assert res_png.valid is True
        assert validate_image_bytes(png_buf, "image.png").valid is True

        jpg_buf = make_jpeg_bytes()
        res_jpg = validate_file_format("photo.jpg", jpg_buf)
        assert res_jpg.valid is True
        assert validate_image_bytes(jpg_buf, "photo.jpg").valid is True

    def test_vqa_and_caption_accept_png_jpeg_without_crs(self):
        png_buf = make_png_bytes()
        res_vqa = validate_image_bytes(png_buf, "sample.png", workflow="vqa")
        assert res_vqa.valid is True

        jpg_buf = make_jpeg_bytes()
        res_cap = validate_image_bytes(jpg_buf, "sample.jpg", workflow="caption")
        assert res_cap.valid is True

    def test_vqa_accepts_optical_and_sar_geotiff(self):
        opt_buf = make_geotiff_bytes()
        assert validate_image_bytes(opt_buf, "optical.tif", workflow="vqa").valid is True

        sar_buf = make_sar_geotiff_bytes()
        assert validate_image_bytes(sar_buf, "sar.tif", workflow="vqa").valid is True


    @pytest.mark.parametrize("bad_ext,bad_content", [
        ("report.txt", b"Plain text content not an image"),
        ("document.pdf", b"%PDF-1.4 binary stream ..."),
        ("analysis.docx", b"PK\x03\x04\x14\x00word/document.xml"),
        ("malware.exe", b"MZ\x90\x00\x03\x00\x00\x00"),
        ("archive.zip", b"PK\x03\x04\x0a\x00\x00\x00"),
    ])
    def test_unsupported_file_extensions_rejected(self, bad_ext, bad_content):
        res = validate_file_format(bad_ext, bad_content)
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.UNSUPPORTED_FILE_TYPE
        assert "Unsupported" in res.message
        assert res.details["received"] in (f".{bad_ext.split('.')[-1]}", "none")
        assert ".tif" in res.details["supported"]

    def test_missing_or_empty_image(self):
        res_empty = validate_image_bytes(b"", "empty.tif")
        assert res_empty.valid is False
        assert res_empty.error_code == ValidationErrorCode.MISSING_IMAGE

    @pytest.mark.parametrize("corrupt_name,corrupt_bytes", [
        ("truncated.tif", b"II*\x00\x08\x00\x00\x00truncated"),
        ("corrupt.png", b"\x89PNG\r\n\x1a\ncorrupted_payload_data_not_valid_chunks"),
        ("corrupt.jpg", b"\xff\xd8\xff\xe0\x00\x10JFIFbroken"),
        ("fake_raster.tif", b"Some random plain text disguised as tiff file"),
        ("fake_png.png", b"random_uncompressed_bytes_that_are_not_png_chunks"),
    ])
    def test_corrupt_images_rejected_cleanly(self, corrupt_name, corrupt_bytes):
        res = validate_image_bytes(corrupt_bytes, corrupt_name)
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.INVALID_IMAGE
        assert "Corrupt" in res.message or "invalid" in res.message.lower() or "unreadable" in res.message.lower()


# ============================================================
# 2. TEMPORAL CHANGE VALIDATION TESTS
# ============================================================

class TestTemporalValidation:
    """Test temporal before/after image pair validation."""

    def test_valid_temporal_pair(self):
        b_bytes = make_geotiff_bytes(64, 64)
        a_bytes = make_geotiff_bytes(64, 64)

        res = validate_temporal_pair(b_bytes, a_bytes, "before.tif", "after.tif")
        assert res.valid is True

    def test_missing_before_or_after_image(self):
        valid_bytes = make_geotiff_bytes()

        # Missing both
        res_both = validate_temporal_pair(None, None)
        assert res_both.valid is False
        assert res_both.error_code == ValidationErrorCode.MISSING_IMAGE

        # Missing before
        res_b = validate_temporal_pair(None, valid_bytes)
        assert res_b.valid is False
        assert res_b.error_code == ValidationErrorCode.MISSING_IMAGE
        assert "before" in res_b.message.lower()

        # Missing after
        res_a = validate_temporal_pair(valid_bytes, None)
        assert res_a.valid is False
        assert res_a.error_code == ValidationErrorCode.MISSING_IMAGE
        assert "after" in res_a.message.lower()

    def test_drastically_mismatched_dimensions_rejected(self):
        # 1000x1000 vs 100x100 standard images without spatial CRS
        img_large = make_png_bytes(500, 500)
        img_tiny = make_png_bytes(50, 50)

        res = validate_temporal_pair(img_large, img_tiny, "large.png", "tiny.png")
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR
        assert "incompatible dimensions" in res.message.lower()

    def test_zero_spatial_overlap_geotiffs_rejected(self):
        # Two GeoTIFFs in different parts of the world
        b_bytes = make_geotiff_bytes(bounds=(500000.0, 3000000.0, 501000.0, 3001000.0))
        a_bytes = make_geotiff_bytes(bounds=(600000.0, 4000000.0, 601000.0, 4001000.0))

        res = validate_temporal_pair(b_bytes, a_bytes, "paris.tif", "tokyo.tif")
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR
        assert "zero geographic overlap" in res.message.lower()

    def test_corrupt_temporal_images_rejected(self):
        valid_bytes = make_geotiff_bytes()
        corrupt_bytes = b"II*\x00truncated"

        # Corrupt before
        res_b = validate_temporal_pair(corrupt_bytes, valid_bytes, "bad_before.tif", "after.tif")
        assert res_b.valid is False
        assert res_b.error_code == ValidationErrorCode.INVALID_IMAGE

        # Corrupt after
        res_a = validate_temporal_pair(valid_bytes, corrupt_bytes, "before.tif", "bad_after.tif")
        assert res_a.valid is False
        assert res_a.error_code == ValidationErrorCode.INVALID_IMAGE

    def test_temporal_optical_optical_accepted(self):
        # Optical before + Optical after is valid
        opt_b = make_geotiff_bytes(64, 64, descriptions=("Red", "Green", "Blue", "NIR"))
        opt_a = make_geotiff_bytes(64, 64, descriptions=("Red", "Green", "Blue", "NIR"))
        res = validate_temporal_pair(opt_b, opt_a, "opt_before.tif", "opt_after.tif")
        assert res.valid is True
        assert res.details["before"]["modality"] == "optical"
        assert res.details["after"]["modality"] == "optical"

    def test_temporal_sar_sar_accepted(self):
        # SAR before + SAR after is valid
        sar_b = make_sar_geotiff_bytes(64, 64)
        sar_a = make_sar_geotiff_bytes(64, 64)
        res = validate_temporal_pair(sar_b, sar_a, "sar_before.tif", "sar_after.tif")
        assert res.valid is True
        assert res.details["before"]["modality"] == "sar"
        assert res.details["after"]["modality"] == "sar"

    def test_temporal_cross_modality_optical_sar_rejected(self):
        # Optical before + SAR after rejected: current temporal engine does not support cross-modality
        opt_b = make_geotiff_bytes(64, 64, descriptions=("Red", "Green", "Blue", "NIR"))
        sar_a = make_sar_geotiff_bytes(64, 64)
        res = validate_temporal_pair(opt_b, sar_a, "opt_before.tif", "sar_after.tif")
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR
        assert "sensor_modality_mismatch" in res.details["reason"]
        assert "optical" in res.message.lower() and "sar" in res.message.lower()

    def test_temporal_cross_modality_sar_optical_rejected(self):
        # SAR before + Optical after rejected
        sar_b = make_sar_geotiff_bytes(64, 64)
        opt_a = make_geotiff_bytes(64, 64, descriptions=("Red", "Green", "Blue", "NIR"))
        res = validate_temporal_pair(sar_b, opt_a, "sar_before.tif", "opt_after.tif")
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR
        assert "sensor_modality_mismatch" in res.details["reason"]

    def test_temporal_require_crs_rejects_missing_crs(self):
        # Temporal operation requiring CRS rejects non-georeferenced images
        opt_with_crs = make_geotiff_bytes(64, 64, crs="EPSG:32643")
        opt_no_crs = make_geotiff_bytes(64, 64, crs="")

        res = validate_temporal_pair(opt_with_crs, opt_no_crs, "with_crs.tif", "no_crs.tif", require_crs=True)
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.MISSING_METADATA
        assert "georeferencing/crs" in res.message.lower() or "crs" in res.message.lower()


# ============================================================
# 3. OPTICAL-SAR MULTIMODAL VALIDATION TESTS
# ============================================================

class TestOpticalSarValidation:
    """Test Optical-SAR pairing and modality consistency validation."""

    def test_optical_sar_request_counts(self):
        # 0 images
        res_0 = validate_optical_sar_request(None, None, [])
        assert res_0.valid is False
        assert res_0.error_code == ValidationErrorCode.MISSING_IMAGE

        # 1 image
        res_1 = validate_optical_sar_request("opt_1", None, [])
        assert res_1.valid is False
        assert res_1.error_code == ValidationErrorCode.WRONG_IMAGE_COUNT

        # 3 images
        res_3 = validate_optical_sar_request("opt_1", "sar_1", ["extra_1"])
        assert res_3.valid is False
        assert res_3.error_code == ValidationErrorCode.WRONG_IMAGE_COUNT

        # Exactly 2 images
        res_2 = validate_optical_sar_request("opt_1", "sar_1", [])
        assert res_2.valid is True

    def test_optical_optical_conflict_rejected(self, client):
        # Direct endpoint test: uploading two optical files to /api/upload/optical-sar
        opt_1 = make_geotiff_bytes(descriptions=("Red", "Green", "Blue", "NIR"))
        opt_2 = make_geotiff_bytes(descriptions=("Red", "Green", "Blue", "NIR"))

        response = client.post(
            "/api/upload/optical-sar",
            files={
                "optical_image": ("optical1.tif", opt_1, "image/tiff"),
                "sar_image": ("optical2_as_sar.tif", opt_2, "image/tiff"),
            },
            data={"query": "Analyze optical and SAR jointly."},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["valid"] is False
        assert data["error_code"] == ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR

    def test_validate_optical_sar_pair_in_memory(self):
        opt = make_geotiff_bytes(descriptions=("Red", "Green", "Blue", "NIR"))
        sar = make_sar_geotiff_bytes()

        # Valid pair
        res_ok = validate_optical_sar_pair(opt, sar, "opt.tif", "sar.tif")
        assert res_ok.valid is True
        assert res_ok.details["optical"]["modality"] == "optical"
        assert res_ok.details["sar"]["modality"] == "sar"

        # Optical + Optical rejected
        res_oo = validate_optical_sar_pair(opt, opt, "opt1.tif", "opt2.tif")
        assert res_oo.valid is False
        assert res_oo.error_code == ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR
        assert "both_optical" in res_oo.details["reason"]

        # SAR + SAR rejected
        res_ss = validate_optical_sar_pair(sar, sar, "sar1.tif", "sar2.tif")
        assert res_ss.valid is False
        assert res_ss.error_code == ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR
        assert "both_sar" in res_ss.details["reason"]

        # Swapped assignments rejected
        res_swap = validate_optical_sar_pair(sar, opt, "opt_field.tif", "sar_field.tif")
        assert res_swap.valid is False
        assert res_swap.error_code == ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR
        assert "modalities_swapped" in res_swap.details["reason"]

        # Zero geographic overlap rejected
        opt_far = make_geotiff_bytes(bounds=(100000.0, 1000000.0, 101000.0, 1001000.0), descriptions=("Red", "Green", "Blue", "NIR"))
        res_no_overlap = validate_optical_sar_pair(opt_far, sar, "opt_far.tif", "sar.tif")
        assert res_no_overlap.valid is False
        assert res_no_overlap.error_code == ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR
        assert "no_spatial_overlap" in res_no_overlap.details["reason"]


# ============================================================
# 4. GEOTIFF WORKFLOW-AWARE VALIDATION TESTS
# ============================================================

class TestGeoTIFFValidation:
    """Test GeoTIFF metadata, dimensions, and CRS requirements."""

    def test_geotiff_missing_crs_rejected_for_geospatial_workflow(self):
        # GeoTIFF without CRS
        buf = make_geotiff_bytes(crs="")
        res = validate_image_bytes(buf, "no_crs.tif", workflow="geospatial")
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.MISSING_METADATA
        assert "Coordinate Reference System (CRS)" in res.message

    def test_geotiff_missing_crs_accepted_for_general_workflow(self):
        # Standard non-geospatial workflow can still read raster dimensions
        buf = make_geotiff_bytes(crs="")
        res = validate_image_bytes(buf, "no_crs.tif", workflow="general")
        assert res.valid is True
        assert res.details["width"] == 64

    def test_unsupported_non_geotiff_for_geospatial_workflow(self):
        png = make_png_bytes()
        res = validate_image_bytes(png, "chart.png", workflow="geospatial")
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.MISSING_METADATA


# ============================================================
# 5. QUERY / INPUT MISMATCH VALIDATION TESTS
# ============================================================

class TestQueryInputMismatchValidation:
    """Test validation of query intent vs provided inputs."""

    def test_temporal_query_with_single_image_rejected(self):
        res = validate_query_input_compatibility(
            query="Compare these two images and detect change.",
            image_ids=["single_image_1"],
            task="temporal_change",
        )
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.INCOMPATIBLE_QUERY_INPUT
        assert "before and after" in res.message.lower()

    def test_temporal_query_with_three_images_rejected(self):
        res = validate_query_input_compatibility(
            query="Detect change between these images.",
            image_ids=["img_1", "img_2", "img_3"],
            task="temporal_change",
        )
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.WRONG_IMAGE_COUNT

    def test_optical_sar_query_with_single_image_rejected(self):
        res = validate_query_input_compatibility(
            query="Compare optical and SAR radar data.",
            image_ids=["optical_only_1"],
            task="optical_sar_analysis",
        )
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.INCOMPATIBLE_QUERY_INPUT
        assert "one optical image and one sar image" in res.message.lower()

    def test_single_image_query_with_multiple_images_rejected(self):
        res = validate_query_input_compatibility(
            query="Describe this image in detail.",
            image_ids=["img_1", "img_2"],
            task="single_image_caption",
        )
        assert res.valid is False
        assert res.error_code == ValidationErrorCode.WRONG_IMAGE_COUNT
        assert "accepts only one image" in res.message.lower()

    def test_valid_single_image_query_passes(self):
        res = validate_query_input_compatibility(
            query="Describe this image in detail.",
            image_ids=["img_1"],
            task="single_image_caption",
        )
        assert res.valid is True

    def test_valid_temporal_query_passes(self):
        res = validate_query_input_compatibility(
            query="What changed between these two images?",
            image_ids=["img_before", "img_after"],
            task="temporal_change",
        )
        assert res.valid is True

    def test_query_with_zero_images_and_no_aoi_rejected(self):
        # Single image intent with 0 images
        res_single = validate_query_input_compatibility(query="Describe this image in detail.", image_ids=[], aoi=None)
        assert res_single.valid is False
        assert res_single.error_code == ValidationErrorCode.MISSING_IMAGE

        # Temporal intent with 0 images
        res_temp = validate_query_input_compatibility(query="Compare change between images.", image_ids=[], aoi=None)
        assert res_temp.valid is False
        assert res_temp.error_code == ValidationErrorCode.MISSING_IMAGE

        # Optical-SAR intent with 0 images
        res_opt_sar = validate_query_input_compatibility(query="Analyze optical and SAR jointly.", image_ids=[], aoi=None)
        assert res_opt_sar.valid is False
        assert res_opt_sar.error_code == ValidationErrorCode.MISSING_IMAGE

    def test_optical_sar_query_missing_modality_clarity(self):
        res_need_sar = validate_query_input_compatibility(
            query="Use the optical and SAR images together.",
            optical_image_id="opt_123",
            sar_image_id=None,
        )
        assert res_need_sar.valid is False
        assert "SAR input is required" in res_need_sar.message

        res_need_opt = validate_query_input_compatibility(
            query="Use the optical and SAR images together.",
            optical_image_id=None,
            sar_image_id="sar_123",
        )
        assert res_need_opt.valid is False
        assert "Optical input is required" in res_need_opt.message


# ============================================================
# 6. HTTP API ERROR BEHAVIOR & STRUCTURE INTEGRITY
# ============================================================

class TestApiErrorBehavior:
    """Test that API endpoints return structured errors and never raise unhandled 500s."""

    def test_upload_image_unsupported_file_returns_structured_400(self, client):
        response = client.post(
            "/api/upload/image",
            files={"file": ("dataset.pdf", b"%PDF-1.4 ...", "application/pdf")},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["valid"] is False
        assert data["error_code"] == ValidationErrorCode.UNSUPPORTED_FILE_TYPE
        assert ".pdf" in data["details"]["received"]
        assert ".tif" in data["details"]["supported"]
        # Compatibility detail string
        assert isinstance(data["detail"], str)

    def test_upload_image_corrupt_file_returns_clean_400(self, client):
        response = client.post(
            "/api/upload/image",
            files={"file": ("damaged.tif", b"II*\x00broken_corrupted_content", "image/tiff")},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["valid"] is False
        assert data["error_code"] == ValidationErrorCode.INVALID_IMAGE
        assert "corrupt" in data["message"].lower() or "invalid" in data["message"].lower()

    def test_upload_analyze_missing_before_returns_structured_400(self, client):
        img_valid = make_geotiff_bytes()
        response = client.post(
            "/api/upload/analyze",
            files={"after_image": ("after.tif", img_valid, "image/tiff")},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["valid"] is False
        assert data["error_code"] == ValidationErrorCode.MISSING_IMAGE

    def test_upload_analyze_empty_file_returns_clean_400(self, client):
        img_valid = make_geotiff_bytes()
        response = client.post(
            "/api/upload/analyze",
            files={
                "before_image": ("before.tif", b"", "image/tiff"),
                "after_image": ("after.tif", img_valid, "image/tiff"),
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["valid"] is False
        assert data["error_code"] == ValidationErrorCode.MISSING_IMAGE

    def test_upload_analyze_incompatible_dimensions_returns_structured_400(self, client):
        large = make_png_bytes(600, 600)
        tiny = make_png_bytes(50, 50)
        response = client.post(
            "/api/upload/analyze",
            files={
                "before_image": ("large.png", large, "image/png"),
                "after_image": ("tiny.png", tiny, "image/png"),
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["valid"] is False
        assert data["error_code"] == ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR
        assert "dimension_mismatch" in data["details"]["reason"]

    def test_query_route_mismatch_returns_structured_400(self, client):
        response = client.post(
            "/api/query",
            json={
                "query": "Compare these two images and detect change.",
                "image_ids": ["only_one_image_id"],
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["valid"] is False
        assert data["error_code"] == ValidationErrorCode.INCOMPATIBLE_QUERY_INPUT
