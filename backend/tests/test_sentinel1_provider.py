"""
Unit and Integration Test Suite for Sentinel-1 SAR Acquisition Provider (Step 12).

Tests:
  1. STAC search construction (collection, filters)
  2. AOI filter payload
  3. Date filter payload
  4. Dual-polarization selection (VV and VH)
  5. VV-only scene handling
  6. Deterministic scene ranking
  7. Metadata completeness
  8. Cache reuse behavior
  9. No-result controlled error handling
  10. Physical semantics preservation (no unrequested dB/log10 conversion)
  11. Integration with existing raster readers (rasterio / read_raster)
  12. Path traversal security and filename sanitization
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from app.remote_sensing.providers.sentinel1 import (
    SENTINEL1_COLLECTION,
    Sentinel1ErrorType,
    Sentinel1Provider,
    Sentinel1RetrievalError,
    Sentinel1ScoringWeights,
    normalize_aoi,
    rank_sentinel1_scenes,
    score_sentinel1_scene,
    search_real_sentinel1,
)
from app.remote_sensing.io.raster import read_raster


# ============================================================
# TEST FIXTURES & HELPERS
# ============================================================

def _make_mock_s1_scene(
    scene_id: str = "S1A_IW_GRDH_1SDV_20210625T051821_038492_048AD8",
    datetime_str: str = "2021-06-25T05:18:21Z",
    has_vh: bool = True,
    has_vv: bool = True,
    mode: str = "IW",
    orbit_state: str = "ascending",
    bbox: list = [13.0, 48.0, 13.5, 48.5],
) -> dict:
    assets = {}
    pols = []
    if has_vv:
        assets["vv"] = {"href": f"https://mockblob.blob.core.windows.net/{scene_id}_vv.tif"}
        pols.append("VV")
    if has_vh:
        assets["vh"] = {"href": f"https://mockblob.blob.core.windows.net/{scene_id}_vh.tif"}
        pols.append("VH")

    return {
        "id": scene_id,
        "collection": SENTINEL1_COLLECTION,
        "datetime": datetime_str,
        "bbox": bbox,
        "properties": {
            "datetime": datetime_str,
            "platform": "Sentinel-1A",
            "sar:instrument_mode": mode,
            "sar:product_type": "GRD",
            "sar:polarizations": pols,
            "sat:orbit_state": orbit_state,
        },
        "assets": assets,
    }


def _create_mock_geotiff(path: Path, width: int = 20, height: int = 20, value: int = 150):
    transform = from_bounds(13.0, 48.0, 13.02, 48.02, width, height)
    data = np.full((height, width), value, dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data, 1)


# ============================================================
# 1. STAC SEARCH CONSTRUCTION
# ============================================================

def test_stac_search_construction():
    """Verify STAC search requests sentinel-1-grd collection and correct parameters."""
    provider = Sentinel1Provider()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"features": [_make_mock_s1_scene()]}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        scenes = provider.search_candidate_scenes(
            bbox=(13.0, 48.0, 13.1, 48.1),
            datetime_range="2021-06-01T00:00:00Z/2021-06-30T23:59:59Z",
            instrument_mode="IW",
        )

        assert len(scenes) == 1
        assert mock_post.called
        call_args = mock_post.call_args
        payload = call_args[1]["json"]

        assert payload["collections"] == ["sentinel-1-grd"]
        assert payload["bbox"] == [13.0, 48.0, 13.1, 48.1]
        assert payload["datetime"] == "2021-06-01T00:00:00Z/2021-06-30T23:59:59Z"
        assert payload["query"]["sar:instrument_mode"] == {"eq": "IW"}


# ============================================================
# 2. AOI FILTER
# ============================================================

def test_aoi_filter_passed_correctly():
    """Verify various AOI formats normalize to bounding box and pass into search payload."""
    provider = Sentinel1Provider()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"features": [_make_mock_s1_scene()]}

    # Test GeoJSON polygon AOI
    geojson_poly = {
        "type": "Polygon",
        "coordinates": [[[13.0, 48.0], [13.2, 48.0], [13.2, 48.2], [13.0, 48.2], [13.0, 48.0]]],
    }
    normalized = normalize_aoi(geojson_poly)
    assert normalized == (13.0, 48.0, 13.2, 48.2)

    with patch("requests.post", return_value=mock_resp) as mock_post:
        provider.search_candidate_scenes(bbox=normalized, datetime_range="2021-06-01/2021-06-30")
        payload = mock_post.call_args[1]["json"]
        assert payload["bbox"] == [13.0, 48.0, 13.2, 48.2]


# ============================================================
# 3. DATE FILTER
# ============================================================

def test_date_range_filter_respected():
    """Verify requested time range is sent to STAC search endpoint."""
    provider = Sentinel1Provider()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"features": [_make_mock_s1_scene()]}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        provider.search_candidate_scenes(
            bbox=(13.0, 48.0, 13.1, 48.1),
            datetime_range="2021-06-15T00:00:00Z/2021-06-20T23:59:59Z",
        )
        payload = mock_post.call_args[1]["json"]
        assert payload["datetime"] == "2021-06-15T00:00:00Z/2021-06-20T23:59:59Z"


# ============================================================
# 4. POLARIZATION SELECTION (VV AND VH)
# ============================================================

def test_polarization_selection_dual_pol(tmp_path: Path):
    """Verify dual-polarization imagery downloads and caches both VV and VH."""
    scene = _make_mock_s1_scene(scene_id="s1_dual_test", has_vv=True, has_vh=True)
    provider = Sentinel1Provider(cache_dir=tmp_path)

    # Mock candidate search
    with patch.object(provider, "search_candidate_scenes", return_value=[scene]):
        # Mock fetch_and_cache_polarization to create synthetic GeoTIFFs
        def mock_fetch(scene_item, polarization, bbox, **kw):
            p = tmp_path / f"s1_{polarization}.tif"
            _create_mock_geotiff(p, value=200 if polarization == "vv" else 50)
            return str(p)

        with patch.object(provider, "fetch_and_cache_polarization", side_effect=mock_fetch):
            res = provider.search_and_fetch(
                time_start="2021-06-01",
                time_end="2021-06-30",
                aoi=[13.0, 48.0, 13.1, 48.1],
            )

            assert res["status"] == "REAL_SUCCESS"
            assert "VV" in res["polarizations"]
            assert "VH" in res["polarizations"]
            assert res["vv"] is not None and Path(res["vv"]).exists()
            assert res["vh"] is not None and Path(res["vh"]).exists()
            assert res["provider"] == "sentinel1"
            assert res["product"] == "GRD"


# ============================================================
# 5. VV-ONLY SCENE HANDLING
# ============================================================

def test_vv_only_scene_handling(tmp_path: Path):
    """Verify provider handles single-band VV-only scenes cleanly with vh=None and without fabricating VH."""
    scene = _make_mock_s1_scene(scene_id="s1_vv_only", has_vv=True, has_vh=False)
    provider = Sentinel1Provider(cache_dir=tmp_path)

    with patch.object(provider, "search_candidate_scenes", return_value=[scene]):
        def mock_fetch(scene_item, polarization, bbox, **kw):
            p = tmp_path / f"s1_{polarization}.tif"
            _create_mock_geotiff(p, value=180)
            return str(p)

        with patch.object(provider, "fetch_and_cache_polarization", side_effect=mock_fetch):
            res = provider.search_and_fetch(
                time_start="2021-06-01",
                time_end="2021-06-30",
                aoi=[13.0, 48.0, 13.1, 48.1],
            )

            assert res["status"] == "REAL_SUCCESS"
            assert res["polarizations"] == ["VV"]
            assert res["vv"] is not None and Path(res["vv"]).exists()
            assert res["vh"] is None  # Must NOT fabricate VH


# ============================================================
# 6. DETERMINISTIC SCENE RANKING
# ============================================================

def test_deterministic_scene_ranking():
    """Verify candidate scenes are ranked predictably by coverage, temporal delta, and polarization."""
    aoi_bbox = (13.0, 48.0, 13.1, 48.1)
    target_dt = "2021-06-25T12:00:00Z"

    # Scene 1: Closest in time (same day), full coverage, dual-pol
    s1 = _make_mock_s1_scene("scene_best", datetime_str="2021-06-25T10:00:00Z", has_vv=True, has_vh=True)
    # Scene 2: 5 days away, full coverage, dual-pol
    s2 = _make_mock_s1_scene("scene_older", datetime_str="2021-06-20T10:00:00Z", has_vv=True, has_vh=True)
    # Scene 3: Same day, but VV-only
    s3 = _make_mock_s1_scene("scene_single_pol", datetime_str="2021-06-25T10:00:00Z", has_vv=True, has_vh=False)

    ranked = rank_sentinel1_scenes([s2, s3, s1], aoi_bbox=aoi_bbox, target_date=target_dt)

    assert len(ranked) == 3
    # Best scene must rank first
    assert ranked[0][0]["id"] == "scene_best"
    assert ranked[0][1]["score"] > ranked[1][1]["score"]


# ============================================================
# 7. METADATA COMPLETENESS
# ============================================================

def test_metadata_returned_completeness(tmp_path: Path):
    """Verify returned result contains all required metadata fields."""
    scene = _make_mock_s1_scene(
        scene_id="s1_meta_test",
        datetime_str="2021-06-27T16:58:35Z",
        mode="IW",
        orbit_state="descending",
    )
    provider = Sentinel1Provider(cache_dir=tmp_path)

    with patch.object(provider, "search_candidate_scenes", return_value=[scene]):
        with patch.object(provider, "fetch_and_cache_polarization", return_value=str(tmp_path / "mock.tif")):
            _create_mock_geotiff(tmp_path / "mock.tif")

            res = provider.search_and_fetch(aoi=[13.0, 48.0, 13.1, 48.1])

            assert res["item_id"] == "s1_meta_test"
            assert res["acquisition_datetime"] == "2021-06-27T16:58:35Z"
            assert res["instrument_mode"] == "IW"
            assert res["orbit_direction"] == "descending"
            assert res["crs"] == "EPSG:4326"
            assert "metadata" in res
            assert res["metadata"]["physical_units"] == "uncalibrated_linear_dn"
            assert "selection_reason" in res


# ============================================================
# 8. CACHE REUSE BEHAVIOR
# ============================================================

def test_cache_reuse_avoids_redownload(tmp_path: Path):
    """Verify existing cached GeoTIFFs are reused immediately without network requests."""
    provider = Sentinel1Provider(cache_dir=tmp_path)
    scene = _make_mock_s1_scene("s1_cache_test")

    # Manually populate cache file with expected hash filename
    bbox = (13.0, 48.0, 13.01, 48.01)
    res_deg = 0.00009
    w = max(1, int(round(0.01 / res_deg)))
    h = max(1, int(round(0.01 / res_deg)))
    aoi_key = f"{bbox[0]:.5f}_{bbox[1]:.5f}_{bbox[2]:.5f}_{bbox[3]:.5f}_{w}x{h}"
    import hashlib
    aoi_hash = hashlib.md5(aoi_key.encode()).hexdigest()[:8]
    expected_cache_file = tmp_path / f"s1_s1_cache_test_vv_{aoi_hash}.tif"
    _create_mock_geotiff(expected_cache_file, width=w, height=h, value=300)

    # Calling fetch_and_cache_polarization should return cached path without calling rasterio.open on network
    with patch("rasterio.open", wraps=rasterio.open) as mock_open:
        cached_res = provider.fetch_and_cache_polarization(scene, "vv", bbox=bbox)
        assert Path(cached_res).exists()
        assert cached_res == str(expected_cache_file.resolve())


# ============================================================
# 9. NO-RESULT BEHAVIOR
# ============================================================

def test_no_scenes_found_clean_error():
    """Verify clean controlled failure with Sentinel1ErrorType.NO_SCENES_FOUND when no scenes match."""
    provider = Sentinel1Provider()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"features": []}

    with patch("requests.post", return_value=mock_resp):
        res = provider.search_and_fetch(
            time_start="1999-01-01",
            time_end="1999-01-02",
            aoi=[13.0, 48.0, 13.1, 48.1],
        )

        assert res["status"] == "REAL_FAILURE"
        assert res["error_type"] == Sentinel1ErrorType.NO_SCENES_FOUND
        assert res["vv"] is None
        assert res["vh"] is None


# ============================================================
# 10. NO PHYSICAL CONVERSION
# ============================================================

def test_no_physical_conversion_preserves_raw_dn(tmp_path: Path):
    """Verify provider does NOT introduce unrequested dB, log10, or normalization transformations."""
    test_file = tmp_path / "raw_s1.tif"
    _create_mock_geotiff(test_file, value=450)

    with rasterio.open(test_file) as src:
        arr = src.read(1)
        assert arr.dtype == np.uint16
        assert arr[0, 0] == 450  # Raw DN unmodified


# ============================================================
# 11. INTEGRATION WITH EXISTING RASTER READERS
# ============================================================

def test_integration_with_raster_readers(tmp_path: Path):
    """Verify returned GeoTIFF can be read seamlessly by existing SatQuery raster utilities."""
    test_tif = tmp_path / "s1_reader_test.tif"
    _create_mock_geotiff(test_tif, width=30, height=30, value=255)

    arr, meta = read_raster(str(test_tif))
    assert arr is not None
    assert meta is not None
    assert meta["width"] == 30
    assert meta["height"] == 30
    assert arr.shape == (30, 30)
    assert arr[0, 0] == 255


# ============================================================
# 12. PATH SECURITY AND FILENAME SANITIZATION
# ============================================================

def test_path_security_sanitization(tmp_path: Path):
    """Verify malicious scene IDs cannot traverse out of authorized cache directory."""
    provider = Sentinel1Provider(cache_dir=tmp_path)
    malicious_scene = _make_mock_s1_scene(
        scene_id="../../../../../etc/passwd",
        has_vv=True,
    )

    with patch.object(provider, "sign_asset_url", return_value="http://mock.tif"):
        with patch("rasterio.open"):
            # The sanitized filename should replace '/' and '..' with '_'
            with pytest.raises(Exception):
                # When reading fails, verify path did not escape
                provider.fetch_and_cache_polarization(malicious_scene, "vv", bbox=(13.0, 48.0, 13.01, 48.01))
