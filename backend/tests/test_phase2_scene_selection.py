from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pytest

from app.remote_sensing.providers.sentinel2 import (
    Sentinel2Provider,
    Sentinel2ErrorType,
    Sentinel2RetrievalError,
    SceneScoringWeights,
    calculate_aoi_coverage,
    calculate_seasonal_similarity,
    score_scene,
    rank_candidate_scenes,
    normalize_aoi,
)


# ============================================================
# 1. AOI COVERAGE TESTS
# ============================================================

def test_aoi_coverage_full_containment():
    """When scene completely encloses the requested AOI, coverage must be 100%."""
    scene = {
        "id": "S2A_MSIL2A_TEST_FULL",
        "bbox": [16.0, 48.0, 17.0, 49.0],
        "properties": {"eo:cloud_cover": 5.0},
    }
    aoi_bbox = (16.40, 48.20, 16.41, 48.21)
    cov = calculate_aoi_coverage(scene, aoi_bbox)

    assert cov["coverage_percentage"] == 100.0
    assert cov["coverage_ratio"] == 1.0
    assert cov["aoi_area"] > 0
    assert cov["intersection_area"] == pytest.approx(cov["aoi_area"])


def test_aoi_coverage_partial_intersection():
    """When scene overlaps half the AOI longitude span, coverage must be 50%."""
    # AOI spans [16.0, 48.0, 17.0, 49.0] (width 1.0, height 1.0)
    aoi_bbox = (16.0, 48.0, 17.0, 49.0)
    # Scene covers eastern half [16.5, 48.0, 17.5, 49.0]
    scene = {
        "id": "S2A_MSIL2A_TEST_HALF",
        "bbox": [16.5, 48.0, 17.5, 49.0],
        "properties": {"eo:cloud_cover": 5.0},
    }
    cov = calculate_aoi_coverage(scene, aoi_bbox)

    assert pytest.approx(cov["coverage_percentage"]) == 50.0
    assert pytest.approx(cov["coverage_ratio"]) == 0.5


def test_aoi_coverage_disjoint():
    """When scene does not intersect the AOI, coverage must be 0%."""
    aoi_bbox = (16.40, 48.20, 16.41, 48.21)
    scene = {
        "id": "S2A_MSIL2A_TEST_DISJOINT",
        "bbox": [20.0, 50.0, 21.0, 51.0],
        "properties": {"eo:cloud_cover": 0.0},
    }
    cov = calculate_aoi_coverage(scene, aoi_bbox)

    assert cov["coverage_percentage"] == 0.0
    assert cov["coverage_ratio"] == 0.0
    assert cov["intersection_area"] == 0.0


# ============================================================
# 2. SEASONAL SIMILARITY TESTS
# ============================================================

def test_seasonal_similarity_identical_dates():
    """Same calendar date across different years must have perfect similarity 1.0."""
    cand_dt = datetime(2025, 6, 15, tzinfo=timezone.utc)
    target_dt = datetime(2021, 6, 15, tzinfo=timezone.utc)

    score = calculate_seasonal_similarity(cand_dt, target_dt)
    assert pytest.approx(score) == 1.0


def test_seasonal_similarity_opposite_seasons():
    """Summer (June) vs Winter (December) ~182 days apart must have near 0.0 similarity."""
    cand_dt = datetime(2025, 12, 15, tzinfo=timezone.utc)
    target_dt = datetime(2021, 6, 15, tzinfo=timezone.utc)

    score = calculate_seasonal_similarity(cand_dt, target_dt)
    assert score < 0.05


def test_seasonal_similarity_month_int():
    """Target specified as integer month (e.g. 6 for June)."""
    cand_dt = datetime(2025, 6, 15, tzinfo=timezone.utc)
    score = calculate_seasonal_similarity(cand_dt, target_dt_or_month=6)
    assert score > 0.95


# ============================================================
# 3. SCENE RANKING & PREFERENCES
# ============================================================

def test_ranking_prefers_lower_cloud_cover():
    """Given identical coverage and dates, the scene with lower cloud cover must win."""
    aoi_bbox = (16.40, 48.20, 16.41, 48.21)
    scene_clear = {
        "id": "S2_CLEAR",
        "bbox": [16.0, 48.0, 17.0, 49.0],
        "properties": {
            "datetime": "2025-06-15T10:00:00Z",
            "eo:cloud_cover": 2.5,
        },
    }
    scene_cloudy = {
        "id": "S2_CLOUDY",
        "bbox": [16.0, 48.0, 17.0, 49.0],
        "properties": {
            "datetime": "2025-06-15T10:00:00Z",
            "eo:cloud_cover": 45.0,
        },
    }

    ranked = rank_candidate_scenes([scene_cloudy, scene_clear], aoi_bbox=aoi_bbox)
    assert ranked[0][0]["id"] == "S2_CLEAR"
    assert ranked[0][1]["score"] > ranked[1][1]["score"]
    assert "low cloud cover (2.5%)" in ranked[0][1]["selection_reason"]


def test_ranking_prefers_better_aoi_coverage():
    """Given identical cloud cover and date, the scene with 100% coverage must beat partial coverage."""
    aoi_bbox = (16.0, 48.0, 17.0, 49.0)
    scene_full = {
        "id": "S2_FULL",
        "bbox": [16.0, 48.0, 17.0, 49.0],  # 100%
        "properties": {
            "datetime": "2025-06-15T10:00:00Z",
            "eo:cloud_cover": 5.0,
        },
    }
    scene_partial = {
        "id": "S2_PARTIAL",
        "bbox": [16.8, 48.0, 17.5, 49.0],  # 20%
        "properties": {
            "datetime": "2025-06-15T10:00:00Z",
            "eo:cloud_cover": 5.0,
        },
    }

    ranked = rank_candidate_scenes([scene_partial, scene_full], aoi_bbox=aoi_bbox)
    assert ranked[0][0]["id"] == "S2_FULL"
    assert ranked[0][1]["score"] > ranked[1][1]["score"]
    assert ranked[0][1]["coverage_details"]["coverage_percentage"] == 100.0
    assert pytest.approx(ranked[1][1]["coverage_details"]["coverage_percentage"]) == 20.0


def test_ranking_prefers_seasonal_match_over_slight_cloud_difference():
    """
    Summer comparison query (target = June 2021):
    A June 2025 scene with 8% cloud cover should beat a December 2025 scene with 1% cloud cover
    because seasonal alignment preserves phenological comparability.
    """
    aoi_bbox = (16.40, 48.20, 16.41, 48.21)
    target_dt = datetime(2021, 6, 15, tzinfo=timezone.utc)

    scene_summer = {
        "id": "S2_SUMMER_2025",
        "bbox": [16.0, 48.0, 17.0, 49.0],
        "properties": {
            "datetime": "2025-06-20T10:00:00Z",
            "eo:cloud_cover": 8.0,
        },
    }
    scene_winter = {
        "id": "S2_WINTER_2025",
        "bbox": [16.0, 48.0, 17.0, 49.0],
        "properties": {
            "datetime": "2025-12-20T10:00:00Z",
            "eo:cloud_cover": 1.0,
        },
    }

    ranked = rank_candidate_scenes(
        [scene_winter, scene_summer],
        aoi_bbox=aoi_bbox,
        target_date=target_dt,
    )
    assert ranked[0][0]["id"] == "S2_SUMMER_2025"
    assert ranked[0][1]["score"] > ranked[1][1]["score"]


# ============================================================
# 4. CANDIDATE SEARCH & FAILURE MODES
# ============================================================

def test_search_candidate_scenes_collects_candidates():
    """search_candidate_scenes collects multiple candidate items from STAC."""
    provider = Sentinel2Provider()
    mock_features = [
        {"id": f"S2_TEST_{i}", "bbox": [16.0, 48.0, 17.0, 49.0], "properties": {"eo:cloud_cover": float(i)}}
        for i in range(5)
    ]

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"features": mock_features}
        mock_post.return_value = mock_resp

        candidates = provider.search_candidate_scenes(
            bbox=(16.40, 48.20, 16.41, 48.21),
            datetime_range="2021-01-01T00:00:00Z/2021-12-31T23:59:59Z",
        )

        assert len(candidates) == 5
        assert candidates[0]["id"] == "S2_TEST_0"


def test_no_scenes_found_returns_real_failure():
    """When STAC returns 0 features, the provider must return REAL_FAILURE with NO_SCENES_FOUND."""
    provider = Sentinel2Provider()

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"features": []}
        mock_post.return_value = mock_resp

        result = provider.search_and_fetch(
            time_start="2021",
            time_end="2025",
            aoi=[16.40, 48.20, 16.41, 48.21],
        )

        assert result["status"] == "REAL_FAILURE"
        assert result["source"] == "REAL_SENTINEL_2"
        assert result["error_type"] == Sentinel2ErrorType.NO_SCENES_FOUND
        assert "No Sentinel-2 Level-2A scenes found" in result["error"]
        assert result["images"] == []
        # MUST NEVER FALL BACK TO SAMPLE IN PRODUCTION
        assert result["status"] != "success"
        assert result["source"] != "SAMPLE_SENTINEL_2_FALLBACK"


def test_stac_unavailable_returns_real_failure():
    """When STAC network request fails, return REAL_FAILURE with STAC_UNAVAILABLE."""
    provider = Sentinel2Provider()

    with patch("requests.post", side_effect=Exception("Connection refused")):
        result = provider.search_and_fetch(
            time_start="2021",
            time_end="2025",
            aoi=[16.40, 48.20, 16.41, 48.21],
        )

        assert result["status"] == "REAL_FAILURE"
        assert result["source"] == "REAL_SENTINEL_2"
        assert result["error_type"] == Sentinel2ErrorType.STAC_UNAVAILABLE
        assert result["images"] == []
        assert result["source"] != "SAMPLE_SENTINEL_2_FALLBACK"


def test_malformed_aoi_raises_error():
    """Invalid bounding box (e.g. west >= east) triggers MALFORMED_AOI error."""
    provider = Sentinel2Provider()
    # Malformed bbox: west 16.50 > east 16.40
    malformed_bbox = (16.50, 48.20, 16.40, 48.21)

    with pytest.raises(Sentinel2RetrievalError) as exc_info:
        provider.search_candidate_scenes(
            bbox=malformed_bbox,
            datetime_range="2021-01-01T00:00:00Z/2021-12-31T23:59:59Z",
        )
    assert exc_info.value.error_type == Sentinel2ErrorType.MALFORMED_AOI


# ============================================================
# 5. REAL METADATA PRESERVATION
# ============================================================

def test_stac_metadata_preserved_in_selection():
    """Selected scene output preserves all real STAC metadata without fabrication."""
    provider = Sentinel2Provider()

    before_item = {
        "id": "S2B_MSIL2A_20210615T100029_N0300_R122_T33UXP_20210615T121500",
        "collection": "sentinel-2-l2a",
        "bbox": [16.0, 48.0, 17.0, 49.0],
        "properties": {
            "datetime": "2021-06-15T10:00:29.024Z",
            "eo:cloud_cover": 1.25,
            "platform": "Sentinel-2B",
            "s2:processing_baseline": "03.00",
            "s2:mgrs_tile": "33UXP",
        },
    }
    after_item = {
        "id": "S2A_MSIL2A_20250618T100031_N0400_R122_T33UXP_20250618T140000",
        "collection": "sentinel-2-l2a",
        "bbox": [16.0, 48.0, 17.0, 49.0],
        "properties": {
            "datetime": "2025-06-18T10:00:31.024Z",
            "eo:cloud_cover": 2.10,
            "platform": "Sentinel-2A",
            "s2:processing_baseline": "04.00",
            "s2:mgrs_tile": "33UXP",
        },
    }

    with patch.object(provider, "search_candidate_scenes") as mock_search, \
         patch.object(provider, "fetch_scene_bands") as mock_fetch:

        mock_search.side_effect = [[before_item], [after_item]]
        mock_fetch.return_value = {
            "red": "/path/to/red.tif",
            "green": "/path/to/green.tif",
            "nir": "/path/to/nir.tif",
            "swir": "/path/to/swir.tif",
        }

        res = provider.search_and_fetch(
            time_start="2021",
            time_end="2025",
            aoi=[16.40, 48.20, 16.41, 48.21],
        )

        assert res["status"] == "REAL_SUCCESS"
        assert res["source"] == "REAL_SENTINEL_2"

        # Check 'before' structure
        b = res["before"]
        assert b["scene_id"] == before_item["id"]
        assert b["date"] == "2021-06-15"
        assert b["cloud_cover"] == 1.25
        assert b["coverage"] == 100.0
        assert b["platform"] == "Sentinel-2B"
        assert b["processing_baseline"] == "03.00"
        assert b["mgrs_tile"] == "33UXP"
        assert "Selected with quality score" in b["selection_reason"]

        # Check 'after' structure
        a = res["after"]
        assert a["scene_id"] == after_item["id"]
        assert a["date"] == "2025-06-18"
        assert a["cloud_cover"] == 2.10
        assert a["coverage"] == 100.0
        assert a["platform"] == "Sentinel-2A"
        assert a["processing_baseline"] == "04.00"
        assert a["mgrs_tile"] == "33UXP"

        # Check compatibility with downstream executor
        assert len(res["images"]) == 2
        assert res["images"][0]["bands"]["nir"] == "/path/to/nir.tif"


# ============================================================
# 6. AOI NORMALIZATION CONTINUITY
# ============================================================

def test_aoi_normalization_formats():
    """Verify normalize_aoi correctly handles list, GeoJSON Polygon, and dict."""
    # List [w, s, e, n]
    assert normalize_aoi([16.40, 48.20, 16.41, 48.21]) == (16.40, 48.20, 16.41, 48.21)

    # GeoJSON Polygon
    geojson = {
        "type": "Polygon",
        "coordinates": [[
            [16.40, 48.20],
            [16.41, 48.20],
            [16.41, 48.21],
            [16.40, 48.21],
            [16.40, 48.20],
        ]]
    }
    w, s, e, n = normalize_aoi(geojson)
    assert pytest.approx(w) == 16.40
    assert pytest.approx(s) == 48.20
    assert pytest.approx(e) == 16.41
    assert pytest.approx(n) == 48.21
