"""
Unit Test Suite for Sentinel-2 + Sentinel-1 Optical-SAR Scene Pairing (Step 13).

Covers all 12 required specifications:
1. Test 1: Best pair selection (multi-candidate scoring selects best pair)
2. Test 2: Temporal proximity (closer SAR acquisition beats farther one)
3. Test 3: Temporal limit (SAR candidate beyond max_temporal_delta_days rejected)
4. Test 4: Spatial compatibility (SAR scene without adequate AOI overlap rejected)
5. Test 5: Dual polarization preference (VV+VH preferred over VV-only)
6. Test 6: VV-only acceptance (valid VV-only scene accepted when no dual-pol candidate)
7. Test 7: No optical scene (clean failure)
8. Test 8: No SAR scene (clean failure)
9. Test 9: No compatible pair (clean failure when candidate separation exceeds window)
10. Test 10: Deterministic ranking (same inputs produce identical winner and scores)
11. Test 11: Selection reason explanation (result includes explainable selection reason)
12. Test 12: Downstream compatibility (returned paths validate via validate_optical_sar_pair)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from app.remote_sensing.multimodal.pairing import (
    DEFAULT_MAX_TEMPORAL_DELTA_DAYS,
    OpticalSarPairingError,
    PairingErrorType,
    evaluate_candidate_pair,
    find_optical_sar_pair,
    rank_candidate_pairs,
)
from app.remote_sensing.multimodal.optical_sar import validate_optical_sar_pair
from app.remote_sensing.providers.sentinel2 import Sentinel2Provider, Sentinel2RetrievalError
from app.remote_sensing.providers.sentinel1 import Sentinel1Provider, Sentinel1RetrievalError


# ============================================================
# MOCK FIXTURES & HELPERS
# ============================================================

def _make_mock_s2_scene(
    scene_id: str = "S2A_MSIL2A_20210627T101031_T32UQA",
    datetime_str: str = "2021-06-27T10:10:31Z",
    cloud_cover: float = 1.0,
    bbox: list = [13.0, 48.0, 13.5, 48.5],
) -> dict:
    return {
        "id": scene_id,
        "collection": "sentinel-2-l2a",
        "datetime": datetime_str,
        "bbox": bbox,
        "properties": {
            "datetime": datetime_str,
            "platform": "Sentinel-2A",
            "eo:cloud_cover": cloud_cover,
            "s2:processing_baseline": "03.00",
        },
        "assets": {
            "B04": {"href": f"https://mockblob.blob.core.windows.net/{scene_id}_B04.tif"},
            "B03": {"href": f"https://mockblob.blob.core.windows.net/{scene_id}_B03.tif"},
            "B02": {"href": f"https://mockblob.blob.core.windows.net/{scene_id}_B02.tif"},
            "B08": {"href": f"https://mockblob.blob.core.windows.net/{scene_id}_B08.tif"},
        },
    }


def _make_mock_s1_scene(
    scene_id: str = "S1B_IW_GRDH_1SDV_20210627T165835_027545",
    datetime_str: str = "2021-06-27T16:58:35Z",
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
        "collection": "sentinel-1-grd",
        "datetime": datetime_str,
        "bbox": bbox,
        "properties": {
            "datetime": datetime_str,
            "platform": "Sentinel-1B",
            "sar:instrument_mode": mode,
            "sar:product_type": "GRD",
            "sar:polarizations": pols,
            "sat:orbit_state": orbit_state,
        },
        "assets": assets,
    }


def _create_mock_geotiff(path: Path, count: int = 1, width: int = 20, height: int = 20, value: int = 150):
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


# ============================================================
# UNIT TESTS
# ============================================================

def test_1_best_pair_selection():
    """
    Test 1: Given multiple optical and SAR candidates, the optimal pair is selected.
    """
    aoi_bbox = (13.0, 48.0, 13.05, 48.05)

    opt1 = _make_mock_s2_scene("opt_clean", "2021-06-27T10:00:00Z", cloud_cover=1.0)
    opt2 = _make_mock_s2_scene("opt_cloudy", "2021-06-25T10:00:00Z", cloud_cover=45.0)

    sar1 = _make_mock_s1_scene("sar_close_dual", "2021-06-27T18:00:00Z", has_vh=True, has_vv=True)
    sar2 = _make_mock_s1_scene("sar_far_single", "2021-06-29T18:00:00Z", has_vh=False, has_vv=True)

    ranked = rank_candidate_pairs(
        optical_candidates=[opt1, opt2],
        sar_candidates=[sar1, sar2],
        aoi_bbox=aoi_bbox,
        max_temporal_delta_days=3.0,
    )

    assert len(ranked) > 0
    best = ranked[0]
    assert best["optical_item"]["id"] == "opt_clean"
    assert best["sar_item"]["id"] == "sar_close_dual"
    assert best["score"] > ranked[-1]["score"]


def test_2_temporal_proximity():
    """
    Test 2: A closer SAR acquisition beats a farther one when both cover the AOI.
    """
    aoi_bbox = (13.0, 48.0, 13.05, 48.05)
    opt = _make_mock_s2_scene("opt_target", "2021-06-27T10:00:00Z", cloud_cover=2.0)

    # sar_close is 6 hours away (~0.25 days)
    sar_close = _make_mock_s1_scene("sar_close", "2021-06-27T16:00:00Z", has_vh=True)
    # sar_far is 2.5 days away
    sar_far = _make_mock_s1_scene("sar_far", "2021-06-29T22:00:00Z", has_vh=True)

    ranked = rank_candidate_pairs(
        optical_candidates=[opt],
        sar_candidates=[sar_close, sar_far],
        aoi_bbox=aoi_bbox,
        max_temporal_delta_days=3.0,
    )

    assert len(ranked) == 2
    assert ranked[0]["sar_item"]["id"] == "sar_close"
    assert ranked[0]["temporal_delta_days"] < ranked[1]["temporal_delta_days"]
    assert ranked[0]["score"] > ranked[1]["score"]


def test_3_temporal_limit():
    """
    Test 3: A SAR candidate beyond max_temporal_delta_days is rejected.
    """
    aoi_bbox = (13.0, 48.0, 13.05, 48.05)
    opt = _make_mock_s2_scene("opt_base", "2021-06-20T10:00:00Z")
    # 5 days later (> 3.0 limit)
    sar_distant = _make_mock_s1_scene("sar_distant", "2021-06-25T10:00:00Z")

    is_compat, eval_info, reason = evaluate_candidate_pair(
        optical_item=opt,
        sar_item=sar_distant,
        aoi_bbox=aoi_bbox,
        max_temporal_delta_days=3.0,
    )

    assert is_compat is False
    assert eval_info is None
    assert "exceeds allowed maximum" in reason


def test_4_spatial_compatibility():
    """
    Test 4: A SAR scene without sufficient AOI overlap is rejected.
    """
    aoi_bbox = (13.0, 48.0, 13.05, 48.05)
    opt = _make_mock_s2_scene("opt_cov", "2021-06-27T10:00:00Z", bbox=[13.0, 48.0, 13.5, 48.5])
    # sar_remote is completely outside the AOI (longitude 20 vs 13)
    sar_remote = _make_mock_s1_scene("sar_remote", "2021-06-27T12:00:00Z", bbox=[20.0, 48.0, 20.5, 48.5])

    is_compat, eval_info, reason = evaluate_candidate_pair(
        optical_item=opt,
        sar_item=sar_remote,
        aoi_bbox=aoi_bbox,
        min_aoi_coverage_percent=50.0,
    )

    assert is_compat is False
    assert eval_info is None
    assert "coverage" in reason.lower() or "overlap" in reason.lower()


def test_5_dual_polarization_preference():
    """
    Test 5: VV+VH is preferred over VV-only when otherwise comparable.
    """
    aoi_bbox = (13.0, 48.0, 13.05, 48.05)
    opt = _make_mock_s2_scene("opt_same", "2021-06-27T10:00:00Z")

    sar_dual = _make_mock_s1_scene("sar_dual", "2021-06-27T12:00:00Z", has_vh=True, has_vv=True)
    sar_single = _make_mock_s1_scene("sar_single", "2021-06-27T12:00:00Z", has_vh=False, has_vv=True)

    ranked = rank_candidate_pairs(
        optical_candidates=[opt],
        sar_candidates=[sar_dual, sar_single],
        aoi_bbox=aoi_bbox,
        prefer_dual_pol=True,
    )

    assert len(ranked) == 2
    assert ranked[0]["sar_item"]["id"] == "sar_dual"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_6_vv_only_acceptance():
    """
    Test 6: A valid VV-only SAR scene remains acceptable when no dual-pol candidate exists.
    """
    aoi_bbox = (13.0, 48.0, 13.05, 48.05)
    opt = _make_mock_s2_scene("opt_single_pol_test", "2021-06-27T10:00:00Z")
    sar_vv = _make_mock_s1_scene("sar_vv_only", "2021-06-27T12:00:00Z", has_vh=False, has_vv=True)

    is_compat, eval_info, _ = evaluate_candidate_pair(
        optical_item=opt,
        sar_item=sar_vv,
        aoi_bbox=aoi_bbox,
    )

    assert is_compat is True
    assert eval_info is not None
    assert eval_info["polarizations"] == ["VV"]


def test_7_no_optical_scene_clean_failure():
    """
    Test 7: Clean failure when no optical scenes match query.
    """
    mock_s2 = MagicMock(spec=Sentinel2Provider)
    mock_s2.search_candidate_scenes.return_value = []
    mock_s1 = MagicMock(spec=Sentinel1Provider)

    res = find_optical_sar_pair(
        aoi=[13.0, 48.0, 13.05, 48.05],
        time_start="2021-06-25",
        time_end="2021-06-28",
        s2_provider=mock_s2,
        s1_provider=mock_s1,
        fetch_data=False,
    )

    assert res["status"] == "REAL_FAILURE"
    assert res["pair_found"] is False
    assert res["error_type"] == PairingErrorType.NO_OPTICAL_SCENES
    assert len(res["errors"]) > 0


def test_8_no_sar_scene_clean_failure():
    """
    Test 8: Clean failure when no SAR scenes match query.
    """
    mock_s2 = MagicMock(spec=Sentinel2Provider)
    mock_s2.search_candidate_scenes.return_value = [_make_mock_s2_scene()]
    mock_s1 = MagicMock(spec=Sentinel1Provider)
    mock_s1.search_candidate_scenes.return_value = []

    res = find_optical_sar_pair(
        aoi=[13.0, 48.0, 13.05, 48.05],
        time_start="2021-06-25",
        time_end="2021-06-28",
        s2_provider=mock_s2,
        s1_provider=mock_s1,
        fetch_data=False,
    )

    assert res["status"] == "REAL_FAILURE"
    assert res["pair_found"] is False
    assert res["error_type"] == PairingErrorType.NO_SAR_SCENES
    assert len(res["errors"]) > 0


def test_9_no_compatible_pair_clean_failure():
    """
    Test 9: Clean failure when candidates exist but temporal separation exceeds configured limit.
    """
    mock_s2 = MagicMock(spec=Sentinel2Provider)
    mock_s2.search_candidate_scenes.return_value = [
        _make_mock_s2_scene("s2_cand", "2021-06-01T10:00:00Z")
    ]
    mock_s1 = MagicMock(spec=Sentinel1Provider)
    mock_s1.search_candidate_scenes.return_value = [
        _make_mock_s1_scene("s1_cand", "2021-06-25T10:00:00Z")  # 24 days gap
    ]

    res = find_optical_sar_pair(
        aoi=[13.0, 48.0, 13.05, 48.05],
        time_start="2021-06-01",
        time_end="2021-06-30",
        max_temporal_delta_days=3.0,
        s2_provider=mock_s2,
        s1_provider=mock_s1,
        fetch_data=False,
    )

    assert res["status"] == "REAL_FAILURE"
    assert res["pair_found"] is False
    assert res["error_type"] == PairingErrorType.NO_TEMPORALLY_COMPATIBLE_PAIR
    assert "No compatible Optical-SAR pair found" in res["error"]


def test_10_deterministic_ranking():
    """
    Test 10: Repeated runs on the same candidates produce identical selected pairs and scores.
    """
    aoi_bbox = (13.0, 48.0, 13.05, 48.05)
    opts = [
        _make_mock_s2_scene("s2_1", "2021-06-27T10:00:00Z", cloud_cover=5.0),
        _make_mock_s2_scene("s2_2", "2021-06-27T10:00:00Z", cloud_cover=12.0),
    ]
    sars = [
        _make_mock_s1_scene("s1_1", "2021-06-27T14:00:00Z", has_vh=True),
        _make_mock_s1_scene("s1_2", "2021-06-27T18:00:00Z", has_vh=True),
    ]

    run1 = rank_candidate_pairs(opts, sars, aoi_bbox)
    run2 = rank_candidate_pairs(opts, sars, aoi_bbox)

    assert len(run1) == len(run2) == 4
    for p1, p2 in zip(run1, run2):
        assert p1["optical_item"]["id"] == p2["optical_item"]["id"]
        assert p1["sar_item"]["id"] == p2["sar_item"]["id"]
        assert p1["score"] == pytest.approx(p2["score"], abs=1e-6)
        assert p1["selection_reason"] == p2["selection_reason"]


def test_11_selection_reason_explanation():
    """
    Test 11: Returned successful pair exposes an explainable selection_reason.
    """
    mock_s2 = MagicMock(spec=Sentinel2Provider)
    mock_s2.search_candidate_scenes.return_value = [
        _make_mock_s2_scene("s2_winner", "2021-06-27T10:00:00Z", cloud_cover=0.8)
    ]
    mock_s1 = MagicMock(spec=Sentinel1Provider)
    mock_s1.search_candidate_scenes.return_value = [
        _make_mock_s1_scene("s1_winner", "2021-06-27T14:00:00Z", has_vh=True)
    ]

    res = find_optical_sar_pair(
        aoi=[13.0, 48.0, 13.05, 48.05],
        time_start="2021-06-25",
        time_end="2021-06-28",
        s2_provider=mock_s2,
        s1_provider=mock_s1,
        fetch_data=False,
    )

    assert res["status"] == "REAL_SUCCESS"
    assert res["pair_found"] is True
    reason = res["selection_reason"]
    assert "s2_winner" in reason
    assert "s1_winner" in reason
    assert "cloud" in reason.lower()
    assert "temporal delta" in reason.lower()
    assert "score" in reason.lower()


def test_12_downstream_compatibility(tmp_path):
    """
    Test 12: Returned local paths can be consumed directly by validate_optical_sar_pair.
    """
    opt_tiff = tmp_path / "mock_opt.tif"
    sar_vv_tiff = tmp_path / "mock_sar_vv.tif"
    sar_vh_tiff = tmp_path / "mock_sar_vh.tif"

    _create_mock_geotiff(opt_tiff, count=3, value=100)
    _create_mock_geotiff(sar_vv_tiff, count=1, value=200)
    _create_mock_geotiff(sar_vh_tiff, count=1, value=50)

    mock_s2 = MagicMock(spec=Sentinel2Provider)
    mock_s2.cache_dir = tmp_path
    mock_s2.search_candidate_scenes.return_value = [_make_mock_s2_scene("s2_test")]
    mock_s2.fetch_scene_bands.return_value = {
        "red": str(opt_tiff),
        "green": str(opt_tiff),
        "blue": str(opt_tiff),
    }

    mock_s1 = MagicMock(spec=Sentinel1Provider)
    mock_s1.cache_dir = tmp_path
    mock_s1.search_candidate_scenes.return_value = [_make_mock_s1_scene("s1_test")]
    mock_s1.fetch_and_cache_polarization.side_effect = lambda scene_item, polarization, bbox: (
        str(sar_vv_tiff) if polarization == "vv" else str(sar_vh_tiff)
    )

    res = find_optical_sar_pair(
        aoi=[13.0, 48.0, 13.02, 48.02],
        time_start="2021-06-25",
        time_end="2021-06-28",
        s2_provider=mock_s2,
        s1_provider=mock_s1,
        fetch_data=True,
    )

    assert res["status"] == "REAL_SUCCESS"
    assert res["pair_found"] is True
    assert res["optical"]["path"] is not None
    assert res["sar"]["path"] is not None

    # Downstream verification with validate_optical_sar_pair
    val = validate_optical_sar_pair(res["optical"]["path"], res["sar"]["path"])
    assert val["valid"] is True
    assert val["compatibility"]["spatial_overlap"] is True
