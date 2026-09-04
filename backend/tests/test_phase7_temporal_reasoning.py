"""
Phase 7 Dedicated Test Suite: Advanced Temporal Reasoning.

Covers:
1. Temporal observation model
2. Multi-observation ordering
3. Missing observation handling
4. Quality filtering
5. Seasonal comparability
6. Net change calculation
7. Annualized slope calculation
8. Stable series (within deadband)
9. Increasing series
10. Decreasing series
11. Mixed series
12. Persistence calculation
13. Direction consistency
14. Sudden change detection
15. Gradual change detection
16. Reversal detection
17. Acceleration detection
18. Deceleration detection
19. Insufficient data handling (bi-temporal limitation)
20. Pixel-level persistence raster generation
21. Spatial-temporal candidate region integration
22. Temporal transition ordering
23. API / Schema integration
24. Phase 5 Multi-Index Evidence regression
25. Phase 6 Spatial Reasoning regression
26. Deterministic repeatability
"""

import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.evidence.temporal import (
    TemporalConfig,
    TemporalObservation,
    TrendResult,
    calculate_seasonal_comparability,
    calculate_temporal_trend,
    calculate_pixel_level_temporal_persistence,
    calculate_spatial_temporal_region_stats,
    evaluate_transition_temporal_ordering,
    build_temporal_analysis_package,
)
from app.schemas.query import QueryPlan, QueryRequest
from app.agent.parser import parse_query, detect_temporal_mode
from app.schemas.analysis import AnalysisResult


# ============================================================
# TEST HELPERS / FACTORIES
# ============================================================

def make_obs(
    obs_id: str,
    date_str: str,
    doy: int,
    ndvi: float = 0.50,
    ndbi: float = -0.20,
    ndwi: float = -0.10,
    valid_fraction: float = 1.0,
    cloud_cover: float = 0.0,
) -> TemporalObservation:
    yr = int(date_str[:4])
    return TemporalObservation(
        observation_id=obs_id,
        scene_id=obs_id,
        datetime_iso=f"{date_str}T10:30:00Z",
        date=date_str,
        year=yr,
        day_of_year=doy,
        cloud_cover=cloud_cover,
        coverage_fraction=1.0,
        valid_fraction=valid_fraction,
        quality_state="high" if valid_fraction >= 0.85 else "low",
        acquisition_score=0.95,
        ndvi_mean=ndvi,
        ndvi_median=ndvi,
        ndvi_std=0.04,
        ndwi_mean=ndwi,
        ndwi_median=ndwi,
        ndwi_std=0.03,
        ndbi_mean=ndbi,
        ndbi_median=ndbi,
        ndbi_std=0.03,
    )


# ============================================================
# 1. OBSERVATION MODEL & SERIALIZATION
# ============================================================

def test_temporal_observation_model():
    obs = make_obs("S2A_2021", "2021-06-15", 166, ndvi=0.55)
    d = obs.to_dict()
    assert d["observation_id"] == "S2A_2021"
    assert d["year"] == 2021
    assert d["day_of_year"] == 166
    assert d["ndvi_mean"] == 0.55
    assert d["valid_fraction"] == 1.0


# ============================================================
# 2. MULTI-OBSERVATION ORDERING
# ============================================================

def test_multi_observation_ordering():
    # Pass out-of-order observations
    obs1 = make_obs("obs_2024", "2024-06-15", 167, ndvi=0.40)
    obs2 = make_obs("obs_2021", "2021-06-15", 166, ndvi=0.55)
    obs3 = make_obs("obs_2023", "2023-06-15", 166, ndvi=0.45)

    trend = calculate_temporal_trend([obs1, obs2, obs3], "ndvi")
    assert trend.first_value == 0.55
    assert trend.last_value == 0.40
    assert trend.net_change == -0.15
    assert trend.direction == "decreasing"


# ============================================================
# 3. MISSING OBSERVATION HANDLING
# ============================================================

def test_missing_observation_handling():
    # Only 2021, 2023, 2025 (2022 and 2024 genuinely missing)
    obs2021 = make_obs("obs_2021", "2021-06-15", 166, ndvi=0.60)
    obs2023 = make_obs("obs_2023", "2023-06-15", 166, ndvi=0.50)
    obs2025 = make_obs("obs_2025", "2025-06-15", 166, ndvi=0.40)

    trend = calculate_temporal_trend([obs2021, obs2023, obs2025], "ndvi")
    assert trend.usable_observation_count == 3
    assert trend.direction == "decreasing"
    assert trend.elapsed_years == 4.0
    # Annualized slope should be ~ -0.05 / yr
    assert pytest.approx(trend.annualized_slope, abs=0.01) == -0.05


# ============================================================
# 4. QUALITY FILTERING
# ============================================================

def test_quality_filtering_excludes_cloudy():
    obs1 = make_obs("obs_1", "2021-06-15", 166, ndvi=0.55, valid_fraction=0.90)
    # Cloud-contaminated observation with low valid fraction
    obs2_cloud = make_obs("obs_2", "2022-06-15", 166, ndvi=0.10, valid_fraction=0.20)
    obs3 = make_obs("obs_3", "2023-06-15", 166, ndvi=0.53, valid_fraction=0.88)

    trend = calculate_temporal_trend([obs1, obs2_cloud, obs3], "ndvi")
    assert trend.observation_count == 3
    assert trend.usable_observation_count == 2
    # obs2_cloud excluded, leaving only 0.55 and 0.53 (net change -0.02 is within deadband)
    assert trend.direction == "stable"


# ============================================================
# 5. SEASONAL COMPARABILITY
# ============================================================

def test_seasonal_comparability_high_and_low():
    # High: All around DOY 165-170
    obs_summer1 = make_obs("o1", "2021-06-15", 166)
    obs_summer2 = make_obs("o2", "2022-06-18", 169)
    comp_high = calculate_seasonal_comparability([obs_summer1, obs_summer2])
    assert comp_high["comparability"] == "high"
    assert comp_high["max_doy_difference"] <= 45

    # Low: June (166) vs December (350) -> ~184 days diff
    obs_winter = make_obs("o3", "2022-12-15", 349)
    comp_low = calculate_seasonal_comparability([obs_summer1, obs_winter])
    assert comp_low["comparability"] == "low"
    assert comp_low["max_doy_difference"] > 90


# ============================================================
# 6. NET CHANGE
# ============================================================

def test_net_change_calculation():
    obs1 = make_obs("o1", "2021-06-15", 166, ndvi=0.45)
    obs2 = make_obs("o2", "2025-06-15", 166, ndvi=0.60)
    trend = calculate_temporal_trend([obs1, obs2], "ndvi")
    assert trend.net_change == 0.15
    assert trend.first_value == 0.45
    assert trend.last_value == 0.60


# ============================================================
# 7. SLOPE CALCULATION
# ============================================================

def test_annualized_slope_linear_regression():
    # Exactly 0.02 increase per year across 4 years (2021 to 2025)
    obs = [
        make_obs("o1", "2021-06-15", 166, ndbi=0.10),
        make_obs("o2", "2022-06-15", 166, ndbi=0.12),
        make_obs("o3", "2023-06-15", 166, ndbi=0.14),
        make_obs("o4", "2024-06-15", 166, ndbi=0.16),
        make_obs("o5", "2025-06-15", 166, ndbi=0.18),
    ]
    trend = calculate_temporal_trend(obs, "ndbi")
    assert pytest.approx(trend.annualized_slope, abs=0.002) == 0.02
    assert trend.direction == "increasing"


# ============================================================
# 8. STABLE SERIES WITHIN DEADBAND
# ============================================================

def test_stable_series_inside_deadband():
    # Minor noise around 0.50 (within 0.03 deadband)
    obs = [
        make_obs("o1", "2021-06-15", 166, ndvi=0.50),
        make_obs("o2", "2022-06-15", 166, ndvi=0.51),
        make_obs("o3", "2023-06-15", 166, ndvi=0.49),
        make_obs("o4", "2024-06-15", 166, ndvi=0.51),
        make_obs("o5", "2025-06-15", 166, ndvi=0.50),
    ]
    trend = calculate_temporal_trend(obs, "ndvi")
    assert trend.direction == "stable"
    assert trend.change_type == "stable"


# ============================================================
# 9. INCREASING SERIES
# ============================================================

def test_increasing_series():
    obs = [
        make_obs("o1", "2021-06-15", 166, ndbi=-0.10),
        make_obs("o2", "2022-06-15", 166, ndbi=-0.05),
        make_obs("o3", "2023-06-15", 166, ndbi=0.02),
        make_obs("o4", "2024-06-15", 166, ndbi=0.08),
    ]
    trend = calculate_temporal_trend(obs, "ndbi")
    assert trend.direction == "increasing"
    assert trend.persistent_change is True
    assert trend.persistence_fraction == 1.0


# ============================================================
# 10. DECREASING SERIES
# ============================================================

def test_decreasing_series():
    obs = [
        make_obs("o1", "2021-06-15", 166, ndvi=0.65),
        make_obs("o2", "2022-06-15", 166, ndvi=0.58),
        make_obs("o3", "2023-06-15", 166, ndvi=0.50),
        make_obs("o4", "2024-06-15", 166, ndvi=0.42),
    ]
    trend = calculate_temporal_trend(obs, "ndvi")
    assert trend.direction == "decreasing"
    assert trend.persistent_change is True
    assert trend.net_change == -0.23


# ============================================================
# 11. MIXED SERIES
# ============================================================

def test_mixed_oscillating_series():
    # Alternating up and down significantly
    obs = [
        make_obs("o1", "2021-06-15", 166, ndvi=0.50),
        make_obs("o2", "2022-06-15", 166, ndvi=0.62),
        make_obs("o3", "2023-06-15", 166, ndvi=0.48),
        make_obs("o4", "2024-06-15", 166, ndvi=0.64),
    ]
    trend = calculate_temporal_trend(obs, "ndvi")
    assert trend.direction == "mixed"
    assert trend.change_type == "mixed"
    assert trend.persistent_change is False


# ============================================================
# 12. PERSISTENCE CALCULATION
# ============================================================

def test_persistence_calculation_threshold():
    # 3 intervals positive, 1 neutral -> 3/4 = 75% persistence
    obs = [
        make_obs("o1", "2021-06-15", 166, ndbi=0.10),
        make_obs("o2", "2022-06-15", 166, ndbi=0.14),
        make_obs("o3", "2023-06-15", 166, ndbi=0.18),
        make_obs("o4", "2024-06-15", 166, ndbi=0.19),  # small shift (+0.01)
        make_obs("o5", "2025-06-15", 166, ndbi=0.25),
    ]
    trend = calculate_temporal_trend(obs, "ndbi")
    assert trend.persistence_fraction >= TemporalConfig.PERSISTENCE_MIN_FRACTION
    assert trend.persistent_change is True


# ============================================================
# 13. DIRECTION CONSISTENCY
# ============================================================

def test_direction_consistency_metric():
    obs = [
        make_obs("o1", "2021-06-15", 166, ndvi=0.50),
        make_obs("o2", "2022-06-15", 166, ndvi=0.45),
        make_obs("o3", "2023-06-15", 166, ndvi=0.40),
        make_obs("o4", "2024-06-15", 166, ndvi=0.35),
    ]
    trend = calculate_temporal_trend(obs, "ndvi")
    assert trend.direction_consistency == 1.0


# ============================================================
# 14. SUDDEN CHANGE DETECTION
# ============================================================

def test_sudden_change_detection():
    # Stable, then one huge jump, then stable
    obs = [
        make_obs("o1", "2021-06-15", 166, ndbi=0.10),
        make_obs("o2", "2022-06-15", 166, ndbi=0.11),
        make_obs("o3", "2023-06-15", 166, ndbi=0.35),  # +0.24 sudden jump
        make_obs("o4", "2024-06-15", 166, ndbi=0.36),
    ]
    trend = calculate_temporal_trend(obs, "ndbi")
    assert trend.change_type == "sudden"


# ============================================================
# 15. GRADUAL CHANGE DETECTION
# ============================================================

def test_gradual_change_detection():
    # Steady increments
    obs = [
        make_obs("o1", "2021-06-15", 166, ndbi=0.10),
        make_obs("o2", "2022-06-15", 166, ndbi=0.15),
        make_obs("o3", "2023-06-15", 166, ndbi=0.20),
        make_obs("o4", "2024-06-15", 166, ndbi=0.26),
    ]
    trend = calculate_temporal_trend(obs, "ndbi")
    assert trend.change_type == "gradual"


# ============================================================
# 16. REVERSAL DETECTION
# ============================================================

def test_reversal_detection():
    # Vegetation decline then full recovery
    obs = [
        make_obs("o1", "2021-06-15", 166, ndvi=0.60),
        make_obs("o2", "2022-06-15", 166, ndvi=0.45),  # -0.15 loss
        make_obs("o3", "2023-06-15", 166, ndvi=0.58),  # +0.13 recovery
        make_obs("o4", "2024-06-15", 166, ndvi=0.61),
    ]
    trend = calculate_temporal_trend(obs, "ndvi")
    assert trend.reversal_detected is True
    assert trend.change_type == "reversal"
    assert trend.reversal_details["reversal_direction"] == "decrease_then_increase"


# ============================================================
# 17. ACCELERATION DETECTION
# ============================================================

def test_acceleration_detection():
    # Rate of growth increases each interval: +0.01, +0.03, +0.07
    obs = [
        make_obs("o1", "2021-06-15", 166, ndbi=0.10),
        make_obs("o2", "2022-06-15", 166, ndbi=0.11),
        make_obs("o3", "2023-06-15", 166, ndbi=0.14),
        make_obs("o4", "2024-06-15", 166, ndbi=0.21),
    ]
    trend = calculate_temporal_trend(obs, "ndbi")
    assert trend.acceleration_state == "accelerating"


# ============================================================
# 18. DECELERATION DETECTION
# ============================================================

def test_deceleration_detection():
    # Rate of growth slows down: +0.08, +0.03, +0.01
    obs = [
        make_obs("o1", "2021-06-15", 166, ndbi=0.10),
        make_obs("o2", "2022-06-15", 166, ndbi=0.18),
        make_obs("o3", "2023-06-15", 166, ndbi=0.21),
        make_obs("o4", "2024-06-15", 166, ndbi=0.22),
    ]
    trend = calculate_temporal_trend(obs, "ndbi")
    assert trend.acceleration_state == "decelerating"


# ============================================================
# 19. INSUFFICIENT DATA HANDLING (BI-TEMPORAL LIMITATION)
# ============================================================

def test_bitemporal_limitation():
    # Only 2 observations: cannot establish gradual/sudden
    obs = [
        make_obs("o1", "2021-06-15", 166, ndvi=0.50),
        make_obs("o2", "2025-06-15", 166, ndvi=0.40),
    ]
    trend = calculate_temporal_trend(obs, "ndvi")
    assert trend.data_sufficiency == "limited_bi_temporal"
    assert trend.change_type == "insufficient_data"
    assert trend.acceleration_state == "insufficient_data"


# ============================================================
# 20. PIXEL-LEVEL PERSISTENCE
# ============================================================

def test_pixel_level_persistence_synthetic(tmp_path):
    import rasterio
    from affine import Affine

    # Create 3 synthetic GeoTIFF observations
    h, w = 10, 10
    transform = Affine(0.0001, 0, 16.40, 0, -0.0001, 48.20)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": h,
        "width": w,
        "crs": "EPSG:4326",
        "transform": transform,
    }

    obs_list = []
    # Persistent urban cluster at center [4:6, 4:6]
    for i, yr in enumerate([2021, 2022, 2023]):
        swir = np.full((h, w), 0.20, dtype=np.float32)
        nir = np.full((h, w), 0.40, dtype=np.float32)
        if yr >= 2022:
            swir[4:7, 4:7] += 0.25  # urban increase

        p_swir = tmp_path / f"swir_{yr}.tif"
        p_nir = tmp_path / f"nir_{yr}.tif"
        with rasterio.open(p_swir, "w", **profile) as dst:
            dst.write(swir, 1)
        with rasterio.open(p_nir, "w", **profile) as dst:
            dst.write(nir, 1)

        o = make_obs(f"s2_{yr}", f"{yr}-06-15", 166)
        o.band_paths = {"swir": str(p_swir), "nir": str(p_nir)}
        obs_list.append(o)

    res = calculate_pixel_level_temporal_persistence(obs_list, "urban", output_dir=tmp_path)
    assert res["available"] is True
    assert res["pixel_counts"]["persistent"] > 0
    assert Path(res["raster_path"]).exists()


# ============================================================
# 21. SPATIAL-TEMPORAL REGION STATISTICS
# ============================================================

def test_spatial_temporal_region_integration():
    spatial_data = {
        "candidate_regions": [
            {
                "region_id": 1,
                "pixel_count": 25,
                "area_hectares": 0.25,
                "spatial_coherence": 0.72,
                "dominant_location": "south-western",
            }
        ]
    }
    obs = [
        make_obs("o1", "2021-06-15", 166, ndbi=0.10),
        make_obs("o2", "2022-06-15", 166, ndbi=0.15),
        make_obs("o3", "2023-06-15", 166, ndbi=0.20),
    ]

    reg_stats = calculate_spatial_temporal_region_stats(spatial_data, obs, "urban")
    assert len(reg_stats) == 1
    r0 = reg_stats[0]
    assert r0["region_id"] == 1
    assert r0["temporal_observation_count"] == 3
    assert r0["temporal_direction"] == "increasing"
    assert r0["persistent_fraction"] == 1.0


# ============================================================
# 22. TRANSITION TEMPORAL ORDERING
# ============================================================

def test_transition_temporal_ordering_valid():
    # Vegetation declines at year 2 (2022), urban expands at year 3 (2023)
    obs = [
        make_obs("o1", "2021-06-15", 166, ndvi=0.60, ndbi=-0.10),
        make_obs("o2", "2022-06-15", 166, ndvi=0.45, ndbi=-0.08),  # Veg drop
        make_obs("o3", "2023-06-15", 166, ndvi=0.42, ndbi=0.08),   # Urban gain
        make_obs("o4", "2024-06-15", 166, ndvi=0.40, ndbi=0.12),
    ]
    t_ord = evaluate_transition_temporal_ordering(obs)
    assert t_ord["available"] is True
    assert t_ord["temporal_order_valid"] is True
    assert t_ord["transition_temporal_support"] == "supported"


def test_transition_temporal_ordering_invalid():
    # Urban expands at year 2, vegetation only drops at year 4
    obs = [
        make_obs("o1", "2021-06-15", 166, ndvi=0.60, ndbi=-0.10),
        make_obs("o2", "2022-06-15", 166, ndvi=0.59, ndbi=0.15),  # Urban jump first
        make_obs("o3", "2023-06-15", 166, ndvi=0.58, ndbi=0.18),
        make_obs("o4", "2024-06-15", 166, ndvi=0.40, ndbi=0.20),  # Veg drop later
    ]
    t_ord = evaluate_transition_temporal_ordering(obs)
    assert t_ord["available"] is True
    assert t_ord["temporal_order_valid"] is False
    assert t_ord["transition_temporal_support"] == "opposing_order"


# ============================================================
# 23. API / SCHEMA / INTENT INTEGRATION
# ============================================================

def test_temporal_mode_detection():
    # 1. Bi-temporal comparison
    p1 = parse_query(QueryRequest(query="Compare urban change between 2021 and 2025"))
    assert p1.temporal_mode == "bi_temporal"

    # 2. Multi-temporal question
    p2 = parse_query(QueryRequest(query="How has vegetation changed from 2021 to 2025?"))
    assert p2.temporal_mode == "multi_temporal"

    # 3. Gradual / sudden inquiry
    p3 = parse_query(QueryRequest(query="Was urban change gradual or sudden between 2021 and 2025?"))
    assert p3.temporal_mode == "trend_analysis"

    # 4. Reversal / recovery inquiry
    p4 = parse_query(QueryRequest(query="Did vegetation recover between 2021 and 2025?"))
    assert p4.temporal_mode == "persistence_reversal"

    # 5. Acceleration inquiry
    p5 = parse_query(QueryRequest(query="Did urban expansion accelerate between 2021 and 2025?"))
    assert p5.temporal_mode == "acceleration"


def test_analysis_result_schema_temporal_field():
    res = AnalysisResult(
        status="success",
        answer="Analysis complete",
        temporal_analysis={"available": True, "observation_count": 4},
    )
    assert res.temporal_analysis["available"] is True
    assert res.temporal_analysis["observation_count"] == 4


# ============================================================
# 24. PHASE 5 MULTI-INDEX REGRESSION
# ============================================================

def test_phase5_regression():
    from app.evidence.multi_index import calculate_multi_index_evidence
    ev = calculate_multi_index_evidence(
        target="urban",
        task="urban_change",
        execution_results={},
        imagery_result=None,
        change_result=None,
    )
    assert "evidence_score" in ev
    assert "signals" in ev


# ============================================================
# 25. PHASE 6 SPATIAL REGRESSION
# ============================================================

def test_phase6_spatial_regression():
    from app.evidence.spatial import SpatialConfig, extract_spatial_candidate_regions
    assert SpatialConfig.MMU_MIN_PIXELS == 5
    sp = extract_spatial_candidate_regions(
        candidate_raster_path=None,
        target="urban",
    )
    assert sp["available"] is False
    assert sp["region_count"] == 0


# ============================================================
# 26. DETERMINISTIC REPEATABILITY
# ============================================================

def test_deterministic_repeatability():
    obs = [
        make_obs("o1", "2021-06-15", 166, ndvi=0.55),
        make_obs("o2", "2022-06-15", 166, ndvi=0.50),
        make_obs("o3", "2023-06-15", 166, ndvi=0.45),
        make_obs("o4", "2024-06-15", 166, ndvi=0.40),
    ]
    t1 = calculate_temporal_trend(obs, "ndvi")
    t2 = calculate_temporal_trend(obs, "ndvi")
    assert t1.to_dict() == t2.to_dict()


# ============================================================
# 27. MULTI-TEMPORAL RESPONSE EXPOSES OBSERVATION DATES & VALUES
# ============================================================

def test_multitemporal_response_exposes_observation_dates_and_values():
    """
    Regression test 1:
    Multi-temporal response exposes 5 observation dates/datetimes and index values (ndvi, ndwi, ndbi)
    consistently so ($r.temporal_analysis.observations.datetime -join ' | ') never yields empty values.
    """
    obs = [
        make_obs("v2020", "2020-04-22", 113, ndvi=0.6004, ndwi=-0.5510, ndbi=-0.1977),
        make_obs("v2021", "2021-03-31", 90,  ndvi=0.3825, ndwi=-0.4295, ndbi=0.0423),
        make_obs("v2022", "2022-03-28", 87,  ndvi=0.3376, ndwi=-0.3762, ndbi=0.0353),
        make_obs("v2023", "2023-02-09", 40,  ndvi=0.3296, ndwi=-0.3829, ndbi=0.0556),
        make_obs("v2024", "2024-05-09", 130, ndvi=0.6706, ndwi=-0.6108, ndbi=-0.2922),
    ]

    # Verify dataclass properties
    for o in obs:
        assert o.datetime == o.datetime_iso
        assert o.ndvi == o.ndvi_mean
        assert o.ndwi == o.ndwi_mean
        assert o.ndbi == o.ndbi_mean

    pkg = build_temporal_analysis_package(
        observations=obs,
        target="vegetation",
        task="vegetation_change",
    )

    assert pkg["available"] is True
    assert pkg["temporal_mode"] == "multi_temporal"
    assert pkg["observation_count"] == 5
    assert pkg["usable_observation_count"] == 5

    serialized_obs = pkg["observations"]
    assert len(serialized_obs) == 5

    # Verify PowerShell expression: ($r.temporal_analysis.observations.datetime -join " | ")
    datetimes = [o.get("datetime") for o in serialized_obs]
    assert all(isinstance(dt, str) and len(dt) > 0 for dt in datetimes)
    joined_dt = " | ".join(datetimes)
    assert joined_dt == "2020-04-22T10:30:00Z | 2021-03-31T10:30:00Z | 2022-03-28T10:30:00Z | 2023-02-09T10:30:00Z | 2024-05-09T10:30:00Z"

    # Verify dates and metrics
    dates = [o.get("date") for o in serialized_obs]
    assert dates == ["2020-04-22", "2021-03-31", "2022-03-28", "2023-02-09", "2024-05-09"]

    ndvis = [o.get("ndvi") for o in serialized_obs]
    assert ndvis == [0.6004, 0.3825, 0.3376, 0.3296, 0.6706]

    ndwis = [o.get("ndwi") for o in serialized_obs]
    assert ndwis == [-0.5510, -0.4295, -0.3762, -0.3829, -0.6108]

    ndbis = [o.get("ndbi") for o in serialized_obs]
    assert ndbis == [-0.1977, 0.0423, 0.0353, 0.0556, -0.2922]


# ============================================================
# 28. BI-TEMPORAL RESPONSE DOES NOT CLAIM MULTI-TEMPORAL PERSISTENCE
# ============================================================

def test_bitemporal_response_does_not_claim_multitemporal_persistence():
    """
    Regression test 2:
    Bi-temporal (N=2) series must not report a normal multi-temporal persistence metric.
    persistence_fraction and direction_consistency must be None / not applicable.
    """
    obs_bitemporal = [
        make_obs("o1", "2020-04-22", 113, ndvi=0.60),
        make_obs("o2", "2024-05-09", 130, ndvi=0.68),
    ]

    trend = calculate_temporal_trend(obs_bitemporal, "ndvi")
    assert trend.observation_count == 2
    assert trend.usable_observation_count == 2
    assert trend.direction == "increasing"
    assert trend.change_type == "insufficient_data"
    assert trend.data_sufficiency == "limited_bi_temporal"
    assert trend.reversal_detected is False

    # CRITICAL: Persistence must be None / unavailable for N=2
    assert trend.persistence_fraction is None
    assert trend.direction_consistency is None
    assert trend.persistent_change is False

    # Verify package serialization retains null / None
    pkg = build_temporal_analysis_package(obs_bitemporal, target="vegetation")
    assert pkg["temporal_mode"] == "bi_temporal"
    veg_domain = pkg["domains"]["vegetation"]
    assert veg_domain["persistence_fraction"] is None
    assert veg_domain["direction_consistency"] is None
    assert veg_domain["persistent_change"] is False
    assert veg_domain["change_type"] == "insufficient_data"

    # Verify interpretation mapping handles None without defaulting to 1.0 or 0.0
    from app.evidence.interpretation import generate_structured_interpretation
    interp = generate_structured_interpretation(
        candidate_package={"candidates": [], "primary_candidate": {}},
        multi_index_evidence={},
        target="vegetation",
        task="vegetation_change",
        temporal_analysis=pkg,
    )
    assert interp.get("persistence_fraction") is None
    assert interp.get("temporal_mode") == "bi_temporal"



# ============================================================
# 29. REVERSAL DETECTION IS BASED ON ACTUAL SEQUENCE NOT QUERY WORDING
# ============================================================

def test_reversal_detection_based_on_actual_sequence_not_query_wording():
    """
    Regression test 3:
    Reversal detection must be evaluated strictly from the actual physical index sequence:
    1. Multi-step decline + rebound [0.60, 0.38, 0.34, 0.33, 0.67] detects reversal regardless of query.
    2. Purely monotonic decrease [0.60, 0.50, 0.40, 0.30, 0.20] does NOT detect reversal even if query contains 'recover'.
    """
    # 1. Genuine physical reversal: 2020-2023 multi-step decline (delta = -0.2708 <= -0.04), 2023-2024 recovery (+0.3410 >= +0.04)
    obs_reversal = [
        make_obs("v2020", "2020-04-22", 113, ndvi=0.6004),
        make_obs("v2021", "2021-03-31", 90,  ndvi=0.3825),
        make_obs("v2022", "2022-03-28", 87,  ndvi=0.3376),
        make_obs("v2023", "2023-02-09", 40,  ndvi=0.3296),
        make_obs("v2024", "2024-05-09", 130, ndvi=0.6706),
    ]
    trend_rev = calculate_temporal_trend(obs_reversal, "ndvi")
    assert trend_rev.reversal_detected is True
    assert trend_rev.change_type == "reversal"
    assert trend_rev.direction == "mixed"
    assert trend_rev.reversal_details["reversal_direction"] == "decrease_then_increase"
    assert trend_rev.reversal_details["observation_inflection"] == "2023-02-09"

    # Even with a neutral or unrelated task, package respects the physical reversal
    pkg_rev = build_temporal_analysis_package(obs_reversal, target="vegetation", task="urban_expansion")
    assert pkg_rev["domains"]["vegetation"]["reversal_detected"] is True
    assert pkg_rev["domains"]["vegetation"]["change_type"] == "reversal"

    # 2. Pure monotonic decline: [0.60, 0.50, 0.40, 0.30, 0.20]
    obs_monotonic_decline = [
        make_obs("d1", "2020-04-22", 113, ndvi=0.60),
        make_obs("d2", "2021-04-22", 113, ndvi=0.50),
        make_obs("d3", "2022-04-22", 113, ndvi=0.40),
        make_obs("d4", "2023-04-22", 113, ndvi=0.30),
        make_obs("d5", "2024-04-22", 113, ndvi=0.20),
    ]

    # Query asking about recovery:
    query_text = "Did vegetation decline and then recover between 2020 and 2024 for AOI [16.40, 48.20, 16.41, 48.21]"
    plan = parse_query(QueryRequest(query=query_text))
    assert plan.temporal_mode == "persistence_reversal"

    # But physical calculation MUST NOT fabricate reversal
    trend_decline = calculate_temporal_trend(obs_monotonic_decline, "ndvi")
    assert trend_decline.reversal_detected is False
    assert trend_decline.direction == "decreasing"
    assert trend_decline.change_type == "gradual"
    assert trend_decline.persistent_change is True
    assert trend_decline.persistence_fraction == 1.0

    pkg_decline = build_temporal_analysis_package(obs_monotonic_decline, target=plan.target, task=plan.task)
    assert pkg_decline["domains"]["vegetation"]["reversal_detected"] is False
    assert pkg_decline["domains"]["vegetation"]["change_type"] != "reversal"

