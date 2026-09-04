"""
Phase 8: Reliability & Confidence Calibration Test Suite.

Verifies:
1. Reliability band classification & quality independence
2. Semantic evidence strength bands & physical signal mapping
3. Spatial assessment & geometric clustering separation
4. Temporal consistency & N=2 bi-temporal limitation
5. Interpretation support & multi-source corroboration (no fake probabilities)
6. Data sufficiency & No-change vs. Insufficient-data distinction
7. Conflicting indicators detection
8. Land-cover transition corroboration
9. Reason code taxonomy & consistency
10. Schema, API, and regression integrity
"""

import pytest
import numpy as np
from app.evidence.calibration import (
    CalibrationConfig,
    CalibrationReasonCodes,
    classify_evidence_strength,
    evaluate_observation_reliability,
    evaluate_semantic_evidence,
    evaluate_spatial_assessment,
    evaluate_temporal_consistency,
    detect_evidence_conflicts,
    evaluate_data_sufficiency,
    evaluate_transition_support,
    evaluate_overall_interpretation_support,
    build_calibration_package,
)
from app.evidence.temporal import TemporalObservation
from app.evidence.interpretation import generate_structured_interpretation
from app.schemas.analysis import AnalysisResult


# ============================================================
# 1. OBSERVATION RELIABILITY TESTS
# ============================================================

def test_observation_reliability_bands():
    """Verify deterministic mapping of reliability bands."""
    # High quality
    res_high = evaluate_observation_reliability(
        imagery_result={
            "images": [
                {"quality": {"valid_percentage": 95.0}, "cloud_cover": 2.0},
                {"quality": {"valid_percentage": 92.0}, "cloud_cover": 4.0},
            ]
        }
    )
    assert res_high.state == "high"
    assert res_high.score >= 0.80
    assert CalibrationReasonCodes.DATA_HIGH_QUALITY in res_high.reason_codes

    # Moderate quality
    res_mod = evaluate_observation_reliability(
        imagery_result={
            "images": [
                {"quality": {"valid_percentage": 65.0}, "cloud_cover": 15.0},
                {"quality": {"valid_percentage": 70.0}, "cloud_cover": 10.0},
            ]
        }
    )
    assert res_mod.state == "moderate"
    assert CalibrationReasonCodes.DATA_MODERATE_QUALITY in res_mod.reason_codes

    # Low quality
    res_low = evaluate_observation_reliability(
        imagery_result={
            "images": [
                {"quality": {"valid_percentage": 40.0}, "cloud_cover": 45.0},
                {"quality": {"valid_percentage": 30.0}, "cloud_cover": 55.0},
            ]
        }
    )
    assert res_low.state == "low"
    assert CalibrationReasonCodes.DATA_LOW_QUALITY in res_low.reason_codes
    assert CalibrationReasonCodes.INSUFFICIENT_VALID_PIXELS in res_low.reason_codes


def test_observation_reliability_zero_valid_pixels():
    """Edge case: 0 valid pixels should produce low reliability and insufficient pixels code."""
    res = evaluate_observation_reliability(
        imagery_result={
            "images": [
                {"quality": {"valid_percentage": 0.0}, "cloud_cover": 100.0}
            ]
        }
    )
    assert res.state == "low"
    assert res.valid_fraction == 0.0
    assert CalibrationReasonCodes.INSUFFICIENT_VALID_PIXELS in res.reason_codes


def test_seasonal_comparability_in_reliability():
    """Verify circular DOY seasonal comparability propagates to reliability notes."""
    obs_series = [
        TemporalObservation("obs1", "s1", "2021-06-15T00:00:00Z", "2021-06-15", 2021, 166, 0.0, 1.0, 0.95, "high", 1.0),
        TemporalObservation("obs2", "s2", "2022-12-15T00:00:00Z", "2022-12-15", 2022, 349, 0.0, 1.0, 0.90, "high", 1.0),
    ]
    res = evaluate_observation_reliability(temporal_observations=obs_series)
    # 349 - 166 = 183 days difference -> seasonal mismatch (> 90 days)
    assert res.seasonal_comparability == "low"
    assert CalibrationReasonCodes.SEASONAL_MISMATCH in res.reason_codes


# ============================================================
# 2. SEMANTIC EVIDENCE STRENGTH TESTS
# ============================================================

def test_evidence_strength_bands():
    """Verify deterministic bands: none, weak, moderate, strong, very_strong."""
    assert classify_evidence_strength(0.05) == "none"
    assert classify_evidence_strength(0.17) == "none"
    assert classify_evidence_strength(0.18) == "weak"
    assert classify_evidence_strength(0.30) == "weak"
    assert classify_evidence_strength(0.35) == "moderate"
    assert classify_evidence_strength(0.55) == "moderate"
    assert classify_evidence_strength(0.60) == "strong"
    assert classify_evidence_strength(0.75) == "strong"
    assert classify_evidence_strength(0.80) == "very_strong"
    assert classify_evidence_strength(0.95) == "very_strong"


def test_semantic_evidence_evaluation():
    """Verify semantic candidate evidence is categorized without changing scores."""
    cand_pkg = {
        "candidates": [
            {
                "target": "urban",
                "hypothesis": "urban_expansion",
                "final_evidence_score": 0.72,
                "state": "candidate",
            },
            {
                "target": "vegetation",
                "hypothesis": "vegetation_loss",
                "final_evidence_score": 0.45,
                "state": "candidate",
            }
        ]
    }
    sem_eval = evaluate_semantic_evidence(cand_pkg, target="urban")
    assert sem_eval.score == 0.72
    assert sem_eval.state == "strong"
    assert sem_eval.hypothesis == "urban_expansion"
    assert CalibrationReasonCodes.STRONG_SEMANTIC_SUPPORT in sem_eval.reason_codes
    assert "vegetation" in sem_eval.domains
    assert sem_eval.domains["vegetation"]["evidence_strength"] == "moderate"


# ============================================================
# 3. SPATIAL ASSESSMENT TESTS
# ============================================================

def test_spatial_assessment_bands():
    """Verify spatial coherence and contiguity mapping."""
    # High spatial coherence
    sp_high = evaluate_spatial_assessment({
        "available": True,
        "spatial_coherence": 0.65,
        "region_count": 4,
        "total_candidate_area_hectares": 12.5,
        "largest_region_area_hectares": 8.0,
        "dominant_location_description": "eastern portion",
    })
    assert sp_high.state == "high"
    assert CalibrationReasonCodes.STRONG_SPATIAL_SUPPORT in sp_high.reason_codes

    # Low / fragmented spatial coherence
    sp_frag = evaluate_spatial_assessment({
        "available": True,
        "spatial_coherence": 0.15,
        "region_count": 22,
        "total_candidate_area_hectares": 3.0,
        "largest_region_area_hectares": 0.3,
        "dominant_location_description": "scattered throughout",
    })
    assert sp_frag.state == "low"
    assert CalibrationReasonCodes.FRAGMENTED_CANDIDATE_REGIONS in sp_frag.reason_codes
    assert CalibrationReasonCodes.LOW_SPATIAL_COHERENCE in sp_frag.reason_codes


def test_spatial_assessment_zero_regions():
    """Zero candidate regions should result in low/no spatial support."""
    sp_zero = evaluate_spatial_assessment({
        "available": True,
        "spatial_coherence": 0.0,
        "region_count": 0,
        "total_candidate_area_hectares": 0.0,
    })
    assert sp_zero.state == "low"
    assert CalibrationReasonCodes.NO_SPATIAL_SUPPORT in sp_zero.reason_codes


# ============================================================
# 4. TEMPORAL CONSISTENCY & N=2 LIMITATION
# ============================================================

def test_temporal_consistency_bitemporal_limitation():
    """
    CRITICAL RULE: For N=2, temporal consistency MUST be 'bi_temporal_only'
    and must NOT assert or imply a trend.
    """
    temp_eval = evaluate_temporal_consistency({
        "available": True,
        "temporal_mode": "bi_temporal",
        "observation_count": 2,
        "usable_observation_count": 2,
    })
    assert temp_eval.state == "bi_temporal_only"
    assert temp_eval.score is None
    assert temp_eval.persistence_fraction is None
    assert CalibrationReasonCodes.BI_TEMPORAL_ONLY in temp_eval.reason_codes
    assert CalibrationReasonCodes.INSUFFICIENT_TEMPORAL_OBSERVATIONS in temp_eval.reason_codes


def test_temporal_consistency_multitemporal_series():
    """Verify N >= 3 series produces persistent/monotonic consistency."""
    temp_eval = evaluate_temporal_consistency({
        "available": True,
        "temporal_mode": "multi_temporal",
        "observation_count": 5,
        "usable_observation_count": 5,
        "primary_domain": "urban",
        "domains": {
            "urban": {
                "persistence_fraction": 0.80,
                "monotonicity": 0.90,
                "direction": "increase",
                "change_type": "gradual",
                "reversal_detected": False,
            }
        }
    })
    assert temp_eval.state == "high"
    assert temp_eval.score == 0.80
    assert CalibrationReasonCodes.STRONG_TEMPORAL_SUPPORT in temp_eval.reason_codes
    assert CalibrationReasonCodes.PERSISTENT_TRAJECTORY in temp_eval.reason_codes
    assert CalibrationReasonCodes.MONOTONIC_TREND in temp_eval.reason_codes


# ============================================================
# 5. DATA SUFFICIENCY & NO-CHANGE VS INSUFFICIENT-DATA
# ============================================================

def test_no_change_vs_insufficient_data():
    """
    CRITICAL DISTINCTION:
    Case A: High-quality observation + stable/neutral indices -> No strong evidence of change (NOT insufficient data).
    Case B: Low-quality imagery -> Insufficient data (NOT no change).
    """
    # Case A: High quality + neutral signals
    obs_rel_high = evaluate_observation_reliability(
        imagery_result={"images": [{"quality": {"valid_percentage": 95.0}, "cloud_cover": 0.0}]}
    )
    sem_ev_none = evaluate_semantic_evidence({
        "candidates": [{"target": "urban", "hypothesis": "neutral", "final_evidence_score": 0.05, "state": "no_support"}]
    })
    sp_eval = evaluate_spatial_assessment({"available": True, "region_count": 0})
    temp_eval = evaluate_temporal_consistency({"available": True, "temporal_mode": "bi_temporal", "observation_count": 2})
    data_suff_a = evaluate_data_sufficiency(obs_rel_high, temp_eval)
    
    interp_a = evaluate_overall_interpretation_support(
        obs_rel_high, sem_ev_none, sp_eval, temp_eval, data_suff_a, []
    )
    # The system correctly reports no strong evidence
    assert "No strong evidence" in interp_a.summary
    assert data_suff_a.state == "limited" or data_suff_a.state == "sufficient"

    # Case B: Low quality observation
    obs_rel_low = evaluate_observation_reliability(
        imagery_result={"images": [{"quality": {"valid_percentage": 25.0}, "cloud_cover": 70.0}]}
    )
    data_suff_b = evaluate_data_sufficiency(obs_rel_low, temp_eval)
    interp_b = evaluate_overall_interpretation_support(
        obs_rel_low, sem_ev_none, sp_eval, temp_eval, data_suff_b, []
    )
    assert interp_b.state == "insufficient_support"
    assert "Available imagery quality was insufficient" in interp_b.summary
    assert data_suff_b.state == "insufficient"


# ============================================================
# 6. CONFLICT DETECTION TESTS
# ============================================================

def test_conflicting_evidence_detection():
    """Verify simultaneous NDBI increase and NDVI increase is flagged as a conflict."""
    multi_index = {
        "signals": {
            "NDBI": {"raw_magnitude": 0.15},
            "NDVI": {"raw_magnitude": 0.14},
            "NDWI": {"raw_magnitude": -0.02},
        }
    }
    conflicts, codes = detect_evidence_conflicts(multi_index_evidence=multi_index)
    assert len(conflicts) > 0
    assert CalibrationReasonCodes.CONFLICTING_NDBI_NDVI_EXPANSION in codes
    assert CalibrationReasonCodes.CONFLICTING_INDICATORS in codes


def test_reversal_conflict_detection():
    """Verify temporal trajectory reversal surfaces in conflict detection."""
    temp_eval = evaluate_temporal_consistency({
        "available": True,
        "temporal_mode": "multi_temporal",
        "observation_count": 4,
        "domains": {
            "urban": {
                "persistence_fraction": 0.40,
                "reversal_detected": True,
            }
        }
    })
    conflicts, codes = detect_evidence_conflicts(temporal_consistency=temp_eval)
    assert any(c["type"] == "temporal_reversal" for c in conflicts)
    assert CalibrationReasonCodes.REVERSAL_PRESENT in codes


# ============================================================
# 7. TRANSITION SUPPORT TESTS
# ============================================================

def test_transition_support():
    """Verify multi-factor evaluation of vegetation-to-urban transition."""
    cand_pkg = {
        "candidates": [
            {
                "target": "transition",
                "hypothesis": "vegetation_to_urban",
                "final_evidence_score": 0.65,
                "details": {
                    "source_support": 0.55,
                    "destination_support": 0.60,
                }
            }
        ]
    }
    sp_analysis = {
        "available": True,
        "transition_overlap_fraction": 0.35,
    }
    temp_analysis = {
        "available": True,
        "transition_temporal_ordering": {
            "available": True,
            "sequence_supported": True,
        }
    }
    obs_rel = evaluate_observation_reliability(
        imagery_result={"images": [{"quality": {"valid_percentage": 90.0}}]}
    )

    trans_supp = evaluate_transition_support(
        candidate_package=cand_pkg,
        spatial_analysis=sp_analysis,
        temporal_analysis=temp_analysis,
        observation_reliability=obs_rel,
    )
    assert trans_supp is not None
    assert trans_supp.state == "supported_transition"
    assert trans_supp.spatial_support == "confirmed_spatial_overlap"
    assert trans_supp.temporal_support == "sequential_transition_supported"
    assert CalibrationReasonCodes.TRANSITION_SPATIAL_OVERLAP_CONFIRMED in trans_supp.reason_codes
    assert CalibrationReasonCodes.TRANSITION_TEMPORAL_ORDER_VALID in trans_supp.reason_codes


# ============================================================
# 8. MASTER CALIBRATION PACKAGE & API SCHEMA INTEGRATION
# ============================================================

def test_build_calibration_package_e2e():
    """Verify master builder produces complete structured package."""
    cand_pkg = {
        "candidates": [
            {
                "target": "urban",
                "hypothesis": "urban_expansion",
                "final_evidence_score": 0.70,
                "state": "candidate",
            }
        ]
    }
    sp_analysis = {
        "available": True,
        "spatial_coherence": 0.55,
        "region_count": 3,
        "total_candidate_area_hectares": 5.4,
        "largest_region_area_hectares": 3.2,
        "dominant_location_description": "northern region",
    }
    temp_obs = [
        TemporalObservation("obs1", "s1", "2021-06-15T00:00:00Z", "2021-06-15", 2021, 166, 0.0, 1.0, 0.95, "high", 1.0),
        TemporalObservation("obs2", "s2", "2025-06-15T00:00:00Z", "2025-06-15", 2025, 166, 0.0, 1.0, 0.92, "high", 1.0),
    ]

    cal_pkg = build_calibration_package(
        candidate_package=cand_pkg,
        spatial_analysis=sp_analysis,
        temporal_observations=temp_obs,
        target="urban",
        task="change_detection",
        temporal_mode="bi_temporal",
    )

    assert "observation_reliability" in cal_pkg
    assert "semantic_evidence" in cal_pkg
    assert "spatial_assessment" in cal_pkg
    assert "temporal_consistency" in cal_pkg
    assert "interpretation_support" in cal_pkg
    assert "data_sufficiency" in cal_pkg
    assert "reason_codes" in cal_pkg
    assert "calibration_notice" in cal_pkg
    assert cal_pkg["temporal_consistency"]["state"] == "bi_temporal_only"
    assert cal_pkg["interpretation_support"]["state"] == "strong_support"

    # Schema integration check
    ar = AnalysisResult(
        status="success",
        calibration=cal_pkg,
    )
    assert ar.calibration is not None
    assert ar.calibration["observation_reliability"]["state"] == "high"


def test_structured_interpretation_integration():
    """Verify generate_structured_interpretation incorporates calibration fields."""
    cand_pkg = {
        "candidates": [
            {
                "target": "urban",
                "hypothesis": "urban_expansion",
                "final_evidence_score": 0.65,
                "state": "candidate",
                "supporting_evidence": {"NDBI": 0.7},
                "opposing_evidence": {},
                "reliability": 0.9,
            }
        ]
    }
    cal_pkg = {
        "interpretation_support": {
            "state": "strong_support",
            "summary": "Multiple independent lines of physical and spatial evidence strongly corroborate the change interpretation.",
        },
        "observation_reliability": {
            "state": "high",
        },
        "semantic_evidence": {
            "state": "strong",
        },
        "data_sufficiency": {
            "state": "sufficient",
        },
        "reason_codes": [CalibrationReasonCodes.DATA_HIGH_QUALITY, CalibrationReasonCodes.STRONG_SEMANTIC_SUPPORT],
    }

    interp = generate_structured_interpretation(
        candidate_package=cand_pkg,
        multi_index_evidence={"signals": {}},
        target="urban",
        task="change_detection",
        calibration=cal_pkg,
    )

    assert interp["calibration_summary"] == cal_pkg["interpretation_support"]["summary"]
    assert interp["interpretation_support_state"] == "strong_support"
    assert interp["data_sufficiency_state"] == "sufficient"
    assert CalibrationReasonCodes.DATA_HIGH_QUALITY in interp["reason_codes"]


# ============================================================
# 9. ADDITIONAL EDGE CASES & REGRESSION TESTS
# ============================================================

def test_insufficient_trend_n_less_than_3():
    """Verify N < 3 observations produce limited/insufficient data for trend analysis."""
    obs_rel = evaluate_observation_reliability(
        imagery_result={"images": [{"quality": {"valid_percentage": 95.0}}, {"quality": {"valid_percentage": 95.0}}]}
    )
    temp_eval = evaluate_temporal_consistency({
        "available": True,
        "temporal_mode": "trend_analysis",
        "observation_count": 2,
        "usable_observation_count": 2,
    })
    data_suff = evaluate_data_sufficiency(obs_rel, temp_eval, task="trend_analysis", temporal_mode="trend_analysis")
    assert data_suff.state == "limited" or data_suff.state == "insufficient"
    assert any("Fewer than 3 usable observations" in m for m in data_suff.missing_components)


def test_insufficient_acceleration_n_less_than_4():
    """Verify N < 4 observations produce missing requirements for acceleration."""
    obs_rel = evaluate_observation_reliability(
        imagery_result={"images": [{"quality": {"valid_percentage": 95.0}}, {"quality": {"valid_percentage": 95.0}}, {"quality": {"valid_percentage": 95.0}}]}
    )
    temp_eval = evaluate_temporal_consistency({
        "available": True,
        "temporal_mode": "acceleration",
        "observation_count": 3,
        "usable_observation_count": 3,
    })
    data_suff = evaluate_data_sufficiency(obs_rel, temp_eval, task="acceleration", temporal_mode="acceleration")
    assert any("Fewer than 4 usable observations" in m for m in data_suff.missing_components)


def test_general_multidomain_calibration():
    """Verify multi-domain queries preserve separate domain assessments without collapsing."""
    cand_pkg = {
        "candidates": [
            {"target": "urban", "hypothesis": "urban_expansion", "final_evidence_score": 0.10, "state": "no_support"},
            {"target": "vegetation", "hypothesis": "vegetation_loss", "final_evidence_score": 0.52, "state": "candidate"},
            {"target": "water", "hypothesis": "water_loss", "final_evidence_score": 0.22, "state": "weak_candidate"},
        ]
    }
    sem_eval = evaluate_semantic_evidence(cand_pkg, target="none")
    assert "urban" in sem_eval.domains
    assert "vegetation" in sem_eval.domains
    assert "water" in sem_eval.domains
    assert sem_eval.domains["urban"]["evidence_strength"] == "none"
    assert sem_eval.domains["vegetation"]["evidence_strength"] == "moderate"
    assert sem_eval.domains["water"]["evidence_strength"] == "weak"


def test_deterministic_repeatability():
    """Verify multiple executions on identical data produce identical calibration packages."""
    pkg_1 = build_calibration_package(
        candidate_package={"candidates": [{"target": "urban", "final_evidence_score": 0.65}]},
        target="urban",
        temporal_mode="bi_temporal",
    )
    pkg_2 = build_calibration_package(
        candidate_package={"candidates": [{"target": "urban", "final_evidence_score": 0.65}]},
        target="urban",
        temporal_mode="bi_temporal",
    )
    assert pkg_1 == pkg_2


def test_edge_case_low_quality_strong_signal():
    """
    Critical scientific safeguard:
    Strong physical signal on low-quality imagery must NOT produce strong interpretation support.
    It MUST be gated to insufficient_support due to low data reliability.
    """
    obs_rel_low = evaluate_observation_reliability(
        imagery_result={"images": [{"quality": {"valid_percentage": 20.0}, "cloud_cover": 80.0}]}
    )
    sem_ev_strong = evaluate_semantic_evidence({
        "candidates": [{"target": "urban", "hypothesis": "urban_expansion", "final_evidence_score": 0.85, "state": "strong_candidate"}]
    })
    sp_eval = evaluate_spatial_assessment({"available": True, "spatial_coherence": 0.70, "region_count": 2})
    temp_eval = evaluate_temporal_consistency({"available": True, "temporal_mode": "bi_temporal", "observation_count": 2})
    data_suff = evaluate_data_sufficiency(obs_rel_low, temp_eval)

    interp = evaluate_overall_interpretation_support(
        obs_rel_low, sem_ev_strong, sp_eval, temp_eval, data_suff, []
    )
    assert interp.state == "insufficient_support"
    assert "Available imagery quality was insufficient" in interp.summary
    assert CalibrationReasonCodes.DATA_LOW_QUALITY in interp.reason_codes


def test_edge_case_one_valid_observation():
    """Only 1 valid observation epoch is insufficient for change detection."""
    obs_rel = evaluate_observation_reliability(
        imagery_result={"images": [{"quality": {"valid_percentage": 95.0}}]}
    )
    temp_eval = evaluate_temporal_consistency(temporal_observations=[TemporalObservation("o1", "s1", "2021-06-15T00:00:00Z", "2021-06-15", 2021, 166, 0.0, 1.0, 0.95, "high", 1.0)])
    data_suff = evaluate_data_sufficiency(obs_rel, temp_eval)
    assert data_suff.state == "limited" or data_suff.state == "insufficient"
    assert any("Fewer than 2 observation epochs" in m for m in data_suff.missing_components)


def test_edge_case_transition_no_spatial_overlap():
    """Transition with 0% overlap reports no_spatial_overlap."""
    cand_pkg = {
        "candidates": [
            {
                "target": "transition",
                "details": {"source_support": 0.50, "destination_support": 0.50},
            }
        ]
    }
    sp_analysis = {"available": True, "transition_overlap_fraction": 0.0}
    trans_supp = evaluate_transition_support(
        candidate_package=cand_pkg,
        spatial_analysis=sp_analysis,
    )
    assert trans_supp is not None
    assert trans_supp.spatial_support == "no_spatial_overlap"


def test_reason_codes_deduplication():
    """Verify all reason codes in master package are unique."""
    cal_pkg = build_calibration_package(
        candidate_package={"candidates": [{"target": "urban", "final_evidence_score": 0.70}]},
        target="urban",
        temporal_mode="bi_temporal",
    )
    rcs = cal_pkg["reason_codes"]
    assert len(rcs) == len(set(rcs))


def test_contradictory_interpretation_support_state():
    """Verify concurrent contradictory indicators result in contradictory_support."""
    obs_rel = evaluate_observation_reliability(
        imagery_result={"images": [{"quality": {"valid_percentage": 95.0}}, {"quality": {"valid_percentage": 95.0}}]}
    )
    sem_ev = evaluate_semantic_evidence({
        "candidates": [{"target": "urban", "final_evidence_score": 0.60, "state": "candidate"}]
    })
    sp_eval = evaluate_spatial_assessment({"available": True, "spatial_coherence": 0.50, "region_count": 2})
    temp_eval = evaluate_temporal_consistency({"available": True, "temporal_mode": "bi_temporal", "observation_count": 2})
    data_suff = evaluate_data_sufficiency(obs_rel, temp_eval)
    
    # Conflict detected between NDBI and NDVI
    conflicts, _ = detect_evidence_conflicts(
        multi_index_evidence={"signals": {"NDBI": {"raw_magnitude": 0.15}, "NDVI": {"raw_magnitude": 0.15}}}
    )
    interp = evaluate_overall_interpretation_support(
        obs_rel, sem_ev, sp_eval, temp_eval, data_suff, conflicts
    )
    assert interp.state == "contradictory_support"
    assert CalibrationReasonCodes.CONTRADICTORY_INTERPRETATION_SUPPORT in interp.reason_codes


def test_conflicting_ndvi_decrease_ndbi_decrease():
    """Verify concurrent decrease in NDVI and NDBI flags conflict."""
    multi_index = {
        "signals": {
            "NDVI": {"raw_magnitude": -0.15},
            "NDBI": {"raw_magnitude": -0.12},
        }
    }
    conflicts, codes = detect_evidence_conflicts(multi_index_evidence=multi_index)
    assert any(c["code"] == CalibrationReasonCodes.CONFLICTING_NDVI_NDBI_DECLINE for c in conflicts)


def test_all_indicators_neutral_edge_case():
    """All indicators neutral deadband produces no semantic support and neutral signals code."""
    multi_index = {
        "signals": {
            "NDVI": {"raw_magnitude": 0.01},
            "NDBI": {"raw_magnitude": -0.01},
            "NDWI": {"raw_magnitude": 0.02},
        }
    }
    sem_ev = evaluate_semantic_evidence({
        "candidates": [{"target": "urban", "hypothesis": "neutral", "final_evidence_score": 0.02, "state": "no_support"}]
    }, multi_index_evidence=multi_index)
    assert sem_ev.state == "none"
    assert CalibrationReasonCodes.NO_SEMANTIC_SUPPORT in sem_ev.reason_codes
    assert CalibrationReasonCodes.NEUTRAL_PHYSICAL_SIGNALS in sem_ev.reason_codes


def test_moderate_semantic_support_and_moderate_spatial():
    """Moderate evidence with moderate spatial support produces moderate_support."""
    obs_rel = evaluate_observation_reliability(
        imagery_result={"images": [{"quality": {"valid_percentage": 90.0}}, {"quality": {"valid_percentage": 90.0}}]}
    )
    sem_ev = evaluate_semantic_evidence({
        "candidates": [{"target": "urban", "hypothesis": "urban_expansion", "final_evidence_score": 0.45, "state": "candidate"}]
    })
    sp_eval = evaluate_spatial_assessment({"available": True, "spatial_coherence": 0.40, "region_count": 2})
    temp_eval = evaluate_temporal_consistency({"available": True, "temporal_mode": "bi_temporal", "observation_count": 2})
    data_suff = evaluate_data_sufficiency(obs_rel, temp_eval)

    interp = evaluate_overall_interpretation_support(
        obs_rel, sem_ev, sp_eval, temp_eval, data_suff, []
    )
    assert interp.state == "moderate_support"


def test_weak_semantic_support_and_limited_spatial():
    """Weak physical evidence produces weak_support."""
    obs_rel = evaluate_observation_reliability(
        imagery_result={"images": [{"quality": {"valid_percentage": 90.0}}, {"quality": {"valid_percentage": 90.0}}]}
    )
    sem_ev = evaluate_semantic_evidence({
        "candidates": [{"target": "urban", "hypothesis": "urban_expansion", "final_evidence_score": 0.25, "state": "weak_candidate"}]
    })
    sp_eval = evaluate_spatial_assessment({"available": True, "spatial_coherence": 0.15, "region_count": 1})
    temp_eval = evaluate_temporal_consistency({"available": True, "temporal_mode": "bi_temporal", "observation_count": 2})
    data_suff = evaluate_data_sufficiency(obs_rel, temp_eval)

    interp = evaluate_overall_interpretation_support(
        obs_rel, sem_ev, sp_eval, temp_eval, data_suff, []
    )
    assert interp.state == "weak_support"


