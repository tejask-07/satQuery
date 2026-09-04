"""
Phase 5B: Evidence Fusion & Semantic Change Candidate Classification Test Suite.

Verifies:
1. Urban Fusion:
   - NDBI increase + NDVI decrease -> urban expansion candidate
   - NDBI decrease + NDVI increase -> urban reduction candidate
   - NDBI increase + NDVI increase -> uncertain / conflicting
   - NDBI decrease + NDVI decrease -> uncertain / conflicting
   - Close expansion / reduction scores within ambiguity margin -> uncertain
   - Low reliability (< 0.50) -> unavailable

2. Vegetation Fusion:
   - NDVI decrease + supporting canopy loss -> vegetation loss candidate
   - NDVI increase + supporting canopy growth -> vegetation gain candidate
   - Contradictory spectral response -> uncertain

3. Water Fusion:
   - NDWI decrease + supporting soil drying -> water loss candidate
   - NDWI increase + supporting water absorption -> water gain candidate
   - Contradictory spectral response -> uncertain

4. General Change (Multi-Target):
   - Multiple domain candidates coexist independently
   - General change query evaluates urban, vegetation, and water
   - Clear sky / high reliability does not independently create a candidate
   - Invalid / masked pixels remain unavailable
   - Deterministic bit-for-bit output across identical inputs

5. Transition Queries:
   - Vegetation loss + urban expansion -> vegetation-to-urban transition candidate
   - Weak or conflicting endpoints -> uncertain / no_support
"""

from unittest.mock import patch
import numpy as np
import pytest

from app.evidence.multi_index import (
    calculate_urban_evidence,
    calculate_vegetation_evidence,
    calculate_water_evidence,
)
from app.evidence.fusion import (
    FusionThresholds,
    SemanticCandidate,
    classify_candidate_state,
    fuse_urban_evidence,
    fuse_vegetation_evidence,
    fuse_water_evidence,
    fuse_transition_evidence,
    fuse_evidence_and_classify_candidates,
)
from app.api.routes_query import process_query
from app.schemas.query import QueryRequest


# ============================================================
# 1. URBAN FUSION TESTS
# ============================================================

def test_urban_expansion_candidate():
    """NDBI increase + NDVI decrease produces urban_expansion candidate."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.22,
        ndvi_delta=-0.18,
        spectral_shifts={"swir": 0.08, "red": 0.04},
        quality_fraction=0.95,
    )
    cand = fuse_urban_evidence(ev)

    assert cand.target == "urban"
    assert cand.hypothesis == "urban_expansion"
    assert cand.state in ["candidate", "strong_candidate"]
    assert cand.final_evidence_score >= FusionThresholds.CANDIDATE_THRESHOLD
    assert "NDBI_INCREASE" in cand.reason_codes
    assert "NDVI_DECREASE" in cand.reason_codes
    assert "urban_expansion_support" in cand.details


def test_urban_reduction_candidate():
    """NDBI decrease + NDVI increase produces urban_reduction candidate."""
    ev = calculate_urban_evidence(
        ndbi_delta=-0.22,
        ndvi_delta=0.20,
        spectral_shifts={"swir": -0.06, "red": -0.03},
        quality_fraction=0.95,
    )
    cand = fuse_urban_evidence(ev)

    assert cand.target == "urban"
    assert cand.hypothesis == "urban_reduction"
    assert cand.state in ["candidate", "strong_candidate"]
    assert "NDBI_DECREASE" in cand.reason_codes
    assert "NDVI_INCREASE" in cand.reason_codes


def test_urban_conflicting_both_increase():
    """NDBI increase + NDVI increase (building while re-greening) produces uncertain state."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.20,
        ndvi_delta=0.20,
        quality_fraction=0.95,
    )
    cand = fuse_urban_evidence(ev)

    assert cand.state == "uncertain"
    assert "CONFLICTING_NDBI_NDVI_INCREASE" in cand.reason_codes
    assert cand.details["is_conflicted"] is True


def test_urban_conflicting_both_decrease():
    """NDBI decrease + NDVI decrease produces uncertain state."""
    ev = calculate_urban_evidence(
        ndbi_delta=-0.20,
        ndvi_delta=-0.20,
        quality_fraction=0.95,
    )
    cand = fuse_urban_evidence(ev)

    assert cand.state == "uncertain"
    assert "CONFLICTING_NDBI_NDVI_DECREASE" in cand.reason_codes
    assert cand.details["is_conflicted"] is True


def test_urban_ambiguity_margin():
    """When expansion and reduction scores are within AMBIGUITY_MARGIN, state is uncertain."""
    # Mock evidence where scores are very close
    ev = {
        "signals": {
            "ndbi": {"direction": "neutral", "normalized_strength": 0.0},
            "ndvi": {"direction": "neutral", "normalized_strength": 0.0},
            "spectral": {"direction": "neutral", "normalized_strength": 0.0},
        },
        "reliability": {"score": 0.95, "valid": True},
        "urban_expansion_support": 0.45,
        "semantic_support": 0.45,
        "counter_hypothesis": {
            "urban_reduction_support": 0.43,  # diff is 0.02 <= 0.12 margin
            "semantic_support": 0.43,
        },
    }
    cand = fuse_urban_evidence(ev)

    assert cand.state == "uncertain"
    assert "AMBIGUOUS_EXPANSION_REDUCTION" in cand.reason_codes
    assert cand.details["is_ambiguous"] is True


def test_urban_low_reliability_unavailable():
    """Low observation reliability (< 0.50) gates candidate state to unavailable."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.30,
        ndvi_delta=-0.30,
        quality_fraction=0.30,
    )
    cand = fuse_urban_evidence(ev)

    assert cand.state == "unavailable"
    assert cand.final_evidence_score == 0.0
    assert "LOW_RELIABILITY_GATED" in cand.reason_codes


# ============================================================
# 2. VEGETATION FUSION TESTS
# ============================================================

def test_vegetation_loss_candidate():
    """NDVI decrease + supporting spectral response produces vegetation_loss candidate."""
    ev = calculate_vegetation_evidence(
        ndvi_delta=-0.25,
        ndbi_delta=0.10,
        spectral_shifts={"nir": -0.12, "red": 0.08},
        quality_fraction=0.95,
    )
    cand = fuse_vegetation_evidence(ev)

    assert cand.target == "vegetation"
    assert cand.hypothesis == "vegetation_loss"
    assert cand.state in ["candidate", "strong_candidate"]
    assert "NDVI_DECREASE" in cand.reason_codes


def test_vegetation_gain_candidate():
    """NDVI increase + supporting spectral response produces vegetation_gain candidate."""
    ev = calculate_vegetation_evidence(
        ndvi_delta=+0.25,
        ndbi_delta=-0.08,
        spectral_shifts={"nir": 0.12, "red": -0.06},
        quality_fraction=0.95,
    )
    cand = fuse_vegetation_evidence(ev)

    assert cand.target == "vegetation"
    assert cand.hypothesis == "vegetation_gain"
    assert cand.state in ["candidate", "strong_candidate"]
    assert "NDVI_INCREASE" in cand.reason_codes


def test_vegetation_conflicting_signals():
    """NDVI decrease combined with spectral canopy growth produces uncertain state."""
    # NDVI drops but NIR strongly increased and Red dropped (contradiction)
    ev = calculate_vegetation_evidence(
        ndvi_delta=-0.25,
        spectral_shifts={"nir": 0.20, "red": -0.15},  # contradicts canopy loss
        quality_fraction=0.95,
    )
    cand = fuse_vegetation_evidence(ev)

    assert cand.state == "uncertain"
    assert "CONFLICTING_NDVI_AND_CANOPY_GROWTH" in cand.reason_codes


# ============================================================
# 3. WATER FUSION TESTS
# ============================================================

def test_water_loss_candidate():
    """NDWI decrease + soil drying spectral shift produces water_loss candidate."""
    ev = calculate_water_evidence(
        ndwi_delta=-0.25,
        spectral_shifts={"nir": 0.10, "swir": 0.12},
        quality_fraction=0.92,
    )
    cand = fuse_water_evidence(ev)

    assert cand.target == "water"
    assert cand.hypothesis == "water_loss"
    assert cand.state in ["candidate", "strong_candidate"]
    assert "NDWI_DECREASE" in cand.reason_codes


def test_water_gain_candidate():
    """NDWI increase + water absorption spectral shift produces water_gain candidate."""
    ev = calculate_water_evidence(
        ndwi_delta=+0.28,
        spectral_shifts={"nir": -0.10, "swir": -0.12},
        quality_fraction=0.92,
    )
    cand = fuse_water_evidence(ev)

    assert cand.target == "water"
    assert cand.hypothesis == "water_gain"
    assert cand.state in ["candidate", "strong_candidate"]
    assert "NDWI_INCREASE" in cand.reason_codes


def test_water_conflicting_signals():
    """NDWI decrease combined with water absorption deepening produces uncertain state."""
    ev = calculate_water_evidence(
        ndwi_delta=-0.22,
        spectral_shifts={"nir": -0.15, "swir": -0.18},  # absorbs like water, contradicts loss
        quality_fraction=0.95,
    )
    cand = fuse_water_evidence(ev)

    assert cand.state == "uncertain"
    assert "CONFLICTING_NDWI_AND_WATER_ABSORPTION" in cand.reason_codes


# ============================================================
# 4. GENERAL CHANGE & INVARIANT TESTS
# ============================================================

def test_general_change_evaluates_multiple_targets():
    """General change ('What changed?') evaluates urban, vegetation, and water independently."""
    # Synthetic execution results with vegetation loss and urban expansion
    exec_results = {
        "calculate_temporal_ndvi": {"mean_ndvi_change": -0.22},
        "calculate_temporal_ndbi": {"mean_ndbi_change": 0.20},
        "calculate_temporal_ndwi": {"mean_ndwi_change": 0.01},
    }
    m_ev = {
        "metadata": {
            "spectral_shifts": {"red": 0.05, "swir": 0.08, "nir": -0.10},
            "quality_fraction": 0.95,
        }
    }

    fusion_res = fuse_evidence_and_classify_candidates(
        target=None,
        task="detect_change",
        multi_index_evidence=m_ev,
        execution_results=exec_results,
    )

    candidates = fusion_res["candidates"]
    targets_evaluated = [c["target"] for c in candidates]
    assert "urban" in targets_evaluated
    assert "vegetation" in targets_evaluated
    assert "water" in targets_evaluated
    assert len(candidates) >= 3


def test_general_change_does_not_collapse_to_one_class():
    """When multiple changes occur simultaneously, both are preserved as candidates."""
    exec_results = {
        "calculate_temporal_ndvi": {"mean_ndvi_change": -0.25},
        "calculate_temporal_ndbi": {"mean_ndbi_change": 0.22},
        "calculate_temporal_ndwi": {"mean_ndwi_change": 0.0},
    }
    m_ev = {
        "metadata": {
            "spectral_shifts": {"red": 0.06, "swir": 0.08, "nir": -0.12},
            "quality_fraction": 0.95,
        }
    }

    res = fuse_evidence_and_classify_candidates(
        target="none",
        task="What changed between 2021 and 2025?",
        multi_index_evidence=m_ev,
        execution_results=exec_results,
    )

    cands_by_target = {c["target"]: c for c in res["candidates"]}
    assert cands_by_target["urban"]["state"] in ["candidate", "strong_candidate"]
    assert cands_by_target["vegetation"]["state"] in ["candidate", "strong_candidate"]


def test_quality_does_not_independently_create_candidate():
    """Clear sky (quality=0.99) with zero physical change produces no_support, not a candidate."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.0,
        ndvi_delta=0.0,
        spectral_shifts={"swir": 0.0, "red": 0.0},
        quality_fraction=0.99,
    )
    cand = fuse_urban_evidence(ev)

    assert cand.state == "no_support"
    assert cand.final_evidence_score == 0.0
    assert cand.semantic_support == 0.0


def test_invalid_pixels_remain_unavailable():
    """Masked / low-quality scenes remain unavailable across all candidates."""
    ev = calculate_vegetation_evidence(
        ndvi_delta=-0.35,
        quality_fraction=0.35,  # unacceptable quality
    )
    cand = fuse_vegetation_evidence(ev)

    assert cand.state == "unavailable"
    assert cand.final_evidence_score == 0.0


def test_fusion_determinism():
    """Fusion output must be bit-for-bit deterministic."""
    ev = calculate_urban_evidence(ndbi_delta=0.18, ndvi_delta=-0.14, quality_fraction=0.92)
    cand1 = fuse_urban_evidence(ev)
    cand2 = fuse_urban_evidence(ev)
    assert cand1.to_dict() == cand2.to_dict()


# ============================================================
# 5. TRANSITION QUERIES
# ============================================================

def test_transition_vegetation_to_urban():
    """Vegetation loss + urban expansion fuses into vegetation_to_urban candidate."""
    ev_veg = calculate_vegetation_evidence(
        ndvi_delta=-0.22,
        spectral_shifts={"nir": -0.10, "red": 0.06},
        quality_fraction=0.95,
    )
    ev_urb = calculate_urban_evidence(
        ndbi_delta=0.20,
        ndvi_delta=-0.22,
        spectral_shifts={"swir": 0.08, "red": 0.06},
        quality_fraction=0.95,
    )

    cand_veg = fuse_vegetation_evidence(ev_veg)
    cand_urb = fuse_urban_evidence(ev_urb)
    trans = fuse_transition_evidence(cand_veg, cand_urb)

    assert trans.target == "transition"
    assert trans.hypothesis == "vegetation_to_urban_transition"
    assert trans.state in ["candidate", "strong_candidate"]
    assert "SOURCE_VEGETATION_LOSS_DETECTED" in trans.reason_codes
    assert "DESTINATION_URBAN_EXPANSION_DETECTED" in trans.reason_codes


def test_transition_weak_or_missing_evidence():
    """If one endpoint has no evidence, the transition candidate receives no_support."""
    ev_veg = calculate_vegetation_evidence(ndvi_delta=0.0, quality_fraction=0.95)
    ev_urb = calculate_urban_evidence(ndbi_delta=0.0, ndvi_delta=0.0, quality_fraction=0.95)

    cand_veg = fuse_vegetation_evidence(ev_veg)
    cand_urb = fuse_urban_evidence(ev_urb)
    trans = fuse_transition_evidence(cand_veg, cand_urb)

    assert trans.state == "no_support"
    assert trans.final_evidence_score == 0.0


# ============================================================
# 6. END-TO-END PIPELINE QUERY INTEGRATION
# ============================================================

@pytest.fixture
def mock_vlm_offline():
    with patch("app.api.routes_query.VLM.generate", return_value="VLM offline test"):
        yield


def test_e2e_query_produces_phase5b_candidates(mock_vlm_offline):
    """Verify that a full end-to-end query produces Phase 5B candidates in the API response."""
    req = QueryRequest(
        query="Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
    )
    result = process_query(req)

    assert result.status == "success"
    assert result.candidates is not None
    assert len(result.candidates) >= 1

    primary_cand = result.candidates[0]
    assert "target" in primary_cand
    assert "hypothesis" in primary_cand
    assert "state" in primary_cand
    assert "supporting_evidence" in primary_cand
    assert "opposing_evidence" in primary_cand
    assert "final_evidence_score" in primary_cand
    assert "reason_codes" in primary_cand

    assert primary_cand["state"] in [
        "strong_candidate",
        "candidate",
        "weak_candidate",
        "uncertain",
        "no_support",
        "unavailable",
    ]

    # Check candidate package is attached to statistics
    assert "candidates" in result.statistics
    assert result.candidate_package is not None
