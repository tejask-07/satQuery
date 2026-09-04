"""
Phase 5A: Multi-Index Evidence Calculation Test Suite.

Verifies:
1. Urban evidence:
   - NDBI increase creates positive urban expansion evidence
   - NDVI decrease creates positive urban expansion evidence (vegetation clearing)
   - NDBI decrease creates negative urban expansion evidence (and supports reduction)
   - Conflicting signals remain inspectable and unsuppressed
2. Vegetation evidence:
   - NDVI decrease creates vegetation loss evidence
   - NDVI increase creates vegetation gain evidence
   - NIR/Red spectral response corroborates canopy loss
3. Water evidence:
   - NDWI decrease creates water loss evidence
   - NDWI increase creates water gain evidence
   - NIR/SWIR reflectance increase corroborates receding waterline
4. General scientific rigor:
   - Normalization is bounded in [0.0, 1.0]
   - Deadbands [-0.05, 0.05] are treated as neutral/noise
   - Invalid/masked pixels yield "unavailable" support
   - Missing signals are gracefully handled
   - Evidence is deterministic across identical inputs
   - No semantic final classification is asserted (evidence scores only)
"""

from unittest.mock import patch
import numpy as np
import pytest

from app.evidence.multi_index import (
    EvidenceThresholds,
    normalize_signal_magnitude,
    evaluate_directional_support,
    calculate_urban_evidence,
    calculate_vegetation_evidence,
    calculate_water_evidence,
    calculate_multi_index_evidence,
)
from app.api.routes_query import process_query
from app.schemas.query import QueryRequest


# ============================================================
# 1. NORMALIZATION & DEADBAND BEHAVIOR
# ============================================================

def test_normalization_bounds_and_deadband():
    """Verify deadband behavior and strict [0.0, 1.0] bounds."""
    # Within deadband (|delta| <= 0.05) -> direction=neutral, strength=0.0
    direction, mag, strength = normalize_signal_magnitude(0.02)
    assert direction == "neutral"
    assert mag == 0.02
    assert strength == 0.0

    direction, mag, strength = normalize_signal_magnitude(-0.04)
    assert direction == "neutral"
    assert strength == 0.0

    # Intermediate positive ramp
    direction, mag, strength = normalize_signal_magnitude(0.175)
    assert direction == "increase"
    # (0.175 - 0.05) / (0.30 - 0.05) = 0.125 / 0.25 = 0.50
    assert np.isclose(strength, 0.50)

    # Saturated high value (|delta| >= 0.30) -> 1.0
    direction, mag, strength = normalize_signal_magnitude(0.45)
    assert direction == "increase"
    assert strength == 1.0

    direction, mag, strength = normalize_signal_magnitude(-0.80)
    assert direction == "decrease"
    assert strength == 1.0

    # None or NaN handling
    direction, mag, strength = normalize_signal_magnitude(None)
    assert direction == "none"
    assert strength == 0.0

    direction, mag, strength = normalize_signal_magnitude(float("nan"))
    assert direction == "none"
    assert strength == 0.0


def test_evidence_determinism():
    """Evidence calculation must be bit-for-bit deterministic."""
    ev1 = calculate_urban_evidence(ndbi_delta=0.15, ndvi_delta=-0.12, quality_fraction=0.95)
    ev2 = calculate_urban_evidence(ndbi_delta=0.15, ndvi_delta=-0.12, quality_fraction=0.95)
    assert ev1 == ev2


# ============================================================
# 2. URBAN EVIDENCE SIGNALS
# ============================================================

def test_urban_ndbi_increase_positive_expansion_evidence():
    """NDBI increase must produce positive urban expansion support."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.20,
        ndvi_delta=0.0,  # neutral
        quality_fraction=1.0,
    )

    ndbi_sig = ev["signals"]["ndbi"]
    assert ndbi_sig["direction"] == "increase"
    assert ndbi_sig["support_state"] == "positive"
    assert ndbi_sig["support_score"] > 0.5
    assert ev["urban_expansion_support"] > 0.2
    assert ev["counter_hypothesis"]["urban_reduction_support"] < ev["urban_expansion_support"]


def test_urban_ndvi_decrease_positive_expansion_evidence():
    """NDVI decrease (vegetation clearing) must produce positive urban expansion support."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.0,  # neutral
        ndvi_delta=-0.25,
        quality_fraction=1.0,
    )

    ndvi_sig = ev["signals"]["ndvi"]
    assert ndvi_sig["direction"] == "decrease"
    assert ndvi_sig["support_state"] == "positive"
    assert ndvi_sig["support_score"] > 0.7
    assert ev["component_evidence"]["ndvi_support"] > 0.7


def test_urban_ndbi_decrease_negative_expansion_evidence():
    """NDBI decrease must produce negative urban expansion support and support reduction."""
    ev = calculate_urban_evidence(
        ndbi_delta=-0.18,
        ndvi_delta=0.15,  # vegetation increase
        quality_fraction=1.0,
    )

    ndbi_sig = ev["signals"]["ndbi"]
    assert ndbi_sig["direction"] == "decrease"
    assert ndbi_sig["support_state"] == "negative"
    assert ndbi_sig["support_score"] == 0.0

    # Counter hypothesis should receive positive support
    counter = ev["counter_hypothesis"]
    assert counter["urban_reduction_support"] >= 0.35
    assert counter["ndbi_reduction_support"] > 0.0


def test_urban_conflicting_signals_remain_inspectable():
    """When NDBI increases and NDVI also increases (re-greening while building), both signals must remain inspectable."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.20,  # supports expansion
        ndvi_delta=0.20,  # opposes expansion
        spectral_shifts={"swir": 0.08, "red": 0.04},
        quality_fraction=0.90,
    )

    assert ev["signals"]["ndbi"]["direction"] == "increase"
    assert ev["signals"]["ndbi"]["support_state"] == "positive"

    assert ev["signals"]["ndvi"]["direction"] == "increase"
    assert ev["signals"]["ndvi"]["support_state"] == "negative"

    # Both values are distinct and not artificially forced to agreement
    assert ev["component_evidence"]["ndbi_support"] > 0.5
    assert ev["component_evidence"]["ndvi_support"] == 0.0
    assert 0.0 <= ev["urban_expansion_support"] <= 1.0


# ============================================================
# 3. VEGETATION EVIDENCE SIGNALS
# ============================================================

def test_vegetation_loss_and_gain_evidence():
    """NDVI decrease produces vegetation loss evidence; NDVI increase produces gain evidence."""
    # Loss scenario
    loss_ev = calculate_vegetation_evidence(
        ndvi_delta=-0.22,
        ndbi_delta=0.10,  # exposed soil
        spectral_shifts={"nir": -0.10, "red": 0.06},
        quality_fraction=0.95,
    )
    assert loss_ev["signals"]["ndvi"]["direction"] == "decrease"
    assert loss_ev["signals"]["ndvi"]["support_state"] == "positive"
    assert loss_ev["vegetation_loss_support"] > 0.4
    assert loss_ev["component_evidence"]["spectral_support"] > 0.3

    # Gain scenario
    gain_ev = calculate_vegetation_evidence(
        ndvi_delta=+0.25,
        ndbi_delta=-0.08,
        spectral_shifts={"nir": 0.12, "red": -0.05},
        quality_fraction=0.95,
    )
    assert gain_ev["signals"]["ndvi"]["direction"] == "increase"
    assert gain_ev["signals"]["ndvi"]["support_state"] == "negative"
    assert gain_ev["counter_hypothesis"]["vegetation_gain_support"] > 0.4


# ============================================================
# 4. WATER EVIDENCE SIGNALS
# ============================================================

def test_water_loss_and_gain_evidence():
    """NDWI decrease produces water loss evidence; NDWI increase produces water gain evidence."""
    # Loss / drying scenario
    loss_ev = calculate_water_evidence(
        ndwi_delta=-0.25,
        spectral_shifts={"nir": 0.08, "swir": 0.10},  # drying exposes reflective soil
        quality_fraction=0.92,
    )
    assert loss_ev["signals"]["ndwi"]["direction"] == "decrease"
    assert loss_ev["signals"]["ndwi"]["support_state"] == "positive"
    assert loss_ev["signals"]["spectral"]["direction"] == "increase"
    assert loss_ev["water_loss_support"] > 0.5

    # Gain / inundation scenario
    gain_ev = calculate_water_evidence(
        ndwi_delta=+0.28,
        spectral_shifts={"nir": -0.09, "swir": -0.12},  # water absorption
        quality_fraction=0.92,
    )
    assert gain_ev["signals"]["ndwi"]["direction"] == "increase"
    assert gain_ev["signals"]["ndwi"]["support_state"] == "negative"
    assert gain_ev["counter_hypothesis"]["water_gain_support"] > 0.5


# ============================================================
# 5. MASKING & QUALITY
# ============================================================

def test_masked_and_invalid_data_yields_unavailable():
    """Poor quality (< 0.50) or unavailable data must yield unavailable support state and 0.0 score."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.25,
        ndvi_delta=-0.25,
        quality_fraction=0.30,  # 70% clouds / invalid
    )
    assert ev["state"] == "unavailable"
    assert ev["urban_expansion_support"] == 0.0
    assert ev["evidence_score"] == 0.0
    assert ev["reliability"]["valid"] is False
    assert ev["signals"]["ndbi"]["support_state"] == "unavailable"
    assert ev["signals"]["ndbi"]["support_score"] == 0.0


def test_missing_signals_handled_explicitly():
    """Missing signals must not throw exceptions; they receive zero/unavailable evidence."""
    ev = calculate_urban_evidence(
        ndbi_delta=None,
        ndvi_delta=None,
        spectral_shifts=None,
        quality_fraction=None,
    )
    assert ev["signals"]["ndbi"]["support_state"] == "unavailable"
    assert ev["signals"]["ndvi"]["support_state"] == "unavailable"
    assert ev["signals"]["spectral"]["support_state"] == "unavailable"
    assert ev["urban_expansion_support"] == 0.0
    assert ev["semantic_support"] == 0.0


# ============================================================
# 6. STEP 10 SCIENTIFIC DECOUPLING & ZERO-CHANGE REGRESSION TESTS
# ============================================================

def test_quality_does_not_create_semantic_evidence():
    """Quality = 0.99 with zero physical change must produce exactly 0.0 evidence."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.0,
        ndvi_delta=0.0,
        spectral_shifts={"swir": 0.0, "red": 0.0},
        quality_fraction=0.99,
    )
    assert ev["semantic_support"] == 0.0
    assert ev["urban_expansion_support"] == 0.0
    assert ev["evidence_score"] == 0.0
    assert ev["reliability"]["score"] == 0.99


def test_zero_physical_change_produces_zero_evidence():
    """All phenomena (urban, vegetation, water) must yield zero evidence when delta = 0."""
    ev_u = calculate_urban_evidence(ndbi_delta=0.0, ndvi_delta=0.0, quality_fraction=0.9628)
    assert ev_u["urban_expansion_support"] == 0.0
    assert ev_u["semantic_support"] == 0.0
    assert ev_u["counter_hypothesis"]["urban_reduction_support"] == 0.0

    ev_v = calculate_vegetation_evidence(ndvi_delta=0.0, quality_fraction=0.9628)
    assert ev_v["vegetation_loss_support"] == 0.0
    assert ev_v["semantic_support"] == 0.0
    assert ev_v["counter_hypothesis"]["vegetation_gain_support"] == 0.0

    ev_w = calculate_water_evidence(ndwi_delta=0.0, quality_fraction=0.9628)
    assert ev_w["water_loss_support"] == 0.0
    assert ev_w["semantic_support"] == 0.0
    assert ev_w["counter_hypothesis"]["water_gain_support"] == 0.0


def test_quality_exposed_separately():
    """Quality must be exposed under a separate reliability structure, not in semantic weights."""
    ev = calculate_urban_evidence(ndbi_delta=0.20, ndvi_delta=-0.15, quality_fraction=0.92)
    assert "reliability" in ev
    rel = ev["reliability"]
    assert "score" in rel
    assert "state" in rel
    assert "valid" in rel
    assert "quality_fraction" in rel
    assert "quality_support" not in ev["component_evidence"]


def test_final_evidence_equals_semantic_support_times_reliability():
    """Final evidence must equal round(semantic_support * reliability, 4)."""
    ev = calculate_urban_evidence(ndbi_delta=0.20, ndvi_delta=-0.15, quality_fraction=0.90)
    expected = round(ev["semantic_support"] * ev["reliability"]["score"], 4)
    assert np.isclose(ev["urban_expansion_support"], expected, atol=1e-4)
    assert ev["evidence_score"] == ev["urban_expansion_support"]


def test_low_quality_gates_evidence():
    """Quality below QUALITY_MIN_ACCEPTABLE must gate evidence to 0.0 and unavailable state."""
    ev = calculate_urban_evidence(ndbi_delta=0.35, ndvi_delta=-0.35, quality_fraction=0.30)
    assert ev["state"] == "unavailable"
    assert ev["urban_expansion_support"] == 0.0
    assert ev["evidence_score"] == 0.0
    assert ev["reliability"]["valid"] is False


def test_counter_hypotheses_not_quality_driven():
    """Counter hypotheses must not receive positive scores purely from clear quality."""
    ev_u = calculate_urban_evidence(ndbi_delta=0.20, ndvi_delta=-0.15, quality_fraction=1.0)
    # Counter for urban is reduction (needs NDBI decrease / NDVI increase)
    assert ev_u["counter_hypothesis"]["urban_reduction_support"] == 0.0

    ev_v = calculate_vegetation_evidence(ndvi_delta=-0.25, quality_fraction=1.0)
    # Counter for vegetation loss is vegetation gain (needs NDVI increase)
    assert ev_v["counter_hypothesis"]["vegetation_gain_support"] == 0.0

    ev_w = calculate_water_evidence(ndwi_delta=-0.25, quality_fraction=1.0)
    # Counter for water loss is water gain (needs NDWI increase)
    assert ev_w["counter_hypothesis"]["water_gain_support"] == 0.0


# ============================================================
# 7. END-TO-END PIPELINE INTEGRATION
# ============================================================

@pytest.fixture
def mock_vlm_offline():
    with patch("app.api.routes_query.VLM.generate", return_value="VLM offline test"):
        yield


def test_e2e_query_produces_multi_index_evidence(mock_vlm_offline):
    """Verify that a full query exposes multi_index_evidence in the API response."""
    req = QueryRequest(
        query="Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
    )
    result = process_query(req)

    assert result.status == "success"
    assert result.multi_index_evidence is not None

    ev = result.multi_index_evidence
    assert ev["target"] == "urban"
    assert "urban_expansion_support" in ev
    assert "evidence_score" in ev
    assert "semantic_support" in ev
    assert "reliability" in ev
    assert "score" in ev["reliability"]
    assert "component_evidence" in ev
    assert "ndbi_support" in ev["component_evidence"]
    assert "ndvi_support" in ev["component_evidence"]
    assert "spectral_support" in ev["component_evidence"]

    # Signals structure
    signals = ev["signals"]
    assert "ndbi" in signals
    assert "ndvi" in signals
    assert "spectral" in signals
    assert "quality" in signals

    for sig_name in ["ndbi", "ndvi", "spectral", "quality"]:
        s = signals[sig_name]
        assert "direction" in s
        assert "normalized_strength" in s
        assert "support_state" in s
        assert "support_score" in s
        assert 0.0 <= s["support_score"] <= 1.0

    # Also check evidence is attached to statistics and evidence_package
    assert result.statistics.get("evidence") is not None
    assert result.evidence_package.get("multi_index_evidence") is not None

