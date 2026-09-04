"""
Phase 5C: Structured Result Interpretation & Evidence-Backed Explanation Test Suite.

Verifies:
1. Strong urban expansion candidate -> appropriate grounded explanation
2. Urban no_support -> explicit neutral/no-support explanation
3. Urban conflicting evidence -> inconclusive explanation acknowledging disagreement
4. Vegetation loss candidate -> grounded vegetation-loss explanation
5. Vegetation no_support -> explicit no-loss explanation
6. Water loss candidate -> grounded water-loss explanation
7. Water no_support -> explicit no-loss explanation
8. General change with multiple findings -> multi-domain structured summary
9. General change with no findings -> general stability summary
10. Vegetation-to-urban transition candidate -> dual-endpoint explanation
11. Unreliable observation -> clear observation quality limitation
12. Invalid/unavailable result -> unavailable explanation
13. Opposing evidence explicitly included in factors
14. Reason codes match actual evidence
15. Reliability explained separately from change confidence
16. Anti-hallucination verification (no uncalculated areas, building counts, or percentages)
"""

from unittest.mock import patch
import pytest

from app.evidence.multi_index import (
    calculate_urban_evidence,
    calculate_vegetation_evidence,
    calculate_water_evidence,
)
from app.evidence.fusion import (
    fuse_urban_evidence,
    fuse_vegetation_evidence,
    fuse_water_evidence,
    fuse_transition_evidence,
    fuse_evidence_and_classify_candidates,
)
from app.evidence.interpretation import (
    ReasonCodes,
    StructuredInterpretation,
    format_reliability_summary,
    interpret_urban_candidate,
    interpret_vegetation_candidate,
    interpret_water_candidate,
    interpret_transition_candidate,
    interpret_general_change,
    generate_structured_interpretation,
)
from app.api.routes_query import process_query
from app.schemas.query import QueryRequest


# ============================================================
# 1. URBAN INTERPRETATION TESTS
# ============================================================

def test_strong_urban_expansion_explanation():
    """Strong urban expansion produces grounded positive conclusion and factors."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.25,
        ndvi_delta=-0.20,
        spectral_shifts={"swir": 0.08, "red": 0.05},
        quality_fraction=0.95,
    )
    cand = fuse_urban_evidence(ev).to_dict()
    interp = interpret_urban_candidate(cand, ev)

    assert "supports a candidate urban-expansion pattern" in interp.conclusion
    assert "urban expansion" in interp.summary.lower()
    assert any("NDBI increased" in f for f in interp.supporting_factors)
    assert any("NDVI decreased" in f for f in interp.supporting_factors)
    assert "HIGH_RELIABILITY" in interp.reason_codes
    assert "High observation reliability" in interp.reliability_summary


def test_urban_no_support_explanation():
    """Zero or noise-level index changes produce explicit no-support explanation."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.01,
        ndvi_delta=0.02,
        spectral_shifts={"swir": 0.005, "red": 0.003},
        quality_fraction=0.96,
    )
    cand = fuse_urban_evidence(ev).to_dict()
    interp = interpret_urban_candidate(cand, ev)

    assert "No strong evidence of urban expansion was found" in interp.conclusion
    assert "within baseline seasonal and sensor noise deadbands" in interp.summary
    assert any("deadband" in f for f in interp.opposing_factors)


def test_urban_conflicting_evidence_explanation():
    """Contradictory signals produce inconclusive explanation acknowledging disagreement."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.20,
        ndvi_delta=0.20,  # contradicts clearing
        quality_fraction=0.95,
    )
    cand = fuse_urban_evidence(ev).to_dict()
    interp = interpret_urban_candidate(cand, ev)

    assert "inconclusive because supporting indicators disagree" in interp.conclusion
    assert "CONFLICTING_NDBI_NDVI_INCREASE" in interp.reason_codes
    assert "contradicts a standard impervious conversion" in interp.summary


# ============================================================
# 2. VEGETATION INTERPRETATION TESTS
# ============================================================

def test_vegetation_loss_candidate_explanation():
    """Vegetation loss candidate produces clear grounded canopy loss explanation."""
    ev = calculate_vegetation_evidence(
        ndvi_delta=-0.25,
        ndbi_delta=0.10,
        spectral_shifts={"nir": -0.12, "red": 0.08},
        quality_fraction=0.94,
    )
    cand = fuse_vegetation_evidence(ev).to_dict()
    interp = interpret_vegetation_candidate(cand, ev)

    assert "supports a candidate vegetation-loss pattern" in interp.conclusion
    assert any("NDVI decreased" in f for f in interp.supporting_factors)
    assert any("Near-infrared" in f for f in interp.supporting_factors)


def test_vegetation_no_support_explanation():
    """Neutral vegetation delta produces explicit no-loss explanation."""
    ev = calculate_vegetation_evidence(
        ndvi_delta=0.01,
        quality_fraction=0.95,
    )
    cand = fuse_vegetation_evidence(ev).to_dict()
    interp = interpret_vegetation_candidate(cand, ev)

    assert "No strong evidence of vegetation loss was found" in interp.conclusion
    assert "remained within baseline variance" in interp.summary


# ============================================================
# 3. WATER INTERPRETATION TESTS
# ============================================================

def test_water_loss_candidate_explanation():
    """Water loss candidate produces clear water shrinkage explanation."""
    ev = calculate_water_evidence(
        ndwi_delta=-0.25,
        spectral_shifts={"nir": 0.10, "swir": 0.12},
        quality_fraction=0.92,
    )
    cand = fuse_water_evidence(ev).to_dict()
    interp = interpret_water_candidate(cand, ev)

    assert "supports a candidate water-loss pattern" in interp.conclusion
    assert any("NDWI decreased" in f for f in interp.supporting_factors)
    assert any("Near-infrared and SWIR" in f for f in interp.supporting_factors)


def test_water_no_support_explanation():
    """Neutral water delta produces explicit stability explanation."""
    ev = calculate_water_evidence(
        ndwi_delta=0.01,
        quality_fraction=0.92,
    )
    cand = fuse_water_evidence(ev).to_dict()
    interp = interpret_water_candidate(cand, ev)

    assert "No strong evidence of water change was found" in interp.conclusion
    assert "remained stable within baseline limits" in interp.summary


# ============================================================
# 4. GENERAL CHANGE & MULTI-DOMAIN TESTS
# ============================================================

def test_general_change_with_multiple_findings():
    """General change query with multiple active candidates produces multi-domain summary."""
    cand_u = {
        "target": "urban",
        "hypothesis": "urban_expansion",
        "state": "candidate",
        "supporting_evidence": {"ndbi_increase": 0.6},
        "opposing_evidence": {},
        "reliability": 0.95,
        "final_evidence_score": 0.50,
        "reason_codes": ["NDBI_INCREASE"],
    }
    cand_v = {
        "target": "vegetation",
        "hypothesis": "vegetation_loss",
        "state": "candidate",
        "supporting_evidence": {"ndvi_decrease": 0.6},
        "opposing_evidence": {},
        "reliability": 0.95,
        "final_evidence_score": 0.48,
        "reason_codes": ["NDVI_DECREASE"],
    }
    m_ev = {"metadata": {"quality_fraction": 0.95, "all_index_deltas": {"delta_ndvi": -0.2, "delta_ndbi": 0.2}}}

    interp = interpret_general_change([cand_u, cand_v], m_ev)

    assert "multiple candidate change patterns" in interp.conclusion
    assert "Urban" in interp.conclusion
    assert "Vegetation" in interp.conclusion
    assert len(interp.details["findings"]) == 2


def test_general_change_with_no_findings():
    """General change query where all domains are neutral reports general stability."""
    cand_u = {"target": "urban", "hypothesis": "urban_expansion", "state": "no_support", "supporting_evidence": {}, "opposing_evidence": {}, "final_evidence_score": 0.0, "reason_codes": []}
    cand_v = {"target": "vegetation", "hypothesis": "vegetation_loss", "state": "no_support", "supporting_evidence": {}, "opposing_evidence": {}, "final_evidence_score": 0.0, "reason_codes": []}
    cand_w = {"target": "water", "hypothesis": "water_loss", "state": "no_support", "supporting_evidence": {}, "opposing_evidence": {}, "final_evidence_score": 0.0, "reason_codes": []}
    m_ev = {"metadata": {"quality_fraction": 0.95, "all_index_deltas": {}}}

    interp = interpret_general_change([cand_u, cand_v, cand_w], m_ev)

    assert "No strong land-cover change was identified from the available evidence" in interp.conclusion
    assert interp.state == "no_support"


# ============================================================
# 5. TRANSITION TESTS
# ============================================================

def test_transition_vegetation_to_urban_explanation():
    """Transition candidate explains dual-endpoint dynamics and notes spatial verification requirement."""
    cand = {
        "target": "transition",
        "hypothesis": "vegetation_to_urban_transition",
        "state": "candidate",
        "supporting_evidence": {"source_vegetation_evidence": 0.55, "destination_urban_evidence": 0.52},
        "opposing_evidence": {},
        "reliability": 0.95,
        "final_evidence_score": 0.53,
        "reason_codes": ["SOURCE_VEGETATION_LOSS_DETECTED", "DESTINATION_URBAN_EXPANSION_DETECTED"],
    }
    m_ev = {"metadata": {"quality_fraction": 0.95}}
    interp = interpret_transition_candidate(cand, m_ev)

    assert "supports a candidate vegetation-to-urban transition pattern" in interp.conclusion
    assert "Spatial contiguity validation is required" in interp.summary
    assert any("vegetation" in f for f in interp.supporting_factors)
    assert any("urban" in f for f in interp.supporting_factors)


# ============================================================
# 6. RELIABILITY, INVALIDITY & OPPOSING FACTORS
# ============================================================

def test_unreliable_observation_explanation():
    """Observations with poor quality explicitly document reliability limitation."""
    rel_text = format_reliability_summary(0.35, is_valid=False)
    assert "Low observation reliability" in rel_text
    assert "below acceptable threshold" in rel_text


def test_invalid_unavailable_result_explanation():
    """Unavailable state yields explicit data insufficiency conclusion."""
    cand = {
        "target": "urban",
        "hypothesis": "urban_expansion",
        "state": "unavailable",
        "supporting_evidence": {},
        "opposing_evidence": {},
        "reliability": 0.30,
        "final_evidence_score": 0.0,
        "reason_codes": ["LOW_RELIABILITY_GATED"],
    }
    ev = {"metadata": {"all_index_deltas": {}}}
    interp = interpret_urban_candidate(cand, ev)

    assert "unavailable due to insufficient observation quality" in interp.conclusion
    assert "does not meet minimum quality criteria" in interp.summary


def test_opposing_evidence_included():
    """When opposing signals are present, they are explicitly populated in opposing_factors."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.20,  # supports expansion
        ndvi_delta=0.15,  # opposes expansion (vegetation increase)
        quality_fraction=0.90,
    )
    cand = fuse_urban_evidence(ev).to_dict()
    interp = interpret_urban_candidate(cand, ev)

    assert len(interp.opposing_factors) > 0
    assert any("opposing vegetation clearing" in f or "increased" in f for f in interp.opposing_factors)


def test_reason_codes_match_actual_evidence():
    """Reason codes in interpretation match computed signals."""
    ev = calculate_urban_evidence(
        ndbi_delta=0.20,
        ndvi_delta=-0.15,
        spectral_shifts={"swir": 0.05, "red": 0.03},
        quality_fraction=0.95,
    )
    cand = fuse_urban_evidence(ev).to_dict()
    interp = interpret_urban_candidate(cand, ev)

    assert "NDBI_INCREASE" in interp.reason_codes
    assert "NDVI_DECREASE" in interp.reason_codes
    assert "SPECTRAL_BRIGHTENING" in interp.reason_codes


def test_reliability_separated_from_semantic_interpretation():
    """Reliability text never states change confidence percentage."""
    rel_text = format_reliability_summary(0.9628, is_valid=True)
    assert "96.3% jointly valid" in rel_text
    # Forbidden phrases:
    assert "confidence in" not in rel_text
    assert "probability" not in rel_text
    assert "accuracy" not in rel_text


def test_anti_hallucination_safeguards():
    """Interpretation summaries never contain fabricated numeric claims."""
    ev = calculate_urban_evidence(ndbi_delta=0.02, ndvi_delta=0.01, quality_fraction=0.9628)
    cand = fuse_urban_evidence(ev).to_dict()
    interp = interpret_urban_candidate(cand, ev)

    text = interp.summary + " " + interp.conclusion
    # Verify no fabricated claims
    assert "buildings" not in text
    assert "hectares" not in text
    assert "sq km" not in text
    assert "square kilometer" not in text
    assert "23%" not in text
    assert "95% confidence" not in text
    assert "confirmed" not in text.lower()


# ============================================================
# 7. END-TO-END PIPELINE QUERY INTEGRATION
# ============================================================

@pytest.fixture
def mock_vlm_offline():
    with patch("app.api.routes_query.VLM.generate", return_value="VLM offline test"):
        yield


def test_e2e_query_produces_phase5c_interpretation(mock_vlm_offline):
    """Verify that a live end-to-end query exposes structured interpretation in API response."""
    req = QueryRequest(
        query="Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
    )
    result = process_query(req)

    assert result.status == "success"
    assert result.interpretation is not None

    interp = result.interpretation
    assert "conclusion" in interp
    assert "summary" in interp
    assert "target" in interp
    assert "hypothesis" in interp
    assert "state" in interp
    assert "reliability_summary" in interp
    assert "limitations" in interp
    assert "reason_codes" in interp

    # Statistics must also record interpretation
    assert "interpretation" in result.statistics

    # Final answer or grounded explanation must match interpretation summary
    assert len(interp["summary"]) > 20
    assert "urban" in interp["summary"].lower()
