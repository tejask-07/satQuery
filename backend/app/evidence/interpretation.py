"""
Phase 5C: Structured Result Interpretation & Evidence-Backed Explanation Engine.

Translates Phase 5B Semantic Candidates and Phase 5A physical evidence packages
into transparent, human-readable, evidence-grounded interpretations.

CORE PRINCIPLES:
1. Grounded in Measured Evidence:
   All claims stem strictly from computed spectral index deltas, reflectance shifts,
   reliability masks, and deterministic candidate classifications.
2. Anti-Hallucination:
   Never invents uncalculated areas (e.g. "increased by 23%"), building counts,
   percentages of change, model confidence probabilities, or causes of change.
3. Separation of Quality & Finding:
   Observation reliability (pixel validity) is explained strictly as observation fidelity,
   never conflated with confidence or probability in the phenomenon.
4. Scientific Nuance:
   Clearly distinguishes supporting factors from opposing/neutral factors,
   and explicitly articulates uncertain or no-support states.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


def _format_delta(val: Optional[Union[float, int]]) -> str:
    """Format an index delta safely, returning 'N/A' if None or non-finite."""
    if val is None:
        return "N/A"
    try:
        fval = float(val)
        if not np.isfinite(fval):
            return "N/A"
        return f"{fval:+.4f}"
    except (ValueError, TypeError):
        return "N/A"



# ============================================================
# STANDARDIZED REASON CODES VOCABULARY
# ============================================================

class ReasonCodes:
    """
    Centralized vocabulary of standardized reason codes.
    """
    # Urban
    NDBI_INCREASE = "NDBI_INCREASE"
    NDBI_DECREASE = "NDBI_DECREASE"
    NDVI_DECREASE = "NDVI_DECREASE"
    NDVI_INCREASE = "NDVI_INCREASE"
    SPECTRAL_BRIGHTENING = "SPECTRAL_BRIGHTENING"
    SPECTRAL_DARKENING = "SPECTRAL_DARKENING"
    CONFLICTING_NDBI_NDVI_INCREASE = "CONFLICTING_NDBI_NDVI_INCREASE"
    CONFLICTING_NDBI_NDVI_DECREASE = "CONFLICTING_NDBI_NDVI_DECREASE"
    AMBIGUOUS_EXPANSION_REDUCTION = "AMBIGUOUS_EXPANSION_REDUCTION"
    URBAN_EXPANSION_FAVORED = "URBAN_EXPANSION_FAVORED"
    URBAN_REDUCTION_FAVORED = "URBAN_REDUCTION_FAVORED"

    # Vegetation
    NDVI_LOSS = "NDVI_LOSS"
    NDVI_GAIN = "NDVI_GAIN"
    SOIL_EXPOSURE = "SOIL_EXPOSURE"
    CANOPY_LOSS_SPECTRAL = "CANOPY_LOSS_SPECTRAL"
    CANOPY_GROWTH_SPECTRAL = "CANOPY_GROWTH_SPECTRAL"
    CONFLICTING_NDVI_AND_CANOPY_GROWTH = "CONFLICTING_NDVI_AND_CANOPY_GROWTH"
    CONFLICTING_NDVI_AND_CANOPY_LOSS = "CONFLICTING_NDVI_AND_CANOPY_LOSS"
    AMBIGUOUS_LOSS_GAIN = "AMBIGUOUS_LOSS_GAIN"
    VEGETATION_LOSS_FAVORED = "VEGETATION_LOSS_FAVORED"
    VEGETATION_GAIN_FAVORED = "VEGETATION_GAIN_FAVORED"

    # Water
    NDWI_LOSS = "NDWI_LOSS"
    NDWI_GAIN = "NDWI_GAIN"
    SOIL_DRYING_SPECTRAL = "SOIL_DRYING_SPECTRAL"
    WATER_ABSORPTION_SPECTRAL = "WATER_ABSORPTION_SPECTRAL"
    CONFLICTING_NDWI_AND_WATER_ABSORPTION = "CONFLICTING_NDWI_AND_WATER_ABSORPTION"
    CONFLICTING_NDWI_AND_SOIL_DRYING = "CONFLICTING_NDWI_AND_SOIL_DRYING"
    AMBIGUOUS_WATER_LOSS_GAIN = "AMBIGUOUS_WATER_LOSS_GAIN"
    WATER_LOSS_FAVORED = "WATER_LOSS_FAVORED"
    WATER_GAIN_FAVORED = "WATER_GAIN_FAVORED"

    # Transition
    SOURCE_VEGETATION_LOSS_DETECTED = "SOURCE_VEGETATION_LOSS_DETECTED"
    DESTINATION_URBAN_EXPANSION_DETECTED = "DESTINATION_URBAN_EXPANSION_DETECTED"
    ENDPOINT_EVIDENCE_UNCERTAIN = "ENDPOINT_EVIDENCE_UNCERTAIN"
    TRANSITION_INCONCLUSIVE = "TRANSITION_INCONCLUSIVE"

    # Quality & General
    HIGH_RELIABILITY = "HIGH_RELIABILITY"
    MODERATE_RELIABILITY = "MODERATE_RELIABILITY"
    LOW_RELIABILITY_GATED = "LOW_RELIABILITY_GATED"
    NO_PHYSICAL_SUPPORT = "NO_PHYSICAL_SUPPORT"
    DEADBAND_NEUTRAL = "DEADBAND_NEUTRAL"


# ============================================================
# STRUCTURED INTERPRETATION DATA MODEL
# ============================================================

@dataclass
class StructuredInterpretation:
    """
    Standardized, inspectable interpretation of an analysis result.
    """
    conclusion: str
    summary: str
    target: str
    hypothesis: str
    state: str
    evidence_summary: str
    supporting_factors: List[str]
    opposing_factors: List[str]
    reliability_summary: str
    limitations: List[str]
    reason_codes: List[str]
    spatial_summary: Optional[str] = None
    region_count: int = 0
    candidate_area_hectares: float = 0.0
    largest_region_area_hectares: float = 0.0
    dominant_location_description: Optional[str] = None
    temporal_summary: Optional[str] = None
    temporal_mode: Optional[str] = None
    temporal_trend: Optional[str] = None
    persistence_fraction: Optional[float] = None
    change_nature: Optional[str] = None
    seasonal_comparability: Optional[str] = None
    calibration_summary: Optional[str] = None
    interpretation_support_state: Optional[str] = None
    data_sufficiency_state: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Common limitations attached to satellite change candidate interpretation
STANDARD_LIMITATIONS: List[str] = [
    "Candidate pattern is based on bi-temporal spectral index shifts and reflectance deltas.",
    "No spatial contiguous aggregation or cluster polygonization has been performed yet.",
    "Observation is not verified against local ground-truth reference data.",
]


def format_reliability_summary(reliability: float, is_valid: bool = True) -> str:
    """
    Formats observation reliability strictly as data validity, avoiding probability terms.
    """
    if not is_valid or reliability < 0.50:
        return f"Low observation reliability ({reliability * 100:.1f}% valid observation pixels); below acceptable threshold for confident interpretation."
    if reliability >= 0.85:
        return f"High observation reliability ({reliability * 100:.1f}% jointly valid, cloud/shadow-free observation pixels)."
    return f"Moderate observation reliability ({reliability * 100:.1f}% jointly valid observation pixels)."


# ============================================================
# DOMAIN INTERPRETERS
# ============================================================

def interpret_urban_candidate(
    candidate: Dict[str, Any],
    multi_index_evidence: Dict[str, Any],
) -> StructuredInterpretation:
    """
    Generates structured interpretation for an urban candidate finding.
    """
    state = candidate.get("state", "no_support")
    hyp = candidate.get("hypothesis", "urban_expansion")
    supp = candidate.get("supporting_evidence", {})
    opp = candidate.get("opposing_evidence", {})
    rel = float(candidate.get("reliability", 1.0))
    score = float(candidate.get("final_evidence_score", 0.0))
    codes = list(candidate.get("reason_codes", []))
    deltas = multi_index_evidence.get("metadata", {}).get("all_index_deltas", {})
    d_ndbi = deltas.get("delta_ndbi")
    if d_ndbi is None:
        d_ndbi = multi_index_evidence.get("signals", {}).get("ndbi", {}).get("details", {}).get("delta")
    d_ndvi = deltas.get("delta_ndvi")
    if d_ndvi is None:
        d_ndvi = multi_index_evidence.get("signals", {}).get("ndvi", {}).get("details", {}).get("delta")

    supporting_factors: List[str] = []
    opposing_factors: List[str] = []

    # Format supporting factors
    if "ndbi_increase" in supp:
        supporting_factors.append(f"NDBI increased by {_format_delta(d_ndbi)}, corroborating increased built-up or impervious surface.")
    if "ndvi_clearing" in supp:
        supporting_factors.append(f"NDVI decreased by {_format_delta(d_ndvi)}, corroborating vegetation canopy clearing.")
    if "spectral_brightening" in supp:
        supporting_factors.append("SWIR and Red surface reflectance brightened, consistent with impervious materials.")

    if hyp == "urban_reduction":
        if "ndbi_decrease" in supp:
            supporting_factors.append(f"NDBI decreased by {_format_delta(d_ndbi)}, suggesting reduction in built-up signature.")
        if "ndvi_regreening" in supp:
            supporting_factors.append(f"NDVI increased by {_format_delta(d_ndvi)}, corroborating revegetation or re-greening.")
        if "spectral_darkening" in supp:
            supporting_factors.append("SWIR and Red reflectance darkened, opposing built-up structures.")

    # Format opposing factors
    if "ndbi_decrease" in opp and hyp == "urban_expansion":
        opposing_factors.append(f"NDBI decreased by {_format_delta(d_ndbi)}, opposing built-up expansion.")
    if "ndvi_regreening" in opp and hyp == "urban_expansion":
        opposing_factors.append(f"NDVI increased by {_format_delta(d_ndvi)}, opposing vegetation clearing.")
    if "spectral_darkening" in opp and hyp == "urban_expansion":
        opposing_factors.append("SWIR/Red reflectance darkened, opposing impervious surface construction.")

    # Check deadband neutral condition
    if (d_ndbi is not None and abs(d_ndbi) <= 0.05) and (d_ndvi is not None and abs(d_ndvi) <= 0.05):
        opposing_factors.append("NDBI and NDVI shifts are within the +/-0.05 sensor noise deadband (no significant physical change).")

    # Generate conclusion and summary based on state
    if state == "unavailable":
        conclusion = "Urban-change assessment is unavailable due to insufficient observation quality."
        summary = "Observation data within the target area does not meet minimum quality criteria due to cloud cover, shadow, or missing pixels. No reliable change determination can be made."
    elif state == "uncertain":
        conclusion = "Urban-change evidence is inconclusive because supporting indicators disagree."
        summary = (
            "Evidence for urban surface change is ambiguous or conflicting. "
            + ("Both built-up (NDBI) and vegetation (NDVI) indices increased simultaneously, which contradicts a standard impervious conversion. " if "CONFLICTING_NDBI_NDVI_INCREASE" in codes else
               "Primary expansion and reduction evidence indicators are closely matched within the ambiguity margin. ")
            + "Further spatial or multi-temporal validation is required."
        )
    elif state in ["candidate", "strong_candidate"]:
        if hyp == "urban_expansion":
            conclusion = "Evidence supports a candidate urban-expansion pattern."
            summary = (
                f"Multi-index satellite analysis provides consistent physical evidence supporting potential urban expansion (evidence score: {score:.3f}). "
                f"Built-up reflectance signatures increased while vegetation signatures decreased across the observation interval, corroborated by surface reflectance shifts."
            )
        else:
            conclusion = "Evidence supports a candidate urban-reduction pattern."
            summary = (
                f"Multi-index satellite analysis provides physical evidence supporting potential urban reduction or surface deconstruction (evidence score: {score:.3f}). "
                f"Built-up reflectance signatures decreased accompanied by vegetation index increases."
            )
    elif state == "weak_candidate":
        conclusion = "Evidence shows weak, isolated support for potential urban change."
        summary = (
            f"Observed index shifts indicate weak potential urban change (evidence score: {score:.3f}), "
            "but signal magnitudes are near threshold levels and do not provide decisive candidate support."
        )
    else:  # no_support
        conclusion = "No strong evidence of urban expansion was found."
        summary = (
            "Multi-index evaluation revealed no significant physical evidence of urban expansion across the evaluated interval. "
            "Observed NDBI, NDVI, and spectral reflectance variations remained within baseline seasonal and sensor noise deadbands."
        )

    ev_summary = f"NDBI delta: {_format_delta(d_ndbi)}, NDVI delta: {_format_delta(d_ndvi)}."

    return StructuredInterpretation(
        conclusion=conclusion,
        summary=summary,
        target="urban",
        hypothesis=hyp,
        state=state,
        evidence_summary=ev_summary,
        supporting_factors=supporting_factors,
        opposing_factors=opposing_factors,
        reliability_summary=format_reliability_summary(rel, is_valid=state != "unavailable"),
        limitations=STANDARD_LIMITATIONS,
        reason_codes=codes,
        details={"candidate": candidate},
    )


def interpret_vegetation_candidate(
    candidate: Dict[str, Any],
    multi_index_evidence: Dict[str, Any],
) -> StructuredInterpretation:
    """
    Generates structured interpretation for a vegetation candidate finding.
    """
    state = candidate.get("state", "no_support")
    hyp = candidate.get("hypothesis", "vegetation_loss")
    supp = candidate.get("supporting_evidence", {})
    opp = candidate.get("opposing_evidence", {})
    rel = float(candidate.get("reliability", 1.0))
    score = float(candidate.get("final_evidence_score", 0.0))
    codes = list(candidate.get("reason_codes", []))
    deltas = multi_index_evidence.get("metadata", {}).get("all_index_deltas", {})
    d_ndvi = deltas.get("delta_ndvi")
    if d_ndvi is None:
        d_ndvi = multi_index_evidence.get("signals", {}).get("ndvi", {}).get("details", {}).get("delta")

    supporting_factors: List[str] = []
    opposing_factors: List[str] = []

    if "ndvi_decrease" in supp:
        supporting_factors.append(f"NDVI decreased by {_format_delta(d_ndvi)}, indicating loss of green photosynthetic canopy.")
    if "soil_exposure" in supp:
        supporting_factors.append("Soil / non-vegetated surface response increased, corroborating canopy removal.")
    if "canopy_loss_spectral" in supp:
        supporting_factors.append("Near-infrared canopy reflectance dropped accompanied by increased visible red reflectance.")

    if hyp == "vegetation_gain":
        if "ndvi_increase" in supp:
            supporting_factors.append(f"NDVI increased by {_format_delta(d_ndvi)}, indicating canopy expansion or vegetative recovery.")
        if "canopy_growth_spectral" in supp:
            supporting_factors.append("Near-infrared reflectance increased, corroborating vegetative biomass growth.")

    if "ndvi_increase" in opp and hyp == "vegetation_loss":
        opposing_factors.append(f"NDVI shifted positively ({_format_delta(d_ndvi)}), opposing vegetation loss.")

    if d_ndvi is not None and abs(d_ndvi) <= 0.05:
        opposing_factors.append("NDVI variation is within the +/-0.05 baseline noise deadband.")

    if state == "unavailable":
        conclusion = "Vegetation assessment is unavailable due to insufficient observation quality."
        summary = "Observation pixels are masked or invalid due to atmospheric interference. No reliable vegetation trend can be established."
    elif state == "uncertain":
        conclusion = "Vegetation-change evidence is inconclusive due to contradictory canopy indicators."
        summary = "Observed vegetative signals disagree across indices and spectral bands, preventing conclusive candidate classification."
    elif state in ["candidate", "strong_candidate"]:
        if hyp == "vegetation_loss":
            conclusion = "Evidence supports a candidate vegetation-loss pattern."
            summary = (
                f"Multi-index satellite analysis provides consistent evidence supporting vegetation loss or canopy thinning (evidence score: {score:.3f}). "
                f"NDVI decreased significantly, corroborated by canopy reflectance and soil response indicators."
            )
        else:
            conclusion = "Evidence supports a candidate vegetation-gain pattern."
            summary = (
                f"Multi-index satellite analysis provides physical evidence supporting vegetative vigor increase or canopy expansion (evidence score: {score:.3f}). "
                f"NDVI shifted positively, consistent with vegetative recovery or seasonal growth."
            )
    elif state == "weak_candidate":
        conclusion = "Evidence shows weak, isolated support for vegetation change."
        summary = f"Observed vegetative variations indicate minor fluctuation (evidence score: {score:.3f}), but remain near detection thresholds."
    else:  # no_support
        conclusion = "No strong evidence of vegetation loss was found."
        summary = (
            "Multi-index evaluation detected no significant physical evidence of vegetation reduction across the evaluated interval. "
            "Vegetation canopy indicators remained within baseline variance."
        )

    ev_summary = f"NDVI delta: {_format_delta(d_ndvi)}."

    return StructuredInterpretation(
        conclusion=conclusion,
        summary=summary,
        target="vegetation",
        hypothesis=hyp,
        state=state,
        evidence_summary=ev_summary,
        supporting_factors=supporting_factors,
        opposing_factors=opposing_factors,
        reliability_summary=format_reliability_summary(rel, is_valid=state != "unavailable"),
        limitations=STANDARD_LIMITATIONS,
        reason_codes=codes,
        details={"candidate": candidate},
    )


def interpret_water_candidate(
    candidate: Dict[str, Any],
    multi_index_evidence: Dict[str, Any],
) -> StructuredInterpretation:
    """
    Generates structured interpretation for a water candidate finding.
    """
    state = candidate.get("state", "no_support")
    hyp = candidate.get("hypothesis", "water_loss")
    supp = candidate.get("supporting_evidence", {})
    opp = candidate.get("opposing_evidence", {})
    rel = float(candidate.get("reliability", 1.0))
    score = float(candidate.get("final_evidence_score", 0.0))
    codes = list(candidate.get("reason_codes", []))
    deltas = multi_index_evidence.get("metadata", {}).get("all_index_deltas", {})
    d_ndwi = deltas.get("delta_ndwi")
    if d_ndwi is None:
        d_ndwi = multi_index_evidence.get("signals", {}).get("ndwi", {}).get("details", {}).get("delta")

    supporting_factors: List[str] = []
    opposing_factors: List[str] = []

    if "ndwi_decrease" in supp:
        supporting_factors.append(f"NDWI decreased by {_format_delta(d_ndwi)}, corroborating surface water shrinkage or shoreline recession.")
    if "soil_drying_spectral" in supp:
        supporting_factors.append("Near-infrared and SWIR reflectance brightened, consistent with exposed soil or drying bed.")

    if hyp == "water_gain":
        if "ndwi_increase" in supp:
            supporting_factors.append(f"NDWI increased by {_format_delta(d_ndwi)}, indicating surface water expansion or inundation.")
        if "water_absorption_spectral" in supp:
            supporting_factors.append("NIR/SWIR reflectance darkened due to enhanced water absorption.")

    if "ndwi_increase" in opp and hyp == "water_loss":
        opposing_factors.append(f"NDWI increased positively ({_format_delta(d_ndwi)}), opposing water loss.")

    if d_ndwi is not None and abs(d_ndwi) <= 0.05:
        opposing_factors.append("NDWI change is within the +/-0.05 baseline noise deadband.")

    if state == "unavailable":
        conclusion = "Water surface assessment is unavailable due to insufficient observation quality."
        summary = "Observation data is obscured or invalid; water surface dynamics cannot be reliably evaluated."
    elif state == "uncertain":
        conclusion = "Water-change evidence is inconclusive due to contradictory spectral absorption signals."
        summary = "Water index and absorption indicators conflict across the observation interval, preventing clear candidate classification."
    elif state in ["candidate", "strong_candidate"]:
        if hyp == "water_loss":
            conclusion = "Evidence supports a candidate water-loss pattern."
            summary = (
                f"Multi-index satellite analysis provides consistent evidence supporting surface water reduction (evidence score: {score:.3f}). "
                f"NDWI decreased significantly, corroborated by soil exposure and diminished water absorption signatures."
            )
        else:
            conclusion = "Evidence supports a candidate water-gain pattern."
            summary = (
                f"Multi-index satellite analysis provides physical evidence supporting surface water expansion or inundation (evidence score: {score:.3f}). "
                f"Water index increased accompanied by characteristic NIR/SWIR absorption deepening."
            )
    elif state == "weak_candidate":
        conclusion = "Evidence shows weak, isolated support for water surface change."
        summary = f"Water index variations indicate minor shoreline movement (evidence score: {score:.3f}), but remain near detection thresholds."
    else:  # no_support
        conclusion = "No strong evidence of water change was found."
        summary = (
            "Multi-index evaluation detected no significant physical evidence of surface water change across the evaluated interval. "
            "Water index signatures remained stable within baseline limits."
        )

    ev_summary = f"NDWI delta: {_format_delta(d_ndwi)}."

    return StructuredInterpretation(
        conclusion=conclusion,
        summary=summary,
        target="water",
        hypothesis=hyp,
        state=state,
        evidence_summary=ev_summary,
        supporting_factors=supporting_factors,
        opposing_factors=opposing_factors,
        reliability_summary=format_reliability_summary(rel, is_valid=state != "unavailable"),
        limitations=STANDARD_LIMITATIONS,
        reason_codes=codes,
        details={"candidate": candidate},
    )


def interpret_transition_candidate(
    candidate: Dict[str, Any],
    multi_index_evidence: Dict[str, Any],
) -> StructuredInterpretation:
    """
    Generates structured interpretation for a land-cover transition candidate (e.g. vegetation to urban).
    """
    state = candidate.get("state", "no_support")
    hyp = candidate.get("hypothesis", "vegetation_to_urban_transition")
    rel = float(candidate.get("reliability", 1.0))
    score = float(candidate.get("final_evidence_score", 0.0))
    codes = list(candidate.get("reason_codes", []))
    supp = candidate.get("supporting_evidence", {})

    supporting_factors: List[str] = []
    opposing_factors: List[str] = []

    if "SOURCE_VEGETATION_LOSS_DETECTED" in codes:
        supporting_factors.append("Source category (vegetation) demonstrated significant canopy reduction.")
    else:
        opposing_factors.append("Source category (vegetation) showed no significant canopy clearing.")

    if "DESTINATION_URBAN_EXPANSION_DETECTED" in codes:
        supporting_factors.append("Destination category (urban) demonstrated concurrent built-up surface expansion.")
    else:
        opposing_factors.append("Destination category (urban) showed no significant built-up expansion.")

    if state in ["candidate", "strong_candidate"]:
        conclusion = "Evidence supports a candidate vegetation-to-urban transition pattern."
        summary = (
            f"Multi-index analysis identified paired evidence of vegetation loss accompanied by concurrent built-up expansion (evidence score: {score:.3f}). "
            "Both conversion endpoints demonstrate corroborating spectral shifts. Spatial contiguity validation is required before confirming land-cover conversion."
        )
    elif state == "uncertain":
        conclusion = "Evidence for vegetation-to-urban transition is inconclusive."
        summary = "Indicators for one or both transition endpoints exhibit contradictory or ambiguous responses across the target area."
    elif state == "unavailable":
        conclusion = "Transition assessment is unavailable due to insufficient observation quality."
        summary = "Atmospheric interference or masked pixels prevent reliable multi-endpoint transition analysis."
    else:  # no_support
        conclusion = "No strong evidence supports a vegetation-to-urban transition."
        summary = (
            "Multi-index evaluation found no dual-endpoint evidence supporting vegetation conversion into built-up land. "
            "Observed index shifts remained within baseline limits for both vegetation and urban indicators."
        )

    return StructuredInterpretation(
        conclusion=conclusion,
        summary=summary,
        target="transition",
        hypothesis=hyp,
        state=state,
        evidence_summary=f"Source support: {supp.get('source_vegetation_evidence', 0.0):.3f}, Destination support: {supp.get('destination_urban_evidence', 0.0):.3f}.",
        supporting_factors=supporting_factors,
        opposing_factors=opposing_factors,
        reliability_summary=format_reliability_summary(rel, is_valid=state != "unavailable"),
        limitations=STANDARD_LIMITATIONS + ["Transition candidates require spatial polygon overlap verification (Phase 6)."],
        reason_codes=codes,
        details={"candidate": candidate},
    )


def interpret_general_change(
    candidates_list: List[Dict[str, Any]],
    multi_index_evidence: Dict[str, Any],
) -> StructuredInterpretation:
    """
    Generates structured multi-domain interpretation for general change queries ("What changed?").
    """
    findings: List[Dict[str, Any]] = []
    active_findings: List[str] = []

    rel = float(multi_index_evidence.get("metadata", {}).get("quality_fraction") or 1.0)
    all_codes: List[str] = []

    for cand in candidates_list:
        tgt = cand.get("target")
        if tgt == "urban":
            interp = interpret_urban_candidate(cand, multi_index_evidence)
        elif tgt == "water":
            interp = interpret_water_candidate(cand, multi_index_evidence)
        elif tgt == "vegetation":
            interp = interpret_vegetation_candidate(cand, multi_index_evidence)
        else:
            continue

        findings.append(interp.to_dict())
        all_codes.extend(interp.reason_codes)

        if interp.state in ["candidate", "strong_candidate"]:
            active_findings.append(f"{tgt.capitalize()} ({interp.hypothesis})")

    if active_findings:
        conclusion = f"Evidence indicates multiple candidate change patterns: {', '.join(active_findings)}."
        summary = (
            f"Multi-domain satellite evaluation identified active change candidates across {len(active_findings)} category domains: "
            f"{', '.join(active_findings)}. Each domain exhibits distinct physical index and spectral movements."
        )
        overall_state = "candidate"
    else:
        conclusion = "No strong land-cover change was identified from the available evidence."
        summary = (
            "Multi-domain satellite analysis evaluated urban, vegetation, and water surface indicators across the target area. "
            "All physical indicators remained within baseline noise deadbands, indicating general stability across the evaluated interval."
        )
        overall_state = "no_support"

    return StructuredInterpretation(
        conclusion=conclusion,
        summary=summary,
        target="general",
        hypothesis="general_change_detection",
        state=overall_state,
        evidence_summary=f"Evaluated {len(findings)} environmental domains.",
        supporting_factors=[f"{f['target'].capitalize()}: {f['conclusion']}" for f in findings if f['state'] in ['candidate', 'strong_candidate']],
        opposing_factors=[f"{f['target'].capitalize()}: {f['conclusion']}" for f in findings if f['state'] not in ['candidate', 'strong_candidate']],
        reliability_summary=format_reliability_summary(rel),
        limitations=STANDARD_LIMITATIONS,
        reason_codes=list(set(all_codes)),
        details={"findings": findings},
    )


# ============================================================
# MAIN INTERPRETATION ORCHESTRATOR
# ============================================================

def generate_structured_interpretation(
    candidate_package: Dict[str, Any],
    multi_index_evidence: Dict[str, Any],
    target: Optional[str] = None,
    task: Optional[str] = None,
    spatial_analysis: Optional[Dict[str, Any]] = None,
    temporal_analysis: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main orchestration entry point: derives structured interpretation from Phase 5B candidate outputs
    and incorporates Phase 6 spatial reasoning, Phase 7 temporal reasoning, and Phase 8 calibration findings.
    """
    target_clean = (target or "").lower().strip()
    task_clean = (task or "").lower().strip()

    candidates = candidate_package.get("candidates", [])
    primary_cand = candidate_package.get("primary_candidate") or (candidates[0] if candidates else {})

    is_general_change = (
        not target_clean
        or target_clean == "none"
        or "what changed" in task_clean
        or "detect_change" in task_clean
    )
    is_transition = (
        target_clean == "transition"
        or "transition" in task_clean
        or "become" in task_clean
        or "turned into" in task_clean
    )

    if is_transition:
        interpretation = interpret_transition_candidate(primary_cand, multi_index_evidence)
    elif is_general_change and len(candidates) > 1:
        interpretation = interpret_general_change(candidates, multi_index_evidence)
    elif target_clean == "urban" or "urban" in task_clean:
        interpretation = interpret_urban_candidate(primary_cand, multi_index_evidence)
    elif target_clean == "water" or "water" in task_clean:
        interpretation = interpret_water_candidate(primary_cand, multi_index_evidence)
    else:
        interpretation = interpret_vegetation_candidate(primary_cand, multi_index_evidence)

    # Incorporate Phase 6 Spatial Reasoning evidence if available
    if spatial_analysis and spatial_analysis.get("available"):
        reg_count = int(spatial_analysis.get("region_count", 0))
        tot_ha = float(spatial_analysis.get("total_candidate_area_hectares", 0.0))
        largest_ha = float(spatial_analysis.get("largest_region", {}).get("area_hectares", 0.0)) if spatial_analysis.get("largest_region") else 0.0
        dom_loc = spatial_analysis.get("dominant_location_description", "across the observation area")
        spatial_sum = spatial_analysis.get("summary", "")

        interpretation.region_count = reg_count
        interpretation.candidate_area_hectares = tot_ha
        interpretation.largest_region_area_hectares = largest_ha
        interpretation.dominant_location_description = dom_loc
        interpretation.spatial_summary = spatial_sum

        if reg_count > 0 and interpretation.state in ["candidate", "strong_candidate"]:
            interpretation.conclusion = (
                f"Potential {interpretation.target} change is concentrated in {reg_count} spatially coherent candidate region(s) "
                f"totaling {tot_ha:.2f} hectares, {dom_loc}."
            )
            interpretation.supporting_factors.append(
                f"Spatial clustering identified {reg_count} coherent candidate region(s) totaling {tot_ha:.2f} hectares, {dom_loc}."
            )
        elif reg_count == 0:
            interpretation.spatial_summary = "No spatially coherent candidate regions met the minimum mapping unit threshold."

    # Incorporate Phase 7 Temporal Reasoning evidence if available
    if temporal_analysis and temporal_analysis.get("available"):
        t_mode = temporal_analysis.get("temporal_mode", "bi_temporal")
        t_sum = temporal_analysis.get("summary", "")
        seas = temporal_analysis.get("seasonal_comparability", {}).get("comparability", "high")
        domains = temporal_analysis.get("domains", {})
        prim_dom = temporal_analysis.get("primary_domain", "vegetation")
        p_trend = domains.get(prim_dom, {})

        interpretation.temporal_mode = t_mode
        interpretation.temporal_summary = t_sum
        interpretation.seasonal_comparability = seas
        interpretation.temporal_trend = p_trend.get("direction", "stable")
        interpretation.persistence_fraction = p_trend.get("persistence_fraction")
        interpretation.change_nature = p_trend.get("change_type", "insufficient_data")

        usable_count = temporal_analysis.get("usable_observation_count", 0)
        if usable_count >= 3:
            dir_str = p_trend.get("direction", "stable")
            slope_str = f"{p_trend.get('annualized_slope', 0.0):+.4f}/yr" if p_trend.get("annualized_slope") is not None else "N/A"
            pers_pct = int((p_trend.get("persistence_fraction") or 0.0) * 100)
            chg_nat = p_trend.get("change_type", "gradual")

            temporal_fact = (
                f"Multi-temporal analysis ({usable_count} usable observations, {seas} seasonal comparability) "
                f"shows a {dir_str} trend (slope: {slope_str}, {pers_pct}% persistence, {chg_nat} change)."
            )
            interpretation.supporting_factors.append(temporal_fact)

            if p_trend.get("reversal_detected"):
                rev_det = p_trend.get("reversal_details", {})
                rev_fact = f"Trajectory reversal ({rev_det.get('reversal_direction')}) detected around {rev_det.get('observation_inflection')}."
                interpretation.supporting_factors.append(rev_fact)

            if interpretation.state in ["candidate", "strong_candidate"]:
                interpretation.conclusion += f" Temporal evolution across {usable_count} observations was {chg_nat}."
        elif usable_count == 2:
            interpretation.supporting_factors.append(
                f"Bi-temporal observation pair evaluated with {seas} seasonal comparability. Multi-year trend analysis requires >= 3 observations."
            )

        # Transition temporal ordering factor
        trans_ord = temporal_analysis.get("transition_temporal_ordering")
        if trans_ord and trans_ord.get("available"):
            if trans_ord.get("temporal_order_valid"):
                interpretation.supporting_factors.append(
                    f"Temporal sequence supported: vegetation decline preceded or coincided with urban expansion."
                )
            else:
                interpretation.opposing_factors.append(
                    f"Temporal sequence inconsistent: urban signal increase preceded detected vegetation loss."
                )

    # Incorporate Phase 8 Calibration evidence if available
    if calibration:
        interp_support = calibration.get("interpretation_support", {})
        obs_rel = calibration.get("observation_reliability", {})
        sem_ev = calibration.get("semantic_evidence", {})
        data_suff = calibration.get("data_sufficiency", {})

        interpretation.calibration_summary = interp_support.get("summary")
        interpretation.interpretation_support_state = interp_support.get("state")
        interpretation.data_sufficiency_state = data_suff.get("state")

        # Distinct explanation nuances based on calibration
        if obs_rel.get("state") == "low":
            interpretation.summary = "Available imagery quality was insufficient to make a reliable change assessment."
            interpretation.conclusion = "Analysis restricted due to low observation reliability (high cloud/shadow contamination or insufficient valid pixels)."
        elif interp_support.get("state") == "contradictory_support":
            interpretation.summary = "Physical indicators materially disagree, producing contradictory support for the candidate hypothesis."
            interpretation.conclusion = "Inconclusive due to conflicting physical signals."
        elif sem_ev.get("state") == "none" and obs_rel.get("state") in ["high", "moderate"]:
            interpretation.summary = f"No strong evidence of {interpretation.target} change was found. Observation reliability was {obs_rel.get('state')}, but the measured physical signals did not strongly support change."
            interpretation.conclusion = f"No significant {interpretation.target} change detected within physical noise bounds."

        # Add distinct reason codes from calibration
        for rc in calibration.get("reason_codes", []):
            if rc not in interpretation.reason_codes:
                interpretation.reason_codes.append(rc)

    return interpretation.to_dict()
