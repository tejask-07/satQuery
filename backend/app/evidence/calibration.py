"""
Phase 8: Reliability & Confidence Calibration Engine.

Establishes mathematically and semantically separate calibration components:
1. Observation Reliability (data quality / observation fidelity only)
2. Semantic Evidence Strength (physical spectral signal support intensity)
3. Spatial Coherence (geometric contiguity, compactness, clustering)
4. Temporal Consistency (multi-observation persistence and trajectory stability)
5. Overall Interpretation Support (independent lines of evidence corroboration)

SCIENTIFIC PRINCIPLES:
- ZERO FAKE PROBABILITIES: Scores and support states are deterministic evidence summaries.
  They are NOT probabilities of semantic correctness and are NOT empirically calibrated probabilities.
- NO BLIND AVERAGING: The five components are never collapsed or multiplied into a single pseudo-confidence.
- NO-CHANGE != INSUFFICIENT DATA: High-quality imagery with neutral physical signals reports
  "no strong evidence of change", NOT "low confidence" or "insufficient data".
- BI-TEMPORAL LIMITATION: For N=2 observations, temporal consistency is strictly classified as
  "bi_temporal_only" and never asserts a multi-temporal trend.
- CONFLICT INTEGRATION: Conflicting indicators (e.g. NDBI up + NDVI up) are explicitly surfaced,
  never hidden or smoothed over.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np


# ============================================================
# CENTRALIZED CALIBRATION CONFIGURATION & THRESHOLDS
# ============================================================

class CalibrationConfig:
    """
    Centralized, documented configuration thresholds and deterministic bands for Phase 8.
    All magic numbers are maintained in this single location.
    """

    # --- Evidence Strength Bands ---
    # 0.00 - 0.17: none
    # 0.18 - 0.34: weak
    # 0.35 - 0.59: moderate
    # 0.60 - 0.79: strong
    # 0.80 - 1.00: very_strong
    EVIDENCE_BAND_NONE_MAX: float = 0.17
    EVIDENCE_BAND_WEAK_MAX: float = 0.34
    EVIDENCE_BAND_MODERATE_MAX: float = 0.59
    EVIDENCE_BAND_STRONG_MAX: float = 0.79

    # --- Observation Reliability Bands ---
    # 0.00 - 0.49: low
    # 0.50 - 0.79: moderate
    # 0.80 - 1.00: high
    RELIABILITY_BAND_LOW_MAX: float = 0.49
    RELIABILITY_BAND_MODERATE_MAX: float = 0.79

    # --- Spatial Coherence Bands ---
    SPATIAL_BAND_LOW_MAX: float = 0.20
    SPATIAL_BAND_MODERATE_MAX: float = 0.50

    # Minimum acceptable valid pixel fraction for an observation to be usable
    MIN_USABLE_VALID_FRACTION: float = 0.50

    # Minimum observations required for trend sufficiency
    MIN_OBSERVATIONS_TREND: int = 3

    # Minimum observations required for acceleration sufficiency
    MIN_OBSERVATIONS_ACCELERATION: int = 4

    # Spatial minimum mapping unit pixel threshold
    SPATIAL_MIN_PIXELS: int = 5

    # Conflict thresholds
    CONFLICT_CONCURRENT_DELTA_THRESHOLD: float = 0.08
    CONFLICT_AMBIGUITY_MARGIN: float = 0.12


# ============================================================
# CENTRALIZED REASON CODES TAXONOMY
# ============================================================

class CalibrationReasonCodes:
    """
    Standardized, centralized vocabulary of calibration reason codes.
    """
    # Observation Quality
    DATA_HIGH_QUALITY = "DATA_HIGH_QUALITY"
    DATA_MODERATE_QUALITY = "DATA_MODERATE_QUALITY"
    DATA_LOW_QUALITY = "DATA_LOW_QUALITY"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    INSUFFICIENT_VALID_PIXELS = "INSUFFICIENT_VALID_PIXELS"
    INSUFFICIENT_TEMPORAL_OBSERVATIONS = "INSUFFICIENT_TEMPORAL_OBSERVATIONS"
    BI_TEMPORAL_ONLY = "BI_TEMPORAL_ONLY"
    HIGH_CLOUD_COVER = "HIGH_CLOUD_COVER"
    SEASONAL_MATCH_HIGH = "SEASONAL_MATCH_HIGH"
    SEASONAL_MATCH_MODERATE = "SEASONAL_MATCH_MODERATE"
    SEASONAL_MISMATCH = "SEASONAL_MISMATCH"

    # Semantic Evidence Strength
    VERY_STRONG_SEMANTIC_SUPPORT = "VERY_STRONG_SEMANTIC_SUPPORT"
    STRONG_SEMANTIC_SUPPORT = "STRONG_SEMANTIC_SUPPORT"
    MODERATE_SEMANTIC_SUPPORT = "MODERATE_SEMANTIC_SUPPORT"
    WEAK_SEMANTIC_SUPPORT = "WEAK_SEMANTIC_SUPPORT"
    NO_SEMANTIC_SUPPORT = "NO_SEMANTIC_SUPPORT"
    SEMANTIC_EVIDENCE_UNAVAILABLE = "SEMANTIC_EVIDENCE_UNAVAILABLE"
    NEUTRAL_PHYSICAL_SIGNALS = "NEUTRAL_PHYSICAL_SIGNALS"

    # Spatial Support
    STRONG_SPATIAL_SUPPORT = "STRONG_SPATIAL_SUPPORT"
    MODERATE_SPATIAL_SUPPORT = "MODERATE_SPATIAL_SUPPORT"
    LIMITED_SPATIAL_SUPPORT = "LIMITED_SPATIAL_SUPPORT"
    NO_SPATIAL_SUPPORT = "NO_SPATIAL_SUPPORT"
    SPATIAL_ASSESSMENT_UNAVAILABLE = "SPATIAL_ASSESSMENT_UNAVAILABLE"
    LOW_SPATIAL_COHERENCE = "LOW_SPATIAL_COHERENCE"
    FRAGMENTED_CANDIDATE_REGIONS = "FRAGMENTED_CANDIDATE_REGIONS"

    # Temporal Consistency
    STRONG_TEMPORAL_SUPPORT = "STRONG_TEMPORAL_SUPPORT"
    MODERATE_TEMPORAL_SUPPORT = "MODERATE_TEMPORAL_SUPPORT"
    LIMITED_TEMPORAL_SUPPORT = "LIMITED_TEMPORAL_SUPPORT"
    NO_TEMPORAL_SUPPORT = "NO_TEMPORAL_SUPPORT"
    TEMPORAL_ASSESSMENT_UNAVAILABLE = "TEMPORAL_ASSESSMENT_UNAVAILABLE"
    PERSISTENT_TRAJECTORY = "PERSISTENT_TRAJECTORY"
    REVERSAL_PRESENT = "REVERSAL_PRESENT"
    MONOTONIC_TREND = "MONOTONIC_TREND"
    MIXED_TEMPORAL_STEPS = "MIXED_TEMPORAL_STEPS"

    # Interpretation Support
    STRONG_INTERPRETATION_SUPPORT = "STRONG_INTERPRETATION_SUPPORT"
    MODERATE_INTERPRETATION_SUPPORT = "MODERATE_INTERPRETATION_SUPPORT"
    WEAK_INTERPRETATION_SUPPORT = "WEAK_INTERPRETATION_SUPPORT"
    INSUFFICIENT_INTERPRETATION_SUPPORT = "INSUFFICIENT_INTERPRETATION_SUPPORT"
    CONTRADICTORY_INTERPRETATION_SUPPORT = "CONTRADICTORY_INTERPRETATION_SUPPORT"

    # Data Sufficiency
    DATA_SUFFICIENT = "DATA_SUFFICIENT"
    DATA_LIMITED = "DATA_LIMITED"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"

    # Conflicts
    CONFLICTING_INDICATORS = "CONFLICTING_INDICATORS"
    CONFLICTING_NDBI_NDVI_EXPANSION = "CONFLICTING_NDBI_NDVI_EXPANSION"
    CONFLICTING_NDVI_NDBI_DECLINE = "CONFLICTING_NDVI_NDBI_DECLINE"
    CONFLICTING_SPECTRAL_INDEX_MISMATCH = "CONFLICTING_SPECTRAL_INDEX_MISMATCH"

    # Transitions
    TRANSITION_SOURCE_SUPPORTED = "TRANSITION_SOURCE_SUPPORTED"
    TRANSITION_DEST_SUPPORTED = "TRANSITION_DEST_SUPPORTED"
    TRANSITION_SPATIAL_OVERLAP_CONFIRMED = "TRANSITION_SPATIAL_OVERLAP_CONFIRMED"
    TRANSITION_TEMPORAL_ORDER_VALID = "TRANSITION_TEMPORAL_ORDER_VALID"
    TRANSITION_INCONCLUSIVE = "TRANSITION_INCONCLUSIVE"


# ============================================================
# DATA STRUCTURES
# ============================================================

ReliabilityState = Literal["high", "moderate", "low", "unavailable"]
EvidenceStrengthState = Literal["very_strong", "strong", "moderate", "weak", "none", "unavailable"]
SpatialState = Literal["high", "moderate", "low", "unavailable"]
TemporalState = Literal["high", "moderate", "limited", "bi_temporal_only", "unavailable"]
InterpretationSupportState = Literal[
    "strong_support",
    "moderate_support",
    "weak_support",
    "insufficient_support",
    "contradictory_support",
    "unavailable",
]
DataSufficiencyState = Literal["sufficient", "limited", "insufficient"]


@dataclass
class ObservationReliabilityAssessment:
    """
    Assessment of observation and data quality only. Strictly independent of phenomenon support.
    """
    score: float
    state: ReliabilityState
    valid_fraction: float
    cloud_fraction: float
    shadow_fraction: float
    coverage_fraction: float
    seasonal_comparability: str
    usable_observations: int
    total_observations: int
    reason_codes: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticEvidenceAssessment:
    """
    Assessment of physical signal intensity supporting a change candidate hypothesis.
    """
    score: float
    state: EvidenceStrengthState
    primary_target: str
    hypothesis: str
    domains: Dict[str, Any]
    reason_codes: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpatialAssessment:
    """
    Assessment of geometric compactness, contiguity, and candidate clustering.
    """
    score: float
    state: SpatialState
    region_count: int
    total_candidate_area_hectares: float
    largest_region_fraction: float
    dominant_location: str
    reason_codes: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalConsistencyAssessment:
    """
    Assessment of multi-observation trajectory stability and persistence.
    """
    score: Optional[float]
    state: TemporalState
    usable_observations: int
    persistence_fraction: Optional[float]
    direction_consistency: str
    monotonicity: Optional[Union[float, str]]
    change_nature: str
    reversal_detected: bool
    reason_codes: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataSufficiencyAssessment:
    """
    Assessment of whether available observations and imagery quality fulfill requirements for the query task.
    """
    state: DataSufficiencyState
    requirements: List[str]
    satisfied_requirements: List[str]
    missing_components: List[str]
    reason_codes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransitionSupportAssessment:
    """
    Assessment of multi-stage land cover transitions (e.g. vegetation -> urban).
    """
    state: str
    source_evidence_score: float
    destination_evidence_score: float
    spatial_support: str
    temporal_support: str
    spatial_overlap_fraction: float
    reason_codes: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OverallInterpretationSupport:
    """
    Multi-source corroboration assessment. Answers how many independent lines of evidence support the conclusion.
    This is NOT a probability.
    """
    state: InterpretationSupportState
    summary: str
    corroborating_factors: List[str]
    non_corroborating_factors: List[str]
    reason_codes: List[str]
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationPackage:
    """
    Comprehensive structured calibration package attaching all 5 separated components.
    """
    observation_reliability: ObservationReliabilityAssessment
    semantic_evidence: SemanticEvidenceAssessment
    spatial_assessment: SpatialAssessment
    temporal_consistency: TemporalConsistencyAssessment
    interpretation_support: OverallInterpretationSupport
    data_sufficiency: DataSufficiencyAssessment
    transition_support: Optional[TransitionSupportAssessment]
    conflicts: List[Dict[str, Any]]
    reason_codes: List[str]
    calibration_notice: str = (
        "Calibration scores and support states are deterministic evidence summaries. "
        "They are not probabilities of semantic correctness and are not empirically calibrated probabilities."
    )

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        return res


# ============================================================
# COMPONENT 1: OBSERVATION RELIABILITY EVALUATION
# ============================================================

def evaluate_observation_reliability(
    temporal_observations: Optional[List[Any]] = None,
    imagery_result: Optional[Dict[str, Any]] = None,
    execution_results: Optional[Dict[str, Any]] = None,
) -> ObservationReliabilityAssessment:
    """
    Evaluates observation reliability from physical data quality only.
    Input signals: valid pixel fraction, cloud cover, shadow cover, AOI coverage, seasonal comparability.
    """
    reason_codes: List[str] = []
    notes: List[str] = []

    # 1. Gather observation-level metrics
    valid_fractions: List[float] = []
    cloud_fractions: List[float] = []
    shadow_fractions: List[float] = []
    coverage_fractions: List[float] = []

    if temporal_observations:
        for obs in temporal_observations:
            # Handle dataclass or dict
            v = getattr(obs, "valid_fraction", None) if not isinstance(obs, dict) else obs.get("valid_fraction")
            c = getattr(obs, "cloud_cover", None) if not isinstance(obs, dict) else obs.get("cloud_cover")
            cov = getattr(obs, "coverage_fraction", None) if not isinstance(obs, dict) else obs.get("coverage_fraction")
            if v is not None:
                valid_fractions.append(float(v))
            if c is not None:
                # Cloud cover is usually percentage 0-100 or fraction 0-1
                c_val = float(c)
                cloud_fractions.append(c_val / 100.0 if c_val > 1.0 else c_val)
            if cov is not None:
                coverage_fractions.append(float(cov))
    elif imagery_result and imagery_result.get("images"):
        for img in imagery_result["images"]:
            q = img.get("quality", {})
            v = q.get("valid_coverage_percentage") or q.get("valid_percentage")
            if v is not None:
                valid_fractions.append(float(v) / 100.0)
            c = img.get("cloud_cover")
            if c is not None:
                c_val = float(c)
                cloud_fractions.append(c_val / 100.0 if c_val > 1.0 else c_val)
            sh = q.get("shadow_percentage_inside_aoi")
            if sh is not None:
                shadow_fractions.append(float(sh) / 100.0)
            coverage_fractions.append(1.0)

    total_obs = max(len(valid_fractions), 1 if valid_fractions else 0)
    usable_obs = sum(1 for v in valid_fractions if v >= CalibrationConfig.MIN_USABLE_VALID_FRACTION)

    if not valid_fractions:
        return ObservationReliabilityAssessment(
            score=0.0,
            state="unavailable",
            valid_fraction=0.0,
            cloud_fraction=0.0,
            shadow_fraction=0.0,
            coverage_fraction=0.0,
            seasonal_comparability="unavailable",
            usable_observations=0,
            total_observations=0,
            reason_codes=[CalibrationReasonCodes.DATA_UNAVAILABLE],
            notes=["No valid satellite observation metadata available."],
        )

    mean_valid = float(np.mean(valid_fractions))
    mean_cloud = float(np.mean(cloud_fractions)) if cloud_fractions else 0.0
    mean_shadow = float(np.mean(shadow_fractions)) if shadow_fractions else 0.0
    mean_coverage = float(np.mean(coverage_fractions)) if coverage_fractions else 1.0

    # Seasonal comparability check if multi-temporal
    seasonal_comp = "high"
    if temporal_observations and len(temporal_observations) >= 2:
        doys = []
        for obs in temporal_observations:
            d = getattr(obs, "day_of_year", None) if not isinstance(obs, dict) else obs.get("day_of_year")
            if d is not None:
                doys.append(int(d))
        if len(doys) >= 2:
            base_doy = doys[0]
            max_doy_diff = 0
            for d in doys[1:]:
                diff = abs(d - base_doy)
                circ_diff = min(diff, 365 - diff)
                if circ_diff > max_doy_diff:
                    max_doy_diff = circ_diff
            if max_doy_diff <= 45:
                seasonal_comp = "high"
                reason_codes.append(CalibrationReasonCodes.SEASONAL_MATCH_HIGH)
            elif max_doy_diff <= 90:
                seasonal_comp = "moderate"
                reason_codes.append(CalibrationReasonCodes.SEASONAL_MATCH_MODERATE)
            else:
                seasonal_comp = "low"
                reason_codes.append(CalibrationReasonCodes.SEASONAL_MISMATCH)
                notes.append(f"Seasonal day-of-year disparity of {max_doy_diff} days may induce phenological shifts.")

    # Empirical data-quality index: 0.70 * valid_fraction + 0.20 * coverage + 0.10 * (1 - cloud)
    reliability_score = (
        0.70 * mean_valid +
        0.20 * min(1.0, mean_coverage) +
        0.10 * max(0.0, 1.0 - mean_cloud)
    )
    reliability_score = float(round(np.clip(reliability_score, 0.0, 1.0), 4))

    # Classify deterministic state
    if reliability_score < CalibrationConfig.RELIABILITY_BAND_LOW_MAX or mean_valid < CalibrationConfig.MIN_USABLE_VALID_FRACTION:
        state: ReliabilityState = "low"
        reason_codes.append(CalibrationReasonCodes.DATA_LOW_QUALITY)
        if mean_valid < CalibrationConfig.MIN_USABLE_VALID_FRACTION:
            reason_codes.append(CalibrationReasonCodes.INSUFFICIENT_VALID_PIXELS)
        notes.append(f"Observation reliability is low ({round(mean_valid * 100, 1)}% valid pixel fraction).")
    elif reliability_score <= CalibrationConfig.RELIABILITY_BAND_MODERATE_MAX:
        state = "moderate"
        reason_codes.append(CalibrationReasonCodes.DATA_MODERATE_QUALITY)
        notes.append(f"Observation reliability is moderate ({round(mean_valid * 100, 1)}% valid pixel fraction).")
    else:
        state = "high"
        reason_codes.append(CalibrationReasonCodes.DATA_HIGH_QUALITY)
        notes.append(f"Observation reliability is high ({round(mean_valid * 100, 1)}% cloud/shadow-free valid pixels).")

    if mean_cloud > 0.30:
        reason_codes.append(CalibrationReasonCodes.HIGH_CLOUD_COVER)
        notes.append(f"Cloud contamination detected ({round(mean_cloud * 100, 1)}% mean cloud cover).")

    return ObservationReliabilityAssessment(
        score=reliability_score,
        state=state,
        valid_fraction=round(mean_valid, 4),
        cloud_fraction=round(mean_cloud, 4),
        shadow_fraction=round(mean_shadow, 4),
        coverage_fraction=round(mean_coverage, 4),
        seasonal_comparability=seasonal_comp,
        usable_observations=usable_obs,
        total_observations=total_obs,
        reason_codes=list(dict.fromkeys(reason_codes)),
        notes=notes,
    )


# ============================================================
# COMPONENT 2: SEMANTIC EVIDENCE STRENGTH EVALUATION
# ============================================================

def classify_evidence_strength(score: float) -> EvidenceStrengthState:
    """
    Deterministic mapping from continuous evidence score to controlled strength band.
    """
    if score <= CalibrationConfig.EVIDENCE_BAND_NONE_MAX:
        return "none"
    if score <= CalibrationConfig.EVIDENCE_BAND_WEAK_MAX:
        return "weak"
    if score <= CalibrationConfig.EVIDENCE_BAND_MODERATE_MAX:
        return "moderate"
    if score <= CalibrationConfig.EVIDENCE_BAND_STRONG_MAX:
        return "strong"
    return "very_strong"


def evaluate_semantic_evidence(
    candidate_package: Optional[Dict[str, Any]] = None,
    multi_index_evidence: Optional[Dict[str, Any]] = None,
    target: str = "urban",
) -> SemanticEvidenceAssessment:
    """
    Evaluates physical signal intensity supporting a change candidate hypothesis.
    Extracts scores directly from Phase 5A / 5B candidate outputs without changing equations.
    """
    reason_codes: List[str] = []
    notes: List[str] = []
    domains: Dict[str, Any] = {}

    primary_score = 0.0
    primary_hyp = "no_hypothesis"

    # 1. Extract domain-specific candidates from candidate_package
    candidates = candidate_package.get("candidates", []) if candidate_package else []
    for cand in candidates:
        t = cand.get("target", "")
        hyp = cand.get("hypothesis", "")
        sc = float(cand.get("final_evidence_score", cand.get("semantic_support", 0.0)))
        state = cand.get("state", "unavailable")
        domains[t] = {
            "hypothesis": hyp,
            "score": round(sc, 4),
            "state": state,
            "evidence_strength": classify_evidence_strength(sc),
            "supporting_evidence": cand.get("supporting_evidence", {}),
            "opposing_evidence": cand.get("opposing_evidence", {}),
        }

    # Select primary target score
    if target in domains:
        primary_score = domains[target]["score"]
        primary_hyp = domains[target]["hypothesis"]
    elif "transition" in domains:
        primary_score = domains["transition"]["score"]
        primary_hyp = domains["transition"]["hypothesis"]
    elif domains:
        # Pick domain with highest score
        best_t = max(domains.keys(), key=lambda k: domains[k]["score"])
        primary_score = domains[best_t]["score"]
        primary_hyp = domains[best_t]["hypothesis"]

    strength_state = classify_evidence_strength(primary_score)

    if strength_state == "very_strong":
        reason_codes.append(CalibrationReasonCodes.VERY_STRONG_SEMANTIC_SUPPORT)
        notes.append(f"Very strong physical spectral evidence supports {primary_hyp} (score={primary_score:.4f}).")
    elif strength_state == "strong":
        reason_codes.append(CalibrationReasonCodes.STRONG_SEMANTIC_SUPPORT)
        notes.append(f"Strong physical spectral evidence supports {primary_hyp} (score={primary_score:.4f}).")
    elif strength_state == "moderate":
        reason_codes.append(CalibrationReasonCodes.MODERATE_SEMANTIC_SUPPORT)
        notes.append(f"Moderate physical evidence supports {primary_hyp} (score={primary_score:.4f}).")
    elif strength_state == "weak":
        reason_codes.append(CalibrationReasonCodes.WEAK_SEMANTIC_SUPPORT)
        notes.append(f"Weak physical spectral signal detected for {primary_hyp} (score={primary_score:.4f}).")
    else:
        reason_codes.append(CalibrationReasonCodes.NO_SEMANTIC_SUPPORT)
        reason_codes.append(CalibrationReasonCodes.NEUTRAL_PHYSICAL_SIGNALS)
        notes.append(f"Measured physical spectral indices are within the neutral deadband for {primary_hyp}.")

    return SemanticEvidenceAssessment(
        score=round(primary_score, 4),
        state=strength_state,
        primary_target=target,
        hypothesis=primary_hyp,
        domains=domains,
        reason_codes=list(dict.fromkeys(reason_codes)),
        notes=notes,
    )


# ============================================================
# COMPONENT 3: SPATIAL ASSESSMENT EVALUATION
# ============================================================

def evaluate_spatial_assessment(
    spatial_analysis: Optional[Dict[str, Any]] = None,
) -> SpatialAssessment:
    """
    Evaluates geometric contiguity, clustering, and spatial coherence from Phase 6.
    """
    reason_codes: List[str] = []
    notes: List[str] = []

    if not spatial_analysis or not spatial_analysis.get("available"):
        return SpatialAssessment(
            score=0.0,
            state="unavailable",
            region_count=0,
            total_candidate_area_hectares=0.0,
            largest_region_fraction=0.0,
            dominant_location="unavailable",
            reason_codes=[CalibrationReasonCodes.SPATIAL_ASSESSMENT_UNAVAILABLE],
            notes=["Spatial reasoning analysis was not executed or not available."],
        )

    coherence = float(spatial_analysis.get("spatial_coherence", 0.0))
    region_count = int(spatial_analysis.get("region_count", 0))
    total_area_ha = float(spatial_analysis.get("total_candidate_area_hectares", 0.0))
    largest_ha = float(spatial_analysis.get("largest_region_area_hectares", 0.0))
    dom_loc = spatial_analysis.get("dominant_location_description", "across the observation area")

    largest_frac = float(largest_ha / total_area_ha) if total_area_ha > 0 else 0.0

    # Determine spatial state deterministically
    if region_count == 0:
        state: SpatialState = "low"
        reason_codes.append(CalibrationReasonCodes.NO_SPATIAL_SUPPORT)
        notes.append("No contiguous candidate regions passed the minimum mapping unit threshold.")
    elif coherence < CalibrationConfig.SPATIAL_BAND_LOW_MAX:
        state = "low"
        reason_codes.append(CalibrationReasonCodes.LIMITED_SPATIAL_SUPPORT)
        reason_codes.append(CalibrationReasonCodes.LOW_SPATIAL_COHERENCE)
        notes.append(f"Candidate regions exhibit low geometric coherence ({coherence:.2f}).")
    elif coherence <= CalibrationConfig.SPATIAL_BAND_MODERATE_MAX:
        state = "moderate"
        reason_codes.append(CalibrationReasonCodes.MODERATE_SPATIAL_SUPPORT)
        notes.append(f"Candidate regions exhibit moderate spatial contiguity ({coherence:.2f}).")
    else:
        state = "high"
        reason_codes.append(CalibrationReasonCodes.STRONG_SPATIAL_SUPPORT)
        notes.append(f"Candidate regions exhibit high spatial contiguity and compactness ({coherence:.2f}).")

    if region_count > 15 and largest_frac < 0.20:
        reason_codes.append(CalibrationReasonCodes.FRAGMENTED_CANDIDATE_REGIONS)
        notes.append("Candidate clusters are widely fragmented into numerous small parcels.")

    return SpatialAssessment(
        score=round(coherence, 4),
        state=state,
        region_count=region_count,
        total_candidate_area_hectares=round(total_area_ha, 4),
        largest_region_fraction=round(largest_frac, 4),
        dominant_location=dom_loc,
        reason_codes=list(dict.fromkeys(reason_codes)),
        notes=notes,
    )


# ============================================================
# COMPONENT 4: TEMPORAL CONSISTENCY EVALUATION
# ============================================================

def evaluate_temporal_consistency(
    temporal_analysis: Optional[Dict[str, Any]] = None,
    temporal_observations: Optional[List[Any]] = None,
    target: str = "urban",
    temporal_mode: str = "bi_temporal",
) -> TemporalConsistencyAssessment:
    """
    Evaluates multi-observation trajectory stability, persistence, and monotonicity from Phase 7.
    Crucially: For N=2, marks state as 'bi_temporal_only' and does NOT imply a trend.
    """
    reason_codes: List[str] = []
    notes: List[str] = []

    # Check if temporal_observations indicates bi-temporal even if temporal_analysis dict is absent
    obs_count = 0
    if temporal_analysis and temporal_analysis.get("available"):
        t_mode = temporal_analysis.get("temporal_mode", temporal_mode)
        obs_count = int(temporal_analysis.get("observation_count", 0))
        usable_obs = int(temporal_analysis.get("usable_observation_count", obs_count))
    elif temporal_observations:
        t_mode = temporal_mode
        obs_count = len(temporal_observations)
        usable_obs = sum(
            1 for o in temporal_observations
            if (getattr(o, "valid_fraction", 1.0) if not isinstance(o, dict) else o.get("valid_fraction", 1.0)) >= CalibrationConfig.MIN_USABLE_VALID_FRACTION
        )
    elif temporal_mode == "bi_temporal":
        t_mode = "bi_temporal"
        obs_count = 2
        usable_obs = 2
    else:
        return TemporalConsistencyAssessment(
            score=None,
            state="unavailable",
            usable_observations=0,
            persistence_fraction=None,
            direction_consistency="unknown",
            monotonicity=None,
            change_nature="unknown",
            reversal_detected=False,
            reason_codes=[CalibrationReasonCodes.TEMPORAL_ASSESSMENT_UNAVAILABLE],
            notes=["Temporal analysis not available."],
        )

    # CRITICAL RULE: Bi-temporal N=2 must not imply trend
    if obs_count < CalibrationConfig.MIN_OBSERVATIONS_TREND or t_mode == "bi_temporal":
        reason_codes.append(CalibrationReasonCodes.BI_TEMPORAL_ONLY)
        reason_codes.append(CalibrationReasonCodes.INSUFFICIENT_TEMPORAL_OBSERVATIONS)
        notes.append("Only bi-temporal observations available (N=2); insufficient to evaluate continuous temporal trend or persistence.")
        return TemporalConsistencyAssessment(
            score=None,
            state="bi_temporal_only",
            usable_observations=usable_obs,
            persistence_fraction=None,
            direction_consistency="bi_temporal_step",
            monotonicity=None,
            change_nature="insufficient_data",
            reversal_detected=False,
            reason_codes=list(dict.fromkeys(reason_codes)),
            notes=notes,
        )

    # Multi-temporal series (N >= 3)
    primary_domain = temporal_analysis.get("primary_domain") or target
    domain_data = temporal_analysis.get("domains", {}).get(primary_domain, {})

    persistence_frac = domain_data.get("persistence_fraction")
    monotonicity = domain_data.get("monotonicity")
    direction = domain_data.get("direction", "neutral")
    change_type = domain_data.get("change_type", "unknown")
    reversal = bool(domain_data.get("reversal_detected", False))

    if reversal:
        reason_codes.append(CalibrationReasonCodes.REVERSAL_PRESENT)
        notes.append("Temporal trajectory reverses direction over the observation series.")

    if persistence_frac is not None and persistence_frac >= 0.75 and not reversal:
        state: TemporalState = "high"
        reason_codes.append(CalibrationReasonCodes.STRONG_TEMPORAL_SUPPORT)
        reason_codes.append(CalibrationReasonCodes.PERSISTENT_TRAJECTORY)
        notes.append(f"Trajectory exhibits high temporal persistence ({round(persistence_frac * 100)}%).")
    elif persistence_frac is not None and persistence_frac >= 0.50:
        state = "moderate"
        reason_codes.append(CalibrationReasonCodes.MODERATE_TEMPORAL_SUPPORT)
        notes.append(f"Trajectory exhibits moderate temporal consistency ({round(persistence_frac * 100)}%).")
    else:
        state = "limited"
        reason_codes.append(CalibrationReasonCodes.LIMITED_TEMPORAL_SUPPORT)
        reason_codes.append(CalibrationReasonCodes.MIXED_TEMPORAL_STEPS)
        notes.append("Temporal trajectory has mixed or irregular directional steps.")

    is_monotonic = False
    if isinstance(monotonicity, (int, float)):
        is_monotonic = abs(monotonicity) >= 0.85
        mono_out: Optional[Union[float, str]] = round(float(monotonicity), 4)
    elif isinstance(monotonicity, str):
        is_monotonic = monotonicity in ["strictly_monotonic", "monotonic"]
        mono_out = monotonicity
    else:
        mono_out = None

    if is_monotonic:
        reason_codes.append(CalibrationReasonCodes.MONOTONIC_TREND)

    temporal_score = persistence_frac if persistence_frac is not None else 0.50

    return TemporalConsistencyAssessment(
        score=round(float(temporal_score), 4) if temporal_score is not None else None,
        state=state,
        usable_observations=usable_obs,
        persistence_fraction=round(float(persistence_frac), 4) if persistence_frac is not None else None,
        direction_consistency=str(direction),
        monotonicity=mono_out,
        change_nature=change_type,
        reversal_detected=reversal,
        reason_codes=list(dict.fromkeys(reason_codes)),
        notes=notes,
    )


# ============================================================
# COMPONENT 5: CONFLICT DETECTION
# ============================================================

def detect_evidence_conflicts(
    multi_index_evidence: Optional[Dict[str, Any]] = None,
    candidate_package: Optional[Dict[str, Any]] = None,
    temporal_consistency: Optional[TemporalConsistencyAssessment] = None,
    spatial_assessment: Optional[SpatialAssessment] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Explicitly identifies conflicting physical evidence or opposing spatial/temporal signals.
    Never hides conflicting evidence.
    """
    conflicts: List[Dict[str, Any]] = []
    conflict_codes: List[str] = []

    # 1. Delta checks across indices
    if multi_index_evidence:
        signals = multi_index_evidence.get("signals", {})
        ndbi_sig = signals.get("NDBI", {})
        ndvi_sig = signals.get("NDVI", {})
        ndwi_sig = signals.get("NDWI", {})

        ndbi_delta = ndbi_sig.get("raw_magnitude", 0.0)
        ndvi_delta = ndvi_sig.get("raw_magnitude", 0.0)
        ndwi_delta = ndwi_sig.get("raw_magnitude", 0.0)

        # Conflict: NDBI increases strongly while NDVI also increases strongly
        if ndbi_delta > CalibrationConfig.CONFLICT_CONCURRENT_DELTA_THRESHOLD and ndvi_delta > CalibrationConfig.CONFLICT_CONCURRENT_DELTA_THRESHOLD:
            conflicts.append({
                "type": "index_conflict",
                "indicators": ["NDBI", "NDVI"],
                "description": f"Both built-up index (NDBI delta = {ndbi_delta:+.4f}) and vegetation index (NDVI delta = {ndvi_delta:+.4f}) show concurrent positive expansion.",
                "code": CalibrationReasonCodes.CONFLICTING_NDBI_NDVI_EXPANSION,
            })
            conflict_codes.append(CalibrationReasonCodes.CONFLICTING_NDBI_NDVI_EXPANSION)

        # Conflict: NDVI decreases strongly while NDBI also decreases strongly
        if ndvi_delta < -CalibrationConfig.CONFLICT_CONCURRENT_DELTA_THRESHOLD and ndbi_delta < -CalibrationConfig.CONFLICT_CONCURRENT_DELTA_THRESHOLD:
            conflicts.append({
                "type": "index_conflict",
                "indicators": ["NDVI", "NDBI"],
                "description": f"Vegetation index declined (NDVI delta = {ndvi_delta:+.4f}) but built-up index also declined (NDBI delta = {ndbi_delta:+.4f}) without urban conversion.",
                "code": CalibrationReasonCodes.CONFLICTING_NDVI_NDBI_DECLINE,
            })
            conflict_codes.append(CalibrationReasonCodes.CONFLICTING_NDVI_NDBI_DECLINE)

    # 2. Temporal reversal conflict
    if temporal_consistency and temporal_consistency.reversal_detected:
        conflicts.append({
            "type": "temporal_reversal",
            "indicators": ["temporal_series"],
            "description": "Multi-observation series displays a net reversal in trajectory direction.",
            "code": CalibrationReasonCodes.REVERSAL_PRESENT,
        })
        conflict_codes.append(CalibrationReasonCodes.REVERSAL_PRESENT)

    # 3. Spatial fragmentation conflict
    if spatial_assessment and CalibrationReasonCodes.FRAGMENTED_CANDIDATE_REGIONS in spatial_assessment.reason_codes:
        conflicts.append({
            "type": "spatial_fragmentation",
            "indicators": ["candidate_regions"],
            "description": "Candidate pixels lack geometric cohesion and are fragmented across isolated micro-parcels.",
            "code": CalibrationReasonCodes.FRAGMENTED_CANDIDATE_REGIONS,
        })

    if conflict_codes:
        conflict_codes.append(CalibrationReasonCodes.CONFLICTING_INDICATORS)

    return conflicts, list(dict.fromkeys(conflict_codes))


# ============================================================
# COMPONENT 6: DATA SUFFICIENCY EVALUATION
# ============================================================

def evaluate_data_sufficiency(
    observation_reliability: ObservationReliabilityAssessment,
    temporal_consistency: TemporalConsistencyAssessment,
    task: str = "change_detection",
    temporal_mode: str = "bi_temporal",
) -> DataSufficiencyAssessment:
    """
    Evaluates whether observation count and quality satisfy the scientific requirements for the query task.
    Distinguishes 'insufficient data' from 'no physical change'.
    """
    requirements: List[str] = ["Minimum valid pixel fraction >= 50%"]
    satisfied: List[str] = []
    missing: List[str] = []
    reason_codes: List[str] = []

    # Reliability requirement
    if observation_reliability.valid_fraction >= CalibrationConfig.MIN_USABLE_VALID_FRACTION and observation_reliability.state != "low":
        satisfied.append("Observation quality adequate for analysis")
    else:
        missing.append("Observation quality below acceptable threshold (valid pixels < 50%)")
        reason_codes.append(CalibrationReasonCodes.INSUFFICIENT_VALID_PIXELS)

    # Task & temporal requirements
    if temporal_mode in ["trend_analysis", "multi_temporal", "persistence_reversal"]:
        requirements.append("Minimum 3 usable temporal observations for trend")
        if temporal_consistency.usable_observations >= CalibrationConfig.MIN_OBSERVATIONS_TREND:
            satisfied.append("Sufficient temporal observations for trend analysis (N >= 3)")
        else:
            missing.append("Fewer than 3 usable observations available for multi-temporal trend")
            reason_codes.append(CalibrationReasonCodes.INSUFFICIENT_TEMPORAL_OBSERVATIONS)
    elif temporal_mode == "acceleration":
        requirements.append("Minimum 4 usable temporal observations for acceleration")
        if temporal_consistency.usable_observations >= CalibrationConfig.MIN_OBSERVATIONS_ACCELERATION:
            satisfied.append("Sufficient observations for acceleration analysis (N >= 4)")
        else:
            missing.append("Fewer than 4 usable observations available for acceleration analysis")
            reason_codes.append(CalibrationReasonCodes.INSUFFICIENT_TEMPORAL_OBSERVATIONS)
    else:
        # Bi-temporal query
        requirements.append("Minimum 2 valid observation epochs for bi-temporal comparison")
        if observation_reliability.total_observations >= 2:
            satisfied.append("Sufficient bi-temporal observation epochs (N=2)")
        else:
            missing.append("Fewer than 2 observation epochs available")
            reason_codes.append(CalibrationReasonCodes.INSUFFICIENT_TEMPORAL_OBSERVATIONS)

    # Determine overall data sufficiency state
    if not missing:
        state: DataSufficiencyState = "sufficient"
        reason_codes.append(CalibrationReasonCodes.DATA_SUFFICIENT)
    elif observation_reliability.state == "low" or len(satisfied) == 0:
        state = "insufficient"
        reason_codes.append(CalibrationReasonCodes.DATA_INSUFFICIENT)
    else:
        state = "limited"
        reason_codes.append(CalibrationReasonCodes.DATA_LIMITED)

    return DataSufficiencyAssessment(
        state=state,
        requirements=requirements,
        satisfied_requirements=satisfied,
        missing_components=missing,
        reason_codes=list(dict.fromkeys(reason_codes)),
    )


# ============================================================
# COMPONENT 7: TRANSITION SUPPORT EVALUATION
# ============================================================

def evaluate_transition_support(
    candidate_package: Optional[Dict[str, Any]] = None,
    spatial_analysis: Optional[Dict[str, Any]] = None,
    temporal_analysis: Optional[Dict[str, Any]] = None,
    observation_reliability: Optional[ObservationReliabilityAssessment] = None,
) -> Optional[TransitionSupportAssessment]:
    """
    Evaluates multi-stage transition corroboration (e.g. vegetation loss -> urban expansion).
    Synthesizes source evidence, destination evidence, spatial overlap, and temporal ordering independently.
    """
    if not candidate_package:
        return None

    trans_cand = None
    for cand in candidate_package.get("candidates", []):
        if cand.get("target") == "transition":
            trans_cand = cand
            break

    if not trans_cand:
        return None

    reason_codes: List[str] = []
    notes: List[str] = []

    src_score = float(trans_cand.get("details", {}).get("source_support", 0.0))
    dest_score = float(trans_cand.get("details", {}).get("destination_support", 0.0))

    if src_score >= CalibrationConfig.EVIDENCE_BAND_WEAK_MAX:
        reason_codes.append(CalibrationReasonCodes.TRANSITION_SOURCE_SUPPORTED)
    if dest_score >= CalibrationConfig.EVIDENCE_BAND_WEAK_MAX:
        reason_codes.append(CalibrationReasonCodes.TRANSITION_DEST_SUPPORTED)

    # Spatial overlap fraction
    overlap_frac = 0.0
    sp_support = "no_spatial_overlap"
    if spatial_analysis and spatial_analysis.get("available"):
        overlap_frac = float(spatial_analysis.get("transition_overlap_fraction", 0.0))
        if overlap_frac >= 0.15:
            sp_support = "confirmed_spatial_overlap"
            reason_codes.append(CalibrationReasonCodes.TRANSITION_SPATIAL_OVERLAP_CONFIRMED)
            notes.append(f"Spatial overlap of {round(overlap_frac * 100, 1)}% confirms co-location of source loss and destination gain.")
        elif overlap_frac > 0.0:
            sp_support = "limited_spatial_overlap"
            notes.append("Spatial co-location between source loss and destination gain is below threshold.")
        else:
            sp_support = "no_spatial_overlap"
            notes.append("Zero spatial overlap between source loss and destination gain.")

    # Temporal ordering
    temp_support = "bi_temporal_concurrent"
    if temporal_analysis and temporal_analysis.get("available"):
        t_order = temporal_analysis.get("transition_temporal_ordering") or temporal_analysis.get("transition_ordering") or {}
        if t_order.get("sequence_supported") or t_order.get("temporal_order_valid"):
            temp_support = "sequential_transition_supported"
            reason_codes.append(CalibrationReasonCodes.TRANSITION_TEMPORAL_ORDER_VALID)
            notes.append("Temporal ordering confirms source decline preceded or accompanied destination increase.")
        elif temporal_analysis.get("temporal_mode") == "bi_temporal":
            temp_support = "bi_temporal_only"

    # Overall transition state
    if (
        src_score > CalibrationConfig.EVIDENCE_BAND_WEAK_MAX
        and dest_score > CalibrationConfig.EVIDENCE_BAND_WEAK_MAX
        and observation_reliability
        and observation_reliability.state != "low"
    ):
        state = "supported_transition"
        notes.append("Substantial physical evidence supports a candidate land-cover transition.")
    elif (
        src_score > CalibrationConfig.EVIDENCE_BAND_NONE_MAX
        and dest_score > CalibrationConfig.EVIDENCE_BAND_NONE_MAX
    ):
        state = "partial_transition"
        notes.append("Partial or emerging physical evidence supports a candidate transition.")
    else:
        state = "inconclusive"
        reason_codes.append(CalibrationReasonCodes.TRANSITION_INCONCLUSIVE)
        notes.append("Physical evidence is insufficient to substantiate a candidate land-cover transition.")

    return TransitionSupportAssessment(
        state=state,
        source_evidence_score=round(src_score, 4),
        destination_evidence_score=round(dest_score, 4),
        spatial_support=sp_support,
        temporal_support=temp_support,
        spatial_overlap_fraction=round(overlap_frac, 4),
        reason_codes=list(dict.fromkeys(reason_codes)),
        notes=notes,
    )


# ============================================================
# COMPONENT 8: OVERALL INTERPRETATION SUPPORT EVALUATION
# ============================================================

def evaluate_overall_interpretation_support(
    observation_reliability: ObservationReliabilityAssessment,
    semantic_evidence: SemanticEvidenceAssessment,
    spatial_assessment: SpatialAssessment,
    temporal_consistency: TemporalConsistencyAssessment,
    data_sufficiency: DataSufficiencyAssessment,
    conflicts: List[Dict[str, Any]],
) -> OverallInterpretationSupport:
    """
    Synthesizes independent lines of evidence into a grounded, deterministic support assessment.
    CRITICAL RULE: This is NOT a probability or mathematical average.
    It evaluates rule-based multi-source corroboration.
    """
    corroborating: List[str] = []
    non_corroborating: List[str] = []
    reason_codes: List[str] = []
    limitations: List[str] = []

    # 1. Evaluate Observation Reliability
    if observation_reliability.state == "high":
        corroborating.append(f"High-quality imagery ({round(observation_reliability.valid_fraction * 100, 1)}% valid pixels)")
    elif observation_reliability.state == "moderate":
        corroborating.append(f"Acceptable imagery quality ({round(observation_reliability.valid_fraction * 100, 1)}% valid pixels)")
    else:
        non_corroborating.append(f"Low observation reliability ({round(observation_reliability.valid_fraction * 100, 1)}% valid pixels)")
        limitations.append("Cloud or shadow contamination restricts spatial clarity.")

    # 2. Evaluate Semantic Evidence Strength
    if semantic_evidence.state in ["very_strong", "strong"]:
        corroborating.append(f"Strong physical spectral signal for {semantic_evidence.hypothesis} (score={semantic_evidence.score})")
    elif semantic_evidence.state == "moderate":
        corroborating.append(f"Meaningful physical spectral signal for {semantic_evidence.hypothesis} (score={semantic_evidence.score})")
    elif semantic_evidence.state == "weak":
        non_corroborating.append(f"Weak physical spectral signal (score={semantic_evidence.score})")
    else:
        non_corroborating.append("Measured spectral indices remain within the neutral noise deadband")

    # 3. Evaluate Spatial Coherence
    if spatial_assessment.state == "high":
        corroborating.append(f"High spatial coherence ({spatial_assessment.score}) across {spatial_assessment.region_count} contiguous candidate regions")
    elif spatial_assessment.state == "moderate":
        corroborating.append(f"Moderate spatial clustering ({spatial_assessment.region_count} candidate regions)")
    elif spatial_assessment.state == "low":
        if spatial_assessment.region_count == 0:
            non_corroborating.append("No contiguous candidate clusters identified")
        else:
            non_corroborating.append(f"Low geometric coherence ({spatial_assessment.score})")

    # 4. Evaluate Temporal Consistency
    if temporal_consistency.state == "high":
        corroborating.append(f"High multi-observation temporal persistence ({round((temporal_consistency.persistence_fraction or 0) * 100)}%)")
    elif temporal_consistency.state == "moderate":
        corroborating.append("Moderate multi-observation temporal consistency")
    elif temporal_consistency.state == "bi_temporal_only":
        limitations.append("Bi-temporal comparison cannot confirm multi-year trend persistence.")
    elif temporal_consistency.state == "limited":
        non_corroborating.append("Irregular or mixed trajectory across temporal observations")

    # 5. Evaluate Conflicts
    has_conflicts = len(conflicts) > 0

    # ============================================================
    # DETERMINISTIC MATRIX FOR INTERPRETATION SUPPORT STATE
    # ============================================================

    # Case A: Low observation quality gates final support
    if observation_reliability.state == "low":
        state: InterpretationSupportState = "insufficient_support"
        summary = "Available imagery quality was insufficient to make a reliable change assessment."
        reason_codes.append(CalibrationReasonCodes.INSUFFICIENT_INTERPRETATION_SUPPORT)
        reason_codes.append(CalibrationReasonCodes.DATA_LOW_QUALITY)

    # Case B: Conflicting physical evidence
    elif has_conflicts and any(c.get("type") == "index_conflict" for c in conflicts):
        state = "contradictory_support"
        summary = "Physical indicators materially disagree, producing contradictory support for the candidate hypothesis."
        reason_codes.append(CalibrationReasonCodes.CONTRADICTORY_INTERPRETATION_SUPPORT)
        reason_codes.append(CalibrationReasonCodes.CONFLICTING_INDICATORS)

    # Case C: No physical evidence (neutral signals) with high quality
    elif semantic_evidence.state == "none":
        state = "insufficient_support"
        summary = "No strong evidence of change was found. Observation reliability was high/adequate, but measured physical signals did not support change."
        reason_codes.append(CalibrationReasonCodes.INSUFFICIENT_INTERPRETATION_SUPPORT)
        reason_codes.append(CalibrationReasonCodes.NO_SEMANTIC_SUPPORT)

    # Case D: Strong physical evidence + adequate quality + no major conflict
    elif semantic_evidence.state in ["very_strong", "strong"]:
        if spatial_assessment.state in ["high", "moderate"] and temporal_consistency.state in ["high", "moderate", "bi_temporal_only"]:
            state = "strong_support"
            summary = "Multiple independent lines of physical and spatial evidence strongly corroborate the change interpretation."
            reason_codes.append(CalibrationReasonCodes.STRONG_INTERPRETATION_SUPPORT)
        else:
            state = "moderate_support"
            summary = "Strong physical spectral evidence exists, but spatial or temporal corroboration is limited."
            reason_codes.append(CalibrationReasonCodes.MODERATE_INTERPRETATION_SUPPORT)

    # Case E: Moderate physical evidence
    elif semantic_evidence.state == "moderate":
        if spatial_assessment.state in ["high", "moderate"]:
            state = "moderate_support"
            summary = "Moderate evidence supports the candidate phenomenon, corroborated by coherent spatial clustering."
            reason_codes.append(CalibrationReasonCodes.MODERATE_INTERPRETATION_SUPPORT)
        else:
            state = "weak_support"
            summary = "Moderate physical signals were detected, but lack strong spatial or temporal corroboration."
            reason_codes.append(CalibrationReasonCodes.WEAK_INTERPRETATION_SUPPORT)

    # Case F: Weak physical evidence
    else:  # weak
        state = "weak_support"
        summary = "Weak physical signals were detected; corroboration is limited across independent dimensions."
        reason_codes.append(CalibrationReasonCodes.WEAK_INTERPRETATION_SUPPORT)

    return OverallInterpretationSupport(
        state=state,
        summary=summary,
        corroborating_factors=corroborating,
        non_corroborating_factors=non_corroborating,
        reason_codes=list(dict.fromkeys(reason_codes)),
        limitations=limitations,
    )


# ============================================================
# COMPONENT 9: MASTER CALIBRATION PACKAGE BUILDER
# ============================================================

def build_calibration_package(
    candidate_package: Optional[Dict[str, Any]] = None,
    multi_index_evidence: Optional[Dict[str, Any]] = None,
    spatial_analysis: Optional[Dict[str, Any]] = None,
    temporal_analysis: Optional[Dict[str, Any]] = None,
    imagery_result: Optional[Dict[str, Any]] = None,
    execution_results: Optional[Dict[str, Any]] = None,
    temporal_observations: Optional[List[Any]] = None,
    target: str = "urban",
    task: str = "change_detection",
    temporal_mode: str = "bi_temporal",
) -> Dict[str, Any]:
    """
    Main entry point for Phase 8. Orchestrates all 5 separate signals into a unified,
    inspectable calibration dictionary.
    """
    # 1. Observation Reliability
    obs_reliability = evaluate_observation_reliability(
        temporal_observations=temporal_observations,
        imagery_result=imagery_result,
        execution_results=execution_results,
    )

    # 2. Semantic Evidence Strength
    sem_evidence = evaluate_semantic_evidence(
        candidate_package=candidate_package,
        multi_index_evidence=multi_index_evidence,
        target=target,
    )

    # 3. Spatial Assessment
    sp_assessment = evaluate_spatial_assessment(
        spatial_analysis=spatial_analysis,
    )

    # 4. Temporal Consistency
    temp_consistency = evaluate_temporal_consistency(
        temporal_analysis=temporal_analysis,
        temporal_observations=temporal_observations,
        target=target,
        temporal_mode=temporal_mode,
    )

    # 5. Conflict Detection
    conflicts, conflict_codes = detect_evidence_conflicts(
        multi_index_evidence=multi_index_evidence,
        candidate_package=candidate_package,
        temporal_consistency=temp_consistency,
        spatial_assessment=sp_assessment,
    )

    # 6. Data Sufficiency
    data_suff = evaluate_data_sufficiency(
        observation_reliability=obs_reliability,
        temporal_consistency=temp_consistency,
        task=task,
        temporal_mode=temporal_mode,
    )

    # 7. Transition Support (if transition target)
    trans_support = evaluate_transition_support(
        candidate_package=candidate_package,
        spatial_analysis=spatial_analysis,
        temporal_analysis=temporal_analysis,
        observation_reliability=obs_reliability,
    )

    # 8. Overall Interpretation Support
    interp_support = evaluate_overall_interpretation_support(
        observation_reliability=obs_reliability,
        semantic_evidence=sem_evidence,
        spatial_assessment=sp_assessment,
        temporal_consistency=temp_consistency,
        data_sufficiency=data_suff,
        conflicts=conflicts,
    )

    # Aggregate all distinct reason codes
    all_codes = list(dict.fromkeys(
        obs_reliability.reason_codes
        + sem_evidence.reason_codes
        + sp_assessment.reason_codes
        + temp_consistency.reason_codes
        + data_suff.reason_codes
        + conflict_codes
        + interp_support.reason_codes
        + (trans_support.reason_codes if trans_support else [])
    ))

    package = CalibrationPackage(
        observation_reliability=obs_reliability,
        semantic_evidence=sem_evidence,
        spatial_assessment=sp_assessment,
        temporal_consistency=temp_consistency,
        interpretation_support=interp_support,
        data_sufficiency=data_suff,
        transition_support=trans_support,
        conflicts=conflicts,
        reason_codes=all_codes,
    )

    return package.to_dict()
