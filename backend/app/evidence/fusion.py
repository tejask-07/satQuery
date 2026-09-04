"""
Phase 5B: Evidence Fusion & Semantic Change Candidate Classification Engine.

Interprets multiple physical evidence signals from Phase 5A and classifies
deterministic, inspectable semantic CHANGE CANDIDATES.

IMPORTANT PRINCIPLES:
- Phase 5A answered: "What physical evidence exists?"
- Phase 5B answers: "What does the combination of physical evidence suggest?"
- Vocabulary is strictly non-predictive:
  "strong_candidate", "candidate", "weak_candidate", "uncertain", "no_support", "unavailable".
- Never asserts probability, confidence accuracy, or confirmed land-cover class.
- All thresholds, weights, and contradiction penalties are centralized.
- Supports single-target, general-change (multi-target), and land-cover transition queries.
- Generates pixel-level candidate rasters and summary statistics without Phase 6 spatial processing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np


# ============================================================
# CENTRALIZED FUSION THRESHOLDS & CONFIGURATION
# ============================================================

class FusionThresholds:
    """
    Documented, centralized thresholds for Phase 5B evidence fusion.
    """
    # Candidate strength categorization
    STRONG_THRESHOLD: float = 0.60
    CANDIDATE_THRESHOLD: float = 0.35
    WEAK_THRESHOLD: float = 0.18

    # Ambiguity margin: if opposing hypotheses are too close, outcome is uncertain
    AMBIGUITY_MARGIN: float = 0.12

    # Contradiction penalty on semantic support when conflicting signals are detected
    CONTRADICTION_PENALTY: float = 0.50

    # Noise deadbands (matching 5A)
    INDEX_DEADBAND: float = 0.05
    SPECTRAL_DEADBAND: float = 0.02


# Controlled vocabulary for candidate state
CandidateState = Literal[
    "strong_candidate",
    "candidate",
    "weak_candidate",
    "uncertain",
    "no_support",
    "unavailable",
]


@dataclass
class SemanticCandidate:
    """
    Structured representation of a semantic change candidate finding.
    """
    target: str
    hypothesis: str
    state: CandidateState
    supporting_evidence: Dict[str, float]
    opposing_evidence: Dict[str, float]
    semantic_support: float
    reliability: float
    final_evidence_score: float
    reason_codes: List[str]
    statistics: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_candidate_state(
    score: float,
    is_conflicted: bool = False,
    is_ambiguous: bool = False,
    is_valid: bool = True,
) -> CandidateState:
    """
    Maps final evidence score and conflict flags to a controlled candidate state.
    """
    if not is_valid:
        return "unavailable"
    if is_conflicted or is_ambiguous:
        return "uncertain"
    if score >= FusionThresholds.STRONG_THRESHOLD:
        return "strong_candidate"
    if score >= FusionThresholds.CANDIDATE_THRESHOLD:
        return "candidate"
    if score >= FusionThresholds.WEAK_THRESHOLD:
        return "weak_candidate"
    return "no_support"


# ============================================================
# DOMAIN FUSION: URBAN
# ============================================================

def fuse_urban_evidence(
    evidence_5a: Dict[str, Any],
) -> SemanticCandidate:
    """
    Interprets urban evidence signals:
    - Primary Hypothesis: urban_expansion
    - Counter Hypothesis: urban_reduction
    """
    signals = evidence_5a.get("signals", {})
    ndbi_sig = signals.get("ndbi", {})
    ndvi_sig = signals.get("ndvi", {})
    spec_sig = signals.get("spectral", {})
    rel_info = evidence_5a.get("reliability", {})

    reliability_score = float(rel_info.get("score", 1.0))
    is_valid = bool(rel_info.get("valid", True))

    ndbi_dir = ndbi_sig.get("direction", "neutral")
    ndbi_str = float(ndbi_sig.get("normalized_strength", 0.0))
    ndvi_dir = ndvi_sig.get("direction", "neutral")
    ndvi_str = float(ndvi_sig.get("normalized_strength", 0.0))
    spec_dir = spec_sig.get("direction", "neutral")
    spec_str = float(spec_sig.get("normalized_strength", 0.0))

    supporting_evidence: Dict[str, float] = {}
    opposing_evidence: Dict[str, float] = {}
    reason_codes: List[str] = []

    # Expansion signals
    if ndbi_dir == "increase":
        supporting_evidence["ndbi_increase"] = ndbi_str
        reason_codes.append("NDBI_INCREASE")
    elif ndbi_dir == "decrease":
        opposing_evidence["ndbi_decrease"] = ndbi_str
        reason_codes.append("NDBI_DECREASE")

    if ndvi_dir == "decrease":
        supporting_evidence["ndvi_clearing"] = ndvi_str
        reason_codes.append("NDVI_DECREASE")
    elif ndvi_dir == "increase":
        opposing_evidence["ndvi_regreening"] = ndvi_str
        reason_codes.append("NDVI_INCREASE")

    if spec_dir == "increase":
        supporting_evidence["spectral_brightening"] = spec_str
        reason_codes.append("SPECTRAL_BRIGHTENING")
    elif spec_dir == "decrease":
        opposing_evidence["spectral_darkening"] = spec_str
        reason_codes.append("SPECTRAL_DARKENING")

    # Conflict check:
    # 1. NDBI increases AND NDVI increases (building while vegetation expands)
    # 2. NDBI decreases AND NDVI decreases (both drop)
    is_conflicted = False
    if ndbi_dir == "increase" and ndvi_dir == "increase" and ndbi_str > 0.0 and ndvi_str > 0.0:
        is_conflicted = True
        reason_codes.append("CONFLICTING_NDBI_NDVI_INCREASE")
    elif ndbi_dir == "decrease" and ndvi_dir == "decrease" and ndbi_str > 0.0 and ndvi_str > 0.0:
        is_conflicted = True
        reason_codes.append("CONFLICTING_NDBI_NDVI_DECREASE")

    expansion_score = float(evidence_5a.get("urban_expansion_support", 0.0))
    counter_data = evidence_5a.get("counter_hypothesis", {})
    reduction_score = float(counter_data.get("urban_reduction_support", 0.0))

    # Ambiguity check: expansion vs reduction are too close
    is_ambiguous = False
    if (
        expansion_score >= FusionThresholds.WEAK_THRESHOLD
        and reduction_score >= FusionThresholds.WEAK_THRESHOLD
        and abs(expansion_score - reduction_score) <= FusionThresholds.AMBIGUITY_MARGIN
    ):
        is_ambiguous = True
        reason_codes.append("AMBIGUOUS_EXPANSION_REDUCTION")

    # Primary hypothesis determination
    if reduction_score > expansion_score and reduction_score >= FusionThresholds.WEAK_THRESHOLD and not is_conflicted and not is_ambiguous:
        primary_hyp = "urban_reduction"
        final_score = reduction_score
        semantic_support = float(counter_data.get("semantic_support", 0.0))
        reason_codes.append("URBAN_REDUCTION_FAVORED")
    else:
        primary_hyp = "urban_expansion"
        final_score = expansion_score
        semantic_support = float(evidence_5a.get("semantic_support", 0.0))
        if final_score > 0.0:
            reason_codes.append("URBAN_EXPANSION_FAVORED")

    state = classify_candidate_state(
        score=final_score,
        is_conflicted=is_conflicted,
        is_ambiguous=is_ambiguous,
        is_valid=is_valid,
    )

    if reliability_score >= 0.85:
        reason_codes.append("HIGH_RELIABILITY")
    elif not is_valid:
        reason_codes.append("LOW_RELIABILITY_GATED")

    return SemanticCandidate(
        target="urban",
        hypothesis=primary_hyp,
        state=state,
        supporting_evidence=supporting_evidence,
        opposing_evidence=opposing_evidence,
        semantic_support=round(semantic_support, 4),
        reliability=round(reliability_score, 4),
        final_evidence_score=round(final_score, 4),
        reason_codes=reason_codes,
        details={
            "urban_expansion_support": expansion_score,
            "urban_reduction_support": reduction_score,
            "is_conflicted": is_conflicted,
            "is_ambiguous": is_ambiguous,
        },
    )


# ============================================================
# DOMAIN FUSION: VEGETATION
# ============================================================

def fuse_vegetation_evidence(
    evidence_5a: Dict[str, Any],
) -> SemanticCandidate:
    """
    Interprets vegetation evidence signals:
    - Primary Hypothesis: vegetation_loss
    - Counter Hypothesis: vegetation_gain
    """
    signals = evidence_5a.get("signals", {})
    ndvi_sig = signals.get("ndvi", {})
    ndbi_sig = signals.get("ndbi", {})
    spec_sig = signals.get("spectral", {})
    rel_info = evidence_5a.get("reliability", {})

    reliability_score = float(rel_info.get("score", 1.0))
    is_valid = bool(rel_info.get("valid", True))

    ndvi_dir = ndvi_sig.get("direction", "neutral")
    ndvi_str = float(ndvi_sig.get("normalized_strength", 0.0))
    ndbi_dir = ndbi_sig.get("direction", "neutral")
    ndbi_str = float(ndbi_sig.get("normalized_strength", 0.0))
    spec_dir = spec_sig.get("direction", "neutral")
    spec_str = float(spec_sig.get("normalized_strength", 0.0))

    supporting_evidence: Dict[str, float] = {}
    opposing_evidence: Dict[str, float] = {}
    reason_codes: List[str] = []

    if ndvi_dir == "decrease":
        supporting_evidence["ndvi_decrease"] = ndvi_str
        reason_codes.append("NDVI_DECREASE")
    elif ndvi_dir == "increase":
        opposing_evidence["ndvi_increase"] = ndvi_str
        reason_codes.append("NDVI_INCREASE")

    if ndbi_dir == "increase":
        supporting_evidence["soil_exposure"] = ndbi_str
        reason_codes.append("SOIL_EXPOSURE")
    elif ndbi_dir == "decrease":
        opposing_evidence["soil_reduction"] = ndbi_str

    if spec_dir == "increase":
        supporting_evidence["canopy_loss_spectral"] = spec_str
        reason_codes.append("CANOPY_LOSS_SPECTRAL")
    elif spec_dir == "decrease":
        opposing_evidence["canopy_growth_spectral"] = spec_str
        reason_codes.append("CANOPY_GROWTH_SPECTRAL")

    # Conflict check: NDVI direction directly contradicts spectral canopy response
    is_conflicted = False
    if ndvi_dir == "decrease" and spec_dir == "decrease" and ndvi_str > 0.0 and spec_str > 0.0:
        is_conflicted = True
        reason_codes.append("CONFLICTING_NDVI_AND_CANOPY_GROWTH")
    elif ndvi_dir == "increase" and spec_dir == "increase" and ndvi_str > 0.0 and spec_str > 0.0:
        is_conflicted = True
        reason_codes.append("CONFLICTING_NDVI_AND_CANOPY_LOSS")

    loss_score = float(evidence_5a.get("vegetation_loss_support", 0.0))
    counter_data = evidence_5a.get("counter_hypothesis", {})
    gain_score = float(counter_data.get("vegetation_gain_support", 0.0))

    # Ambiguity check
    is_ambiguous = False
    if (
        loss_score >= FusionThresholds.WEAK_THRESHOLD
        and gain_score >= FusionThresholds.WEAK_THRESHOLD
        and abs(loss_score - gain_score) <= FusionThresholds.AMBIGUITY_MARGIN
    ):
        is_ambiguous = True
        reason_codes.append("AMBIGUOUS_LOSS_GAIN")

    # Primary hypothesis determination
    if gain_score > loss_score and gain_score >= FusionThresholds.WEAK_THRESHOLD and not is_conflicted and not is_ambiguous:
        primary_hyp = "vegetation_gain"
        final_score = gain_score
        semantic_support = float(counter_data.get("semantic_support", 0.0))
        reason_codes.append("VEGETATION_GAIN_FAVORED")
    else:
        primary_hyp = "vegetation_loss"
        final_score = loss_score
        semantic_support = float(evidence_5a.get("semantic_support", 0.0))
        if final_score > 0.0:
            reason_codes.append("VEGETATION_LOSS_FAVORED")

    state = classify_candidate_state(
        score=final_score,
        is_conflicted=is_conflicted,
        is_ambiguous=is_ambiguous,
        is_valid=is_valid,
    )

    if reliability_score >= 0.85:
        reason_codes.append("HIGH_RELIABILITY")
    elif not is_valid:
        reason_codes.append("LOW_RELIABILITY_GATED")

    return SemanticCandidate(
        target="vegetation",
        hypothesis=primary_hyp,
        state=state,
        supporting_evidence=supporting_evidence,
        opposing_evidence=opposing_evidence,
        semantic_support=round(semantic_support, 4),
        reliability=round(reliability_score, 4),
        final_evidence_score=round(final_score, 4),
        reason_codes=reason_codes,
        details={
            "vegetation_loss_support": loss_score,
            "vegetation_gain_support": gain_score,
            "is_conflicted": is_conflicted,
            "is_ambiguous": is_ambiguous,
        },
    )


# ============================================================
# DOMAIN FUSION: WATER
# ============================================================

def fuse_water_evidence(
    evidence_5a: Dict[str, Any],
) -> SemanticCandidate:
    """
    Interprets water evidence signals:
    - Primary Hypothesis: water_loss
    - Counter Hypothesis: water_gain
    """
    signals = evidence_5a.get("signals", {})
    ndwi_sig = signals.get("ndwi", {})
    spec_sig = signals.get("spectral", {})
    rel_info = evidence_5a.get("reliability", {})

    reliability_score = float(rel_info.get("score", 1.0))
    is_valid = bool(rel_info.get("valid", True))

    ndwi_dir = ndwi_sig.get("direction", "neutral")
    ndwi_str = float(ndwi_sig.get("normalized_strength", 0.0))
    spec_dir = spec_sig.get("direction", "neutral")
    spec_str = float(spec_sig.get("normalized_strength", 0.0))

    supporting_evidence: Dict[str, float] = {}
    opposing_evidence: Dict[str, float] = {}
    reason_codes: List[str] = []

    if ndwi_dir == "decrease":
        supporting_evidence["ndwi_decrease"] = ndwi_str
        reason_codes.append("NDWI_DECREASE")
    elif ndwi_dir == "increase":
        opposing_evidence["ndwi_increase"] = ndwi_str
        reason_codes.append("NDWI_INCREASE")

    if spec_dir == "increase":
        supporting_evidence["soil_drying_spectral"] = spec_str
        reason_codes.append("SOIL_DRYING_SPECTRAL")
    elif spec_dir == "decrease":
        opposing_evidence["water_absorption_spectral"] = spec_str
        reason_codes.append("WATER_ABSORPTION_SPECTRAL")

    # Conflict check: NDWI direction contradicts spectral absorption response
    is_conflicted = False
    if ndwi_dir == "decrease" and spec_dir == "decrease" and ndwi_str > 0.0 and spec_str > 0.0:
        is_conflicted = True
        reason_codes.append("CONFLICTING_NDWI_AND_WATER_ABSORPTION")
    elif ndwi_dir == "increase" and spec_dir == "increase" and ndwi_str > 0.0 and spec_str > 0.0:
        is_conflicted = True
        reason_codes.append("CONFLICTING_NDWI_AND_SOIL_DRYING")

    loss_score = float(evidence_5a.get("water_loss_support", 0.0))
    counter_data = evidence_5a.get("counter_hypothesis", {})
    gain_score = float(counter_data.get("water_gain_support", 0.0))

    # Ambiguity check
    is_ambiguous = False
    if (
        loss_score >= FusionThresholds.WEAK_THRESHOLD
        and gain_score >= FusionThresholds.WEAK_THRESHOLD
        and abs(loss_score - gain_score) <= FusionThresholds.AMBIGUITY_MARGIN
    ):
        is_ambiguous = True
        reason_codes.append("AMBIGUOUS_WATER_LOSS_GAIN")

    # Primary hypothesis determination
    if gain_score > loss_score and gain_score >= FusionThresholds.WEAK_THRESHOLD and not is_conflicted and not is_ambiguous:
        primary_hyp = "water_gain"
        final_score = gain_score
        semantic_support = float(counter_data.get("semantic_support", 0.0))
        reason_codes.append("WATER_GAIN_FAVORED")
    else:
        primary_hyp = "water_loss"
        final_score = loss_score
        semantic_support = float(evidence_5a.get("semantic_support", 0.0))
        if final_score > 0.0:
            reason_codes.append("WATER_LOSS_FAVORED")

    state = classify_candidate_state(
        score=final_score,
        is_conflicted=is_conflicted,
        is_ambiguous=is_ambiguous,
        is_valid=is_valid,
    )

    if reliability_score >= 0.85:
        reason_codes.append("HIGH_RELIABILITY")
    elif not is_valid:
        reason_codes.append("LOW_RELIABILITY_GATED")

    return SemanticCandidate(
        target="water",
        hypothesis=primary_hyp,
        state=state,
        supporting_evidence=supporting_evidence,
        opposing_evidence=opposing_evidence,
        semantic_support=round(semantic_support, 4),
        reliability=round(reliability_score, 4),
        final_evidence_score=round(final_score, 4),
        reason_codes=reason_codes,
        details={
            "water_loss_support": loss_score,
            "water_gain_support": gain_score,
            "is_conflicted": is_conflicted,
            "is_ambiguous": is_ambiguous,
        },
    )


# ============================================================
# TRANSITION FUSION (e.g. VEGETATION TO URBAN)
# ============================================================

def fuse_transition_evidence(
    source_candidate: SemanticCandidate,
    destination_candidate: SemanticCandidate,
) -> SemanticCandidate:
    """
    Fuses two domain candidates to evaluate a land-cover transition candidate.
    Example: Vegetation Loss + Urban Expansion -> vegetation_to_urban_candidate.
    """
    source_hyp = source_candidate.hypothesis
    dest_hyp = destination_candidate.hypothesis
    hyp_name = f"{source_candidate.target}_to_{destination_candidate.target}_transition"

    is_valid = source_candidate.state != "unavailable" and destination_candidate.state != "unavailable"

    # Both source reduction (e.g. veg loss) and destination increase (e.g. urban expansion) must have positive evidence
    src_score = source_candidate.final_evidence_score
    dest_score = destination_candidate.final_evidence_score

    # Geometric mean / harmonic balance ensures both endpoints must support the transition
    if src_score <= 0.0 or dest_score <= 0.0 or not is_valid:
        transition_score = 0.0
        semantic_support = 0.0
    else:
        semantic_support = (source_candidate.semantic_support + destination_candidate.semantic_support) / 2.0
        rel = min(source_candidate.reliability, destination_candidate.reliability)
        transition_score = round(semantic_support * rel, 4)

    is_conflicted = (source_candidate.state == "uncertain") or (destination_candidate.state == "uncertain")

    reason_codes: List[str] = []
    if src_score >= FusionThresholds.WEAK_THRESHOLD:
        reason_codes.append(f"SOURCE_{source_hyp.upper()}_DETECTED")
    if dest_score >= FusionThresholds.WEAK_THRESHOLD:
        reason_codes.append(f"DESTINATION_{dest_hyp.upper()}_DETECTED")
    if is_conflicted:
        reason_codes.append("ENDPOINT_EVIDENCE_UNCERTAIN")

    state = classify_candidate_state(
        score=transition_score,
        is_conflicted=is_conflicted,
        is_ambiguous=False,
        is_valid=is_valid,
    )

    return SemanticCandidate(
        target="transition",
        hypothesis=hyp_name,
        state=state,
        supporting_evidence={
            f"source_{source_candidate.target}_evidence": src_score,
            f"destination_{destination_candidate.target}_evidence": dest_score,
        },
        opposing_evidence={},
        semantic_support=round(semantic_support, 4),
        reliability=round(min(source_candidate.reliability, destination_candidate.reliability), 4),
        final_evidence_score=transition_score,
        reason_codes=reason_codes,
        details={
            "source_candidate": source_candidate.to_dict(),
            "destination_candidate": destination_candidate.to_dict(),
        },
    )


# ============================================================
# PIXEL-LEVEL CANDIDATE RASTER GENERATION & STATISTICS
# ============================================================

def generate_pixel_candidate_raster(
    target: str,
    execution_results: Dict[str, Any],
    imagery_result: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Generates a deterministic integer candidate GeoTIFF raster:
      0: Valid No Support / Invalid Masked
      1: Primary Candidate (e.g. Expansion / Loss)
      2: Counter Candidate (e.g. Reduction / Gain)
      3: Uncertain / Conflicting

    Returns:
      (raster_path, summary_statistics)
    """
    try:
        import rasterio
        from app.evidence.multi_index import EvidenceThresholds

        # Determine reference delta raster and joint mask
        t_ndvi = execution_results.get("calculate_temporal_ndvi", {})
        t_ndwi = execution_results.get("calculate_temporal_ndwi", {})
        t_ndbi = execution_results.get("calculate_temporal_ndbi", {})

        ref_path = None
        for res in [t_ndvi, t_ndbi, t_ndwi]:
            if res.get("difference_raster"):
                ref_path = res["difference_raster"]
                break

        if not ref_path or not Path(ref_path).exists():
            if imagery_result and imagery_result.get("images"):
                bands = imagery_result["images"][0].get("bands", {})
                for b_key in ["red", "mask", "nir", "green", "blue", "swir"]:
                    candidate_p = bands.get(b_key)
                    if candidate_p and Path(candidate_p).exists():
                        ref_path = candidate_p
                        break

        if not ref_path or not Path(ref_path).exists():
            return None, {}

        # Open reference raster to get geometry
        with rasterio.open(ref_path) as ref_ds:
            profile = ref_ds.profile.copy()
            shape = (ref_ds.height, ref_ds.width)

        # Read joint mask if available
        j_mask = np.ones(shape, dtype=bool)
        if imagery_result and len(imagery_result.get("images", [])) >= 2:
            m_b_path = imagery_result["images"][0].get("bands", {}).get("mask")
            m_a_path = imagery_result["images"][1].get("bands", {}).get("mask")
            if m_b_path and m_a_path and Path(m_b_path).exists() and Path(m_a_path).exists():
                with rasterio.open(m_b_path) as mb, rasterio.open(m_a_path) as ma:
                    j_mask = mb.read(1).astype(bool) & ma.read(1).astype(bool)

        target_clean = (target or "").lower().strip()
        candidate_map = np.zeros(shape, dtype=np.uint8)

        # Read index deltas
        ndvi_arr = None
        ndbi_arr = None
        ndwi_arr = None

        if t_ndvi.get("difference_raster") and Path(t_ndvi["difference_raster"]).exists():
            with rasterio.open(t_ndvi["difference_raster"]) as ds:
                ndvi_arr = ds.read(1)
        if t_ndbi.get("difference_raster") and Path(t_ndbi["difference_raster"]).exists():
            with rasterio.open(t_ndbi["difference_raster"]) as ds:
                ndbi_arr = ds.read(1)
        if t_ndwi.get("difference_raster") and Path(t_ndwi["difference_raster"]).exists():
            with rasterio.open(t_ndwi["difference_raster"]) as ds:
                ndwi_arr = ds.read(1)

        # Fallback to in-memory deltas or detect_change all_changes
        dc_all = execution_results.get("detect_change", {}).get("all_changes", {})
        if ndvi_arr is None:
            if "ndvi" in dc_all and "change_map" in dc_all["ndvi"]:
                ndvi_arr = np.array(dc_all["ndvi"]["change_map"], dtype=np.float32)
            elif t_ndvi.get("ndvi_after") is not None and t_ndvi.get("ndvi_before") is not None:
                ndvi_arr = np.array(t_ndvi["ndvi_after"], dtype=np.float32) - np.array(t_ndvi["ndvi_before"], dtype=np.float32)

        if ndbi_arr is None:
            if "ndbi" in dc_all and "change_map" in dc_all["ndbi"]:
                ndbi_arr = np.array(dc_all["ndbi"]["change_map"], dtype=np.float32)
            elif t_ndbi.get("ndbi_after") is not None and t_ndbi.get("ndbi_before") is not None:
                ndbi_arr = np.array(t_ndbi["ndbi_after"], dtype=np.float32) - np.array(t_ndbi["ndbi_before"], dtype=np.float32)

        if ndwi_arr is None:
            if "ndwi" in dc_all and "change_map" in dc_all["ndwi"]:
                ndwi_arr = np.array(dc_all["ndwi"]["change_map"], dtype=np.float32)
            elif t_ndwi.get("ndwi_after") is not None and t_ndwi.get("ndwi_before") is not None:
                ndwi_arr = np.array(t_ndwi["ndwi_after"], dtype=np.float32) - np.array(t_ndwi["ndwi_before"], dtype=np.float32)

        deadband = EvidenceThresholds.INDEX_DEADBAND

        if target_clean == "urban":
            if ndbi_arr is not None and ndvi_arr is not None:
                # Valid pixels
                valid = j_mask & np.isfinite(ndbi_arr) & np.isfinite(ndvi_arr)
                ndbi_inc = valid & (ndbi_arr > deadband)
                ndbi_dec = valid & (ndbi_arr < -deadband)
                ndvi_inc = valid & (ndvi_arr > deadband)
                ndvi_dec = valid & (ndvi_arr < -deadband)

                # Expansion: NDBI inc & NDVI dec (or neutral)
                expansion = ndbi_inc & ~ndvi_inc
                # Reduction: NDBI dec & NDVI inc (or neutral)
                reduction = ndbi_dec & ~ndvi_dec
                # Uncertain: conflicting directions
                uncertain = (ndbi_inc & ndvi_inc) | (ndbi_dec & ndvi_dec)

                candidate_map[expansion] = 1
                candidate_map[reduction] = 2
                candidate_map[uncertain] = 3

        elif target_clean == "water":
            if ndwi_arr is not None:
                valid = j_mask & np.isfinite(ndwi_arr)
                candidate_map[valid & (ndwi_arr < -deadband)] = 1  # water loss
                candidate_map[valid & (ndwi_arr > deadband)] = 2   # water gain

        else:
            # Vegetation
            if ndvi_arr is not None:
                valid = j_mask & np.isfinite(ndvi_arr)
                candidate_map[valid & (ndvi_arr < -deadband)] = 1  # vegetation loss
                candidate_map[valid & (ndvi_arr > deadband)] = 2   # vegetation gain

        # Mask invalid pixels back to 0
        candidate_map[~j_mask] = 0

        # Calculate statistics
        total_valid = int(np.sum(j_mask))
        cand_1_count = int(np.sum(candidate_map == 1))
        cand_2_count = int(np.sum(candidate_map == 2))
        cand_3_count = int(np.sum(candidate_map == 3))
        total_candidate = cand_1_count + cand_2_count

        cand_pct = float(round((cand_1_count / max(1, total_valid)) * 100.0, 2))
        counter_pct = float(round((cand_2_count / max(1, total_valid)) * 100.0, 2))
        uncertain_pct = float(round((cand_3_count / max(1, total_valid)) * 100.0, 2))

        stats = {
            "total_valid_pixels": total_valid,
            "primary_candidate_pixels": cand_1_count,
            "primary_candidate_percentage": cand_pct,
            "counter_candidate_pixels": cand_2_count,
            "counter_candidate_percentage": counter_pct,
            "uncertain_pixels": cand_3_count,
            "uncertain_percentage": uncertain_pct,
            "total_candidate_pixels": total_candidate,
        }

        # Save raster GeoTIFF
        out_dir = Path(output_dir) if output_dir else Path(ref_path).parent
        out_path = out_dir / f"candidate_{target_clean}.tif"
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)

        with rasterio.open(out_path, "w", **profile) as out_ds:
            out_ds.write(candidate_map, 1)

        return str(out_path), stats

    except Exception as exc:
        print(f"[FUSION ENGINE WARNING] Candidate raster generation: {exc}")
        return None, {}


# ============================================================
# MAIN FUSION ORCHESTRATOR
# ============================================================

def fuse_evidence_and_classify_candidates(
    target: Optional[str],
    task: Optional[str],
    multi_index_evidence: Dict[str, Any],
    execution_results: Dict[str, Any],
    imagery_result: Optional[Dict[str, Any]] = None,
    change_result: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for Phase 5B Evidence Fusion.
    Consumes Phase 5A evidence packages and produces classified semantic candidates.
    """
    target_clean = (target or "").lower().strip()
    task_clean = (task or "").lower().strip()

    from app.evidence.multi_index import (
        calculate_urban_evidence,
        calculate_vegetation_evidence,
        calculate_water_evidence,
    )

    # Extract common metrics for secondary domain evaluations
    t_ndvi = execution_results.get("calculate_temporal_ndvi", {})
    t_ndwi = execution_results.get("calculate_temporal_ndwi", {})
    t_ndbi = execution_results.get("calculate_temporal_ndbi", {})

    ndvi_delta = t_ndvi.get("mean_ndvi_change")
    ndwi_delta = t_ndwi.get("mean_ndwi_change")
    ndbi_delta = t_ndbi.get("mean_ndbi_change")

    spectral_shifts = multi_index_evidence.get("metadata", {}).get("spectral_shifts", {})
    quality_fraction = multi_index_evidence.get("metadata", {}).get("quality_fraction")

    candidates_list: List[Dict[str, Any]] = []

    # Check query type:
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
        # Evaluate vegetation and urban domains to fuse transition
        ev_veg = calculate_vegetation_evidence(
            ndvi_delta=ndvi_delta,
            ndbi_delta=ndbi_delta,
            spectral_shifts=spectral_shifts,
            quality_fraction=quality_fraction,
        )
        ev_urb = calculate_urban_evidence(
            ndbi_delta=ndbi_delta,
            ndvi_delta=ndvi_delta,
            spectral_shifts=spectral_shifts,
            quality_fraction=quality_fraction,
        )

        cand_veg = fuse_vegetation_evidence(ev_veg)
        cand_urb = fuse_urban_evidence(ev_urb)
        trans_cand = fuse_transition_evidence(cand_veg, cand_urb)

        candidates_list.append(trans_cand.to_dict())
        candidates_list.append(cand_urb.to_dict())
        candidates_list.append(cand_veg.to_dict())
        primary_candidate = trans_cand

    elif is_general_change:
        # General Change Query: evaluate urban, vegetation, and water independently
        ev_urb = calculate_urban_evidence(
            ndbi_delta=ndbi_delta,
            ndvi_delta=ndvi_delta,
            spectral_shifts=spectral_shifts,
            quality_fraction=quality_fraction,
        )
        ev_veg = calculate_vegetation_evidence(
            ndvi_delta=ndvi_delta,
            ndbi_delta=ndbi_delta,
            spectral_shifts=spectral_shifts,
            quality_fraction=quality_fraction,
        )
        ev_wat = calculate_water_evidence(
            ndwi_delta=ndwi_delta,
            spectral_shifts=spectral_shifts,
            quality_fraction=quality_fraction,
        )

        cand_urb = fuse_urban_evidence(ev_urb)
        cand_veg = fuse_vegetation_evidence(ev_veg)
        cand_wat = fuse_water_evidence(ev_wat)

        candidates_list.append(cand_urb.to_dict())
        candidates_list.append(cand_veg.to_dict())
        candidates_list.append(cand_wat.to_dict())

        # Select highest-scoring active candidate as primary, but keep all
        primary_candidate = max(
            [cand_urb, cand_veg, cand_wat],
            key=lambda c: c.final_evidence_score,
        )

    elif target_clean == "urban" or "urban" in task_clean:
        cand_urb = fuse_urban_evidence(multi_index_evidence)
        candidates_list.append(cand_urb.to_dict())
        primary_candidate = cand_urb

    elif target_clean == "water" or "water" in task_clean:
        cand_wat = fuse_water_evidence(multi_index_evidence)
        candidates_list.append(cand_wat.to_dict())
        primary_candidate = cand_wat

    else:
        # Vegetation
        cand_veg = fuse_vegetation_evidence(multi_index_evidence)
        candidates_list.append(cand_veg.to_dict())
        primary_candidate = cand_veg

    # Generate pixel-level candidate raster & statistics
    raster_target = primary_candidate.target if primary_candidate.target != "transition" else "urban"
    cand_raster_path, cand_stats = generate_pixel_candidate_raster(
        target=raster_target,
        execution_results=execution_results,
        imagery_result=imagery_result,
        output_dir=output_dir,
    )

    primary_dict = primary_candidate.to_dict()
    if cand_stats:
        primary_dict["statistics"] = cand_stats

    return {
        "engine_version": "5B.1",
        "primary_candidate": primary_dict,
        "candidates": candidates_list,
        "candidate_raster": cand_raster_path,
        "statistics": cand_stats,
        "thresholds": {
            "strong_threshold": FusionThresholds.STRONG_THRESHOLD,
            "candidate_threshold": FusionThresholds.CANDIDATE_THRESHOLD,
            "weak_threshold": FusionThresholds.WEAK_THRESHOLD,
            "ambiguity_margin": FusionThresholds.AMBIGUITY_MARGIN,
            "contradiction_penalty": FusionThresholds.CONTRADICTION_PENALTY,
        },
    }
