"""
Phase 5A: Multi-Index Evidence Calculation Engine.

Derives structured, inspectable evidence signals from:
1. Spectral indices (NDVI, NDWI, NDBI) and their temporal deltas
2. Spectral reflectance bands (Red, Green, NIR, SWIR)
3. Joint temporal observation quality masks

IMPORTANT SCIENTIFIC PRINCIPLES:
- An index change is an EVIDENCE SIGNAL, NOT a confirmed land-cover class.
- All evidence scores are bounded in [0.0, 1.0].
- Evidence scores represent physical signal support intensity, NOT probabilities or percentages.
- Quality masks are strictly respected: invalid pixels yield "unavailable" evidence.
- Directional alignment and counter-hypotheses are explicitly inspectable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np


# ============================================================
# CENTRALIZED THRESHOLDS & CONFIGURATION (NO MAGIC NUMBERS)
# ============================================================

class EvidenceThresholds:
    """
    Centralized, documented thresholds for scientific evidence evaluation.
    
    All index normalization maps:
      |delta| <= INDEX_DEADBAND  -> strength = 0.0 (noise / neutral)
      |delta| >= INDEX_STRONG    -> strength = 1.0 (saturated strong evidence)
      intermediate               -> piecewise linear interpolation in (0.0, 1.0)
    """

    # Spectral indices (|delta| in [-1.0, 1.0])
    INDEX_DEADBAND: float = 0.05
    INDEX_STRONG: float = 0.30

    # Surface reflectance bands (delta in [-1.0, 1.0], typical BOA reflectance 0.0 - 0.6)
    SPECTRAL_DEADBAND: float = 0.02
    SPECTRAL_STRONG: float = 0.15

    # Observation quality thresholds (fraction of valid pixels 0.0 - 1.0)
    QUALITY_HIGH: float = 0.85
    QUALITY_MIN_ACCEPTABLE: float = 0.50


class EvidenceWeights:
    """
    Centralized, documented semantic evidence weights.
    Weights sum to 1.0 for each hypothesis and represent purely physical signals.
    Quality is NOT an additive evidence weight.
    """

    # Urban Expansion (sum = 1.0)
    URBAN_EXPANSION_NDBI: float = 0.50
    URBAN_EXPANSION_NDVI: float = 0.35
    URBAN_EXPANSION_SPECTRAL: float = 0.15

    # Urban Reduction (sum = 1.0)
    URBAN_REDUCTION_NDBI: float = 0.50
    URBAN_REDUCTION_NDVI: float = 0.35
    URBAN_REDUCTION_SPECTRAL: float = 0.15

    # Vegetation Loss (sum = 1.0)
    VEGETATION_LOSS_NDVI: float = 0.60
    VEGETATION_LOSS_SPECTRAL: float = 0.25
    VEGETATION_LOSS_NDBI: float = 0.15

    # Vegetation Gain (sum = 1.0)
    VEGETATION_GAIN_NDVI: float = 0.70
    VEGETATION_GAIN_SPECTRAL: float = 0.30

    # Water Loss (sum = 1.0)
    WATER_LOSS_NDWI: float = 0.70
    WATER_LOSS_SPECTRAL: float = 0.30

    # Water Gain (sum = 1.0)
    WATER_GAIN_NDWI: float = 0.70
    WATER_GAIN_SPECTRAL: float = 0.30



# ============================================================
# DATA STRUCTURES
# ============================================================

SignalDirection = Literal["increase", "decrease", "neutral", "none"]
SupportState = Literal["positive", "negative", "neutral", "uncertain", "unavailable"]


@dataclass
class EvidenceSignal:
    """
    Structured representation of an individual scientific evidence signal.
    """
    signal_name: str
    display_name: str
    direction: SignalDirection
    raw_magnitude: float
    normalized_strength: float
    support_state: SupportState
    support_score: float
    valid: bool
    interpretation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def normalize_signal_magnitude(
    raw_delta: Optional[float],
    deadband: float = EvidenceThresholds.INDEX_DEADBAND,
    strong: float = EvidenceThresholds.INDEX_STRONG,
) -> Tuple[SignalDirection, float, float]:
    """
    Map a raw numeric delta to (direction, magnitude, normalized_strength).
    
    Returns
    -------
    direction : SignalDirection ("increase", "decrease", "neutral", or "none")
    magnitude : float (absolute value of delta)
    normalized_strength : float in [0.0, 1.0]
    """
    if raw_delta is None or not np.isfinite(raw_delta):
        return "none", 0.0, 0.0

    delta = float(raw_delta)
    magnitude = abs(delta)

    if magnitude <= deadband:
        direction: SignalDirection = "neutral"
        strength = 0.0
    elif delta > 0:
        direction = "increase"
        strength = (magnitude - deadband) / max(1e-6, strong - deadband)
    else:
        direction = "decrease"
        strength = (magnitude - deadband) / max(1e-6, strong - deadband)

    normalized_strength = float(np.clip(strength, 0.0, 1.0))
    return direction, magnitude, normalized_strength


def evaluate_directional_support(
    direction: SignalDirection,
    strength: float,
    expected_direction: SignalDirection,
    valid: bool = True,
) -> Tuple[SupportState, float]:
    """
    Evaluate directional support for a specific hypothesis.

    Parameters
    ----------
    direction : Observed signal direction ("increase", "decrease", "neutral", "none").
    strength : Normalized magnitude [0.0, 1.0].
    expected_direction : Direction that supports the hypothesis.
    valid : Whether the underlying data is valid.

    Returns
    -------
    support_state : "positive", "negative", "neutral", "uncertain", or "unavailable".
    support_score : Bounded float [0.0, 1.0].
    """
    if not valid or direction == "none":
        return "unavailable", 0.0

    if direction == "neutral":
        return "neutral", 0.0

    if direction == expected_direction:
        return "positive", strength

    # Signal points in opposite direction -> opposes the hypothesis
    return "negative", 0.0


# ============================================================
# TARGET-SPECIFIC EVIDENCE CALCULATORS
# ============================================================

def calculate_urban_evidence(
    ndbi_delta: Optional[float],
    ndvi_delta: Optional[float],
    spectral_shifts: Optional[Dict[str, float]] = None,
    quality_fraction: Optional[float] = None,
    joint_valid_mask: Optional[np.ndarray] = None,
    ndbi_before: Optional[float] = None,
    ndbi_after: Optional[float] = None,
    ndvi_before: Optional[float] = None,
    ndvi_after: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate structured evidence for urban/built-up surface change.

    Hypothesis: Potential Urban Expansion
      - NDBI increase supports urban expansion (+).
      - NDVI decrease supports urban expansion (+) (vegetation clearing).
      - SWIR and Red brightening corroborate impervious/cleared surfaces.
      - Quality score weights reliability.

    Counter-Hypothesis: Potential Urban Reduction
      - NDBI decrease supports urban reduction (+).
      - NDVI increase supports urban reduction (+) (re-greening).
    """
    is_valid = quality_fraction is None or quality_fraction >= EvidenceThresholds.QUALITY_MIN_ACCEPTABLE
    spectral_shifts = spectral_shifts or {}

    # 1. NDBI Signal
    ndbi_dir, ndbi_mag, ndbi_str = normalize_signal_magnitude(ndbi_delta)
    ndbi_state_exp, ndbi_score_exp = evaluate_directional_support(ndbi_dir, ndbi_str, expected_direction="increase", valid=is_valid and ndbi_delta is not None)
    ndbi_state_red, ndbi_score_red = evaluate_directional_support(ndbi_dir, ndbi_str, expected_direction="decrease", valid=is_valid and ndbi_delta is not None)

    ndbi_interp = (
        f"NDBI shifted by {ndbi_delta:+.4f} ({ndbi_dir}). "
        + ("Supports urban expansion." if ndbi_dir == "increase" else
           "Opposes urban expansion (supports reduction)." if ndbi_dir == "decrease" else
           "Within neutral deadband (|delta| <= 0.05).")
        if ndbi_delta is not None else "NDBI data unavailable."
    )

    ndbi_signal = EvidenceSignal(
        signal_name="ndbi",
        display_name="Normalized Difference Built-up Index (NDBI)",
        direction=ndbi_dir,
        raw_magnitude=ndbi_mag,
        normalized_strength=ndbi_str,
        support_state=ndbi_state_exp,
        support_score=ndbi_score_exp,
        valid=is_valid and ndbi_delta is not None,
        interpretation=ndbi_interp,
        details={
            "delta": ndbi_delta,
            "mean_before": ndbi_before,
            "mean_after": ndbi_after,
            "counter_support_state": ndbi_state_red,
            "counter_support_score": ndbi_score_red,
        },
    )

    # 2. NDVI Signal (Inverse response for urban expansion: vegetation clearing)
    ndvi_dir, ndvi_mag, ndvi_str = normalize_signal_magnitude(ndvi_delta)
    ndvi_state_exp, ndvi_score_exp = evaluate_directional_support(ndvi_dir, ndvi_str, expected_direction="decrease", valid=is_valid and ndvi_delta is not None)
    ndvi_state_red, ndvi_score_red = evaluate_directional_support(ndvi_dir, ndvi_str, expected_direction="increase", valid=is_valid and ndvi_delta is not None)

    ndvi_interp = (
        f"NDVI shifted by {ndvi_delta:+.4f} ({ndvi_dir}). "
        + ("Vegetation loss corroborates potential urban expansion." if ndvi_dir == "decrease" else
           "Vegetation increase opposes urban expansion (vegetation gain)." if ndvi_dir == "increase" else
           "Vegetation change is within neutral deadband (|delta| <= 0.05).")
        if ndvi_delta is not None else "NDVI data unavailable."
    )

    ndvi_signal = EvidenceSignal(
        signal_name="ndvi",
        display_name="Normalized Difference Vegetation Index (NDVI)",
        direction=ndvi_dir,
        raw_magnitude=ndvi_mag,
        normalized_strength=ndvi_str,
        support_state=ndvi_state_exp,
        support_score=ndvi_score_exp,
        valid=is_valid and ndvi_delta is not None,
        interpretation=ndvi_interp,
        details={
            "delta": ndvi_delta,
            "mean_before": ndvi_before,
            "mean_after": ndvi_after,
            "counter_support_state": ndvi_state_red,
            "counter_support_score": ndvi_score_red,
        },
    )

    # 3. Spectral Signal (SWIR and Red reflectance shift)
    swir_delta = spectral_shifts.get("swir")
    red_delta = spectral_shifts.get("red")
    
    if swir_delta is not None or red_delta is not None:
        # Combined spectral shift: positive shift in SWIR/Red supports built-up/cleared surface
        spec_val = (swir_delta or 0.0) * 0.6 + (red_delta or 0.0) * 0.4
        spec_dir, spec_mag, spec_str = normalize_signal_magnitude(
            spec_val,
            deadband=EvidenceThresholds.SPECTRAL_DEADBAND,
            strong=EvidenceThresholds.SPECTRAL_STRONG,
        )
        spec_state, spec_score = evaluate_directional_support(spec_dir, spec_str, expected_direction="increase", valid=is_valid)
        spec_interp = (
            f"SWIR shift: {swir_delta:+.4f}, Red shift: {red_delta:+.4f}. "
            + ("Brightening in SWIR/Red corroborates impervious or cleared surfaces." if spec_dir == "increase" else
               "Darkening in SWIR/Red opposes impervious surface construction." if spec_dir == "decrease" else
               "Spectral reflectance change is within baseline variance.")
        )
    else:
        spec_dir = "none"
        spec_mag = 0.0
        spec_str = 0.0
        spec_state = "unavailable"
        spec_score = 0.0
        spec_interp = "Spectral reflectance bands unavailable for direct evaluation."

    spectral_signal = EvidenceSignal(
        signal_name="spectral",
        display_name="Spectral Reflectance Shift (SWIR / Red)",
        direction=spec_dir,
        raw_magnitude=spec_mag,
        normalized_strength=spec_str,
        support_state=spec_state,
        support_score=spec_score,
        valid=spec_state != "unavailable" and is_valid,
        interpretation=spec_interp,
        details={
            "swir_delta": swir_delta,
            "red_delta": red_delta,
            "bands_used": ["B11_SWIR", "B04_Red"],
        },
    )

    # 4. Quality Signal
    q_val = quality_fraction if quality_fraction is not None else 1.0
    q_score = float(np.clip(q_val, 0.0, 1.0))
    q_state: SupportState = "positive" if q_score >= EvidenceThresholds.QUALITY_HIGH else ("neutral" if q_score >= EvidenceThresholds.QUALITY_MIN_ACCEPTABLE else "uncertain")
    quality_signal = EvidenceSignal(
        signal_name="quality",
        display_name="Observation Quality & Joint Validity",
        direction="neutral",
        raw_magnitude=q_score,
        normalized_strength=q_score,
        support_state=q_state,
        support_score=q_score,
        valid=q_score >= EvidenceThresholds.QUALITY_MIN_ACCEPTABLE,
        interpretation=f"Joint temporal observation validity is {q_score * 100:.2f}%.",
        details={"valid_pixel_ratio": q_score},
    )

    # Component Evidence Scores (pure physical signals)
    ndbi_support = round(ndbi_score_exp, 4)
    ndvi_support = round(ndvi_score_exp, 4)
    spectral_support = round(spec_score, 4)

    # Calculate overall expansion support (bounded [0.0, 1.0])
    # Semantic weighting: NDBI (50%), NDVI (35%), Spectral (15%) - Pure Physical
    if not is_valid:
        semantic_expansion = 0.0
        final_expansion = 0.0
        semantic_reduction = 0.0
        final_reduction = 0.0
        spec_score_red = 0.0
        state = "unavailable"
    else:
        semantic_expansion = (
            EvidenceWeights.URBAN_EXPANSION_NDBI * ndbi_support
            + EvidenceWeights.URBAN_EXPANSION_NDVI * ndvi_support
            + EvidenceWeights.URBAN_EXPANSION_SPECTRAL * spectral_support
        )
        # If NDBI strongly decreases, expansion is negated (contradictory evidence)
        if ndbi_dir == "decrease" and ndbi_str > 0.1:
            semantic_expansion = max(0.0, semantic_expansion - ndbi_str * 0.5)

        semantic_expansion = float(np.clip(semantic_expansion, 0.0, 1.0))
        final_expansion = round(float(np.clip(semantic_expansion * q_score, 0.0, 1.0)), 4)

        # Counter-hypothesis: Urban Reduction Support
        # Supported by NDBI decrease, NDVI increase, and SWIR/Red darkening
        spec_red_val = -spec_val if (swir_delta is not None or red_delta is not None) else None
        if spec_red_val is not None:
            spec_red_dir, spec_red_mag, spec_red_str = normalize_signal_magnitude(
                spec_red_val,
                deadband=EvidenceThresholds.SPECTRAL_DEADBAND,
                strong=EvidenceThresholds.SPECTRAL_STRONG,
            )
            _, spec_score_red = evaluate_directional_support(spec_red_dir, spec_red_str, expected_direction="increase", valid=is_valid)
        else:
            spec_score_red = 0.0

        semantic_reduction = (
            EvidenceWeights.URBAN_REDUCTION_NDBI * ndbi_score_red
            + EvidenceWeights.URBAN_REDUCTION_NDVI * ndvi_score_red
            + EvidenceWeights.URBAN_REDUCTION_SPECTRAL * spec_score_red
        )
        semantic_reduction = float(np.clip(semantic_reduction, 0.0, 1.0))
        final_reduction = round(float(np.clip(semantic_reduction * q_score, 0.0, 1.0)), 4)
        state = "available"

    return {
        "target": "urban",
        "primary_hypothesis": "urban_expansion",
        "state": state,
        "urban_expansion_support": final_expansion,
        "evidence_score": final_expansion,
        "semantic_support": round(semantic_expansion, 4),
        "reliability": {
            "score": round(q_score, 4),
            "state": q_state,
            "valid": is_valid,
            "quality_fraction": quality_fraction,
            "quality_support": round(q_score, 4),
        },
        "component_evidence": {
            "ndbi_support": ndbi_support,
            "ndvi_support": ndvi_support,
            "spectral_support": spectral_support,
        },
        "signals": {
            "ndbi": ndbi_signal.to_dict(),
            "ndvi": ndvi_signal.to_dict(),
            "spectral": spectral_signal.to_dict(),
            "quality": quality_signal.to_dict(),
        },
        "counter_hypothesis": {
            "hypothesis": "urban_reduction",
            "state": state,
            "urban_reduction_support": final_reduction,
            "evidence_score": final_reduction,
            "semantic_support": round(semantic_reduction, 4),
            "ndbi_reduction_support": round(ndbi_score_red, 4),
            "ndvi_reduction_support": round(ndvi_score_red, 4),
            "spectral_reduction_support": round(spec_score_red, 4),
        },
    }


def calculate_vegetation_evidence(
    ndvi_delta: Optional[float],
    ndbi_delta: Optional[float] = None,
    spectral_shifts: Optional[Dict[str, float]] = None,
    quality_fraction: Optional[float] = None,
    ndvi_before: Optional[float] = None,
    ndvi_after: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate structured evidence for vegetation change.

    Hypothesis: Potential Vegetation Loss
      - NDVI decrease supports vegetation loss (+).
      - NDBI/soil increase corroborates exposed surface (+).
      - NIR decrease and Red increase corroborates loss of photosynthetic canopy.

    Counter-Hypothesis: Potential Vegetation Gain
      - NDVI increase supports vegetation gain (+).
    """
    is_valid = quality_fraction is None or quality_fraction >= EvidenceThresholds.QUALITY_MIN_ACCEPTABLE
    spectral_shifts = spectral_shifts or {}

    # 1. NDVI Signal
    ndvi_dir, ndvi_mag, ndvi_str = normalize_signal_magnitude(ndvi_delta)
    ndvi_loss_state, ndvi_loss_score = evaluate_directional_support(ndvi_dir, ndvi_str, expected_direction="decrease", valid=is_valid and ndvi_delta is not None)
    ndvi_gain_state, ndvi_gain_score = evaluate_directional_support(ndvi_dir, ndvi_str, expected_direction="increase", valid=is_valid and ndvi_delta is not None)

    ndvi_interp = (
        f"NDVI changed by {ndvi_delta:+.4f} ({ndvi_dir}). "
        + ("Provides evidence supporting vegetation loss." if ndvi_dir == "decrease" else
           "Provides evidence supporting vegetation gain." if ndvi_dir == "increase" else
           "NDVI change is within neutral deadband (|delta| <= 0.05).")
        if ndvi_delta is not None else "NDVI data unavailable."
    )

    ndvi_signal = EvidenceSignal(
        signal_name="ndvi",
        display_name="Normalized Difference Vegetation Index (NDVI)",
        direction=ndvi_dir,
        raw_magnitude=ndvi_mag,
        normalized_strength=ndvi_str,
        support_state=ndvi_loss_state,
        support_score=ndvi_loss_score,
        valid=is_valid and ndvi_delta is not None,
        interpretation=ndvi_interp,
        details={"delta": ndvi_delta, "mean_before": ndvi_before, "mean_after": ndvi_after},
    )

    # 2. Corroborating NDBI / Soil Signal
    ndbi_dir, ndbi_mag, ndbi_str = normalize_signal_magnitude(ndbi_delta)
    # Built-up / dry soil increase corroborates vegetation loss
    ndbi_state, ndbi_score = evaluate_directional_support(ndbi_dir, ndbi_str, expected_direction="increase", valid=is_valid and ndbi_delta is not None)

    ndbi_interp = (
        f"NDBI / soil response shifted by {ndbi_delta:+.4f} ({ndbi_dir}). "
        + ("NDBI/soil increase corroborates vegetation removal/disturbance." if ndbi_dir == "increase" else
           "NDBI decrease does not suggest bare soil exposure." if ndbi_dir == "decrease" else
           "NDBI is neutral.")
        if ndbi_delta is not None else "NDBI unavailable."
    )

    ndbi_signal = EvidenceSignal(
        signal_name="ndbi",
        display_name="Soil / Built-up Corroboration Index (NDBI)",
        direction=ndbi_dir,
        raw_magnitude=ndbi_mag,
        normalized_strength=ndbi_str,
        support_state=ndbi_state,
        support_score=ndbi_score,
        valid=is_valid and ndbi_delta is not None,
        interpretation=ndbi_interp,
        details={"delta": ndbi_delta},
    )

    # 3. Spectral Signal (NIR drop & Red increase for vegetation loss)
    nir_delta = spectral_shifts.get("nir")
    red_delta = spectral_shifts.get("red")

    if nir_delta is not None or red_delta is not None:
        # Canopy loss: NIR drops (-), Red rises (+) -> spec_loss = (-nir_delta + red_delta) / 2
        spec_loss_val = (-(nir_delta or 0.0) + (red_delta or 0.0)) / 2.0
        spec_dir, spec_mag, spec_str = normalize_signal_magnitude(
            spec_loss_val,
            deadband=EvidenceThresholds.SPECTRAL_DEADBAND,
            strong=EvidenceThresholds.SPECTRAL_STRONG,
        )
        spec_state, spec_score = evaluate_directional_support(spec_dir, spec_str, expected_direction="increase", valid=is_valid)
        spec_interp = (
            f"NIR shift: {nir_delta:+.4f}, Red shift: {red_delta:+.4f}. "
            + ("NIR loss and Red increase corroborates canopy reduction." if spec_dir == "increase" else
               "Spectral response does not indicate canopy reduction." if spec_dir == "decrease" else
               "Spectral change is within baseline variance.")
        )
    else:
        spec_dir = "none"
        spec_mag = 0.0
        spec_str = 0.0
        spec_state = "unavailable"
        spec_score = 0.0
        spec_interp = "Spectral bands unavailable."

    spectral_signal = EvidenceSignal(
        signal_name="spectral",
        display_name="Canopy Reflectance Shift (NIR / Red)",
        direction=spec_dir,
        raw_magnitude=spec_mag,
        normalized_strength=spec_str,
        support_state=spec_state,
        support_score=spec_score,
        valid=spec_state != "unavailable" and is_valid,
        interpretation=spec_interp,
        details={"nir_delta": nir_delta, "red_delta": red_delta, "bands_used": ["B08_NIR", "B04_Red"]},
    )

    # 4. Quality Signal
    q_val = quality_fraction if quality_fraction is not None else 1.0
    q_score = float(np.clip(q_val, 0.0, 1.0))
    q_state: SupportState = "positive" if q_score >= EvidenceThresholds.QUALITY_HIGH else ("neutral" if q_score >= EvidenceThresholds.QUALITY_MIN_ACCEPTABLE else "uncertain")
    quality_signal = EvidenceSignal(
        signal_name="quality",
        display_name="Observation Quality & Joint Validity",
        direction="neutral",
        raw_magnitude=q_score,
        normalized_strength=q_score,
        support_state=q_state,
        support_score=q_score,
        valid=q_score >= EvidenceThresholds.QUALITY_MIN_ACCEPTABLE,
        interpretation=f"Joint temporal observation validity is {q_score * 100:.2f}%.",
        details={"valid_pixel_ratio": q_score},
    )

    ndvi_support = round(ndvi_loss_score, 4)
    ndbi_support = round(ndbi_score, 4)
    spectral_support = round(spec_score, 4)

    # Counter-hypothesis: canopy gain spectral shift (NIR rises, Red drops)
    if nir_delta is not None or red_delta is not None:
        spec_gain_val = ((nir_delta or 0.0) - (red_delta or 0.0)) / 2.0
        spec_gain_dir, spec_gain_mag, spec_gain_str = normalize_signal_magnitude(
            spec_gain_val,
            deadband=EvidenceThresholds.SPECTRAL_DEADBAND,
            strong=EvidenceThresholds.SPECTRAL_STRONG,
        )
        _, spec_gain_score = evaluate_directional_support(spec_gain_dir, spec_gain_str, expected_direction="increase", valid=is_valid)
    else:
        spec_gain_score = 0.0

    if not is_valid:
        semantic_loss = 0.0
        final_loss = 0.0
        semantic_gain = 0.0
        final_gain = 0.0
        state = "unavailable"
    else:
        # Semantic weighting: NDVI (60%), Spectral (25%), NDBI (15%) - Pure Physical
        semantic_loss = (
            EvidenceWeights.VEGETATION_LOSS_NDVI * ndvi_support
            + EvidenceWeights.VEGETATION_LOSS_SPECTRAL * spectral_support
            + EvidenceWeights.VEGETATION_LOSS_NDBI * ndbi_support
        )
        semantic_loss = float(np.clip(semantic_loss, 0.0, 1.0))
        final_loss = round(float(np.clip(semantic_loss * q_score, 0.0, 1.0)), 4)

        # Counter-hypothesis: Vegetation Gain (NDVI 70%, Spectral 30%) - Pure Physical
        semantic_gain = (
            EvidenceWeights.VEGETATION_GAIN_NDVI * ndvi_gain_score
            + EvidenceWeights.VEGETATION_GAIN_SPECTRAL * spec_gain_score
        )
        semantic_gain = float(np.clip(semantic_gain, 0.0, 1.0))
        final_gain = round(float(np.clip(semantic_gain * q_score, 0.0, 1.0)), 4)
        state = "available"

    return {
        "target": "vegetation",
        "primary_hypothesis": "vegetation_loss",
        "state": state,
        "vegetation_loss_support": final_loss,
        "evidence_score": final_loss,
        "semantic_support": round(semantic_loss, 4),
        "reliability": {
            "score": round(q_score, 4),
            "state": q_state,
            "valid": is_valid,
            "quality_fraction": quality_fraction,
            "quality_support": round(q_score, 4),
        },
        "component_evidence": {
            "ndvi_support": ndvi_support,
            "ndbi_support": ndbi_support,
            "spectral_support": spectral_support,
        },
        "signals": {
            "ndvi": ndvi_signal.to_dict(),
            "ndbi": ndbi_signal.to_dict(),
            "spectral": spectral_signal.to_dict(),
            "quality": quality_signal.to_dict(),
        },
        "counter_hypothesis": {
            "hypothesis": "vegetation_gain",
            "state": state,
            "vegetation_gain_support": final_gain,
            "evidence_score": final_gain,
            "semantic_support": round(semantic_gain, 4),
            "ndvi_gain_support": round(ndvi_gain_score, 4),
            "spectral_gain_support": round(spec_gain_score, 4),
        },
    }


def calculate_water_evidence(
    ndwi_delta: Optional[float],
    spectral_shifts: Optional[Dict[str, float]] = None,
    quality_fraction: Optional[float] = None,
    ndwi_before: Optional[float] = None,
    ndwi_after: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate structured evidence for water body change.

    Hypothesis: Potential Water Loss (drying / shrinkage)
      - NDWI decrease supports water loss (+).
      - NIR and SWIR increase supports water loss (+) (water absorbs strongly in NIR/SWIR; exposed soil reflects them).

    Counter-Hypothesis: Potential Water Gain (inundation / reservoir rise)
      - NDWI increase supports water gain (+).
      - NIR/SWIR decrease supports water gain (+).
    """
    is_valid = quality_fraction is None or quality_fraction >= EvidenceThresholds.QUALITY_MIN_ACCEPTABLE
    spectral_shifts = spectral_shifts or {}

    # 1. NDWI Signal
    ndwi_dir, ndwi_mag, ndwi_str = normalize_signal_magnitude(ndwi_delta)
    ndwi_loss_state, ndwi_loss_score = evaluate_directional_support(ndwi_dir, ndwi_str, expected_direction="decrease", valid=is_valid and ndwi_delta is not None)
    ndwi_gain_state, ndwi_gain_score = evaluate_directional_support(ndwi_dir, ndwi_str, expected_direction="increase", valid=is_valid and ndwi_delta is not None)

    ndwi_interp = (
        f"NDWI changed by {ndwi_delta:+.4f} ({ndwi_dir}). "
        + ("Provides evidence supporting water loss / body shrinkage." if ndwi_dir == "decrease" else
           "Provides evidence supporting water gain / expansion." if ndwi_dir == "increase" else
           "NDWI change is within neutral deadband (|delta| <= 0.05).")
        if ndwi_delta is not None else "NDWI data unavailable."
    )

    ndwi_signal = EvidenceSignal(
        signal_name="ndwi",
        display_name="Normalized Difference Water Index (NDWI)",
        direction=ndwi_dir,
        raw_magnitude=ndwi_mag,
        normalized_strength=ndwi_str,
        support_state=ndwi_loss_state,
        support_score=ndwi_loss_score,
        valid=is_valid and ndwi_delta is not None,
        interpretation=ndwi_interp,
        details={"delta": ndwi_delta, "mean_before": ndwi_before, "mean_after": ndwi_after},
    )

    # 2. Spectral Signal (NIR & SWIR absorption behavior)
    # When water recedes, NIR and SWIR reflectance rise significantly (drying)
    nir_delta = spectral_shifts.get("nir")
    swir_delta = spectral_shifts.get("swir")

    if nir_delta is not None or swir_delta is not None:
        water_loss_spectral = ((nir_delta or 0.0) + (swir_delta or 0.0)) / 2.0
        spec_dir, spec_mag, spec_str = normalize_signal_magnitude(
            water_loss_spectral,
            deadband=EvidenceThresholds.SPECTRAL_DEADBAND,
            strong=EvidenceThresholds.SPECTRAL_STRONG,
        )
        spec_state, spec_score = evaluate_directional_support(spec_dir, spec_str, expected_direction="increase", valid=is_valid)
        spec_interp = (
            f"NIR shift: {nir_delta:+.4f}, SWIR shift: {swir_delta:+.4f}. "
            + ("NIR/SWIR brightening corroborates receding water line / soil exposure." if spec_dir == "increase" else
               "NIR/SWIR darkening corroborates increased water absorption." if spec_dir == "decrease" else
               "Spectral absorption change is within baseline variance.")
        )
    else:
        spec_dir = "none"
        spec_mag = 0.0
        spec_str = 0.0
        spec_state = "unavailable"
        spec_score = 0.0
        spec_interp = "Spectral bands unavailable."

    spectral_signal = EvidenceSignal(
        signal_name="spectral",
        display_name="Water Absorption Reflectance Shift (NIR / SWIR)",
        direction=spec_dir,
        raw_magnitude=spec_mag,
        normalized_strength=spec_str,
        support_state=spec_state,
        support_score=spec_score,
        valid=spec_state != "unavailable" and is_valid,
        interpretation=spec_interp,
        details={"nir_delta": nir_delta, "swir_delta": swir_delta, "bands_used": ["B08_NIR", "B11_SWIR"]},
    )

    # 3. Quality Signal
    q_val = quality_fraction if quality_fraction is not None else 1.0
    q_score = float(np.clip(q_val, 0.0, 1.0))
    q_state: SupportState = "positive" if q_score >= EvidenceThresholds.QUALITY_HIGH else ("neutral" if q_score >= EvidenceThresholds.QUALITY_MIN_ACCEPTABLE else "uncertain")
    quality_signal = EvidenceSignal(
        signal_name="quality",
        display_name="Observation Quality & Joint Validity",
        direction="neutral",
        raw_magnitude=q_score,
        normalized_strength=q_score,
        support_state=q_state,
        support_score=q_score,
        valid=q_score >= EvidenceThresholds.QUALITY_MIN_ACCEPTABLE,
        interpretation=f"Joint temporal observation validity is {q_score * 100:.2f}%.",
        details={"valid_pixel_ratio": q_score},
    )

    ndwi_support = round(ndwi_loss_score, 4)
    spectral_support = round(spec_score, 4)

    # Counter-hypothesis: water gain spectral shift (NIR & SWIR decrease due to water absorption)
    if nir_delta is not None or swir_delta is not None:
        water_gain_spectral = (-(nir_delta or 0.0) - (swir_delta or 0.0)) / 2.0
        spec_gain_dir, spec_gain_mag, spec_gain_str = normalize_signal_magnitude(
            water_gain_spectral,
            deadband=EvidenceThresholds.SPECTRAL_DEADBAND,
            strong=EvidenceThresholds.SPECTRAL_STRONG,
        )
        _, spec_gain_score = evaluate_directional_support(spec_gain_dir, spec_gain_str, expected_direction="increase", valid=is_valid)
    else:
        spec_gain_score = 0.0

    if not is_valid:
        semantic_loss = 0.0
        final_loss = 0.0
        semantic_gain = 0.0
        final_gain = 0.0
        state = "unavailable"
    else:
        # Semantic weighting: NDWI (70%), Spectral (30%) - Pure Physical
        semantic_loss = (
            EvidenceWeights.WATER_LOSS_NDWI * ndwi_support
            + EvidenceWeights.WATER_LOSS_SPECTRAL * spectral_support
        )
        semantic_loss = float(np.clip(semantic_loss, 0.0, 1.0))
        final_loss = round(float(np.clip(semantic_loss * q_score, 0.0, 1.0)), 4)

        # Counter-hypothesis: Water Gain (NDWI 70%, Spectral 30%) - Pure Physical
        semantic_gain = (
            EvidenceWeights.WATER_GAIN_NDWI * ndwi_gain_score
            + EvidenceWeights.WATER_GAIN_SPECTRAL * spec_gain_score
        )
        semantic_gain = float(np.clip(semantic_gain, 0.0, 1.0))
        final_gain = round(float(np.clip(semantic_gain * q_score, 0.0, 1.0)), 4)
        state = "available"

    return {
        "target": "water",
        "primary_hypothesis": "water_loss",
        "state": state,
        "water_loss_support": final_loss,
        "evidence_score": final_loss,
        "semantic_support": round(semantic_loss, 4),
        "reliability": {
            "score": round(q_score, 4),
            "state": q_state,
            "valid": is_valid,
            "quality_fraction": quality_fraction,
            "quality_support": round(q_score, 4),
        },
        "component_evidence": {
            "ndwi_support": ndwi_support,
            "spectral_support": spectral_support,
        },
        "signals": {
            "ndwi": ndwi_signal.to_dict(),
            "spectral": spectral_signal.to_dict(),
            "quality": quality_signal.to_dict(),
        },
        "counter_hypothesis": {
            "hypothesis": "water_gain",
            "state": state,
            "water_gain_support": final_gain,
            "evidence_score": final_gain,
            "semantic_support": round(semantic_gain, 4),
            "ndwi_gain_support": round(ndwi_gain_score, 4),
            "spectral_gain_support": round(spec_gain_score, 4),
        },
    }


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def calculate_multi_index_evidence(
    target: Optional[str],
    task: Optional[str],
    execution_results: Dict[str, Any],
    imagery_result: Optional[Dict[str, Any]] = None,
    change_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main orchestration entry point: derives complete multi-index evidence package.
    Consumes already calculated temporal index outputs and cached bands without recalculation.
    """
    target_clean = (target or "").lower().strip()
    task_clean = (task or "").lower().strip()

    # Extract temporal index results
    t_ndvi = execution_results.get("calculate_temporal_ndvi", {})
    t_ndwi = execution_results.get("calculate_temporal_ndwi", {})
    t_ndbi = execution_results.get("calculate_temporal_ndbi", {})

    ndvi_delta = t_ndvi.get("mean_ndvi_change")
    ndwi_delta = t_ndwi.get("mean_ndwi_change")
    ndbi_delta = t_ndbi.get("mean_ndbi_change")

    # Joint validity fraction
    quality_fraction: Optional[float] = None
    for res in [t_ndvi, t_ndwi, t_ndbi, change_result or {}]:
        tot = res.get("total_pixels")
        val = res.get("valid_pixels")
        if tot and val and tot > 0:
            quality_fraction = float(val) / float(tot)
            break

    # Calculate spectral shifts from cached bands if images are present
    spectral_shifts: Dict[str, float] = {}
    if imagery_result and len(imagery_result.get("images", [])) >= 2:
        try:
            import rasterio
            b_bands = imagery_result["images"][0].get("bands", {})
            a_bands = imagery_result["images"][1].get("bands", {})
            
            # Use joint mask if available
            mask_b_p = b_bands.get("mask")
            mask_a_p = a_bands.get("mask")
            m_b = rasterio.open(mask_b_p).read(1).astype(bool) if mask_b_p else None
            m_a = rasterio.open(mask_a_p).read(1).astype(bool) if mask_a_p else None
            j_mask = (m_b & m_a) if (m_b is not None and m_a is not None) else None

            for band_name in ["red", "green", "blue", "nir", "swir"]:
                p_b = b_bands.get(band_name)
                p_a = a_bands.get(band_name)
                if p_b and p_a:
                    arr_b = rasterio.open(p_b).read(1).astype(np.float32)
                    arr_a = rasterio.open(p_a).read(1).astype(np.float32)
                    diff = arr_a - arr_b
                    valid_px = (j_mask & np.isfinite(diff)) if j_mask is not None else np.isfinite(diff)
                    if np.any(valid_px):
                        spectral_shifts[band_name] = float(np.mean(diff[valid_px]))
        except Exception as exc:
            print(f"[EVIDENCE ENGINE WARNING] Spectral shift calculation: {exc}")

    # Determine primary target
    if target_clean == "urban" or "urban" in task_clean:
        evidence = calculate_urban_evidence(
            ndbi_delta=ndbi_delta,
            ndvi_delta=ndvi_delta,
            spectral_shifts=spectral_shifts,
            quality_fraction=quality_fraction,
            ndbi_before=t_ndbi.get("mean_ndbi_before"),
            ndbi_after=t_ndbi.get("mean_ndbi_after"),
            ndvi_before=t_ndvi.get("mean_ndvi_before"),
            ndvi_after=t_ndvi.get("mean_ndvi_after"),
        )
    elif target_clean == "water" or "water" in task_clean:
        evidence = calculate_water_evidence(
            ndwi_delta=ndwi_delta,
            spectral_shifts=spectral_shifts,
            quality_fraction=quality_fraction,
            ndwi_before=t_ndwi.get("mean_ndwi_before"),
            ndwi_after=t_ndwi.get("mean_ndwi_after"),
        )
    else:
        # Default / vegetation
        evidence = calculate_vegetation_evidence(
            ndvi_delta=ndvi_delta,
            ndbi_delta=ndbi_delta,
            spectral_shifts=spectral_shifts,
            quality_fraction=quality_fraction,
            ndvi_before=t_ndvi.get("mean_ndvi_before"),
            ndvi_after=t_ndvi.get("mean_ndvi_after"),
        )

    # Attach shared metadata and cross-index summary
    evidence["metadata"] = {
        "engine_version": "5A.2",
        "thresholds": {
            "index_deadband": EvidenceThresholds.INDEX_DEADBAND,
            "index_strong": EvidenceThresholds.INDEX_STRONG,
            "spectral_deadband": EvidenceThresholds.SPECTRAL_DEADBAND,
            "spectral_strong": EvidenceThresholds.SPECTRAL_STRONG,
            "quality_high": EvidenceThresholds.QUALITY_HIGH,
            "quality_min_acceptable": EvidenceThresholds.QUALITY_MIN_ACCEPTABLE,
        },
        "weights": {
            "urban_expansion": {
                "ndbi": EvidenceWeights.URBAN_EXPANSION_NDBI,
                "ndvi": EvidenceWeights.URBAN_EXPANSION_NDVI,
                "spectral": EvidenceWeights.URBAN_EXPANSION_SPECTRAL,
            },
            "urban_reduction": {
                "ndbi": EvidenceWeights.URBAN_REDUCTION_NDBI,
                "ndvi": EvidenceWeights.URBAN_REDUCTION_NDVI,
                "spectral": EvidenceWeights.URBAN_REDUCTION_SPECTRAL,
            },
            "vegetation_loss": {
                "ndvi": EvidenceWeights.VEGETATION_LOSS_NDVI,
                "spectral": EvidenceWeights.VEGETATION_LOSS_SPECTRAL,
                "ndbi": EvidenceWeights.VEGETATION_LOSS_NDBI,
            },
            "vegetation_gain": {
                "ndvi": EvidenceWeights.VEGETATION_GAIN_NDVI,
                "spectral": EvidenceWeights.VEGETATION_GAIN_SPECTRAL,
            },
            "water_loss": {
                "ndwi": EvidenceWeights.WATER_LOSS_NDWI,
                "spectral": EvidenceWeights.WATER_LOSS_SPECTRAL,
            },
            "water_gain": {
                "ndwi": EvidenceWeights.WATER_GAIN_NDWI,
                "spectral": EvidenceWeights.WATER_GAIN_SPECTRAL,
            },
        },
        "all_index_deltas": {
            "delta_ndvi": ndvi_delta,
            "delta_ndwi": ndwi_delta,
            "delta_ndbi": ndbi_delta,
        },
        "spectral_shifts": spectral_shifts,
        "quality_fraction": quality_fraction,
    }

    return evidence
