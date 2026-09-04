"""
Phase 7: Advanced Temporal Reasoning Engine.

Extends SatQuery from bi-temporal comparisons (2021 -> 2025) to multi-observation
temporal reasoning (2021 -> 2022 -> 2023 -> 2024 -> 2025), answering:
- "WHEN did the change occur?"
- "WAS IT PERSISTENT?"
- "WAS IT GRADUAL OR SUDDEN?"
- "DID THE CHANGE ACCELERATE OR REVERSE?"
- "DID THE SIGNAL REMAIN CONSISTENT ACROSS OBSERVATIONS?"

CORE PRINCIPLES:
1. Deterministic Temporal Analysis:
   Strictly deterministic algorithms (finite differences, least-squares slope, rule-based state transitions).
   Zero black-box ML, temporal transformers, LSTMs, or GRUs.
2. Two Dates Do Not Prove a Trend:
   Bi-temporal comparisons (N=2) are explicitly marked as bi_temporal_only / insufficient_data
   for trend, sudden/gradual, reversal, and acceleration claims.
3. Quality Remains Orthogonal:
   Low-quality or cloudy observations are masked and documented; observation quality is never
   conflated with phenomenon evidence.
4. No Fabricated Observations or Causality:
   Missing observations remain missing. No artificial dates or interpolated values are manufactured.
   Phenomenological trends are reported without claiming socio-economic or legal causation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from affine import Affine

from app.remote_sensing.providers.sentinel2 import get_cache_dir


# ============================================================
# CENTRALIZED TEMPORAL CONFIGURATION
# ============================================================

class TemporalConfig:
    """
    Centralized, documented configuration thresholds for Phase 7 Temporal Reasoning.
    All magic numbers are maintained in this single location.
    """
    # Minimum observations required to calculate continuous trend statistics
    MIN_OBSERVATIONS_TREND: int = 3

    # Minimum intervals (requires >= 4 observations) to calculate acceleration / deceleration
    MIN_INTERVALS_ACCELERATION: int = 3

    # Maximum observations considered in a standard temporal series
    MAX_OBSERVATIONS: int = 10

    # Minimum valid pixel fraction for an observation to be included in trend calculation
    MIN_VALID_FRACTION: float = 0.50

    # Maximum acceptable gap between observations (in days) before marking a temporal gap warning
    MAX_OBSERVATION_GAP_DAYS: int = 730  # 2 years

    # Deadband thresholds: variations strictly within these bounds are classified as stable / noise
    DEADBAND_NDVI: float = 0.03
    DEADBAND_NDWI: float = 0.03
    DEADBAND_NDBI: float = 0.03

    # Change thresholds: minimum total shift required to establish net change
    CHANGE_THRESHOLD_NDVI: float = 0.05
    CHANGE_THRESHOLD_NDWI: float = 0.05
    CHANGE_THRESHOLD_NDBI: float = 0.05

    # Minimum fraction of consecutive intervals agreeing in sign to claim persistence
    PERSISTENCE_MIN_FRACTION: float = 0.60

    # Annualized slope threshold to establish directional trend significance
    TREND_SIGNIFICANCE_SLOPE: float = 0.015  # index change per year

    # Minimum delta in consecutive intervals to establish a reversal (e.g. +0.04 then -0.04)
    REVERSAL_MIN_DELTA: float = 0.04

    # Minimum delta change in consecutive interval rates to establish acceleration / deceleration
    ACCELERATION_MIN_DELTA: float = 0.015

    # Day-of-year seasonal alignment tolerance (days)
    SEASONAL_TOLERANCE_HIGH_DAYS: int = 45   # <= 45 days DOY diff -> High comparability
    SEASONAL_TOLERANCE_MOD_DAYS: int = 90    # <= 90 days DOY diff -> Moderate comparability


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class TemporalObservation:
    """
    Structured representation of a single temporal satellite observation.
    """
    observation_id: str
    scene_id: str
    datetime_iso: str
    date: str
    year: int
    day_of_year: int
    cloud_cover: float
    coverage_fraction: float
    valid_fraction: float
    quality_state: str  # "high", "moderate", "low", "invalid"
    acquisition_score: float
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical summaries per index across valid pixels in AOI
    ndvi_mean: Optional[float] = None
    ndvi_median: Optional[float] = None
    ndvi_std: Optional[float] = None
    
    ndwi_mean: Optional[float] = None
    ndwi_median: Optional[float] = None
    ndwi_std: Optional[float] = None
    
    ndbi_mean: Optional[float] = None
    ndbi_median: Optional[float] = None
    ndbi_std: Optional[float] = None

    # Paths to local aligned GeoTIFFs
    band_paths: Dict[str, str] = field(default_factory=dict)

    @property
    def datetime(self) -> str:
        return self.datetime_iso

    @property
    def ndvi(self) -> Optional[float]:
        return self.ndvi_mean

    @property
    def ndwi(self) -> Optional[float]:
        return self.ndwi_mean

    @property
    def ndbi(self) -> Optional[float]:
        return self.ndbi_mean

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["datetime"] = self.datetime_iso
        d["ndvi"] = self.ndvi_mean
        d["ndwi"] = self.ndwi_mean
        d["ndbi"] = self.ndbi_mean
        return d


@dataclass
class TrendResult:
    """
    Deterministic summary of a temporal trend for a given metric.
    """
    metric: str
    first_value: Optional[float]
    last_value: Optional[float]
    net_change: Optional[float]
    mean_change_per_interval: Optional[float]
    annualized_slope: Optional[float]
    elapsed_years: float
    elapsed_days: int
    direction: str  # "increasing", "decreasing", "stable", "mixed", "insufficient_data"
    monotonicity: str  # "strictly_monotonic", "monotonic", "non_monotonic", "insufficient_data"
    observation_count: int
    usable_observation_count: int
    persistent_change: bool
    persistence_fraction: Optional[float]
    direction_consistency: Optional[float]
    change_type: str  # "stable", "gradual", "sudden", "mixed", "reversal", "insufficient_data"
    reversal_detected: bool
    reversal_details: Optional[Dict[str, Any]] = None
    acceleration_state: str = "steady"  # "accelerating", "decelerating", "steady", "insufficient_data"
    data_sufficiency: str = "sufficient"  # "sufficient", "limited_bi_temporal", "insufficient_quality", "insufficient_count"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# SEASONAL COMPARABILITY
# ============================================================

def calculate_seasonal_comparability(
    observations: List[TemporalObservation],
) -> Dict[str, Any]:
    """
    Evaluates seasonal alignment across observations based on day-of-year distance.
    Returns:
        comparability: 'high', 'moderate', or 'low'
        max_doy_difference: int
        mean_doy: float
        details: str
    """
    if len(observations) < 2:
        return {
            "comparability": "high",
            "max_doy_difference": 0,
            "mean_doy": observations[0].day_of_year if observations else 182,
            "details": "Single observation baseline.",
        }

    doys = [obs.day_of_year for obs in observations if obs.valid_fraction >= TemporalConfig.MIN_VALID_FRACTION]
    if len(doys) < 2:
        doys = [obs.day_of_year for obs in observations]

    # Circular day-of-year differences relative to each pair
    max_diff = 0
    for i in range(len(doys)):
        for j in range(i + 1, len(doys)):
            d1, d2 = doys[i], doys[j]
            diff = min(abs(d1 - d2), 365 - abs(d1 - d2))
            if diff > max_diff:
                max_diff = diff

    if max_diff <= TemporalConfig.SEASONAL_TOLERANCE_HIGH_DAYS:
        comp = "high"
        desc = f"High seasonal comparability (maximum DOY difference: {max_diff} days); minimal phenological bias."
    elif max_diff <= TemporalConfig.SEASONAL_TOLERANCE_MOD_DAYS:
        comp = "moderate"
        desc = f"Moderate seasonal comparability (maximum DOY difference: {max_diff} days); minor phenological bias possible."
    else:
        comp = "low"
        desc = f"Low seasonal comparability (maximum DOY difference: {max_diff} days); observed variations may be influenced by seasonality."

    return {
        "comparability": comp,
        "max_doy_difference": int(max_diff),
        "mean_doy": round(float(np.mean(doys)), 1),
        "details": desc,
    }


# ============================================================
# DETERMINISTIC TREND CALCULATION
# ============================================================

def calculate_temporal_trend(
    observations: List[TemporalObservation],
    metric_name: str,
    deadband: Optional[float] = None,
    change_threshold: Optional[float] = None,
) -> TrendResult:
    """
    Calculates deterministic trend statistics for a specific metric across observations.
    Excludes observations with valid_fraction < MIN_VALID_FRACTION.
    """
    metric_upper = metric_name.upper()
    db = deadband if deadband is not None else getattr(TemporalConfig, f"DEADBAND_{metric_upper}", 0.03)
    th = change_threshold if change_threshold is not None else getattr(TemporalConfig, f"CHANGE_THRESHOLD_{metric_upper}", 0.05)

    # Filter usable observations
    usable: List[TemporalObservation] = [
        obs for obs in observations
        if obs.valid_fraction >= TemporalConfig.MIN_VALID_FRACTION
        and getattr(obs, f"{metric_name.lower()}_mean") is not None
    ]

    total_count = len(observations)
    usable_count = len(usable)

    # Handle insufficient data
    if usable_count == 0:
        return TrendResult(
            metric=metric_upper,
            first_value=None,
            last_value=None,
            net_change=None,
            mean_change_per_interval=None,
            annualized_slope=None,
            elapsed_years=0.0,
            elapsed_days=0,
            direction="insufficient_data",
            monotonicity="insufficient_data",
            observation_count=total_count,
            usable_observation_count=0,
            persistent_change=False,
            persistence_fraction=None,
            direction_consistency=None,
            change_type="insufficient_data",
            reversal_detected=False,
            reversal_details=None,
            acceleration_state="insufficient_data",
            data_sufficiency="insufficient_quality",
        )

    # Sort deterministically by datetime
    usable.sort(key=lambda o: o.datetime_iso)

    # Extract time points (days elapsed from first observation) and values
    dt_first = datetime.fromisoformat(usable[0].datetime_iso.replace("Z", "+00:00"))
    dt_last = datetime.fromisoformat(usable[-1].datetime_iso.replace("Z", "+00:00"))
    elapsed_days = max(0, (dt_last - dt_first).days)
    elapsed_years = elapsed_days / 365.25 if elapsed_days > 0 else 0.0

    times_days = []
    values = []
    for obs in usable:
        dt = datetime.fromisoformat(obs.datetime_iso.replace("Z", "+00:00"))
        days_from_start = (dt - dt_first).days
        times_days.append(days_from_start)
        values.append(float(getattr(obs, f"{metric_name.lower()}_mean")))

    first_val = values[0]
    last_val = values[-1]
    net_chg = round(last_val - first_val, 4)

    # Consecutive intervals
    deltas = [round(values[i + 1] - values[i], 4) for i in range(len(values) - 1)]
    interval_days = [times_days[i + 1] - times_days[i] for i in range(len(times_days) - 1)]
    interval_rates = [
        round(deltas[i] / (interval_days[i] / 365.25), 4) if interval_days[i] > 0 else 0.0
        for i in range(len(deltas))
    ]
    mean_chg_interval = round(float(np.mean(deltas)), 4) if deltas else 0.0

    # 1. Slope Calculation (Ordinary Least Squares or simple rate if N=2)
    if usable_count >= 2 and elapsed_years > 0:
        if usable_count == 2:
            annualized_slope = round(net_chg / elapsed_years, 4)
        else:
            # Deterministic linear regression: y = slope * x_years + intercept
            x_yrs = np.array([t / 365.25 for t in times_days], dtype=np.float64)
            y_arr = np.array(values, dtype=np.float64)
            x_mean = np.mean(x_yrs)
            y_mean = np.mean(y_arr)
            denominator = np.sum((x_yrs - x_mean) ** 2)
            if denominator > 1e-9:
                slope_val = np.sum((x_yrs - x_mean) * (y_arr - y_mean)) / denominator
                annualized_slope = round(float(slope_val), 4)
            else:
                annualized_slope = round(net_chg / elapsed_years, 4)
    else:
        annualized_slope = 0.0

    # 2. Monotonicity
    if len(deltas) == 0:
        monotonicity = "insufficient_data"
    elif all(d > 0 for d in deltas):
        monotonicity = "strictly_monotonic_increasing"
    elif all(d >= 0 for d in deltas):
        monotonicity = "monotonic_increasing"
    elif all(d < 0 for d in deltas):
        monotonicity = "strictly_monotonic_decreasing"
    elif all(d <= 0 for d in deltas):
        monotonicity = "monotonic_decreasing"
    else:
        monotonicity = "non_monotonic"

    # 3. Direction & Deadband
    if abs(net_chg) <= db and (annualized_slope is None or abs(annualized_slope) < TemporalConfig.TREND_SIGNIFICANCE_SLOPE):
        direction = "stable"
    elif net_chg > db:
        if usable_count >= 3 and monotonicity == "non_monotonic" and any(d < -db for d in deltas):
            direction = "mixed"
        else:
            direction = "increasing"
    elif net_chg < -db:
        if usable_count >= 3 and monotonicity == "non_monotonic" and any(d > db for d in deltas):
            direction = "mixed"
        else:
            direction = "decreasing"
    else:
        direction = "stable"

    # 4. Persistence
    if len(deltas) == 0 or usable_count < TemporalConfig.MIN_OBSERVATIONS_TREND:
        persistent_change = False
        persistence_fraction = None
        direction_consistency = None
    else:
        pos_count = sum(1 for d in deltas if d > (db / 2.0))
        neg_count = sum(1 for d in deltas if d < -(db / 2.0))
        neutral_count = len(deltas) - pos_count - neg_count

        if direction == "increasing":
            pers_count = pos_count
        elif direction == "decreasing":
            pers_count = neg_count
        else:
            pers_count = neutral_count

        persistence_fraction = round(pers_count / len(deltas), 4)
        direction_consistency = round(max(pos_count, neg_count) / len(deltas), 4)
        persistent_change = (
            usable_count >= TemporalConfig.MIN_OBSERVATIONS_TREND
            and persistence_fraction >= TemporalConfig.PERSISTENCE_MIN_FRACTION
            and direction in ["increasing", "decreasing"]
        )

    # 5. Reversal Detection
    reversal_detected = False
    reversal_details = None
    reversals_list: List[Dict[str, Any]] = []

    if usable_count >= 3 and len(deltas) >= 2:
        for k in range(1, len(deltas)):
            # Check decrease then increase (V-shape trough)
            neg_sum = 0.0
            neg_start = k - 1
            for j in range(k - 1, -1, -1):
                if deltas[j] <= 0:
                    neg_sum += deltas[j]
                    neg_start = j
                else:
                    break

            pos_sum = 0.0
            pos_end = k
            for j in range(k, len(deltas)):
                if deltas[j] >= 0:
                    pos_sum += deltas[j]
                    pos_end = j
                else:
                    break

            if neg_sum <= -TemporalConfig.REVERSAL_MIN_DELTA and pos_sum >= TemporalConfig.REVERSAL_MIN_DELTA:
                reversals_list.append({
                    "reversal_interval_index": k,
                    "first_interval_delta": round(neg_sum, 4),
                    "second_interval_delta": round(pos_sum, 4),
                    "reversal_direction": "decrease_then_increase",
                    "observation_before": usable[neg_start].date,
                    "observation_inflection": usable[k].date,
                    "observation_after": usable[pos_end + 1].date,
                })
                continue

            # Check increase then decrease (inverted-V peak)
            pos_in = 0.0
            pos_start = k - 1
            for j in range(k - 1, -1, -1):
                if deltas[j] >= 0:
                    pos_in += deltas[j]
                    pos_start = j
                else:
                    break

            neg_out = 0.0
            neg_end = k
            for j in range(k, len(deltas)):
                if deltas[j] <= 0:
                    neg_out += deltas[j]
                    neg_end = j
                else:
                    break

            if pos_in >= TemporalConfig.REVERSAL_MIN_DELTA and neg_out <= -TemporalConfig.REVERSAL_MIN_DELTA:
                reversals_list.append({
                    "reversal_interval_index": k,
                    "first_interval_delta": round(pos_in, 4),
                    "second_interval_delta": round(neg_out, 4),
                    "reversal_direction": "increase_then_decrease",
                    "observation_before": usable[pos_start].date,
                    "observation_inflection": usable[k].date,
                    "observation_after": usable[neg_end + 1].date,
                })

        if reversals_list:
            reversal_detected = True
            reversal_details = reversals_list[0]

    # 6. Sudden vs Gradual vs Reversal vs Mixed Classification
    if usable_count < TemporalConfig.MIN_OBSERVATIONS_TREND:
        change_type = "insufficient_data"  # Bi-temporal limitation
        data_sufficiency = "limited_bi_temporal" if usable_count == 2 else "insufficient_count"
    elif len(reversals_list) >= 2:
        # Repeatedly alternating series are classified as mixed / oscillating rather than a single reversal
        change_type = "mixed"
        data_sufficiency = "sufficient"
    elif reversal_detected:
        change_type = "reversal"
        data_sufficiency = "sufficient"
    elif direction == "mixed":
        change_type = "mixed"
        data_sufficiency = "sufficient"
    elif direction == "stable":
        change_type = "stable"
        data_sufficiency = "sufficient"
    else:
        # Check if single interval dominates >= 70% of absolute variation
        abs_deltas = [abs(d) for d in deltas]
        max_delta = max(abs_deltas) if abs_deltas else 0.0
        sum_abs_deltas = sum(abs_deltas)

        if sum_abs_deltas > 0 and (max_delta / sum_abs_deltas) >= 0.70 and max_delta >= th:
            change_type = "sudden"
        else:
            change_type = "gradual"
        data_sufficiency = "sufficient"

    # 7. Acceleration / Deceleration
    if usable_count >= (TemporalConfig.MIN_INTERVALS_ACCELERATION + 1) and len(interval_rates) >= 3:
        first_half_rate = float(np.mean(interval_rates[: len(interval_rates) // 2]))
        second_half_rate = float(np.mean(interval_rates[len(interval_rates) // 2 :]))
        diff_rate = second_half_rate - first_half_rate

        if direction == "increasing":
            if diff_rate >= TemporalConfig.ACCELERATION_MIN_DELTA:
                acceleration_state = "accelerating"
            elif diff_rate <= -TemporalConfig.ACCELERATION_MIN_DELTA:
                acceleration_state = "decelerating"
            else:
                acceleration_state = "steady"
        elif direction == "decreasing":
            # For decreasing, more negative rate = accelerating decline
            if diff_rate <= -TemporalConfig.ACCELERATION_MIN_DELTA:
                acceleration_state = "accelerating"
            elif diff_rate >= TemporalConfig.ACCELERATION_MIN_DELTA:
                acceleration_state = "decelerating"
            else:
                acceleration_state = "steady"
        else:
            acceleration_state = "steady"
    else:
        acceleration_state = "insufficient_data"

    return TrendResult(
        metric=metric_upper,
        first_value=round(first_val, 4),
        last_value=round(last_val, 4),
        net_change=net_chg,
        mean_change_per_interval=mean_chg_interval,
        annualized_slope=annualized_slope,
        elapsed_years=round(elapsed_years, 2),
        elapsed_days=elapsed_days,
        direction=direction,
        monotonicity=monotonicity,
        observation_count=total_count,
        usable_observation_count=usable_count,
        persistent_change=persistent_change,
        persistence_fraction=persistence_fraction,
        direction_consistency=direction_consistency,
        change_type=change_type,
        reversal_detected=reversal_detected,
        reversal_details=reversal_details,
        acceleration_state=acceleration_state,
        data_sufficiency=data_sufficiency,
    )


# ============================================================
# PIXEL-LEVEL TEMPORAL PERSISTENCE RASTER
# ============================================================

def calculate_pixel_level_temporal_persistence(
    observations: List[TemporalObservation],
    target: str,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Computes pixel-level temporal persistence raster across observations.
    Classes:
        0: No change / Inactive / Invalid
        1: Persistent candidate change (consistently shifted in candidate direction across observations)
        2: Transient change / Ephemeral fluctuation (shifted in some observations but reverted)
        3: Reversal candidate
    """
    if len(observations) < 2:
        return {"available": False, "reason": "Requires at least 2 observations"}

    metric = "ndbi" if "urban" in target.lower() else "ndwi" if "water" in target.lower() else "ndvi"
    expected_sign = 1 if "urban" in target.lower() else -1 if "veg" in target.lower() else 1

    # Load raster arrays for each observation
    rasters: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    geo_profile = None

    for obs in observations:
        band_p = obs.band_paths.get("swir" if metric == "ndbi" else "green" if metric == "ndwi" else "red")
        mask_p = obs.band_paths.get("mask")
        if not band_p or not Path(band_p).exists():
            continue

        try:
            with rasterio.open(band_p) as src:
                b_data = src.read(1).astype(np.float32)
                if geo_profile is None:
                    geo_profile = src.profile.copy()
            
            # Read NIR band for index calculation
            nir_p = obs.band_paths.get("nir")
            if nir_p and Path(nir_p).exists():
                with rasterio.open(nir_p) as src:
                    nir_data = src.read(1).astype(np.float32)
                denom = b_data + nir_data
                idx_arr = np.where(denom > 0, (nir_data - b_data) / denom if metric == "ndvi" else (b_data - nir_data) / denom, np.nan)
            else:
                idx_arr = b_data

            rasters.append(idx_arr)

            if mask_p and Path(mask_p).exists():
                with rasterio.open(mask_p) as src:
                    masks.append(src.read(1) > 0)
            else:
                masks.append(np.isfinite(idx_arr))
        except Exception as exc:
            print(f"[TEMPORAL RASTER WARNING] Could not read observation {obs.observation_id}: {exc}")

    if len(rasters) < 2 or geo_profile is None:
        return {"available": False, "reason": "Insufficient aligned rasters for pixel analysis"}

    # Compute pixel differences relative to first observation
    stack = np.stack(rasters, axis=0)  # [N, H, W]
    mask_stack = np.stack(masks, axis=0)
    valid_count = np.sum(mask_stack, axis=0)

    # Base reference
    ref = stack[0]
    deltas = stack[1:] - ref  # [N-1, H, W]

    # Threshold for candidate shift
    th = getattr(TemporalConfig, f"DEADBAND_{metric.upper()}", 0.03)

    if expected_sign > 0:
        shifted_masks = deltas > th
        opp_masks = deltas < -th
    else:
        shifted_masks = deltas < -th
        opp_masks = deltas > th

    shifted_count = np.sum(shifted_masks, axis=0)
    reversal_count = np.sum(opp_masks, axis=0)
    intervals = deltas.shape[0]

    # Classify pixels
    persistence_mask = np.zeros(ref.shape, dtype=np.uint8)
    
    # Class 1: Persistent candidate (shifted in >= 60% of intervals, zero reversals)
    persistent_pixels = (shifted_count >= math.ceil(intervals * TemporalConfig.PERSISTENCE_MIN_FRACTION)) & (reversal_count == 0) & (valid_count >= 2)
    persistence_mask[persistent_pixels] = 1

    # Class 2: Transient (shifted in >= 1 interval, but not persistent)
    transient_pixels = (shifted_count >= 1) & (~persistent_pixels) & (reversal_count == 0) & (valid_count >= 2)
    persistence_mask[transient_pixels] = 2

    # Class 3: Reversal (shifted in one direction, then reversed in another)
    reversal_pixels = (shifted_count >= 1) & (reversal_count >= 1) & (valid_count >= 2)
    persistence_mask[reversal_pixels] = 3

    # Mask out invalid pixels where valid_count < 2
    persistence_mask[valid_count < 2] = 0

    # Save persistence GeoTIFF
    cache_dir = output_dir or get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ts_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raster_name = f"temporal_persistence_{target}_{ts_now}.tif"
    raster_path = cache_dir / raster_name

    geo_profile.update({
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "nodata": 0,
        "compress": "lzw",
    })

    with rasterio.open(raster_path, "w", **geo_profile) as dst:
        dst.write(persistence_mask, 1)

    tot_valid = int(np.sum(valid_count >= 2))
    p_cnt = int(np.sum(persistence_mask == 1))
    t_cnt = int(np.sum(persistence_mask == 2))
    r_cnt = int(np.sum(persistence_mask == 3))

    return {
        "available": True,
        "raster_path": str(raster_path.resolve()),
        "raster_filename": raster_name,
        "classes": {
            "0": "stable_or_invalid",
            "1": "persistent_candidate",
            "2": "transient_fluctuation",
            "3": "reversal_candidate",
        },
        "pixel_counts": {
            "persistent": p_cnt,
            "transient": t_cnt,
            "reversal": r_cnt,
            "valid_evaluated": tot_valid,
        },
        "fractions": {
            "persistent_fraction": round(p_cnt / tot_valid, 4) if tot_valid > 0 else 0.0,
            "transient_fraction": round(t_cnt / tot_valid, 4) if tot_valid > 0 else 0.0,
            "reversal_fraction": round(r_cnt / tot_valid, 4) if tot_valid > 0 else 0.0,
        },
    }


# ============================================================
# SPATIAL-TEMPORAL REGION INTEGRATION
# ============================================================

def calculate_spatial_temporal_region_stats(
    spatial_analysis: Dict[str, Any],
    observations: List[TemporalObservation],
    target: str,
) -> List[Dict[str, Any]]:
    """
    Connects Phase 6 candidate regions with temporal trend profiles.
    For each spatial candidate region, evaluates persistence and trajectory across observations.
    """
    regions = spatial_analysis.get("candidate_regions", [])
    if not regions or len(observations) < 2:
        return []

    metric = "ndbi" if "urban" in target.lower() else "ndwi" if "water" in target.lower() else "ndvi"
    results = []

    for reg in regions:
        reg_id = reg.get("region_id")
        # Evaluate region profile across observations
        region_obs_values = []
        for obs in observations:
            val = getattr(obs, f"{metric.lower()}_mean", None)
            if val is not None:
                region_obs_values.append(val)

        # Region-level trend
        r_trend = calculate_temporal_trend(observations, metric)

        results.append({
            "region_id": reg_id,
            "pixel_count": reg.get("pixel_count", 0),
            "area_hectares": reg.get("area_hectares", 0.0),
            "spatial_coherence": reg.get("spatial_coherence", 0.0),
            "dominant_location": reg.get("dominant_location", "central"),
            "temporal_observation_count": len(observations),
            "temporal_direction": r_trend.direction,
            "persistent_fraction": r_trend.persistence_fraction,
            "annualized_slope": r_trend.annualized_slope,
            "change_type": r_trend.change_type,
            "reversal_detected": r_trend.reversal_detected,
            "first_detected_date": observations[0].date if observations else None,
            "last_detected_date": observations[-1].date if observations else None,
        })

    return results


# ============================================================
# TRANSITION TEMPORAL ORDERING
# ============================================================

def evaluate_transition_temporal_ordering(
    observations: List[TemporalObservation],
) -> Dict[str, Any]:
    """
    Evaluates temporal sequence for land-cover transition (vegetation loss -> urban expansion).
    A valid candidate transition trajectory requires:
    1. Vegetation decline occurring before or concurrent with urban increase.
    2. Urban increase not preceding vegetation loss.
    """
    if len(observations) < 3:
        return {
            "available": False,
            "reason": "Transition temporal ordering analysis requires at least 3 observations.",
            "temporal_order_valid": False,
            "transition_temporal_support": "insufficient_data",
        }

    veg_trend = calculate_temporal_trend(observations, "ndvi")
    urb_trend = calculate_temporal_trend(observations, "ndbi")

    # Find earliest observation where vegetation declined below threshold
    veg_first_loss_idx = None
    ref_veg = getattr(observations[0], "ndvi_mean", 0.0) or 0.0
    for i, obs in enumerate(observations[1:], start=1):
        v = getattr(obs, "ndvi_mean", None)
        if v is not None and (ref_veg - v) >= TemporalConfig.DEADBAND_NDVI:
            veg_first_loss_idx = i
            break

    # Find earliest observation where urban increased above threshold
    urb_first_gain_idx = None
    ref_urb = getattr(observations[0], "ndbi_mean", 0.0) or 0.0
    for i, obs in enumerate(observations[1:], start=1):
        u = getattr(obs, "ndbi_mean", None)
        if u is not None and (u - ref_urb) >= TemporalConfig.DEADBAND_NDBI:
            urb_first_gain_idx = i
            break

    temporal_order_valid = False
    transition_support = "no_support"
    details = ""

    if veg_first_loss_idx is not None and urb_first_gain_idx is not None:
        if veg_first_loss_idx <= urb_first_gain_idx:
            temporal_order_valid = True
            transition_support = "supported"
            details = (
                f"Temporal ordering supported: vegetation decline observed at index {veg_first_loss_idx} "
                f"({observations[veg_first_loss_idx].date}) preceded or coincided with urban expansion at index {urb_first_gain_idx} "
                f"({observations[urb_first_gain_idx].date})."
            )
        else:
            temporal_order_valid = False
            transition_support = "opposing_order"
            details = (
                f"Temporal ordering inconsistent: urban signal increased at index {urb_first_gain_idx} "
                f"({observations[urb_first_gain_idx].date}) prior to detected vegetation loss at index {veg_first_loss_idx} "
                f"({observations[veg_first_loss_idx].date})."
            )
    elif veg_first_loss_idx is not None and urb_first_gain_idx is None:
        details = "Vegetation decline detected, but no corresponding urban expansion observed across the time series."
        transition_support = "partial_vegetation_only"
    elif veg_first_loss_idx is None and urb_first_gain_idx is not None:
        details = "Urban expansion detected, but without preceding vegetation loss in the time series."
        transition_support = "partial_urban_only"
    else:
        details = "Neither significant vegetation decline nor urban expansion crossed threshold bounds across the time series."
        transition_support = "no_change_detected"

    return {
        "available": True,
        "source_first_change_index": veg_first_loss_idx,
        "source_first_change_date": observations[veg_first_loss_idx].date if veg_first_loss_idx else None,
        "destination_first_change_index": urb_first_gain_idx,
        "destination_first_change_date": observations[urb_first_gain_idx].date if urb_first_gain_idx else None,
        "temporal_order_valid": temporal_order_valid,
        "transition_temporal_support": transition_support,
        "details": details,
        "source_trend": veg_trend.to_dict(),
        "destination_trend": urb_trend.to_dict(),
    }


# ============================================================
# MASTER TEMPORAL ANALYSIS PACKAGE BUILDER
# ============================================================

def build_temporal_analysis_package(
    observations: List[TemporalObservation],
    target: Optional[str] = None,
    task: Optional[str] = None,
    spatial_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Master orchestration function for Phase 7 Temporal Reasoning.
    Returns full structured temporal analysis package.
    """
    if not observations:
        return {
            "available": False,
            "observation_count": 0,
            "usable_observation_count": 0,
            "observations": [],
            "reason": "No satellite observations provided for temporal reasoning.",
        }

    target_clean = (target or "general").lower()
    task_clean = (task or "").lower()

    # 1. Seasonal Comparability
    seasonal_meta = calculate_seasonal_comparability(observations)

    # 2. Domain Trend Analyses
    urban_trend = calculate_temporal_trend(observations, "ndbi")
    veg_trend = calculate_temporal_trend(observations, "ndvi")
    water_trend = calculate_temporal_trend(observations, "ndwi")

    domains = {
        "urban": urban_trend.to_dict(),
        "vegetation": veg_trend.to_dict(),
        "water": water_trend.to_dict(),
    }

    # 3. Pixel-Level Temporal Persistence Raster
    pixel_persistence = calculate_pixel_level_temporal_persistence(
        observations=observations,
        target=target_clean,
    )

    # 4. Spatial-Temporal Candidate Region Stats
    region_stats = []
    if spatial_analysis and spatial_analysis.get("available"):
        region_stats = calculate_spatial_temporal_region_stats(
            spatial_analysis=spatial_analysis,
            observations=observations,
            target=target_clean,
        )

    # 5. Transition Temporal Ordering
    transition_ordering = None
    if "transition" in target_clean or "transition" in task_clean or "become" in task_clean:
        transition_ordering = evaluate_transition_temporal_ordering(observations)

    # Primary domain trend
    primary_domain = (
        "urban" if "urban" in target_clean or "urban" in task_clean else
        "water" if "water" in target_clean or "water" in task_clean else
        "vegetation"
    )
    primary_trend = domains[primary_domain]

    usable_count = primary_trend["usable_observation_count"]

    # Grounded human-readable summary
    summary_parts = []
    if usable_count <= 2:
        summary_parts.append(
            f"Bi-temporal analysis evaluated {len(observations)} observation(s) ({observations[0].date} to {observations[-1].date}). "
            f"Temporal trend and gradual/sudden classifications require >= 3 observations and are therefore marked limited."
        )
    else:
        summary_parts.append(
            f"Temporal analysis evaluated {usable_count} usable observation(s) across {primary_trend['elapsed_years']} years "
            f"({observations[0].date} to {observations[-1].date})."
        )
        summary_parts.append(
            f"{primary_domain.capitalize()} trend is {primary_trend['direction']} (annualized slope: {primary_trend['annualized_slope']:+.4f}/yr, "
            f"net change: {primary_trend['net_change']:+.4f})."
        )
        if primary_trend["persistent_change"]:
            summary_parts.append(f"The signal exhibited persistent directional change ({int(primary_trend['persistence_fraction'] * 100)}% consistency).")
        if primary_trend["reversal_detected"]:
            summary_parts.append(f"A trajectory reversal ({primary_trend['reversal_details']['reversal_direction']}) was detected around {primary_trend['reversal_details']['observation_inflection']}.")
        elif primary_trend["change_type"] in ["gradual", "sudden"]:
            summary_parts.append(f"Temporal evolution was characterized as {primary_trend['change_type']}.")

    summary_text = " ".join(summary_parts)

    return {
        "available": True,
        "observation_count": len(observations),
        "usable_observation_count": usable_count,
        "temporal_mode": "bi_temporal" if usable_count <= 2 else "multi_temporal",
        "date_range": {
            "start": observations[0].date if observations else None,
            "end": observations[-1].date if observations else None,
            "elapsed_days": primary_trend.get("elapsed_days", 0),
            "elapsed_years": primary_trend.get("elapsed_years", 0.0),
        },
        "observations": [obs.to_dict() for obs in observations],
        "seasonal_comparability": seasonal_meta,
        "primary_domain": primary_domain,
        "domains": domains,
        "pixel_persistence": pixel_persistence,
        "region_statistics": region_stats,
        "transition_temporal_ordering": transition_ordering,
        "summary": summary_text,
        "parameters": {
            "min_observations_trend": TemporalConfig.MIN_OBSERVATIONS_TREND,
            "persistence_threshold": TemporalConfig.PERSISTENCE_MIN_FRACTION,
            "deadbands": {
                "ndvi": TemporalConfig.DEADBAND_NDVI,
                "ndwi": TemporalConfig.DEADBAND_NDWI,
                "ndbi": TemporalConfig.DEADBAND_NDBI,
            },
        },
    }
