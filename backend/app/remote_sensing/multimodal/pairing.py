"""
Optical-SAR Scene Pairing and Coordination Module (Step 13).

Automatically matches, scores, ranks, and retrieves compatible Sentinel-2 (optical)
and Sentinel-1 (SAR) scene pairs for a given Area of Interest (AOI) and time period.

Coordinates existing Sentinel2Provider and Sentinel1Provider without duplicating
STAC querying, SAS URL signing, or local caching infrastructure.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS

from app import config  # Ensures .env variables are loaded
from app.remote_sensing.providers.sentinel2 import (
    Sentinel2Provider,
    Sentinel2RetrievalError,
    calculate_aoi_coverage as calculate_s2_aoi_coverage,
    normalize_aoi,
)
from app.remote_sensing.providers.sentinel1 import (
    SENTINEL1_COLLECTION,
    Sentinel1Provider,
    Sentinel1RetrievalError,
    calculate_aoi_coverage as calculate_s1_aoi_coverage,
)
from app.remote_sensing.multimodal.optical_sar import (
    inspect_geotiff,
    validate_optical_sar_pair,
)


# ============================================================
# CONSTANTS & ERROR DEFINITIONS
# ============================================================

DEFAULT_MAX_TEMPORAL_DELTA_DAYS = 3.0
DEFAULT_MIN_AOI_COVERAGE_PERCENT = 50.0
DEFAULT_CLOUD_COVER_LIMIT = 30.0


class PairingErrorType:
    MALFORMED_AOI = "MALFORMED_AOI"
    INVALID_DATES = "INVALID_DATES"
    NO_OPTICAL_SCENES = "NO_OPTICAL_SCENES"
    NO_SAR_SCENES = "NO_SAR_SCENES"
    NO_TEMPORALLY_COMPATIBLE_PAIR = "NO_TEMPORALLY_COMPATIBLE_PAIR"
    INSUFFICIENT_SPATIAL_OVERLAP = "INSUFFICIENT_SPATIAL_OVERLAP"
    MISSING_POLARIZATION = "MISSING_POLARIZATION"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    UNREADABLE_RASTER = "UNREADABLE_RASTER"


class OpticalSarPairingError(RuntimeError):
    """
    Actionable structured exception for Optical-SAR pairing failures.
    """
    def __init__(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{error_type}] {message}")
        self.error_type = error_type
        self.message = message
        self.details = details or {}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_iso_datetime(dt_val: Union[str, datetime]) -> datetime:
    """
    Parse an ISO-8601 string or datetime into a UTC-aware datetime.
    """
    if isinstance(dt_val, datetime):
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=timezone.utc)
        return dt_val.astimezone(timezone.utc)

    s = str(dt_val).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as exc:
        raise ValueError(f"Cannot parse datetime '{dt_val}': {exc}") from exc


def format_iso_range(
    time_start: Optional[str],
    time_end: Optional[str],
    buffer_days: float = 0.0,
    default_year: str = "2021",
) -> str:
    """
    Construct an ISO-8601 interval string (start/end) with optional day buffer.
    """
    if not time_start and not time_end:
        start_dt = datetime(int(default_year), 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = datetime(int(default_year), 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    elif time_start and not time_end:
        s = str(time_start).strip()
        if len(s) == 4 and s.isdigit():
            start_dt = datetime(int(s), 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            end_dt = datetime(int(s), 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        elif "/" in s:
            parts = s.split("/")
            start_dt = parse_iso_datetime(parts[0])
            end_dt = parse_iso_datetime(parts[1])
        else:
            dt = parse_iso_datetime(s)
            start_dt = dt.replace(hour=0, minute=0, second=0)
            end_dt = dt.replace(hour=23, minute=59, second=59)
    elif not time_start and time_end:
        dt = parse_iso_datetime(time_end)
        start_dt = dt.replace(hour=0, minute=0, second=0)
        end_dt = dt.replace(hour=23, minute=59, second=59)
    else:
        start_dt = parse_iso_datetime(time_start)
        end_dt = parse_iso_datetime(time_end)

    if buffer_days > 0:
        start_dt -= timedelta(days=buffer_days)
        end_dt += timedelta(days=buffer_days)

    return f"{start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def extract_item_bbox(item: Dict[str, Any]) -> List[float]:
    """
    Extract [west, south, east, north] bounding box from a STAC item.
    """
    if "bbox" in item and item["bbox"]:
        return [float(c) for c in item["bbox"][:4]]

    geom = item.get("geometry", {})
    coords = geom.get("coordinates", [])
    flat_pts: List[Tuple[float, float]] = []

    def _collect(c: Any):
        if isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (int, float)):
            flat_pts.append((float(c[0]), float(c[1])))
        elif isinstance(c, list):
            for sub in c:
                _collect(sub)

    _collect(coords)
    if flat_pts:
        return [
            min(p[0] for p in flat_pts),
            min(p[1] for p in flat_pts),
            max(p[0] for p in flat_pts),
            max(p[1] for p in flat_pts),
        ]

    return [-180.0, -90.0, 180.0, 90.0]


def calculate_spatial_intersection(
    opt_bbox: List[float],
    sar_bbox: List[float],
    aoi_bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """
    Compute intersection bounding box between optical footprint, SAR footprint, and requested AOI.
    """
    aoi_w, aoi_s, aoi_e, aoi_n = aoi_bbox
    inter_w = max(opt_bbox[0], sar_bbox[0], aoi_w)
    inter_s = max(opt_bbox[1], sar_bbox[1], aoi_s)
    inter_e = min(opt_bbox[2], sar_bbox[2], aoi_e)
    inter_n = min(opt_bbox[3], sar_bbox[3], aoi_n)

    has_overlap = (inter_e > inter_w) and (inter_n > inter_s)
    inter_area = max(0.0, (inter_e - inter_w) * (inter_n - inter_s)) if has_overlap else 0.0
    aoi_area = max(0.0, (aoi_e - aoi_w) * (aoi_n - aoi_s))
    overlap_ratio = (inter_area / aoi_area) if aoi_area > 0 else 1.0

    return {
        "has_overlap": has_overlap,
        "intersection_bbox": [inter_w, inter_s, inter_e, inter_n] if has_overlap else None,
        "intersection_area": inter_area,
        "aoi_coverage_ratio": min(1.0, max(0.0, overlap_ratio)),
        "aoi_coverage_percent": round(min(100.0, max(0.0, overlap_ratio * 100.0)), 2),
    }


# ============================================================
# CANDIDATE PAIR EVALUATION & RANKING
# ============================================================

def evaluate_candidate_pair(
    optical_item: Dict[str, Any],
    sar_item: Dict[str, Any],
    aoi_bbox: Tuple[float, float, float, float],
    max_temporal_delta_days: float = DEFAULT_MAX_TEMPORAL_DELTA_DAYS,
    prefer_dual_pol: bool = True,
    min_aoi_coverage_percent: float = DEFAULT_MIN_AOI_COVERAGE_PERCENT,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Evaluate a candidate Optical (Sentinel-2) + SAR (Sentinel-1) pair.

    Returns:
        (is_compatible, evaluation_dict_or_none, rejection_reason_or_none)
    """
    # 1. Parse acquisition datetimes
    opt_props = optical_item.get("properties", {})
    sar_props = sar_item.get("properties", {})

    opt_dt_raw = opt_props.get("datetime") or optical_item.get("datetime")
    sar_dt_raw = sar_props.get("datetime") or sar_item.get("datetime")

    if not opt_dt_raw or not sar_dt_raw:
        return False, None, "Missing acquisition datetime metadata on one or both candidate items."

    try:
        opt_dt = parse_iso_datetime(opt_dt_raw)
        sar_dt = parse_iso_datetime(sar_dt_raw)
    except Exception as exc:
        return False, None, f"Failed to parse acquisition datetimes: {exc}"

    temporal_delta_days = abs((sar_dt - opt_dt).total_seconds()) / 86400.0

    # Strict temporal delta check (Step 13E)
    if temporal_delta_days > max_temporal_delta_days:
        return (
            False,
            None,
            f"Temporal separation {temporal_delta_days:.2f} days exceeds allowed maximum of {max_temporal_delta_days:.2f} days.",
        )

    # 2. Spatial coverage evaluation
    opt_cov = calculate_s2_aoi_coverage(optical_item, aoi_bbox)
    sar_cov = calculate_s1_aoi_coverage(sar_item, aoi_bbox)

    opt_cov_pct = float(opt_cov.get("coverage_percentage", 100.0))
    sar_cov_pct = float(sar_cov.get("coverage_percentage", 100.0))

    if opt_cov_pct < min_aoi_coverage_percent:
        return (
            False,
            None,
            f"Optical AOI coverage {opt_cov_pct:.1f}% is below minimum threshold of {min_aoi_coverage_percent:.1f}%.",
        )

    if sar_cov_pct < min_aoi_coverage_percent:
        return (
            False,
            None,
            f"SAR AOI coverage {sar_cov_pct:.1f}% is below minimum threshold of {min_aoi_coverage_percent:.1f}%.",
        )

    opt_bbox = extract_item_bbox(optical_item)
    sar_bbox = extract_item_bbox(sar_item)
    spatial_inter = calculate_spatial_intersection(opt_bbox, sar_bbox, aoi_bbox)

    if not spatial_inter["has_overlap"]:
        return False, None, "No geographic intersection between optical footprint, SAR footprint, and AOI."

    # 3. Polarization & Product Verification
    sar_assets = sar_item.get("assets", {})
    available_pols: List[str] = []
    if "vv" in sar_assets:
        available_pols.append("VV")
    if "vh" in sar_assets:
        available_pols.append("VH")

    # If assets are not populated, check STAC properties
    if not available_pols:
        prop_pols = sar_props.get("sar:polarizations", [])
        if isinstance(prop_pols, list):
            available_pols = [str(p).upper() for p in prop_pols if str(p).upper() in ("VV", "VH")]

    # Accept VV or VH (at least one valid SAR channel required)
    if not available_pols:
        return False, None, f"Candidate SAR scene {sar_item.get('id')} contains neither VV nor VH polarizations."

    has_dual_pol = ("VV" in available_pols) and ("VH" in available_pols)

    # 4. Multi-factor Scoring Strategy (Step 13G)
    # Optical cloud score (20%)
    cloud_cover = float(opt_props.get("eo:cloud_cover", 0.0))
    cloud_score = max(0.0, 1.0 - (cloud_cover / 100.0))

    # Optical coverage score (20%)
    opt_cov_score = min(1.0, opt_cov_pct / 100.0)

    # Temporal proximity score (35%)
    proximity_ratio = max(0.0, 1.0 - (temporal_delta_days / max(1e-5, max_temporal_delta_days)))

    # SAR coverage score (15%)
    sar_cov_score = min(1.0, sar_cov_pct / 100.0)

    # Polarization preference (7%)
    if has_dual_pol:
        pol_score = 1.0
    elif "VV" in available_pols:
        pol_score = 0.5 if prefer_dual_pol else 0.9
    else:
        pol_score = 0.3

    # Mode preference (3%)
    mode = str(sar_props.get("sar:instrument_mode", "IW")).upper()
    mode_score = 1.0 if mode == "IW" else 0.5

    composite_score = (
        0.20 * cloud_score
        + 0.20 * opt_cov_score
        + 0.35 * proximity_ratio
        + 0.15 * sar_cov_score
        + 0.07 * pol_score
        + 0.03 * mode_score
    )

    opt_id = optical_item.get("id", "unknown_s2")
    sar_id = sar_item.get("id", "unknown_s1")

    selection_reason = (
        f"Selected Optical {opt_id} ({cloud_cover:.1f}% cloud, {opt_cov_pct:.1f}% AOI coverage) "
        f"and SAR {sar_id} (temporal delta: {temporal_delta_days:.2f} days, polarizations: {available_pols}, "
        f"mode: {mode}, {sar_cov_pct:.1f}% AOI coverage) with composite score {composite_score:.4f}."
    )

    eval_result: Dict[str, Any] = {
        "score": composite_score,
        "temporal_delta_days": temporal_delta_days,
        "optical_item": optical_item,
        "sar_item": sar_item,
        "optical_datetime": opt_dt.isoformat(),
        "sar_datetime": sar_dt.isoformat(),
        "optical_cloud_cover": cloud_cover,
        "optical_coverage_percent": opt_cov_pct,
        "sar_coverage_percent": sar_cov_pct,
        "polarizations": available_pols,
        "sar_mode": mode,
        "spatial_overlap": spatial_inter,
        "selection_reason": selection_reason,
    }

    return True, eval_result, None


def rank_candidate_pairs(
    optical_candidates: List[Dict[str, Any]],
    sar_candidates: List[Dict[str, Any]],
    aoi_bbox: Tuple[float, float, float, float],
    max_temporal_delta_days: float = DEFAULT_MAX_TEMPORAL_DELTA_DAYS,
    prefer_dual_pol: bool = True,
    min_aoi_coverage_percent: float = DEFAULT_MIN_AOI_COVERAGE_PERCENT,
) -> List[Dict[str, Any]]:
    """
    Evaluate and deterministically rank all candidate Optical x SAR pairs.
    """
    evaluated_pairs: List[Dict[str, Any]] = []

    for opt in optical_candidates:
        for sar in sar_candidates:
            is_compat, eval_info, _ = evaluate_candidate_pair(
                optical_item=opt,
                sar_item=sar,
                aoi_bbox=aoi_bbox,
                max_temporal_delta_days=max_temporal_delta_days,
                prefer_dual_pol=prefer_dual_pol,
                min_aoi_coverage_percent=min_aoi_coverage_percent,
            )
            if is_compat and eval_info is not None:
                evaluated_pairs.append(eval_info)

    # Deterministic multi-key sorting:
    # 1. Primary: Descending composite score
    # 2. Secondary: Ascending temporal delta
    # 3. Tertiary: Ascending cloud cover
    # 4. Tie-breaker: Scene IDs
    evaluated_pairs.sort(
        key=lambda p: (
            -p["score"],
            p["temporal_delta_days"],
            p["optical_cloud_cover"],
            p["optical_item"].get("id", ""),
            p["sar_item"].get("id", ""),
        )
    )

    return evaluated_pairs


# ============================================================
# OPTICAL GEOTIFF COMPOSITE HELPER
# ============================================================

def create_optical_rgb_geotiff(
    band_paths: Dict[str, str],
    scene_id: str,
    bbox: Tuple[float, float, float, float],
    cache_dir: Path,
) -> str:
    """
    Create or retrieve a stacked 3-band (RGB) GeoTIFF from downloaded individual bands.
    Uses Red (B04), Green (B03), and Blue (B02) if present; falls back to Red/Green/NIR.
    """
    min_lng, min_lat, max_lng, max_lat = bbox
    aoi_key = f"{min_lng:.5f}_{min_lat:.5f}_{max_lng:.5f}_{max_lat:.5f}"
    aoi_hash = hashlib.md5(aoi_key.encode()).hexdigest()[:8]
    rgb_cache_name = f"s2_{scene_id}_rgb_{aoi_hash}.tif"
    rgb_cache_path = cache_dir / rgb_cache_name

    if rgb_cache_path.exists():
        try:
            with rasterio.open(rgb_cache_path) as src:
                if src.count >= 3 and src.width > 0 and src.height > 0:
                    return str(rgb_cache_path.resolve())
        except Exception:
            pass

    # Determine RGB band sources
    r_path = band_paths.get("red")
    g_path = band_paths.get("green")
    b_path = band_paths.get("blue") or band_paths.get("nir")

    if not r_path or not Path(r_path).exists():
        # Return whatever optical band is available
        for p in band_paths.values():
            if p and Path(p).exists():
                return str(Path(p).resolve())
        raise RuntimeError(f"No valid optical band files found for scene {scene_id}")

    if not g_path or not b_path or not Path(g_path).exists() or not Path(b_path).exists():
        # Return red band as single-band GeoTIFF
        return str(Path(r_path).resolve())

    with rasterio.open(r_path) as src_r, rasterio.open(g_path) as src_g, rasterio.open(b_path) as src_b:
        profile = src_r.profile.copy()
        profile.update(
            count=3,
            driver="GTiff",
            dtype=src_r.dtypes[0],
            compress="lzw",
        )

        arr_r = src_r.read(1)
        arr_g = src_g.read(1)
        arr_b = src_b.read(1)

        with rasterio.open(rgb_cache_path, "w", **profile) as dst:
            dst.write(arr_r, 1)
            dst.write(arr_g, 2)
            dst.write(arr_b, 3)
            dst.set_band_description(1, "Red (B04)")
            dst.set_band_description(2, "Green (B03)")
            dst.set_band_description(3, "Blue (B02)" if "blue" in band_paths else "NIR (B08)")

    return str(rgb_cache_path.resolve())


# ============================================================
# PRIMARY COORDINATION FUNCTION
# ============================================================

def find_optical_sar_pair(
    aoi: Any,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    target_date: Optional[Union[datetime, str]] = None,
    max_temporal_delta_days: float = DEFAULT_MAX_TEMPORAL_DELTA_DAYS,
    prefer_dual_polarization: bool = True,
    cloud_cover_limit: float = DEFAULT_CLOUD_COVER_LIMIT,
    min_aoi_coverage_percent: float = DEFAULT_MIN_AOI_COVERAGE_PERCENT,
    s2_provider: Optional[Sentinel2Provider] = None,
    s1_provider: Optional[Sentinel1Provider] = None,
    fetch_data: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Search, match, score, and retrieve the best compatible Sentinel-2 (optical)
    and Sentinel-1 (SAR) scene pair for the specified AOI and time period.

    Parameters:
        aoi: Area of Interest (bounding box, Leaflet coords, or GeoJSON geometry).
        time_start: Start date string or ISO interval (e.g. "2021-06-25").
        time_end: End date string (e.g. "2021-06-28").
        target_date: Optional specific target anchor datetime.
        max_temporal_delta_days: Maximum allowable temporal separation (default 3.0 days).
        prefer_dual_polarization: Whether to prefer dual-pol (VV+VH) over single-pol (VV-only).
        cloud_cover_limit: Initial cloud cover ceiling for Sentinel-2 candidate search.
        min_aoi_coverage_percent: Minimum acceptable percentage of AOI covered by each scene.
        s2_provider: Optional injected Sentinel2Provider instance.
        s1_provider: Optional injected Sentinel1Provider instance.
        fetch_data: When True, retrieves and caches local GeoTIFFs and verifies compatibility.

    Returns:
        Structured dict containing:
            - status: "REAL_SUCCESS" or "REAL_FAILURE"
            - pair_found: bool
            - temporal_delta_days: float
            - spatial_overlap: dict
            - selection_reason: str
            - optical: metadata dict + local path(s)
            - sar: metadata dict + local path(s)
            - errors: list of error strings
    """
    # 1. Normalize AOI
    try:
        bbox = normalize_aoi(aoi)
    except Exception as exc:
        return {
            "status": "REAL_FAILURE",
            "pair_found": False,
            "error_type": PairingErrorType.MALFORMED_AOI,
            "error": f"Failed to parse or normalize AOI: {exc}",
            "details": {"aoi": aoi},
            "temporal_delta_days": None,
            "spatial_overlap": None,
            "selection_reason": None,
            "optical": None,
            "sar": None,
            "errors": [f"Malformed AOI: {exc}"],
        }

    # Validate bbox coordinates
    w, s, e, n = bbox
    if w >= e or s >= n or not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90):
        return {
            "status": "REAL_FAILURE",
            "pair_found": False,
            "error_type": PairingErrorType.MALFORMED_AOI,
            "error": f"Invalid bounding box coordinates: {bbox}. Expected [west, south, east, north] with west < east and south < north.",
            "details": {"bbox": bbox},
            "temporal_delta_days": None,
            "spatial_overlap": None,
            "selection_reason": None,
            "optical": None,
            "sar": None,
            "errors": [f"Invalid bounding box: {bbox}"],
        }

    # 2. Build temporal query envelopes
    try:
        # Optical search envelope
        dt_optical_range = format_iso_range(time_start, time_end, buffer_days=0.0)
        # SAR search envelope expanded by max_temporal_delta_days to discover matching S1 scenes
        dt_sar_range = format_iso_range(time_start, time_end, buffer_days=max_temporal_delta_days)
    except Exception as exc:
        return {
            "status": "REAL_FAILURE",
            "pair_found": False,
            "error_type": PairingErrorType.INVALID_DATES,
            "error": f"Invalid date range parameters: {exc}",
            "details": {"time_start": time_start, "time_end": time_end},
            "temporal_delta_days": None,
            "spatial_overlap": None,
            "selection_reason": None,
            "optical": None,
            "sar": None,
            "errors": [f"Invalid dates: {exc}"],
        }

    # 3. Instantiate providers if not supplied
    provider_s2 = s2_provider or Sentinel2Provider()
    provider_s1 = s1_provider or Sentinel1Provider()

    # 4. Query candidate scenes (Step 13C & 13D)
    try:
        optical_candidates = provider_s2.search_candidate_scenes(
            bbox=bbox,
            datetime_range=dt_optical_range,
            cloud_cover_limit=cloud_cover_limit,
        )
    except Sentinel2RetrievalError as exc:
        return {
            "status": "REAL_FAILURE",
            "pair_found": False,
            "error_type": PairingErrorType.NO_OPTICAL_SCENES,
            "error": f"Sentinel-2 search returned no candidates: {exc.message}",
            "details": exc.details,
            "temporal_delta_days": None,
            "spatial_overlap": None,
            "selection_reason": None,
            "optical": None,
            "sar": None,
            "errors": [exc.message],
        }
    except Exception as exc:
        return {
            "status": "REAL_FAILURE",
            "pair_found": False,
            "error_type": PairingErrorType.NO_OPTICAL_SCENES,
            "error": f"Unexpected error during Sentinel-2 candidate search: {exc}",
            "details": {},
            "temporal_delta_days": None,
            "spatial_overlap": None,
            "selection_reason": None,
            "optical": None,
            "sar": None,
            "errors": [str(exc)],
        }

    if not optical_candidates:
        return {
            "status": "REAL_FAILURE",
            "pair_found": False,
            "error_type": PairingErrorType.NO_OPTICAL_SCENES,
            "error": f"No Sentinel-2 optical candidate scenes found for AOI {bbox} in date range '{dt_optical_range}'.",
            "details": {"bbox": bbox, "datetime_range": dt_optical_range},
            "temporal_delta_days": None,
            "spatial_overlap": None,
            "selection_reason": None,
            "optical": None,
            "sar": None,
            "errors": ["No Sentinel-2 candidate scenes found."],
        }

    try:
        sar_candidates = provider_s1.search_candidate_scenes(
            bbox=bbox,
            datetime_range=dt_sar_range,
        )
    except Sentinel1RetrievalError as exc:
        return {
            "status": "REAL_FAILURE",
            "pair_found": False,
            "error_type": PairingErrorType.NO_SAR_SCENES,
            "error": f"Sentinel-1 search returned no candidates: {exc.message}",
            "details": exc.details,
            "temporal_delta_days": None,
            "spatial_overlap": None,
            "selection_reason": None,
            "optical": None,
            "sar": None,
            "errors": [exc.message],
        }
    except Exception as exc:
        return {
            "status": "REAL_FAILURE",
            "pair_found": False,
            "error_type": PairingErrorType.NO_SAR_SCENES,
            "error": f"Unexpected error during Sentinel-1 candidate search: {exc}",
            "details": {},
            "temporal_delta_days": None,
            "spatial_overlap": None,
            "selection_reason": None,
            "optical": None,
            "sar": None,
            "errors": [str(exc)],
        }

    if not sar_candidates:
        return {
            "status": "REAL_FAILURE",
            "pair_found": False,
            "error_type": PairingErrorType.NO_SAR_SCENES,
            "error": f"No Sentinel-1 SAR candidate scenes found for AOI {bbox} in date range '{dt_sar_range}'.",
            "details": {"bbox": bbox, "datetime_range": dt_sar_range},
            "temporal_delta_days": None,
            "spatial_overlap": None,
            "selection_reason": None,
            "optical": None,
            "sar": None,
            "errors": ["No Sentinel-1 candidate scenes found."],
        }

    # 5. Evaluate and rank candidate pairs (Step 13E, 13F, 13G)
    ranked_pairs = rank_candidate_pairs(
        optical_candidates=optical_candidates,
        sar_candidates=sar_candidates,
        aoi_bbox=bbox,
        max_temporal_delta_days=max_temporal_delta_days,
        prefer_dual_pol=prefer_dual_polarization,
        min_aoi_coverage_percent=min_aoi_coverage_percent,
    )

    if not ranked_pairs:
        return {
            "status": "REAL_FAILURE",
            "pair_found": False,
            "error_type": PairingErrorType.NO_TEMPORALLY_COMPATIBLE_PAIR,
            "error": (
                f"No compatible Optical-SAR pair found for AOI {bbox} within "
                f"maximum temporal separation of {max_temporal_delta_days:.1f} days. "
                f"Evaluated {len(optical_candidates)} optical and {len(sar_candidates)} SAR candidates."
            ),
            "details": {
                "optical_candidates_count": len(optical_candidates),
                "sar_candidates_count": len(sar_candidates),
                "max_temporal_delta_days": max_temporal_delta_days,
            },
            "temporal_delta_days": None,
            "spatial_overlap": None,
            "selection_reason": None,
            "optical": None,
            "sar": None,
            "errors": ["No temporally and spatially compatible Optical-SAR pair found."],
        }

    best_pair = ranked_pairs[0]
    best_opt_item = best_pair["optical_item"]
    best_sar_item = best_pair["sar_item"]

    opt_id = best_opt_item.get("id", "unknown_s2")
    sar_id = best_sar_item.get("id", "unknown_s1")
    opt_props = best_opt_item.get("properties", {})
    sar_props = best_sar_item.get("properties", {})

    opt_path: Optional[str] = None
    opt_bands: Dict[str, str] = {}
    sar_path: Optional[str] = None
    sar_vv_path: Optional[str] = None
    sar_vh_path: Optional[str] = None

    # 6. Fetch data and verify rasters if requested (Step 13H & 13I)
    if fetch_data:
        try:
            # Fetch Sentinel-2 bands
            opt_bands = provider_s2.fetch_scene_bands(
                scene_item=best_opt_item,
                bbox=bbox,
            )
            # Create/get 3-band RGB GeoTIFF
            opt_path = create_optical_rgb_geotiff(
                band_paths=opt_bands,
                scene_id=opt_id,
                bbox=bbox,
                cache_dir=provider_s2.cache_dir,
            )
        except Exception as exc:
            return {
                "status": "REAL_FAILURE",
                "pair_found": False,
                "error_type": PairingErrorType.DOWNLOAD_FAILED,
                "error": f"Failed to retrieve or cache optical raster bands for scene {opt_id}: {exc}",
                "details": {"scene_id": opt_id},
                "temporal_delta_days": best_pair["temporal_delta_days"],
                "spatial_overlap": best_pair["spatial_overlap"],
                "selection_reason": best_pair["selection_reason"],
                "optical": None,
                "sar": None,
                "errors": [f"Optical download failed: {exc}"],
            }

        try:
            # Fetch Sentinel-1 polarizations
            sar_assets = best_sar_item.get("assets", {})
            if "vv" in sar_assets:
                sar_vv_path = provider_s1.fetch_and_cache_polarization(
                    scene_item=best_sar_item,
                    polarization="vv",
                    bbox=bbox,
                )
            if "vh" in sar_assets:
                sar_vh_path = provider_s1.fetch_and_cache_polarization(
                    scene_item=best_sar_item,
                    polarization="vh",
                    bbox=bbox,
                )

            sar_path = sar_vv_path or sar_vh_path
            if not sar_path:
                raise RuntimeError(f"No VV or VH polarization raster could be retrieved for SAR scene {sar_id}.")

        except Exception as exc:
            return {
                "status": "REAL_FAILURE",
                "pair_found": False,
                "error_type": PairingErrorType.DOWNLOAD_FAILED,
                "error": f"Failed to retrieve or cache SAR polarization raster for scene {sar_id}: {exc}",
                "details": {"scene_id": sar_id},
                "temporal_delta_days": best_pair["temporal_delta_days"],
                "spatial_overlap": best_pair["spatial_overlap"],
                "selection_reason": best_pair["selection_reason"],
                "optical": None,
                "sar": None,
                "errors": [f"SAR download failed: {exc}"],
            }

        # Validate GeoTIFF pair with existing validator (Step 13H)
        val_res = validate_optical_sar_pair(str(opt_path), str(sar_path))
        if not val_res.get("valid", False):
            return {
                "status": "REAL_FAILURE",
                "pair_found": False,
                "error_type": PairingErrorType.UNREADABLE_RASTER,
                "error": f"Acquired Optical-SAR raster pair failed validation: {val_res.get('errors')}",
                "details": val_res,
                "temporal_delta_days": best_pair["temporal_delta_days"],
                "spatial_overlap": best_pair["spatial_overlap"],
                "selection_reason": best_pair["selection_reason"],
                "optical": None,
                "sar": None,
                "errors": val_res.get("errors", []),
            }

    # 7. Construct and return successful metadata contract (Step 13J & 13M)
    opt_bounds = extract_item_bbox(best_opt_item)
    sar_bounds = extract_item_bbox(best_sar_item)

    optical_meta: Dict[str, Any] = {
        "item_id": opt_id,
        "acquisition_datetime": best_pair["optical_datetime"],
        "sensor": str(opt_props.get("platform", "Sentinel-2")),
        "product": "Level-2A",
        "cloud_cover": best_pair["optical_cloud_cover"],
        "coverage_percentage": best_pair["optical_coverage_percent"],
        "crs": "EPSG:4326",
        "bounds": opt_bounds,
        "path": opt_path,
        "bands": opt_bands,
    }

    sar_meta: Dict[str, Any] = {
        "item_id": sar_id,
        "acquisition_datetime": best_pair["sar_datetime"],
        "sensor": str(sar_props.get("platform", "Sentinel-1")),
        "product": str(sar_props.get("sar:product_type", "GRD")),
        "mode": str(sar_props.get("sar:instrument_mode", "IW")),
        "orbit_direction": str(sar_props.get("sat:orbit_state", "unknown")),
        "polarizations": best_pair["polarizations"],
        "coverage_percentage": best_pair["sar_coverage_percent"],
        "crs": "EPSG:4326",
        "bounds": sar_bounds,
        "path": sar_path,
        "vv": sar_vv_path,
        "vh": sar_vh_path,
    }

    return {
        "status": "REAL_SUCCESS",
        "pair_found": True,
        "temporal_delta_days": round(best_pair["temporal_delta_days"], 3),
        "spatial_overlap": best_pair["spatial_overlap"],
        "selection_reason": best_pair["selection_reason"],
        "optical": optical_meta,
        "sar": sar_meta,
        "candidate_pairs_evaluated": len(ranked_pairs),
        "errors": [],
    }
