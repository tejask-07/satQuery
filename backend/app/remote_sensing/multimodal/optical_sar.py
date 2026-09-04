"""
Optical-SAR GeoTIFF Pair Validator & Alignment Module.

Provides standalone, metadata-preserving validation and spatial alignment / co-registration
between Optical (e.g. Sentinel-2) and SAR (e.g. Sentinel-1) GeoTIFF rasters.
Does NOT perform physical SAR calibration (dB conversion) or VLM integration.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.warp import reproject, transform_bounds
from PIL import Image

from app.remote_sensing.io.raster import read_raster_metadata

logger = logging.getLogger(__name__)


def inspect_geotiff(
    path: str,
    label: str = "raster",
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Inspect a GeoTIFF and extract comprehensive metadata without reading pixel data.

    Returns:
        (metadata_dict, errors_list)
    """
    errors: List[str] = []
    p = Path(path)

    if not p.exists():
        errors.append(f"{label} file not found: {path}")
        return None, errors

    if not p.is_file():
        errors.append(f"{label} path is not a regular file: {path}")
        return None, errors

    try:
        raw_meta = read_raster_metadata(str(p))
    except (RasterioIOError, Exception) as exc:
        errors.append(f"Unreadable {label} raster file: {exc}")
        return None, errors

    # Check CRS
    crs_obj = raw_meta.get("crs")
    if crs_obj is None:
        errors.append(f"{label} raster missing coordinate reference system (CRS)")
        return None, errors

    try:
        crs_str = crs_obj.to_string() if hasattr(crs_obj, "to_string") else str(crs_obj)
        # Verify valid CRS instance
        _ = CRS.from_user_input(crs_obj)
    except Exception as exc:
        errors.append(f"{label} raster has invalid or unparseable CRS: {exc}")
        return None, errors

    # Check Bounds
    b = raw_meta.get("bounds")
    if b is None or any(coord is None or not math.isfinite(coord) for coord in (b.left, b.bottom, b.right, b.top)):
        errors.append(f"{label} raster has missing or non-finite bounds")
        return None, errors

    if b.left >= b.right or b.bottom >= b.top:
        errors.append(f"{label} raster has invalid bounds geometry (left >= right or bottom >= top)")
        return None, errors

    # Check Dimensions
    width = raw_meta.get("width", 0)
    height = raw_meta.get("height", 0)
    if width <= 0 or height <= 0:
        errors.append(f"{label} raster has invalid dimensions: width={width}, height={height}")
        return None, errors

    # Resolution
    res = raw_meta.get("resolution")
    res_list = [float(res[0]), float(res[1])] if res else [0.0, 0.0]

    # Transform
    transform = raw_meta.get("transform")
    transform_list = list(transform) if transform is not None else []

    # Band count & dtypes
    band_count = raw_meta.get("count", 1)
    dtypes = raw_meta.get("dtypes", ())
    dtype_str = str(dtypes[0]) if dtypes else "unknown"

    # Band names / descriptions
    descriptions = raw_meta.get("descriptions") or ()
    band_names: List[str] = []
    for i in range(band_count):
        if i < len(descriptions) and descriptions[i]:
            band_names.append(str(descriptions[i]))
        else:
            band_names.append(f"band_{i+1}")

    info: Dict[str, Any] = {
        "path": str(p.resolve()),
        "crs": crs_str,
        "bounds": [float(b.left), float(b.bottom), float(b.right), float(b.top)],
        "width": int(width),
        "height": int(height),
        "resolution": res_list,
        "transform": transform_list,
        "band_count": int(band_count),
        "dtype": dtype_str,
        "nodata": raw_meta.get("nodata"),
        "band_names": band_names,
    }

    return info, errors


def check_spatial_overlap(
    optical_info: Dict[str, Any],
    sar_info: Dict[str, Any],
) -> Tuple[bool, bool]:
    """
    Determine if CRS matches and if there is geographic/spatial overlap.

    Returns:
        (crs_match, spatial_overlap)
    """
    opt_crs = CRS.from_user_input(optical_info["crs"])
    sar_crs = CRS.from_user_input(sar_info["crs"])

    crs_match = (opt_crs == sar_crs)

    opt_b = optical_info["bounds"]  # [left, bottom, right, top]
    sar_b = sar_info["bounds"]

    if crs_match:
        # Same CRS: direct bounding box intersection
        inter_left = max(opt_b[0], sar_b[0])
        inter_bottom = max(opt_b[1], sar_b[1])
        inter_right = min(opt_b[2], sar_b[2])
        inter_top = min(opt_b[3], sar_b[3])
        overlap = (inter_right > inter_left) and (inter_top > inter_bottom)
        return True, overlap

    # Different CRS: reproject SAR bounds into Optical CRS
    try:
        sar_in_opt = transform_bounds(sar_crs, opt_crs, sar_b[0], sar_b[1], sar_b[2], sar_b[3], densify_pts=21)
        inter_left = max(opt_b[0], sar_in_opt[0])
        inter_bottom = max(opt_b[1], sar_in_opt[1])
        inter_right = min(opt_b[2], sar_in_opt[2])
        inter_top = min(opt_b[3], sar_in_opt[3])
        overlap = (inter_right > inter_left) and (inter_top > inter_bottom)
        if overlap:
            return False, True
    except Exception:
        pass

    # Double check in WGS84 (EPSG:4326)
    try:
        wgs84 = CRS.from_epsg(4326)
        opt_wgs = transform_bounds(opt_crs, wgs84, opt_b[0], opt_b[1], opt_b[2], opt_b[3], densify_pts=21)
        sar_wgs = transform_bounds(sar_crs, wgs84, sar_b[0], sar_b[1], sar_b[2], sar_b[3], densify_pts=21)
        inter_left = max(opt_wgs[0], sar_wgs[0])
        inter_bottom = max(opt_wgs[1], sar_wgs[1])
        inter_right = min(opt_wgs[2], sar_wgs[2])
        inter_top = min(opt_wgs[3], sar_wgs[3])
        overlap = (inter_right > inter_left) and (inter_top > inter_bottom)
        return False, overlap
    except Exception:
        return False, False


def check_resolution_compatible(
    optical_info: Dict[str, Any],
    sar_info: Dict[str, Any],
) -> bool:
    """
    Check whether optical and SAR pixel resolutions are within a compatible scale factor.
    """
    opt_res = optical_info.get("resolution", [0.0, 0.0])
    sar_res = sar_info.get("resolution", [0.0, 0.0])

    if not (math.isfinite(opt_res[0]) and math.isfinite(opt_res[1]) and opt_res[0] > 0 and opt_res[1] > 0):
        return False
    if not (math.isfinite(sar_res[0]) and math.isfinite(sar_res[1]) and sar_res[0] > 0 and sar_res[1] > 0):
        return False

    opt_crs = CRS.from_user_input(optical_info["crs"])
    sar_crs = CRS.from_user_input(sar_info["crs"])

    # If both are in the same unit system (e.g. both projected meters or both degrees)
    if opt_crs.is_projected == sar_crs.is_projected:
        ratio_x = opt_res[0] / sar_res[0]
        ratio_y = opt_res[1] / sar_res[1]
        # Allow resolutions within 0.05x to 20x (e.g. 10m vs 20m or 10m vs 60m)
        return (0.05 <= ratio_x <= 20.0) and (0.05 <= ratio_y <= 20.0)

    # Cross-unit comparison: one is degrees (e.g. ~0.0001 deg) and one is projected (e.g. ~10m)
    # 1 degree latitude ~ 111,320m
    def to_approx_meters(res: List[float], crs: CRS, bounds: List[float]) -> Tuple[float, float]:
        if crs.is_projected:
            return float(res[0]), float(res[1])
        # Geographic degrees: approximate meter scale based on center latitude
        lat_center = (bounds[1] + bounds[3]) / 2.0
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_center))
        return float(res[0] * m_per_deg_lon), float(res[1] * m_per_deg_lat)

    opt_m_x, opt_m_y = to_approx_meters(opt_res, opt_crs, optical_info["bounds"])
    sar_m_x, sar_m_y = to_approx_meters(sar_res, sar_crs, sar_info["bounds"])

    if opt_m_x <= 0 or opt_m_y <= 0 or sar_m_x <= 0 or sar_m_y <= 0:
        return False

    ratio_x = opt_m_x / sar_m_x
    ratio_y = opt_m_y / sar_m_y
    return (0.05 <= ratio_x <= 20.0) and (0.05 <= ratio_y <= 20.0)


def check_alignment_required(
    crs_match: bool,
    optical_info: Dict[str, Any],
    sar_info: Dict[str, Any],
) -> bool:
    """
    Determine if reprojection or spatial alignment is required between optical and SAR rasters.
    """
    if not crs_match:
        return True

    if optical_info["width"] != sar_info["width"] or optical_info["height"] != sar_info["height"]:
        return True

    # Check bounds tolerance
    for b_opt, b_sar in zip(optical_info["bounds"], sar_info["bounds"]):
        if abs(b_opt - b_sar) > 1e-5:
            return True

    # Check transform tolerance
    opt_tf = optical_info.get("transform", [])
    sar_tf = sar_info.get("transform", [])
    if len(opt_tf) != len(sar_tf) or not opt_tf:
        return True

    for t_opt, t_sar in zip(opt_tf, sar_tf):
        if abs(t_opt - t_sar) > 1e-6:
            return True

    return False


def validate_optical_sar_pair(
    optical_path: str,
    sar_path: str,
) -> Dict[str, Any]:
    """
    Validate an Optical and SAR GeoTIFF pair for multimodal processing.

    Collects metadata, verifies spatial overlap, checks resolution compatibility,
    and determines whether reprojection/alignment will be required.

    Parameters:
        optical_path: Local filesystem path to Optical GeoTIFF.
        sar_path: Local filesystem path to SAR GeoTIFF.

    Returns:
        Structured dict with validation status, metadata for optical and sar,
        compatibility flags, and any accumulated errors.
    """
    errors: List[str] = []

    optical_info, opt_errors = inspect_geotiff(optical_path, label="Optical")
    errors.extend(opt_errors)

    sar_info, sar_errors = inspect_geotiff(sar_path, label="SAR")
    errors.extend(sar_errors)

    if optical_info is None or sar_info is None:
        return {
            "valid": False,
            "optical": optical_info,
            "sar": sar_info,
            "compatibility": {
                "crs_match": False,
                "spatial_overlap": False,
                "resolution_compatible": False,
                "alignment_required": False,
            },
            "errors": errors,
        }

    # Compatibility checks
    crs_match, spatial_overlap = check_spatial_overlap(optical_info, sar_info)
    resolution_compatible = check_resolution_compatible(optical_info, sar_info)
    alignment_required = check_alignment_required(crs_match, optical_info, sar_info)

    if not spatial_overlap:
        errors.append(
            "No spatial overlap detected between optical and SAR imagery."
        )

    is_valid = (len(errors) == 0) and spatial_overlap

    return {
        "valid": is_valid,
        "optical": optical_info,
        "sar": sar_info,
        "compatibility": {
            "crs_match": crs_match,
            "spatial_overlap": spatial_overlap,
            "resolution_compatible": resolution_compatible,
            "alignment_required": alignment_required,
        },
        "errors": errors,
    }


def align_optical_sar_pair(
    optical_path: str,
    sar_path: str,
    resampling: str = "bilinear",
    sar_vh_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Spatially align and co-register a SAR GeoTIFF onto an Optical reference grid.

    The Optical raster serves as the authoritative REFERENCE GRID.
    The SAR data (VV, VH, or multi-band) is reprojected and resampled to match
    the Optical grid's CRS, affine transform, dimensions (height, width), and extent.

    Parameters:
        optical_path: Local filesystem path to Optical GeoTIFF (reference grid).
        sar_path: Local filesystem path to SAR GeoTIFF (source to align).
        resampling: Resampling algorithm for continuous SAR values ('bilinear', 'cubic', 'nearest').
                    Default is 'bilinear'.
        sar_vh_path: Optional local filesystem path to companion VH SAR GeoTIFF.

    Returns:
        Structured dict with:
            - success: bool
            - optical: dict with 'data' array and 'metadata'
            - sar: dict with 'vv' array, 'vh' array, 'data' (all bands), and 'metadata'
            - alignment: dict with alignment audit details
            - valid_mask: 2D boolean array representing joint validity across modalities
            - errors: list of error strings if any
    """
    # 1. Step 2 Pre-validation
    validation = validate_optical_sar_pair(optical_path, sar_path)
    if not validation["valid"]:
        return {
            "success": False,
            "optical": None,
            "sar": None,
            "alignment": None,
            "valid_mask": None,
            "errors": validation.get("errors", ["Optical-SAR validation failed."]),
        }

    alignment_required = validation["compatibility"]["alignment_required"]

    # Map resampling string to rasterio enum
    resampling_lower = str(resampling).lower()
    if resampling_lower == "bilinear":
        resampling_enum = Resampling.bilinear
    elif resampling_lower == "cubic":
        resampling_enum = Resampling.cubic
    elif resampling_lower == "nearest":
        resampling_enum = Resampling.nearest
    else:
        resampling_enum = Resampling.bilinear

    # 2. Ingest Optical Reference Grid
    with rasterio.open(optical_path) as opt_src:
        opt_data = opt_src.read()  # shape: (count, height, width)
        opt_crs = opt_src.crs
        opt_transform = opt_src.transform
        target_width = opt_src.width
        target_height = opt_src.height
        opt_nodata = opt_src.nodata
        opt_descriptions = list(opt_src.descriptions or [])
        opt_dtypes = opt_src.dtypes
        opt_bounds = opt_src.bounds
        opt_res = opt_src.res

    optical_metadata: Dict[str, Any] = {
        "crs": opt_crs.to_string() if hasattr(opt_crs, "to_string") else str(opt_crs),
        "transform": list(opt_transform),
        "width": target_width,
        "height": target_height,
        "bounds": [float(opt_bounds.left), float(opt_bounds.bottom), float(opt_bounds.right), float(opt_bounds.top)],
        "resolution": [float(opt_res[0]), float(opt_res[1])],
        "band_count": opt_src.count,
        "dtypes": [str(d) for d in opt_dtypes],
        "nodata": opt_nodata,
        "band_names": [
            opt_descriptions[i] if (i < len(opt_descriptions) and opt_descriptions[i]) else f"band_{i+1}"
            for i in range(opt_src.count)
        ],
    }

    # 3. Ingest and Align SAR
    with rasterio.open(sar_path) as sar_src:
        sar_crs = sar_src.crs
        sar_transform = sar_src.transform
        sar_nodata = sar_src.nodata
        sar_count = sar_src.count
        sar_descriptions = list(sar_src.descriptions or [])

        # Choose destination nodata (use SAR nodata if finite; otherwise np.nan)
        dst_nodata = float(sar_nodata) if sar_nodata is not None and math.isfinite(sar_nodata) else np.nan

        if not alignment_required:
            aligned_sar = sar_src.read().astype(np.float32)
            reprojected = False
        else:
            aligned_sar = np.full((sar_count, target_height, target_width), dst_nodata, dtype=np.float32)
            for b in range(1, sar_count + 1):
                src_band = sar_src.read(b).astype(np.float32)
                reproject(
                    source=src_band,
                    destination=aligned_sar[b - 1],
                    src_transform=sar_transform,
                    src_crs=sar_crs,
                    dst_transform=opt_transform,
                    dst_crs=opt_crs,
                    resampling=resampling_enum,
                    src_nodata=sar_nodata,
                    dst_nodata=dst_nodata,
                )
            reprojected = True

    # 4. Identify VV and VH bands
    sar_band_names: List[str] = []
    for i in range(sar_count):
        desc = sar_descriptions[i] if (i < len(sar_descriptions) and sar_descriptions[i]) else None
        sar_band_names.append(str(desc).strip() if desc else f"band_{i+1}")

    vv_band: Optional[np.ndarray] = None
    vh_band: Optional[np.ndarray] = None

    for i, name in enumerate(sar_band_names):
        name_upper = name.upper()
        if "VV" in name_upper and vv_band is None:
            vv_band = aligned_sar[i]
        elif "VH" in name_upper and vh_band is None:
            vh_band = aligned_sar[i]

    # Fallback to filename or convention if not discovered in band descriptions
    if vv_band is None and vh_band is None:
        p_upper = Path(sar_path).name.upper()
        if "_VV" in p_upper or "VV" in p_upper:
            vv_band = aligned_sar[0]
        elif "_VH" in p_upper or "VH" in p_upper:
            vh_band = aligned_sar[0]
        elif sar_count >= 2:
            vv_band = aligned_sar[0]
            vh_band = aligned_sar[1]
        else:
            vv_band = aligned_sar[0]
    elif vv_band is None and sar_count == 1:
        vv_band = aligned_sar[0]

    # Auto-detect or align companion VH band if vh_band is None
    if vh_band is None:
        target_vh = sar_vh_path
        if not target_vh:
            sar_p = Path(sar_path)
            s_name = sar_p.name
            if "_vv.tif" in s_name.lower():
                vh_name = re.sub(r"_vv\.tif$", "_vh.tif", s_name, flags=re.IGNORECASE)
                cand = sar_p.parent / vh_name
                if cand.is_file():
                    target_vh = str(cand)
            elif "_vv" in s_name.lower():
                vh_name = re.sub(r"_vv", "_vh", s_name, flags=re.IGNORECASE)
                cand = sar_p.parent / vh_name
                if cand.is_file():
                    target_vh = str(cand)

        if target_vh and Path(target_vh).is_file():
            try:
                with rasterio.open(target_vh) as vh_src:
                    vh_crs = vh_src.crs
                    vh_tf = vh_src.transform
                    vh_nodata = vh_src.nodata
                    vh_raw = vh_src.read(1).astype(np.float32)
                    dst_vh_nodata = float(vh_nodata) if vh_nodata is not None and math.isfinite(vh_nodata) else np.nan

                    if not alignment_required and vh_src.width == target_width and vh_src.height == target_height:
                        vh_band = vh_raw
                    else:
                        aligned_vh = np.full((target_height, target_width), dst_vh_nodata, dtype=np.float32)
                        reproject(
                            source=vh_raw,
                            destination=aligned_vh,
                            src_transform=vh_tf,
                            src_crs=vh_crs,
                            dst_transform=opt_transform,
                            dst_crs=opt_crs,
                            resampling=resampling_enum,
                            src_nodata=vh_nodata,
                            dst_nodata=dst_vh_nodata,
                        )
                        vh_band = aligned_vh
            except Exception as vh_err:
                logger.warning(f"[OPTICAL-SAR ALIGN] Could not align companion VH raster {target_vh}: {vh_err}")

    # 5. Compute Validity Masks (Optical, SAR, and Joint)
    # Optical validity: all optical bands are finite and != nodata
    opt_valid = np.ones((target_height, target_width), dtype=bool)
    for b in range(opt_data.shape[0]):
        band = opt_data[b]
        b_finite = np.isfinite(band)
        if opt_nodata is not None and math.isfinite(opt_nodata):
            b_finite = b_finite & (band != opt_nodata)
        opt_valid = opt_valid & b_finite

    # SAR validity: all aligned SAR bands are finite and != nodata
    sar_valid = np.ones((target_height, target_width), dtype=bool)
    for b in range(aligned_sar.shape[0]):
        band = aligned_sar[b]
        b_finite = np.isfinite(band)
        if math.isfinite(dst_nodata):
            b_finite = b_finite & (band != dst_nodata)
        sar_valid = sar_valid & b_finite

    if vh_band is not None and sar_count == 1:
        vh_finite = np.isfinite(vh_band)
        if math.isfinite(dst_nodata):
            vh_finite = vh_finite & (vh_band != dst_nodata)
        sar_valid = sar_valid & vh_finite

    # Joint valid mask: finite in both optical and all SAR bands
    joint_valid_mask = opt_valid & sar_valid

    sar_band_count = sar_count
    effective_sar_names = list(sar_band_names)
    if vh_band is not None and sar_count == 1:
        sar_band_count = 2
        effective_sar_names = ["VV", "VH"]

    sar_metadata: Dict[str, Any] = {
        "crs": optical_metadata["crs"],
        "transform": optical_metadata["transform"],
        "width": target_width,
        "height": target_height,
        "bounds": optical_metadata["bounds"],
        "resolution": optical_metadata["resolution"],
        "band_count": sar_band_count,
        "dtypes": ["float32"] * sar_band_count,
        "nodata": dst_nodata if reprojected else sar_nodata,
        "band_names": effective_sar_names,
    }

    return {
        "success": True,
        "optical": {
            "data": opt_data,
            "metadata": optical_metadata,
        },
        "sar": {
            "vv": vv_band,
            "vh": vh_band,
            "data": aligned_sar,
            "metadata": sar_metadata,
        },
        "alignment": {
            "reference": "optical",
            "target_crs": optical_metadata["crs"],
            "target_width": target_width,
            "target_height": target_height,
            "target_transform": optical_metadata["transform"],
            "same_grid": True,
            "reprojected": reprojected,
            "resampling": resampling_lower,
        },
        "valid_mask": joint_valid_mask,
        "errors": [],
    }


def normalize_band_visual(
    band: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """
    Robustly normalize a 2D raster band to uint8 [0, 255] for visual rendering.

    Excludes NaNs, non-finite values, and invalid/nodata pixels from percentile calculation.
    If the band has constant value or valid range is degenerate (hi <= lo),
    returns an array of zeros without dividing by zero or crashing.
    Invalid / masked pixels are forced to 0 (neutral black).
    Does NOT modify the input array in place.
    """
    arr = np.asarray(band, dtype=np.float32)

    # Determine finite values
    finite_mask = np.isfinite(arr)

    # Combine with valid_mask if provided
    if valid_mask is not None:
        effective_mask = finite_mask & np.asarray(valid_mask, dtype=bool)
    else:
        effective_mask = finite_mask

    if not np.any(effective_mask):
        return np.zeros(arr.shape, dtype=np.uint8)

    vals = arr[effective_mask]
    lo = float(np.percentile(vals, low_percentile))
    hi = float(np.percentile(vals, high_percentile))

    if hi <= lo or abs(hi - lo) < 1e-7:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = np.zeros(arr.shape, dtype=np.float32)
    norm_vals = (arr[effective_mask] - lo) / (hi - lo)
    scaled[effective_mask] = np.clip(norm_vals, 0.0, 1.0) * 255.0
    scaled[~effective_mask] = 0.0

    return scaled.astype(np.uint8)


def _detect_optical_rgb_channels(
    band_names: List[str],
    band_count: int,
) -> Tuple[int, int, int, bool, str, List[str]]:
    """
    Identify band indices for (R, G, B) display channels and determine if false-color.

    Returns:
        (r_idx, g_idx, b_idx, is_false_color, description, channels_used)
    """
    red_idx = None
    green_idx = None
    blue_idx = None
    nir_idx = None

    for i in range(band_count):
        raw_name = band_names[i] if i < len(band_names) else ""
        n = str(raw_name).strip().upper()
        # Red candidate: RED, B04, B4, BAND_4, B_RED
        if n in ("RED", "B04", "B4", "BAND_4", "B_RED", "RED_BAND") or (n.startswith("RED") and len(n) <= 6):
            if red_idx is None:
                red_idx = i
        # Green candidate: GREEN, B03, B3, BAND_3, B_GREEN
        elif n in ("GREEN", "B03", "B3", "BAND_3", "B_GREEN", "GREEN_BAND") or (n.startswith("GREEN") and len(n) <= 8):
            if green_idx is None:
                green_idx = i
        # Blue candidate: BLUE, B02, B2, BAND_2, B_BLUE
        elif n in ("BLUE", "B02", "B2", "BAND_2", "B_BLUE", "BLUE_BAND") or (n.startswith("BLUE") and len(n) <= 7):
            if blue_idx is None:
                blue_idx = i
        # NIR candidate: NIR, B08, B8, BAND_8, B_NIR, NEAR_INFRARED
        elif n in ("NIR", "B08", "B8", "BAND_8", "B_NIR", "NEAR_INFRARED") or "NIR" in n:
            if nir_idx is None:
                nir_idx = i

    # True-color RGB: all three Red, Green, Blue are identified
    if red_idx is not None and green_idx is not None and blue_idx is not None:
        r_name = band_names[red_idx] if red_idx < len(band_names) else f"band_{red_idx+1}"
        g_name = band_names[green_idx] if green_idx < len(band_names) else f"band_{green_idx+1}"
        b_name = band_names[blue_idx] if blue_idx < len(band_names) else f"band_{blue_idx+1}"
        desc = f"True-color RGB (R={r_name}, G={g_name}, B={b_name})"
        return red_idx, green_idx, blue_idx, False, desc, [r_name, g_name, b_name]

    # False-color NIR composite: Red, Green, and NIR are present
    if red_idx is not None and green_idx is not None and nir_idx is not None:
        r_name = band_names[red_idx] if red_idx < len(band_names) else f"band_{red_idx+1}"
        g_name = band_names[green_idx] if green_idx < len(band_names) else f"band_{green_idx+1}"
        b_name = band_names[nir_idx] if nir_idx < len(band_names) else f"band_{nir_idx+1}"
        desc = f"False-color NIR composite (R={r_name}, G={g_name}, B={b_name} [NIR])"
        return red_idx, green_idx, nir_idx, True, desc, [r_name, g_name, b_name]

    # Generic 3+ bands: fallback to first 3 bands
    if band_count >= 3:
        b0_name = band_names[0] if len(band_names) > 0 else "band_1"
        b1_name = band_names[1] if len(band_names) > 1 else "band_2"
        b2_name = band_names[2] if len(band_names) > 2 else "band_3"
        desc = f"False-color band-sequence fallback (R={b0_name}, G={b1_name}, B={b2_name})"
        return 0, 1, 2, True, desc, [b0_name, b1_name, b2_name]

    # 2 bands: R=band_0, G=band_1, B=blank
    if band_count == 2:
        b0_name = band_names[0] if len(band_names) > 0 else "band_1"
        b1_name = band_names[1] if len(band_names) > 1 else "band_2"
        desc = f"False-color 2-band composite (R={b0_name}, G={b1_name}, B=0)"
        return 0, 1, -1, True, desc, [b0_name, b1_name, "none"]

    # 1 band: replicate single band across RGB
    b0_name = band_names[0] if len(band_names) > 0 else "band_1"
    desc = f"Single-band grayscale ({b0_name}) rendered in RGB mode"
    return 0, 0, 0, True, desc, [b0_name]


def build_optical_sar_visuals(
    aligned_result: Dict[str, Any],
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> Dict[str, Any]:
    """
    Convert aligned Optical and SAR raster arrays into displayable, VLM-ready visual representations.

    Parameters:
        aligned_result: Output dict from align_optical_sar_pair(...) containing:
            - optical: {'data', 'metadata'}
            - sar: {'vv', 'vh', 'data', 'metadata'}
            - valid_mask: 2D boolean array
            - alignment: {'target_width', 'target_height', 'target_crs', ...}
        low_percentile: Lower clipping percentile for contrast stretching (default 2.0).
        high_percentile: Upper clipping percentile for contrast stretching (default 98.0).

    Returns:
        Structured dict with:
            - optical: {'image': PIL.Image, 'mode': 'rgb', 'is_false_color': bool, 'description': str, 'bands_used': list}
            - s1_vv: {'image': PIL.Image | None, 'mode': 'L' | None, 'description': str}
            - s1_vh: {'image': PIL.Image | None, 'mode': 'L' | None, 'description': str}
            - s1_composite: {'image': PIL.Image | None, 'mode': 'rgb' | None, 'description': str}
            - metadata: {'width': int, 'height': int, 'crs': str, 'reference': 'optical', ...}
    """
    if not isinstance(aligned_result, dict) or not aligned_result.get("success", False):
        err_msg = "Cannot build visuals: aligned_result is invalid or alignment was unsuccessful."
        if isinstance(aligned_result, dict) and aligned_result.get("errors"):
            err_msg += f" Errors: {aligned_result['errors']}"
        raise ValueError(err_msg)

    optical_info = aligned_result.get("optical") or {}
    opt_data = optical_info.get("data")
    opt_meta = optical_info.get("metadata") or {}

    if opt_data is None:
        raise ValueError("Missing optical data in aligned_result")

    sar_info = aligned_result.get("sar") or {}
    sar_vv = sar_info.get("vv")
    sar_vh = sar_info.get("vh")

    valid_mask = aligned_result.get("valid_mask")
    alignment_meta = aligned_result.get("alignment") or {}

    target_width = int(alignment_meta.get("target_width", opt_meta.get("width", opt_data.shape[-1])))
    target_height = int(alignment_meta.get("target_height", opt_meta.get("height", opt_data.shape[-2])))
    target_crs = str(alignment_meta.get("target_crs", opt_meta.get("crs", "")))

    # Ensure valid_mask is 2D boolean array matching target dimensions if present
    if valid_mask is not None:
        valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    else:
        valid_mask_arr = None

    # 1. OPTICAL VISUALIZATION
    band_count = opt_data.shape[0]
    band_names = opt_meta.get("band_names", [])
    r_idx, g_idx, b_idx, is_false_color, opt_desc, bands_used = _detect_optical_rgb_channels(band_names, band_count)

    r_norm = (
        normalize_band_visual(opt_data[r_idx], valid_mask=valid_mask_arr, low_percentile=low_percentile, high_percentile=high_percentile)
        if r_idx >= 0
        else np.zeros((target_height, target_width), dtype=np.uint8)
    )
    g_norm = (
        normalize_band_visual(opt_data[g_idx], valid_mask=valid_mask_arr, low_percentile=low_percentile, high_percentile=high_percentile)
        if g_idx >= 0
        else np.zeros((target_height, target_width), dtype=np.uint8)
    )
    b_norm = (
        normalize_band_visual(opt_data[b_idx], valid_mask=valid_mask_arr, low_percentile=low_percentile, high_percentile=high_percentile)
        if b_idx >= 0
        else np.zeros((target_height, target_width), dtype=np.uint8)
    )

    opt_rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
    opt_image = Image.fromarray(opt_rgb, mode="RGB")

    optical_output = {
        "image": opt_image,
        "mode": "rgb",
        "is_false_color": is_false_color,
        "description": opt_desc,
        "bands_used": bands_used,
    }

    # 2. SAR VV VISUALIZATION
    if sar_vv is not None:
        vv_norm = normalize_band_visual(sar_vv, valid_mask=valid_mask_arr, low_percentile=low_percentile, high_percentile=high_percentile)
        vv_image = Image.fromarray(vv_norm, mode="L")
        s1_vv_output = {
            "image": vv_image,
            "mode": "L",
            "description": "Sentinel-1 VV polarization grayscale radar backscatter",
        }
    else:
        vv_norm = None
        s1_vv_output = {
            "image": None,
            "mode": None,
            "description": "VV band not available in SAR data",
        }

    # 3. SAR VH VISUALIZATION
    if sar_vh is not None:
        vh_norm = normalize_band_visual(sar_vh, valid_mask=valid_mask_arr, low_percentile=low_percentile, high_percentile=high_percentile)
        vh_image = Image.fromarray(vh_norm, mode="L")
        s1_vh_output = {
            "image": vh_image,
            "mode": "L",
            "description": "Sentinel-1 VH cross-polarization grayscale radar backscatter",
        }
    else:
        vh_norm = None
        s1_vh_output = {
            "image": None,
            "mode": None,
            "description": "VH band not available in SAR data",
        }

    # 4. SAR DUAL-POLARIZATION COMPOSITE
    if vv_norm is not None and vh_norm is not None:
        # Derived polarization contrast: |VV_norm - VH_norm|
        contrast = np.clip(np.abs(vv_norm.astype(np.float32) - vh_norm.astype(np.float32)), 0.0, 255.0).astype(np.uint8)
        if valid_mask_arr is not None:
            contrast[~valid_mask_arr] = 0

        composite_rgb = np.stack([vv_norm, vh_norm, contrast], axis=-1)
        composite_image = Image.fromarray(composite_rgb, mode="RGB")
        s1_composite_output = {
            "image": composite_image,
            "mode": "rgb",
            "description": "VV/VH dual-polarization radar visualization (R=normalized VV, G=normalized VH, B=|normalized VV - normalized VH| polarization contrast)",
        }
    else:
        s1_composite_output = {
            "image": None,
            "mode": None,
            "description": "Dual-polarization composite requires both VV and VH bands",
        }

    # 5. METADATA
    valid_count = int(np.count_nonzero(valid_mask_arr)) if valid_mask_arr is not None else target_width * target_height
    total_count = target_width * target_height

    metadata = {
        "width": target_width,
        "height": target_height,
        "crs": target_crs,
        "reference": "optical",
        "valid_pixel_count": valid_count,
        "total_pixel_count": total_count,
        "valid_fraction": float(valid_count / total_count) if total_count > 0 else 0.0,
    }

    return {
        "optical": optical_output,
        "s1_vv": s1_vv_output,
        "s1_vh": s1_vh_output,
        "s1_composite": s1_composite_output,
        "metadata": metadata,
    }

