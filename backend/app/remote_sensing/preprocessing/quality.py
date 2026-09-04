from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional, Tuple
import numpy as np


# ============================================================
# SENTINEL-2 L2A SCENE CLASSIFICATION LAYER (SCL) CODES
# ============================================================

class SCLClass(IntEnum):
    """
    Standard ESA Sentinel-2 Level-2A Scene Classification Layer (SCL) classes.
    """
    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    DARK_AREA_PIXELS = 2             # Cast shadows, topographic shadows
    CLOUD_SHADOWS = 3                # Cloud shadows
    VEGETATION = 4                   # Healthy & sparse vegetation
    NOT_VEGETATED = 5                # Bare soil, rock, urban / built-up
    WATER = 6                        # Water bodies (ocean, lakes, rivers)
    UNCLASSIFIED = 7                 # Unclassified ground surfaces
    CLOUD_MEDIUM_PROBABILITY = 8     # Medium probability cloud
    CLOUD_HIGH_PROBABILITY = 9       # High probability cloud
    THIN_CIRRUS = 10                 # Thin cirrus cloud
    SNOW_ICE = 11                    # Snow or ice surfaces


# ============================================================
# QUALITY MASKS COMPUTATION
# ============================================================

def compute_quality_masks(
    scl_raster: Optional[np.ndarray] = None,
    band_data: Optional[np.ndarray] = None,
    shape: Optional[Tuple[int, int]] = None,
    reflectance_min: float = 0.0,
    reflectance_max: float = 1.5,
) -> Dict[str, np.ndarray]:
    """
    Generate authoritative scientific quality masks from Sentinel-2 SCL and band data.

    Parameters
    ----------
    scl_raster : np.ndarray, optional
        2D integer array of Sentinel-2 Scene Classification Layer (SCL) values.
    band_data : np.ndarray, optional
        2D or 3D float array of surface reflectance values to check for finite/range validity.
    shape : tuple of int, optional
        Target shape (height, width) if neither scl_raster nor band_data is loaded.
    reflectance_min : float
        Minimum physically plausible surface reflectance (default: 0.0).
    reflectance_max : float
        Maximum physically plausible surface reflectance (default: 1.5).

    Returns
    -------
    dict
        Dictionary containing boolean masks (True indicates condition is met):
        - 'nodata_mask': Pixels with missing/defective data
        - 'cloud_mask': Pixels classified as medium or high probability cloud
        - 'cirrus_mask': Pixels classified as thin cirrus
        - 'shadow_mask': Pixels classified as cloud shadow or dark shadow
        - 'reflectance_invalid_mask': Pixels outside valid physical reflectance range
        - 'valid_mask': Analysis-ready pixels (clean, cloud-free, shadow-free, valid)
    """
    # Determine target shape
    if scl_raster is not None:
        target_shape = scl_raster.shape
    elif band_data is not None:
        target_shape = band_data.shape if band_data.ndim == 2 else band_data.shape[-2:]
    elif shape is not None:
        target_shape = shape
    else:
        raise ValueError("At least one of scl_raster, band_data, or shape must be provided.")

    # 1. Evaluate SCL classifications if available
    if scl_raster is not None:
        scl = np.asarray(scl_raster, dtype=np.int32)
        nodata_mask = (scl == SCLClass.NO_DATA) | (scl == SCLClass.SATURATED_OR_DEFECTIVE)
        cloud_mask = (scl == SCLClass.CLOUD_MEDIUM_PROBABILITY) | (scl == SCLClass.CLOUD_HIGH_PROBABILITY)
        cirrus_mask = (scl == SCLClass.THIN_CIRRUS)
        shadow_mask = (scl == SCLClass.CLOUD_SHADOWS) | (scl == SCLClass.DARK_AREA_PIXELS)
    else:
        nodata_mask = np.zeros(target_shape, dtype=bool)
        cloud_mask = np.zeros(target_shape, dtype=bool)
        cirrus_mask = np.zeros(target_shape, dtype=bool)
        shadow_mask = np.zeros(target_shape, dtype=bool)


    # 2. Evaluate Reflectance and Finite Validity
    if band_data is not None:
        b_data = np.asarray(band_data, dtype=np.float32)
        non_finite = ~np.isfinite(b_data)
        out_of_range = (b_data < reflectance_min) | (b_data > reflectance_max)

        if b_data.ndim > 2:
            # If multiple bands are provided, any band being invalid invalidates the pixel
            non_finite = np.any(non_finite, axis=0)
            out_of_range = np.any(out_of_range, axis=0)

        reflectance_invalid_mask = non_finite | out_of_range
        nodata_mask = nodata_mask | non_finite
    else:
        reflectance_invalid_mask = np.zeros(target_shape, dtype=bool)


    # 3. Unified Analysis-Ready Valid Mask
    # A pixel is analysis-ready ONLY when it is free from clouds, cirrus, shadows, nodata, and invalid values
    valid_mask = (
        ~nodata_mask
        & ~cloud_mask
        & ~cirrus_mask
        & ~shadow_mask
        & ~reflectance_invalid_mask
    )

    return {
        "nodata_mask": nodata_mask,
        "cloud_mask": cloud_mask,
        "cirrus_mask": cirrus_mask,
        "shadow_mask": shadow_mask,
        "reflectance_invalid_mask": reflectance_invalid_mask,
        "valid_mask": valid_mask,
    }


# ============================================================
# QUALITY METRICS COMPUTATION
# ============================================================

def compute_quality_metrics(
    valid_mask: np.ndarray,
    cloud_mask: Optional[np.ndarray] = None,
    cirrus_mask: Optional[np.ndarray] = None,
    shadow_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute rigorous quality statistics from quality masks without fabricating metrics.

    Parameters
    ----------
    valid_mask : np.ndarray
        Boolean array where True indicates analysis-ready pixels.
    cloud_mask : np.ndarray, optional
        Boolean array where True indicates cloud pixels.
    cirrus_mask : np.ndarray, optional
        Boolean array where True indicates cirrus pixels.
    shadow_mask : np.ndarray, optional
        Boolean array where True indicates shadow pixels.

    Returns
    -------
    dict
        Quality metrics dictionary containing:
        - total_pixels: int
        - valid_pixels: int
        - invalid_pixels: int
        - valid_coverage_percentage: float
        - cloud_pixels: Optional[int]
        - cirrus_pixels: Optional[int]
        - shadow_pixels: Optional[int]
        - cloud_percentage_inside_aoi: Optional[float]
        - shadow_percentage_inside_aoi: Optional[float]
    """
    total_pixels = int(valid_mask.size)
    if total_pixels == 0:
        return {
            "total_pixels": 0,
            "valid_pixels": 0,
            "invalid_pixels": 0,
            "valid_coverage_percentage": 0.0,
            "cloud_pixels": None,
            "cirrus_pixels": None,
            "shadow_pixels": None,
            "cloud_percentage_inside_aoi": None,
            "shadow_percentage_inside_aoi": None,
        }

    valid_pixels = int(np.sum(valid_mask))
    invalid_pixels = total_pixels - valid_pixels
    valid_coverage_pct = round((valid_pixels / total_pixels) * 100.0, 2)

    has_scl_info = (cloud_mask is not None or cirrus_mask is not None or shadow_mask is not None)

    if has_scl_info:
        c_pix = int(np.sum(cloud_mask)) if cloud_mask is not None else 0
        cir_pix = int(np.sum(cirrus_mask)) if cirrus_mask is not None else 0
        s_pix = int(np.sum(shadow_mask)) if shadow_mask is not None else 0

        cloud_pct = round(((c_pix + cir_pix) / total_pixels) * 100.0, 2)
        shadow_pct = round((s_pix / total_pixels) * 100.0, 2)
    else:
        c_pix = None
        cir_pix = None
        s_pix = None
        cloud_pct = None
        shadow_pct = None

    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "invalid_pixels": invalid_pixels,
        "valid_coverage_percentage": valid_coverage_pct,
        "cloud_pixels": c_pix,
        "cirrus_pixels": cir_pix,
        "shadow_pixels": s_pix,
        "cloud_percentage_inside_aoi": cloud_pct,
        "shadow_percentage_inside_aoi": shadow_pct,
    }
