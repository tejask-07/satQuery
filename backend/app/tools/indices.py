import numpy as np

from app.remote_sensing.indices.ndvi import (
    calculate_ndvi as calculate_ndvi_raster,
)

from app.remote_sensing.indices.ndwi import (
    calculate_ndwi as calculate_ndwi_raster,
)

from app.remote_sensing.indices.ndbi import (
    calculate_ndbi as calculate_ndbi_raster,
)


# ============================================================
# Helper
# ============================================================

def _mean_valid(array: np.ndarray):
    """
    Calculate the mean of all finite values.

    Returns None when there are no valid pixels.
    """

    valid = np.isfinite(array)

    if not np.any(valid):
        return None

    return float(np.mean(array[valid]))


# ============================================================
# NDVI
# ============================================================

def calculate_ndvi(
    red: np.ndarray,
    nir: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """
    Calculate NDVI for a single image.

    NDVI = (NIR - Red) / (NIR + Red)
    """

    ndvi = calculate_ndvi_raster(
        red,
        nir,
        valid_mask=valid_mask,
    )

    valid = np.isfinite(ndvi)
    if valid_mask is not None:
        valid = valid & np.asarray(valid_mask, dtype=bool)

    valid_pixels = int(np.sum(valid))
    total_pixels = int(ndvi.size)
    min_val = float(np.min(ndvi[valid])) if valid_pixels > 0 else None
    max_val = float(np.max(ndvi[valid])) if valid_pixels > 0 else None
    mean_val = float(np.mean(ndvi[valid])) if valid_pixels > 0 else None

    return {
        "status": "success",
        "index": "NDVI",
        "data": ndvi,
        "mean": mean_val,
        "min_value": min_val,
        "max_value": max_val,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
    }


# ============================================================
# NDWI
# ============================================================

def calculate_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """
    Calculate NDWI for a single image.

    NDWI = (Green - NIR) / (Green + NIR)
    """

    ndwi = calculate_ndwi_raster(
        green,
        nir,
        valid_mask=valid_mask,
    )

    valid = np.isfinite(ndwi)
    if valid_mask is not None:
        valid = valid & np.asarray(valid_mask, dtype=bool)

    valid_pixels = int(np.sum(valid))
    total_pixels = int(ndwi.size)
    min_val = float(np.min(ndwi[valid])) if valid_pixels > 0 else None
    max_val = float(np.max(ndwi[valid])) if valid_pixels > 0 else None
    mean_val = float(np.mean(ndwi[valid])) if valid_pixels > 0 else None

    return {
        "status": "success",
        "index": "NDWI",
        "data": ndwi,
        "mean": mean_val,
        "min_value": min_val,
        "max_value": max_val,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
    }


# ============================================================
# NDBI
# ============================================================

def calculate_ndbi(
    swir: np.ndarray,
    nir: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """
    Calculate NDBI for a single image.

    NDBI = (SWIR - NIR) / (SWIR + NIR)
    """

    ndbi = calculate_ndbi_raster(
        swir,
        nir,
        valid_mask=valid_mask,
    )

    valid = np.isfinite(ndbi)
    if valid_mask is not None:
        valid = valid & np.asarray(valid_mask, dtype=bool)

    valid_pixels = int(np.sum(valid))
    total_pixels = int(ndbi.size)
    min_val = float(np.min(ndbi[valid])) if valid_pixels > 0 else None
    max_val = float(np.max(ndbi[valid])) if valid_pixels > 0 else None
    mean_val = float(np.mean(ndbi[valid])) if valid_pixels > 0 else None

    return {
        "status": "success",
        "index": "NDBI",
        "data": ndbi,
        "mean": mean_val,
        "min_value": min_val,
        "max_value": max_val,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
    }


# ============================================================
# TEMPORAL NDVI
# ============================================================

def calculate_temporal_ndvi(
    red_before: np.ndarray,
    nir_before: np.ndarray,
    red_after: np.ndarray,
    nir_after: np.ndarray,
    mask_before: np.ndarray | None = None,
    mask_after: np.ndarray | None = None,
) -> dict:
    """
    Calculate NDVI for two dates with quality mask filtering.

    Only jointly valid pixels contribute to temporal statistics.
    """

    ndvi_before = calculate_ndvi_raster(
        red_before,
        nir_before,
        valid_mask=mask_before,
    )

    ndvi_after = calculate_ndvi_raster(
        red_after,
        nir_after,
        valid_mask=mask_after,
    )

    # Compute joint validity mask: finite in both and valid in both masks
    valid = np.isfinite(ndvi_before) & np.isfinite(ndvi_after)
    if mask_before is not None:
        valid = valid & np.asarray(mask_before, dtype=bool)
    if mask_after is not None:
        valid = valid & np.asarray(mask_after, dtype=bool)

    # Apply joint valid mask: set invalid pixels to NaN
    ndvi_before = np.where(valid, ndvi_before, np.nan)
    ndvi_after = np.where(valid, ndvi_after, np.nan)

    valid_pixels = int(np.sum(valid))
    total_pixels = int(valid.size)

    if valid_pixels == 0:
        mean_before = None
        mean_after = None
        mean_change = None
        min_before = None
        max_before = None
        min_after = None
        max_after = None
    else:
        mean_before = float(np.mean(ndvi_before[valid]))
        mean_after = float(np.mean(ndvi_after[valid]))
        mean_change = mean_after - mean_before
        min_before = float(np.min(ndvi_before[valid]))
        max_before = float(np.max(ndvi_before[valid]))
        min_after = float(np.min(ndvi_after[valid]))
        max_after = float(np.max(ndvi_after[valid]))

    return {
        "status": "success",
        "index": "NDVI",
        "ndvi_before": ndvi_before,
        "ndvi_after": ndvi_after,
        "mean_ndvi_before": mean_before,
        "mean_ndvi_after": mean_after,
        "min_ndvi_before": min_before,
        "max_ndvi_before": max_before,
        "min_ndvi_after": min_after,
        "max_ndvi_after": max_after,
        "mean_ndvi_change": mean_change,
        "valid_mask": valid,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
    }


# ============================================================
# TEMPORAL NDWI
# ============================================================

def calculate_temporal_ndwi(
    green_before: np.ndarray,
    nir_before: np.ndarray,
    green_after: np.ndarray,
    nir_after: np.ndarray,
    mask_before: np.ndarray | None = None,
    mask_after: np.ndarray | None = None,
) -> dict:
    """
    Calculate NDWI for two dates with quality mask filtering.
    """

    ndwi_before = calculate_ndwi_raster(
        green_before,
        nir_before,
        valid_mask=mask_before,
    )

    ndwi_after = calculate_ndwi_raster(
        green_after,
        nir_after,
        valid_mask=mask_after,
    )

    valid = np.isfinite(ndwi_before) & np.isfinite(ndwi_after)
    if mask_before is not None:
        valid = valid & np.asarray(mask_before, dtype=bool)
    if mask_after is not None:
        valid = valid & np.asarray(mask_after, dtype=bool)

    ndwi_before = np.where(valid, ndwi_before, np.nan)
    ndwi_after = np.where(valid, ndwi_after, np.nan)

    valid_pixels = int(np.sum(valid))
    total_pixels = int(valid.size)

    if valid_pixels == 0:
        mean_before = None
        mean_after = None
        mean_change = None
        min_before = None
        max_before = None
        min_after = None
        max_after = None
    else:
        mean_before = float(np.mean(ndwi_before[valid]))
        mean_after = float(np.mean(ndwi_after[valid]))
        mean_change = mean_after - mean_before
        min_before = float(np.min(ndwi_before[valid]))
        max_before = float(np.max(ndwi_before[valid]))
        min_after = float(np.min(ndwi_after[valid]))
        max_after = float(np.max(ndwi_after[valid]))

    return {
        "status": "success",
        "index": "NDWI",
        "ndwi_before": ndwi_before,
        "ndwi_after": ndwi_after,
        "mean_ndwi_before": mean_before,
        "mean_ndwi_after": mean_after,
        "min_ndwi_before": min_before,
        "max_ndwi_before": max_before,
        "min_ndwi_after": min_after,
        "max_ndwi_after": max_after,
        "mean_ndwi_change": mean_change,
        "valid_mask": valid,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
    }


# ============================================================
# TEMPORAL NDBI
# ============================================================

def calculate_temporal_ndbi(
    swir_before: np.ndarray,
    nir_before: np.ndarray,
    swir_after: np.ndarray,
    nir_after: np.ndarray,
    mask_before: np.ndarray | None = None,
    mask_after: np.ndarray | None = None,
) -> dict:
    """
    Calculate NDBI for two dates with quality mask filtering.
    """

    ndbi_before = calculate_ndbi_raster(
        swir_before,
        nir_before,
        valid_mask=mask_before,
    )

    ndbi_after = calculate_ndbi_raster(
        swir_after,
        nir_after,
        valid_mask=mask_after,
    )

    valid = np.isfinite(ndbi_before) & np.isfinite(ndbi_after)
    if mask_before is not None:
        valid = valid & np.asarray(mask_before, dtype=bool)
    if mask_after is not None:
        valid = valid & np.asarray(mask_after, dtype=bool)

    ndbi_before = np.where(valid, ndbi_before, np.nan)
    ndbi_after = np.where(valid, ndbi_after, np.nan)

    valid_pixels = int(np.sum(valid))
    total_pixels = int(valid.size)

    if valid_pixels == 0:
        mean_before = None
        mean_after = None
        mean_change = None
        min_before = None
        max_before = None
        min_after = None
        max_after = None
    else:
        mean_before = float(np.mean(ndbi_before[valid]))
        mean_after = float(np.mean(ndbi_after[valid]))
        mean_change = mean_after - mean_before
        min_before = float(np.min(ndbi_before[valid]))
        max_before = float(np.max(ndbi_before[valid]))
        min_after = float(np.min(ndbi_after[valid]))
        max_after = float(np.max(ndbi_after[valid]))

    return {
        "status": "success",
        "index": "NDBI",
        "ndbi_before": ndbi_before,
        "ndbi_after": ndbi_after,
        "mean_ndbi_before": mean_before,
        "mean_ndbi_after": mean_after,
        "min_ndbi_before": min_before,
        "max_ndbi_before": max_before,
        "min_ndbi_after": min_after,
        "max_ndbi_after": max_after,
        "mean_ndbi_change": mean_change,
        "valid_mask": valid,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
    }