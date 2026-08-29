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
) -> dict:
    """
    Calculate NDVI for a single image.

    NDVI = (NIR - Red) / (NIR + Red)
    """

    ndvi = calculate_ndvi_raster(
        red,
        nir,
    )

    return {
        "status": "success",
        "index": "NDVI",
        "data": ndvi,
        "mean": _mean_valid(ndvi),
    }


# ============================================================
# NDWI
# ============================================================

def calculate_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
) -> dict:
    """
    Calculate NDWI for a single image.

    NDWI = (Green - NIR) / (Green + NIR)
    """

    ndwi = calculate_ndwi_raster(
        green,
        nir,
    )

    return {
        "status": "success",
        "index": "NDWI",
        "data": ndwi,
        "mean": _mean_valid(ndwi),
    }


# ============================================================
# NDBI
# ============================================================

def calculate_ndbi(
    swir: np.ndarray,
    nir: np.ndarray,
) -> dict:
    """
    Calculate NDBI for a single image.

    NDBI = (SWIR - NIR) / (SWIR + NIR)
    """

    ndbi = calculate_ndbi_raster(
        swir,
        nir,
    )

    return {
        "status": "success",
        "index": "NDBI",
        "data": ndbi,
        "mean": _mean_valid(ndbi),
    }


# ============================================================
# TEMPORAL NDVI
# ============================================================

def calculate_temporal_ndvi(
    red_before: np.ndarray,
    nir_before: np.ndarray,
    red_after: np.ndarray,
    nir_after: np.ndarray,
) -> dict:
    """
    Calculate NDVI for two dates.

    Returns both rasters and their mean values.
    """

    ndvi_before = calculate_ndvi_raster(
        red_before,
        nir_before,
    )

    ndvi_after = calculate_ndvi_raster(
        red_after,
        nir_after,
    )

    valid = (
        np.isfinite(ndvi_before)
        & np.isfinite(ndvi_after)
    )

    if not np.any(valid):
        mean_before = None
        mean_after = None
        mean_change = None
    else:
        mean_before = float(
            np.mean(ndvi_before[valid])
        )

        mean_after = float(
            np.mean(ndvi_after[valid])
        )

        mean_change = (
            mean_after - mean_before
        )

    return {
        "status": "success",
        "index": "NDVI",

        "ndvi_before": ndvi_before,
        "ndvi_after": ndvi_after,

        "mean_ndvi_before": mean_before,
        "mean_ndvi_after": mean_after,
        "mean_ndvi_change": mean_change,
    }


# ============================================================
# TEMPORAL NDWI
# ============================================================

def calculate_temporal_ndwi(
    green_before: np.ndarray,
    nir_before: np.ndarray,
    green_after: np.ndarray,
    nir_after: np.ndarray,
) -> dict:
    """
    Calculate NDWI for two dates.

    Used for water-change detection.
    """

    ndwi_before = calculate_ndwi_raster(
        green_before,
        nir_before,
    )

    ndwi_after = calculate_ndwi_raster(
        green_after,
        nir_after,
    )

    valid = (
        np.isfinite(ndwi_before)
        & np.isfinite(ndwi_after)
    )

    if not np.any(valid):
        mean_before = None
        mean_after = None
        mean_change = None
    else:
        mean_before = float(
            np.mean(ndwi_before[valid])
        )

        mean_after = float(
            np.mean(ndwi_after[valid])
        )

        mean_change = (
            mean_after - mean_before
        )

    return {
        "status": "success",
        "index": "NDWI",

        "ndwi_before": ndwi_before,
        "ndwi_after": ndwi_after,

        "mean_ndwi_before": mean_before,
        "mean_ndwi_after": mean_after,
        "mean_ndwi_change": mean_change,
    }


# ============================================================
# TEMPORAL NDBI
# ============================================================

def calculate_temporal_ndbi(
    swir_before: np.ndarray,
    nir_before: np.ndarray,
    swir_after: np.ndarray,
    nir_after: np.ndarray,
) -> dict:
    """
    Calculate NDBI for two dates.

    Used for urban/built-up change detection.
    """

    ndbi_before = calculate_ndbi_raster(
        swir_before,
        nir_before,
    )

    ndbi_after = calculate_ndbi_raster(
        swir_after,
        nir_after,
    )

    valid = (
        np.isfinite(ndbi_before)
        & np.isfinite(ndbi_after)
    )

    if not np.any(valid):
        mean_before = None
        mean_after = None
        mean_change = None
    else:
        mean_before = float(
            np.mean(ndbi_before[valid])
        )

        mean_after = float(
            np.mean(ndbi_after[valid])
        )

        mean_change = (
            mean_after - mean_before
        )

    return {
        "status": "success",
        "index": "NDBI",

        "ndbi_before": ndbi_before,
        "ndbi_after": ndbi_after,

        "mean_ndbi_before": mean_before,
        "mean_ndbi_after": mean_after,
        "mean_ndbi_change": mean_change,
    }