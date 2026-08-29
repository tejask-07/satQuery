import numpy as np


def calculate_ndvi_change(
    ndvi_before: np.ndarray,
    ndvi_after: np.ndarray,
) -> np.ndarray:
    """
    Calculate NDVI change between two dates.

    NDVI change = NDVI after - NDVI before

    Parameters
    ----------
    ndvi_before : np.ndarray
        NDVI values from the earlier date.

    ndvi_after : np.ndarray
        NDVI values from the later date.

    Returns
    -------
    np.ndarray
        Pixel-wise NDVI change.
        Negative values indicate vegetation decrease.
        Positive values indicate vegetation increase.
    """

    ndvi_before = np.asarray(ndvi_before, dtype=np.float32)
    ndvi_after = np.asarray(ndvi_after, dtype=np.float32)

    if ndvi_before.shape != ndvi_after.shape:
        raise ValueError(
            "Before and after NDVI arrays must have the same shape."
        )

    return ndvi_after - ndvi_before
def detect_vegetation_decrease(
    ndvi_change: np.ndarray,
    threshold: float = -0.2,
) -> np.ndarray:
    """
    Identify pixels with significant vegetation decrease.

    Parameters
    ----------
    ndvi_change : np.ndarray
        Pixel-wise NDVI change values.

    threshold : float, optional
        Maximum NDVI change considered significant decrease.
        Pixels with change <= threshold are marked True.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates vegetation decrease.
    """

    ndvi_change = np.asarray(ndvi_change, dtype=np.float32)

    return np.isfinite(ndvi_change) & (ndvi_change <= threshold)
def summarize_vegetation_change(
    ndvi_change: np.ndarray,
    decrease_mask: np.ndarray,
) -> dict:
    """
    Summarize vegetation change statistics.

    Parameters
    ----------
    ndvi_change : np.ndarray
        Pixel-wise NDVI change values.

    decrease_mask : np.ndarray
        Boolean mask identifying vegetation decrease.

    Returns
    -------
    dict
        Summary statistics for vegetation change.
    """

    ndvi_change = np.asarray(ndvi_change, dtype=np.float32)
    decrease_mask = np.asarray(decrease_mask, dtype=bool)

    if ndvi_change.shape != decrease_mask.shape:
        raise ValueError(
            "NDVI change and decrease mask must have the same shape."
        )

    valid_change = ndvi_change[np.isfinite(ndvi_change)]

    if valid_change.size == 0:
        return {
            "valid_pixel_count": 0,
            "mean_ndvi_change": None,
            "min_ndvi_change": None,
            "max_ndvi_change": None,
            "decreased_pixel_count": 0,
            "decreased_pixel_percentage": 0.0,
        }

    decreased_pixel_count = int(np.count_nonzero(decrease_mask))
    valid_pixel_count = int(valid_change.size)

    return {
        "valid_pixel_count": valid_pixel_count,
        "mean_ndvi_change": float(np.mean(valid_change)),
        "min_ndvi_change": float(np.min(valid_change)),
        "max_ndvi_change": float(np.max(valid_change)),
        "decreased_pixel_count": decreased_pixel_count,
        "decreased_pixel_percentage": (
            decreased_pixel_count / valid_pixel_count
        ) * 100.0,
    }
from typing import Any


def build_vegetation_analysis(
    ndvi_before: np.ndarray,
    ndvi_after: np.ndarray,
    threshold: float = -0.2,
) -> dict[str, Any]:
    """
    Run the complete vegetation-change analysis.

    Parameters
    ----------
    ndvi_before : np.ndarray
        NDVI values from the earlier date.

    ndvi_after : np.ndarray
        NDVI values from the later date.

    threshold : float, optional
        NDVI change threshold used to identify vegetation decrease.

    Returns
    -------
    dict[str, Any]
        Complete vegetation-change analysis containing:
        - NDVI change array
        - vegetation decrease mask
        - summary statistics
    """

    ndvi_change = calculate_ndvi_change(
        ndvi_before,
        ndvi_after,
    )

    decrease_mask = detect_vegetation_decrease(
        ndvi_change,
        threshold=threshold,
    )

    statistics = summarize_vegetation_change(
        ndvi_change,
        decrease_mask,
    )

    return {
        "ndvi_change": ndvi_change,
        "decrease_mask": decrease_mask,
        "statistics": statistics,
    }