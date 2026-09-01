from pathlib import Path
from typing import Optional

import rasterio
import numpy as np


def _read_band(path: str) -> np.ndarray:
    """
    Read the first raster band and return it as a float array.
    """
    with rasterio.open(path) as src:
        return src.read(1).astype(float)


def _compare_arrays(
    before: np.ndarray,
    after: np.ndarray,
    threshold: float = 10.0,
) -> dict:
    """
    Calculate pixel-level comparison statistics.

    The default threshold is 10.0 because this function compares
    raw raster pixel values rather than normalized indices such
    as NDVI or NDWI.
    """

    if before.shape != after.shape:
        raise ValueError(
            "The two images must have the same dimensions."
        )

    valid = (
        np.isfinite(before)
        & np.isfinite(after)
    )

    if not np.any(valid):
        return {
            "valid_pixels": 0,
            "mean_before": None,
            "mean_after": None,
            "mean_change": None,
            "changed_pixels": 0,
            "change_ratio": 0.0,
            "increased_pixels": 0,
            "decreased_pixels": 0,
            "threshold": threshold,
        }

    before_valid = before[valid]
    after_valid = after[valid]

    difference = after_valid - before_valid

    changed = np.abs(difference) > threshold

    changed_pixels = int(np.sum(changed))
    valid_pixels = int(len(difference))

    increased_pixels = int(
        np.sum(difference > threshold)
    )

    decreased_pixels = int(
        np.sum(difference < -threshold)
    )

    if increased_pixels > decreased_pixels:
        change_type = "increase"
    elif decreased_pixels > increased_pixels:
        change_type = "decrease"
    else:
        change_type = "mixed"

    return {
        "valid_pixels": valid_pixels,
        "mean_before": float(np.mean(before_valid)),
        "mean_after": float(np.mean(after_valid)),
        "mean_change": float(np.mean(difference)),
        "changed_pixels": changed_pixels,
        "change_ratio": float(
            changed_pixels / valid_pixels
        ),
        "increased_pixels": increased_pixels,
        "decreased_pixels": decreased_pixels,
        "change_type": change_type,
        "threshold": threshold,
    }


def compare_images(
    before_image=None,
    after_image=None,
    threshold: Optional[float] = None,
    **kwargs,
):
    """
    Compare two satellite raster images.

    The executor may provide image information through
    different keyword names, so this function accepts
    both explicit arguments and **kwargs.
    """

    # Support common argument names used by the executor.
    if before_image is None:
        before_image = kwargs.get("before")

    if before_image is None:
        before_image = kwargs.get("before_path")

    if after_image is None:
        after_image = kwargs.get("after")

    if after_image is None:
        after_image = kwargs.get("after_path")

    # Allow the executor to pass a comparison threshold.
    if threshold is None:
        threshold = kwargs.get("change_threshold")

    if threshold is None:
        threshold = 10.0

    threshold = float(threshold)

    # If image records were passed instead of direct paths,
    # extract the first available band.
    if isinstance(before_image, dict):
        before_bands = before_image.get("bands", {})

        before_image = (
            before_bands.get("nir")
            or before_bands.get("red")
            or before_bands.get("green")
        )

    if isinstance(after_image, dict):
        after_bands = after_image.get("bands", {})

        after_image = (
            after_bands.get("nir")
            or after_bands.get("red")
            or after_bands.get("green")
        )

    if not before_image or not after_image:
        raise ValueError(
            "compare_images requires both before and after images."
        )

    before_path = Path(before_image)
    after_path = Path(after_image)

    if not before_path.exists():
        raise FileNotFoundError(
            f"Before image not found: {before_path}"
        )

    if not after_path.exists():
        raise FileNotFoundError(
            f"After image not found: {after_path}"
        )

    before = _read_band(str(before_path))
    after = _read_band(str(after_path))

    statistics = _compare_arrays(
        before,
        after,
        threshold=threshold,
    )

    return {
        "status": "success",
        "operation": "image_comparison",
        "before_image": str(before_path),
        "after_image": str(after_path),
        "statistics": statistics,
    }