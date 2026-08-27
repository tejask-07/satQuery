from __future__ import annotations

import cv2
import numpy as np

from app.vision.preprocessing import prepare_images
from app.vision.utils import (
    clean_mask,
    extract_regions,
    threshold_difference,
    threshold_difference_adaptive,
)


def detect_change(
    before: np.ndarray,
    after: np.ndarray,
    threshold: float = 0.10,
    min_region_area: int = 20,
    method: str = "fixed",
) -> dict:
    """
    Detect visual changes between two aligned images.

    Args:
        before: Earlier image as a NumPy array.
        after: Later image as a NumPy array.
        threshold: Threshold used by the fixed method.
        min_region_area: Minimum connected-component area.
        method: "fixed" or "adaptive".
    """
    if method not in {"fixed", "adaptive"}:
        raise ValueError(
            "method must be either 'fixed' or 'adaptive'"
        )

    before, after = prepare_images(before, after)

    # Compute absolute pixel-wise difference.
    difference = cv2.absdiff(before, after)

    # Convert multi-channel difference to a single intensity map.
    if difference.ndim == 3:
        difference_gray = np.mean(difference, axis=2)
    else:
        difference_gray = difference

    # Convert difference map into a binary change mask.
    if method == "fixed":
        mask = threshold_difference(
            difference_gray,
            threshold=threshold,
        )
    else:
        mask = threshold_difference_adaptive(
            difference_gray,
        )

    # Remove isolated noise and close small gaps.
    mask = clean_mask(mask)

    # Extract connected changed regions.
    regions = extract_regions(
        mask,
        min_area=min_region_area,
    )

    changed_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.shape[0] * mask.shape[1])

    change_ratio = (
        changed_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    return {
        "status": "success",
        "method": method,
        "changed_pixels": changed_pixels,
        "change_ratio": float(change_ratio),
        "regions_detected": len(regions),
        "regions": regions,
        "change_mask": mask,
    }