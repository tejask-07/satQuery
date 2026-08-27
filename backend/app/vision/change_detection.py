from __future__ import annotations

import cv2
import numpy as np

from app.vision.preprocessing import prepare_images
from app.vision.utils import (
    clean_mask,
    extract_regions,
    threshold_difference,
)


def detect_change(
    before: np.ndarray,
    after: np.ndarray,
    threshold: float = 0.10,
    min_region_area: int = 20,
) -> dict:
    """
    Detect visual changes between two aligned images.

    This is the initial classical CV baseline for SatQuery AI.
    """
    before, after = prepare_images(before, after)

    # Absolute difference between the two images.
    difference = cv2.absdiff(before, after)

    # Convert multi-channel differences into a single intensity map.
    if difference.ndim == 3:
        difference_gray = np.mean(difference, axis=2)
    else:
        difference_gray = difference

    # Turn the difference into a candidate change mask.
    mask = threshold_difference(
        difference_gray,
        threshold=threshold,
    )

    # Remove small isolated noise.
    mask = clean_mask(mask)

    # Extract meaningful connected regions.
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
        "changed_pixels": changed_pixels,
        "change_ratio": float(change_ratio),
        "regions_detected": len(regions),
        "regions": regions,
        "change_mask": mask,
    }