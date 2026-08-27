from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np


def threshold_difference(
    difference: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convert a normalized difference image into a binary change mask.
    """
    if difference.ndim != 2:
        raise ValueError("difference must be a 2D array")

    mask = (difference >= threshold).astype(np.uint8) * 255

    return mask


def clean_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Remove small noise and close small gaps using morphology.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask


def extract_regions(
    mask: np.ndarray,
    min_area: int = 20,
) -> list[dict]:
    """
    Extract connected changed regions from a binary mask.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    regions: list[dict] = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        center_x, center_y = centroids[label]

        regions.append(
            {
                "area_pixels": area,
                "bbox": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                },
                "centroid": {
                    "x": float(center_x),
                    "y": float(center_y),
                },
            }
        )

    regions.sort(
        key=lambda region: region["area_pixels"],
        reverse=True,
    )

    return regions

def save_mask(
    mask: np.ndarray,
    output_path: str | Path,
) -> Path:
    """
    Save a binary change mask as an image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), mask)

    if not success:
        raise IOError(f"Failed to save mask to {output_path}")

    return output_path

def draw_regions(
    image: np.ndarray,
    regions: list[dict],
) -> np.ndarray:
    """
    Draw bounding boxes around detected change regions.
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D or 3D")

    if image.ndim == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()

    for region in regions:
        bbox = region["bbox"]

        x = bbox["x"]
        y = bbox["y"]
        width = bbox["width"]
        height = bbox["height"]

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (255, 255, 255),
            2,
        )

    return output

def threshold_difference_adaptive(
    difference: np.ndarray,
) -> np.ndarray:
    """
    Automatically determine a threshold from the difference image
    using Otsu's method.

    Input:
        difference: 2D float image in the [0, 1] range.

    Returns:
        Binary uint8 mask where changed pixels are 255.
    """
    if difference.ndim != 2:
        raise ValueError("difference must be a 2D array")

    # Convert [0, 1] difference to 8-bit [0, 255].
    difference_uint8 = np.clip(
        difference * 255.0,
        0,
        255,
    ).astype(np.uint8)

    _, mask = cv2.threshold(
        difference_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return mask