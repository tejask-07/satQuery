from __future__ import annotations

import numpy as np


def validate_images(
    before: np.ndarray,
    after: np.ndarray,
) -> None:
    """
    Validate that two images are NumPy arrays with compatible shapes.
    """
    if not isinstance(before, np.ndarray):
        raise TypeError("before must be a NumPy array")

    if not isinstance(after, np.ndarray):
        raise TypeError("after must be a NumPy array")

    if before.size == 0 or after.size == 0:
        raise ValueError("Images must not be empty")

    if before.shape[:2] != after.shape[:2]:
        raise ValueError(
            f"Image dimensions do not match: "
            f"{before.shape[:2]} vs {after.shape[:2]}"
        )


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert image values to float32 in the [0, 1] range.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    image = image.astype(np.float32)

    if image.max() > 1.0:
        image /= 255.0

    return np.clip(image, 0.0, 1.0)


def prepare_images(
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and normalize two images for change detection.

    For the MVP, images must already be spatially aligned.
    """
    validate_images(before, after)

    before = normalize_image(before)
    after = normalize_image(after)

    return before, after