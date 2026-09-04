from pathlib import Path

import cv2
import numpy as np


def detect_change(
    before: np.ndarray,
    after: np.ndarray,
    threshold: float = 0.01,
    output_path: str | None = None,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """
    Detect temporal change between two raster/index arrays.

    A pixel is considered changed when:

        abs(after - before) >= threshold

    Positive changed pixels:
        after > before

    Negative changed pixels:
        after < before
    """

    before = np.asarray(
        before,
        dtype=np.float32,
    )

    after = np.asarray(
        after,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    if before.shape != after.shape:
        raise ValueError(
            "Before and after arrays must have the same shape."
        )

    # --------------------------------------------------------
    # Valid pixels
    # --------------------------------------------------------

    finite_mask = (
        np.isfinite(before)
        & np.isfinite(after)
    )

    if valid_mask is not None:
        valid_mask = finite_mask & np.asarray(valid_mask, dtype=bool)
    else:
        valid_mask = finite_mask

    valid_pixels = int(
        np.sum(valid_mask)
    )


    # --------------------------------------------------------
    # Signed change raster
    # --------------------------------------------------------

    change = np.full(
        before.shape,
        np.nan,
        dtype=np.float32,
    )

    change[valid_mask] = (
        after[valid_mask]
        - before[valid_mask]
    )

    # --------------------------------------------------------
    # Significant changed pixels
    # --------------------------------------------------------

    changed_mask = (
        valid_mask
        & (
            np.abs(change)
            >= threshold
        )
    )

    changed_pixels = int(
        np.sum(changed_mask)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    if valid_pixels > 0:

        mean_before = float(
            np.mean(
                before[valid_mask]
            )
        )

        mean_after = float(
            np.mean(
                after[valid_mask]
            )
        )

        mean_change = (
            mean_after
            - mean_before
        )

        change_ratio = (
            changed_pixels
            / valid_pixels
        )

    else:

        mean_before = None
        mean_after = None
        mean_change = None
        change_ratio = 0.0

    # --------------------------------------------------------
    # Increase / decrease
    #
    # IMPORTANT:
    # These count ONLY statistically significant pixels.
    # --------------------------------------------------------

    increased_pixels = int(
        np.sum(
            changed_mask
            & (change > 0)
        )
    )

    decreased_pixels = int(
        np.sum(
            changed_mask
            & (change < 0)
        )
    )

    # --------------------------------------------------------
    # Overall change type
    #
    # IMPORTANT:
    # If there are zero significant changed pixels,
    # report no_change regardless of the sign of mean_change.
    # --------------------------------------------------------

    if valid_pixels == 0:

        change_type = "no_data"

    elif changed_pixels == 0:

        change_type = "no_change"

    elif increased_pixels > decreased_pixels:

        change_type = "increase"

    elif decreased_pixels > increased_pixels:

        change_type = "decrease"

    else:

        change_type = "mixed"

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    visualization_path = None

    if output_path is not None:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        absolute_change = np.abs(
            np.nan_to_num(
                change,
                nan=0.0,
            )
        )

        max_change = float(
            np.max(
                absolute_change
            )
        )

        if max_change > 0:

            visualization = (
                absolute_change
                / max_change
                * 255.0
            ).astype(np.uint8)

        else:

            visualization = np.zeros(
                change.shape,
                dtype=np.uint8,
            )

        success = cv2.imwrite(
            str(output_path),
            visualization,
        )

        if not success:
            raise IOError(
                "Failed to save change visualization: "
                f"{output_path}"
            )

        visualization_path = str(
            output_path
        )

    min_value = float(np.min(change[valid_mask])) if valid_pixels > 0 else None
    max_value = float(np.max(change[valid_mask])) if valid_pixels > 0 else None

    return {
        "status": "success",

        "mean_before": mean_before,
        "mean_after": mean_after,
        "mean_change": mean_change,
        "min_value": min_value,
        "max_value": max_value,

        "changed_pixels": changed_pixels,
        "valid_pixels": valid_pixels,
        "total_pixels": int(
            before.size
        ),

        "change_ratio": float(
            change_ratio
        ),

        "increased_pixels": increased_pixels,
        "decreased_pixels": decreased_pixels,

        "change_type": change_type,

        "threshold": float(
            threshold
        ),

        "change_map": change,

        "visualization_path": visualization_path,
    }