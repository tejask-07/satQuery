import numpy as np


def detect_change(
    before: np.ndarray,
    after: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """
    Detect temporal change between two raster/index arrays.

    Positive change:
        after > before

    Negative change:
        after < before

    A pixel is considered changed when:

        abs(after - before) >= threshold
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

    valid_mask = (
        np.isfinite(before)
        & np.isfinite(after)
    )

    valid_pixels = int(
        np.sum(valid_mask)
    )

    # --------------------------------------------------------
    # Change raster
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
    # Changed pixels
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
    # --------------------------------------------------------

    if mean_change is None:
        change_type = "no_data"

    elif mean_change > 0:
        change_type = "increase"

    elif mean_change < 0:
        change_type = "decrease"

    else:
        change_type = "no_change"

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "status": "success",

        "mean_before": mean_before,
        "mean_after": mean_after,
        "mean_change": mean_change,

        "changed_pixels": changed_pixels,
        "valid_pixels": valid_pixels,
        "total_pixels": int(before.size),

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
    }