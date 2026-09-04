from __future__ import annotations

from typing import Optional, Union
import numpy as np


def compute_joint_valid_mask(
    mask_before: np.ndarray,
    mask_after: np.ndarray,
) -> np.ndarray:
    """
    Compute joint analysis validity mask as the intersection of before and after masks.

    A pixel is jointly valid if and only if it is valid (free from clouds, shadows,
    and nodata) in BOTH observations.

    Parameters
    ----------
    mask_before : np.ndarray
        Boolean mask for the before observation (True = valid).
    mask_after : np.ndarray
        Boolean mask for the after observation (True = valid).

    Returns
    -------
    np.ndarray
        Boolean mask representing jointly valid pixels.
    """
    m_before = np.asarray(mask_before, dtype=bool)
    m_after = np.asarray(mask_after, dtype=bool)

    if m_before.shape != m_after.shape:
        raise ValueError(
            f"Shape mismatch in joint mask computation: before {m_before.shape} vs after {m_after.shape}"
        )

    return m_before & m_after


def apply_mask(
    data: np.ndarray,
    valid_mask: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Apply a boolean validity mask to raster data.

    Parameters
    ----------
    data : np.ndarray
        Input raster array.
    valid_mask : np.ndarray
        Boolean mask where True means valid and False means invalid.
    fill_value : float
        Value to assign to invalid pixels (default: np.nan).

    Returns
    -------
    np.ndarray
        New array with invalid pixels replaced by fill_value.
    """
    arr = np.asarray(data, dtype=np.float32).copy()
    mask = np.asarray(valid_mask, dtype=bool)

    if arr.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: data {arr.shape} vs mask {mask.shape}"
        )

    arr[~mask] = fill_value
    return arr


def create_valid_mask(data: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask identifying finite, valid pixels.
    """
    data = np.asarray(data)
    return np.isfinite(data)
