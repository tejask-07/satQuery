"""
Phase 9: Evidence-Backed Error Analysis Engine.

Classifies false-positive, false-negative, and ambiguous detection failures
into structured, scientifically grounded failure categories.

CATEGORIES:
- URBAN_FALSE_POSITIVE_NDBI: High NDBI on bare agricultural soil / fallow land
- VEGETATION_SEASONAL_EFFECT: Phenological leaf-fall or crop calendar discrepancy
- WATER_EPHEMERAL_CHANGE: Transient surface ponding / ephemeral inundation
- CLOUD_CONTAMINATION: Residual cloud edge, thin cirrus, or shadow boundary
- SMALL_REGION_FILTERED: Sub-MMU change parcel filtered out by spatial contiguity
- MIXED_PIXEL: Boundary interface between distinct land cover classes
- INSUFFICIENT_TEMPORAL_EVIDENCE: Single-pair transient fluctuation without persistence
- CONFLICTING_INDICES: Opposing index trajectories (e.g. concurrent high NDBI & NDWI)
- UNCATEGORIZED: Residual errors not meeting physical diagnostic rules
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class ErrorCategory(str, Enum):
    URBAN_FALSE_POSITIVE_NDBI = "URBAN_FALSE_POSITIVE_NDBI"
    VEGETATION_SEASONAL_EFFECT = "VEGETATION_SEASONAL_EFFECT"
    WATER_EPHEMERAL_CHANGE = "WATER_EPHEMERAL_CHANGE"
    CLOUD_CONTAMINATION = "CLOUD_CONTAMINATION"
    SMALL_REGION_FILTERED = "SMALL_REGION_FILTERED"
    MIXED_PIXEL = "MIXED_PIXEL"
    INSUFFICIENT_TEMPORAL_EVIDENCE = "INSUFFICIENT_TEMPORAL_EVIDENCE"
    CONFLICTING_INDICES = "CONFLICTING_INDICES"
    UNCATEGORIZED = "UNCATEGORIZED"


def diagnose_pixel_error(
    r: int,
    c: int,
    true_cls: int,
    pred_cls: int,
    delta_ndvi: float,
    delta_ndwi: float,
    delta_ndbi: float,
    ndvi_before: float,
    is_boundary_pixel: bool,
    was_filtered_by_mmu: bool,
    is_cloud_adjacent: bool,
    seasonal_doy_diff: int,
    observation_count: int,
) -> ErrorCategory:
    """
    Deterministically assigns an evidence-based error category to a misclassified pixel.
    """
    # 1. Cloud contamination
    if is_cloud_adjacent:
        return ErrorCategory.CLOUD_CONTAMINATION

    # 2. MMU filtering (was detected initially but cluster was < MMU)
    if was_filtered_by_mmu and true_cls > 0 and pred_cls == 0:
        return ErrorCategory.SMALL_REGION_FILTERED

    # 3. Conflicting indices (opposing deltas)
    if abs(delta_ndbi) > 0.12 and abs(delta_ndwi) > 0.12:
        return ErrorCategory.CONFLICTING_INDICES

    # 4. Urban false positive due to bare soil / fallow land
    if pred_cls == 1 and true_cls != 1:
        if delta_ndbi > 0.05 and ndvi_before < 0.20:
            return ErrorCategory.URBAN_FALSE_POSITIVE_NDBI

    # 5. Seasonal vegetation effects (calendar discrepancy > 60 days)
    if (pred_cls in (3, 4) or true_cls in (3, 4)) and seasonal_doy_diff > 60:
        if abs(delta_ndvi) > 0.10:
            return ErrorCategory.VEGETATION_SEASONAL_EFFECT

    # 6. Ephemeral water change
    if (pred_cls in (5, 6) or true_cls in (5, 6)) and observation_count <= 2:
        if delta_ndwi > 0.12 and not is_boundary_pixel:
            return ErrorCategory.WATER_EPHEMERAL_CHANGE

    # 7. Mixed boundary pixel
    if is_boundary_pixel:
        return ErrorCategory.MIXED_PIXEL

    # 8. Insufficient temporal evidence
    if observation_count < 3 and (pred_cls > 0 or true_cls > 0):
        return ErrorCategory.INSUFFICIENT_TEMPORAL_EVIDENCE

    return ErrorCategory.UNCATEGORIZED


def analyze_scene_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ndvi_before: np.ndarray,
    ndvi_after: np.ndarray,
    ndwi_before: np.ndarray,
    ndwi_after: np.ndarray,
    ndbi_before: np.ndarray,
    ndbi_after: np.ndarray,
    scl_mask: Optional[np.ndarray] = None,
    raw_pred_before_mmu: Optional[np.ndarray] = None,
    seasonal_doy_diff: int = 0,
    observation_count: int = 2,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Performs comprehensive diagnostic error analysis on a prediction raster.

    Returns:
        Structured summary with counts and percentage breakdown per ErrorCategory.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred shapes must match.")

    # Mask of misclassifications
    if valid_mask is None:
        eval_mask = (y_true >= 0) & (y_true < 8) & (y_pred >= 0) & (y_pred < 8)
    else:
        eval_mask = valid_mask & (y_true >= 0) & (y_true < 8) & (y_pred >= 0) & (y_pred < 8)

    error_mask = eval_mask & (y_true != y_pred)
    total_errors = int(np.sum(error_mask))

    if total_errors == 0:
        return {
            "total_evaluated_pixels": int(np.sum(eval_mask)),
            "total_error_pixels": 0,
            "error_rate": 0.0,
            "category_counts": {cat.value: 0 for cat in ErrorCategory},
            "category_percentages": {cat.value: 0.0 for cat in ErrorCategory},
            "primary_failure_mode": "NONE",
        }

    # Index deltas
    delta_ndvi = ndvi_after - ndvi_before
    delta_ndwi = ndwi_after - ndwi_before
    delta_ndbi = ndbi_after - ndbi_before

    # Boundary detection (mixed pixels) using morphological gradient on y_true
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(y_true.astype(np.uint8), kernel)
    eroded = cv2.erode(y_true.astype(np.uint8), kernel)
    boundary_mask = (dilated != eroded)

    # Cloud adjacency
    if scl_mask is not None:
        cloud_pixels = np.isin(scl_mask, [3, 8, 9, 10]).astype(np.uint8)
        cloud_buffer = cv2.dilate(cloud_pixels, kernel, iterations=2) > 0
    else:
        cloud_buffer = np.zeros(y_true.shape, dtype=bool)

    # MMU filtering flag
    if raw_pred_before_mmu is not None:
        mmu_filtered_mask = (raw_pred_before_mmu > 0) & (y_pred == 0)
    else:
        mmu_filtered_mask = np.zeros(y_true.shape, dtype=bool)

    category_counts = {cat.value: 0 for cat in ErrorCategory}

    error_rows, error_cols = np.where(error_mask)
    for r, c in zip(error_rows, error_cols):
        cat = diagnose_pixel_error(
            r=int(r),
            c=int(c),
            true_cls=int(y_true[r, c]),
            pred_cls=int(y_pred[r, c]),
            delta_ndvi=float(delta_ndvi[r, c]),
            delta_ndwi=float(delta_ndwi[r, c]),
            delta_ndbi=float(delta_ndbi[r, c]),
            ndvi_before=float(ndvi_before[r, c]),
            is_boundary_pixel=bool(boundary_mask[r, c]),
            was_filtered_by_mmu=bool(mmu_filtered_mask[r, c]),
            is_cloud_adjacent=bool(cloud_buffer[r, c]),
            seasonal_doy_diff=seasonal_doy_diff,
            observation_count=observation_count,
        )
        category_counts[cat.value] += 1

    category_percentages = {
        k: round(v / total_errors * 100.0, 2)
        for k, v in category_counts.items()
    }

    # Identify primary failure mode
    primary = max(category_counts.items(), key=lambda x: x[1])[0]

    return {
        "total_evaluated_pixels": int(np.sum(eval_mask)),
        "total_error_pixels": total_errors,
        "error_rate": round(total_errors / int(np.sum(eval_mask)), 4),
        "category_counts": category_counts,
        "category_percentages": category_percentages,
        "primary_failure_mode": primary,
    }
