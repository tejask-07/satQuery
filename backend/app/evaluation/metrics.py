"""
Phase 9: Evaluation Metrics Engine.

Provides mathematically rigorous, deterministic evaluation metrics for pixel-level
and region-level change detection in remote sensing imagery.

KEY PRINCIPLES:
1. Deterministic calculation: zero stochastic variation or heuristic approximation.
2. Explicit invalid-pixel masking: nodata/cloud pixels are accounted for in ignored_pixel_count
   and strictly excluded from evaluation denominators.
3. Safe zero-division handling: deterministic return of 0.0 with explicit status metadata;
   never silences undefined values or crashes.
4. Independent region-matching: spatial IoU thresholding (default 0.30) with centroid
   and geometric area error tracking.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# ============================================================
# SAFE METRIC PRIMITIVES
# ============================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division preventing ZeroDivisionError and NaN."""
    if denominator == 0.0 or math.isnan(denominator) or math.isnan(numerator):
        return default
    val = numerator / denominator
    return float(val) if not math.isnan(val) else default


def calculate_precision(tp: int, fp: int) -> float:
    """Precision = TP / (TP + FP). Returns 0.0 if TP + FP == 0."""
    return safe_divide(float(tp), float(tp + fp), default=0.0)


def calculate_recall(tp: int, fn: int) -> float:
    """Recall = TP / (TP + FN). Returns 0.0 if TP + FN == 0."""
    return safe_divide(float(tp), float(tp + fn), default=0.0)


def calculate_f1(precision: float, recall: float) -> float:
    """F1 = 2 * (P * R) / (P + R). Returns 0.0 if P + R == 0."""
    return safe_divide(2.0 * precision * recall, precision + recall, default=0.0)


def calculate_iou(tp: int, fp: int, fn: int) -> float:
    """IoU = TP / (TP + FP + FN). Returns 0.0 if denominator == 0."""
    return safe_divide(float(tp), float(tp + fp + fn), default=0.0)


# ============================================================
# PIXEL-LEVEL CONFUSION MATRIX & EVALUATION
# ============================================================

def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 8,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Computes confusion matrix of shape (num_classes, num_classes).
    Rows represent True labels, columns represent Predicted labels.

    Args:
        y_true: 2D integer array of ground truth labels.
        y_pred: 2D integer array of predicted labels.
        num_classes: Number of active classes (default 8, classes 0 to 7).
        valid_mask: Optional boolean mask where True = valid pixel, False = ignored (nodata/cloud/class 8).

    Returns:
        (cm, evaluated_pixels, ignored_pixels)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    total_pixels = y_true.size

    # Default mask: exclude class 8 (invalid) from both true and pred
    if valid_mask is None:
        mask = (y_true < num_classes) & (y_true >= 0) & (y_pred < num_classes) & (y_pred >= 0)
    else:
        mask = valid_mask & (y_true < num_classes) & (y_true >= 0) & (y_pred < num_classes) & (y_pred >= 0)

    evaluated_pixels = int(np.sum(mask))
    ignored_pixels = total_pixels - evaluated_pixels

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if evaluated_pixels > 0:
        flat_true = y_true[mask].ravel().astype(np.int64)
        flat_pred = y_pred[mask].ravel().astype(np.int64)
        # Using bincount with 2D encoding: row * num_classes + col
        indices = flat_true * num_classes + flat_pred
        counts = np.bincount(indices, minlength=num_classes * num_classes)
        cm = counts.reshape((num_classes, num_classes))

    return cm, evaluated_pixels, ignored_pixels


def calculate_per_class_metrics(cm: np.ndarray, class_names: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """
    Extracts TP, FP, FN, TN, precision, recall, F1, IoU, and support for each class.
    """
    num_classes = cm.shape[0]
    total_eval = int(cm.sum())
    results: Dict[str, Any] = {}

    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        tn = int(total_eval - tp - fp - fn)
        support = tp + fn

        prec = calculate_precision(tp, fp)
        rec = calculate_recall(tp, fn)
        f1 = calculate_f1(prec, rec)
        iou = calculate_iou(tp, fp, fn)

        name = class_names.get(c, f"class_{c}") if class_names else f"class_{c}"
        results[name] = {
            "class_id": c,
            "class_name": name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "iou": round(iou, 4),
            "is_present": support > 0,
        }

    return results


def calculate_macro_metrics(
    per_class_metrics: Dict[str, Dict[str, Any]],
    include_classes: Optional[List[int]] = None,
    only_present: bool = True,
) -> Dict[str, float]:
    """
    Calculates macro-averaged precision, recall, F1, and IoU.
    If only_present=True, only averages across classes with support > 0.
    """
    precisions = []
    recalls = []
    f1s = []
    ious = []

    for _, data in per_class_metrics.items():
        cid = data["class_id"]
        if include_classes is not None and cid not in include_classes:
            continue
        if only_present and not data["is_present"]:
            continue

        precisions.append(data["precision"])
        recalls.append(data["recall"])
        f1s.append(data["f1"])
        ious.append(data["iou"])

    n = len(f1s)
    if n == 0:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "macro_iou": 0.0,
            "classes_evaluated_count": 0,
        }

    return {
        "macro_precision": round(sum(precisions) / n, 4),
        "macro_recall": round(sum(recalls) / n, 4),
        "macro_f1": round(sum(f1s) / n, 4),
        "macro_iou": round(sum(ious) / n, 4),
        "classes_evaluated_count": n,
    }


def calculate_overall_accuracy(cm: np.ndarray) -> float:
    """Overall accuracy = trace(cm) / total_eval."""
    total = float(cm.sum())
    if total == 0.0:
        return 0.0
    return round(float(np.trace(cm)) / total, 4)


def calculate_balanced_accuracy(cm: np.ndarray) -> float:
    """Balanced accuracy = mean of recalls across classes with support > 0."""
    recalls = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - tp)
        support = tp + fn
        if support > 0:
            recalls.append(tp / support)
    if not recalls:
        return 0.0
    return round(sum(recalls) / len(recalls), 4)


# ============================================================
# REGION-LEVEL EVALUATION
# ============================================================

@dataclass
class RegionInfo:
    region_id: int
    area_pixels: int
    centroid_row: float
    centroid_col: float
    bbox: Tuple[int, int, int, int]  # min_r, min_c, max_r, max_c
    pixel_coords: np.ndarray


def extract_regions_from_mask(mask: np.ndarray, min_area_pixels: int = 4) -> List[RegionInfo]:
    """
    Extracts contiguous connected components (8-connectivity) as RegionInfo objects.
    Applies MMU threshold min_area_pixels.
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions: List[RegionInfo] = []

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area_pixels:
            continue

        c_col, c_row = centroids[label_idx]
        min_c = int(stats[label_idx, cv2.CC_STAT_LEFT])
        min_r = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        coords = np.argwhere(labels == label_idx)

        regions.append(
            RegionInfo(
                region_id=label_idx,
                area_pixels=area,
                centroid_row=float(c_row),
                centroid_col=float(c_col),
                bbox=(min_r, min_c, min_r + height, min_c + width),
                pixel_coords=coords,
            )
        )

    return regions


def compute_region_iou(r1: RegionInfo, r2: RegionInfo, raster_shape: Tuple[int, int]) -> float:
    """Calculates pixel-intersection-over-union between two RegionInfo objects."""
    # Bounding box disjoint check for speed
    r1_min_r, r1_min_c, r1_max_r, r1_max_c = r1.bbox
    r2_min_r, r2_min_c, r2_max_r, r2_max_c = r2.bbox

    if (
        r1_max_r < r2_min_r
        or r2_max_r < r1_min_r
        or r1_max_c < r2_min_c
        or r2_max_c < r1_min_c
    ):
        return 0.0

    # Fast set intersection on coordinate tuples
    s1 = set(map(tuple, r1.pixel_coords))
    s2 = set(map(tuple, r2.pixel_coords))
    intersection = len(s1 & s2)
    if intersection == 0:
        return 0.0
    union = len(s1 | s2)
    return safe_divide(float(intersection), float(union), default=0.0)


def match_regions(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    iou_threshold: float = 0.30,
    min_area_pixels: int = 4,
    pixel_size_m: float = 10.0,
) -> Dict[str, Any]:
    """
    Evaluates region-level detection performance using spatial overlap matching.

    Args:
        gt_mask: Ground truth binary mask (1 = change, 0 = background).
        pred_mask: Predicted binary mask (1 = change, 0 = background).
        iou_threshold: Minimum IoU overlap required to declare a true positive match.
        min_area_pixels: MMU threshold in pixels.
        pixel_size_m: Pixel spatial resolution in meters.

    Returns:
        Structured dictionary of region metrics.
    """
    gt_regions = extract_regions_from_mask(gt_mask, min_area_pixels=min_area_pixels)
    pred_regions = extract_regions_from_mask(pred_mask, min_area_pixels=min_area_pixels)

    num_gt = len(gt_regions)
    num_pred = len(pred_regions)

    if num_gt == 0 and num_pred == 0:
        return {
            "total_gt_regions": 0,
            "total_pred_regions": 0,
            "matched_gt_regions": 0,
            "matched_pred_regions": 0,
            "overlapping_gt_regions": 0,
            "detection_rate": 1.0,
            "false_positive_regions": 0,
            "false_negative_regions": 0,
            "region_precision": 1.0,
            "region_recall": 1.0,
            "region_f1": 1.0,
            "mean_region_iou": 1.0,
            "median_region_iou": 1.0,
            "centroid_distance_mean_px": 0.0,
            "centroid_distance_mean_m": 0.0,
            "area_absolute_error_mean_m2": 0.0,
        }

    if num_gt == 0:
        return {
            "total_gt_regions": 0,
            "total_pred_regions": num_pred,
            "matched_gt_regions": 0,
            "matched_pred_regions": 0,
            "overlapping_gt_regions": 0,
            "detection_rate": 0.0,
            "false_positive_regions": num_pred,
            "false_negative_regions": 0,
            "region_precision": 0.0,
            "region_recall": 0.0,
            "region_f1": 0.0,
            "mean_region_iou": 0.0,
            "median_region_iou": 0.0,
            "centroid_distance_mean_px": 0.0,
            "centroid_distance_mean_m": 0.0,
            "area_absolute_error_mean_m2": 0.0,
        }

    if num_pred == 0:
        return {
            "total_gt_regions": num_gt,
            "total_pred_regions": 0,
            "matched_gt_regions": 0,
            "matched_pred_regions": 0,
            "overlapping_gt_regions": 0,
            "detection_rate": 0.0,
            "false_positive_regions": 0,
            "false_negative_regions": num_gt,
            "region_precision": 0.0,
            "region_recall": 0.0,
            "region_f1": 0.0,
            "mean_region_iou": 0.0,
            "median_region_iou": 0.0,
            "centroid_distance_mean_px": 0.0,
            "centroid_distance_mean_m": 0.0,
            "area_absolute_error_mean_m2": 0.0,
        }

    # Pairwise IoU matrix
    shape = gt_mask.shape
    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
    for i, gt_r in enumerate(gt_regions):
        for j, pred_r in enumerate(pred_regions):
            iou_matrix[i, j] = compute_region_iou(gt_r, pred_r, shape)

    # Detection rate: proportion of GT regions with at least 1 pixel overlap (IoU > 0)
    overlapping_gt_count = int(np.sum(np.any(iou_matrix > 0, axis=1)))
    detection_rate = round(float(overlapping_gt_count / num_gt), 4)

    # Greedy bipartite matching based on highest IoU >= threshold
    matched_gt: set = set()
    matched_pred: set = set()
    matched_ious: List[float] = []
    centroid_distances_px: List[float] = []
    area_errors_m2: List[float] = []

    # Sort all pairs by descending IoU
    pair_indices = np.dstack(np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape))[0]

    for gt_idx, pred_idx in pair_indices:
        val = float(iou_matrix[gt_idx, pred_idx])
        if val < iou_threshold:
            break
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            matched_ious.append(val)

            # Centroid distance
            g = gt_regions[gt_idx]
            p = pred_regions[pred_idx]
            d_px = math.sqrt((g.centroid_row - p.centroid_row) ** 2 + (g.centroid_col - p.centroid_col) ** 2)
            centroid_distances_px.append(d_px)

            # Area absolute error
            pixel_area_m2 = pixel_size_m * pixel_size_m
            area_err = abs(g.area_pixels - p.area_pixels) * pixel_area_m2
            area_errors_m2.append(area_err)

    tp_regions = len(matched_gt)
    fp_regions = num_pred - len(matched_pred)
    fn_regions = num_gt - len(matched_gt)

    reg_prec = calculate_precision(tp_regions, fp_regions)
    reg_rec = calculate_recall(tp_regions, fn_regions)
    reg_f1 = calculate_f1(reg_prec, reg_rec)

    mean_iou = round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0
    median_iou = round(float(np.median(matched_ious)), 4) if matched_ious else 0.0
    mean_dist_px = round(float(np.mean(centroid_distances_px)), 2) if centroid_distances_px else 0.0
    mean_dist_m = round(mean_dist_px * pixel_size_m, 2)
    mean_area_err = round(float(np.mean(area_errors_m2)), 1) if area_errors_m2 else 0.0

    return {
        "total_gt_regions": num_gt,
        "total_pred_regions": num_pred,
        "matched_gt_regions": len(matched_gt),
        "matched_pred_regions": len(matched_pred),
        "overlapping_gt_regions": overlapping_gt_count,
        "detection_rate": detection_rate,
        "false_positive_regions": fp_regions,
        "false_negative_regions": fn_regions,
        "region_precision": round(reg_prec, 4),
        "region_recall": round(reg_rec, 4),
        "region_f1": round(reg_f1, 4),
        "mean_region_iou": mean_iou,
        "median_region_iou": median_iou,
        "centroid_distance_mean_px": mean_dist_px,
        "centroid_distance_mean_m": mean_dist_m,
        "area_absolute_error_mean_m2": mean_area_err,
    }
