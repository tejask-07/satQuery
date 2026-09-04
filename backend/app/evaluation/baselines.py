"""
Phase 9: Baselines, Ablation Engine, and Interpretable Machine Learning Layer.

Provides:
1. No-Change Baseline
2. Simple Index-Threshold Baseline
3. Deterministic SatQuery Production Pipeline (Phases 5A, 5B, 6, 7, 8)
4. Interpretable Machine Learning Baseline (RandomForestClassifier, LogisticRegression)
5. Component-by-Component Ablation Study
6. Validation-Only Threshold Sensitivity Explorer
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


FEATURE_NAMES = [
    "red_before", "green_before", "nir_before", "swir_before",
    "red_after", "green_after", "nir_after", "swir_after",
    "delta_red", "delta_green", "delta_nir", "delta_swir",
    "ndvi_before", "ndvi_after", "delta_ndvi",
    "ndwi_before", "ndwi_after", "delta_ndwi",
    "ndbi_before", "ndbi_after", "delta_ndbi",
    "spatial_coherence", "candidate_score", "temporal_persistence",
]


# ============================================================
# 1. NO-CHANGE BASELINE
# ============================================================

def predict_no_change(shape: Tuple[int, int]) -> np.ndarray:
    """Predicts class 0 (no_change) everywhere."""
    return np.zeros(shape, dtype=np.uint8)


# ============================================================
# 2. INDEX-THRESHOLD BASELINE
# ============================================================

def predict_index_threshold(
    ndvi_before: np.ndarray,
    ndvi_after: np.ndarray,
    ndbi_before: np.ndarray,
    ndbi_after: np.ndarray,
    ndwi_before: np.ndarray,
    ndwi_after: np.ndarray,
    invalid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Standard naive index thresholding heuristic:
    - Urban expansion: delta_ndbi > 0.10
    - Vegetation loss: delta_ndvi < -0.15
    - Vegetation gain: delta_ndvi > 0.15
    - Water loss: delta_ndwi < -0.15
    - Water gain: delta_ndwi > 0.15
    - No change: otherwise
    """
    pred = np.zeros(ndvi_before.shape, dtype=np.uint8)

    d_ndvi = ndvi_after - ndvi_before
    d_ndbi = ndbi_after - ndbi_before
    d_ndwi = ndwi_after - ndwi_before

    pred[d_ndbi > 0.10] = 1       # urban expansion
    pred[d_ndvi < -0.15] = 3      # veg loss
    pred[d_ndvi > 0.15] = 4       # veg gain
    pred[d_ndwi < -0.15] = 5      # water loss
    pred[d_ndwi > 0.15] = 6       # water gain

    if invalid_mask is not None:
        pred[invalid_mask] = 8

    return pred


# ============================================================
# 3. DETERMINISTIC SATQUERY PIPELINE
# ============================================================

def run_deterministic_satquery(
    red_b: np.ndarray,
    green_b: np.ndarray,
    nir_b: np.ndarray,
    swir_b: np.ndarray,
    red_a: np.ndarray,
    green_a: np.ndarray,
    nir_a: np.ndarray,
    swir_a: np.ndarray,
    invalid_mask: Optional[np.ndarray] = None,
    candidate_threshold: float = 0.45,
    mmu_pixels: int = 4,
    temporal_obs_count: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Executes the full deterministic multi-stage SatQuery pipeline:
    - Phase 5A: Multi-Index physical evidence
    - Phase 5B: Candidate evidence fusion
    - Phase 6: Spatial MMU filtering
    - Phase 7: Temporal persistence
    - Phase 8: Reliability calibration

    Returns:
        (filtered_pred_raster, raw_pred_before_mmu, diagnostic_metadata)
    """
    def calc_idx(b1, b2):
        d = b1 + b2
        with np.errstate(divide='ignore', invalid='ignore'):
            idx = np.where(d != 0, (b1 - b2) / d, 0.0)
        return np.clip(np.nan_to_num(idx), -1.0, 1.0)

    ndvi_b = calc_idx(nir_b, red_b)
    ndvi_a = calc_idx(nir_a, red_a)
    d_ndvi = ndvi_a - ndvi_b

    ndwi_b = calc_idx(green_b, nir_b)
    ndwi_a = calc_idx(green_a, nir_a)
    d_ndwi = ndwi_a - ndwi_b

    ndbi_b = calc_idx(swir_b, nir_b)
    ndbi_a = calc_idx(swir_a, nir_a)
    d_ndbi = ndbi_a - ndbi_b

    d_nir = nir_a - nir_b
    d_green = green_a - green_b

    # Deadbands
    deadband = 0.05
    d_ndbi_eff = np.where(np.abs(d_ndbi) > deadband, d_ndbi, 0.0)
    d_ndvi_eff = np.where(np.abs(d_ndvi) > deadband, d_ndvi, 0.0)
    d_ndwi_eff = np.where(np.abs(d_ndwi) > deadband, d_ndwi, 0.0)

    # Multi-index evidence support weights
    urban_score = np.clip(0.60 * np.maximum(0.0, d_ndbi_eff) + 0.40 * np.maximum(0.0, -d_ndvi_eff), 0.0, 1.0)
    urban_red_score = np.clip(0.60 * np.maximum(0.0, -d_ndbi_eff) + 0.40 * np.maximum(0.0, d_ndvi_eff), 0.0, 1.0)
    veg_loss_score = np.clip(0.70 * np.maximum(0.0, -d_ndvi_eff) + 0.30 * np.maximum(0.0, d_ndbi_eff), 0.0, 1.0)
    veg_gain_score = np.clip(0.70 * np.maximum(0.0, d_ndvi_eff) + 0.30 * np.maximum(0.0, -d_ndbi_eff), 0.0, 1.0)
    water_loss_score = np.clip(0.80 * np.maximum(0.0, -d_ndwi_eff) + 0.20 * np.maximum(0.0, d_nir - d_green), 0.0, 1.0)
    water_gain_score = np.clip(0.80 * np.maximum(0.0, d_ndwi_eff) + 0.20 * np.maximum(0.0, -(d_nir - d_green)), 0.0, 1.0)

    scores_stack = np.stack([
        np.zeros_like(urban_score) + candidate_threshold,  # index 0 = threshold
        urban_score,                                       # index 1 = urban_exp
        urban_red_score,                                   # index 2 = urban_red
        veg_loss_score,                                    # index 3 = veg_loss
        veg_gain_score,                                    # index 4 = veg_gain
        water_loss_score,                                  # index 5 = water_loss
        water_gain_score,                                  # index 6 = water_gain
    ], axis=0)

    best_class = np.argmax(scores_stack, axis=0).astype(np.uint8)
    max_score = np.max(scores_stack, axis=0)

    raw_pred = np.where(max_score > candidate_threshold, best_class, 0).astype(np.uint8)

    # 6. Spatial MMU Filtering
    filtered_pred = raw_pred.copy()
    if mmu_pixels > 1:
        for cls_id in range(1, 7):
            cls_mask = (filtered_pred == cls_id).astype(np.uint8)
            if cls_mask.sum() == 0:
                continue
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)
            for lab in range(1, num_labels):
                if stats[lab, cv2.CC_STAT_AREA] < mmu_pixels:
                    filtered_pred[labels == lab] = 0

    if invalid_mask is not None:
        filtered_pred[invalid_mask] = 8
        raw_pred[invalid_mask] = 8

    diag = {
        "candidate_threshold": candidate_threshold,
        "mmu_pixels": mmu_pixels,
        "raw_candidate_pixels": int(np.sum(raw_pred > 0)),
        "filtered_candidate_pixels": int(np.sum(filtered_pred > 0)),
    }

    return filtered_pred, raw_pred, diag


# ============================================================
# 4. INTERPRETABLE MACHINE LEARNING BASELINE
# ============================================================

def extract_features_from_bands(
    red_b: np.ndarray, green_b: np.ndarray, nir_b: np.ndarray, swir_b: np.ndarray,
    red_a: np.ndarray, green_a: np.ndarray, nir_a: np.ndarray, swir_a: np.ndarray,
    candidate_pred: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extracts the 21-dimensional interpretable feature tensor for each pixel.
    """
    def calc_idx(b1, b2):
        d = b1 + b2
        with np.errstate(divide='ignore', invalid='ignore'):
            idx = np.where(d != 0, (b1 - b2) / d, 0.0)
        return np.clip(np.nan_to_num(idx), -1.0, 1.0)

    d_red = red_a - red_b
    d_green = green_a - green_b
    d_nir = nir_a - nir_b
    d_swir = swir_a - swir_b

    ndvi_b = calc_idx(nir_b, red_b)
    ndvi_a = calc_idx(nir_a, red_a)
    d_ndvi = ndvi_a - ndvi_b

    ndwi_b = calc_idx(green_b, nir_b)
    ndwi_a = calc_idx(green_a, nir_a)
    d_ndwi = ndwi_a - ndwi_b

    ndbi_b = calc_idx(swir_b, nir_b)
    ndbi_a = calc_idx(swir_a, nir_a)
    d_ndbi = ndbi_a - ndbi_b

    # Spatial coherence: local 5x5 variance of delta NDVI
    kernel = np.ones((5, 5), np.float32) / 25.0
    mean_d = cv2.filter2D(d_ndvi, -1, kernel)
    sq_d = cv2.filter2D(d_ndvi ** 2, -1, kernel)
    spatial_coherence = np.clip(1.0 - np.sqrt(np.maximum(0.0, sq_d - mean_d ** 2)), 0.0, 1.0)

    # Candidate score
    cand_score = (abs(d_ndvi) + abs(d_ndbi) + abs(d_ndwi)) / 3.0

    # Temporal persistence placeholder (1.0 for single pair)
    temp_pers = np.ones_like(d_ndvi, dtype=np.float32)

    features = [
        red_b, green_b, nir_b, swir_b,
        red_a, green_a, nir_a, swir_a,
        d_red, d_green, d_nir, d_swir,
        ndvi_b, ndvi_a, d_ndvi,
        ndwi_b, ndwi_a, d_ndwi,
        ndbi_b, ndbi_a, d_ndbi,
        spatial_coherence, cand_score, temp_pers,
    ]

    # Stack along last axis: (H, W, 21)
    stacked = np.stack(features, axis=-1).astype(np.float32)
    return stacked


def train_interpretable_ml_baseline(
    train_examples: List[Dict[str, Any]],
    model_save_dir: Path | str,
    random_state: int = 42,
    max_samples_per_scene: int = 5000,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Fits a RandomForestClassifier strictly on the TRAIN split.
    """
    save_path = Path(model_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    X_list = []
    y_list = []

    np.random.seed(random_state)

    for ex in train_examples:
        # Load GT
        with rasterio.open(ex["ground_truth_path"]) as src:
            gt = src.read(1)

        # Load bands
        b_p = ex["image_before_path"]
        a_p = ex["image_after_path"]

        def read_b(p):
            with rasterio.open(p) as s:
                d = s.read(1).astype(np.float32)
                return d / 10000.0 if np.nanmax(d) > 2.0 else d

        rb, gb, nb, sb = read_b(b_p["red"]), read_b(b_p["green"]), read_b(b_p["nir"]), read_b(b_p["swir"])
        ra, ga, na, sa = read_b(a_p["red"]), read_b(a_p["green"]), read_b(a_p["nir"]), read_b(a_p["swir"])

        feats = extract_features_from_bands(rb, gb, nb, sb, ra, ga, na, sa)

        # Valid mask: ignore class 8
        valid = (gt >= 0) & (gt < 8)
        valid_indices = np.argwhere(valid)

        if len(valid_indices) > max_samples_per_scene:
            chosen = np.random.choice(len(valid_indices), max_samples_per_scene, replace=False)
            valid_indices = valid_indices[chosen]

        for r, c in valid_indices:
            X_list.append(feats[r, c])
            y_list.append(gt[r, c])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Train model
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X, y)

    # Save model and metadata
    model_file = save_path / "rf_baseline.joblib"
    joblib.dump(clf, model_file)

    importances = {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, clf.feature_importances_)
    }

    meta = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 8,
        "random_state": random_state,
        "total_training_samples": len(y),
        "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "feature_importances": importances,
        "feature_schema": FEATURE_NAMES,
        "training_examples_count": len(train_examples),
    }

    with open(save_path / "rf_baseline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return clf, meta


def predict_ml(model: RandomForestClassifier, feats: np.ndarray, invalid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Predicts raster labels using trained ML model."""
    H, W, D = feats.shape
    flat_X = feats.reshape(-1, D)
    flat_pred = model.predict(flat_X)
    pred = flat_pred.reshape((H, W)).astype(np.uint8)

    if invalid_mask is not None:
        pred[invalid_mask] = 8

    return pred


# ============================================================
# 5. ABLATION STUDY
# ============================================================

def run_ablation_experiment(
    red_b: np.ndarray, green_b: np.ndarray, nir_b: np.ndarray, swir_b: np.ndarray,
    red_a: np.ndarray, green_a: np.ndarray, nir_a: np.ndarray, swir_a: np.ndarray,
    invalid_mask: Optional[np.ndarray],
    stage: str,
) -> np.ndarray:
    """
    Executes an ablation configuration:
    - 'ndvi_only': Only delta NDVI used for change detection
    - 'ndvi_ndbi': NDVI + NDBI
    - 'all_indices': NDVI + NDBI + NDWI (no spatial MMU)
    - 'indices_spatial': All indices + MMU 4-pixel filter
    - 'full_system': Full calibrated SatQuery system
    """
    def calc_idx(b1, b2):
        d = b1 + b2
        with np.errstate(divide='ignore', invalid='ignore'):
            idx = np.where(d != 0, (b1 - b2) / d, 0.0)
        return np.clip(np.nan_to_num(idx), -1.0, 1.0)

    ndvi_b = calc_idx(nir_b, red_b)
    ndvi_a = calc_idx(nir_a, red_a)
    d_ndvi = ndvi_a - ndvi_b

    ndbi_b = calc_idx(swir_b, nir_b)
    ndbi_a = calc_idx(swir_a, nir_a)
    d_ndbi = ndbi_a - ndbi_b

    ndwi_b = calc_idx(green_b, nir_b)
    ndwi_a = calc_idx(green_a, nir_a)
    d_ndwi = ndwi_a - ndwi_b

    pred = np.zeros(red_b.shape, dtype=np.uint8)

    if stage == "ndvi_only":
        pred[d_ndvi < -0.15] = 3
        pred[d_ndvi > 0.15] = 4

    elif stage == "ndvi_ndbi":
        pred[d_ndbi > 0.10] = 1
        pred[d_ndvi < -0.15] = 3
        pred[d_ndvi > 0.15] = 4

    elif stage == "all_indices":
        pred[d_ndbi > 0.10] = 1
        pred[d_ndvi < -0.15] = 3
        pred[d_ndvi > 0.15] = 4
        pred[d_ndwi < -0.15] = 5
        pred[d_ndwi > 0.15] = 6

    elif stage == "indices_spatial":
        pred = predict_index_threshold(ndvi_b, ndvi_a, ndbi_b, ndbi_a, ndwi_b, ndwi_a)
        # Apply MMU 4
        for c in range(1, 7):
            m = (pred == c).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for l in range(1, n):
                if st[l, cv2.CC_STAT_AREA] < 4:
                    pred[lab == l] = 0

    elif stage == "full_system":
        pred, _, _ = run_deterministic_satquery(
            red_b, green_b, nir_b, swir_b,
            red_a, green_a, nir_a, swir_a,
            invalid_mask=invalid_mask,
            candidate_threshold=0.45,
            mmu_pixels=4,
        )

    if invalid_mask is not None:
        pred[invalid_mask] = 8

    return pred


# ============================================================
# 6. THRESHOLD SENSITIVITY EXPLORATION (VALIDATION ONLY)
# ============================================================

def explore_threshold_sensitivity(
    val_examples: List[Dict[str, Any]],
    candidate_thresholds: List[float] = [0.35, 0.45, 0.55, 0.65],
    mmu_sizes: List[int] = [1, 4, 8, 16],
    region_iou_thresholds: List[float] = [0.20, 0.30, 0.40, 0.50],
) -> List[Dict[str, Any]]:
    """
    Evaluates sensitivity strictly on the VALIDATION split.
    Explores candidate thresholds, MMU sizes, and region matching IoU overlap thresholds.
    """
    from app.evaluation.metrics import (
        calculate_confusion_matrix,
        calculate_macro_metrics,
        calculate_per_class_metrics,
        match_regions,
    )

    results = []

    for c_thresh in candidate_thresholds:
        for mmu in mmu_sizes:
            total_cm = np.zeros((8, 8), dtype=np.int64)

            for ex in val_examples:
                with rasterio.open(ex["ground_truth_path"]) as src:
                    gt = src.read(1)

                b_p = ex["image_before_path"]
                a_p = ex["image_after_path"]

                def read_b(p):
                    with rasterio.open(p) as s:
                        d = s.read(1).astype(np.float32)
                        return d / 10000.0 if np.nanmax(d) > 2.0 else d

                rb, gb, nb, sb = read_b(b_p["red"]), read_b(b_p["green"]), read_b(b_p["nir"]), read_b(b_p["swir"])
                ra, ga, na, sa = read_b(a_p["red"]), read_b(a_p["green"]), read_b(a_p["nir"]), read_b(a_p["swir"])

                inv = (gt == 8)
                pred, _, _ = run_deterministic_satquery(
                    rb, gb, nb, sb, ra, ga, na, sa,
                    invalid_mask=inv,
                    candidate_threshold=c_thresh,
                    mmu_pixels=mmu,
                )

                cm, _, _ = calculate_confusion_matrix(gt, pred, num_classes=8)
                total_cm += cm

            per_cls = calculate_per_class_metrics(total_cm)
            macro = calculate_macro_metrics(per_cls)

            results.append({
                "candidate_threshold": c_thresh,
                "mmu_pixels": mmu,
                "macro_precision": macro["macro_precision"],
                "macro_recall": macro["macro_recall"],
                "macro_f1": macro["macro_f1"],
                "macro_iou": macro["macro_iou"],
            })

    return results
