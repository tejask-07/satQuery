from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
import rasterio

from app.remote_sensing.preprocessing.quality import SCLClass

VISUALIZATION_DIR = Path(__file__).resolve().parent / "visualizations"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISPLAY NORMALIZATION & STRETCHING
# ============================================================

def normalize_channel_for_display(
    band_array: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    gamma: float = 1.15,
) -> np.ndarray:
    """
    Apply a scientific display stretch to a single surface reflectance channel.

    NEVER modifies the input array. Uses in-memory copies exclusively.

    Parameters
    ----------
    band_array : np.ndarray
        2D float reflectance raster.
    valid_mask : np.ndarray, optional
        Boolean mask where True indicates valid pixels.
    percentile_low : float
        Lower cutoff percentile (default: 2.0).
    percentile_high : float
        Upper cutoff percentile (default: 98.0).
    gamma : float
        Gamma brightness adjustment curve (default: 1.15).

    Returns
    -------
    np.ndarray
        Uint8 display channel (0 to 255).
    """
    arr = np.asarray(band_array, dtype=np.float32).copy()
    finite_mask = np.isfinite(arr) & (arr >= 0.0)

    if valid_mask is not None:
        eval_mask = finite_mask & np.asarray(valid_mask, dtype=bool)
    else:
        eval_mask = finite_mask

    if np.any(eval_mask):
        p_low = float(np.percentile(arr[eval_mask], percentile_low))
        p_high = float(np.percentile(arr[eval_mask], percentile_high))
        if p_high <= p_low:
            p_low = 0.0
            p_high = max(0.30, float(np.max(arr[eval_mask])))
    else:
        p_low = 0.0
        p_high = 0.35

    # Safe linear stretch
    p_high = max(p_high, p_low + 1e-4)
    stretched = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)

    # Gamma correction for natural human visual perception
    if gamma != 1.0:
        stretched = np.power(stretched, 1.0 / gamma)

    stretched_clean = np.nan_to_num(stretched, nan=0.0)
    return (stretched_clean * 255.0).astype(np.uint8)



# ============================================================
# COMPOSITE BUILDERS (TRUE COLOR & FALSE COLOR)
# ============================================================

def build_true_color_rgba(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a Sentinel-2 True-Color (RGB: B04, B03, B02) display composite.

    Returns a 4-channel BGRA uint8 image with alpha transparency for invalid pixels.
    NEVER modifies input reflectance arrays.
    """
    r_in = np.asarray(red, dtype=np.float32)
    g_in = np.asarray(green, dtype=np.float32)
    b_in = np.asarray(blue, dtype=np.float32)

    if r_in.shape != g_in.shape or r_in.shape != b_in.shape:
        raise ValueError(
            f"Shape mismatch in true-color composite: red {r_in.shape}, green {g_in.shape}, blue {b_in.shape}"
        )

    # Determine pixel validity
    finite = np.isfinite(r_in) & np.isfinite(g_in) & np.isfinite(b_in)
    if valid_mask is not None:
        effective_valid = finite & np.asarray(valid_mask, dtype=bool)
    else:
        effective_valid = finite

    # Display-only stretch on each channel
    r_disp = normalize_channel_for_display(r_in, valid_mask=effective_valid, gamma=1.15)
    g_disp = normalize_channel_for_display(g_in, valid_mask=effective_valid, gamma=1.15)
    b_disp = normalize_channel_for_display(b_in, valid_mask=effective_valid, gamma=1.15)

    # Alpha channel: 255 for valid surface, 0 for invalid/cloud/shadow/nodata
    alpha = np.full(r_in.shape, 255, dtype=np.uint8)
    alpha[~effective_valid] = 0

    # OpenCV BGRA: [Blue, Green, Red, Alpha]
    return np.stack([b_disp, g_disp, r_disp, alpha], axis=-1)


def build_false_color_rgba(
    nir: np.ndarray,
    red: np.ndarray,
    green: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build an optical False-Color NIR composite (NIR = B08, Red = B04, Green = B03).

    Vegetation biomass appears vivid red, urban appears cyan/gray, water appears dark.
    Returns 4-channel BGRA uint8 image. NEVER modifies input rasters.
    """
    nir_in = np.asarray(nir, dtype=np.float32)
    red_in = np.asarray(red, dtype=np.float32)
    green_in = np.asarray(green, dtype=np.float32)

    if nir_in.shape != red_in.shape or nir_in.shape != green_in.shape:
        raise ValueError(
            f"Shape mismatch in false-color composite: nir {nir_in.shape}, red {red_in.shape}, green {green_in.shape}"
        )

    finite = np.isfinite(nir_in) & np.isfinite(red_in) & np.isfinite(green_in)
    if valid_mask is not None:
        effective_valid = finite & np.asarray(valid_mask, dtype=bool)
    else:
        effective_valid = finite

    # Red channel = NIR (B08)
    r_disp = normalize_channel_for_display(nir_in, valid_mask=effective_valid, gamma=1.10)
    # Green channel = Red (B04)
    g_disp = normalize_channel_for_display(red_in, valid_mask=effective_valid, gamma=1.15)
    # Blue channel = Green (B03)
    b_disp = normalize_channel_for_display(green_in, valid_mask=effective_valid, gamma=1.15)

    alpha = np.full(nir_in.shape, 255, dtype=np.uint8)
    alpha[~effective_valid] = 0

    return np.stack([b_disp, g_disp, r_disp, alpha], axis=-1)


# ============================================================
# SCIENTIFIC INDEX VISUALIZATIONS (NDVI, NDWI, NDBI)
# ============================================================

def build_index_rgba(
    index_data: np.ndarray,
    index_name: str,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate an authoritative scientific colormap overlay for an index raster.

    The numeric meaning is preserved with transparent background for masked regions.
    NEVER modifies the underlying index array.
    """
    idx = np.asarray(index_data, dtype=np.float32).copy()
    height, width = idx.shape

    finite = np.isfinite(idx)
    if valid_mask is not None:
        effective_valid = finite & np.asarray(valid_mask, dtype=bool)
    else:
        effective_valid = finite

    name = index_name.upper()

    if name == "NDVI":
        # Anchors for vegetation: lower signal -> higher signal
        anchors = [-0.10, 0.00, 0.18, 0.35, 0.55, 0.80]
        # BGR tuples
        b_anch = [180, 140, 90,  60,  40,  20]
        g_anch = [180, 190, 180, 200, 160, 100]
        r_anch = [180, 210, 190, 140,  50,  20]

    elif name == "NDWI":
        # Anchors for water: lower response -> higher response
        anchors = [-0.40, -0.10, 0.00, 0.15, 0.40]
        b_anch = [140, 180, 220, 240, 220]
        g_anch = [150, 180, 195, 160,  70]
        r_anch = [160, 170, 130,  50,  15]

    elif name == "NDBI":
        # Anchors for built-up: lower response -> higher response
        anchors = [-0.35, -0.10, 0.00, 0.12, 0.35]
        b_anch = [100, 160, 170,  70, 180]
        g_anch = [160, 170, 180, 140,  40]
        r_anch = [100, 150, 210, 240, 200]

    else:
        # Default grayscale with contrast
        anchors = [-1.0, 0.0, 1.0]
        b_anch = [40, 128, 240]
        g_anch = [40, 128, 240]
        r_anch = [40, 128, 240]

    flat_idx = np.nan_to_num(idx.flatten(), nan=0.0)

    b = np.interp(flat_idx, anchors, b_anch).reshape(height, width).astype(np.uint8)
    g = np.interp(flat_idx, anchors, g_anch).reshape(height, width).astype(np.uint8)
    r = np.interp(flat_idx, anchors, r_anch).reshape(height, width).astype(np.uint8)

    alpha = np.full((height, width), 225, dtype=np.uint8)
    alpha[~effective_valid] = 0

    return np.stack([b, g, r, alpha], axis=-1)


# ============================================================
# RAW & CLASSIFIED CHANGE VISUALIZATIONS
# ============================================================

def build_raw_change_rgba(
    change_map: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Build a scientific continuous diverging change layer.

    - Decrease: cool blue/cyan
    - Neutral (|delta| < threshold): transparent so base imagery remains visible
    - Increase: warm amber/orange
    - Masked/NaN: transparent (alpha = 0)
    """
    c_map = np.asarray(change_map, dtype=np.float32).copy()
    height, width = c_map.shape

    finite = np.isfinite(c_map)
    if valid_mask is not None:
        effective_valid = finite & np.asarray(valid_mask, dtype=bool)
    else:
        effective_valid = finite

    th = max(0.005, float(threshold))
    th_mod = max(0.08, th * 2.5)
    th_high = max(0.20, th * 5.0)

    anchors = [-th_high, -th_mod, -th, 0.0, th, th_mod, th_high]
    b_anch = [240, 240, 200, 128, 50,  30,  20]
    g_anch = [80,  150, 190, 128, 150, 110, 40]
    r_anch = [30,  40,  80,  128, 240, 240, 240]
    # Neutral zone inside [-th, th] is transparent
    a_anch = [230, 210, 80,  0,   80,  210, 230]

    flat_c = np.nan_to_num(c_map.flatten(), nan=0.0)

    b = np.interp(flat_c, anchors, b_anch).reshape(height, width).astype(np.uint8)
    g = np.interp(flat_c, anchors, g_anch).reshape(height, width).astype(np.uint8)
    r = np.interp(flat_c, anchors, r_anch).reshape(height, width).astype(np.uint8)
    a = np.interp(flat_c, anchors, a_anch).reshape(height, width).astype(np.uint8)

    # Enforce exact transparency for invalid or near-zero changes
    a[~effective_valid] = 0

    return np.stack([b, g, r, a], axis=-1)


def build_classified_change_rgba(
    change_map: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Build discrete classified change categories layer (no semantic claims).

    Labels:
    - High Decrease
    - Moderate Decrease
    - Slight Decrease
    - No Significant Change (transparent)
    - Slight Increase
    - Moderate Increase
    - High Increase
    """
    c_map = np.asarray(change_map, dtype=np.float32).copy()
    height, width = c_map.shape

    finite = np.isfinite(c_map)
    if valid_mask is not None:
        effective_valid = finite & np.asarray(valid_mask, dtype=bool)
    else:
        effective_valid = finite

    th_slight = max(0.005, float(threshold))
    th_mod = max(0.08, th_slight * 2.5)
    th_high = max(0.20, th_slight * 5.0)

    bgra = np.zeros((height, width, 4), dtype=np.uint8)

    # Decreases (Blue/Cyan tones)
    m_hd = effective_valid & (c_map < -th_high)
    bgra[m_hd] = [235, 75, 30, 230]   # High decrease

    m_md = effective_valid & (c_map >= -th_high) & (c_map < -th_mod)
    bgra[m_md] = [240, 150, 40, 215]  # Moderate decrease

    m_sd = effective_valid & (c_map >= -th_mod) & (c_map <= -th_slight)
    bgra[m_sd] = [245, 205, 70, 195]  # Slight decrease

    # Increases (Orange/Red tones)
    m_si = effective_valid & (c_map >= th_slight) & (c_map < th_mod)
    bgra[m_si] = [60, 185, 245, 195]  # Slight increase

    m_mi = effective_valid & (c_map >= th_mod) & (c_map < th_high)
    bgra[m_mi] = [40, 120, 240, 215]  # Moderate increase

    m_hi = effective_valid & (c_map >= th_high)
    bgra[m_hi] = [20, 45, 235, 230]   # High increase

    # No change & invalid remain completely transparent
    return bgra


# ============================================================
# QUALITY MASK VISUALIZATION
# ============================================================

def build_quality_mask_rgba(
    scl_raster: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Build a visual quality inspection overlay.

    - Clouds: bright white/cyan
    - Cirrus: translucent cyan
    - Shadows: dark charcoal/slate
    - Valid ground: subtle transparent green
    - Nodata: transparent
    """
    if scl_raster is not None:
        scl = np.asarray(scl_raster, dtype=np.int32)
        height, width = scl.shape
    elif valid_mask is not None:
        height, width = valid_mask.shape
        scl = None
    elif target_shape is not None:
        height, width = target_shape
        scl = None
    else:
        raise ValueError("At least one of scl_raster, valid_mask, or target_shape required.")

    bgra = np.zeros((height, width, 4), dtype=np.uint8)

    if scl is not None:
        # Clouds: High & Medium Probability
        m_cloud = (scl == SCLClass.CLOUD_HIGH_PROBABILITY) | (scl == SCLClass.CLOUD_MEDIUM_PROBABILITY)
        bgra[m_cloud] = [245, 245, 245, 230]

        # Thin Cirrus
        m_cirrus = (scl == SCLClass.THIN_CIRRUS)
        bgra[m_cirrus] = [250, 220, 180, 190]

        # Shadows: Cloud & Cast Shadows
        m_shadow = (scl == SCLClass.CLOUD_SHADOWS) | (scl == SCLClass.DARK_AREA_PIXELS)
        bgra[m_shadow] = [60, 50, 50, 210]

        # Valid Surface
        m_valid = (
            (scl == SCLClass.VEGETATION)
            | (scl == SCLClass.NOT_VEGETATED)
            | (scl == SCLClass.WATER)
            | (scl == SCLClass.UNCLASSIFIED)
            | (scl == SCLClass.SNOW_ICE)
        )
        bgra[m_valid] = [80, 180, 60, 70]  # Subtle green wash

    elif valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        bgra[vm] = [80, 180, 60, 70]
        bgra[~vm] = [245, 245, 245, 210]

    return bgra


# ============================================================
# PERSISTENCE & GEOREFERENCING
# ============================================================

def save_visualization_layer(
    image_bgra: np.ndarray,
    filename: str,
    source_raster_path: Optional[str] = None,
    aoi_bbox: Optional[Tuple[float, float, float, float]] = None,
    upscale_factor: int = 2,
) -> Dict[str, Any]:
    """
    Save a georeferenced RGBA display PNG and return complete layer metadata.

    Enlarges with nearest-neighbor interpolation to retain pixel clarity.
    Extracts exact geographic coordinates in [[south, west], [north, east]].
    """
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VISUALIZATION_DIR / filename

    height, width = image_bgra.shape[:2]

    # Optional nearest-neighbor upscale for crisp web viewing without smoothing
    if upscale_factor > 1:
        target_w = max(1, width * upscale_factor)
        target_h = max(1, height * upscale_factor)
        disp_img = cv2.resize(
            image_bgra,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        disp_img = image_bgra
        target_w, target_h = width, height

    success = cv2.imwrite(str(out_path), disp_img)
    if not success:
        raise IOError(f"Failed to write visualization to {out_path}")

    # Extract exact geographic bounds
    bounds = None
    if source_raster_path and Path(source_raster_path).exists():
        try:
            with rasterio.open(source_raster_path) as src:
                b = src.bounds
                # Leaflet convention: [[south, west], [north, east]] = [[min_lat, min_lng], [max_lat, max_lng]]
                bounds = [
                    [float(b.bottom), float(b.left)],
                    [float(b.top), float(b.right)],
                ]
        except Exception:
            bounds = None

    if bounds is None and aoi_bbox is not None:
        w, s, e, n = aoi_bbox
        bounds = [
            [float(s), float(w)],
            [float(n), float(e)],
        ]

    return {
        "status": "success",
        "filename": filename,
        "path": str(out_path.resolve()),
        "url": f"/visualizations/{filename}",
        "bounds": bounds,
        "width": target_w,
        "height": target_h,
        "source_width": width,
        "source_height": height,
        "crs": "EPSG:4326" if bounds is not None else None,
    }
