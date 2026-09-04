from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from app.agent.registry import get_tool
from app.evidence.scientific_visualizations import (
    build_index_rgba,
    build_raw_change_rgba,
    build_classified_change_rgba,
    save_visualization_layer,
)
from app.remote_sensing.multimodal.pairing import (
    find_optical_sar_pair,
    PairingErrorType,
)



# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parents[1]

VISUALIZATION_DIR = (
    APP_DIR
    / "evidence"
    / "visualizations"
)

VISUALIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RASTER HELPERS
# ============================================================

def _read_raster(path: str) -> np.ndarray:
    """
    Read the first raster band from a GeoTIFF
    and return it as a float32 NumPy array.
    """

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _get_raster_bounds(
    path: str,
) -> list[list[float]]:
    """
    Return GeoTIFF bounds in Leaflet format:

        [
            [south, west],
            [north, east]
        ]

    The returned values are geographic coordinates in EPSG:4326.
    """

    if not path:
        raise ValueError(
            "A source raster path is required "
            "to calculate bounds."
        )

    with rasterio.open(path) as src:

        if src.crs is None:
            raise ValueError(
                f"Source raster has no CRS: {path}"
            )

        bounds = src.bounds

        try:
            if src.crs.to_epsg() != 4326:
                west, south, east, north = transform_bounds(
                    src.crs,
                    "EPSG:4326",
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top,
                )
            else:
                west = float(bounds.left)
                south = float(bounds.bottom)
                east = float(bounds.right)
                north = float(bounds.top)
        except Exception:
            west = float(bounds.left)
            south = float(bounds.bottom)
            east = float(bounds.right)
            north = float(bounds.top)

        return [
            [
                float(south),
                float(west),
            ],
            [
                float(north),
                float(east),
            ],
        ]


def _get_band_path(
    image: dict,
    band_name: str,
    year: str,
) -> str:
    """
    Safely retrieve a band path from an imagery record.
    """

    bands = image.get(
        "bands",
        {},
    )

    path = bands.get(
        band_name
    )

    if not path:
        raise RuntimeError(
            f"{year} imagery is missing "
            f"{band_name} band."
        )

    return path


def _get_images(
    imagery_result: dict,
) -> tuple[dict, dict]:
    """
    Return the before and after images
    from an imagery search result.
    """
    if imagery_result.get("status") == "REAL_FAILURE":
        err_msg = imagery_result.get("error") or "Unknown retrieval error"
        err_type = imagery_result.get("error_type") or "REAL_FAILURE"
        raise RuntimeError(f"Sentinel-2 retrieval failed [{err_type}]: {err_msg}")

    images = imagery_result.get(
        "images",
        [],
    )

    if len(images) < 2:
        raise RuntimeError(
            "At least two images are required "
            "for comparison."
        )

    return images[0], images[1]



def _get_visualization_source_path(
    image: dict,
    preferred_bands: tuple[str, ...] = (
        "nir",
        "red",
        "green",
        "swir",
    ),
) -> str | None:
    """
    Select a georeferenced source raster for visualization.

    The PNG itself is not georeferenced, so the frontend
    uses this source raster's bounds to position the PNG.
    """

    bands = image.get(
        "bands",
        {},
    )

    for band_name in preferred_bands:

        path = bands.get(
            band_name
        )

        if path:
            return str(path)

    return None


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def _build_continuous_change_rgba(
    change_map: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Map a continuous numeric change field to a smooth RGBA gradient.

    The color gradient is anchored precisely at the 7 semantic classification
    thresholds matching the legend, but smoothly and continuously interpolates
    all intermediate values:

        High decrease:     #e33420 (BGR: [32, 52, 227])
        Moderate decrease: #f27624 (BGR: [36, 118, 242])
        Slight decrease:   #f4c43b (BGR: [59, 196, 244])
        Neutral / Zero:    Transparent (alpha = 0, clean basemap overlay)
        Slight increase:   #a3d977 (BGR: [119, 217, 163])
        Moderate increase: #5b9b42 (BGR: [66, 155, 91])
        High increase:     #2e7d32 (BGR: [50, 125, 46])

    Alpha smoothly scales with change magnitude to prevent hard-edged borders.
    """

    height, width = change_map.shape

    th_slight = float(threshold) if threshold is not None else 0.01
    th_mod = max(0.08, th_slight * 3.0)
    th_high = max(0.20, th_slight * 6.0)

    # 9 anchor points spanning extreme decrease to extreme increase
    anchors_val = np.array(
        [
            -1.0,
            -th_high,
            -th_mod,
            -th_slight,
            0.0,
            th_slight,
            th_mod,
            th_high,
            1.0,
        ],
        dtype=np.float32,
    )

    # Corresponding BGR colors
    b_anchors = np.array(
        [32, 32, 36, 59, 132, 119, 66, 50, 50],
        dtype=np.float32,
    )
    g_anchors = np.array(
        [52, 52, 118, 196, 137, 217, 155, 125, 125],
        dtype=np.float32,
    )
    r_anchors = np.array(
        [227, 227, 242, 244, 136, 163, 91, 46, 46],
        dtype=np.float32,
    )

    # Smooth alpha: 0 at delta=0, scaling up smoothly to ~230
    a_anchors = np.array(
        [230, 230, 220, 190, 0, 190, 220, 230, 230],
        dtype=np.float32,
    )

    flat_vals = np.nan_to_num(
        change_map.flatten(),
        nan=0.0,
    )

    b = np.interp(
        flat_vals,
        anchors_val,
        b_anchors,
    ).reshape(height, width).astype(np.uint8)

    g = np.interp(
        flat_vals,
        anchors_val,
        g_anchors,
    ).reshape(height, width).astype(np.uint8)

    r = np.interp(
        flat_vals,
        anchors_val,
        r_anchors,
    ).reshape(height, width).astype(np.uint8)

    a = np.interp(
        flat_vals,
        anchors_val,
        a_anchors,
    ).reshape(height, width).astype(np.uint8)

    # Strictly set invalid pixels to transparent
    a[~valid_mask] = 0

    return np.stack([b, g, r, a], axis=-1)


def _build_classified_change_rgba(
    change_map: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Build discrete classified BGRA image matching exact legend tiers.
    """

    height, width = change_map.shape
    bgra = np.zeros(
        (height, width, 4),
        dtype=np.uint8,
    )

    th_slight = float(threshold) if threshold is not None else 0.01
    th_mod = max(0.08, th_slight * 3.0)
    th_high = max(0.20, th_slight * 6.0)

    # High decrease: delta < -th_high
    m_hd = valid_mask & (change_map < -th_high)
    bgra[m_hd] = [32, 52, 227, 230]

    # Moderate decrease: -th_high <= delta < -th_mod
    m_md = (
        valid_mask
        & (change_map >= -th_high)
        & (change_map < -th_mod)
    )
    bgra[m_md] = [36, 118, 242, 220]

    # Slight decrease: -th_mod <= delta <= -th_slight
    m_sd = (
        valid_mask
        & (change_map >= -th_mod)
        & (change_map <= -th_slight)
    )
    bgra[m_sd] = [59, 196, 244, 210]

    # Slight increase: th_slight <= delta < th_mod
    m_si = (
        valid_mask
        & (change_map >= th_slight)
        & (change_map < th_mod)
    )
    bgra[m_si] = [119, 217, 163, 210]

    # Moderate increase: th_mod <= delta < th_high
    m_mi = (
        valid_mask
        & (change_map >= th_mod)
        & (change_map < th_high)
    )
    bgra[m_mi] = [66, 155, 91, 220]

    # High increase: delta >= th_high
    m_hi = valid_mask & (change_map >= th_high)
    bgra[m_hi] = [50, 125, 46, 230]

    # No change: -th_slight < delta < th_slight (transparent for clean overlay)
    m_nc = valid_mask & (np.abs(change_map) < th_slight)
    bgra[m_nc] = [132, 137, 136, 0]

    return bgra

def _save_change_map_visualization(
    change_map: np.ndarray,
    prefix: str = "change_map",
    source_raster_path: str | None = None,
    threshold: float | None = None,
) -> dict:
    """
    Convert a floating-point change raster into both:

    1. A smooth continuous spatial gradient visualization PNG.
    2. A discrete classified visualization PNG.

    The continuous numeric change field is bilinearly interpolated
    BEFORE color mapping so that the final visualization has smooth
    spatial transitions.

    The classified layer remains categorical and uses nearest-neighbor
    interpolation so classification boundaries are preserved.
    """

    # --------------------------------------------------------
    # Normalize input
    # --------------------------------------------------------

    change_map = np.asarray(
        change_map,
        dtype=np.float32,
    )

    if change_map.ndim != 2:
        raise ValueError(
            "change_map must be a 2D array."
        )

    height, width = change_map.shape

    if height == 0 or width == 0:
        raise ValueError(
            "change_map cannot be empty."
        )

    # --------------------------------------------------------
    # Valid pixels
    # --------------------------------------------------------

    valid_mask = np.isfinite(
        change_map
    )

    if not np.any(valid_mask):
        raise ValueError(
            "Cannot create change-map visualization: "
            "no valid pixels."
        )

    valid_values = change_map[
        valid_mask
    ]

    max_abs = float(
        np.max(
            np.abs(valid_values)
        )
    )

    if max_abs == 0.0:
        max_abs = 1.0

    # --------------------------------------------------------
    # Display resolution
    #
    # Original Sentinel-2 change raster may be 600x600.
    #
    # Render at 4x resolution:
    #
    #   600 x 600 -> 2400 x 2400
    #
    # This improves Leaflet rendering without changing the
    # underlying scientific raster.
    # --------------------------------------------------------

    max_source_dim = max(
        width,
        height,
    )

    target_dim = max(
        2400,
        max_source_dim * 4,
    )

    scale = (
        target_dim
        / max_source_dim
    )

    display_width = max(
        1,
        int(
            round(
                width * scale
            )
        ),
    )

    display_height = max(
        1,
        int(
            round(
                height * scale
            )
        ),
    )

    # --------------------------------------------------------
    # Continuous numeric interpolation
    #
    # IMPORTANT:
    # Interpolate the FLOAT CHANGE VALUES first.
    # Color mapping happens AFTER interpolation.
    #
    # This prevents blocky categorical-looking pixels.
    # --------------------------------------------------------

    clean_change = np.nan_to_num(
        change_map,
        nan=0.0,
    )

    interp_change = cv2.resize(
        clean_change,
        (
            display_width,
            display_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # --------------------------------------------------------
    # Interpolate validity mask separately
    #
    # This prevents invalid pixels from becoming visible.
    # --------------------------------------------------------

    valid_float = (
        valid_mask.astype(
            np.float32
        )
    )

    interp_valid = (
        cv2.resize(
            valid_float,
            (
                display_width,
                display_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )
        > 0.5
    )

    # --------------------------------------------------------
    # Continuous gradient PNG
    # --------------------------------------------------------

    continuous_rgba = (
        _build_continuous_change_rgba(
            interp_change,
            interp_valid,
            threshold=threshold,
        )
    )

    # --------------------------------------------------------
    # Discrete classified PNG
    #
    # Classification is performed on the ORIGINAL raster,
    # then enlarged using nearest-neighbor interpolation.
    #
    # This preserves exact classification categories.
    # --------------------------------------------------------

    classified_bgra = (
        _build_classified_change_rgba(
            change_map,
            valid_mask,
            threshold=threshold,
        )
    )

    classified_rgba = cv2.resize(
        classified_bgra,
        (
            display_width,
            display_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    # --------------------------------------------------------
    # Unique filenames
    # --------------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    filename = (
        f"{prefix}_{timestamp}.png"
    )

    classified_filename = (
        f"{prefix}_{timestamp}_classified.png"
    )

    output_path = (
        VISUALIZATION_DIR
        / filename
    )

    classified_output_path = (
        VISUALIZATION_DIR
        / classified_filename
    )

    # --------------------------------------------------------
    # Save continuous PNG
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(output_path),
        continuous_rgba,
    )

    if not success:
        raise IOError(
            "Failed to save change-map visualization: "
            f"{output_path}"
        )

    # --------------------------------------------------------
    # Save classified PNG
    # --------------------------------------------------------

    classified_success = cv2.imwrite(
        str(classified_output_path),
        classified_rgba,
    )

    if not classified_success:
        raise IOError(
            "Failed to save classified change-map visualization: "
            f"{classified_output_path}"
        )

    # --------------------------------------------------------
    # Geographic bounds
    # --------------------------------------------------------

    bounds = None

    if source_raster_path:

        bounds = _get_raster_bounds(
            source_raster_path
        )

    # --------------------------------------------------------
    # Relative paths
    # --------------------------------------------------------

    relative_path = (
        output_path
        .relative_to(APP_DIR)
        .as_posix()
    )

    classified_relative_path = (
        classified_output_path
        .relative_to(APP_DIR)
        .as_posix()
    )

    # --------------------------------------------------------
    # Return metadata
    # --------------------------------------------------------

    return {
        "type": "change_map",

        "status": "success",

        "mode": "continuous",

        "filename": filename,

        "path": str(
            output_path
        ),

        "relative_path": relative_path,

        "classified_filename": classified_filename,

        "classified_path": str(
            classified_output_path
        ),

        "classified_relative_path": (
            classified_relative_path
        ),

        "media_type": "image/png",

        "bounds": bounds,

        # Original scientific raster statistics
        "min_value": float(
            np.min(
                valid_values
            )
        ),

        "max_value": float(
            np.max(
                valid_values
            )
        ),

        "max_abs_value": max_abs,

        # ----------------------------------------------------
        # Explicit resolution information
        # ----------------------------------------------------

        "width": display_width,

        "height": display_height,

        "source_width": width,

        "source_height": height,

        "scale_factor": float(
            scale
        ),

        "valid_pixels": int(
            np.sum(
                valid_mask
            )
        ),

       "total_pixels": int(height * width),

        "crs": (
            "EPSG:4326"
            if bounds is not None
            else None
        ),
    }


def _attach_change_map_visualization(
    result: dict,
    prefix: str,
    source_raster_path: str | None = None,
) -> dict:
    """
    Save the change_map contained in a detector result
    and attach visualization metadata.

    The original change_map remains in the internal result.
    """

    if not isinstance(
        result,
        dict,
    ):
        return result

    change_map = result.get(
        "change_map"
    )

    if change_map is None:
        return result

    try:

        visualization = (
            _save_change_map_visualization(
                change_map,
                prefix=prefix,
                source_raster_path=source_raster_path,
                threshold=float(result.get("threshold", 0.05)),
            )
        )

        result[
            "visualization"
        ] = visualization

        # Also expose bounds directly.
        if visualization.get(
            "bounds"
        ) is not None:

            result[
                "bounds"
            ] = visualization[
                "bounds"
            ]

    except Exception as exc:

        result[
            "visualization"
        ] = {
            "type": "change_map",
            "status": "error",
            "error": str(exc),
        }

    return result


# ============================================================
# COMPARISON NORMALIZATION
# ============================================================

def _normalize_comparison_result(
    comparison_result: dict,
) -> dict:
    """
    Convert compare_images() output into the flat
    structure expected by the API.
    """

    if not isinstance(
        comparison_result,
        dict,
    ):
        raise RuntimeError(
            "Image comparison returned "
            "an invalid result."
        )

    statistics = comparison_result.get(
        "statistics"
    )

    if not isinstance(
        statistics,
        dict,
    ):

        statistics = (
            comparison_result.copy()
        )

    normalized = dict(
        statistics
    )

    normalized[
        "metric"
    ] = "IMAGE"

    if normalized.get(
        "total_pixels"
    ) is None:

        normalized[
            "total_pixels"
        ] = normalized.get(
            "valid_pixels"
        )

    mean_change = normalized.get(
        "mean_change"
    )

    if normalized.get(
        "change_type"
    ) is None:

        if mean_change is None:

            normalized[
                "change_type"
            ] = None

        elif mean_change > 0.05:

            normalized[
                "change_type"
            ] = "increase"

        elif mean_change < -0.05:

            normalized[
                "change_type"
            ] = "decrease"

        else:

            normalized[
                "change_type"
            ] = "no_change"

    normalized[
        "operation"
    ] = comparison_result.get(
        "operation",
        "image_comparison",
    )

    normalized[
        "status"
    ] = comparison_result.get(
        "status",
        "success",
    )

    if comparison_result.get(
        "before_image"
    ) is not None:

        normalized[
            "before_image"
        ] = comparison_result[
            "before_image"
        ]

    if comparison_result.get(
        "after_image"
    ) is not None:

        normalized[
            "after_image"
        ] = comparison_result[
            "after_image"
        ]

    return normalized


# ============================================================
# EXECUTION
# ============================================================

def execute_plan(
    tools: list[str],
    context: dict | None = None,
):
    """
    Execute tools sequentially.

    Supports:

    - Image search
    - Temporal NDVI
    - Temporal NDWI
    - Temporal NDBI
    - Single-image NDVI
    - Single-image NDWI
    - Single-image NDBI
    - Vegetation change analysis
    - Image comparison
    - Generic change detection

    Change-detection outputs additionally receive:

    - transparent PNG visualization
    - geographic bounds
    - source raster dimensions
    - visualization dimensions
    """

    context = context or {}

    results = {}

    # ---------------------------------------------------------
    # Requested temporal range
    # ---------------------------------------------------------

    time_start = str(
        context.get(
            "time_start",
            "2021",
        )
    )

    time_end = str(
        context.get(
            "time_end",
            "2025",
        )
    )

    imagery_result = None

    # =========================================================
    # Execute tools
    # =========================================================

    for tool_name in tools:

        print(
            f"Executing: {tool_name}"
        )

        tool = get_tool(
            tool_name
        )

        result: dict = {}

        # =====================================================
        # SEARCH IMAGERY
        # =====================================================

        if tool_name == "search_imagery":
            is_multi = (
                context.get("temporal_mode") in ["multi_temporal", "trend_analysis", "persistence_reversal", "acceleration"]
                or bool(context.get("multi_temporal", False))
            )

            result = tool(
                time_start=time_start,
                time_end=time_end,
                aoi=context.get(
                    "aoi"
                ),
                multi_temporal=is_multi,
            )

            imagery_result = result

        # =====================================================
        # TEMPORAL NDVI
        # =====================================================

        elif tool_name == "calculate_temporal_ndvi":

            if imagery_result is None:
                raise RuntimeError(
                    "search_imagery must run before "
                    "calculate_temporal_ndvi."
                )

            before, after = _get_images(
                imagery_result
            )

            red_before_path = _get_band_path(
                before,
                "red",
                time_start,
            )

            nir_before_path = _get_band_path(
                before,
                "nir",
                time_start,
            )

            red_after_path = _get_band_path(
                after,
                "red",
                time_end,
            )

            nir_after_path = _get_band_path(
                after,
                "nir",
                time_end,
            )

            red_before = _read_raster(
                red_before_path
            )

            nir_before = _read_raster(
                nir_before_path
            )

            red_after = _read_raster(
                red_after_path
            )

            nir_after = _read_raster(
                nir_after_path
            )

            mask_before_path = before.get("bands", {}).get("mask")
            mask_after_path = after.get("bands", {}).get("mask")
            mask_before = _read_raster(mask_before_path) if mask_before_path else None
            mask_after = _read_raster(mask_after_path) if mask_after_path else None

            result = tool(
                red_before=red_before,
                nir_before=nir_before,
                red_after=red_after,
                nir_after=nir_after,
                mask_before=mask_before,
                mask_after=mask_after,
            )

            ts_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            idx_vis = {}
            if "ndvi_before" in result and result["ndvi_before"] is not None:
                try:
                    vis_b = build_index_rgba(result["ndvi_before"], "NDVI", valid_mask=mask_before)
                    idx_vis["before"] = save_visualization_layer(
                        vis_b, f"ndvi_before_{ts_now}.png", source_raster_path=red_before_path
                    )
                except Exception as exc:
                    print(f"[VIS WARNING] NDVI before: {exc}")
            if "ndvi_after" in result and result["ndvi_after"] is not None:
                try:
                    vis_a = build_index_rgba(result["ndvi_after"], "NDVI", valid_mask=mask_after)
                    idx_vis["after"] = save_visualization_layer(
                        vis_a, f"ndvi_after_{ts_now}.png", source_raster_path=red_after_path
                    )
                except Exception as exc:
                    print(f"[VIS WARNING] NDVI after: {exc}")
            result["visualizations"] = idx_vis

        # =====================================================
        # TEMPORAL NDWI
        # =====================================================

        elif tool_name == "calculate_temporal_ndwi":

            if imagery_result is None:
                raise RuntimeError(
                    "search_imagery must run before "
                    "calculate_temporal_ndwi."
                )

            before, after = _get_images(
                imagery_result
            )

            green_before_path = _get_band_path(
                before,
                "green",
                time_start,
            )

            nir_before_path = _get_band_path(
                before,
                "nir",
                time_start,
            )

            green_after_path = _get_band_path(
                after,
                "green",
                time_end,
            )

            nir_after_path = _get_band_path(
                after,
                "nir",
                time_end,
            )

            green_before = _read_raster(
                green_before_path
            )

            nir_before = _read_raster(
                nir_before_path
            )

            green_after = _read_raster(
                green_after_path
            )

            nir_after = _read_raster(
                nir_after_path
            )

            mask_before_path = before.get("bands", {}).get("mask")
            mask_after_path = after.get("bands", {}).get("mask")
            mask_before = _read_raster(mask_before_path) if mask_before_path else None
            mask_after = _read_raster(mask_after_path) if mask_after_path else None

            result = tool(
                green_before=green_before,
                nir_before=nir_before,
                green_after=green_after,
                nir_after=nir_after,
                mask_before=mask_before,
                mask_after=mask_after,
            )

            ts_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            idx_vis = {}
            if "ndwi_before" in result and result["ndwi_before"] is not None:
                try:
                    vis_b = build_index_rgba(result["ndwi_before"], "NDWI", valid_mask=mask_before)
                    idx_vis["before"] = save_visualization_layer(
                        vis_b, f"ndwi_before_{ts_now}.png", source_raster_path=green_before_path
                    )
                except Exception as exc:
                    print(f"[VIS WARNING] NDWI before: {exc}")
            if "ndwi_after" in result and result["ndwi_after"] is not None:
                try:
                    vis_a = build_index_rgba(result["ndwi_after"], "NDWI", valid_mask=mask_after)
                    idx_vis["after"] = save_visualization_layer(
                        vis_a, f"ndwi_after_{ts_now}.png", source_raster_path=green_after_path
                    )
                except Exception as exc:
                    print(f"[VIS WARNING] NDWI after: {exc}")
            result["visualizations"] = idx_vis

        # =====================================================
        # TEMPORAL NDBI
        # =====================================================

        elif tool_name == "calculate_temporal_ndbi":

            if imagery_result is None:
                raise RuntimeError(
                    "search_imagery must run before "
                    "calculate_temporal_ndbi."
                )

            before, after = _get_images(
                imagery_result
            )

            swir_before_path = _get_band_path(
                before,
                "swir",
                time_start,
            )

            nir_before_path = _get_band_path(
                before,
                "nir",
                time_start,
            )

            swir_after_path = _get_band_path(
                after,
                "swir",
                time_end,
            )

            nir_after_path = _get_band_path(
                after,
                "nir",
                time_end,
            )

            swir_before = _read_raster(
                swir_before_path
            )

            nir_before = _read_raster(
                nir_before_path
            )

            swir_after = _read_raster(
                swir_after_path
            )

            nir_after = _read_raster(
                nir_after_path
            )

            mask_before_path = before.get("bands", {}).get("mask")
            mask_after_path = after.get("bands", {}).get("mask")
            mask_before = _read_raster(mask_before_path) if mask_before_path else None
            mask_after = _read_raster(mask_after_path) if mask_after_path else None

            result = tool(
                swir_before=swir_before,
                nir_before=nir_before,
                swir_after=swir_after,
                nir_after=nir_after,
                mask_before=mask_before,
                mask_after=mask_after,
            )

            ts_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            idx_vis = {}
            if "ndbi_before" in result and result["ndbi_before"] is not None:
                try:
                    vis_b = build_index_rgba(result["ndbi_before"], "NDBI", valid_mask=mask_before)
                    idx_vis["before"] = save_visualization_layer(
                        vis_b, f"ndbi_before_{ts_now}.png", source_raster_path=swir_before_path
                    )
                except Exception as exc:
                    print(f"[VIS WARNING] NDBI before: {exc}")
            if "ndbi_after" in result and result["ndbi_after"] is not None:
                try:
                    vis_a = build_index_rgba(result["ndbi_after"], "NDBI", valid_mask=mask_after)
                    idx_vis["after"] = save_visualization_layer(
                        vis_a, f"ndbi_after_{ts_now}.png", source_raster_path=swir_after_path
                    )
                except Exception as exc:
                    print(f"[VIS WARNING] NDBI after: {exc}")
            result["visualizations"] = idx_vis



        # =====================================================
        # SINGLE IMAGE NDWI
        # =====================================================

        elif tool_name == "calculate_ndwi":

            green = context.get("green")
            nir = context.get("nir")

            if green is None or nir is None:
                if imagery_result is None:
                    raise RuntimeError(
                        "search_imagery must run before "
                        "calculate_ndwi."
                    )

                images = imagery_result.get(
                    "images",
                    [],
                )

                if not images:
                    raise RuntimeError(
                        "No imagery available."
                    )

                image = images[0]

                green = _read_raster(
                    _get_band_path(
                        image,
                        "green",
                        time_start,
                    )
                )

                nir = _read_raster(
                    _get_band_path(
                        image,
                        "nir",
                        time_start,
                    )
                )

            result = tool(
                green=green,
                nir=nir,
            )

            if "data" in result and result["data"] is not None:
                source_path = None
                if imagery_result and imagery_result.get("images"):
                    source_path = _get_band_path(imagery_result["images"][0], "green", time_start)
                try:
                    vis_meta = _save_change_map_visualization(
                        change_map=result["data"],
                        prefix="ndwi_index",
                        source_raster_path=source_path,
                    )
                    result["visualization"] = vis_meta
                except Exception as exc:
                    print(f"[NDWI VISUALIZATION WARNING]: {exc}")

        # =====================================================
        # SINGLE IMAGE NDBI
        # =====================================================

        elif tool_name == "calculate_ndbi":

            swir = context.get("swir")
            nir = context.get("nir")

            if swir is None or nir is None:
                if imagery_result is None:
                    raise RuntimeError(
                        "search_imagery must run before "
                        "calculate_ndbi."
                    )

                images = imagery_result.get(
                    "images",
                    [],
                )

                if not images:
                    raise RuntimeError(
                        "No imagery available."
                    )

                image = images[0]

                swir = _read_raster(
                    _get_band_path(
                        image,
                        "swir",
                        time_start,
                    )
                )

                nir = _read_raster(
                    _get_band_path(
                        image,
                        "nir",
                        time_start,
                    )
                )

            result = tool(
                swir=swir,
                nir=nir,
            )

            if "data" in result and result["data"] is not None:
                source_path = None
                if imagery_result and imagery_result.get("images"):
                    source_path = _get_band_path(imagery_result["images"][0], "nir", time_start)
                try:
                    vis_meta = _save_change_map_visualization(
                        change_map=result["data"],
                        prefix="ndbi_index",
                        source_raster_path=source_path,
                    )
                    result["visualization"] = vis_meta
                except Exception as exc:
                    print(f"[NDBI VISUALIZATION WARNING]: {exc}")

        # =====================================================
        # SINGLE IMAGE NDVI
        # =====================================================

        elif tool_name == "calculate_ndvi":

            red = context.get("red")
            nir = context.get("nir")

            if red is None or nir is None:
                if imagery_result is None:
                    raise RuntimeError(
                        "search_imagery must run before "
                        "calculate_ndvi."
                    )

                images = imagery_result.get(
                    "images",
                    [],
                )

                if not images:
                    raise RuntimeError(
                        "No imagery available."
                    )

                image = images[0]

                red = _read_raster(
                    _get_band_path(
                        image,
                        "red",
                        time_start,
                    )
                )

                nir = _read_raster(
                    _get_band_path(
                        image,
                        "nir",
                        time_start,
                    )
                )

            result = tool(
                red=red,
                nir=nir,
            )

            if "data" in result and result["data"] is not None:
                source_path = None
                if imagery_result and imagery_result.get("images"):
                    source_path = _get_band_path(imagery_result["images"][0], "red", time_start)
                try:
                    vis_meta = _save_change_map_visualization(
                        change_map=result["data"],
                        prefix="ndvi_index",
                        source_raster_path=source_path,
                    )
                    result["visualization"] = vis_meta
                except Exception as exc:
                    print(f"[NDVI VISUALIZATION WARNING]: {exc}")

        # =====================================================
        # VEGETATION CHANGE
        # =====================================================

        elif tool_name == "analyze_vegetation_change":

            ndvi_result = results.get(
                "calculate_temporal_ndvi"
            )

            if ndvi_result is None:
                raise RuntimeError(
                    "calculate_temporal_ndvi must run "
                    "before analyze_vegetation_change."
                )

            ndvi_before = ndvi_result.get(
                "ndvi_before"
            )

            ndvi_after = ndvi_result.get(
                "ndvi_after"
            )

            if (
                ndvi_before is None
                or ndvi_after is None
            ):
                raise RuntimeError(
                    "Temporal NDVI did not return "
                    "ndvi_before and ndvi_after."
                )

            result = tool(
                ndvi_before=ndvi_before,
                ndvi_after=ndvi_after,
            )

        # =====================================================
        # IMAGE COMPARISON
        # =====================================================

        elif tool_name == "compare_images":

            if imagery_result is None:
                raise RuntimeError(
                    "search_imagery must run before "
                    "compare_images."
                )

            before, after = _get_images(
                imagery_result
            )

            result = tool(
                before=before,
                after=after,
                before_image=before,
                after_image=after,
            )

            try:

                before_bands = before.get(
                    "bands",
                    {},
                )

                after_bands = after.get(
                    "bands",
                    {},
                )

                before_path = (
                    before_bands.get("nir")
                    or before_bands.get("red")
                    or before_bands.get("green")
                    or before_bands.get("swir")
                )

                after_path = (
                    after_bands.get("nir")
                    or after_bands.get("red")
                    or after_bands.get("green")
                    or after_bands.get("swir")
                )

                if (
                    before_path
                    and after_path
                ):

                    before_array = (
                        _read_raster(
                            before_path
                        )
                    )

                    after_array = (
                        _read_raster(
                            after_path
                        )
                    )

                    if (
                        before_array.shape
                        == after_array.shape
                    ):

                        raw_difference = (
                            after_array
                            - before_array
                        )

                        visualization = (
                            _save_change_map_visualization(
                                raw_difference,
                                prefix="image_comparison",
                                source_raster_path=before_path,
                            )
                        )

                        result[
                            "visualization"
                        ] = visualization

                        result[
                            "bounds"
                        ] = visualization.get(
                            "bounds"
                        )

            except Exception as exc:

                result[
                    "visualization"
                ] = {
                    "type": "change_map",
                    "status": "error",
                    "error": str(exc),
                }

        # =====================================================
        # CHANGE DETECTION
        # =====================================================

        elif tool_name == "detect_change":

            # Ensure all three temporal indices (NDVI, NDWI, NDBI) are calculated
            # so the full scientific layer package is available for every temporal analysis
            if imagery_result is not None and len(imagery_result.get("images", [])) >= 2:
                before_img, after_img = _get_images(imagery_result)
                b_bands = before_img.get("bands", {})
                a_bands = after_img.get("bands", {})
                m_b_path = b_bands.get("mask")
                m_a_path = a_bands.get("mask")
                m_b = _read_raster(m_b_path) if m_b_path else None
                m_a = _read_raster(m_a_path) if m_a_path else None

                # 1. NDVI
                if "calculate_temporal_ndvi" not in results and b_bands.get("red") and b_bands.get("nir") and a_bands.get("red") and a_bands.get("nir"):
                    try:
                        ndvi_tool = get_tool("calculate_temporal_ndvi")
                        r_b = _read_raster(b_bands["red"])
                        n_b = _read_raster(b_bands["nir"])
                        r_a = _read_raster(a_bands["red"])
                        n_a = _read_raster(a_bands["nir"])
                        res_ndvi = ndvi_tool(red_before=r_b, nir_before=n_b, red_after=r_a, nir_after=n_a, mask_before=m_b, mask_after=m_a)
                        ts_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        vis_dict = {}
                        if "ndvi_before" in res_ndvi and res_ndvi["ndvi_before"] is not None:
                            vb = build_index_rgba(res_ndvi["ndvi_before"], "NDVI", valid_mask=m_b)
                            vis_dict["before"] = save_visualization_layer(vb, f"ndvi_before_{ts_now}.png", source_raster_path=b_bands["red"])
                        if "ndvi_after" in res_ndvi and res_ndvi["ndvi_after"] is not None:
                            va = build_index_rgba(res_ndvi["ndvi_after"], "NDVI", valid_mask=m_a)
                            vis_dict["after"] = save_visualization_layer(va, f"ndvi_after_{ts_now}.png", source_raster_path=a_bands["red"])
                        res_ndvi["visualizations"] = vis_dict
                        results["calculate_temporal_ndvi"] = res_ndvi
                    except Exception as exc:
                        print(f"[EXECUTOR WARNING] Auto-calc temporal NDVI: {exc}")

                # 2. NDWI
                if "calculate_temporal_ndwi" not in results and b_bands.get("green") and b_bands.get("nir") and a_bands.get("green") and a_bands.get("nir"):
                    try:
                        ndwi_tool = get_tool("calculate_temporal_ndwi")
                        g_b = _read_raster(b_bands["green"])
                        n_b = _read_raster(b_bands["nir"])
                        g_a = _read_raster(a_bands["green"])
                        n_a = _read_raster(a_bands["nir"])
                        res_ndwi = ndwi_tool(green_before=g_b, nir_before=n_b, green_after=g_a, nir_after=n_a, mask_before=m_b, mask_after=m_a)
                        ts_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        vis_dict = {}
                        if "ndwi_before" in res_ndwi and res_ndwi["ndwi_before"] is not None:
                            vb = build_index_rgba(res_ndwi["ndwi_before"], "NDWI", valid_mask=m_b)
                            vis_dict["before"] = save_visualization_layer(vb, f"ndwi_before_{ts_now}.png", source_raster_path=b_bands["green"])
                        if "ndwi_after" in res_ndwi and res_ndwi["ndwi_after"] is not None:
                            va = build_index_rgba(res_ndwi["ndwi_after"], "NDWI", valid_mask=m_a)
                            vis_dict["after"] = save_visualization_layer(va, f"ndwi_after_{ts_now}.png", source_raster_path=a_bands["green"])
                        res_ndwi["visualizations"] = vis_dict
                        results["calculate_temporal_ndwi"] = res_ndwi
                    except Exception as exc:
                        print(f"[EXECUTOR WARNING] Auto-calc temporal NDWI: {exc}")

                # 3. NDBI
                if "calculate_temporal_ndbi" not in results and b_bands.get("swir") and b_bands.get("nir") and a_bands.get("swir") and a_bands.get("nir"):
                    try:
                        ndbi_tool = get_tool("calculate_temporal_ndbi")
                        s_b = _read_raster(b_bands["swir"])
                        n_b = _read_raster(b_bands["nir"])
                        s_a = _read_raster(a_bands["swir"])
                        n_a = _read_raster(a_bands["nir"])
                        res_ndbi = ndbi_tool(swir_before=s_b, nir_before=n_b, swir_after=s_a, nir_after=n_a, mask_before=m_b, mask_after=m_a)
                        ts_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        vis_dict = {}
                        if "ndbi_before" in res_ndbi and res_ndbi["ndbi_before"] is not None:
                            vb = build_index_rgba(res_ndbi["ndbi_before"], "NDBI", valid_mask=m_b)
                            vis_dict["before"] = save_visualization_layer(vb, f"ndbi_before_{ts_now}.png", source_raster_path=b_bands["swir"])
                        if "ndbi_after" in res_ndbi and res_ndbi["ndbi_after"] is not None:
                            va = build_index_rgba(res_ndbi["ndbi_after"], "NDBI", valid_mask=m_a)
                            vis_dict["after"] = save_visualization_layer(va, f"ndbi_after_{ts_now}.png", source_raster_path=a_bands["swir"])
                        res_ndbi["visualizations"] = vis_dict
                        results["calculate_temporal_ndbi"] = res_ndbi
                    except Exception as exc:
                        print(f"[EXECUTOR WARNING] Auto-calc temporal NDBI: {exc}")

            # Determine primary index based on query context / target / task
            req_metric = str(context.get("metric") or "").lower()
            req_target = str(context.get("target") or "").lower()
            req_task = str(context.get("task") or "").lower()

            if req_metric == "ndbi" or req_target == "urban" or "urban" in req_task:
                primary_index = "ndbi"
            elif req_metric == "ndwi" or req_target == "water" or "water" in req_task:
                primary_index = "ndwi"
            else:
                primary_index = "ndvi"

            threshold = float(context.get("threshold", 0.05))

            # Run change detection for all available temporal indices
            all_changes: dict[str, dict] = {}
            for idx_name in ["ndvi", "ndwi", "ndbi"]:
                t_key = f"calculate_temporal_{idx_name}"
                if t_key in results and results[t_key]:
                    t_res = results[t_key]
                    b_arr = t_res.get(f"{idx_name}_before")
                    a_arr = t_res.get(f"{idx_name}_after")
                    v_mask = t_res.get("valid_mask")
                    if b_arr is not None and a_arr is not None:
                        try:
                            chg_res = tool(
                                before=b_arr,
                                after=a_arr,
                                threshold=threshold,
                                valid_mask=v_mask,
                            )
                            # Find georeferenced source raster
                            src_path = None
                            if imagery_result is not None:
                                before_img, _ = _get_images(imagery_result)
                                b_bands = before_img.get("bands", {})
                                if idx_name == "ndbi":
                                    src_path = b_bands.get("swir") or b_bands.get("nir")
                                elif idx_name == "ndwi":
                                    src_path = b_bands.get("green") or b_bands.get("nir")
                                else:
                                    src_path = b_bands.get("red") or b_bands.get("nir")

                            c_map = chg_res.get("change_map")
                            if c_map is not None:
                                vis_chg = _save_change_map_visualization(
                                    change_map=c_map,
                                    prefix=f"{idx_name}_change",
                                    source_raster_path=src_path,
                                    threshold=threshold,
                                )
                                chg_res["visualization"] = vis_chg
                                chg_res["bounds"] = vis_chg.get("bounds")

                            valid_diff = c_map[np.isfinite(c_map)] if c_map is not None else []
                            chg_res["min_change"] = float(np.min(valid_diff)) if len(valid_diff) > 0 else None
                            chg_res["max_change"] = float(np.max(valid_diff)) if len(valid_diff) > 0 else None
                            all_changes[idx_name.lower()] = chg_res
                            all_changes[idx_name.upper()] = chg_res
                        except Exception as exc:
                            print(f"[EXECUTOR WARNING] Change detection for {idx_name}: {exc}")

            # Pick primary result matching target semantics
            if primary_index.lower() in all_changes:
                result = dict(all_changes[primary_index.lower()])
            elif primary_index.upper() in all_changes:
                result = dict(all_changes[primary_index.upper()])
            elif all_changes:
                first_k = list(all_changes.keys())[0]
                result = dict(all_changes[first_k])
            elif "compare_images" in results:
                comparison_result = results["compare_images"]
                result = _normalize_comparison_result(comparison_result)
                if comparison_result.get("visualization") is not None:
                    result["visualization"] = comparison_result["visualization"]
                if comparison_result.get("bounds") is not None:
                    result["bounds"] = comparison_result["bounds"]
            elif context.get("before") is not None and context.get("after") is not None:
                before = context["before"]
                after = context["after"]
                result = tool(before=before, after=after, threshold=threshold)
                source_raster_path = context.get("source_raster_path")
                change_map = result.get("change_map")
                if change_map is not None:
                    visualization = _save_change_map_visualization(
                        change_map=change_map,
                        prefix="change",
                        source_raster_path=source_raster_path,
                        threshold=threshold,
                    )
                    result["visualization"] = visualization
                    result["bounds"] = visualization.get("bounds")
            else:
                raise RuntimeError(
                    "detect_change requires a temporal index calculation or image comparison first."
                )

            result["all_changes"] = all_changes
            result["primary_metric"] = primary_index.upper()

        # =====================================================
        # OPTICAL-SAR MULTIMODAL ANALYSIS
        # =====================================================

        elif tool_name == "optical_sar_analysis":

            opt_path = context.get("optical_path") or context.get("optical")
            s_path = context.get("sar_path") or context.get("sar")

            # Fallback extraction from images or image_ids if provided
            if (not opt_path or not s_path) and context.get("images"):
                imgs = context.get("images")
                if isinstance(imgs, list) and len(imgs) >= 2:
                    if not opt_path:
                        opt_path = imgs[0]
                    if not s_path:
                        s_path = imgs[1]
                elif isinstance(imgs, dict):
                    if not opt_path:
                        opt_path = imgs.get("optical") or imgs.get("optical_path")
                    if not s_path:
                        s_path = imgs.get("sar") or imgs.get("sar_path")

            if (not opt_path or not s_path) and context.get("image_ids"):
                ids = context.get("image_ids")
                if isinstance(ids, list) and len(ids) >= 2:
                    if not opt_path:
                        opt_path = ids[0]
                    if not s_path:
                        s_path = ids[1]

            pair_metadata = None
            pair_res = None

            # Mode B: Automatic Acquisition when explicit raster paths are absent
            if not opt_path and not s_path:
                aoi = context.get("aoi")
                time_start = context.get("time_start")
                time_end = context.get("time_end")
                target_date = context.get("target_date")

                if aoi is not None and (time_start is not None or time_end is not None or target_date is not None):
                    print(f"[OPTICAL-SAR AGENT] Triggering automatic Optical-SAR acquisition for AOI={aoi}, start={time_start}, end={time_end}")
                    pair_res = find_optical_sar_pair(
                        aoi=aoi,
                        time_start=time_start,
                        time_end=time_end,
                        target_date=target_date,
                        fetch_data=True,
                    )
                    if not pair_res.get("pair_found", False):
                        err_type = pair_res.get("error_type", PairingErrorType.NO_TEMPORALLY_COMPATIBLE_PAIR)
                        err_msg = pair_res.get("error") or "Optical-SAR automatic acquisition failed: no compatible scene pair found."
                        result = {
                            "success": False,
                            "error": err_msg,
                            "error_type": err_type,
                            "details": pair_res.get("details", {}),
                            "answer": None,
                            "modalities": [],
                            "metadata": {},
                            "evidence_used": False,
                            "visuals": {},
                            "fallback": False,
                        }
                        results[tool_name] = result
                        continue

                    opt_path = pair_res["optical"]["path"]
                    s_path = pair_res["sar"]["path"]
                    pair_metadata = {
                        "source": "automatic",
                        "pair_found": True,
                        "optical_item_id": pair_res["optical"]["item_id"],
                        "sar_item_id": pair_res["sar"]["item_id"],
                        "optical_acquisition_datetime": pair_res["optical"]["acquisition_datetime"],
                        "sar_acquisition_datetime": pair_res["sar"]["acquisition_datetime"],
                        "temporal_delta_days": pair_res["temporal_delta_days"],
                        "polarizations": pair_res["sar"]["polarizations"],
                        "coverage": pair_res["spatial_overlap"],
                        "spatial_overlap": pair_res["spatial_overlap"],
                        "selection_reason": pair_res["selection_reason"],
                        "optical": {
                            "item_id": pair_res["optical"]["item_id"],
                            "acquisition_datetime": pair_res["optical"]["acquisition_datetime"],
                            "cloud_cover": pair_res["optical"].get("cloud_cover"),
                            "path": str(pair_res["optical"]["path"]),
                        },
                        "sar": {
                            "item_id": pair_res["sar"]["item_id"],
                            "acquisition_datetime": pair_res["sar"]["acquisition_datetime"],
                            "polarizations": pair_res["sar"]["polarizations"],
                            "mode": pair_res["sar"].get("mode"),
                            "path": str(pair_res["sar"]["path"]),
                            "vh": str(pair_res["sar"].get("vh")),
                        },
                    }
                    context["optical_sar_pair"] = pair_metadata
                    context["optical_path"] = str(opt_path)
                    context["sar_path"] = str(s_path)
                    if pair_res["sar"].get("vh"):
                        context["sar_vh_path"] = str(pair_res["sar"]["vh"])

            q_text = context.get("question") or context.get("query") or "Analyze optical and SAR imagery."
            ev = context.get("evidence")
            vlm_inst = context.get("vlm")
            sar_vh_arg = context.get("sar_vh_path") or context.get("sar_vh") or (pair_res.get("sar", {}).get("vh") if pair_res else None)

            print(f"[OPTICAL-SAR AGENT] Executing multimodal analysis with optical={opt_path}, sar={s_path}, sar_vh={sar_vh_arg}")

            result = tool(
                optical_path=opt_path,
                sar_path=s_path,
                question=q_text,
                evidence=ev,
                vlm=vlm_inst,
                sar_vh_path=str(sar_vh_arg) if sar_vh_arg else None,
            )

            if pair_metadata:
                result["optical_sar_pair"] = pair_metadata
                if "metadata" in result and isinstance(result["metadata"], dict):
                    result["metadata"]["optical_sar_pair"] = pair_metadata

        # =====================================================
        # FALLBACK
        # =====================================================

        else:

            result = tool()

        # =====================================================
        # SAVE RESULT
        # =====================================================

        results[
            tool_name
        ] = result

        context[
            tool_name
        ] = result

    return results