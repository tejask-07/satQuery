from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from app.agent.registry import get_tool


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

            result = tool(
                time_start=time_start,
                time_end=time_end,
                aoi=context.get(
                    "aoi"
                ),
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

            result = tool(
                red_before=red_before,
                nir_before=nir_before,
                red_after=red_after,
                nir_after=nir_after,
            )

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

            result = tool(
                green_before=green_before,
                nir_before=nir_before,
                green_after=green_after,
                nir_after=nir_after,
            )

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

            result = tool(
                swir_before=swir_before,
                nir_before=nir_before,
                swir_after=swir_after,
                nir_after=nir_after,
            )

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

            temporal_result = None
            index_name = None

            # -------------------------------------------------
            # Temporal NDVI
            # -------------------------------------------------

            if (
                "calculate_temporal_ndvi"
                in results
            ):

                temporal_result = results[
                    "calculate_temporal_ndvi"
                ]

                index_name = "ndvi"

            # -------------------------------------------------
            # Temporal NDWI
            # -------------------------------------------------

            elif (
                "calculate_temporal_ndwi"
                in results
            ):

                temporal_result = results[
                    "calculate_temporal_ndwi"
                ]

                index_name = "ndwi"

            # -------------------------------------------------
            # Temporal NDBI
            # -------------------------------------------------

            elif (
                "calculate_temporal_ndbi"
                in results
            ):

                temporal_result = results[
                    "calculate_temporal_ndbi"
                ]

                index_name = "ndbi"

            # -------------------------------------------------
            # IMAGE COMPARISON
            # -------------------------------------------------

            elif "compare_images" in results:

                comparison_result = results[
                    "compare_images"
                ]

                result = (
                    _normalize_comparison_result(
                        comparison_result
                    )
                )

                if comparison_result.get(
                    "visualization"
                ) is not None:

                    result[
                        "visualization"
                    ] = comparison_result[
                        "visualization"
                    ]

                if comparison_result.get(
                    "bounds"
                ) is not None:

                    result[
                        "bounds"
                    ] = comparison_result[
                        "bounds"
                    ]

            elif context.get("before") is not None and context.get("after") is not None:

                before = context["before"]
                after = context["after"]
                threshold = float(context.get("threshold", 0.05))

                result = tool(
                    before=before,
                    after=after,
                    threshold=threshold,
                )

                source_raster_path = context.get(
                    "source_raster_path"
                )

                change_map = result.get(
                    "change_map"
                )

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
                    "detect_change requires a temporal "
                    "index calculation or image comparison first."
                )

            # -------------------------------------------------
            # Temporal index change detection
            # -------------------------------------------------

            if temporal_result is not None and index_name is not None:

                before_key = (
                    f"{index_name}_before"
                )

                after_key = (
                    f"{index_name}_after"
                )

                before = temporal_result.get(
                    before_key
                )

                after = temporal_result.get(
                    after_key
                )

                if (
                    before is None
                    or after is None
                ):

                    raise RuntimeError(
                        f"Temporal {index_name.upper()} "
                        f"result is missing "
                        f"{before_key} or {after_key}."
                    )

                # ---------------------------------------------
                # Scientific change detection
                # ---------------------------------------------

                threshold = float(
                    context.get(
                        "threshold",
                        0.05,
                    )
                )

                result = tool(
                    before=before,
                    after=after,
                    threshold=threshold,
                )

                # ---------------------------------------------
                # Find source GeoTIFF.
                # ---------------------------------------------

                source_raster_path = None

                if imagery_result is not None:

                    before_image, _ = _get_images(
                        imagery_result
                    )

                    before_bands = (
                        before_image.get(
                            "bands",
                            {},
                        )
                    )

                    if index_name == "ndbi":

                        source_raster_path = (
                            before_bands.get(
                                "swir"
                            )
                            or before_bands.get(
                                "nir"
                            )
                        )

                    elif index_name == "ndwi":

                        source_raster_path = (
                            before_bands.get(
                                "green"
                            )
                            or before_bands.get(
                                "nir"
                            )
                        )

                    elif index_name == "ndvi":

                        source_raster_path = (
                            before_bands.get(
                                "red"
                            )
                            or before_bands.get(
                                "nir"
                            )
                        )

                    if source_raster_path is None:

                        source_raster_path = (
                            before_bands.get(
                                "nir"
                            )
                            or before_bands.get(
                                "red"
                            )
                            or before_bands.get(
                                "green"
                            )
                            or before_bands.get(
                                "swir"
                            )
                        )

                # ---------------------------------------------
                # Make sure we actually received change_map.
                # ---------------------------------------------

                change_map = result.get(
                    "change_map"
                )

                if change_map is None:

                    raise RuntimeError(
                        f"Change detection for "
                        f"{index_name.upper()} did not return "
                        "a change_map."
                    )

                # ---------------------------------------------
                # Create visualization.
                # ---------------------------------------------

                visualization = _save_change_map_visualization(
                    change_map=change_map,
                    prefix=f"{index_name}_change",
                    source_raster_path=source_raster_path,
                    threshold=float(
                        result.get(
                            "threshold",
                            0.05,
                        )
                    ),
                )

                # ---------------------------------------------
                # Attach visualization.
                # ---------------------------------------------

                result[
                    "visualization"
                ] = visualization

                # ---------------------------------------------
                # Expose bounds directly.
                # ---------------------------------------------

                result[
                    "bounds"
                ] = visualization.get(
                    "bounds"
                )

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