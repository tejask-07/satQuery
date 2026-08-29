import numpy as np
import rasterio

from app.agent.registry import get_tool


def _read_raster(path: str) -> np.ndarray:
    """
    Read the first band from a raster file.
    """

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _get_band_path(image: dict, band_name: str, year: str) -> str:
    """
    Safely retrieve a band path from an imagery record.
    """

    bands = image.get("bands", {})

    path = bands.get(band_name)

    if not path:
        raise RuntimeError(
            f"{year} imagery is missing {band_name} band."
        )

    return path


def execute_plan(
    tools: list[str],
    context: dict | None = None,
):
    """
    Execute tools sequentially.

    Supports:

    - Temporal NDVI
    - Temporal NDWI
    - Temporal NDBI
    - Generic change detection
    """

    context = context or {}
    results = {}

    imagery_result = None

    for tool_name in tools:

        print(f"Executing: {tool_name}")

        tool = get_tool(tool_name)

        # ---------------------------------------------------------
        # SEARCH IMAGERY
        # ---------------------------------------------------------

        if tool_name == "search_imagery":

            result = tool()

            imagery_result = result

        # ---------------------------------------------------------
        # TEMPORAL NDVI
        # ---------------------------------------------------------

        elif tool_name == "calculate_temporal_ndvi":

            if imagery_result is None:
                raise RuntimeError(
                    "search_imagery must run before "
                    "calculate_temporal_ndvi."
                )

            images = imagery_result.get("images", [])

            if len(images) < 2:
                raise RuntimeError(
                    "At least two images are required "
                    "for temporal NDVI."
                )

            before = images[0]
            after = images[1]

            red_before_path = _get_band_path(
                before,
                "red",
                "2021",
            )

            nir_before_path = _get_band_path(
                before,
                "nir",
                "2021",
            )

            red_after_path = _get_band_path(
                after,
                "red",
                "2025",
            )

            nir_after_path = _get_band_path(
                after,
                "nir",
                "2025",
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

        # ---------------------------------------------------------
        # TEMPORAL NDWI
        # ---------------------------------------------------------

        elif tool_name == "calculate_temporal_ndwi":

            if imagery_result is None:
                raise RuntimeError(
                    "search_imagery must run before "
                    "calculate_temporal_ndwi."
                )

            images = imagery_result.get("images", [])

            if len(images) < 2:
                raise RuntimeError(
                    "At least two images are required "
                    "for temporal NDWI."
                )

            before = images[0]
            after = images[1]

            green_before_path = _get_band_path(
                before,
                "green",
                "2021",
            )

            nir_before_path = _get_band_path(
                before,
                "nir",
                "2021",
            )

            green_after_path = _get_band_path(
                after,
                "green",
                "2025",
            )

            nir_after_path = _get_band_path(
                after,
                "nir",
                "2025",
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

        # ---------------------------------------------------------
        # TEMPORAL NDBI
        # ---------------------------------------------------------

        elif tool_name == "calculate_temporal_ndbi":

            if imagery_result is None:
                raise RuntimeError(
                    "search_imagery must run before "
                    "calculate_temporal_ndbi."
                )

            images = imagery_result.get("images", [])

            if len(images) < 2:
                raise RuntimeError(
                    "At least two images are required "
                    "for temporal NDBI."
                )

            before = images[0]
            after = images[1]

            swir_before_path = _get_band_path(
                before,
                "swir",
                "2021",
            )

            nir_before_path = _get_band_path(
                before,
                "nir",
                "2021",
            )

            swir_after_path = _get_band_path(
                after,
                "swir",
                "2025",
            )

            nir_after_path = _get_band_path(
                after,
                "nir",
                "2025",
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

        # ---------------------------------------------------------
        # NDWI SINGLE IMAGE
        # ---------------------------------------------------------

        elif tool_name == "calculate_ndwi":

            if imagery_result is None:
                raise RuntimeError(
                    "search_imagery must run before "
                    "calculate_ndwi."
                )

            images = imagery_result.get("images", [])

            if not images:
                raise RuntimeError(
                    "No imagery available."
                )

            image = images[0]

            green = _read_raster(
                _get_band_path(
                    image,
                    "green",
                    "2021",
                )
            )

            nir = _read_raster(
                _get_band_path(
                    image,
                    "nir",
                    "2021",
                )
            )

            result = tool(
                green=green,
                nir=nir,
            )

        # ---------------------------------------------------------
        # NDBI SINGLE IMAGE
        # ---------------------------------------------------------

        elif tool_name == "calculate_ndbi":

            if imagery_result is None:
                raise RuntimeError(
                    "search_imagery must run before "
                    "calculate_ndbi."
                )

            images = imagery_result.get("images", [])

            if not images:
                raise RuntimeError(
                    "No imagery available."
                )

            image = images[0]

            swir = _read_raster(
                _get_band_path(
                    image,
                    "swir",
                    "2021",
                )
            )

            nir = _read_raster(
                _get_band_path(
                    image,
                    "nir",
                    "2021",
                )
            )

            result = tool(
                swir=swir,
                nir=nir,
            )

        # ---------------------------------------------------------
        # NDVI SINGLE IMAGE
        # ---------------------------------------------------------

        elif tool_name == "calculate_ndvi":

            if imagery_result is None:
                raise RuntimeError(
                    "search_imagery must run before "
                    "calculate_ndvi."
                )

            images = imagery_result.get("images", [])

            if not images:
                raise RuntimeError(
                    "No imagery available."
                )

            image = images[0]

            red = _read_raster(
                _get_band_path(
                    image,
                    "red",
                    "2021",
                )
            )

            nir = _read_raster(
                _get_band_path(
                    image,
                    "nir",
                    "2021",
                )
            )

            result = tool(
                red=red,
                nir=nir,
            )

        # ---------------------------------------------------------
        # VEGETATION CHANGE
        # ---------------------------------------------------------

        elif tool_name == "analyze_vegetation_change":

            ndvi_result = results.get(
                "calculate_temporal_ndvi"
            )

            if ndvi_result is None:
                raise RuntimeError(
                    "calculate_temporal_ndvi must run before "
                    "analyze_vegetation_change."
                )

            ndvi_before = ndvi_result.get(
                "ndvi_before"
            )

            ndvi_after = ndvi_result.get(
                "ndvi_after"
            )

            if ndvi_before is None or ndvi_after is None:
                raise RuntimeError(
                    "Temporal NDVI did not return "
                    "ndvi_before and ndvi_after."
                )

            result = tool(
                ndvi_before=ndvi_before,
                ndvi_after=ndvi_after,
            )

        # ---------------------------------------------------------
        # CHANGE DETECTION
        # ---------------------------------------------------------

        elif tool_name == "detect_change":

            # Determine which temporal index ran immediately
            # before change detection.

            temporal_result = None
            index_name = None

            if "calculate_temporal_ndvi" in results:
                temporal_result = results[
                    "calculate_temporal_ndvi"
                ]
                index_name = "ndvi"

            elif "calculate_temporal_ndwi" in results:
                temporal_result = results[
                    "calculate_temporal_ndwi"
                ]
                index_name = "ndwi"

            elif "calculate_temporal_ndbi" in results:
                temporal_result = results[
                    "calculate_temporal_ndbi"
                ]
                index_name = "ndbi"

            if temporal_result is None:
                raise RuntimeError(
                    "detect_change requires a temporal "
                    "index calculation first."
                )

            before_key = f"{index_name}_before"
            after_key = f"{index_name}_after"

            before = temporal_result.get(
                before_key
            )

            after = temporal_result.get(
                after_key
            )

            if before is None or after is None:
                raise RuntimeError(
                    f"Temporal {index_name.upper()} result "
                    f"is missing {before_key} or {after_key}."
                )

            result = tool(
                before=before,
                after=after,
            )

        # ---------------------------------------------------------
        # FALLBACK
        # ---------------------------------------------------------

        else:
            result = tool()

        # ---------------------------------------------------------
        # SAVE RESULT
        # ---------------------------------------------------------

        results[tool_name] = result
        context[tool_name] = result

    return results