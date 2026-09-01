from app.tools.imagery import search_imagery

from app.tools.indices import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndbi,
    calculate_temporal_ndvi,
    calculate_temporal_ndwi,
    calculate_temporal_ndbi,
)

from app.tools.change import detect_change

# Image comparison tool
from app.tools.comparison import compare_images


TOOL_REGISTRY = {
    # Imagery
    "search_imagery": search_imagery,

    # Single-image indices
    "calculate_ndvi": calculate_ndvi,
    "calculate_ndwi": calculate_ndwi,
    "calculate_ndbi": calculate_ndbi,

    # Temporal indices
    "calculate_temporal_ndvi": calculate_temporal_ndvi,
    "calculate_temporal_ndwi": calculate_temporal_ndwi,
    "calculate_temporal_ndbi": calculate_temporal_ndbi,

    # Change detection
    "detect_change": detect_change,

    # Image comparison
    "compare_images": compare_images,
}


def get_tool(tool_name: str):
    """
    Return a registered tool by name.
    """

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    return TOOL_REGISTRY[tool_name]