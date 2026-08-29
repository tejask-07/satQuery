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


TOOL_REGISTRY = {
    "search_imagery": search_imagery,

    "calculate_ndvi": calculate_ndvi,
    "calculate_ndwi": calculate_ndwi,
    "calculate_ndbi": calculate_ndbi,

    "calculate_temporal_ndvi": calculate_temporal_ndvi,
    "calculate_temporal_ndwi": calculate_temporal_ndwi,
    "calculate_temporal_ndbi": calculate_temporal_ndbi,

    "detect_change": detect_change,
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