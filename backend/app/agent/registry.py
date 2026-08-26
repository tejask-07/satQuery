from app.tools.imagery import search_imagery
from app.tools.indices import calculate_ndvi, calculate_ndwi, calculate_ndbi
from app.tools.change import compare_images, detect_change


TOOL_REGISTRY = {
    "search_imagery": search_imagery,

    "calculate_ndvi": calculate_ndvi,
    "calculate_ndwi": calculate_ndwi,
    "calculate_ndbi": calculate_ndbi,

    "compare_images": compare_images,
    "detect_change": detect_change,
}


def get_tool(tool_name: str):
    """
    Return a registered tool by name.
    """

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")

    return TOOL_REGISTRY[tool_name]