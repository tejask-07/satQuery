from typing import List

from app.schemas.query import QueryPlan


def create_execution_plan(query_plan: QueryPlan) -> List[str]:
    """
    Convert a QueryPlan into an ordered list of tools.
    """

    task = query_plan.task
    target = (query_plan.target or "").lower()

    if task == "urban_change" or (task == "change_detection" and target == "urban"):
        return [
            "search_imagery",
            "calculate_temporal_ndbi",
            "calculate_temporal_ndvi",
            "calculate_temporal_ndwi",
            "detect_change",
        ]

    if task == "water_change" or (task == "change_detection" and target == "water"):
        return [
            "search_imagery",
            "calculate_temporal_ndwi",
            "calculate_temporal_ndvi",
            "calculate_temporal_ndbi",
            "detect_change",
        ]

    if task in {"change_detection", "vegetation_change"} or (task == "change_detection" and target == "vegetation"):
        return [
            "search_imagery",
            "calculate_temporal_ndvi",
            "calculate_temporal_ndwi",
            "calculate_temporal_ndbi",
            "detect_change",
        ]

    if task in {"general_change_detection", "general_change"}:
        return [
            "search_imagery",
            "calculate_temporal_ndvi",
            "calculate_temporal_ndwi",
            "calculate_temporal_ndbi",
            "detect_change",
        ]

    if task == "land_cover_transition":
        return [
            "search_imagery",
            "calculate_temporal_ndvi",
            "calculate_temporal_ndwi",
            "calculate_temporal_ndbi",
            "detect_change",
        ]

    if task == "image_comparison":
        return [
            "search_imagery",
            "compare_images",
            "detect_change",
        ]

    if task == "vegetation_index":
        return [
            "search_imagery",
            "calculate_ndvi",
        ]

    if task == "water_index":
        return [
            "search_imagery",
            "calculate_ndwi",
        ]

    if task == "urban_index":
        return [
            "search_imagery",
            "calculate_ndbi",
        ]

    if task == "image_search":
        return [
            "search_imagery",
        ]

    return []