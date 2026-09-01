from typing import List

from app.schemas.query import QueryPlan


def create_execution_plan(query_plan: QueryPlan) -> List[str]:
    """
    Convert a QueryPlan into an ordered list of tools.
    """

    if query_plan.task in {"change_detection", "vegetation_change"}:
        return [
            "search_imagery",
            "calculate_temporal_ndvi",
            "detect_change",
        ]

    if query_plan.task == "water_change":
        return [
            "search_imagery",
            "calculate_temporal_ndwi",
            "detect_change",
        ]

    if query_plan.task == "urban_change":
        return [
            "search_imagery",
            "calculate_temporal_ndbi",
            "detect_change",
        ]

    if query_plan.task == "image_comparison":
        return [
            "search_imagery",
            "compare_images",
            "detect_change",
        ]

    if query_plan.task == "vegetation_index":
        return [
            "search_imagery",
            "calculate_ndvi",
        ]

    if query_plan.task == "water_index":
        return [
            "search_imagery",
            "calculate_ndwi",
        ]

    if query_plan.task == "urban_index":
        return [
            "search_imagery",
            "calculate_ndbi",
        ]

    if query_plan.task == "image_search":
        return [
            "search_imagery",
        ]

    return []