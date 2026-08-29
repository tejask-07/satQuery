from typing import List

from app.schemas.query import QueryPlan


def create_execution_plan(query_plan: QueryPlan) -> List[str]:
    """
    Convert a QueryPlan into an ordered list of tools.
    """

    if query_plan.task == "change_detection":
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
            "compare_images",
            "detect_change",
        ]

    if query_plan.task == "image_search":
        return [
            "search_imagery",
        ]

    return []