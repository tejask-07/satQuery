from typing import List

from app.schemas.query import QueryPlan


def create_execution_plan(query_plan: QueryPlan) -> List[str]:
    """
    Convert a QueryPlan into an ordered list of tools.
    """

    task = query_plan.task
    target = (query_plan.target or "").lower()

    if task == "optical_sar_analysis":
        return [
            "optical_sar_analysis",
        ]

    if task == "single_image_vqa":
        return [
            "single_image_vqa",
        ]

    explicit_metric = (
        getattr(query_plan, "explicit_metric", None)
        or ("ndvi" if task in {"ndvi_change", "calculate_temporal_ndvi"}
            else "ndwi" if task in {"ndwi_change", "calculate_temporal_ndwi"}
            else "ndbi" if task in {"ndbi_change", "calculate_temporal_ndbi"}
            else None)
    )
    if not explicit_metric and query_plan.metric:
        if query_plan.target is None and query_plan.metric.lower() in {"ndvi", "ndwi", "ndbi"}:
            explicit_metric = query_plan.metric.lower()

    if explicit_metric:
        explicit_metric = explicit_metric.lower()

    if explicit_metric == "ndvi":
        return [
            "search_imagery",
            "calculate_temporal_ndvi",
            "detect_change",
        ]

    if explicit_metric == "ndwi":
        return [
            "search_imagery",
            "calculate_temporal_ndwi",
            "detect_change",
        ]

    if explicit_metric == "ndbi":
        return [
            "search_imagery",
            "calculate_temporal_ndbi",
            "detect_change",
        ]

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