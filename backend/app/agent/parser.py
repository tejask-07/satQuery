from app.schemas.query import QueryRequest, QueryPlan


def parse_query(request: QueryRequest) -> QueryPlan:
    query = request.query.lower()

    # Vegetation change
    if "vegetation" in query and (
        "decreased" in query
        or "decrease" in query
        or "loss" in query
        or "decline" in query
    ):
        return QueryPlan(
            task="change_detection",
            target="vegetation",
            metric="NDVI",
            direction="decrease",
            modalities=["multispectral"],
            time_start="2021",
            time_end="2025",
            analysis=[
                "search_imagery",
                "calculate_ndvi",
                "compare_images",
                "detect_change",
            ],
            output=[
                "change_map",
                "statistics",
                "explanation",
            ],
        )

    # Water change
    if "water" in query or "flood" in query:
        return QueryPlan(
            task="water_change",
            target="water",
            metric="NDWI",
            modalities=["multispectral"],
            analysis=[
                "search_imagery",
                "calculate_ndwi",
                "detect_change",
            ],
            output=[
                "change_map",
                "statistics",
                "explanation",
            ],
        )

    # Urban change
    if (
        "urban" in query
        or "building" in query
        or "urban expansion" in query
    ):
        return QueryPlan(
            task="urban_change",
            target="urban",
            metric="NDBI",
            modalities=["multispectral"],
            analysis=[
                "search_imagery",
                "calculate_ndbi",
                "detect_change",
            ],
            output=[
                "change_map",
                "statistics",
                "explanation",
            ],
        )

    # Generic image comparison
    if "compare" in query or "changed" in query:
        return QueryPlan(
            task="image_comparison",
            analysis=[
                "compare_images",
                "detect_change",
            ],
            output=[
                "comparison",
                "change_map",
                "explanation",
            ],
        )

    # Fallback
    return QueryPlan(
        task="unknown",
        analysis=[],
        output=["explanation"],
    )