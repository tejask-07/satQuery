import re
from app.schemas.query import QueryRequest, QueryPlan


def parse_query(request: QueryRequest) -> QueryPlan:
    query = request.query.lower()

    # Extract years
    years = re.findall(r"\b(?:19|20)\d{2}\b", query)
    if len(years) >= 2:
        time_start = years[0]
        time_end = years[1]
    else:
        time_start = "2021"
        time_end = "2025"

    # Vegetation change (NDVI)
    if any(
        w in query
        for w in [
            "vegetation",
            "ndvi",
            "forest",
            "crop",
            "tree",
            "greenery",
            "plant",
        ]
    ):
        return QueryPlan(
            task="change_detection",
            target="vegetation",
            metric="NDVI",
            direction="decrease" if any(d in query for d in ["decrease", "loss", "decline", "reduced"]) else "unknown",
            modalities=["multispectral"],
            time_start=time_start,
            time_end=time_end,
            aoi=request.aoi,
            analysis=[
                "search_imagery",
                "calculate_temporal_ndvi",
                "detect_change",
            ],
            output=[
                "change_map",
                "statistics",
                "explanation",
            ],
        )

    # Water change (NDWI)
    if any(
        w in query
        for w in [
            "water",
            "flood",
            "lake",
            "river",
            "ndwi",
            "reservoir",
            "wetland",
        ]
    ):
        return QueryPlan(
            task="water_change",
            target="water",
            metric="NDWI",
            direction="decrease" if any(d in query for d in ["decrease", "loss", "shrink", "dry"]) else "unknown",
            modalities=["multispectral"],
            time_start=time_start,
            time_end=time_end,
            aoi=request.aoi,
            analysis=[
                "search_imagery",
                "calculate_temporal_ndwi",
                "detect_change",
            ],
            output=[
                "change_map",
                "statistics",
                "explanation",
            ],
        )

    # Urban change (NDBI)
    if any(
        w in query
        for w in [
            "urban",
            "city",
            "building",
            "built-up",
            "construction",
            "ndbi",
            "expansion",
            "settlement",
        ]
    ):
        return QueryPlan(
            task="urban_change",
            target="urban",
            metric="NDBI",
            direction="increase" if any(d in query for d in ["increase", "growth", "expansion", "expanded"]) else "unknown",
            modalities=["multispectral"],
            time_start=time_start,
            time_end=time_end,
            aoi=request.aoi,
            analysis=[
                "search_imagery",
                "calculate_temporal_ndbi",
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
            aoi=request.aoi,
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
        aoi=request.aoi,
        analysis=[],
        output=["explanation"],
    )