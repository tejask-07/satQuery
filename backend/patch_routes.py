import re

with open("app/api/routes_query.py", "r") as f:
    content = f.read()

# Fix 1: classified_visualization_url
target1_old = """
        visualization_url = None
        visualization_bounds = None
"""
target1_new = """
        visualization_url = None
        classified_visualization_url = None
        visualization_bounds = None
"""
content = content.replace(target1_old, target1_new)

target2_old = """
            classified_filename = visualization.get(
                "classified_filename"
            )

            classified_visualization_url = (
                f"/visualizations/{classified_filename}"
                if classified_filename
                else None
            )
"""
target2_new = """
            classified_filename = visualization.get(
                "classified_filename"
            )

            if classified_filename:
                classified_visualization_url = (
                    f"/visualizations/{classified_filename}"
                )
"""
content = content.replace(target2_old, target2_new)

# Fix 2: AOI parsing and passing
target3_old = """
    query = request.query.lower()
"""
target3_new = """
    query_str = request.query
    parsed_aoi = None
    import re
    aoi_match = re.search(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', query_str)
    if aoi_match:
        minLon, minLat, maxLon, maxLat = [float(x) for x in aoi_match.groups()]
        parsed_aoi = {
            "type": "Polygon",
            "coordinates": [[
                [minLon, minLat],
                [maxLon, minLat],
                [maxLon, maxLat],
                [minLon, maxLat],
                [minLon, minLat]
            ]]
        }
        query_str = query_str[:aoi_match.start()] + query_str[aoi_match.end():]
        query_str = re.sub(r'\bfor\s+aoi\b', '', query_str, flags=re.IGNORECASE).strip()
        query_str = re.sub(r'\baoi\b', '', query_str, flags=re.IGNORECASE).strip()

    final_aoi = request.aoi if request.aoi else parsed_aoi
    query = query_str.lower()
"""
content = content.replace(target3_old, target3_new)

target4_old = """
    return QueryPlan(
        task=task,
        target=target,
        time_start="2021",
        time_end="2025",
        modalities=["optical"],
        metric=metric,
        direction="unknown",
        analysis=analysis,
        output=[
            "map",
            "statistics",
            "explanation",
        ],
    )
"""
target4_new = """
    return QueryPlan(
        task=task,
        target=target,
        time_start="2021",
        time_end="2025",
        modalities=["optical"],
        metric=metric,
        direction="unknown",
        analysis=analysis,
        output=[
            "map",
            "statistics",
            "explanation",
        ],
        aoi=final_aoi,
    )
"""
content = content.replace(target4_old, target4_new)

with open("app/api/routes_query.py", "w") as f:
    f.write(content)

print("Patch applied successfully.")
