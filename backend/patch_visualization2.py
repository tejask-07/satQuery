with open("app/api/routes_query.py", "r") as f:
    content = f.read()

target = """
    if index_result:

        layers.append(
            {
                "type": "index",
                "name": (
                    f"{index_result.get('index')} "
                    "analysis"
                ),
                "metric": (
                    index_result.get(
                        "index"
                    )
                ),
                "mean": (
                    index_result.get(
                        "mean"
                    )
                ),
            }
        )
"""
new = """
    if index_result:
        
        visualization = index_result.get("visualization")
        visualization_url = None
        classified_visualization_url = None
        visualization_bounds = None
        mode = "continuous"

        if visualization and visualization.get("status") != "error":
            filename = visualization.get("filename")
            if filename:
                visualization_url = f"/visualizations/{filename}"
            
            classified_filename = visualization.get("classified_filename")
            if classified_filename:
                classified_visualization_url = f"/visualizations/{classified_filename}"
            
            visualization_bounds = visualization.get("bounds")
            mode = visualization.get("mode", "continuous")

        layers.append(
            {
                "type": "index",
                "name": (
                    f"{index_result.get('index')} "
                    "analysis"
                ),
                "metric": (
                    index_result.get(
                        "index"
                    )
                ),
                "mean": (
                    index_result.get(
                        "mean"
                    )
                ),
                "visualization_url": visualization_url,
                "classified_visualization_url": classified_visualization_url,
                "bounds": visualization_bounds,
                "mode": mode,
            }
        )
"""
content = content.replace(target, new)
with open("app/api/routes_query.py", "w") as f:
    f.write(content)
print("done2")
