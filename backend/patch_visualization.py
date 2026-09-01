with open("app/api/routes_query.py", "r") as f:
    content = f.read()

target = """
    if change_result:

        layers.append(
            {
                "type": "change_detection",
                "name": (
                    f"{plan.target or 'Image'} "
                    f"change map"
                ),
                "change_ratio": (
                    change_result.get(
                        "change_ratio"
                    )
                ),
                "regions_detected": (
                    change_result.get(
                        "regions_detected"
                    )
                ),
                "changed_pixels": (
                    change_result.get(
                        "changed_pixels"
                    )
                ),
            }
        )
"""
new = """
    if change_result:

        visualization = change_result.get("visualization")
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
                "type": "change_detection",
                "name": (
                    f"{plan.target or 'Image'} "
                    f"change map"
                ),
                "change_ratio": (
                    change_result.get(
                        "change_ratio"
                    )
                ),
                "regions_detected": (
                    change_result.get(
                        "regions_detected"
                    )
                ),
                "changed_pixels": (
                    change_result.get(
                        "changed_pixels"
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
print("done")
