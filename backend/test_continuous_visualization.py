import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agent.executor import execute_plan


def test_indicator_pipeline(indicator: str, tool_name: str):
    print("=" * 80)
    print(f"TESTING PIPELINE: {indicator.upper()}")
    print("=" * 80)

    tools = ["search_imagery", tool_name, "detect_change"]
    context = {
        "time_start": "2021",
        "time_end": "2025",
        "threshold": 0.05,
    }

    results = execute_plan(tools, context=context)
    change_result = results["detect_change"]
    change_map = change_result.get("change_map")
    assert change_map is not None, f"Missing change_map for {indicator}"

    # 1. Analytical Raster & Statistics
    src_h, src_w = change_map.shape
    valid_mask = np.isfinite(change_map)
    valid_pixels = int(np.sum(valid_mask))
    mean_change = float(np.mean(change_map[valid_mask]))
    changed_pixels = change_result.get("changed_pixels")
    change_ratio = change_result.get("change_ratio")

    print(f"Analytical Shape:       {src_h} x {src_w} ({change_map.size} total cells)")
    print(f"Valid Pixels:           {valid_pixels}")
    print(f"Mean Change:            {mean_change:.6f}")
    print(f"Changed Pixels Count:   {changed_pixels}")
    print(f"Change Ratio:           {change_ratio}")

    # 2. Continuous Visualization PNG
    vis = change_result.get("visualization", {})
    png_path = vis.get("path")
    assert png_path and os.path.exists(png_path), f"Continuous PNG missing: {png_path}"
    assert vis.get("mode") == "continuous", f"Expected mode 'continuous', got {vis.get('mode')}"

    im_cont = Image.open(png_path)
    cw, ch = im_cont.size
    c_size_kb = round(os.path.getsize(png_path) / 1024, 2)
    arr_cont = np.array(im_cont)

    # Count unique colors in continuous PNG
    # Reshape to (N, 4)
    flat_colors = arr_cont.reshape(-1, arr_cont.shape[-1])
    unique_continuous_colors = len(np.unique(flat_colors, axis=0))

    # Compute step-difference gradient in continuous PNG (smooth transitions)
    # Check max adjacent pixel step in RGB
    rgb_float = arr_cont[:, :, :3].astype(np.float32)
    diff_y = np.abs(np.diff(rgb_float, axis=0))
    diff_x = np.abs(np.diff(rgb_float, axis=1))
    mean_adjacent_diff = float((np.mean(diff_y) + np.mean(diff_x)) / 2.0)

    print(f"Continuous Display PNG: {cw} x {ch} px ({c_size_kb} KB)")
    print(f"Continuous Bounds:      {vis.get('bounds')}")
    print(f"Continuous Unique Colors: {unique_continuous_colors:,} distinct color gradations")
    print(f"Mean Adjacent Pixel Diff: {mean_adjacent_diff:.3f} RGB levels (smooth continuous transition)")

    # 3. Classified Layer PNG
    classified_path = vis.get("classified_path")
    assert classified_path and os.path.exists(classified_path), f"Classified PNG missing: {classified_path}"
    im_class = Image.open(classified_path)
    cl_w, cl_h = im_class.size
    cl_size_kb = round(os.path.getsize(classified_path) / 1024, 2)
    arr_class = np.array(im_class)
    unique_class_colors = len(np.unique(arr_class.reshape(-1, arr_class.shape[-1]), axis=0))

    print(f"Classified Layer PNG:   {cl_w} x {cl_h} px ({cl_size_kb} KB)")
    print(f"Classified Unique Colors: {unique_class_colors} discrete classes (<= 7 legend tiers + transparent)")

    # 4. Strict Assertions
    # a. Continuous visualization must have thousands of smooth color gradations, NOT 7 flat blocks
    assert unique_continuous_colors > 1000, (
        f"Continuous visualization has only {unique_continuous_colors} colors; expected > 1000"
    )
    # b. Classified layer must strictly have discrete classes (<= 8 including transparent)
    assert unique_class_colors <= 8, (
        f"Classified layer has {unique_class_colors} colors; expected discrete classes <= 8"
    )
    # c. Geographic bounds must match
    assert vis.get("bounds") == [[18.5, 73.8], [18.56, 73.86]], f"Invalid bounds: {vis.get('bounds')}"
    # d. Valid pixels count must equal analytical
    assert vis.get("valid_pixels") == valid_pixels
    # e. Smoothness check: adjacent pixel difference must be gentle
    assert mean_adjacent_diff < 15.0, (
        f"Adjacent pixel difference {mean_adjacent_diff} is too high; visualization is not smooth"
    )

    print(f"-> {indicator.upper()} VERIFICATION PASSED SUCCESSFULLY.")
    return {
        "indicator": indicator,
        "source_shape": f"{src_h}x{src_w}",
        "display_shape": f"{cw}x{ch}",
        "unique_continuous_colors": unique_continuous_colors,
        "mean_adjacent_diff": round(mean_adjacent_diff, 2),
        "changed_pixels": changed_pixels,
        "discrete_classes": unique_class_colors,
    }


def main():
    print("\nSTARTING VERIFICATION OF ALL THREE CONTINUOUS VISUALIZATION PIPELINES\n")
    pipelines = [
        ("NDVI / Vegetation", "calculate_temporal_ndvi"),
        ("NDWI / Water", "calculate_temporal_ndwi"),
        ("NDBI / Urban", "calculate_temporal_ndbi"),
    ]

    summary = []
    for indicator, tool in pipelines:
        res = test_indicator_pipeline(indicator, tool)
        summary.append(res)

    print("\n" + "=" * 80)
    print("CONTINUOUS VISUALIZATION VERIFICATION SUMMARY")
    print("=" * 80)
    header = f"{'Indicator':<20} | {'Source':<10} | {'Display':<11} | {'Continuous Colors':<18} | {'Class Colors':<14} | {'Mean Diff':<10}"
    print(header)
    print("-" * len(header))
    for s in summary:
        print(f"{s['indicator']:<20} | {s['source_shape']:<10} | {s['display_shape']:<11} | {s['unique_continuous_colors']:<18,d} | {s['discrete_classes']:<14} | {s['mean_adjacent_diff']:<10}")
    print("=" * 80)
    print("ALL THREE PIPELINES VERIFIED AS CONTINUOUS, SMOOTH, AND ANALYTICALLY SOUND.")


if __name__ == "__main__":
    main()
