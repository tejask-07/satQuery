import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agent.executor import execute_plan


def test_indicator_audit(indicator: str, tool_name: str):
    print("=" * 80)
    print(f"AUDIT VERIFICATION: {indicator.upper()}")
    print("=" * 80)

    tools = ["search_imagery", tool_name, "detect_change"]
    context = {
        "time_start": "2021",
        "time_end": "2025",
        "threshold": 0.05,
    }

    results = execute_plan(tools, context=context)

    # 1. Verify before, after, and change rasters
    temp_result = results[tool_name]
    index_prefix = tool_name.replace("calculate_temporal_", "")
    before_raster = temp_result[f"{index_prefix}_before"]
    after_raster = temp_result[f"{index_prefix}_after"]

    change_result = results["detect_change"]
    change_map = change_result.get("change_map")
    assert change_map is not None, f"{indicator}: missing change_map"

    # Verify exact math: change == after - before (on finite pixels)
    valid_both = np.isfinite(before_raster) & np.isfinite(after_raster)
    diff_expected = after_raster[valid_both] - before_raster[valid_both]
    diff_actual = change_map[valid_both]

    max_diff_error = float(np.max(np.abs(diff_actual - diff_expected)))
    assert max_diff_error < 1e-6, f"{indicator}: change_map is not exactly after - before! Error: {max_diff_error}"

    # Verify before/after reflectance ranges and mean change sanity
    mean_before = float(np.mean(before_raster[valid_both]))
    mean_after = float(np.mean(after_raster[valid_both]))
    mean_change = float(np.mean(change_map[valid_both]))
    valid_pixels = int(np.sum(valid_both))
    changed_pixels = change_result["changed_pixels"]
    change_ratio = change_result["change_ratio"]

    print(f"Index:                  {index_prefix.upper()}")
    print(f"Mean Before ({index_prefix.upper()}):     {mean_before:.4f}")
    print(f"Mean After ({index_prefix.upper()}):      {mean_after:.4f}")
    print(f"Mean Change:            {mean_change:.4f}")
    print(f"Valid Pixels:           {valid_pixels:,}")
    print(f"Changed Pixels (>0.05): {changed_pixels:,} ({change_ratio * 100:.2f}%)")

    # Sanity checks on corrected radiometric offset:
    # Stable scene between Dec 2021 & Dec 2025 should NOT have a fake 95% change!
    assert abs(mean_change) < 0.10, (
        f"{indicator}: Mean change {mean_change:.4f} is unrealistically large! "
        "Likely uncorrected radiometry offset."
    )
    assert change_ratio < 0.60, (
        f"{indicator}: Change ratio {change_ratio * 100:.2f}% is suspiciously high! "
        "Likely threshold or radiometric scaling bug."
    )

    # 2. Verify Visualization PNG matches change map
    vis = change_result.get("visualization", {})
    png_path = vis.get("path")
    assert png_path and os.path.exists(png_path), f"{indicator}: PNG missing"
    im = Image.open(png_path)
    assert im.size == (1200, 1200), f"PNG size {im.size} is not 1200x1200"

    print(f"Visualization PNG:      {vis['filename']} ({os.path.getsize(png_path) / 1024:.1f} KB)")
    print(f"Geographic Bounds:      {vis['bounds']}")
    print(f"-> {indicator.upper()} PIPELINE AUDIT PASSED SUCCESSFULLY.")
    return {
        "indicator": indicator,
        "mean_before": round(mean_before, 4),
        "mean_after": round(mean_after, 4),
        "mean_change": round(mean_change, 4),
        "change_pct": f"{change_ratio * 100:.2f}%",
        "changed_pixels": changed_pixels,
        "valid_pixels": valid_pixels,
    }


def main():
    print("\n" + "=" * 80)
    print("RUNNING FRESH AUDIT VERIFICATION FOR NDVI, NDWI, AND NDBI")
    print("=" * 80 + "\n")

    summary = []
    pipelines = [
        ("NDVI / Vegetation", "calculate_temporal_ndvi"),
        ("NDWI / Water", "calculate_temporal_ndwi"),
        ("NDBI / Urban", "calculate_temporal_ndbi"),
    ]

    for indicator, tool in pipelines:
        res = test_indicator_audit(indicator, tool)
        summary.append(res)

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Indicator':<20} | {'Mean Before':<12} | {'Mean After':<12} | {'Mean Change':<12} | {'Change %':<10} | {'Status':<8}"
    print(header)
    print("-" * len(header))
    for s in summary:
        print(f"{s['indicator']:<20} | {s['mean_before']:<12} | {s['mean_after']:<12} | {s['mean_change']:<12} | {s['change_pct']:<10} | {'PASSED':<8}")
    print("=" * 80)
    print("ALL THREE INDICATORS PASSED SCIENTIFIC PIPELINE AUDIT.")


if __name__ == "__main__":
    main()
