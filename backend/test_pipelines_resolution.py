import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agent.executor import execute_plan


def run_pipeline_test(indicator: str, tool_name: str):
    """
    Run execution plan for a specific indicator pipeline and extract
    spatial raster and PNG visualization metrics.
    """
    tools = ["search_imagery", tool_name, "detect_change"]
    context = {
        "time_start": "2021",
        "time_end": "2025",
    }

    results = execute_plan(tools, context=context)

    change_result = results["detect_change"]
    change_map = change_result.get("change_map")
    assert change_map is not None, f"{indicator}: change_map missing from detect_change"

    source_raster_shape = change_map.shape
    vis = change_result.get("visualization", {})
    png_path = vis.get("path")
    assert png_path and os.path.exists(png_path), f"{indicator}: visualization PNG not found at {png_path}"

    # Load PNG and inspect
    im = Image.open(png_path)
    png_width, png_height = im.size
    display_raster_shape = (png_height, png_width)
    file_size_bytes = os.path.getsize(png_path)
    file_size_kb = round(file_size_bytes / 1024, 2)

    bounds = vis.get("bounds")
    crs = vis.get("crs")
    valid_pixels = vis.get("valid_pixels")
    if valid_pixels is None:
        valid_pixels = int(np.sum(np.isfinite(change_map)))

    # Spatial variance test: ensure displayed pixels actually vary spatially across the AOI
    arr = np.array(im)
    unique_colors = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))

    # Print required logs
    print("-" * 75)
    print(f"INDICATOR:              {indicator.upper()}")
    print(f"Source Raster Shape:    {source_raster_shape}")
    print(f"Display Raster Shape:   {display_raster_shape}")
    print(f"PNG Width:              {png_width} px")
    print(f"PNG Height:             {png_height} px")
    print(f"PNG File Size:          {file_size_kb} KB ({file_size_bytes} bytes)")
    print(f"Bounds:                 {bounds}")
    print(f"CRS:                    {crs}")
    print(f"Number of Valid Pixels: {valid_pixels}")
    print(f"Unique Color Signatures: {unique_colors}")
    print("-" * 75)

    # Assertions
    assert source_raster_shape[0] >= 500 and source_raster_shape[1] >= 500, (
        f"Source raster shape {source_raster_shape} is unexpectedly small!"
    )
    assert png_width >= 500 and png_height >= 500, (
        f"PNG dimensions {png_width}x{png_height} are unexpectedly small!"
    )
    assert file_size_kb > 2.0, (
        f"PNG file size {file_size_kb} KB is too small (expected > 2 KB)!"
    )
    assert valid_pixels >= 100_000, (
        f"Valid pixel count {valid_pixels} is too low!"
    )
    assert unique_colors >= 2, (
        f"Image has insufficient spatial color variation ({unique_colors} colors)!"
    )

    return {
        "indicator": indicator,
        "source_raster_shape": str(source_raster_shape),
        "display_raster_shape": str(display_raster_shape),
        "png_width": png_width,
        "png_height": png_height,
        "file_size": f"{file_size_kb} KB",
        "bounds": str(bounds),
        "crs": str(crs),
        "valid_pixels": valid_pixels,
    }


def main():
    print("=" * 75)
    print("RUNNING REAL PIPELINE TESTS FOR ALL THREE INDICATORS")
    print("=" * 75)

    pipelines = [
        ("NDVI / Vegetation", "calculate_temporal_ndvi"),
        ("NDWI / Water", "calculate_temporal_ndwi"),
        ("NDBI / Urban", "calculate_temporal_ndbi"),
    ]

    summary = []
    for indicator, tool in pipelines:
        res = run_pipeline_test(indicator, tool)
        summary.append(res)

    print("\n" + "=" * 75)
    print("PIPELINE TEST SUMMARY TABLE")
    print("=" * 75)
    header = f"{'Indicator':<20} | {'Source Shape':<14} | {'PNG Dimensions':<16} | {'File Size':<10} | {'Valid Pixels':<12} | {'CRS':<10}"
    print(header)
    print("-" * len(header))
    for s in summary:
        dim_str = f"{s['png_width']}x{s['png_height']}"
        print(f"{s['indicator']:<20} | {s['source_raster_shape']:<14} | {dim_str:<16} | {s['file_size']:<10} | {s['valid_pixels']:<12} | {s['crs']:<10}")
    print("=" * 75)
    print("ALL THREE PIPELINES PASSED VERIFICATION.")


if __name__ == "__main__":
    main()
