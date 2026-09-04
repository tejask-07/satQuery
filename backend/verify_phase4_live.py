"""
Live Verification Script for Phase 4 Complete Scientific Layer Package.

Query: "Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
"""

import os
from pathlib import Path
from pprint import pprint

from app.api.routes_query import process_query
from app.schemas.query import QueryRequest


def main():
    query = "Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
    print(f"\n========================================================")
    print(f"RUNNING LIVE QUERY:")
    print(f"{query}")
    print(f"========================================================\n")

    req = QueryRequest(query=query)
    result = process_query(req)

    print(f"Status: {result.status}")
    print(f"Confidence: {result.confidence}")
    print(f"\nPrimary Metric: {result.statistics.get('metric')}")
    print(f"Mean Before: {result.statistics.get('mean_before')}")
    print(f"Mean After: {result.statistics.get('mean_after')}")
    print(f"Mean Change: {result.statistics.get('mean_change')}")
    print(f"Changed Pixels: {result.statistics.get('changed_pixels')}")
    print(f"Valid Pixels: {result.statistics.get('valid_pixels')}")
    print(f"Change Ratio: {result.statistics.get('change_ratio')}")

    print("\n--------------------------------------------------------")
    print("INDICES STATISTICS (ALL 3 INDICES):")
    print("--------------------------------------------------------")
    pprint(result.statistics.get("indices"))

    print("\n--------------------------------------------------------")
    print("COMPLETE LAYER PACKAGE:")
    print("--------------------------------------------------------")
    pkg = result.layer_package
    assert pkg is not None, "layer_package must not be None"

    print("\n[BEFORE LAYERS]")
    for k, v in pkg["before"].items():
        print(f"  - {k}: url={v.get('url')} | bounds={v.get('bounds')}")

    print("\n[AFTER LAYERS]")
    for k, v in pkg["after"].items():
        print(f"  - {k}: url={v.get('url')} | bounds={v.get('bounds')}")

    print("\n[CHANGE LAYERS]")
    for k, v in pkg["change"].items():
        print(f"  - {k}: url={v.get('url')} | classified_url={v.get('classified_url')} | bounds={v.get('bounds')}")

    print("\n[QUALITY LAYERS]")
    for k, v in pkg["quality"].items():
        print(f"  - {k}: url={v.get('url')} | bounds={v.get('bounds')} | platform={v.get('metadata', {}).get('platform') if v.get('metadata') else None}")

    print("\n--------------------------------------------------------")
    print("FILE EXISTENCE & INDEPENDENCE VERIFICATION:")
    print("--------------------------------------------------------")
    from app.evidence.scientific_visualizations import VISUALIZATION_DIR
    vis_dir = VISUALIZATION_DIR
    all_urls = []
    
    # Check change deltas
    deltas = [
        pkg["change"]["delta_ndvi"]["url"],
        pkg["change"]["delta_ndwi"]["url"],
        pkg["change"]["delta_ndbi"]["url"],
    ]
    for d_url in deltas:
        assert d_url is not None, f"Delta url missing: {d_url}"
        fname = Path(d_url).name
        fpath = vis_dir / fname
        exists = fpath.exists()
        size = fpath.stat().st_size if exists else 0
        print(f"Delta file: {fname} -> exists={exists}, size={size} bytes")
        assert exists, f"File {fpath} must exist on disk!"
        all_urls.append(d_url)

    # Check distinctness
    assert len(set(deltas)) == 3, "All 3 deltas must have distinct filenames/URLs!"
    print(f"\nDelta distinctness verified: 3 distinct rasters generated.")

    print("\n[LAYERS LIST]")
    print(f"Total layers returned: {len(result.layers)}")
    layer_ids = [l["id"] for l in result.layers]
    print(f"Layer IDs: {layer_ids}")

    print("\nSUCCESS: Phase 4 Complete Scientific Layer Package is fully functional!")


if __name__ == "__main__":
    main()
