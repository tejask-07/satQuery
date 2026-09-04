"""
Phase 4 Verification Test Suite: Complete Scientific Layer Package.

Verifies that for every temporal analysis (urban, vegetation, water):
1. All three spectral indices (NDVI, NDWI, NDBI) are computed for BEFORE and AFTER.
2. True-color, false-color, and quality masks are generated for BEFORE and AFTER.
3. All three delta change rasters (Delta NDVI, Delta NDWI, Delta NDBI) are computed
   with both continuous and classified visualizations.
4. The complete scientific layer package is returned in:
   - result.layer_package (structured dictionary: before, after, change, quality)
   - result.layers (distinct layer objects with unique IDs)
5. Raster URLs are distinct and independent (no sharing single URL or fallback).
6. Target semantics are preserved (Urban -> NDBI, Vegetation -> NDVI, Water -> NDWI).
"""

from unittest.mock import patch
import pytest

from app.api.routes_query import process_query
from app.schemas.query import QueryRequest


@pytest.fixture
def mock_vlm_offline():
    """Mock VLM so tests don't require external HuggingFace API calls."""
    with patch("app.api.routes_query.VLM.generate", return_value="Verified scientific layer package."):
        yield


def test_urban_query_generates_all_scientific_layers(mock_vlm_offline):
    """
    Test that an urban change query generates:
    - Primary metric: NDBI
    - All 3 indices for Before and After (NDVI, NDWI, NDBI)
    - All 3 deltas (Delta NDVI, Delta NDWI, Delta NDBI)
    - True-color and False-color composites
    - Quality masks for Before and After
    - Distinct, non-colliding URLs for each raster
    """
    req = QueryRequest(
        query="Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
    )
    result = process_query(req)

    assert result.status == "success"

    # 1. Target semantics: Urban query primary metric must be NDBI
    assert result.statistics.get("metric") == "NDBI"
    assert "mean_before" in result.statistics
    assert "mean_after" in result.statistics
    assert "mean_change" in result.statistics

    # 2. Multi-index statistics
    indices_stats = result.statistics.get("indices")
    assert indices_stats is not None
    assert "NDVI" in indices_stats
    assert "NDWI" in indices_stats
    assert "NDBI" in indices_stats
    for k in ["NDVI", "NDWI", "NDBI"]:
        assert indices_stats[k].get("mean_before") is not None
        assert indices_stats[k].get("mean_after") is not None
        assert indices_stats[k].get("mean_change") is not None

    # 3. Layer Package Structure
    pkg = result.layer_package
    assert pkg is not None
    assert "before" in pkg
    assert "after" in pkg
    assert "change" in pkg
    assert "quality" in pkg

    # Before layers
    assert pkg["before"]["true_color"]["url"] is not None
    assert pkg["before"]["false_color"]["url"] is not None
    assert pkg["before"]["ndvi"]["url"] is not None
    assert pkg["before"]["ndwi"]["url"] is not None
    assert pkg["before"]["ndbi"]["url"] is not None

    # After layers
    assert pkg["after"]["true_color"]["url"] is not None
    assert pkg["after"]["false_color"]["url"] is not None
    assert pkg["after"]["ndvi"]["url"] is not None
    assert pkg["after"]["ndwi"]["url"] is not None
    assert pkg["after"]["ndbi"]["url"] is not None

    # Change layers: continuous & classified for all 3
    assert pkg["change"]["delta_ndvi"]["url"] is not None
    assert pkg["change"]["delta_ndvi"]["classified_url"] is not None
    assert pkg["change"]["delta_ndwi"]["url"] is not None
    assert pkg["change"]["delta_ndwi"]["classified_url"] is not None
    assert pkg["change"]["delta_ndbi"]["url"] is not None
    assert pkg["change"]["delta_ndbi"]["classified_url"] is not None

    # Quality layers
    assert pkg["quality"]["mask_before"]["url"] is not None
    assert pkg["quality"]["mask_after"]["url"] is not None

    # 4. URL Independence (no sharing a single raster URL)
    delta_urls = [
        pkg["change"]["delta_ndvi"]["url"],
        pkg["change"]["delta_ndwi"]["url"],
        pkg["change"]["delta_ndbi"]["url"],
    ]
    assert len(set(delta_urls)) == 3, f"Delta URLs must be distinct: {delta_urls}"

    before_index_urls = [
        pkg["before"]["ndvi"]["url"],
        pkg["before"]["ndwi"]["url"],
        pkg["before"]["ndbi"]["url"],
    ]
    assert len(set(before_index_urls)) == 3, f"Before index URLs must be distinct: {before_index_urls}"

    after_index_urls = [
        pkg["after"]["ndvi"]["url"],
        pkg["after"]["ndwi"]["url"],
        pkg["after"]["ndbi"]["url"],
    ]
    assert len(set(after_index_urls)) == 3, f"After index URLs must be distinct: {after_index_urls}"

    # 5. Layers list backwards compatibility and distinct IDs
    layer_ids = {l["id"] for l in result.layers}
    expected_ids = {
        "true_color_before",
        "true_color_after",
        "false_color_before",
        "false_color_after",
        "quality_mask_before",
        "quality_mask_after",
        "ndvi_before",
        "ndvi_after",
        "ndwi_before",
        "ndwi_after",
        "ndbi_before",
        "ndbi_after",
        "change_ndvi",
        "change_ndwi",
        "change_ndbi",
        "change_continuous",
    }
    assert expected_ids.issubset(layer_ids), f"Missing layers: {expected_ids - layer_ids}"


def test_vegetation_query_generates_all_scientific_layers(mock_vlm_offline):
    """
    Test that a vegetation change query generates:
    - Primary metric: NDVI
    - Complete layer package with all 3 indices and 3 deltas
    """
    req = QueryRequest(
        query="Compare vegetation change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
    )
    result = process_query(req)

    assert result.status == "success"
    assert result.statistics.get("metric") == "NDVI"

    pkg = result.layer_package
    assert pkg is not None
    assert pkg["before"]["ndvi"]["url"] is not None
    assert pkg["before"]["ndwi"]["url"] is not None
    assert pkg["before"]["ndbi"]["url"] is not None
    assert pkg["change"]["delta_ndvi"]["url"] is not None
    assert pkg["change"]["delta_ndwi"]["url"] is not None
    assert pkg["change"]["delta_ndbi"]["url"] is not None


def test_water_query_generates_all_scientific_layers(mock_vlm_offline):
    """
    Test that a water change query generates:
    - Primary metric: NDWI
    - Complete layer package with all 3 indices and 3 deltas
    """
    req = QueryRequest(
        query="Compare water change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
    )
    result = process_query(req)

    assert result.status == "success"
    assert result.statistics.get("metric") == "NDWI"

    pkg = result.layer_package
    assert pkg is not None
    assert pkg["before"]["ndvi"]["url"] is not None
    assert pkg["before"]["ndwi"]["url"] is not None
    assert pkg["before"]["ndbi"]["url"] is not None
    assert pkg["change"]["delta_ndvi"]["url"] is not None
    assert pkg["change"]["delta_ndwi"]["url"] is not None
    assert pkg["change"]["delta_ndbi"]["url"] is not None
