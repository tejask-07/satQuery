"""
Phase 6 Test Suite: Spatial Reasoning, Candidate Region Clustering & Geometric Contiguity.

Verifies:
1. Connected-component detection (8-connectivity vs 4-connectivity)
2. MMU filtering (removes < 5 pixels, retains >= 5 pixels)
3. Region pixel counts
4. Region georeferenced area calculation (scaled by latitude for WGS84)
5. Geographic centroid calculation
6. Bounding box generation (pixel and geographic)
7. Per-region scientific statistics from continuous delta arrays
8. Spatial coherence calculation and bounds [0.0, 1.0]
9. Strict invalid-pixel exclusion (masked pixels never form candidate regions)
10. Raw vs. filtered candidate masks distinction
11. Polygon vectorization
12. GeoJSON FeatureCollection validity
13. Multiple independent regions
14. Spatial transition matching
15. Spatial transition rejection when no overlap/proximity exists
16. Relative location description generator
17. Edge case: all zero candidate raster
18. Edge case: single pixel candidate (< MMU)
19. Edge case: region touching raster boundary and diagonal connectivity
20. API / Schema / Pipeline integration with Phase 5B and Phase 5C
"""

import math
import numpy as np
import pytest
from affine import Affine

from app.evidence.spatial import (
    CandidateRegion,
    SpatialConfig,
    calculate_pixel_area_m2,
    calculate_spatial_coherence,
    compute_geographic_bbox,
    describe_spatial_location,
    extract_connected_candidate_regions,
    extract_spatial_candidate_regions,
    match_spatial_transitions,
    pixel_to_geo,
    vectorize_candidate_regions,
)


@pytest.fixture
def dummy_transform():
    # Affine: pixel size 0.0001 degrees, top-left at (16.40, 48.21)
    return Affine(0.0001, 0.0, 16.40, 0.0, -0.0001, 48.21)


def test_connected_component_detection(dummy_transform):
    """Verifies that 8-connected components are detected and uniquely labeled."""
    cand_map = np.zeros((30, 30), dtype=np.uint8)
    cand_map[5:10, 5:10] = 1  # 5x5 = 25 pixels (primary candidate)

    regions, filtered_mask, labeled = extract_connected_candidate_regions(
        candidate_map=cand_map,
        transform=dummy_transform,
        mmu_min_pixels=5,
        connectivity=8,
    )

    assert len(regions) == 1
    reg = regions[0]
    assert reg.region_id == 1
    assert reg.candidate_class == 1
    assert reg.pixel_count == 25
    assert np.sum(filtered_mask == 1) == 25
    assert np.sum(labeled == 1) == 25


def test_4_vs_8_connectivity(dummy_transform):
    """Diagonal pixels form one region under 8-connectivity, but two regions under 4-connectivity."""
    cand_map = np.zeros((10, 10), dtype=np.uint8)
    cand_map[2, 2] = 1
    cand_map[3, 3] = 1
    cand_map[4, 4] = 1
    cand_map[5, 5] = 1
    cand_map[6, 6] = 1

    # Under 8-connectivity, diagonal line of 5 pixels is ONE region >= MMU(5)
    reg_8, _, _ = extract_connected_candidate_regions(
        candidate_map=cand_map,
        transform=dummy_transform,
        mmu_min_pixels=5,
        connectivity=8,
    )
    assert len(reg_8) == 1
    assert reg_8[0].pixel_count == 5

    # Under 4-connectivity, each pixel is isolated (pixel_count=1 < MMU), so 0 regions retained
    reg_4, _, _ = extract_connected_candidate_regions(
        candidate_map=cand_map,
        transform=dummy_transform,
        mmu_min_pixels=5,
        connectivity=4,
    )
    assert len(reg_4) == 0


def test_mmu_filtering_removes_noise(dummy_transform):
    """Small clusters (< MMU_MIN_PIXELS) are filtered out, while >= MMU are retained."""
    cand_map = np.zeros((30, 30), dtype=np.uint8)
    cand_map[2:4, 2:4] = 1    # 2x2 = 4 pixels (< MMU 5)
    cand_map[10:15, 10:15] = 1 # 5x5 = 25 pixels (>= MMU 5)

    regions, filtered_mask, labeled = extract_connected_candidate_regions(
        candidate_map=cand_map,
        transform=dummy_transform,
        mmu_min_pixels=5,
        connectivity=8,
    )

    assert len(regions) == 1
    assert regions[0].pixel_count == 25

    # Raw candidate map has 29 candidate pixels; filtered mask has exactly 25
    assert np.sum(cand_map == 1) == 29
    assert np.sum(filtered_mask == 1) == 25
    assert filtered_mask[2, 2] == 0  # small cluster zeroed out


def test_region_georeferenced_area_calculation(dummy_transform):
    """Area calculation must scale longitude resolution by latitude cosine for EPSG:4326."""
    lat_vienna = 48.20
    px_area = calculate_pixel_area_m2(dummy_transform, centroid_lat=lat_vienna, crs="EPSG:4326")

    # Expected: dx_m = 0.0001 * 111320 * cos(48.2 deg) ~ 7.42 m
    # dy_m = 0.0001 * 111320 ~ 11.13 m
    # Area ~ 82.6 m2 per pixel
    expected_px_area = (0.0001 * 111320.0 * math.cos(math.radians(lat_vienna))) * (0.0001 * 111320.0)
    assert pytest.approx(px_area, rel=1e-3) == expected_px_area

    # Projected CRS should not use cosine scaling
    proj_transform = Affine(10.0, 0.0, 500000, 0.0, -10.0, 5300000)
    px_area_proj = calculate_pixel_area_m2(proj_transform, centroid_lat=lat_vienna, crs="EPSG:32633")
    assert pytest.approx(px_area_proj, rel=1e-3) == 100.0


def test_geographic_centroid_calculation(dummy_transform):
    """Pixel centroid translates to correct geographic lon/lat."""
    lon, lat = pixel_to_geo(dummy_transform, col=10.0, row=20.0)
    # col 10: 16.40 + 10.5 * 0.0001 = 16.40105
    # row 20: 48.21 - 20.5 * 0.0001 = 48.20795
    assert pytest.approx(lon, abs=1e-4) == 16.40105
    assert pytest.approx(lat, abs=1e-4) == 48.20795


def test_geographic_bounding_box(dummy_transform):
    """Bounding box covers the correct geographic extent."""
    bbox = compute_geographic_bbox(dummy_transform, min_col=5, min_row=10, width=15, height=20)
    min_lon, min_lat, max_lon, max_lat = bbox
    assert min_lon < max_lon
    assert min_lat < max_lat
    assert 16.39 < min_lon < 16.42
    assert 48.19 < min_lat < 48.22


def test_region_statistics_sampling(dummy_transform):
    """Region statistics sample continuous scientific delta rasters."""
    cand_map = np.zeros((20, 20), dtype=np.uint8)
    cand_map[5:10, 5:10] = 1

    ndvi_arr = np.full((20, 20), -0.25, dtype=np.float32)
    ndbi_arr = np.full((20, 20), 0.32, dtype=np.float32)

    regions, _, _ = extract_connected_candidate_regions(
        candidate_map=cand_map,
        transform=dummy_transform,
        mmu_min_pixels=5,
        delta_arrays={"ndvi": ndvi_arr, "ndbi": ndbi_arr},
    )

    assert len(regions) == 1
    reg = regions[0]
    assert reg.mean_ndvi_delta == -0.25
    assert reg.mean_ndbi_delta == 0.32


def test_spatial_coherence_bounds():
    """Spatial coherence score is strictly bounded in [0.0, 1.0]."""
    # Compact square
    square = np.ones((10, 10), dtype=np.uint8)
    score_compact = calculate_spatial_coherence(square, pixel_count=100)
    assert 0.0 <= score_compact <= 1.0

    # Irregular / scattered line
    sparse = np.zeros((100, 100), dtype=np.uint8)
    sparse[::5, ::5] = 1
    score_sparse = calculate_spatial_coherence(sparse, pixel_count=20)
    assert 0.0 <= score_sparse <= 1.0
    assert score_compact > score_sparse


def test_strict_invalid_pixel_exclusion(dummy_transform):
    """Masked / invalid pixels must NEVER contribute to candidate regions."""
    cand_map = np.ones((20, 20), dtype=np.uint8)  # all pixels marked candidate
    joint_mask = np.zeros((20, 20), dtype=bool)   # all pixels masked invalid

    regions, filtered_mask, labeled = extract_connected_candidate_regions(
        candidate_map=cand_map,
        transform=dummy_transform,
        joint_mask=joint_mask,
        mmu_min_pixels=5,
    )

    assert len(regions) == 0
    assert np.sum(filtered_mask) == 0
    assert np.sum(labeled) == 0


def test_partial_validity_masking(dummy_transform):
    """Only valid pixels within a candidate cluster are retained."""
    cand_map = np.zeros((20, 20), dtype=np.uint8)
    cand_map[5:15, 5:15] = 1  # 10x10 = 100 pixels

    joint_mask = np.zeros((20, 20), dtype=bool)
    joint_mask[5:15, 5:10] = True  # only half is valid (50 pixels)

    regions, filtered_mask, _ = extract_connected_candidate_regions(
        candidate_map=cand_map,
        transform=dummy_transform,
        joint_mask=joint_mask,
        mmu_min_pixels=5,
    )

    assert len(regions) == 1
    assert regions[0].pixel_count == 50
    assert np.sum(filtered_mask == 1) == 50


def test_vectorization_geojson_validity(dummy_transform):
    """Candidate regions vectorize to a valid GeoJSON FeatureCollection."""
    cand_map = np.zeros((30, 30), dtype=np.uint8)
    cand_map[5:12, 5:12] = 1

    regions, _, _ = extract_connected_candidate_regions(
        candidate_map=cand_map,
        transform=dummy_transform,
        mmu_min_pixels=5,
    )

    geojson_fc = vectorize_candidate_regions(regions)

    assert geojson_fc["type"] == "FeatureCollection"
    assert len(geojson_fc["features"]) == 1
    feat = geojson_fc["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] in ["Polygon", "MultiPolygon"]
    assert feat["properties"]["region_id"] == 1
    assert feat["properties"]["pixel_count"] == 49
    assert feat["properties"]["area_hectares"] > 0


def test_multiple_independent_regions(dummy_transform):
    """Multiple disjoint clusters are extracted as separate regions with unique IDs."""
    cand_map = np.zeros((50, 50), dtype=np.uint8)
    cand_map[5:12, 5:12] = 1     # Region 1: 49 pixels
    cand_map[25:35, 25:35] = 1   # Region 2: 100 pixels
    cand_map[40:48, 10:18] = 2   # Region 3 (counter candidate): 64 pixels

    regions, filtered_mask, labeled = extract_connected_candidate_regions(
        candidate_map=cand_map,
        transform=dummy_transform,
        mmu_min_pixels=5,
    )

    assert len(regions) == 3
    ids = [r.region_id for r in regions]
    assert ids == [1, 2, 3]

    # Check distinct classes
    classes = [r.candidate_class for r in regions]
    assert classes == [1, 1, 2]


def test_spatial_transition_matching(dummy_transform):
    """Overlapping source and destination regions produce candidate transition."""
    # Source region (vegetation loss) at [10:20, 10:20]
    s_reg = CandidateRegion(
        region_id=1,
        candidate_class=1,
        class_label="vegetation_loss",
        pixel_count=100,
        area_m2=8200.0,
        area_hectares=0.82,
        centroid_pixel=[15.0, 15.0],
        centroid_geo=[16.4015, 48.2085],
        bbox_pixel=[10, 10, 10, 10],
        bbox_geo=[16.401, 48.208, 16.402, 48.209],
        spatial_coherence=0.75,
        valid_pixel_fraction=1.0,
    )

    # Destination region (urban expansion) overlapping at [12:22, 12:22]
    d_reg = CandidateRegion(
        region_id=2,
        candidate_class=1,
        class_label="urban_expansion",
        pixel_count=100,
        area_m2=8200.0,
        area_hectares=0.82,
        centroid_pixel=[17.0, 17.0],
        centroid_geo=[16.4017, 48.2083],
        bbox_pixel=[12, 12, 10, 10],
        bbox_geo=[16.4012, 48.2082, 16.4022, 48.2092],
        spatial_coherence=0.80,
        valid_pixel_fraction=1.0,
    )

    transitions = match_spatial_transitions(
        source_regions=[s_reg],
        dest_regions=[d_reg],
        transform=dummy_transform,
        shape=(50, 50),
    )

    assert len(transitions) == 1
    assert transitions[0]["source_region_id"] == 1
    assert transitions[0]["destination_region_id"] == 2
    assert transitions[0]["overlap_fraction"] > 0.50
    assert transitions[0]["status"] == "candidate_transition"


def test_spatial_transition_rejection_no_overlap(dummy_transform):
    """Disjoint source and destination regions reject transition claim."""
    s_reg = CandidateRegion(
        region_id=1,
        candidate_class=1,
        class_label="vegetation_loss",
        pixel_count=50,
        area_m2=4100.0,
        area_hectares=0.41,
        centroid_pixel=[5.0, 5.0],
        centroid_geo=[16.4005, 48.2095],
        bbox_pixel=[2, 2, 6, 6],
        bbox_geo=[16.4002, 48.2092, 16.4008, 48.2098],
        spatial_coherence=0.70,
        valid_pixel_fraction=1.0,
    )

    # Far away destination region
    d_reg = CandidateRegion(
        region_id=2,
        candidate_class=1,
        class_label="urban_expansion",
        pixel_count=50,
        area_m2=4100.0,
        area_hectares=0.41,
        centroid_pixel=[40.0, 40.0],
        centroid_geo=[16.4040, 48.2060],
        bbox_pixel=[38, 38, 6, 6],
        bbox_geo=[16.4038, 48.2058, 16.4044, 48.2064],
        spatial_coherence=0.70,
        valid_pixel_fraction=1.0,
    )

    transitions = match_spatial_transitions(
        source_regions=[s_reg],
        dest_regions=[d_reg],
        transform=dummy_transform,
        shape=(50, 50),
    )

    assert len(transitions) == 0


def test_relative_location_description():
    """Centroids correctly translate to relative quadrant/cardinal directions."""
    aoi_bounds = [[48.20, 16.40], [48.21, 16.41]]

    # Eastern side centroid
    east_desc = describe_spatial_location([(16.409, 48.205)], aoi_bounds=aoi_bounds)
    assert "eastern" in east_desc

    # Northern side centroid
    north_desc = describe_spatial_location([(16.405, 48.209)], aoi_bounds=aoi_bounds)
    assert "northern" in north_desc

    # Central centroid
    center_desc = describe_spatial_location([(16.405, 48.205)], aoi_bounds=aoi_bounds)
    assert "central" in center_desc


def test_edge_case_all_zero_candidate_raster(dummy_transform):
    """Zero candidate map produces 0 regions cleanly without error."""
    zero_map = np.zeros((20, 20), dtype=np.uint8)
    regions, filtered, labeled = extract_connected_candidate_regions(
        candidate_map=zero_map,
        transform=dummy_transform,
        mmu_min_pixels=5,
    )
    assert len(regions) == 0
    assert np.sum(filtered) == 0
    assert np.sum(labeled) == 0


def test_edge_case_single_pixel_candidate(dummy_transform):
    """Single candidate pixel is filtered out by MMU."""
    cand_map = np.zeros((20, 20), dtype=np.uint8)
    cand_map[10, 10] = 1

    regions, filtered, _ = extract_connected_candidate_regions(
        candidate_map=cand_map,
        transform=dummy_transform,
        mmu_min_pixels=5,
    )
    assert len(regions) == 0
    assert np.sum(filtered) == 0


def test_edge_case_region_touching_boundary(dummy_transform):
    """Region touching raster boundary is extracted with valid bbox."""
    cand_map = np.zeros((20, 20), dtype=np.uint8)
    cand_map[0:5, 0:5] = 1  # touches top-left border

    regions, filtered, _ = extract_connected_candidate_regions(
        candidate_map=cand_map,
        transform=dummy_transform,
        mmu_min_pixels=5,
    )
    assert len(regions) == 1
    assert regions[0].bbox_pixel == [0, 0, 5, 5]
    assert regions[0].pixel_count == 25


def test_deterministic_repeatability(dummy_transform):
    """Running extraction multiple times produces identical region counts, areas, and IDs."""
    cand_map = np.zeros((30, 30), dtype=np.uint8)
    cand_map[5:12, 5:12] = 1
    cand_map[18:25, 18:25] = 2

    res1, f1, _ = extract_connected_candidate_regions(cand_map, dummy_transform, mmu_min_pixels=5)
    res2, f2, _ = extract_connected_candidate_regions(cand_map, dummy_transform, mmu_min_pixels=5)

    assert len(res1) == len(res2)
    assert np.array_equal(f1, f2)
    for r1, r2 in zip(res1, res2):
        assert r1.region_id == r2.region_id
        assert r1.area_m2 == r2.area_m2
        assert r1.centroid_geo == r2.centroid_geo
        assert r1.spatial_coherence == r2.spatial_coherence


def test_e2e_query_produces_phase6_spatial_analysis(monkeypatch):
    """End-to-end query execution verifies that spatial_analysis is computed and attached."""
    from unittest.mock import patch
    from app.api.routes_query import process_query
    from app.schemas.query import QueryRequest

    with patch("app.api.routes_query.VLM.generate", return_value="VLM offline test"):
        req = QueryRequest(
            query="Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
        )
        result = process_query(req)

        assert result.status == "success"
        assert result.spatial_analysis is not None
        assert "spatial_analysis" in result.statistics

        sp = result.spatial_analysis
        assert "region_count" in sp
        assert "total_candidate_area_hectares" in sp
        assert "geojson" in sp
        assert sp["geojson"]["type"] == "FeatureCollection"
        assert "parameters" in sp
        assert "mmu_min_pixels" in sp["parameters"]

        # Check execution trace
        assert any("Phase 6 Spatial" in t for t in result.execution_trace)

        # Check layer package includes spatial
        if sp.get("available"):
            assert "spatial" in result.layer_package
            assert "geojson" in result.layer_package["spatial"]
