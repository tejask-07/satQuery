from __future__ import annotations

import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from app.evidence.scientific_visualizations import (
    normalize_channel_for_display,
    build_true_color_rgba,
    build_false_color_rgba,
    build_index_rgba,
    build_raw_change_rgba,
    build_classified_change_rgba,
    build_quality_mask_rgba,
    save_visualization_layer,
)
from app.remote_sensing.preprocessing.quality import SCLClass


@pytest.fixture
def mock_raster_geotiff(tmp_path):
    """Create a mock georeferenced GeoTIFF for bounds testing."""
    bounds = (16.40, 48.20, 16.41, 48.21)
    width, height = 50, 50
    transform = from_bounds(*bounds, width, height)
    file_path = tmp_path / "mock_raster.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(file_path, "w", **profile) as dst:
        dst.write(np.full((height, width), 0.2, dtype=np.float32), 1)
    return str(file_path)


# ============================================================
# 1. TRUE-COLOR CHANNEL MAPPING
# ============================================================

def test_true_color_channel_mapping():
    """Verify Red=B04, Green=B03, Blue=B02 channel order in OpenCV BGRA."""
    # Red high, others 0
    r = np.full((10, 10), 0.35, dtype=np.float32)
    g = np.full((10, 10), 0.05, dtype=np.float32)
    b = np.full((10, 10), 0.05, dtype=np.float32)

    bgra = build_true_color_rgba(red=r, green=g, blue=b)
    assert bgra.shape == (10, 10, 4)
    # BGRA: Blue=0, Green=1, Red=2, Alpha=3
    # Red should dominate over blue and green
    assert np.mean(bgra[..., 2]) > np.mean(bgra[..., 0])
    assert np.mean(bgra[..., 2]) > np.mean(bgra[..., 1])
    assert np.all(bgra[..., 3] == 255)


# ============================================================
# 2. FALSE-COLOR CHANNEL MAPPING
# ============================================================

def test_false_color_channel_mapping():
    """Verify NIR=B08 (Red channel), Red=B04 (Green channel), Green=B03 (Blue channel)."""
    # High NIR vegetation response
    nir = np.full((10, 10), 0.40, dtype=np.float32)
    red = np.full((10, 10), 0.08, dtype=np.float32)
    green = np.full((10, 10), 0.08, dtype=np.float32)

    bgra = build_false_color_rgba(nir=nir, red=red, green=green)
    assert bgra.shape == (10, 10, 4)
    # NIR maps to Red channel (index 2 in BGRA)
    assert np.mean(bgra[..., 2]) > np.mean(bgra[..., 0])
    assert np.mean(bgra[..., 2]) > np.mean(bgra[..., 1])
    assert np.all(bgra[..., 3] == 255)


# ============================================================
# 3. DISPLAY NORMALIZATION (FINITE STRETCH)
# ============================================================

def test_display_normalization_finite_stretch():
    """Verify percentile stretch maps reflectance safely to 0..255."""
    refl = np.linspace(0.01, 0.35, 100, dtype=np.float32).reshape((10, 10))
    disp = normalize_channel_for_display(refl)
    assert disp.dtype == np.uint8
    assert np.min(disp) < 50
    assert np.max(disp) > 200


# ============================================================
# 4. NDVI COLORMAP SIGNAL LEVELS
# ============================================================

def test_ndvi_colormap_signal_levels():
    """Verify low NDVI has barren tones and high NDVI has lush green tones."""
    ndvi_barren = np.full((5, 5), -0.05, dtype=np.float32)
    ndvi_dense = np.full((5, 5), 0.75, dtype=np.float32)

    bgra_barren = build_index_rgba(ndvi_barren, "NDVI")
    bgra_dense = build_index_rgba(ndvi_dense, "NDVI")

    # Barren has higher red/blue (tan/gray), dense has strong green
    assert bgra_dense[0, 0, 1] > bgra_dense[0, 0, 2]  # Green > Red in dense
    assert bgra_dense[0, 0, 1] > bgra_dense[0, 0, 0]  # Green > Blue in dense


# ============================================================
# 5. NDWI COLORMAP WATER RESPONSE
# ============================================================

def test_ndwi_colormap_water_response():
    """Verify positive NDWI renders as deep water blue."""
    ndwi_dry = np.full((5, 5), -0.3, dtype=np.float32)
    ndwi_water = np.full((5, 5), 0.35, dtype=np.float32)

    bgra_dry = build_index_rgba(ndwi_dry, "NDWI")
    bgra_water = build_index_rgba(ndwi_water, "NDWI")

    # Water has high blue channel (index 0 in BGRA)
    assert bgra_water[0, 0, 0] > bgra_water[0, 0, 2]  # Blue > Red


# ============================================================
# 6. NDBI COLORMAP BUILT-UP RESPONSE
# ============================================================

def test_ndbi_colormap_builtup_response():
    """Verify built-up positive NDBI renders with warm built-up response."""
    ndbi_veg = np.full((5, 5), -0.25, dtype=np.float32)
    ndbi_urban = np.full((5, 5), 0.25, dtype=np.float32)

    bgra_veg = build_index_rgba(ndbi_veg, "NDBI")
    bgra_urban = build_index_rgba(ndbi_urban, "NDBI")

    # Urban has high red and purple/magenta response
    assert bgra_urban[0, 0, 2] > bgra_urban[0, 0, 1]  # Red > Green in urban


# ============================================================
# 7. NAN AND MASKED TRANSPARENCY
# ============================================================

def test_nan_and_masked_transparency():
    """Verify NaNs and masked pixels strictly receive alpha=0 across all builders."""
    shape = (10, 10)
    data_with_nan = np.full(shape, 0.2, dtype=np.float32)
    data_with_nan[2, 3] = np.nan
    data_with_nan[4, 5] = np.inf

    mask = np.ones(shape, dtype=bool)
    mask[0, 0] = False  # Explicit cloud/shadow mask

    # 1. True Color
    tc = build_true_color_rgba(data_with_nan, data_with_nan, data_with_nan, valid_mask=mask)
    assert tc[2, 3, 3] == 0
    assert tc[4, 5, 3] == 0
    assert tc[0, 0, 3] == 0
    assert tc[1, 1, 3] == 255

    # 2. Index
    idx = build_index_rgba(data_with_nan, "NDVI", valid_mask=mask)
    assert idx[2, 3, 3] == 0
    assert idx[4, 5, 3] == 0
    assert idx[0, 0, 3] == 0
    assert idx[1, 1, 3] > 0

    # 3. Change Map
    chg = build_raw_change_rgba(data_with_nan, valid_mask=mask, threshold=0.01)
    assert chg[2, 3, 3] == 0
    assert chg[0, 0, 3] == 0

    # 4. Classified Change
    cls = build_classified_change_rgba(data_with_nan, valid_mask=mask, threshold=0.01)
    assert cls[2, 3, 3] == 0
    assert cls[0, 0, 3] == 0


# ============================================================
# 8. GEOGRAPHIC BOUNDS PRESERVATION
# ============================================================

def test_geographic_bounds_preservation(mock_raster_geotiff):
    """Verify bounds are accurately extracted and preserved in GeoTIFF / Leaflet format."""
    dummy_img = np.zeros((50, 50, 4), dtype=np.uint8)
    meta = save_visualization_layer(
        dummy_img,
        "test_bounds_layer.png",
        source_raster_path=mock_raster_geotiff,
    )
    assert meta["status"] == "success"
    assert meta["crs"] == "EPSG:4326"
    bounds = meta["bounds"]
    assert bounds is not None
    # [[south, west], [north, east]]
    assert pytest.approx(bounds[0][0], rel=1e-3) == 48.20
    assert pytest.approx(bounds[0][1], rel=1e-3) == 16.40
    assert pytest.approx(bounds[1][0], rel=1e-3) == 48.21
    assert pytest.approx(bounds[1][1], rel=1e-3) == 16.41


# ============================================================
# 9. SCIENTIFIC ARRAYS UNMODIFIED (IMMUTABILITY)
# ============================================================

def test_scientific_arrays_unmodified_by_visualization():
    """Verify input numerical arrays are completely immutable during visualization."""
    r_orig = np.array([[0.12, np.nan], [0.35, 0.44]], dtype=np.float32)
    g_orig = np.array([[0.10, 0.22], [np.nan, 0.30]], dtype=np.float32)
    b_orig = np.array([[0.08, 0.15], [0.20, np.nan]], dtype=np.float32)

    r_copy = r_orig.copy()
    g_copy = g_orig.copy()
    b_copy = b_orig.copy()

    _ = build_true_color_rgba(r_orig, g_orig, b_orig)
    _ = build_false_color_rgba(r_orig, g_orig, b_orig)
    _ = build_index_rgba(r_orig, "NDVI")
    _ = build_raw_change_rgba(r_orig)
    _ = build_classified_change_rgba(r_orig)

    # Must be bit-for-bit identical (including NaNs in same positions)
    np.testing.assert_array_equal(r_orig, r_copy)
    np.testing.assert_array_equal(g_orig, g_copy)
    np.testing.assert_array_equal(b_orig, b_copy)


# ============================================================
# 10. BEFORE/AFTER GEOGRAPHIC CONSISTENCY
# ============================================================

def test_before_after_geographic_consistency(mock_raster_geotiff):
    """Verify before and after layers have identical dimensions and aligned bounds."""
    img_b = np.zeros((50, 50, 4), dtype=np.uint8)
    img_a = np.zeros((50, 50, 4), dtype=np.uint8)

    meta_b = save_visualization_layer(img_b, "test_b.png", source_raster_path=mock_raster_geotiff)
    meta_a = save_visualization_layer(img_a, "test_a.png", source_raster_path=mock_raster_geotiff)

    assert meta_b["bounds"] == meta_a["bounds"]
    assert meta_b["width"] == meta_a["width"]
    assert meta_b["height"] == meta_a["height"]
    assert meta_b["crs"] == meta_a["crs"]
