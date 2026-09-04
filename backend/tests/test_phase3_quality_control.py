import numpy as np
import pytest

from app.remote_sensing.preprocessing.quality import (
    SCLClass,
    compute_quality_masks,
    compute_quality_metrics,
)
from app.remote_sensing.preprocessing.masks import (
    compute_joint_valid_mask,
    apply_mask,
)
from app.tools.indices import (
    calculate_ndvi,
    calculate_temporal_ndvi,
    calculate_temporal_ndwi,
    calculate_temporal_ndbi,
)
from app.tools.change import detect_change


# ============================================================
# TEST 1: NODATA MASKING
# ============================================================

def test_nodata_masking():
    """Verify that SCLClass.NO_DATA and SATURATED_OR_DEFECTIVE are identified as nodata."""
    scl = np.array([
        [SCLClass.NO_DATA, SCLClass.VEGETATION],
        [SCLClass.SATURATED_OR_DEFECTIVE, SCLClass.NOT_VEGETATED],
    ], dtype=np.int32)

    masks = compute_quality_masks(scl_raster=scl)

    assert masks["nodata_mask"][0, 0] is np.True_
    assert masks["nodata_mask"][1, 0] is np.True_
    assert masks["nodata_mask"][0, 1] is np.False_
    assert masks["nodata_mask"][1, 1] is np.False_
    assert masks["valid_mask"][0, 0] is np.False_
    assert masks["valid_mask"][1, 0] is np.False_
    assert masks["valid_mask"][0, 1] is np.True_
    assert masks["valid_mask"][1, 1] is np.True_


# ============================================================
# TEST 2: NAN AND NON-FINITE MASKING
# ============================================================

def test_nan_masking():
    """Verify that NaN and inf values in band data are masked out."""
    band = np.array([
        [0.25, np.nan],
        [np.inf, 0.45],
    ], dtype=np.float32)

    masks = compute_quality_masks(band_data=band)

    assert masks["reflectance_invalid_mask"][0, 1] is np.True_
    assert masks["reflectance_invalid_mask"][1, 0] is np.True_
    assert masks["valid_mask"][0, 1] is np.False_
    assert masks["valid_mask"][1, 0] is np.False_
    assert masks["valid_mask"][0, 0] is np.True_
    assert masks["valid_mask"][1, 1] is np.True_


# ============================================================
# TEST 3: CLOUD MASKING (SCL 8 & 9)
# ============================================================

def test_cloud_masking():
    """Verify that medium and high probability clouds are correctly masked."""
    scl = np.array([
        [SCLClass.CLOUD_MEDIUM_PROBABILITY, SCLClass.VEGETATION],
        [SCLClass.CLOUD_HIGH_PROBABILITY, SCLClass.WATER],
    ], dtype=np.int32)

    masks = compute_quality_masks(scl_raster=scl)

    assert masks["cloud_mask"][0, 0] is np.True_
    assert masks["cloud_mask"][1, 0] is np.True_
    assert masks["cloud_mask"][0, 1] is np.False_
    assert masks["cloud_mask"][1, 1] is np.False_
    assert masks["valid_mask"][0, 0] is np.False_
    assert masks["valid_mask"][1, 0] is np.False_


# ============================================================
# TEST 4: CIRRUS MASKING (SCL 10)
# ============================================================

def test_cirrus_masking():
    """Verify that thin cirrus clouds are specifically identified."""
    scl = np.array([
        [SCLClass.THIN_CIRRUS, SCLClass.VEGETATION],
        [SCLClass.WATER, SCLClass.NOT_VEGETATED],
    ], dtype=np.int32)

    masks = compute_quality_masks(scl_raster=scl)

    assert masks["cirrus_mask"][0, 0] is np.True_
    assert masks["cirrus_mask"][0, 1] is np.False_
    assert masks["valid_mask"][0, 0] is np.False_
    assert masks["valid_mask"][0, 1] is np.True_


# ============================================================
# TEST 5: SHADOW MASKING (SCL 2 & 3)
# ============================================================

def test_shadow_masking():
    """Verify that cloud shadows and topographic shadows are masked."""
    scl = np.array([
        [SCLClass.CLOUD_SHADOWS, SCLClass.VEGETATION],
        [SCLClass.DARK_AREA_PIXELS, SCLClass.NOT_VEGETATED],
    ], dtype=np.int32)

    masks = compute_quality_masks(scl_raster=scl)

    assert masks["shadow_mask"][0, 0] is np.True_
    assert masks["shadow_mask"][1, 0] is np.True_
    assert masks["valid_mask"][0, 0] is np.False_
    assert masks["valid_mask"][1, 0] is np.False_


# ============================================================
# TEST 6: COMBINED VALIDITY MASK
# ============================================================

def test_combined_validity_mask():
    """Verify that valid_mask requires absence of all invalid categories."""
    scl = np.array([
        [SCLClass.NO_DATA, SCLClass.CLOUD_HIGH_PROBABILITY],
        [SCLClass.CLOUD_SHADOWS, SCLClass.VEGETATION],
    ], dtype=np.int32)
    band = np.array([
        [0.2, 0.2],
        [0.2, -0.05],  # last pixel is negative reflectance
    ], dtype=np.float32)

    masks = compute_quality_masks(scl_raster=scl, band_data=band)

    # Pixel (0,0): nodata -> invalid
    assert masks["valid_mask"][0, 0] is np.False_
    # Pixel (0,1): cloud -> invalid
    assert masks["valid_mask"][0, 1] is np.False_
    # Pixel (1,0): shadow -> invalid
    assert masks["valid_mask"][1, 0] is np.False_
    # Pixel (1,1): negative reflectance -> invalid
    assert masks["valid_mask"][1, 1] is np.False_
    # No pixels are valid
    assert int(np.sum(masks["valid_mask"])) == 0


# ============================================================
# TEST 7: REFLECTANCE RANGE VALIDITY
# ============================================================

def test_reflectance_range_validity():
    """Verify that reflectances outside [0.0, 1.5] are marked invalid."""
    band = np.array([
        [-0.01, 0.0],
        [0.5, 1.5],
        [1.51, 3.0],
    ], dtype=np.float32)

    masks = compute_quality_masks(band_data=band, reflectance_min=0.0, reflectance_max=1.5)

    assert masks["valid_mask"][0, 0] is np.False_  # negative
    assert masks["valid_mask"][0, 1] is np.True_   # 0.0 is valid boundary
    assert masks["valid_mask"][1, 0] is np.True_   # 0.5 is valid
    assert masks["valid_mask"][1, 1] is np.True_   # 1.5 is valid boundary
    assert masks["valid_mask"][2, 0] is np.False_  # 1.51 exceeds max
    assert masks["valid_mask"][2, 1] is np.False_  # 3.0 unphysical


# ============================================================
# TEST 8: BEFORE/AFTER JOINT TEMPORAL MASK
# ============================================================

def test_joint_temporal_mask():
    """Verify that joint_valid_mask = before_mask AND after_mask."""
    mask_before = np.array([
        [True, True],
        [False, False],
    ], dtype=bool)
    mask_after = np.array([
        [True, False],
        [True, False],
    ], dtype=bool)

    joint = compute_joint_valid_mask(mask_before, mask_after)

    expected = np.array([
        [True, False],
        [False, False],
    ], dtype=bool)

    np.testing.assert_array_equal(joint, expected)


# ============================================================
# TEST 9: STATISTICS IGNORE INVALID PIXELS
# ============================================================

def test_statistics_ignore_masked_pixels():
    """Verify that indices and change detection strictly exclude invalid pixels from statistics."""
    # 2x2 raster where pixel (0, 0) is valid and others are cloudy
    red_b = np.array([[0.1, 0.9], [0.9, 0.9]], dtype=np.float32)
    nir_b = np.array([[0.5, 0.9], [0.9, 0.9]], dtype=np.float32)
    red_a = np.array([[0.2, 0.9], [0.9, 0.9]], dtype=np.float32)
    nir_a = np.array([[0.6, 0.9], [0.9, 0.9]], dtype=np.float32)

    mask = np.array([
        [True, False],
        [False, False],
    ], dtype=bool)

    result = calculate_temporal_ndvi(
        red_before=red_b,
        nir_before=nir_b,
        red_after=red_a,
        nir_after=nir_a,
        mask_before=mask,
        mask_after=mask,
    )

    # Expected NDVI at (0, 0):
    # before: (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6667
    # after:  (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5000
    # change: 0.5000 - 0.6667 = -0.1667
    assert result["valid_pixels"] == 1
    assert result["total_pixels"] == 4
    assert pytest.approx(result["mean_ndvi_before"], rel=1e-3) == 0.6667
    assert pytest.approx(result["mean_ndvi_after"], rel=1e-3) == 0.5000
    assert pytest.approx(result["mean_ndvi_change"], rel=1e-3) == -0.1667

    # In detect_change:
    change_res = detect_change(
        before=result["ndvi_before"],
        after=result["ndvi_after"],
        threshold=0.05,
        valid_mask=result["valid_mask"],
    )
    assert change_res["valid_pixels"] == 1
    assert change_res["changed_pixels"] == 1  # abs(-0.1667) >= 0.05


# ============================================================
# TEST 10: SCENE WITH ALL VALID PIXELS
# ============================================================

def test_scene_all_valid_pixels():
    """Verify that an unobstructed 100% valid scene reports 100% coverage."""
    scl = np.full((10, 10), SCLClass.VEGETATION, dtype=np.int32)
    masks = compute_quality_masks(scl_raster=scl)
    metrics = compute_quality_metrics(
        valid_mask=masks["valid_mask"],
        cloud_mask=masks["cloud_mask"],
        cirrus_mask=masks["cirrus_mask"],
        shadow_mask=masks["shadow_mask"],
    )

    assert metrics["total_pixels"] == 100
    assert metrics["valid_pixels"] == 100
    assert metrics["invalid_pixels"] == 0
    assert metrics["valid_coverage_percentage"] == 100.0
    assert metrics["cloud_pixels"] == 0
    assert metrics["cloud_percentage_inside_aoi"] == 0.0
    assert metrics["shadow_pixels"] == 0
    assert metrics["shadow_percentage_inside_aoi"] == 0.0


# ============================================================
# TEST 11: SCENE WITH ALL INVALID PIXELS
# ============================================================

def test_scene_all_invalid_pixels():
    """Verify that a 100% cloud/nodata scene gracefully returns None for stats without crashing."""
    scl = np.full((10, 10), SCLClass.CLOUD_HIGH_PROBABILITY, dtype=np.int32)
    masks = compute_quality_masks(scl_raster=scl)
    metrics = compute_quality_metrics(
        valid_mask=masks["valid_mask"],
        cloud_mask=masks["cloud_mask"],
        cirrus_mask=masks["cirrus_mask"],
        shadow_mask=masks["shadow_mask"],
    )

    assert metrics["total_pixels"] == 100
    assert metrics["valid_pixels"] == 0
    assert metrics["invalid_pixels"] == 100
    assert metrics["valid_coverage_percentage"] == 0.0
    assert metrics["cloud_pixels"] == 100
    assert metrics["cloud_percentage_inside_aoi"] == 100.0

    # Index calculation with all invalid mask
    red = np.full((10, 10), 0.2, dtype=np.float32)
    nir = np.full((10, 10), 0.5, dtype=np.float32)
    idx_res = calculate_ndvi(red=red, nir=nir, valid_mask=masks["valid_mask"])

    assert idx_res["valid_pixels"] == 0
    assert idx_res["mean"] is None

    # Temporal index calculation with all invalid mask
    t_res = calculate_temporal_ndvi(
        red_before=red,
        nir_before=nir,
        red_after=red,
        nir_after=nir,
        mask_before=masks["valid_mask"],
        mask_after=masks["valid_mask"],
    )
    assert t_res["valid_pixels"] == 0
    assert t_res["mean_ndvi_before"] is None
    assert t_res["mean_ndvi_after"] is None
    assert t_res["mean_ndvi_change"] is None


# ============================================================
# TEST 12: PARTIAL CLOUD AND SHADOW COVERAGE METRICS
# ============================================================

def test_partial_cloud_coverage_metrics():
    """Verify precision of quality metrics with mixed classes."""
    # Grid of 100 pixels:
    # 60 vegetation, 20 high cloud, 10 thin cirrus, 10 shadow
    grid = [SCLClass.VEGETATION] * 60 + [SCLClass.CLOUD_HIGH_PROBABILITY] * 20 + [SCLClass.THIN_CIRRUS] * 10 + [SCLClass.CLOUD_SHADOWS] * 10
    scl = np.array(grid, dtype=np.int32).reshape((10, 10))

    masks = compute_quality_masks(scl_raster=scl)
    metrics = compute_quality_metrics(
        valid_mask=masks["valid_mask"],
        cloud_mask=masks["cloud_mask"],
        cirrus_mask=masks["cirrus_mask"],
        shadow_mask=masks["shadow_mask"],
    )

    assert metrics["total_pixels"] == 100
    assert metrics["valid_pixels"] == 60
    assert metrics["invalid_pixels"] == 40
    assert metrics["valid_coverage_percentage"] == 60.0
    assert metrics["cloud_pixels"] == 20
    assert metrics["cirrus_pixels"] == 10
    assert metrics["shadow_pixels"] == 10
    # cloud_percentage_inside_aoi includes high/medium cloud + cirrus = 30%
    assert metrics["cloud_percentage_inside_aoi"] == 30.0
    assert metrics["shadow_percentage_inside_aoi"] == 10.0
