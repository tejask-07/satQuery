"""
Unit and Integration Tests for Optical-SAR GeoTIFF Pair Validator.
"""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

from PIL import Image

from app.remote_sensing.multimodal.optical_sar import (
    validate_optical_sar_pair,
    align_optical_sar_pair,
    build_optical_sar_visuals,
    normalize_band_visual,
    inspect_geotiff,
    check_spatial_overlap,
    check_resolution_compatible,
    check_alignment_required,
)


def _create_test_raster(
    path: Path,
    bounds: tuple[float, float, float, float],
    crs: str = "EPSG:4326",
    width: int = 100,
    height: int = 100,
    count: int = 1,
    dtype: str = "float32",
    nodata: float | None = None,
    descriptions: list[str] | None = None,
) -> Path:
    """Helper to generate a test GeoTIFF with specific metadata."""
    left, bottom, right, top = bounds
    transform = from_bounds(left, bottom, right, top, width, height)
    data = np.ones((count, height, width), dtype=dtype)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
        if descriptions:
            for i, desc in enumerate(descriptions, 1):
                dst.set_band_description(i, desc)

    return path


def test_missing_files():
    """Verify clean errors when one or both files are missing."""
    res = validate_optical_sar_pair("nonexistent_optical.tif", "nonexistent_sar.tif")
    assert res["valid"] is False
    assert res["optical"] is None
    assert res["sar"] is None
    assert len(res["errors"]) >= 2
    assert any("Optical file not found" in e for e in res["errors"])
    assert any("SAR file not found" in e for e in res["errors"])


def test_corrupted_file(tmp_path):
    """Verify error handling for invalid/corrupted raster file."""
    bad_file = tmp_path / "corrupt.tif"
    bad_file.write_text("This is not a valid GeoTIFF file header.")

    valid_file = _create_test_raster(
        tmp_path / "valid.tif",
        bounds=(10.0, 10.0, 10.1, 10.1),
    )

    res = validate_optical_sar_pair(str(bad_file), str(valid_file))
    assert res["valid"] is False
    assert res["optical"] is None
    assert res["sar"] is not None
    assert any("Unreadable Optical raster file" in e for e in res["errors"])


def test_missing_crs(tmp_path):
    """Verify error handling when a TIFF has no CRS."""
    nocrs_file = tmp_path / "nocrs.tif"
    transform = from_bounds(0, 0, 10, 10, 50, 50)
    with rasterio.open(
        nocrs_file,
        "w",
        driver="GTiff",
        height=50,
        width=50,
        count=1,
        dtype="uint8",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((1, 50, 50), dtype="uint8"))

    valid_sar = _create_test_raster(
        tmp_path / "sar.tif",
        bounds=(0.0, 0.0, 1.0, 1.0),
        crs="EPSG:4326",
    )

    res = validate_optical_sar_pair(str(nocrs_file), str(valid_sar))
    assert res["valid"] is False
    assert any("Optical raster missing coordinate reference system" in e for e in res["errors"])


def test_valid_identical_grid_pair(tmp_path):
    """Verify valid overlapping pair with identical CRS and grid."""
    bounds = (73.80, 18.50, 73.86, 18.56)
    opt_path = _create_test_raster(
        tmp_path / "optical.tif",
        bounds=bounds,
        crs="EPSG:4326",
        width=200,
        height=200,
        count=4,
        dtype="uint16",
        nodata=0,
        descriptions=["Red", "Green", "Blue", "NIR"],
    )
    sar_path = _create_test_raster(
        tmp_path / "sar.tif",
        bounds=bounds,
        crs="EPSG:4326",
        width=200,
        height=200,
        count=2,
        dtype="float32",
        nodata=-9999.0,
        descriptions=["VV", "VH"],
    )

    res = validate_optical_sar_pair(str(opt_path), str(sar_path))

    assert res["valid"] is True
    assert len(res["errors"]) == 0

    # Optical metadata checks
    assert res["optical"]["crs"] == "EPSG:4326"
    assert res["optical"]["width"] == 200
    assert res["optical"]["height"] == 200
    assert res["optical"]["band_count"] == 4
    assert res["optical"]["dtype"] == "uint16"
    assert res["optical"]["nodata"] == 0
    assert res["optical"]["band_names"] == ["Red", "Green", "Blue", "NIR"]

    # SAR metadata checks
    assert res["sar"]["crs"] == "EPSG:4326"
    assert res["sar"]["width"] == 200
    assert res["sar"]["height"] == 200
    assert res["sar"]["band_count"] == 2
    assert res["sar"]["dtype"] == "float32"
    assert res["sar"]["nodata"] == -9999.0
    assert res["sar"]["band_names"] == ["VV", "VH"]

    # Compatibility
    assert res["compatibility"]["crs_match"] is True
    assert res["compatibility"]["spatial_overlap"] is True
    assert res["compatibility"]["resolution_compatible"] is True
    assert res["compatibility"]["alignment_required"] is False


def test_different_crs_overlapping_pair(tmp_path):
    """
    Verify pair with DIFFERENT CRS but overlapping geographic bounds.
    Must produce:
      crs_match = False
      spatial_overlap = True
      alignment_required = True
      valid = True
    """
    # Optical in WGS84 over Vienna area
    wgs_bounds = (16.40, 48.20, 16.42, 48.22)
    opt_path = _create_test_raster(
        tmp_path / "optical_wgs.tif",
        bounds=wgs_bounds,
        crs="EPSG:4326",
        width=100,
        height=100,
    )

    # SAR in UTM Zone 33N (EPSG:32633) overlapping Vienna
    utm_bounds = transform_bounds("EPSG:4326", "EPSG:32633", *wgs_bounds)
    sar_path = _create_test_raster(
        tmp_path / "sar_utm.tif",
        bounds=utm_bounds,
        crs="EPSG:32633",
        width=150,
        height=150,
        count=2,
    )

    res = validate_optical_sar_pair(str(opt_path), str(sar_path))

    assert res["valid"] is True
    assert len(res["errors"]) == 0
    assert res["compatibility"]["crs_match"] is False
    assert res["compatibility"]["spatial_overlap"] is True
    assert res["compatibility"]["resolution_compatible"] is True
    assert res["compatibility"]["alignment_required"] is True


def test_no_spatial_overlap(tmp_path):
    """Verify pair with no geographic overlap produces valid=False and spatial_overlap=False."""
    # Optical in Pune
    opt_path = _create_test_raster(
        tmp_path / "optical_pune.tif",
        bounds=(73.80, 18.50, 73.86, 18.56),
        crs="EPSG:4326",
    )
    # SAR in Vienna
    sar_path = _create_test_raster(
        tmp_path / "sar_vienna.tif",
        bounds=(16.40, 48.20, 16.46, 48.26),
        crs="EPSG:4326",
    )

    res = validate_optical_sar_pair(str(opt_path), str(sar_path))

    assert res["valid"] is False
    assert res["compatibility"]["spatial_overlap"] is False
    assert any("No spatial overlap detected" in e for e in res["errors"])


def test_real_repository_data():
    """Verify metadata extraction on actual real sample files existing in the repo."""
    backend_dir = Path(__file__).resolve().parents[3]
    opt_file = backend_dir / "data" / "samples" / "before_red.tif"
    sar_file = backend_dir / "data" / "s1_cache" / "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57" / "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57_VV.tif"

    if not (opt_file.exists() and sar_file.exists()):
        pytest.skip(f"Sample raster files not found at {opt_file} or {sar_file}")

    # Separately inspect each
    opt_info, opt_err = inspect_geotiff(str(opt_file), label="Optical")
    assert opt_err == []
    assert opt_info is not None
    assert opt_info["crs"] == "EPSG:4326"
    assert opt_info["width"] == 600
    assert opt_info["height"] == 600

    sar_info, sar_err = inspect_geotiff(str(sar_file), label="SAR")
    assert sar_err == []
    assert sar_info is not None
    assert "32633" in sar_info["crs"]
    assert sar_info["width"] == 120
    assert sar_info["height"] == 120

    # Cross-validation: Pune vs Austria has no overlap
    res = validate_optical_sar_pair(str(opt_file), str(sar_file))
    assert res["valid"] is False
    assert res["compatibility"]["spatial_overlap"] is False
    assert res["compatibility"]["crs_match"] is False


# ============================================================
# STEP 3: ALIGNMENT & REPROJECTION TESTS
# ============================================================

def test_align_identical_grid(tmp_path):
    """Verify alignment when Optical and SAR already share identical grid."""
    bounds = (73.80, 18.50, 73.86, 18.56)
    opt_path = _create_test_raster(
        tmp_path / "optical.tif",
        bounds=bounds,
        crs="EPSG:4326",
        width=120,
        height=120,
        count=4,
        dtype="uint16",
        descriptions=["Red", "Green", "Blue", "NIR"],
    )
    sar_path = _create_test_raster(
        tmp_path / "sar.tif",
        bounds=bounds,
        crs="EPSG:4326",
        width=120,
        height=120,
        count=2,
        dtype="float32",
        descriptions=["VV", "VH"],
    )

    result = align_optical_sar_pair(str(opt_path), str(sar_path))

    assert result["success"] is True
    assert result["alignment"]["same_grid"] is True
    assert result["alignment"]["reprojected"] is False
    assert result["alignment"]["reference"] == "optical"
    assert result["alignment"]["target_crs"] == "EPSG:4326"
    assert result["alignment"]["target_width"] == 120
    assert result["alignment"]["target_height"] == 120

    # SAR VV and VH checks
    assert result["sar"]["vv"] is not None
    assert result["sar"]["vh"] is not None
    assert result["sar"]["vv"].shape == (120, 120)
    assert result["sar"]["vh"].shape == (120, 120)
    assert result["sar"]["data"].shape == (2, 120, 120)
    assert result["optical"]["data"].shape == (4, 120, 120)

    # Valid mask
    assert result["valid_mask"].shape == (120, 120)
    assert result["valid_mask"].all()


def test_align_different_crs(tmp_path):
    """
    Verify alignment when Optical is EPSG:4326 and SAR is EPSG:32633.
    Optical is reference grid; SAR must reproject to match Optical.
    """
    wgs_bounds = (16.40, 48.20, 16.42, 48.22)
    opt_path = _create_test_raster(
        tmp_path / "optical_wgs.tif",
        bounds=wgs_bounds,
        crs="EPSG:4326",
        width=80,
        height=80,
        count=3,
        descriptions=["Red", "Green", "NIR"],
    )

    utm_bounds = transform_bounds("EPSG:4326", "EPSG:32633", *wgs_bounds)
    sar_path = _create_test_raster(
        tmp_path / "sar_utm.tif",
        bounds=utm_bounds,
        crs="EPSG:32633",
        width=140,
        height=140,
        count=2,
        descriptions=["VV", "VH"],
    )

    result = align_optical_sar_pair(str(opt_path), str(sar_path), resampling="bilinear")

    assert result["success"] is True
    assert result["alignment"]["same_grid"] is True
    assert result["alignment"]["reprojected"] is True
    assert result["alignment"]["resampling"] == "bilinear"

    # Verify SAR matches target optical grid exactly
    opt_meta = result["optical"]["metadata"]
    sar_meta = result["sar"]["metadata"]
    assert sar_meta["crs"] == opt_meta["crs"]
    assert sar_meta["width"] == opt_meta["width"] == 80
    assert sar_meta["height"] == opt_meta["height"] == 80
    assert sar_meta["transform"] == opt_meta["transform"]

    assert result["sar"]["vv"].shape == (80, 80)
    assert result["sar"]["vh"].shape == (80, 80)
    assert result["valid_mask"].shape == (80, 80)
    # Interior pixels should be valid
    assert result["valid_mask"][20:60, 20:60].all()


def test_align_different_resolution(tmp_path):
    """Verify SAR is resampled to the optical grid resolution."""
    bounds = (73.80, 18.50, 73.86, 18.56)
    # Optical high-res 200x200
    opt_path = _create_test_raster(
        tmp_path / "optical_fine.tif",
        bounds=bounds,
        crs="EPSG:4326",
        width=200,
        height=200,
    )
    # SAR coarser 50x50
    sar_path = _create_test_raster(
        tmp_path / "sar_coarse.tif",
        bounds=bounds,
        crs="EPSG:4326",
        width=50,
        height=50,
        count=2,
        descriptions=["VV", "VH"],
    )

    result = align_optical_sar_pair(str(opt_path), str(sar_path))

    assert result["success"] is True
    assert result["alignment"]["reprojected"] is True
    assert result["sar"]["vv"].shape == (200, 200)
    assert result["sar"]["vh"].shape == (200, 200)
    assert result["sar"]["data"].shape == (2, 200, 200)


def test_align_no_spatial_overlap(tmp_path):
    """Verify alignment fails cleanly when there is no geographic overlap."""
    opt_path = _create_test_raster(
        tmp_path / "pune.tif",
        bounds=(73.80, 18.50, 73.86, 18.56),
        crs="EPSG:4326",
    )
    sar_path = _create_test_raster(
        tmp_path / "vienna.tif",
        bounds=(16.40, 48.20, 16.46, 48.26),
        crs="EPSG:4326",
    )

    result = align_optical_sar_pair(str(opt_path), str(sar_path))

    assert result["success"] is False
    assert result["optical"] is None
    assert result["sar"] is None
    assert result["alignment"] is None
    assert len(result["errors"]) > 0
    assert any("No spatial overlap detected" in e for e in result["errors"])


def test_align_nodata_and_valid_mask(tmp_path):
    """Verify joint validity mask excludes nodata/invalid pixels from optical and SAR."""
    bounds = (73.80, 18.50, 73.86, 18.56)

    # 1. Optical with nodata in top-left corner
    opt_data = np.full((1, 100, 100), 500.0, dtype=np.float32)
    opt_data[0, :25, :25] = -9999.0  # optical nodata
    opt_path = tmp_path / "opt_nodata.tif"
    transform = from_bounds(*bounds, 100, 100)
    with rasterio.open(
        opt_path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(opt_data)

    # 2. SAR with nodata in bottom-right corner
    sar_data = np.full((2, 100, 100), 1.5, dtype=np.float32)
    sar_data[:, 75:, 75:] = -9999.0  # sar nodata
    sar_path = tmp_path / "sar_nodata.tif"
    with rasterio.open(
        sar_path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(sar_data)
        dst.set_band_description(1, "VV")
        dst.set_band_description(2, "VH")

    result = align_optical_sar_pair(str(opt_path), str(sar_path))

    assert result["success"] is True
    mask = result["valid_mask"]

    # Optical nodata corner must be False
    assert not mask[:25, :25].any()

    # SAR nodata corner must be False
    assert not mask[75:, 75:].any()

    # Central area with valid data in both must be True
    assert mask[30:70, 30:70].all()


def test_align_real_repository_data_overlap():
    """Verify alignment on real repository rasters rejects non-overlapping pair cleanly."""
    backend_dir = Path(__file__).resolve().parents[3]
    opt_file = backend_dir / "data" / "samples" / "before_red.tif"
    sar_file = backend_dir / "data" / "s1_cache" / "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57" / "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57_VV.tif"

    if not (opt_file.exists() and sar_file.exists()):
        pytest.skip("Real repository rasters not available")

    res = align_optical_sar_pair(str(opt_file), str(sar_file))
    assert res["success"] is False
    assert res["optical"] is None
    assert len(res["errors"]) > 0


# ============================================================
# STEP 4: VISUAL REPRESENTATIONS TESTS
# ============================================================

def test_visuals_optical_true_color_rgb(tmp_path):
    """Verify true-color RGB optical image generation when R, G, B bands exist."""
    bounds = (73.80, 18.50, 73.86, 18.56)
    opt_path = _create_test_raster(
        tmp_path / "opt_true_rgb.tif",
        bounds=bounds,
        width=100,
        height=80,
        count=4,
        descriptions=["Red", "Green", "Blue", "NIR"],
    )
    sar_path = _create_test_raster(
        tmp_path / "sar_true_rgb.tif",
        bounds=bounds,
        width=100,
        height=80,
        count=2,
        descriptions=["VV", "VH"],
    )

    aligned = align_optical_sar_pair(str(opt_path), str(sar_path))
    visuals = build_optical_sar_visuals(aligned)

    opt_vis = visuals["optical"]
    assert isinstance(opt_vis["image"], Image.Image)
    assert opt_vis["image"].mode == "RGB"
    assert opt_vis["image"].size == (100, 80)
    assert opt_vis["mode"] == "rgb"
    assert opt_vis["is_false_color"] is False
    assert "True-color RGB" in opt_vis["description"]
    assert opt_vis["bands_used"] == ["Red", "Green", "Blue"]


def test_visuals_optical_false_color_fallback(tmp_path):
    """Verify false-color NIR fallback when Blue band is missing."""
    bounds = (73.80, 18.50, 73.86, 18.56)
    opt_path = _create_test_raster(
        tmp_path / "opt_nir_fallback.tif",
        bounds=bounds,
        width=90,
        height=70,
        count=3,
        descriptions=["Red", "Green", "NIR"],
    )
    sar_path = _create_test_raster(
        tmp_path / "sar_nir_fallback.tif",
        bounds=bounds,
        width=90,
        height=70,
        count=2,
        descriptions=["VV", "VH"],
    )

    aligned = align_optical_sar_pair(str(opt_path), str(sar_path))
    visuals = build_optical_sar_visuals(aligned)

    opt_vis = visuals["optical"]
    assert isinstance(opt_vis["image"], Image.Image)
    assert opt_vis["image"].mode == "RGB"
    assert opt_vis["image"].size == (90, 70)
    assert opt_vis["is_false_color"] is True
    assert "False-color NIR" in opt_vis["description"]
    assert opt_vis["bands_used"] == ["Red", "Green", "NIR"]


def test_visuals_sar_vv_and_vh(tmp_path):
    """Verify VV and VH grayscale visual representations."""
    bounds = (73.80, 18.50, 73.86, 18.56)
    opt_path = _create_test_raster(
        tmp_path / "opt_sar_bands.tif",
        bounds=bounds,
        width=110,
        height=95,
        count=3,
        descriptions=["Red", "Green", "Blue"],
    )
    sar_path = _create_test_raster(
        tmp_path / "sar_sar_bands.tif",
        bounds=bounds,
        width=110,
        height=95,
        count=2,
        descriptions=["VV", "VH"],
    )

    aligned = align_optical_sar_pair(str(opt_path), str(sar_path))
    visuals = build_optical_sar_visuals(aligned)

    # VV checks
    vv_vis = visuals["s1_vv"]
    assert isinstance(vv_vis["image"], Image.Image)
    assert vv_vis["image"].mode == "L"
    assert vv_vis["image"].size == (110, 95)
    assert vv_vis["mode"] == "L"
    assert "VV polarization" in vv_vis["description"]

    # VH checks
    vh_vis = visuals["s1_vh"]
    assert isinstance(vh_vis["image"], Image.Image)
    assert vh_vis["image"].mode == "L"
    assert vh_vis["image"].size == (110, 95)
    assert vh_vis["mode"] == "L"
    assert "VH cross-polarization" in vh_vis["description"]


def test_visuals_sar_vv_vh_composite(tmp_path):
    """Verify dual-polarization composite with polarization contrast B channel."""
    bounds = (73.80, 18.50, 73.86, 18.56)
    opt_path = _create_test_raster(
        tmp_path / "opt_comp.tif",
        bounds=bounds,
        width=80,
        height=80,
        count=3,
        descriptions=["Red", "Green", "Blue"],
    )
    sar_path = _create_test_raster(
        tmp_path / "sar_comp.tif",
        bounds=bounds,
        width=80,
        height=80,
        count=2,
        descriptions=["VV", "VH"],
    )

    aligned = align_optical_sar_pair(str(opt_path), str(sar_path))
    visuals = build_optical_sar_visuals(aligned)

    comp_vis = visuals["s1_composite"]
    assert isinstance(comp_vis["image"], Image.Image)
    assert comp_vis["image"].mode == "RGB"
    assert comp_vis["image"].size == (80, 80)
    assert comp_vis["mode"] == "rgb"
    assert "dual-polarization radar visualization" in comp_vis["description"]
    assert "polarization contrast" in comp_vis["description"]


def test_visuals_nodata_masking_during_normalization(tmp_path):
    """Verify nodata/invalid pixels do not distort normalization and render as neutral black (0)."""
    bounds = (73.80, 18.50, 73.86, 18.56)

    # Optical with nodata in top-left 30x30
    opt_data = np.full((3, 100, 100), 200.0, dtype=np.float32)
    opt_data[:, :30, :30] = -9999.0
    # Gradient in valid area to test stretching
    for y in range(100):
        opt_data[:, y, 30:] += y * 2.0

    opt_path = tmp_path / "opt_nodata_vis.tif"
    transform = from_bounds(*bounds, 100, 100)
    with rasterio.open(
        opt_path, "w", driver="GTiff", height=100, width=100, count=3,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(opt_data)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")

    # SAR with nodata in bottom-right 30x30
    sar_data = np.full((2, 100, 100), 5.0, dtype=np.float32)
    sar_data[:, 70:, 70:] = -9999.0
    sar_path = tmp_path / "sar_nodata_vis.tif"
    with rasterio.open(
        sar_path, "w", driver="GTiff", height=100, width=100, count=2,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(sar_data)
        dst.set_band_description(1, "VV")
        dst.set_band_description(2, "VH")

    aligned = align_optical_sar_pair(str(opt_path), str(sar_path))
    visuals = build_optical_sar_visuals(aligned)

    opt_arr = np.array(visuals["optical"]["image"])
    vv_arr = np.array(visuals["s1_vv"]["image"])
    vh_arr = np.array(visuals["s1_vh"]["image"])
    comp_arr = np.array(visuals["s1_composite"]["image"])

    # Optical nodata corner (:30, :30) must be masked out to 0 in all visuals
    assert (opt_arr[:30, :30] == 0).all()
    assert (vv_arr[:30, :30] == 0).all()
    assert (vh_arr[:30, :30] == 0).all()
    assert (comp_arr[:30, :30] == 0).all()

    # SAR nodata corner (70:, 70:) must be masked out to 0 in all visuals
    assert (opt_arr[70:, 70:] == 0).all()
    assert (vv_arr[70:, 70:] == 0).all()
    assert (vh_arr[70:, 70:] == 0).all()
    assert (comp_arr[70:, 70:] == 0).all()

    # Center valid area (40:60, 40:60) must have valid non-zero content
    assert np.any(opt_arr[40:60, 40:60] > 0)


def test_visuals_output_dimensions_match_optical_grid(tmp_path):
    """Verify all visual outputs strictly match the optical reference dimensions (125x85)."""
    bounds = (73.80, 18.50, 73.86, 18.56)
    opt_path = _create_test_raster(
        tmp_path / "opt_dim.tif",
        bounds=bounds,
        width=125,
        height=85,
        count=3,
        descriptions=["Red", "Green", "Blue"],
    )
    sar_path = _create_test_raster(
        tmp_path / "sar_dim.tif",
        bounds=bounds,
        width=50,
        height=40,
        count=2,
        descriptions=["VV", "VH"],
    )

    aligned = align_optical_sar_pair(str(opt_path), str(sar_path))
    visuals = build_optical_sar_visuals(aligned)

    assert visuals["optical"]["image"].size == (125, 85)
    assert visuals["s1_vv"]["image"].size == (125, 85)
    assert visuals["s1_vh"]["image"].size == (125, 85)
    assert visuals["s1_composite"]["image"].size == (125, 85)

    meta = visuals["metadata"]
    assert meta["width"] == 125
    assert meta["height"] == 85
    assert meta["reference"] == "optical"


def test_visuals_all_outputs_are_pil_images(tmp_path):
    """Explicitly verify that all 4 visual outputs are valid PIL Image instances."""
    bounds = (73.80, 18.50, 73.86, 18.56)
    opt_path = _create_test_raster(tmp_path / "opt_pil.tif", bounds=bounds, count=3, descriptions=["Red", "Green", "Blue"])
    sar_path = _create_test_raster(tmp_path / "sar_pil.tif", bounds=bounds, count=2, descriptions=["VV", "VH"])

    aligned = align_optical_sar_pair(str(opt_path), str(sar_path))
    visuals = build_optical_sar_visuals(aligned)

    for key in ("optical", "s1_vv", "s1_vh", "s1_composite"):
        img = visuals[key]["image"]
        assert isinstance(img, Image.Image), f"{key} output is not a PIL.Image.Image instance"


def test_visuals_constant_value_does_not_crash():
    """Verify that normalize_band_visual and visuals builder handle constant inputs gracefully."""
    # 1. Direct function check
    const_arr = np.full((60, 60), 42.0, dtype=np.float32)
    norm = normalize_band_visual(const_arr)
    assert norm.shape == (60, 60)
    assert norm.dtype == np.uint8
    assert (norm == 0).all()

    # 2. End-to-end synthetic aligned result with constant values
    synth_aligned = {
        "success": True,
        "optical": {
            "data": np.full((3, 50, 50), 100.0, dtype=np.float32),
            "metadata": {"band_names": ["Red", "Green", "Blue"], "width": 50, "height": 50, "crs": "EPSG:4326"},
        },
        "sar": {
            "vv": np.full((50, 50), 1.0, dtype=np.float32),
            "vh": np.full((50, 50), 0.5, dtype=np.float32),
            "metadata": {"width": 50, "height": 50, "crs": "EPSG:4326"},
        },
        "alignment": {"target_width": 50, "target_height": 50, "target_crs": "EPSG:4326"},
        "valid_mask": np.ones((50, 50), dtype=bool),
    }

    visuals = build_optical_sar_visuals(synth_aligned)
    assert isinstance(visuals["optical"]["image"], Image.Image)
    assert isinstance(visuals["s1_vv"]["image"], Image.Image)
    assert isinstance(visuals["s1_vh"]["image"], Image.Image)
    assert isinstance(visuals["s1_composite"]["image"], Image.Image)


def test_visuals_nan_inf_values_do_not_crash():
    """Verify that NaNs, positive infinity, and negative infinity do not cause crashes."""
    # 1. Direct function check with mixed NaNs and Infs
    mixed_arr = np.array([
        [np.nan, 10.0, np.inf],
        [-np.inf, 20.0, 30.0],
        [np.nan, 40.0, 50.0],
    ], dtype=np.float32)

    norm = normalize_band_visual(mixed_arr)
    assert norm.shape == (3, 3)
    assert norm.dtype == np.uint8
    assert norm[0, 0] == 0  # NaN
    assert norm[0, 2] == 0  # Inf
    assert norm[1, 0] == 0  # -Inf
    assert norm[2, 0] == 0  # NaN
    assert norm[1, 1] > 0   # finite value

    # 2. All NaNs array
    all_nan = np.full((20, 20), np.nan, dtype=np.float32)
    norm_nan = normalize_band_visual(all_nan)
    assert (norm_nan == 0).all()

    # 3. All Infs array
    all_inf = np.full((20, 20), np.inf, dtype=np.float32)
    norm_inf = normalize_band_visual(all_inf)
    assert (norm_inf == 0).all()


def test_visuals_raw_sar_values_unmodified(tmp_path):
    """Verify that building visuals does not alter the underlying SAR data arrays."""
    bounds = (73.80, 18.50, 73.86, 18.56)
    opt_path = _create_test_raster(tmp_path / "opt_unmod.tif", bounds=bounds, count=3, descriptions=["Red", "Green", "Blue"])
    sar_path = _create_test_raster(tmp_path / "sar_unmod.tif", bounds=bounds, count=2, descriptions=["VV", "VH"])

    aligned = align_optical_sar_pair(str(opt_path), str(sar_path))

    vv_copy = np.copy(aligned["sar"]["vv"])
    vh_copy = np.copy(aligned["sar"]["vh"])

    _ = build_optical_sar_visuals(aligned)

    assert np.array_equal(aligned["sar"]["vv"], vv_copy)
    assert np.array_equal(aligned["sar"]["vh"], vh_copy)


def test_visuals_single_band_sar_vv_only(tmp_path):
    """Verify clean behavior when SAR only contains a single VV band."""
    bounds = (73.80, 18.50, 73.86, 18.56)
    opt_path = _create_test_raster(tmp_path / "opt_vv_only.tif", bounds=bounds, count=3, descriptions=["Red", "Green", "Blue"])
    sar_path = _create_test_raster(tmp_path / "sar_vv_only.tif", bounds=bounds, count=1, descriptions=["VV"])

    aligned = align_optical_sar_pair(str(opt_path), str(sar_path))
    visuals = build_optical_sar_visuals(aligned)

    assert visuals["s1_vv"]["image"] is not None
    assert isinstance(visuals["s1_vv"]["image"], Image.Image)
    assert visuals["s1_vh"]["image"] is None
    assert visuals["s1_composite"]["image"] is None


def test_visuals_failed_alignment_raises():
    """Verify build_optical_sar_visuals raises ValueError on unsuccessful alignment dict."""
    with pytest.raises(ValueError, match="Cannot build visuals"):
        build_optical_sar_visuals({"success": False, "errors": ["No overlap"]})

