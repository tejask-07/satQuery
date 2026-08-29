import numpy as np
import pytest
from app.remote_sensing.io.raster import read_raster, write_raster
from app.agent.executor import execute_plan

from app.remote_sensing.indices.ndvi import calculate_ndvi
from app.remote_sensing.indices.ndwi import calculate_ndwi
from app.remote_sensing.indices.ndbi import calculate_ndbi
from app.remote_sensing.analysis.vegetation import (
    calculate_ndvi_change,
    detect_vegetation_decrease,
    summarize_vegetation_change,
    build_vegetation_analysis,
)


def test_ndvi_basic():
    red = np.array([100, 200, 300], dtype=np.float32)
    nir = np.array([300, 400, 500], dtype=np.float32)

    result = calculate_ndvi(red, nir)

    expected = np.array(
        [0.5, 0.33333334, 0.25],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_ndwi_basic():
    green = np.array([100, 200, 300], dtype=np.float32)
    nir = np.array([50, 200, 600], dtype=np.float32)

    result = calculate_ndwi(green, nir)

    expected = np.array(
        [0.33333334, 0.0, -0.33333334],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_ndbi_basic():
    swir = np.array([300, 400, 600], dtype=np.float32)
    nir = np.array([100, 200, 300], dtype=np.float32)

    result = calculate_ndbi(swir, nir)

    expected = np.array(
        [0.5, 0.33333334, 0.33333334],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_ndvi_change():
    before = np.array(
        [0.8, 0.6, 0.4],
        dtype=np.float32,
    )

    after = np.array(
        [0.5, 0.7, 0.4],
        dtype=np.float32,
    )

    result = calculate_ndvi_change(before, after)

    expected = np.array(
        [-0.3, 0.1, 0.0],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_vegetation_decrease():
    change = np.array(
        [-0.4, -0.2, -0.1, 0.0, 0.3, np.nan],
        dtype=np.float32,
    )

    result = detect_vegetation_decrease(change)

    expected = np.array(
        [True, True, False, False, False, False]
    )

    np.testing.assert_array_equal(result, expected)


def test_vegetation_statistics():
    change = np.array(
        [-0.4, -0.2, -0.1, 0.0, 0.3, np.nan],
        dtype=np.float32,
    )

    mask = detect_vegetation_decrease(change)

    result = summarize_vegetation_change(change, mask)

    assert result["valid_pixel_count"] == 5
    assert result["decreased_pixel_count"] == 2
    assert result["decreased_pixel_percentage"] == 40.0
    assert np.isclose(result["mean_ndvi_change"], -0.08)
    assert np.isclose(result["min_ndvi_change"], -0.4)
    assert np.isclose(result["max_ndvi_change"], 0.3)


def test_vegetation_analysis_pipeline():
    ndvi_before = np.array(
        [
            [0.8, 0.6],
            [0.5, 0.4],
        ],
        dtype=np.float32,
    )

    ndvi_after = np.array(
        [
            [0.5, 0.7],
            [0.2, 0.4],
        ],
        dtype=np.float32,
    )

    result = build_vegetation_analysis(
        ndvi_before,
        ndvi_after,
        threshold=-0.1,
    )

    assert set(result.keys()) == {
        "ndvi_change",
        "decrease_mask",
        "statistics",
    }

    assert result["ndvi_change"].shape == (2, 2)
    assert result["decrease_mask"].shape == (2, 2)

    assert result["statistics"]["valid_pixel_count"] == 4
    assert result["statistics"]["decreased_pixel_count"] == 2

def test_ndvi_rejects_mismatched_shapes():
    red = np.array(
        [100, 200, 300],
        dtype=np.float32,
    )

    nir = np.array(
        [300, 400],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="same shape"):
        calculate_ndvi(red, nir)


def test_ndvi_handles_zero_denominator():
    red = np.array(
        [100, 0],
        dtype=np.float32,
    )

    nir = np.array(
        [100, 0],
        dtype=np.float32,
    )

    result = calculate_ndvi(red, nir)

    assert np.isclose(result[0], 0.0)
    assert np.isnan(result[1])
def test_read_raster():
    data, metadata = read_raster("test_band.tif")

    assert data.shape == (3, 3)
    assert metadata["width"] == 3
    assert metadata["height"] == 3
    assert metadata["resolution"] == (1.0, 1.0)


def test_write_and_read_raster(tmp_path):
    data, metadata = read_raster("test_band.tif")

    output_path = tmp_path / "test_output.tif"

    write_raster(
        str(output_path),
        data,
        metadata,
    )

    output_data, output_metadata = read_raster(
        str(output_path)
    )

    np.testing.assert_array_equal(output_data, data)

    assert output_metadata["width"] == metadata["width"]
    assert output_metadata["height"] == metadata["height"]
    assert output_metadata["resolution"] == metadata["resolution"]
def test_execute_plan_calculate_ndvi():
    red = np.array(
        [
            [100, 200],
            [150, 250],
        ],
        dtype=np.float32,
    )

    nir = np.array(
        [
            [300, 400],
            [350, 450],
        ],
        dtype=np.float32,
    )

    result = execute_plan(
        ["calculate_ndvi"],
        {
            "red": red,
            "nir": nir,
        },
    )

    ndvi_result = result["calculate_ndvi"]

    assert ndvi_result["status"] == "success"
    assert ndvi_result["index"] == "NDVI"

    expected = np.array(
        [
            [0.5, 0.33333334],
            [0.4, 0.2857143],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        ndvi_result["data"],
        expected,
        rtol=1e-5,
    )

    assert np.isclose(
        ndvi_result["mean"],
        0.3797619,
        rtol=1e-5,
    )
    