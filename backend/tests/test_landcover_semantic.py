import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.agent.executor import _build_landcover_semantic_result
from app.evaluation.label_ingestion import (
    WC_CROPLAND,
    WC_GRASSLAND,
    WC_SHRUBLAND,
    WC_TREE_COVER,
    WC_WATER,
)


def _grid_path(tmp_path):
    path = tmp_path / "grid.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=1,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(73.0, 19.0, 0.0001, 0.0001),
    ) as dst:
        dst.write(np.zeros((1, 4), dtype=np.float32), 1)
    return str(path)


def _index_result(index, before, after):
    return {
        "index": index,
        f"{index.lower()}_before": np.asarray([before], dtype=np.float32),
        f"{index.lower()}_after": np.asarray([after], dtype=np.float32),
        "valid_mask": np.ones((1, 4), dtype=bool),
    }


def test_worldcover_vegetation_classes_are_defined():
    assert {WC_TREE_COVER, WC_SHRUBLAND, WC_GRASSLAND, WC_CROPLAND} == {10, 20, 30, 40}


def test_worldcover_water_class_is_defined():
    assert WC_WATER == 80


def test_ndvi_directional_and_uncertain_semantics(tmp_path):
    path = _grid_path(tmp_path)
    result = _build_landcover_semantic_result(
        _index_result("NDVI", [.2, .2, .2, .2], [.3, .1, .3, .1]),
        np.array([[1, 0, 1, 0]], dtype=np.uint8),
        "vegetation",
        {"id": "before", "date": "2021"},
        {"id": "after", "date": "2024"},
        path,
        path,
    )
    assert result["directional_expansion_pixels"] == 2
    assert result["directional_reduction_pixels"] == 2
    assert result["confirmed_expansion_pixels"] == 2
    assert result["confirmed_reduction_pixels"] == 0
    assert result["uncertain_pixels"] == 2


def test_ndwi_directional_semantics(tmp_path):
    path = _grid_path(tmp_path)
    result = _build_landcover_semantic_result(
        _index_result("NDWI", [.0, .0, .0, .0], [.1, -.1, .1, -.1]),
        np.array([[1, 0, 1, 0]], dtype=np.uint8),
        "water",
        {"id": "before", "date": "2021"},
        {"id": "after", "date": "2024"},
        path,
        path,
    )
    assert result["confirmed_expansion_pixels"] == 2
    assert result["confirmed_reduction_pixels"] == 0
    assert result["uncertain_pixels"] == 2
    assert result["same_grid"] is True


def test_semantic_provenance_is_reproducible(tmp_path):
    path = _grid_path(tmp_path)
    result = _build_landcover_semantic_result(
        _index_result("NDVI", [.2, .2, .2, .2], [.2, .2, .2, .2]),
        np.ones((1, 4), dtype=np.uint8),
        "vegetation",
        {"id": "scene-before", "date": "2021-01-01"},
        {"id": "scene-after", "date": "2024-01-01"},
        path,
        path,
    )
    assert result["semantic_valid_pixels"] == 4
    assert result["candidate_pixels"] == 0
    assert result["before_scene_id"] == "scene-before"
    assert result["after_scene_id"] == "scene-after"
    assert result["same_grid"] is True
