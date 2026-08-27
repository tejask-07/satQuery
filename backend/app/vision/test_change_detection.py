import cv2
import numpy as np

from app.vision.change_detection import detect_change
from app.vision.utils import draw_regions


def test_detect_change_with_synthetic_images():
    """
    Test the baseline change detector using deterministic synthetic images.

    The before image is a black image.
    The after image contains one white rectangular changed region.
    """

    before = np.zeros((100, 100, 3), dtype=np.uint8)

    after = before.copy()

    # Create a deterministic changed region.
    after[30:60, 40:70] = 255

    result = detect_change(
        before,
        after,
        threshold=0.10,
        min_region_area=20,
    )

    assert result["status"] == "success"

    # The synthetic rectangle is 30 x 30 pixels.
    assert result["changed_pixels"] > 0

    assert result["regions_detected"] == 1

    region = result["regions"][0]

    assert region["area_pixels"] > 0

    bbox = region["bbox"]

    assert bbox["x"] == 40
    assert bbox["y"] == 30
    assert bbox["width"] == 30
    assert bbox["height"] == 30


def test_detect_change_with_identical_images():
    """
    Identical images should produce no detected changes.
    """

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = detect_change(
        image,
        image.copy(),
        threshold=0.10,
        min_region_area=20,
    )

    assert result["status"] == "success"
    assert result["changed_pixels"] == 0
    assert result["regions_detected"] == 0


def test_draw_regions():
    """
    Verify that detected regions can be drawn onto an image.
    """

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    regions = [
        {
            "area_pixels": 900,
            "bbox": {
                "x": 40,
                "y": 30,
                "width": 30,
                "height": 30,
            },
            "centroid": {
                "x": 54.5,
                "y": 44.5,
            },
        }
    ]

    overlay = draw_regions(image, regions)

    assert overlay.shape == image.shape
    assert overlay.dtype == image.dtype

def test_detect_change_with_adaptive_threshold():
    """
    Adaptive thresholding should detect a clear synthetic change.
    """
    before = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    after = before.copy()
    after[30:60, 40:70] = 255

    result = detect_change(
        before,
        after,
        method="adaptive",
        min_region_area=20,
    )

    assert result["status"] == "success"
    assert result["method"] == "adaptive"
    assert result["changed_pixels"] > 0
    assert result["regions_detected"] == 1


def test_invalid_change_detection_method():
    """
    Invalid detection methods should fail clearly.
    """
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    try:
        detect_change(
            image,
            image.copy(),
            method="invalid",
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "fixed" in str(exc)
        assert "adaptive" in str(exc)

def test_small_noise_is_filtered():
    """
    Small isolated changes should be removed while a larger
    meaningful region is preserved.
    """
    before = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    after = before.copy()

    # Meaningful changed region: 30 x 30 pixels.
    after[30:60, 40:70] = 255

    # Tiny isolated noise.
    after[5:7, 5:7] = 255

    result = detect_change(
        before,
        after,
        threshold=0.10,
        min_region_area=20,
    )

    assert result["status"] == "success"

    # The large region should remain.
    assert result["regions_detected"] == 1

    region = result["regions"][0]

    assert region["area_pixels"] >= 800

def test_detect_multiple_change_regions():
    """
    Multiple spatially separated changes should be detected as
    separate regions.
    """
    before = np.zeros(
        (120, 120, 3),
        dtype=np.uint8,
    )

    after = before.copy()

    # Region 1: 20 x 20
    after[10:30, 10:30] = 255

    # Region 2: 25 x 25
    after[60:85, 60:85] = 255

    # Region 3: 15 x 30
    after[90:105, 20:50] = 255

    result = detect_change(
        before,
        after,
        threshold=0.10,
        min_region_area=20,
    )

    assert result["status"] == "success"
    assert result["regions_detected"] == 3

    # Regions should be sorted by area, largest first.
    areas = [
        region["area_pixels"]
        for region in result["regions"]
    ]

    assert areas == sorted(areas, reverse=True)