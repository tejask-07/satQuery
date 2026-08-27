from pathlib import Path

import cv2

from app.vision.change_detection import detect_change as run_change_detection


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def compare_images(**kwargs):
    """
    Compare the selected before/after images.

    This remains a lightweight tool for the execution pipeline.
    The actual change detection is performed by the vision module.
    """
    return {
        "status": "success",
        "comparison": "ready",
    }


def detect_change(**kwargs):
    """
    Run the real classical CV change-detection pipeline.

    For the MVP, imagery returned by the mock imagery tool maps to
    local sample images in data/samples/.
    """

    before_path = SAMPLES_DIR / "before.png"
    after_path = SAMPLES_DIR / "after.png"

    before = cv2.imread(str(before_path))
    after = cv2.imread(str(after_path))

    if before is None:
        raise FileNotFoundError(
            f"Could not load before image: {before_path}"
        )

    if after is None:
        raise FileNotFoundError(
            f"Could not load after image: {after_path}"
        )

    result = run_change_detection(
        before,
        after,
        threshold=0.10,
        min_region_area=20,
    )

    # Do not return the raw NumPy change mask through the API.
    # It is not JSON serializable.
    return {
        "status": result["status"],
        "changed_pixels": result["changed_pixels"],
        "change_ratio": result["change_ratio"],
        "regions_detected": result["regions_detected"],
        "regions": result["regions"],
        "change_type": "candidate_change",
    }
