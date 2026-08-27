from pathlib import Path

import cv2

from app.vision.change_detection import detect_change
from app.vision.utils import draw_regions, save_mask


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
OUTPUT_DIR = SAMPLES_DIR / "outputs"


def test_detect_change_from_files():
    before_path = SAMPLES_DIR / "before.png"
    after_path = SAMPLES_DIR / "after.png"

    mask_path = OUTPUT_DIR / "change_mask.png"
    overlay_path = OUTPUT_DIR / "change_overlay.png"

    before = cv2.imread(str(before_path))
    after = cv2.imread(str(after_path))

    assert before is not None, f"Could not read {before_path}"
    assert after is not None, f"Could not read {after_path}"

    result = detect_change(
        before,
        after,
        threshold=0.10,
        min_region_area=20,
    )

    assert result["status"] == "success"
    assert result["changed_pixels"] > 0
    assert result["regions_detected"] >= 1

    save_mask(
        result["change_mask"],
        mask_path,
    )

    assert mask_path.exists()

    overlay = draw_regions(
        after,
        result["regions"],
    )

    success = cv2.imwrite(
        str(overlay_path),
        overlay,
    )

    assert success
    assert overlay_path.exists()