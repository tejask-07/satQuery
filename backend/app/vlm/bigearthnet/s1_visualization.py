from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


PATCH_DIR = Path(
    "data/s1_cache/"
    "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"
)

OUTPUT_DIR = Path(
    "data/s1_visualizations"
)


def normalize_percentile(
    image: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:

    image = image.astype(np.float32)

    lo = np.percentile(image, low)
    hi = np.percentile(image, high)

    if hi <= lo:
        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    scaled = (
        (image - lo)
        / (hi - lo)
        * 255.0
    )

    return np.clip(
        scaled,
        0,
        255,
    ).astype(np.uint8)


def load_tif(path: Path) -> np.ndarray:

    with rasterio.open(path) as src:
        return src.read(1)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vv_path = next(
        PATCH_DIR.glob("*_VV.tif")
    )

    vh_path = next(
        PATCH_DIR.glob("*_VH.tif")
    )

    vv = load_tif(vv_path)
    vh = load_tif(vh_path)

    print("=" * 60)
    print("S1 VISUALIZATION")
    print("=" * 60)

    print("\nVV:")
    print("Shape:", vv.shape)
    print("Mean:", float(np.mean(vv)))

    print("\nVH:")
    print("Shape:", vh.shape)
    print("Mean:", float(np.mean(vh)))

    # --------------------------------------------------
    # Individual grayscale images
    # --------------------------------------------------

    vv_img = Image.fromarray(
        normalize_percentile(vv),
        mode="L",
    )

    vh_img = Image.fromarray(
        normalize_percentile(vh),
        mode="L",
    )

    vv_output = (
        OUTPUT_DIR
        / "s1_vv.png"
    )

    vh_output = (
        OUTPUT_DIR
        / "s1_vh.png"
    )

    vv_img.save(vv_output)
    vh_img.save(vh_output)

    # --------------------------------------------------
    # Two-channel composite
    #
    # VV → red
    # VH → green
    # Blue → zero
    # --------------------------------------------------

    vv_norm = normalize_percentile(vv)
    vh_norm = normalize_percentile(vh)

    composite = np.stack(
        [
            vv_norm,
            vh_norm,
            np.zeros_like(vv_norm),
        ],
        axis=-1,
    )

    composite_img = Image.fromarray(
        composite,
        mode="RGB",
    )

    composite_output = (
        OUTPUT_DIR
        / "s1_vv_vh_composite.png"
    )

    composite_img.save(
        composite_output
    )

    print("\nSaved:")
    print(vv_output)
    print(vh_output)
    print(composite_output)


if __name__ == "__main__":
    main()