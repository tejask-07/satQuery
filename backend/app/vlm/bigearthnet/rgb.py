from pathlib import Path

import numpy as np
from PIL import Image


OUTPUT_DIR = Path("data/bigearthnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def percentile_stretch(
    image: np.ndarray,
    low: float = 2,
    high: float = 98,
) -> np.ndarray:
    """
    Convert uint16 satellite reflectance values into
    an 8-bit displayable image using percentile stretching.
    """

    image = image.astype(np.float32)

    lo = np.percentile(image, low)
    hi = np.percentile(image, high)

    if hi <= lo:
        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    image = (
        (image - lo)
        / (hi - lo)
        * 255.0
    )

    image = np.clip(
        image,
        0,
        255,
    )

    return image.astype(np.uint8)


def create_rgb_image(bands):
    """
    Convert BigEarthNet Sentinel-2 bands to RGB.

    Sentinel-2:
        B04 = Red
        B03 = Green
        B02 = Blue
    """

    red = percentile_stretch(
        bands["B04"]
    )

    green = percentile_stretch(
        bands["B03"]
    )

    blue = percentile_stretch(
        bands["B02"]
    )

    rgb = np.stack(
        [red, green, blue],
        axis=-1,
    )

    return Image.fromarray(
        rgb,
        mode="RGB",
    )


if __name__ == "__main__":

    # Import the remote loader we already built.
    from app.vlm.bigearthnet.remote_tar import (
        load_patch_bands,
    )

    print("=" * 60)
    print("CREATING RGB IMAGE")
    print("=" * 60)

    bands = load_patch_bands()

    rgb = create_rgb_image(
        bands
    )

    output_path = (
        OUTPUT_DIR /
        "sample_rgb.png"
    )

    rgb.save(output_path)

    print("\nRGB image created:")
    print(output_path)

    print(
        "Image size:",
        rgb.size,
    )