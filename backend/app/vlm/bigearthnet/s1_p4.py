from pathlib import Path

import numpy as np
from PIL import Image

from app.vlm.bigearthnet.remote_s1 import load_s1_bands


S1_NAME = (
    "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"
)


def normalize_percentile(
    image: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:

    image = image.astype(np.float32)

    lo = np.nanpercentile(
        image,
        low,
    )

    hi = np.nanpercentile(
        image,
        high,
    )

    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    if hi <= lo:
        return np.zeros(
            image.shape,
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


def build_s1_visualization(
    s1_name: str = S1_NAME,
) -> Image.Image:
    """
    Load real S1 VV/VH data from the local cache
    and return an RGB composite suitable for P4.

    VV -> red
    VH -> green
    blue -> zero
    """

    bands = load_s1_bands(
        s1_name,
    )

    vv = bands["VV"]
    vh = bands["VH"]

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

    return Image.fromarray(
        composite,
        mode="RGB",
    )


def save_s1_visualization(
    s1_name: str = S1_NAME,
) -> Path:

    output_dir = Path(
        "data/s1_visualizations"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{s1_name}_p4_composite.png"
    )

    image = build_s1_visualization(
        s1_name
    )

    image.save(
        output_path
    )

    return output_path