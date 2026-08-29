import numpy as np
import rasterio
from pathlib import Path


# create_samples.py is already inside backend/data/samples
SAMPLES_DIR = Path(__file__).resolve().parent


def write_raster(path, data, reference_path):
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    profile.update(
        dtype="float32",
        count=1,
        compress="lzw",
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


# ------------------------------------------------------------
# Sample Green bands
# ------------------------------------------------------------

green_before = np.array(
    [
        [350, 420, 500],
        [380, 450, 530],
        [400, 480, 550],
    ],
    dtype=np.float32,
)

green_after = np.array(
    [
        [300, 360, 430],
        [330, 390, 450],
        [350, 420, 470],
    ],
    dtype=np.float32,
)


# ------------------------------------------------------------
# Sample SWIR bands
# ------------------------------------------------------------

swir_before = np.array(
    [
        [300, 380, 450],
        [330, 420, 500],
        [360, 450, 540],
    ],
    dtype=np.float32,
)

swir_after = np.array(
    [
        [380, 470, 560],
        [420, 520, 620],
        [460, 570, 680],
    ],
    dtype=np.float32,
)


# ------------------------------------------------------------
# Write files
# ------------------------------------------------------------

write_raster(
    SAMPLES_DIR / "before_green.tif",
    green_before,
    SAMPLES_DIR / "before_red.tif",
)

write_raster(
    SAMPLES_DIR / "after_green.tif",
    green_after,
    SAMPLES_DIR / "after_red.tif",
)

write_raster(
    SAMPLES_DIR / "before_swir.tif",
    swir_before,
    SAMPLES_DIR / "before_red.tif",
)

write_raster(
    SAMPLES_DIR / "after_swir.tif",
    swir_after,
    SAMPLES_DIR / "after_red.tif",
)


print("Created Green and SWIR sample rasters.")