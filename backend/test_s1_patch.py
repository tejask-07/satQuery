import rasterio
from pathlib import Path
import numpy as np

PATCH_DIR = Path(
    "data/s1_cache/"
    "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"
)


def inspect_band(path: Path):
    with rasterio.open(path) as src:
        image = src.read(1)

        print(f"\n{path.name}")
        print("-" * 60)
        print("Shape:", image.shape)
        print("Dtype:", image.dtype)
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Min:", float(np.nanmin(image)))
        print("Max:", float(np.nanmax(image)))
        print("Mean:", float(np.nanmean(image)))


def main():
    print("=" * 60)
    print("S1 PATCH VERIFICATION")
    print("=" * 60)

    vv = next(PATCH_DIR.glob("*_VV.tif"))
    vh = next(PATCH_DIR.glob("*_VH.tif"))

    inspect_band(vv)
    inspect_band(vh)


if __name__ == "__main__":
    main()