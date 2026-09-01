import numpy as np
import rasterio
from pathlib import Path
from rasterio.crs import CRS
from rasterio.transform import from_bounds

SAMPLES_DIR = Path(__file__).resolve().parent

# High-resolution geographic extent for the Pune/Mumbai test scene (~6km x 6km)
WEST = 73.80
SOUTH = 18.50
EAST = 73.86
NORTH = 18.56

WIDTH = 600
HEIGHT = 600

CRS_WGS84 = CRS.from_epsg(4326)

TRANSFORM = from_bounds(
    WEST,
    SOUTH,
    EAST,
    NORTH,
    WIDTH,
    HEIGHT,
)


def write_raster(path: Path, data: np.ndarray) -> None:
    data = np.asarray(data, dtype=np.float32)
    h, w = data.shape

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": h,
        "width": w,
        "crs": CRS_WGS84,
        "transform": TRANSFORM,
        "compress": "lzw",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def generate_landscape() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Generate realistic multi-spectral surface reflectance bands
    with rich spatial structure (water bodies, vegetation corridors,
    urban fabric, agricultural fields) and genuine temporal changes.
    """
    y_coords, x_coords = np.mgrid[0:HEIGHT, 0:WIDTH]
    y_norm = y_coords / HEIGHT
    x_norm = x_coords / WIDTH

    # Continuous natural spatial frequency variation
    np.random.seed(42)
    noise_base = (
        0.03 * np.sin(x_norm * 18.0) * np.cos(y_norm * 14.0)
        + 0.02 * np.sin(x_norm * 35.0 + y_norm * 25.0)
        + 0.01 * np.random.randn(HEIGHT, WIDTH)
    )

    # 1. Water Body (Meandering river & lake in northwest)
    dist_river = np.abs(y_norm - 0.25 - 0.12 * np.sin(x_norm * 8.0))
    lake_dist = np.sqrt((x_norm - 0.22) ** 2 + (y_norm - 0.22) ** 2)
    is_water_before = (dist_river < 0.035) | (lake_dist < 0.12)
    # Seasonal water change in after image (shoreline recession)
    is_water_after = (dist_river < 0.030) | (lake_dist < 0.09)

    # 2. Dense Forest / Vegetation Corridor (South and West)
    dist_forest = np.sqrt((x_norm - 0.25) ** 2 + (y_norm - 0.75) ** 2)
    is_forest = dist_forest < 0.28
    # Deforestation patch in after image
    deforest_patch = np.sqrt((x_norm - 0.28) ** 2 + (y_norm - 0.78) ** 2) < 0.09

    # 3. Existing Urban Core (Central East)
    dist_urban = np.sqrt((x_norm - 0.65) ** 2 + (y_norm - 0.45) ** 2)
    is_urban_core = dist_urban < 0.18

    # 4. Urban Expansion Zone (East and North-East)
    # Was agricultural/open land before, converted into built-up after
    urban_expand_zone = (
        (x_norm >= 0.60)
        & (x_norm <= 0.88)
        & (y_norm >= 0.15)
        & (y_norm <= 0.40)
    )

    # ----------------------------------------------------
    # Baseline Reflectance - BEFORE (2021)
    # ----------------------------------------------------
    # Default: Agricultural / Mixed vegetation
    red_b = 0.07 + 0.02 * y_norm + noise_base
    green_b = 0.09 + 0.01 * x_norm + noise_base
    nir_b = 0.38 - 0.05 * y_norm + noise_base
    swir_b = 0.16 + 0.02 * x_norm + noise_base

    # Apply Forest (High NIR, low Red, low SWIR)
    red_b[is_forest] = 0.04 + 0.01 * noise_base[is_forest]
    green_b[is_forest] = 0.07 + 0.01 * noise_base[is_forest]
    nir_b[is_forest] = 0.52 + 0.02 * noise_base[is_forest]
    swir_b[is_forest] = 0.10 + 0.01 * noise_base[is_forest]

    # Apply Urban Core (Moderate NIR, high SWIR, high Red)
    red_b[is_urban_core] = 0.19 + 0.02 * noise_base[is_urban_core]
    green_b[is_urban_core] = 0.15 + 0.02 * noise_base[is_urban_core]
    nir_b[is_urban_core] = 0.20 + 0.02 * noise_base[is_urban_core]
    swir_b[is_urban_core] = 0.35 + 0.03 * noise_base[is_urban_core]

    # Apply Water (Low Red, moderate Green, very low NIR/SWIR)
    red_b[is_water_before] = 0.025 + 0.005 * noise_base[is_water_before]
    green_b[is_water_before] = 0.080 + 0.010 * noise_base[is_water_before]
    nir_b[is_water_before] = 0.015 + 0.003 * noise_base[is_water_before]
    swir_b[is_water_before] = 0.008 + 0.002 * noise_base[is_water_before]

    # ----------------------------------------------------
    # Baseline Reflectance - AFTER (2025)
    # ----------------------------------------------------
    red_a = red_b.copy()
    green_a = green_b.copy()
    nir_a = nir_b.copy()
    swir_a = swir_b.copy()

    # Apply Urban Expansion (Agricultural -> Built-up)
    red_a[urban_expand_zone] = 0.21 + 0.02 * noise_base[urban_expand_zone]
    green_a[urban_expand_zone] = 0.16 + 0.02 * noise_base[urban_expand_zone]
    nir_a[urban_expand_zone] = 0.18 + 0.02 * noise_base[urban_expand_zone]
    swir_a[urban_expand_zone] = 0.38 + 0.03 * noise_base[urban_expand_zone]

    # Apply Deforestation (Forest -> Cleared/Exposed Soil)
    red_a[deforest_patch] = 0.16 + 0.02 * noise_base[deforest_patch]
    green_a[deforest_patch] = 0.12 + 0.02 * noise_base[deforest_patch]
    nir_a[deforest_patch] = 0.17 + 0.02 * noise_base[deforest_patch]
    swir_a[deforest_patch] = 0.27 + 0.03 * noise_base[deforest_patch]

    # Apply Water Recession (Water -> Exposed mud/sand)
    water_lost = is_water_before & (~is_water_after)
    red_a[water_lost] = 0.13 + 0.02 * noise_base[water_lost]
    green_a[water_lost] = 0.11 + 0.02 * noise_base[water_lost]
    nir_a[water_lost] = 0.14 + 0.02 * noise_base[water_lost]
    swir_a[water_lost] = 0.18 + 0.02 * noise_base[water_lost]

    # Clip to physical surface reflectance range [0.001, 1.0]
    before_bands = {
        "red": np.clip(red_b, 0.001, 1.0).astype(np.float32),
        "green": np.clip(green_b, 0.001, 1.0).astype(np.float32),
        "nir": np.clip(nir_b, 0.001, 1.0).astype(np.float32),
        "swir": np.clip(swir_b, 0.001, 1.0).astype(np.float32),
    }

    after_bands = {
        "red": np.clip(red_a, 0.001, 1.0).astype(np.float32),
        "green": np.clip(green_a, 0.001, 1.0).astype(np.float32),
        "nir": np.clip(nir_a, 0.001, 1.0).astype(np.float32),
        "swir": np.clip(swir_a, 0.001, 1.0).astype(np.float32),
    }

    return before_bands, after_bands


def main():
    before_bands, after_bands = generate_landscape()

    for band_name, data in before_bands.items():
        write_raster(SAMPLES_DIR / f"before_{band_name}.tif", data)

    for band_name, data in after_bands.items():
        write_raster(SAMPLES_DIR / f"after_{band_name}.tif", data)

    print("Successfully created high-resolution georeferenced sample Sentinel-2 rasters:")
    print(f"Dimensions: {WIDTH} x {HEIGHT} ({WIDTH * HEIGHT} pixels)")
    print(f"CRS: {CRS_WGS84}")
    print(f"Bounds: west={WEST}, south={SOUTH}, east={EAST}, north={NORTH}")


if __name__ == "__main__":
    main()
