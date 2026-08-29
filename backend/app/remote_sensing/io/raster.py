import numpy as np
import rasterio


def read_raster(path: str) -> tuple[np.ndarray, dict]:
    """
    Read the first band of a raster file along with its metadata.
    """

    with rasterio.open(path) as src:
        data = src.read(1)

        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "resolution": src.res,
            "nodata": src.nodata,
        }

    return data, metadata


def write_raster(
    path: str,
    data: np.ndarray,
    metadata: dict,
) -> None:
    """
    Write a NumPy array to a GeoTIFF using existing raster metadata.
    """

    output_metadata = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype,
        "crs": metadata["crs"],
        "transform": metadata["transform"],
        "nodata": metadata["nodata"],
    }

    with rasterio.open(path, "w", **output_metadata) as dst:
        dst.write(data, 1)