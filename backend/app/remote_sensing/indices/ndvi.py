
import numpy as np


def calculate_ndvi(
    red: np.ndarray,
    nir: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate NDVI from Red and NIR spectral bands.

    NDVI = (NIR - Red) / (NIR + Red)

    Parameters
    ----------
    red : np.ndarray
        Red spectral band values.
    nir : np.ndarray
        Near-infrared spectral band values.
    valid_mask : np.ndarray, optional
        Boolean mask where True represents a valid pixel.

    Returns
    -------
    np.ndarray
        NDVI values. Invalid pixels are represented as NaN.
    """

    red = np.asarray(red, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)

    if red.shape != nir.shape:
        raise ValueError("Red and NIR bands must have the same shape.")

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if valid_mask.shape != red.shape:
            raise ValueError("Valid mask must have the same shape as the bands.")
    else:
        valid_mask = np.isfinite(red) & np.isfinite(nir)

    denominator = nir + red

    valid_pixels = valid_mask & (denominator != 0)

    ndvi = np.full(red.shape, np.nan, dtype=np.float32)

    ndvi[valid_pixels] = (
        (nir[valid_pixels] - red[valid_pixels])
        / denominator[valid_pixels]
    )

    return ndvi
