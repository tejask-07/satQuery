import numpy as np


def calculate_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate NDWI from Green and NIR spectral bands.

    NDWI = (Green - NIR) / (Green + NIR)

    Parameters
    ----------
    green : np.ndarray
        Green spectral band values.

    nir : np.ndarray
        Near-infrared spectral band values.

    valid_mask : np.ndarray, optional
        Boolean mask where True represents a valid pixel.

    Returns
    -------
    np.ndarray
        NDWI values. Invalid pixels are represented as NaN.
    """

    green = np.asarray(green, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)

    if green.shape != nir.shape:
        raise ValueError("Green and NIR bands must have the same shape.")

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if valid_mask.shape != green.shape:
            raise ValueError(
                "Valid mask must have the same shape as the bands."
            )
    else:
        valid_mask = np.isfinite(green) & np.isfinite(nir)

    denominator = green + nir

    valid_pixels = valid_mask & (denominator != 0)

    ndwi = np.full(
        green.shape,
        np.nan,
        dtype=np.float32,
    )

    ndwi[valid_pixels] = (
        (green[valid_pixels] - nir[valid_pixels])
        / denominator[valid_pixels]
    )

    return ndwi