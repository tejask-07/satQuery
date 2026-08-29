import numpy as np


def calculate_ndbi(
    swir: np.ndarray,
    nir: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate NDBI from SWIR and NIR spectral bands.

    NDBI = (SWIR - NIR) / (SWIR + NIR)

    Parameters
    ----------
    swir : np.ndarray
        Short-wave infrared spectral band values.

    nir : np.ndarray
        Near-infrared spectral band values.

    valid_mask : np.ndarray, optional
        Boolean mask where True represents a valid pixel.

    Returns
    -------
    np.ndarray
        NDBI values. Invalid pixels are represented as NaN.
    """

    swir = np.asarray(swir, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)

    if swir.shape != nir.shape:
        raise ValueError("SWIR and NIR bands must have the same shape.")

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if valid_mask.shape != swir.shape:
            raise ValueError(
                "Valid mask must have the same shape as the bands."
            )
    else:
        valid_mask = np.isfinite(swir) & np.isfinite(nir)

    denominator = swir + nir

    valid_pixels = valid_mask & (denominator != 0)

    ndbi = np.full(
        swir.shape,
        np.nan,
        dtype=np.float32,
    )

    ndbi[valid_pixels] = (
        (swir[valid_pixels] - nir[valid_pixels])
        / denominator[valid_pixels]
    )

    return ndbi