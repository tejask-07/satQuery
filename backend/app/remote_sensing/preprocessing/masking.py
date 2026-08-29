
import numpy as np


def create_valid_mask(data: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask identifying valid pixels.

    A pixel is considered valid when:
    - it is not NaN
    - it is finite

    Parameters
    ----------
    data : np.ndarray
        Raster data array.

    Returns
    -------
    np.ndarray
        Boolean mask where True means valid and False means invalid.
    """

    data = np.asarray(data)

    return np.isfinite(data)

