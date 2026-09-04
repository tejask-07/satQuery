from typing import Any, Dict, Optional

from app.remote_sensing.providers.sentinel2 import (
    Sentinel2Provider,
    search_real_sentinel2,
)


def search_imagery(
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    aoi: Optional[Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Search and download real Sentinel-2 Level-2A satellite imagery.

    Retrieves actual before/after scenes intersecting the requested AOI
    from Microsoft Planetary Computer STAC, crops/warps all bands
    (Red, Green, NIR, SWIR) to an aligned EPSG:4326 grid, and caches them locally.

    Returns standard imagery result structure with source 'REAL_SENTINEL_2'.
    """
    return search_real_sentinel2(
        time_start=time_start,
        time_end=time_end,
        aoi=aoi,
        **kwargs,
    )