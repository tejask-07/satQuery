from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
import requests

from app import config  # Ensures .env variables are loaded


# ============================================================
# CONSTANTS & DEFAULTS
# ============================================================

DEFAULT_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEFAULT_SAS_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"

# Default AOI for Pune / Mumbai region (~6km x 6km) if AOI is omitted
DEFAULT_BBOX = (73.80, 18.50, 73.86, 18.56)

# Sentinel-2 Level-2A Band Mappings
S2_BAND_KEYS = {
    "red": "B04",    # 10m
    "green": "B03",  # 10m
    "nir": "B08",    # 10m
    "swir": "B11",   # 20m (resampled to 10m grid)
}

BACKEND_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def get_cache_dir() -> Path:
    """
    Return the resolved local cache directory for GeoTIFF downloads.
    """
    env_cache = os.getenv("SATQUERY_CACHE_DIR", "backend/data/cache").strip()

    cache_path = Path(env_cache)
    if cache_path.is_absolute():
        resolved = cache_path
    elif cache_path.parts and cache_path.parts[0] == "backend":
        resolved = PROJECT_ROOT / cache_path
    else:
        resolved = BACKEND_DIR / cache_path

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved.resolve()


# ============================================================
# AOI & DATE NORMALIZATION
# ============================================================

def normalize_aoi(aoi: Any) -> Tuple[float, float, float, float]:
    """
    Normalize various AOI representations into a bounding box:
        (min_lng, min_lat, max_lng, max_lat)
    """
    if aoi is None:
        return DEFAULT_BBOX

    # 1. Direct bounding box tuple/list: [min_lng, min_lat, max_lng, max_lat]
    if isinstance(aoi, (list, tuple)):
        if len(aoi) == 4 and all(isinstance(v, (int, float)) for v in aoi):
            return (float(aoi[0]), float(aoi[1]), float(aoi[2]), float(aoi[3]))

        # Leaflet polygon list: [[lat, lng], [lat, lng], ...]
        if len(aoi) >= 3 and isinstance(aoi[0], (list, tuple)) and len(aoi[0]) >= 2:
            if isinstance(aoi[0][0], (int, float)):
                lats = [float(p[0]) for p in aoi]
                lngs = [float(p[1]) for p in aoi]
                return (min(lngs), min(lats), max(lngs), max(lats))
            # GeoJSON coordinates list: [[[lng, lat], ...]]
            if isinstance(aoi[0][0], (list, tuple)) and len(aoi[0][0]) >= 2:
                ring = aoi[0]
                lngs = [float(p[0]) for p in ring]
                lats = [float(p[1]) for p in ring]
                return (min(lngs), min(lats), max(lngs), max(lats))

    # 2. GeoJSON Feature or Geometry dict
    if isinstance(aoi, dict):
        if "type" in aoi:
            geom = aoi.get("geometry", aoi)
            coords = geom.get("coordinates", [])
            if coords and isinstance(coords, list):
                # Polygon coordinates: [[[lng, lat], ...]]
                flat_points = []

                def extract_points(lst: list):
                    if not lst:
                        return
                    if isinstance(lst[0], (int, float)) and len(lst) >= 2:
                        flat_points.append((float(lst[0]), float(lst[1])))
                    elif isinstance(lst[0], list):
                        for sub in lst:
                            extract_points(sub)

                extract_points(coords)
                if flat_points:
                    lngs = [p[0] for p in flat_points]
                    lats = [p[1] for p in flat_points]
                    return (min(lngs), min(lats), max(lngs), max(lats))

        # Dict with named coordinates
        if "coordinates" in aoi:
            return normalize_aoi(aoi["coordinates"])

        if all(k in aoi for k in ("west", "south", "east", "north")):
            return (
                float(aoi["west"]),
                float(aoi["south"]),
                float(aoi["east"]),
                float(aoi["north"]),
            )

        if all(k in aoi for k in ("min_lng", "min_lat", "max_lng", "max_lat")):
            return (
                float(aoi["min_lng"]),
                float(aoi["min_lat"]),
                float(aoi["max_lng"]),
                float(aoi["max_lat"]),
            )

    return DEFAULT_BBOX


def build_datetime_range(time_value: Optional[str], default_year: str) -> str:
    """
    Convert time input (e.g. '2021', '2021-04-17', '2021-01-01/2021-12-31')
    into a valid ISO-8601 STAC datetime range string.
    """
    if time_value is None:
        return f"{default_year}-01-01T00:00:00Z/{default_year}-12-31T23:59:59Z"

    val = str(time_value).strip()

    # Four-digit year
    if len(val) == 4 and val.isdigit():
        return f"{val}-01-01T00:00:00Z/{val}-12-31T23:59:59Z"

    # Full range already
    if "/" in val:
        return val

    # Single ISO date (e.g. 2021-04-17)
    if len(val) >= 10 and val[4] == "-" and val[7] == "-":
        year = val[:4]
        return f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"

    return f"{default_year}-01-01T00:00:00Z/{default_year}-12-31T23:59:59Z"


# ============================================================
# SENTINEL-2 PROVIDER
# ============================================================

class Sentinel2Provider:
    """
    Real Sentinel-2 Level-2A Surface Reflectance Provider.

    Uses Microsoft Planetary Computer STAC API with on-the-fly SAS URL signing,
    windowed reading/warping, and local caching.
    """

    def __init__(
        self,
        stac_url: Optional[str] = None,
        sign_url: Optional[str] = None,
    ):
        self.stac_url = stac_url or os.getenv("SENTINEL_STAC_URL", DEFAULT_STAC_URL)
        self.sign_url = sign_url or os.getenv("SENTINEL_SAS_SIGN_URL", DEFAULT_SAS_SIGN_URL)
        self.cache_dir = get_cache_dir()

    def search_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        datetime_range: str,
        cloud_cover_limit: float = 20.0,
    ) -> List[Dict[str, Any]]:
        """
        Search STAC for Sentinel-2 Level-2A scenes intersecting the AOI and datetime range.
        Tries progressive cloud cover thresholds to find the clearest scene.
        """
        search_endpoint = f"{self.stac_url.rstrip('/')}/search"

        # Try low cloud cover first, fallback to wider thresholds if needed
        thresholds = [cloud_cover_limit, 40.0, 80.0, None]
        items: List[Dict[str, Any]] = []
        last_error = None

        for threshold in thresholds:
            payload: Dict[str, Any] = {
                "collections": ["sentinel-2-l2a"],
                "bbox": list(bbox),
                "datetime": datetime_range,
                "limit": 10,
                "sortby": [
                    {
                        "field": "properties.eo:cloud_cover",
                        "direction": "asc",
                    }
                ],
            }

            if threshold is not None:
                payload["query"] = {
                    "eo:cloud_cover": {"lt": threshold}
                }

            try:
                response = requests.post(
                    search_endpoint,
                    json=payload,
                    timeout=20,
                )
                if response.status_code == 200:
                    data = response.json()
                    found = data.get("features", [])
                    if found:
                        items = found
                        break
                else:
                    last_error = f"STAC status {response.status_code}: {response.text}"
            except Exception as exc:
                last_error = str(exc)

        if not items:
            raise RuntimeError(
                f"No Sentinel-2 Level-2A scenes found for AOI {bbox} in date range '{datetime_range}' "
                f"on STAC API ({self.stac_url}). Troubleshooting: {last_error or 'No intersecting scenes with valid cloud cover.'}"
            )

        return items

    def sign_asset_url(self, raw_url: str) -> str:
        """
        Sign a Planetary Computer Azure asset URL using the public SAS signing endpoint.
        """
        try:
            resp = requests.get(
                self.sign_url,
                params={"href": raw_url},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json().get("href", raw_url)
            return raw_url
        except Exception as exc:
            print(f"[SENTINEL2 SIGN WARNING] Could not sign {raw_url}: {exc}")
            return raw_url

    def fetch_and_cache_band(
        self,
        scene_item: Dict[str, Any],
        band_name: str,
        bbox: Tuple[float, float, float, float],
        dst_crs: CRS,
        dst_transform: Any,
        dst_width: int,
        dst_height: int,
    ) -> str:
        """
        Download and crop the requested band for the given scene and AOI window.
        Saves as an aligned GeoTIFF in the local cache.
        """
        scene_id = scene_item["id"]
        band_key = S2_BAND_KEYS.get(band_name)

        if not band_key:
            raise ValueError(f"Unknown Sentinel-2 band: {band_name}")

        assets = scene_item.get("assets", {})
        if band_key not in assets:
            raise RuntimeError(
                f"Scene {scene_id} is missing required band asset '{band_key}' ({band_name}). "
                f"Available assets: {list(assets.keys())}"
            )

        raw_href = assets[band_key]["href"]

        # Check Processing Baseline & BOA_ADD_OFFSET
        # ESA introduced BOA_ADD_OFFSET = -1000 DN (-0.1000 reflectance) in PB 04.00 (Jan 2022 onwards)
        properties = scene_item.get("properties", {})
        baseline_str = str(properties.get("s2:processing_baseline", "03.00"))

        has_baseline_offset = False
        try:
            baseline_val = float(baseline_str)
            if baseline_val >= 4.0:
                has_baseline_offset = True
        except ValueError:
            if baseline_str >= "04.00":
                has_baseline_offset = True

        band_asset = assets.get(band_key, {})
        raster_bands = band_asset.get("raster:bands", [])
        if raster_bands and isinstance(raster_bands, list) and len(raster_bands) > 0:
            offset_val = raster_bands[0].get("offset")
            if offset_val is not None and float(offset_val) < 0:
                has_baseline_offset = True

        # Deterministic cache filename based on scene ID, band, AOI, and baseline
        min_lng, min_lat, max_lng, max_lat = bbox
        aoi_key = f"{min_lng:.5f}_{min_lat:.5f}_{max_lng:.5f}_{max_lat:.5f}_{dst_width}x{dst_height}"
        aoi_hash = hashlib.md5(aoi_key.encode()).hexdigest()[:8]
        pb_tag = "pb4" if has_baseline_offset else "pb3"
        cache_filename = f"s2_{scene_id}_{band_name}_{aoi_hash}_{pb_tag}.tif"
        cache_path = self.cache_dir / cache_filename

        # If cache exists and is valid, return immediately
        if cache_path.exists():
            try:
                with rasterio.open(cache_path) as src:
                    if src.width == dst_width and src.height == dst_height:
                        return str(cache_path.resolve())
            except Exception:
                pass  # Re-download if corrupted

        # Sign the asset URL
        signed_href = self.sign_asset_url(raw_href)

        # Open and crop/warp window directly using WarpedVRT
        try:
            with rasterio.open(signed_href) as src:
                with WarpedVRT(src, crs=dst_crs, resampling=Resampling.bilinear) as vrt:
                    window = from_bounds(min_lng, min_lat, max_lng, max_lat, vrt.transform)
                    data = vrt.read(
                        1,
                        window=window,
                        out_shape=(dst_height, dst_width),
                        resampling=Resampling.bilinear,
                    ).astype(np.float32)

                    # Sentinel-2 Level-2A surface reflectance scaling:
                    # Stored as DN integer where 10000 = 1.0 surface reflectance.
                    # DN = 0 is nodata.
                    # For Processing Baseline >= 04.00, subtract 1000 DN (0.10 reflectance) offset.
                    nodata_mask = (data <= 0) | (~np.isfinite(data))
                    if has_baseline_offset:
                        scaled_data = np.maximum(0.0, (data - 1000.0) / 10000.0)
                    else:
                        scaled_data = data / 10000.0

                    scaled_data[nodata_mask] = np.nan

                    profile = {
                        "driver": "GTiff",
                        "dtype": "float32",
                        "count": 1,
                        "height": dst_height,
                        "width": dst_width,
                        "crs": dst_crs,
                        "transform": dst_transform,
                        "nodata": np.nan,
                        "compress": "lzw",
                    }

                    with rasterio.open(cache_path, "w", **profile) as dst:
                        dst.write(scaled_data, 1)

        except Exception as exc:
            raise RuntimeError(
                f"Failed to retrieve and process Sentinel-2 band '{band_name}' ({band_key}) "
                f"from scene {scene_id} for AOI {bbox}. Error: {exc}"
            ) from exc

        return str(cache_path.resolve())

    def fetch_scene_bands(
        self,
        scene_item: Dict[str, Any],
        bbox: Tuple[float, float, float, float],
    ) -> Dict[str, str]:
        """
        Fetch and align all 4 required bands (red, green, nir, swir) to the exact same grid.
        """
        min_lng, min_lat, max_lng, max_lat = bbox
        dst_crs = CRS.from_epsg(4326)

        # Compute resolution (~10m in degrees ~ 0.0001 deg)
        res_deg = 0.0001
        dst_width = max(10, int(round((max_lng - min_lng) / res_deg)))
        dst_height = max(10, int(round((max_lat - min_lat) / res_deg)))
        dst_transform = rasterio.transform.from_bounds(
            min_lng, min_lat, max_lng, max_lat, dst_width, dst_height
        )

        band_paths: Dict[str, str] = {}

        for band_name in ("red", "green", "nir", "swir"):
            path = self.fetch_and_cache_band(
                scene_item=scene_item,
                band_name=band_name,
                bbox=bbox,
                dst_crs=dst_crs,
                dst_transform=dst_transform,
                dst_width=dst_width,
                dst_height=dst_height,
            )
            band_paths[band_name] = path

        return band_paths

    def search_and_fetch(
        self,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        aoi: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search and fetch real Sentinel-2 satellite imagery for before and after periods.
        """
        bbox = normalize_aoi(aoi)

        # 1. Build date ranges for before and after
        dt_before = build_datetime_range(time_start, "2021")
        dt_after = build_datetime_range(time_end, "2025")

        try:
            # 2. Search STAC for best before & after scenes
            before_scenes = self.search_scenes(bbox=bbox, datetime_range=dt_before)
            after_scenes = self.search_scenes(bbox=bbox, datetime_range=dt_after)

            best_before = before_scenes[0]
            best_after = after_scenes[0]

            before_id = best_before["id"]
            before_date = best_before["properties"].get("datetime", "")[:10]
            before_cloud = float(best_before["properties"].get("eo:cloud_cover", 0.0))

            after_id = best_after["id"]
            after_date = best_after["properties"].get("datetime", "")[:10]
            after_cloud = float(best_after["properties"].get("eo:cloud_cover", 0.0))

            # 3. Download/extract and cache all 4 bands aligned to the exact same grid
            before_bands = self.fetch_scene_bands(scene_item=best_before, bbox=bbox)
            after_bands = self.fetch_scene_bands(scene_item=best_after, bbox=bbox)

            return {
                "status": "success",
                "source": "REAL_SENTINEL_2",
                "query": {
                    "time_start": str(time_start or "2021"),
                    "time_end": str(time_end or "2025"),
                    "aoi": aoi,
                    "bbox": list(bbox),
                },
                "images": [
                    {
                        "id": before_id,
                        "date": before_date,
                        "cloud_cover": before_cloud,
                        "bands": before_bands,
                    },
                    {
                        "id": after_id,
                        "date": after_date,
                        "cloud_cover": after_cloud,
                        "bands": after_bands,
                    },
                ],
            }
        except Exception as exc:
            # Fallback to high-resolution sample Sentinel-2 rasters if available
            sample_dir = BACKEND_DIR / "data" / "samples"
            sample_before = {
                b: str((sample_dir / f"before_{b}.tif").resolve())
                for b in ("red", "green", "nir", "swir")
            }
            sample_after = {
                b: str((sample_dir / f"after_{b}.tif").resolve())
                for b in ("red", "green", "nir", "swir")
            }
            all_samples_exist = all(
                Path(p).exists()
                for p in list(sample_before.values()) + list(sample_after.values())
            )

            if all_samples_exist:
                print(
                    f"[SENTINEL2 WARNING] Planetary Computer STAC search failed ({exc}). "
                    "Using high-resolution sample rasters."
                )
                return {
                    "status": "success",
                    "source": "SAMPLE_SENTINEL_2_FALLBACK",
                    "query": {
                        "time_start": str(time_start or "2021"),
                        "time_end": str(time_end or "2025"),
                        "aoi": aoi,
                        "bbox": list(bbox),
                    },
                    "images": [
                        {
                            "id": "sample_sentinel2_2021",
                            "date": "2021-06-15",
                            "cloud_cover": 0.0,
                            "bands": sample_before,
                        },
                        {
                            "id": "sample_sentinel2_2025",
                            "date": "2025-06-15",
                            "cloud_cover": 0.0,
                            "bands": sample_after,
                        },
                    ],
                }
            raise


# ============================================================
# SEARCH IMAGERY FUNCTION WRAPPER
# ============================================================

def search_real_sentinel2(
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    aoi: Optional[Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Search and fetch real Sentinel-2 satellite imagery for before and after periods.
    """
    provider = Sentinel2Provider()
    return provider.search_and_fetch(
        time_start=time_start,
        time_end=time_end,
        aoi=aoi,
        **kwargs,
    )
