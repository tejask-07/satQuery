from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
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
from app.remote_sensing.preprocessing.quality import (
    compute_quality_masks,
    compute_quality_metrics,
)
from app.remote_sensing.preprocessing.masks import (
    compute_joint_valid_mask,
)
from app.evidence.scientific_visualizations import (
    build_true_color_rgba,
    build_false_color_rgba,
    build_quality_mask_rgba,
    save_visualization_layer,
)




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
    "blue": "B02",   # 10m
    "nir": "B08",    # 10m
    "swir": "B11",   # 20m (resampled to 10m grid)
    "scl": "SCL",    # 20m Scene Classification Layer (nearest neighbor resampled)
}



BACKEND_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[4]


# ============================================================
# SCENE SCORING CONFIGURATION & WEIGHTS
# ============================================================

@dataclass
class SceneScoringWeights:
    """
    Centralized, explainable weights for ranking Sentinel-2 candidate scenes.
    Weights sum to 1.0.
    """
    weight_cloud: float = 0.40       # Prioritize low cloud contamination
    weight_coverage: float = 0.35    # Prioritize full AOI spatial coverage
    weight_seasonal: float = 0.20    # Prioritize comparable seasonal acquisition
    weight_quality: float = 0.05     # Prioritize high data completeness


class Sentinel2ErrorType:
    NO_SCENES_FOUND = "NO_SCENES_FOUND"
    STAC_UNAVAILABLE = "STAC_UNAVAILABLE"
    MALFORMED_AOI = "MALFORMED_AOI"
    INVALID_DATES = "INVALID_DATES"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"
    QUALITY_CRITERIA_EXCEEDED = "QUALITY_CRITERIA_EXCEEDED"


class Sentinel2RetrievalError(RuntimeError):
    """
    Structured actionable error for Sentinel-2 retrieval failures.
    """
    def __init__(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{error_type}] {message}")
        self.error_type = error_type
        self.message = message
        self.details = details or {}


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
# AOI COVERAGE & SEASONAL SCORING HELPERS
# ============================================================

def calculate_aoi_coverage(
    scene_item: Dict[str, Any],
    aoi_bbox: Tuple[float, float, float, float],
) -> Dict[str, float]:
    """
    Calculate spatial intersection area and coverage percentage between scene and requested AOI.
    Returns:
        aoi_area: float
        intersection_area: float
        coverage_percentage: float (0.0 to 100.0)
        coverage_ratio: float (0.0 to 1.0)
    """
    aoi_w, aoi_s, aoi_e, aoi_n = aoi_bbox
    aoi_area = max(0.0, (aoi_e - aoi_w) * (aoi_n - aoi_s))
    if aoi_area <= 0:
        return {
            "aoi_area": 0.0,
            "intersection_area": 0.0,
            "coverage_percentage": 100.0,
            "coverage_ratio": 1.0,
        }

    # Extract scene bounding box from item bbox or geometry
    scene_bbox = scene_item.get("bbox")
    if not scene_bbox and "geometry" in scene_item:
        geom = scene_item["geometry"]
        if geom and "coordinates" in geom:
            pts: List[Tuple[float, float]] = []

            def _extract(c):
                if isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (int, float)):
                    pts.append((float(c[0]), float(c[1])))
                elif isinstance(c, list):
                    for sub in c:
                        _extract(sub)

            _extract(geom["coordinates"])
            if pts:
                scene_bbox = [
                    min(p[0] for p in pts),
                    min(p[1] for p in pts),
                    max(p[0] for p in pts),
                    max(p[1] for p in pts),
                ]

    if not scene_bbox:
        return {
            "aoi_area": round(aoi_area, 6),
            "intersection_area": round(aoi_area, 6),
            "coverage_percentage": 100.0,
            "coverage_ratio": 1.0,
        }

    scene_w, scene_s, scene_e, scene_n = [float(x) for x in scene_bbox]

    inter_w = max(aoi_w, scene_w)
    inter_s = max(aoi_s, scene_s)
    inter_e = min(aoi_e, scene_e)
    inter_n = min(aoi_n, scene_n)

    if inter_e > inter_w and inter_n > inter_s:
        inter_area = (inter_e - inter_w) * (inter_n - inter_s)
    else:
        inter_area = 0.0

    coverage_ratio = min(1.0, max(0.0, inter_area / aoi_area))
    coverage_pct = round(coverage_ratio * 100.0, 2)

    return {
        "aoi_area": round(aoi_area, 6),
        "intersection_area": round(inter_area, 6),
        "coverage_percentage": coverage_pct,
        "coverage_ratio": coverage_ratio,
    }


def calculate_seasonal_similarity(
    candidate_dt: datetime,
    target_dt_or_month: datetime | int | None = None,
) -> float:
    """
    Calculate seasonal similarity score (0.0 to 1.0) based on circular day-of-year distance.
    If target is None, defaults to mid-year peak (July 1, DOY ~182).
    """
    cand_doy = candidate_dt.timetuple().tm_yday

    if isinstance(target_dt_or_month, datetime):
        target_doy = target_dt_or_month.timetuple().tm_yday
    elif isinstance(target_dt_or_month, int):
        target_doy = (target_dt_or_month - 1) * 30 + 15
    else:
        target_doy = 182  # July 1 midpoint

    doy_diff = min(abs(cand_doy - target_doy), 365 - abs(cand_doy - target_doy))
    return max(0.0, 1.0 - (doy_diff / 182.5))


def score_scene(
    scene_item: Dict[str, Any],
    aoi_bbox: Tuple[float, float, float, float],
    target_date: Optional[datetime | str | int] = None,
    weights: Optional[SceneScoringWeights] = None,
) -> Dict[str, Any]:
    """
    Score a candidate Sentinel-2 scene based on:
    1. Cloud contamination
    2. AOI coverage
    3. Seasonal/temporal matching
    4. Observation data completeness
    """
    weights = weights or SceneScoringWeights()
    props = scene_item.get("properties", {})

    # 1. Cloud Score (0.0 - 1.0)
    cloud_cover = float(props.get("eo:cloud_cover", 0.0))
    cloud_cover = min(100.0, max(0.0, cloud_cover))
    cloud_score = 1.0 - (cloud_cover / 100.0)

    # 2. AOI Coverage Score (0.0 - 1.0)
    cov_details = calculate_aoi_coverage(scene_item, aoi_bbox)
    coverage_score = cov_details["coverage_ratio"]

    # 3. Seasonal Similarity Score (0.0 - 1.0)
    dt_str = props.get("datetime") or scene_item.get("datetime", "")
    cand_dt = None
    if dt_str:
        try:
            cand_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            pass
    if cand_dt is None:
        cand_dt = datetime(2021, 6, 15, tzinfo=timezone.utc)

    target_ref = None
    if isinstance(target_date, str):
        try:
            target_ref = datetime.fromisoformat(target_date.replace("Z", "+00:00"))
        except Exception:
            target_ref = None
    elif isinstance(target_date, (datetime, int)):
        target_ref = target_date

    seasonal_score = calculate_seasonal_similarity(cand_dt, target_ref)

    # 4. Data Quality Score (0.0 - 1.0)
    nodata_pct = float(props.get("s2:nodata_pixel_percentage", 0.0))
    degraded_pct = float(props.get("s2:degraded_msi_data_percentage", 0.0))
    quality_score = max(0.0, 1.0 - ((nodata_pct + degraded_pct) / 100.0))

    # Total Score
    total_w = weights.weight_cloud + weights.weight_coverage + weights.weight_seasonal + weights.weight_quality
    total_score = (
        weights.weight_cloud * cloud_score
        + weights.weight_coverage * coverage_score
        + weights.weight_seasonal * seasonal_score
        + weights.weight_quality * quality_score
    ) / total_w
    total_score = round(total_score, 4)

    # Human-readable selection reason
    month_name = cand_dt.strftime("%B")
    reason_parts = [
        f"low cloud cover ({cloud_cover:.1f}%)",
        f"AOI coverage {cov_details['coverage_percentage']:.1f}%",
        f"seasonal alignment in {month_name}",
    ]
    selection_reason = (
        f"Selected with quality score {total_score:.4f} based on "
        + ", ".join(reason_parts)
        + "."
    )

    return {
        "score": total_score,
        "cloud_score": round(cloud_score, 4),
        "coverage_score": round(coverage_score, 4),
        "seasonal_score": round(seasonal_score, 4),
        "quality_score": round(quality_score, 4),
        "coverage_details": cov_details,
        "selection_reason": selection_reason,
        "datetime": dt_str,
        "cloud_cover": cloud_cover,
    }


def rank_candidate_scenes(
    candidates: List[Dict[str, Any]],
    aoi_bbox: Tuple[float, float, float, float],
    target_date: Optional[datetime | str | int] = None,
    weights: Optional[SceneScoringWeights] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Score and rank candidate scenes deterministically by total score descending.
    Returns list of (candidate_item, score_result).
    """
    scored = []
    for c in candidates:
        sc = score_scene(c, aoi_bbox=aoi_bbox, target_date=target_date, weights=weights)
        scored.append((c, sc))

    # Deterministic sort: score desc, cloud_cover asc, id asc
    scored.sort(
        key=lambda pair: (
            -pair[1]["score"],
            pair[1]["cloud_cover"],
            pair[0].get("id", ""),
        )
    )
    return scored


# ============================================================
# SENTINEL-2 PROVIDER
# ============================================================

class Sentinel2Provider:
    """
    Real Sentinel-2 Level-2A Surface Reflectance Provider.

    Uses Microsoft Planetary Computer STAC API with SAS URL signing,
    windowed WarpedVRT reading, explainable scene selection, and local caching.
    Production requests never fall back silently to fake local samples.
    """

    def __init__(
        self,
        stac_url: Optional[str] = None,
        sign_url: Optional[str] = None,
        weights: Optional[SceneScoringWeights] = None,
    ):
        self.stac_url = stac_url or os.getenv("SENTINEL_STAC_URL", DEFAULT_STAC_URL)
        self.sign_url = sign_url or os.getenv("SENTINEL_SAS_SIGN_URL", DEFAULT_SAS_SIGN_URL)
        self.cache_dir = get_cache_dir()
        self.weights = weights or SceneScoringWeights()

    def search_candidate_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        datetime_range: str,
        cloud_cover_limit: float = 20.0,
        max_candidates: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search Planetary Computer STAC for candidate Sentinel-2 Level-2A scenes.
        Collects a pool of candidate scenes across progressive quality thresholds.
        """
        # Validate bbox
        w, s, e, n = bbox
        if w >= e or s >= n or not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90):
            raise Sentinel2RetrievalError(
                Sentinel2ErrorType.MALFORMED_AOI,
                f"Invalid bounding box coordinates: {bbox}. Expected [west, south, east, north] with west < east and south < north.",
                details={"bbox": bbox},
            )

        search_endpoint = f"{self.stac_url.rstrip('/')}/search"
        thresholds = [cloud_cover_limit, 40.0, 75.0, None]
        candidates: List[Dict[str, Any]] = []
        last_error = None

        for threshold in thresholds:
            payload: Dict[str, Any] = {
                "collections": ["sentinel-2-l2a"],
                "bbox": list(bbox),
                "datetime": datetime_range,
                "limit": max_candidates,
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
                    timeout=25,
                )
                if response.status_code == 200:
                    data = response.json()
                    found = data.get("features", [])
                    if found:
                        candidates = found
                        break
                else:
                    last_error = f"STAC API returned HTTP {response.status_code}: {response.text}"
            except Exception as exc:
                last_error = f"STAC network error: {exc}"

        if not candidates:
            # Distinguish STAC network unavailability from no scenes found
            if last_error and "network error" in last_error.lower():
                raise Sentinel2RetrievalError(
                    Sentinel2ErrorType.STAC_UNAVAILABLE,
                    f"Planetary Computer STAC endpoint unavailable ({self.stac_url}): {last_error}",
                    details={"bbox": bbox, "datetime_range": datetime_range},
                )
            raise Sentinel2RetrievalError(
                Sentinel2ErrorType.NO_SCENES_FOUND,
                f"No Sentinel-2 Level-2A scenes found for AOI {bbox} in date range '{datetime_range}'. Details: {last_error or 'No intersecting scenes found.'}",
                details={"bbox": bbox, "datetime_range": datetime_range},
            )

        return candidates

    def search_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        datetime_range: str,
        cloud_cover_limit: float = 20.0,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible scene search returning candidates sorted by cloud cover.
        """
        return self.search_candidate_scenes(
            bbox=bbox,
            datetime_range=datetime_range,
            cloud_cover_limit=cloud_cover_limit,
        )

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

        is_scl = (band_name == "scl")
        resampling_method = Resampling.nearest if is_scl else Resampling.bilinear

        # Open and crop/warp window directly using WarpedVRT
        try:
            with rasterio.open(signed_href) as src:
                with WarpedVRT(src, crs=dst_crs, resampling=resampling_method) as vrt:
                    window = from_bounds(min_lng, min_lat, max_lng, max_lat, vrt.transform)
                    data = vrt.read(
                        1,
                        window=window,
                        out_shape=(dst_height, dst_width),
                        resampling=resampling_method,
                    )

                    if is_scl:
                        scaled_data = data.astype(np.uint8)
                        profile = {
                            "driver": "GTiff",
                            "dtype": "uint8",
                            "count": 1,
                            "height": dst_height,
                            "width": dst_width,
                            "crs": dst_crs,
                            "transform": dst_transform,
                            "nodata": 0,
                            "compress": "lzw",
                        }
                    else:
                        data = data.astype(np.float32)
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
        Fetch and align all required bands (red, green, nir, swir, and optional scl) to the exact same grid.
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

        bands_to_fetch = ["red", "green", "nir", "swir"]
        if "B02" in scene_item.get("assets", {}):
            bands_to_fetch.append("blue")
        if "SCL" in scene_item.get("assets", {}):
            bands_to_fetch.append("scl")

        for band_name in bands_to_fetch:
            try:
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
            except Exception as exc:
                if band_name in ("scl", "blue"):
                    print(f"[SENTINEL2 WARNING] Optional {band_name} asset could not be fetched: {exc}")
                else:
                    raise


        return band_paths

    def search_and_fetch(
        self,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        aoi: Optional[Any] = None,
        preferred_season_month: int = 6,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search, score, rank, and retrieve real Sentinel-2 Level-2A satellite imagery.
        Never falls back to local sample rasters in production.
        """
        bbox = normalize_aoi(aoi)

        dt_before = build_datetime_range(time_start, "2021")
        dt_after = build_datetime_range(time_end, "2025")

        try:
            # 1. Collect candidate scenes from Microsoft Planetary Computer STAC
            before_candidates = self.search_candidate_scenes(
                bbox=bbox,
                datetime_range=dt_before,
            )
            after_candidates = self.search_candidate_scenes(
                bbox=bbox,
                datetime_range=dt_after,
            )

            # 2. Stage 1: Rank before scenes using preferred seasonal anchor (default June)
            ranked_before = rank_candidate_scenes(
                candidates=before_candidates,
                aoi_bbox=bbox,
                target_date=preferred_season_month,
                weights=self.weights,
            )
            if not ranked_before:
                raise Sentinel2RetrievalError(
                    Sentinel2ErrorType.NO_SCENES_FOUND,
                    f"No candidate scenes for 'before' period in range {dt_before}.",
                )

            best_before, best_before_score = ranked_before[0]
            before_dt = best_before_score.get("datetime") or ""

            # 3. Stage 2: Rank after scenes matching the exact seasonal window of best_before
            ranked_after = rank_candidate_scenes(
                candidates=after_candidates,
                aoi_bbox=bbox,
                target_date=before_dt,
                weights=self.weights,
            )
            if not ranked_after:
                raise Sentinel2RetrievalError(
                    Sentinel2ErrorType.NO_SCENES_FOUND,
                    f"No candidate scenes for 'after' period in range {dt_after}.",
                )

            best_after, best_after_score = ranked_after[0]

            # 4. Extract provenance metadata
            before_id = best_before["id"]
            before_date = (best_before["properties"].get("datetime", "") or "")[:10]
            before_cloud = float(best_before["properties"].get("eo:cloud_cover", 0.0))
            before_coverage = best_before_score["coverage_details"]["coverage_percentage"]
            before_score = best_before_score["score"]
            before_reason = best_before_score["selection_reason"]
            before_platform = best_before["properties"].get("platform", "Sentinel-2")
            before_baseline = str(best_before["properties"].get("s2:processing_baseline", "03.00"))
            before_mgrs = best_before["properties"].get("s2:mgrs_tile", "")

            after_id = best_after["id"]
            after_date = (best_after["properties"].get("datetime", "") or "")[:10]
            after_cloud = float(best_after["properties"].get("eo:cloud_cover", 0.0))
            after_coverage = best_after_score["coverage_details"]["coverage_percentage"]
            after_score = best_after_score["score"]
            after_reason = best_after_score["selection_reason"]
            after_platform = best_after["properties"].get("platform", "Sentinel-2")
            after_baseline = str(best_after["properties"].get("s2:processing_baseline", "03.00"))
            after_mgrs = best_after["properties"].get("s2:mgrs_tile", "")

            # 5. Fetch bands for best scenes
            before_bands = self.fetch_scene_bands(scene_item=best_before, bbox=bbox)
            after_bands = self.fetch_scene_bands(scene_item=best_after, bbox=bbox)

            # 6. Quality control & Analysis-Ready masking
            min_lng, min_lat, max_lng, max_lat = bbox
            res_deg = 0.0001
            dst_width = max(10, int(round((max_lng - min_lng) / res_deg)))
            dst_height = max(10, int(round((max_lat - min_lat) / res_deg)))
            dst_transform = rasterio.transform.from_bounds(
                min_lng, min_lat, max_lng, max_lat, dst_width, dst_height
            )
            mask_profile = {
                "driver": "GTiff",
                "dtype": "uint8",
                "count": 1,
                "height": dst_height,
                "width": dst_width,
                "crs": CRS.from_epsg(4326),
                "transform": dst_transform,
                "nodata": 0,
                "compress": "lzw",
            }
            aoi_key = f"{min_lng:.5f}_{min_lat:.5f}_{max_lng:.5f}_{max_lat:.5f}_{dst_width}x{dst_height}"
            aoi_hash = hashlib.md5(aoi_key.encode()).hexdigest()[:8]

            # Before quality
            scl_before = None
            if "scl" in before_bands and Path(before_bands["scl"]).exists():
                try:
                    with rasterio.open(before_bands["scl"]) as src:
                        scl_before = src.read(1)
                except Exception:
                    scl_before = None

            red_before = None
            if "red" in before_bands and Path(before_bands["red"]).exists():
                try:
                    with rasterio.open(before_bands["red"]) as src:
                        red_before = src.read(1)
                except Exception:
                    red_before = None

            masks_before = compute_quality_masks(
                scl_raster=scl_before,
                band_data=red_before,
                shape=(dst_height, dst_width),
            )
            quality_before = compute_quality_metrics(
                valid_mask=masks_before["valid_mask"],
                cloud_mask=masks_before["cloud_mask"] if scl_before is not None else None,
                cirrus_mask=masks_before["cirrus_mask"] if scl_before is not None else None,
                shadow_mask=masks_before["shadow_mask"] if scl_before is not None else None,
            )
            mask_path_before = self.cache_dir / f"s2_{before_id}_mask_{aoi_hash}.tif"
            try:
                with rasterio.open(mask_path_before, "w", **mask_profile) as dst:
                    dst.write(masks_before["valid_mask"].astype(np.uint8), 1)
                before_bands["mask"] = str(mask_path_before.resolve())
            except Exception as exc:
                print(f"[SENTINEL2 WARNING] Could not write before mask: {exc}")

            # After quality
            scl_after = None
            if "scl" in after_bands and Path(after_bands["scl"]).exists():
                try:
                    with rasterio.open(after_bands["scl"]) as src:
                        scl_after = src.read(1)
                except Exception:
                    scl_after = None

            red_after = None
            if "red" in after_bands and Path(after_bands["red"]).exists():
                try:
                    with rasterio.open(after_bands["red"]) as src:
                        red_after = src.read(1)
                except Exception:
                    red_after = None

            masks_after = compute_quality_masks(
                scl_raster=scl_after,
                band_data=red_after,
                shape=(dst_height, dst_width),
            )
            quality_after = compute_quality_metrics(
                valid_mask=masks_after["valid_mask"],
                cloud_mask=masks_after["cloud_mask"] if scl_after is not None else None,
                cirrus_mask=masks_after["cirrus_mask"] if scl_after is not None else None,
                shadow_mask=masks_after["shadow_mask"] if scl_after is not None else None,
            )
            mask_path_after = self.cache_dir / f"s2_{after_id}_mask_{aoi_hash}.tif"
            try:
                with rasterio.open(mask_path_after, "w", **mask_profile) as dst:
                    dst.write(masks_after["valid_mask"].astype(np.uint8), 1)
                after_bands["mask"] = str(mask_path_after.resolve())
            except Exception as exc:
                print(f"[SENTINEL2 WARNING] Could not write after mask: {exc}")

            # Joint temporal validity
            joint_valid = compute_joint_valid_mask(masks_before["valid_mask"], masks_after["valid_mask"])
            joint_valid_pixels = int(np.sum(joint_valid))
            total_pixels = int(joint_valid.size)
            joint_quality = {
                "joint_valid_pixels": joint_valid_pixels,
                "total_pixels": total_pixels,
                "joint_valid_percentage": round((joint_valid_pixels / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0,
            }
            joint_mask_path = self.cache_dir / f"s2_joint_mask_{before_id}_{after_id}_{aoi_hash}.tif"
            try:
                with rasterio.open(joint_mask_path, "w", **mask_profile) as dst:
                    dst.write(joint_valid.astype(np.uint8), 1)
            except Exception as exc:
                print(f"[SENTINEL2 WARNING] Could not write joint mask: {exc}")


            # 7. Generate scientific display visualizations (True-Color, False-Color, Quality)
            before_vis: Dict[str, Any] = {}
            after_vis: Dict[str, Any] = {}

            green_before = None
            if "green" in before_bands and Path(before_bands["green"]).exists():
                try:
                    with rasterio.open(before_bands["green"]) as src:
                        green_before = src.read(1)
                except Exception:
                    green_before = None

            blue_before = None
            if "blue" in before_bands and Path(before_bands["blue"]).exists():
                try:
                    with rasterio.open(before_bands["blue"]) as src:
                        blue_before = src.read(1)
                except Exception:
                    blue_before = None

            nir_before = None
            if "nir" in before_bands and Path(before_bands["nir"]).exists():
                try:
                    with rasterio.open(before_bands["nir"]) as src:
                        nir_before = src.read(1)
                except Exception:
                    nir_before = None

            # True-Color Before (B04, B03, B02)
            if red_before is not None and green_before is not None:
                b_chan = blue_before if blue_before is not None else (green_before * 0.8)
                try:
                    tc_b = build_true_color_rgba(
                        red=red_before, green=green_before, blue=b_chan,
                        valid_mask=masks_before["valid_mask"]
                    )
                    before_vis["true_color"] = save_visualization_layer(
                        tc_b, f"true_color_before_{before_id}_{aoi_hash}.png",
                        source_raster_path=before_bands.get("red"), aoi_bbox=bbox
                    )
                except Exception as exc:
                    print(f"[SENTINEL2 VIS WARNING] Could not build true-color before: {exc}")

            # False-Color Before (B08, B04, B03)
            if nir_before is not None and red_before is not None and green_before is not None:
                try:
                    fc_b = build_false_color_rgba(
                        nir=nir_before, red=red_before, green=green_before,
                        valid_mask=masks_before["valid_mask"]
                    )
                    before_vis["false_color"] = save_visualization_layer(
                        fc_b, f"false_color_before_{before_id}_{aoi_hash}.png",
                        source_raster_path=before_bands.get("nir"), aoi_bbox=bbox
                    )
                except Exception as exc:
                    print(f"[SENTINEL2 VIS WARNING] Could not build false-color before: {exc}")

            # Quality Mask Before
            try:
                qm_b = build_quality_mask_rgba(
                    scl_raster=scl_before, valid_mask=masks_before["valid_mask"],
                    target_shape=(dst_height, dst_width)
                )
                before_vis["quality_mask"] = save_visualization_layer(
                    qm_b, f"quality_mask_before_{before_id}_{aoi_hash}.png",
                    source_raster_path=before_bands.get("red"), aoi_bbox=bbox
                )
            except Exception as exc:
                print(f"[SENTINEL2 VIS WARNING] Could not build quality mask before: {exc}")

            # Read bands for after display
            green_after = None
            if "green" in after_bands and Path(after_bands["green"]).exists():
                try:
                    with rasterio.open(after_bands["green"]) as src:
                        green_after = src.read(1)
                except Exception:
                    green_after = None

            blue_after = None
            if "blue" in after_bands and Path(after_bands["blue"]).exists():
                try:
                    with rasterio.open(after_bands["blue"]) as src:
                        blue_after = src.read(1)
                except Exception:
                    blue_after = None

            nir_after = None
            if "nir" in after_bands and Path(after_bands["nir"]).exists():
                try:
                    with rasterio.open(after_bands["nir"]) as src:
                        nir_after = src.read(1)
                except Exception:
                    nir_after = None

            # True-Color After (B04, B03, B02)
            if red_after is not None and green_after is not None:
                b_chan_a = blue_after if blue_after is not None else (green_after * 0.8)
                try:
                    tc_a = build_true_color_rgba(
                        red=red_after, green=green_after, blue=b_chan_a,
                        valid_mask=masks_after["valid_mask"]
                    )
                    after_vis["true_color"] = save_visualization_layer(
                        tc_a, f"true_color_after_{after_id}_{aoi_hash}.png",
                        source_raster_path=after_bands.get("red"), aoi_bbox=bbox
                    )
                except Exception as exc:
                    print(f"[SENTINEL2 VIS WARNING] Could not build true-color after: {exc}")

            # False-Color After (B08, B04, B03)
            if nir_after is not None and red_after is not None and green_after is not None:
                try:
                    fc_a = build_false_color_rgba(
                        nir=nir_after, red=red_after, green=green_after,
                        valid_mask=masks_after["valid_mask"]
                    )
                    after_vis["false_color"] = save_visualization_layer(
                        fc_a, f"false_color_after_{after_id}_{aoi_hash}.png",
                        source_raster_path=after_bands.get("nir"), aoi_bbox=bbox
                    )
                except Exception as exc:
                    print(f"[SENTINEL2 VIS WARNING] Could not build false-color after: {exc}")

            # Quality Mask After
            try:
                qm_a = build_quality_mask_rgba(
                    scl_raster=scl_after, valid_mask=masks_after["valid_mask"],
                    target_shape=(dst_height, dst_width)
                )
                after_vis["quality_mask"] = save_visualization_layer(
                    qm_a, f"quality_mask_after_{after_id}_{aoi_hash}.png",
                    source_raster_path=after_bands.get("red"), aoi_bbox=bbox
                )
            except Exception as exc:
                print(f"[SENTINEL2 VIS WARNING] Could not build quality mask after: {exc}")

            selection_before = {
                "scene_id": before_id,
                "date": before_date,
                "cloud_cover": before_cloud,
                "coverage": before_coverage,
                "score": before_score,
                "selection_reason": before_reason,
                "collection": "sentinel-2-l2a",
                "platform": before_platform,
                "processing_baseline": before_baseline,
                "mgrs_tile": before_mgrs,
                "quality": quality_before,
                "visualizations": before_vis,
            }

            selection_after = {
                "scene_id": after_id,
                "date": after_date,
                "cloud_cover": after_cloud,
                "coverage": after_coverage,
                "score": after_score,
                "selection_reason": after_reason,
                "collection": "sentinel-2-l2a",
                "platform": after_platform,
                "processing_baseline": after_baseline,
                "mgrs_tile": after_mgrs,
                "quality": quality_after,
                "visualizations": after_vis,
            }

            return {
                "status": "REAL_SUCCESS",
                "source": "REAL_SENTINEL_2",
                "query": {
                    "time_start": str(time_start or "2021"),
                    "time_end": str(time_end or "2025"),
                    "aoi": aoi,
                    "bbox": list(bbox),
                },
                "before": selection_before,
                "after": selection_after,
                "joint_quality": joint_quality,
                "joint_mask": str(joint_mask_path.resolve()),
                "visualizations": {
                    "before": before_vis,
                    "after": after_vis,
                },
                "selection": {
                    "before": selection_before,
                    "after": selection_after,
                },
                "images": [
                    {
                        "id": before_id,
                        "date": before_date,
                        "cloud_cover": before_cloud,
                        "bands": before_bands,
                        "metadata": selection_before,
                        "quality": quality_before,
                        "visualizations": before_vis,
                        "selection_reason": before_reason,
                    },
                    {
                        "id": after_id,
                        "date": after_date,
                        "cloud_cover": after_cloud,
                        "bands": after_bands,
                        "metadata": selection_after,
                        "quality": quality_after,
                        "visualizations": after_vis,
                        "selection_reason": after_reason,
                    },
                ],
            }



        except Sentinel2RetrievalError as exc:
            print(f"[SENTINEL2 RETRIEVAL ERROR] {exc}")
            return {
                "status": "REAL_FAILURE",
                "source": "REAL_SENTINEL_2",
                "error_type": exc.error_type,
                "error": exc.message,
                "details": exc.details,
                "query": {
                    "time_start": str(time_start or "2021"),
                    "time_end": str(time_end or "2025"),
                    "aoi": aoi,
                    "bbox": list(bbox),
                },
                "images": [],
            }
        except Exception as exc:
            print(f"[SENTINEL2 ERROR] Unexpected retrieval error: {exc}")
            return {
                "status": "REAL_FAILURE",
                "source": "REAL_SENTINEL_2",
                "error_type": Sentinel2ErrorType.STAC_UNAVAILABLE,
                "error": str(exc),
                "query": {
                    "time_start": str(time_start or "2021"),
                    "time_end": str(time_end or "2025"),
                    "aoi": aoi,
                    "bbox": list(bbox),
                },
                "images": [],
            }


    def search_and_fetch_temporal_series(
        self,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        aoi: Optional[Any] = None,
        preferred_season_month: int = 6,
        max_observations: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search, score, rank, and retrieve a multi-observation temporal series of
        Sentinel-2 Level-2A satellite imagery (e.g. 2021, 2022, 2023, 2024, 2025).
        Uses seasonal alignment matching across observations to minimize phenological distortion.
        """
        bbox = normalize_aoi(aoi)

        # Parse year endpoints
        str_start = str(time_start or "2021").strip()
        str_end = str(time_end or "2025").strip()
        yr_start = int(str_start[:4]) if str_start[:4].isdigit() else 2021
        yr_end = int(str_end[:4]) if str_end[:4].isdigit() else 2025
        if yr_end < yr_start:
            yr_start, yr_end = yr_end, yr_start

        # Determine target annual checkpoints
        if yr_end == yr_start:
            target_years = [yr_start]
        elif (yr_end - yr_start + 1) <= max_observations:
            target_years = list(range(yr_start, yr_end + 1))
        else:
            linspace_years = np.linspace(yr_start, yr_end, max_observations, dtype=int).tolist()
            target_years = sorted(list(set(linspace_years)))

        try:
            # 1. Base observation for start year
            dt_base_range = build_datetime_range(str(target_years[0]), str(target_years[0]))
            base_candidates = self.search_candidate_scenes(bbox=bbox, datetime_range=dt_base_range)
            ranked_base = rank_candidate_scenes(
                candidates=base_candidates,
                aoi_bbox=bbox,
                target_date=preferred_season_month,
                weights=self.weights,
            )
            if not ranked_base:
                raise Sentinel2RetrievalError(
                    Sentinel2ErrorType.NO_SCENES_FOUND,
                    f"No candidate scenes for base year {target_years[0]} in range {dt_base_range}.",
                )

            selected_scenes: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [ranked_base[0]]
            base_dt_str = ranked_base[0][1].get("datetime") or ""

            # 2. Subsequent observations matching base seasonal window
            for yr in target_years[1:]:
                dt_yr_range = build_datetime_range(str(yr), str(yr))
                try:
                    yr_cands = self.search_candidate_scenes(bbox=bbox, datetime_range=dt_yr_range)
                    ranked_yr = rank_candidate_scenes(
                        candidates=yr_cands,
                        aoi_bbox=bbox,
                        target_date=base_dt_str or preferred_season_month,
                        weights=self.weights,
                    )
                    if ranked_yr:
                        selected_scenes.append(ranked_yr[0])
                except Exception as yr_exc:
                    print(f"[SENTINEL2 TEMPORAL] Warning: No scene retrieved for year {yr}: {yr_exc}")

            if len(selected_scenes) < 1:
                raise Sentinel2RetrievalError(
                    Sentinel2ErrorType.NO_SCENES_FOUND,
                    f"No usable scenes found for temporal series {target_years}.",
                )

            # 3. Setup spatial raster parameters
            min_lng, min_lat, max_lng, max_lat = bbox
            res_deg = 0.0001
            dst_width = max(10, int(round((max_lng - min_lng) / res_deg)))
            dst_height = max(10, int(round((max_lat - min_lat) / res_deg)))
            dst_transform = rasterio.transform.from_bounds(
                min_lng, min_lat, max_lng, max_lat, dst_width, dst_height
            )
            aoi_key = f"{min_lng:.5f}_{min_lat:.5f}_{max_lng:.5f}_{max_lat:.5f}_{dst_width}x{dst_height}"
            aoi_hash = hashlib.md5(aoi_key.encode()).hexdigest()[:8]
            mask_profile = {
                "driver": "GTiff",
                "dtype": "uint8",
                "count": 1,
                "height": dst_height,
                "width": dst_width,
                "crs": CRS.from_epsg(4326),
                "transform": dst_transform,
                "nodata": 0,
                "compress": "lzw",
            }

            observations: List[Dict[str, Any]] = []

            for item, score_info in selected_scenes:
                sc_id = item["id"]
                props = item.get("properties", {})
                sc_dt_str = props.get("datetime") or item.get("datetime", "")
                sc_date = sc_dt_str[:10]
                sc_cloud = float(props.get("eo:cloud_cover", 0.0))
                sc_coverage = score_info["coverage_details"]["coverage_percentage"]
                sc_score = score_info["score"]
                sc_reason = score_info["selection_reason"]
                sc_platform = props.get("platform", "Sentinel-2")
                sc_baseline = str(props.get("s2:processing_baseline", "03.00"))
                sc_mgrs = props.get("s2:mgrs_tile", "")

                try:
                    sc_dt = datetime.fromisoformat(sc_dt_str.replace("Z", "+00:00"))
                    sc_year = sc_dt.year
                    sc_doy = sc_dt.timetuple().tm_yday
                except Exception:
                    sc_year = int(sc_date[:4]) if len(sc_date) >= 4 and sc_date[:4].isdigit() else 2021
                    sc_doy = 182

                bands = self.fetch_scene_bands(scene_item=item, bbox=bbox)

                scl_arr = None
                if "scl" in bands and Path(bands["scl"]).exists():
                    try:
                        with rasterio.open(bands["scl"]) as src:
                            scl_arr = src.read(1)
                    except Exception:
                        scl_arr = None

                red_arr = None
                if "red" in bands and Path(bands["red"]).exists():
                    try:
                        with rasterio.open(bands["red"]) as src:
                            red_arr = src.read(1)
                    except Exception:
                        red_arr = None

                q_masks = compute_quality_masks(
                    scl_raster=scl_arr,
                    band_data=red_arr,
                    shape=(dst_height, dst_width),
                )
                quality_metrics = compute_quality_metrics(
                    valid_mask=q_masks["valid_mask"],
                    cloud_mask=q_masks["cloud_mask"] if scl_arr is not None else None,
                    cirrus_mask=q_masks["cirrus_mask"] if scl_arr is not None else None,
                    shadow_mask=q_masks["shadow_mask"] if scl_arr is not None else None,
                )
                mask_p = self.cache_dir / f"s2_{sc_id}_mask_{aoi_hash}.tif"
                try:
                    with rasterio.open(mask_p, "w", **mask_profile) as dst:
                        dst.write(q_masks["valid_mask"].astype(np.uint8), 1)
                    bands["mask"] = str(mask_p.resolve())
                except Exception as m_exc:
                    print(f"[SENTINEL2 WARNING] Could not write mask for {sc_id}: {m_exc}")

                v_mask = q_masks["valid_mask"]
                nir_arr = None
                if "nir" in bands and Path(bands["nir"]).exists():
                    try:
                        with rasterio.open(bands["nir"]) as src:
                            nir_arr = src.read(1)
                    except Exception:
                        nir_arr = None

                green_arr = None
                if "green" in bands and Path(bands["green"]).exists():
                    try:
                        with rasterio.open(bands["green"]) as src:
                            green_arr = src.read(1)
                    except Exception:
                        green_arr = None

                swir_arr = None
                if "swir" in bands and Path(bands["swir"]).exists():
                    try:
                        with rasterio.open(bands["swir"]) as src:
                            swir_arr = src.read(1)
                    except Exception:
                        swir_arr = None

                def _calc_stats(arr1, arr2, sign_pos=True):
                    if arr1 is None or arr2 is None:
                        return None, None, None
                    denom = arr1 + arr2
                    valid = (denom > 0) & v_mask & np.isfinite(arr1) & np.isfinite(arr2)
                    if not np.any(valid):
                        return 0.0, 0.0, 0.0
                    idx = (arr1 - arr2) / denom if sign_pos else (arr2 - arr1) / denom
                    idx_v = idx[valid]
                    return (
                        round(float(np.mean(idx_v)), 4),
                        round(float(np.median(idx_v)), 4),
                        round(float(np.std(idx_v)), 4),
                    )

                ndvi_m, ndvi_med, ndvi_s = _calc_stats(nir_arr, red_arr, True)
                ndwi_m, ndwi_med, ndwi_s = _calc_stats(green_arr, nir_arr, True)
                ndbi_m, ndbi_med, ndbi_s = _calc_stats(swir_arr, nir_arr, True)

                vis: Dict[str, Any] = {}
                if red_arr is not None and green_arr is not None:
                    blue_arr = None
                    if "blue" in bands and Path(bands["blue"]).exists():
                        try:
                            with rasterio.open(bands["blue"]) as src:
                                blue_arr = src.read(1)
                        except Exception:
                            blue_arr = None
                    b_chan = blue_arr if blue_arr is not None else (green_arr * 0.8)
                    try:
                        tc = build_true_color_rgba(red=red_arr, green=green_arr, blue=b_chan, valid_mask=v_mask)
                        vis["true_color"] = save_visualization_layer(
                            tc, f"true_color_{sc_id}_{aoi_hash}.png", source_raster_path=bands.get("red"), aoi_bbox=bbox
                        )
                    except Exception as exc:
                        print(f"[SENTINEL2 VIS WARNING] {exc}")

                if nir_arr is not None and red_arr is not None and green_arr is not None:
                    try:
                        fc = build_false_color_rgba(nir=nir_arr, red=red_arr, green=green_arr, valid_mask=v_mask)
                        vis["false_color"] = save_visualization_layer(
                            fc, f"false_color_{sc_id}_{aoi_hash}.png", source_raster_path=bands.get("nir"), aoi_bbox=bbox
                        )
                    except Exception as exc:
                        print(f"[SENTINEL2 VIS WARNING] {exc}")

                try:
                    qm = build_quality_mask_rgba(scl_raster=scl_arr, valid_mask=v_mask, target_shape=(dst_height, dst_width))
                    vis["quality_mask"] = save_visualization_layer(
                        qm, f"quality_mask_{sc_id}_{aoi_hash}.png", source_raster_path=bands.get("red"), aoi_bbox=bbox
                    )
                except Exception as exc:
                    print(f"[SENTINEL2 VIS WARNING] {exc}")

                valid_cov = float(quality_metrics.get("valid_coverage_percentage", 100.0))
                q_state = "high" if valid_cov >= 85.0 else "moderate" if valid_cov >= 50.0 else "low"

                obs_record = {
                    "observation_id": sc_id,
                    "scene_id": sc_id,
                    "datetime": sc_dt_str,
                    "datetime_iso": sc_dt_str,
                    "date": sc_date,
                    "year": sc_year,
                    "day_of_year": sc_doy,
                    "cloud_cover": sc_cloud,
                    "coverage_fraction": sc_coverage / 100.0,
                    "valid_fraction": round(valid_cov / 100.0, 4),
                    "quality_state": q_state,
                    "acquisition_score": sc_score,
                    "provenance": {
                        "platform": sc_platform,
                        "processing_baseline": sc_baseline,
                        "mgrs_tile": sc_mgrs,
                        "selection_reason": sc_reason,
                    },
                    "ndvi": ndvi_m,
                    "ndvi_mean": ndvi_m,
                    "ndvi_median": ndvi_med,
                    "ndvi_std": ndvi_s,
                    "ndwi": ndwi_m,
                    "ndwi_mean": ndwi_m,
                    "ndwi_median": ndwi_med,
                    "ndwi_std": ndwi_s,
                    "ndbi": ndbi_m,
                    "ndbi_mean": ndbi_m,
                    "ndbi_median": ndbi_med,
                    "ndbi_std": ndbi_s,
                    "band_paths": bands,
                    "bands": bands,
                    "quality": quality_metrics,
                    "visualizations": vis,
                }
                observations.append(obs_record)

            observations.sort(key=lambda o: o["datetime_iso"])

            before_obs = observations[0]
            after_obs = observations[-1]

            v1_p = before_obs["bands"].get("mask")
            v2_p = after_obs["bands"].get("mask")
            joint_valid_pct = 100.0
            if v1_p and v2_p and Path(v1_p).exists() and Path(v2_p).exists():
                try:
                    with rasterio.open(v1_p) as s1, rasterio.open(v2_p) as s2:
                        jv = (s1.read(1) > 0) & (s2.read(1) > 0)
                        joint_valid_pct = round(float(np.mean(jv) * 100.0), 2)
                except Exception:
                    pass

            return {
                "status": "REAL_SUCCESS",
                "source": "REAL_SENTINEL_2",
                "query": {
                    "time_start": str_start,
                    "time_end": str_end,
                    "aoi": aoi,
                    "bbox": list(bbox),
                },
                "temporal_observations": observations,
                "observation_count": len(observations),
                "before": before_obs,
                "after": after_obs,
                "images": [before_obs, after_obs],
                "joint_quality": {
                    "joint_valid_percentage": joint_valid_pct,
                },
                "visualizations": {
                    "before": before_obs["visualizations"],
                    "after": after_obs["visualizations"],
                },
                "selection": {
                    "before": before_obs,
                    "after": after_obs,
                },
            }

        except Sentinel2RetrievalError as exc:
            print(f"[SENTINEL2 RETRIEVAL ERROR] {exc}")
            return {
                "status": "REAL_FAILURE",
                "source": "REAL_SENTINEL_2",
                "error_type": exc.error_type,
                "error": exc.message,
                "details": exc.details,
                "query": {
                    "time_start": str(time_start or "2021"),
                    "time_end": str(time_end or "2025"),
                    "aoi": aoi,
                    "bbox": list(bbox),
                },
                "images": [],
                "temporal_observations": [],
            }
        except Exception as exc:
            print(f"[SENTINEL2 ERROR] Unexpected retrieval error: {exc}")
            return {
                "status": "REAL_FAILURE",
                "source": "REAL_SENTINEL_2",
                "error_type": Sentinel2ErrorType.STAC_UNAVAILABLE,
                "error": str(exc),
                "query": {
                    "time_start": str(time_start or "2021"),
                    "time_end": str(time_end or "2025"),
                    "aoi": aoi,
                    "bbox": list(bbox),
                },
                "images": [],
                "temporal_observations": [],
            }


# ============================================================
# SEARCH IMAGERY FUNCTION WRAPPER
# ============================================================

def search_real_sentinel2(
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    aoi: Optional[Any] = None,
    multi_temporal: bool = False,
    max_observations: int = 5,
    **kwargs,
) -> Dict[str, Any]:
    """
    Search and fetch real Sentinel-2 satellite imagery for before and after periods,
    or an entire multi-observation temporal series when multi_temporal is True.
    """
    provider = Sentinel2Provider()
    if multi_temporal:
        return provider.search_and_fetch_temporal_series(
            time_start=time_start,
            time_end=time_end,
            aoi=aoi,
            max_observations=max_observations,
            **kwargs,
        )
    return provider.search_and_fetch(
        time_start=time_start,
        time_end=time_end,
        aoi=aoi,
        **kwargs,
    )

