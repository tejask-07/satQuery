"""
Sentinel-1 SAR Satellite Imagery Acquisition Provider (Step 12).

Retrieves real Sentinel-1 Level-1 Ground Range Detected (GRD) synthetic aperture radar
imagery from Microsoft Planetary Computer STAC API, applies SAS token URL signing,
crops/rectifies to an aligned GeoTIFF grid, and preserves raw physical SAR semantics.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
SENTINEL1_COLLECTION = "sentinel-1-grd"

# Default AOI for Pune / Mumbai region (~6km x 6km) if AOI is omitted
DEFAULT_BBOX = (73.80, 18.50, 73.86, 18.56)

# ~10 meters in EPSG:4326 degrees (~0.0001 deg)
DEFAULT_PIXEL_RES_DEG = 0.00009

BACKEND_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[4]


# ============================================================
# ERROR TYPES & STRUCTURED EXCEPTIONS
# ============================================================

class Sentinel1ErrorType:
    NO_SCENES_FOUND = "NO_SCENES_FOUND"
    STAC_UNAVAILABLE = "STAC_UNAVAILABLE"
    MALFORMED_AOI = "MALFORMED_AOI"
    INVALID_DATES = "INVALID_DATES"
    ASSET_UNAVAILABLE = "ASSET_UNAVAILABLE"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"


class Sentinel1RetrievalError(RuntimeError):
    """
    Structured actionable error for Sentinel-1 retrieval failures.
    """
    def __init__(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{error_type}] {message}")
        self.error_type = error_type
        self.message = message
        self.details = details or {}


# ============================================================
# CACHE DIRECTORY RESOLUTION
# ============================================================

def get_cache_dir() -> Path:
    """
    Return the resolved local cache directory for GeoTIFF downloads.
    Follows existing SatQuery cache convention (SATQUERY_CACHE_DIR).
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
    Accepts bbox lists/tuples, GeoJSON geometries, or dicts with coordinate bounds.
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


def build_datetime_range(time_value: Optional[str], default_year: str = "2021") -> str:
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
# SCENE SCORING & RANKING
# ============================================================

@dataclass
class Sentinel1ScoringWeights:
    """
    Explainable weights for ranking Sentinel-1 candidate SAR scenes.
    Weights sum to 1.0. Note: Cloud cover is NOT used as a SAR criterion.
    """
    weight_coverage: float = 0.40      # Prioritize full AOI spatial coverage
    weight_temporal: float = 0.35      # Prioritize temporal closeness to requested/target date
    weight_polarization: float = 0.15  # Prioritize dual-pol (VV+VH) over single-pol
    weight_mode: float = 0.10          # Prioritize standard Interferometric Wide (IW) swath


def calculate_aoi_coverage(
    scene_item: Dict[str, Any],
    aoi_bbox: Tuple[float, float, float, float],
) -> Dict[str, float]:
    """
    Calculate spatial intersection area and coverage ratio between scene and requested AOI.
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


def score_sentinel1_scene(
    scene_item: Dict[str, Any],
    aoi_bbox: Tuple[float, float, float, float],
    target_date: Optional[Union[datetime, str]] = None,
    weights: Optional[Sentinel1ScoringWeights] = None,
    prefer_dual_pol: bool = True,
) -> Dict[str, Any]:
    """
    Score a candidate Sentinel-1 scene based on:
    1. AOI spatial coverage
    2. Temporal closeness to target date
    3. Dual-polarization (VV + VH) availability
    4. Instrument mode (IW preferred)
    """
    weights = weights or Sentinel1ScoringWeights()
    props = scene_item.get("properties", {})

    # 1. Coverage Score (0.0 to 1.0)
    cov_details = calculate_aoi_coverage(scene_item, aoi_bbox)
    coverage_score = cov_details["coverage_ratio"]

    # 2. Temporal Proximity Score (0.0 to 1.0)
    cand_dt = None
    dt_str = props.get("datetime") or scene_item.get("datetime", "")
    if dt_str:
        try:
            cand_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            pass

    temporal_score = 0.8  # default when target date unspecified
    delta_days = 0.0
    if target_date is not None and cand_dt is not None:
        target_dt = None
        if isinstance(target_date, str):
            try:
                target_dt = datetime.fromisoformat(target_date.replace("Z", "+00:00"))
            except Exception:
                pass
        elif isinstance(target_date, datetime):
            target_dt = target_date

        if target_dt is not None:
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=timezone.utc)
            if cand_dt.tzinfo is None:
                cand_dt = cand_dt.replace(tzinfo=timezone.utc)
            delta_days = abs((cand_dt - target_dt).total_seconds()) / 86400.0
            # Full score at 0 days delta; decays over 14 days
            temporal_score = max(0.0, 1.0 - (delta_days / 14.0))

    # 3. Polarization Score (0.0 to 1.0)
    pols = props.get("sar:polarizations", [])
    assets = scene_item.get("assets", {})
    has_vv = "vv" in assets or "VV" in pols
    has_vh = "vh" in assets or "VH" in pols

    if prefer_dual_pol:
        if has_vv and has_vh:
            pol_score = 1.0
        elif has_vv:
            pol_score = 0.6
        else:
            pol_score = 0.3
    else:
        pol_score = 1.0 if has_vv else 0.5

    # 4. Mode Score (0.0 to 1.0)
    mode = props.get("sar:instrument_mode", "").upper()
    if mode == "IW":
        mode_score = 1.0
    elif mode in ("EW", "SM"):
        mode_score = 0.6
    else:
        mode_score = 0.4

    # Weighted Total Score
    total_w = (
        weights.weight_coverage
        + weights.weight_temporal
        + weights.weight_polarization
        + weights.weight_mode
    )
    total_score = (
        weights.weight_coverage * coverage_score
        + weights.weight_temporal * temporal_score
        + weights.weight_polarization * pol_score
        + weights.weight_mode * mode_score
    ) / total_w
    total_score = round(total_score, 4)

    pol_desc = "dual-pol (VV+VH)" if (has_vv and has_vh) else ("VV-only" if has_vv else "other-pol")
    selection_reason = (
        f"Selected with score {total_score:.4f} based on {cov_details['coverage_percentage']:.1f}% AOI coverage, "
        f"{pol_desc}, mode '{mode or 'unknown'}', and {delta_days:.1f} days delta from target date."
    )

    return {
        "score": total_score,
        "coverage_score": round(coverage_score, 4),
        "temporal_score": round(temporal_score, 4),
        "pol_score": round(pol_score, 4),
        "mode_score": round(mode_score, 4),
        "coverage_details": cov_details,
        "selection_reason": selection_reason,
        "datetime": dt_str,
        "has_vv": has_vv,
        "has_vh": has_vh,
        "delta_days": round(delta_days, 2),
    }


def rank_sentinel1_scenes(
    candidates: List[Dict[str, Any]],
    aoi_bbox: Tuple[float, float, float, float],
    target_date: Optional[Union[datetime, str]] = None,
    weights: Optional[Sentinel1ScoringWeights] = None,
    prefer_dual_pol: bool = True,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Score and rank candidate Sentinel-1 scenes deterministically by total score descending.
    Returns list of (candidate_item, score_result).
    """
    scored = []
    for c in candidates:
        sc = score_sentinel1_scene(
            c,
            aoi_bbox=aoi_bbox,
            target_date=target_date,
            weights=weights,
            prefer_dual_pol=prefer_dual_pol,
        )
        scored.append((c, sc))

    # Deterministic sort: score desc, delta_days asc, id asc
    scored.sort(
        key=lambda pair: (
            -pair[1]["score"],
            pair[1]["delta_days"],
            pair[0].get("id", ""),
        )
    )
    return scored


# ============================================================
# SENTINEL-1 ACQUISITION PROVIDER
# ============================================================

class Sentinel1Provider:
    """
    Real Sentinel-1 Ground Range Detected (GRD) Synthetic Aperture Radar Provider.

    Queries Microsoft Planetary Computer STAC API (`sentinel-1-grd` collection),
    uses SAS URL signing, applies rasterio WarpedVRT to rectify Ground Control Points (GCPs)
    onto a regular EPSG:4326 grid, crops to requested AOI, caches locally as GeoTIFFs,
    and preserves raw physical SAR semantics without uncalibrated conversions.
    """

    def __init__(
        self,
        stac_url: Optional[str] = None,
        sign_url: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        weights: Optional[Sentinel1ScoringWeights] = None,
    ):
        self.stac_url = stac_url or os.getenv("SENTINEL_STAC_URL", DEFAULT_STAC_URL)
        self.sign_url = sign_url or os.getenv("SENTINEL_SAS_SIGN_URL", DEFAULT_SAS_SIGN_URL)
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else get_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.weights = weights or Sentinel1ScoringWeights()

    def search_candidate_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        datetime_range: str,
        instrument_mode: str = "IW",
        orbit_direction: Optional[str] = None,
        max_candidates: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search Planetary Computer STAC for candidate Sentinel-1 GRD scenes.
        """
        w, s, e, n = bbox
        if w >= e or s >= n or not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90):
            raise Sentinel1RetrievalError(
                Sentinel1ErrorType.MALFORMED_AOI,
                f"Invalid bounding box coordinates: {bbox}. Expected [west, south, east, north] with west < east and south < north.",
                details={"bbox": bbox},
            )

        search_endpoint = f"{self.stac_url.rstrip('/')}/search"
        payload: Dict[str, Any] = {
            "collections": [SENTINEL1_COLLECTION],
            "bbox": list(bbox),
            "datetime": datetime_range,
            "limit": max_candidates,
            "sortby": [
                {
                    "field": "properties.datetime",
                    "direction": "desc",
                }
            ],
        }

        query_filters: Dict[str, Any] = {}
        if instrument_mode:
            query_filters["sar:instrument_mode"] = {"eq": instrument_mode}
        if orbit_direction:
            query_filters["sat:orbit_state"] = {"eq": orbit_direction.lower()}

        if query_filters:
            payload["query"] = query_filters

        candidates: List[Dict[str, Any]] = []
        last_error = None

        try:
            response = requests.post(
                search_endpoint,
                json=payload,
                timeout=25,
            )
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("features", [])
            else:
                last_error = f"STAC API returned HTTP {response.status_code}: {response.text}"
        except Exception as exc:
            last_error = f"STAC network error: {exc}"

        # If mode filter yielded 0 results, retry without mode restriction
        if not candidates and instrument_mode:
            payload.pop("query", None)
            try:
                retry_resp = requests.post(search_endpoint, json=payload, timeout=25)
                if retry_resp.status_code == 200:
                    candidates = retry_resp.json().get("features", [])
            except Exception:
                pass

        if not candidates:
            if last_error and "network error" in last_error.lower():
                raise Sentinel1RetrievalError(
                    Sentinel1ErrorType.STAC_UNAVAILABLE,
                    f"Planetary Computer STAC endpoint unavailable ({self.stac_url}): {last_error}",
                    details={"bbox": bbox, "datetime_range": datetime_range},
                )
            raise Sentinel1RetrievalError(
                Sentinel1ErrorType.NO_SCENES_FOUND,
                f"No Sentinel-1 GRD scenes found for AOI {bbox} in date range '{datetime_range}'. Details: {last_error or 'No intersecting scenes found.'}",
                details={"bbox": bbox, "datetime_range": datetime_range},
            )

        return candidates

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
            return raw_url

    def fetch_and_cache_polarization(
        self,
        scene_item: Dict[str, Any],
        polarization: str,
        bbox: Tuple[float, float, float, float],
        dst_crs: CRS = CRS.from_epsg(4326),
        pixel_res_deg: float = DEFAULT_PIXEL_RES_DEG,
    ) -> str:
        """
        Download and crop the requested polarization band for the given scene and AOI window.
        Saves as an aligned GeoTIFF in the local cache, preserving raw physical units.
        """
        pol_key = polarization.lower()
        scene_id = scene_item.get("id", "s1_scene")
        safe_scene_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", scene_id)

        assets = scene_item.get("assets", {})
        if pol_key not in assets:
            raise Sentinel1RetrievalError(
                Sentinel1ErrorType.ASSET_UNAVAILABLE,
                f"Scene {scene_id} does not contain asset for polarization '{polarization}'. Available assets: {list(assets.keys())}",
                details={"scene_id": scene_id, "polarization": polarization},
            )

        raw_href = assets[pol_key]["href"]

        min_lng, min_lat, max_lng, max_lat = bbox
        width = max(1, int(round((max_lng - min_lng) / pixel_res_deg)))
        height = max(1, int(round((max_lat - min_lat) / pixel_res_deg)))
        dst_transform = rasterio.transform.from_bounds(min_lng, min_lat, max_lng, max_lat, width, height)

        # Deterministic cache filename based on sanitized scene ID, polarization, and AOI window
        aoi_key = f"{min_lng:.5f}_{min_lat:.5f}_{max_lng:.5f}_{max_lat:.5f}_{width}x{height}"
        aoi_hash = hashlib.md5(aoi_key.encode()).hexdigest()[:8]
        cache_filename = f"s1_{safe_scene_id}_{pol_key}_{aoi_hash}.tif"
        cache_path = (self.cache_dir / cache_filename).resolve()

        # Security check: ensure path is within cache_dir
        if not str(cache_path).startswith(str(self.cache_dir.resolve())):
            raise PermissionError(f"Unauthorized path escape detected: {cache_path}")

        # Check existing cache
        if cache_path.exists():
            try:
                with rasterio.open(cache_path) as src:
                    if src.width == width and src.height == height:
                        return str(cache_path)
            except Exception:
                pass  # Re-download if corrupt

        # Sign asset URL
        signed_href = self.sign_asset_url(raw_href)

        # Read and rectify GCPs using WarpedVRT
        try:
            with rasterio.open(signed_href) as src:
                with WarpedVRT(src, crs=dst_crs, resampling=Resampling.bilinear) as vrt:
                    window = from_bounds(min_lng, min_lat, max_lng, max_lat, vrt.transform)
                    data = vrt.read(
                        1,
                        window=window,
                        out_shape=(height, width),
                        resampling=Resampling.bilinear,
                    )
        except Exception as exc:
            raise Sentinel1RetrievalError(
                Sentinel1ErrorType.DOWNLOAD_FAILED,
                f"Failed to read/warp Sentinel-1 asset from {signed_href[:60]}...: {exc}",
                details={"scene_id": scene_id, "polarization": polarization},
            )

        # Write rectified GeoTIFF preserving raw uint16 values
        try:
            with rasterio.open(
                cache_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=data.dtype,
                crs=dst_crs,
                transform=dst_transform,
                nodata=0,
                compress="lzw",
            ) as dst:
                dst.write(data, 1)
                dst.set_band_description(1, f"Sentinel-1 {polarization.upper()} Backscatter DN")
        except Exception as exc:
            raise Sentinel1RetrievalError(
                Sentinel1ErrorType.DOWNLOAD_FAILED,
                f"Failed to write cached GeoTIFF to {cache_path}: {exc}",
                details={"cache_path": str(cache_path)},
            )

        return str(cache_path)

    def search_and_fetch(
        self,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        aoi: Optional[Any] = None,
        target_date: Optional[Union[datetime, str]] = None,
        prefer_dual_pol: bool = True,
        orbit_direction: Optional[str] = None,
        pixel_res_deg: float = DEFAULT_PIXEL_RES_DEG,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        End-to-end acquisition:
        1. Normalizes AOI and datetime range.
        2. Queries STAC for candidate Sentinel-1 GRD scenes.
        3. Deterministically scores and ranks candidates.
        4. Fetches and caches VV and VH bands (or VV-only).
        5. Returns structured acquisition result with metadata.
        """
        bbox = normalize_aoi(aoi)

        # Build datetime range
        if time_start and time_end:
            datetime_range = f"{time_start}/{time_end}"
        elif time_start:
            datetime_range = build_datetime_range(time_start)
        else:
            datetime_range = build_datetime_range(None, default_year="2021")

        try:
            candidates = self.search_candidate_scenes(
                bbox=bbox,
                datetime_range=datetime_range,
                orbit_direction=orbit_direction,
            )

            ranked = rank_sentinel1_scenes(
                candidates=candidates,
                aoi_bbox=bbox,
                target_date=target_date,
                weights=self.weights,
                prefer_dual_pol=prefer_dual_pol,
            )

            if not ranked:
                raise Sentinel1RetrievalError(
                    Sentinel1ErrorType.NO_SCENES_FOUND,
                    f"No rankable Sentinel-1 scenes found for AOI {bbox}.",
                    details={"bbox": bbox, "datetime_range": datetime_range},
                )

            selected_item, score_info = ranked[0]
            scene_id = selected_item.get("id", "unknown_s1")
            props = selected_item.get("properties", {})
            assets = selected_item.get("assets", {})

            # Fetch VV
            vv_path = None
            if "vv" in assets:
                vv_path = self.fetch_and_cache_polarization(
                    scene_item=selected_item,
                    polarization="vv",
                    bbox=bbox,
                    pixel_res_deg=pixel_res_deg,
                )

            # Fetch VH if available
            vh_path = None
            if "vh" in assets:
                vh_path = self.fetch_and_cache_polarization(
                    scene_item=selected_item,
                    polarization="vh",
                    bbox=bbox,
                    pixel_res_deg=pixel_res_deg,
                )

            if not vv_path and not vh_path:
                raise Sentinel1RetrievalError(
                    Sentinel1ErrorType.ASSET_UNAVAILABLE,
                    f"Selected scene {scene_id} contains neither VV nor VH assets.",
                    details={"scene_id": scene_id, "assets": list(assets.keys())},
                )

            available_pols = []
            if vv_path:
                available_pols.append("VV")
            if vh_path:
                available_pols.append("VH")

            ref_path = vv_path or vh_path
            dims = [0, 0]
            if ref_path and Path(ref_path).exists():
                with rasterio.open(ref_path) as src:
                    dims = [src.width, src.height]

            dt_str = props.get("datetime") or selected_item.get("datetime", "")
            product_type = props.get("sar:product_type", "GRD")
            platform = props.get("platform", "Sentinel-1")
            mode = props.get("sar:instrument_mode", "IW")
            orbit_state = props.get("sat:orbit_state", "unknown")

            return {
                "status": "REAL_SUCCESS",
                "provider": "sentinel1",
                "collection": SENTINEL1_COLLECTION,
                "product": product_type,
                "item_id": scene_id,
                "acquisition_datetime": dt_str,
                "platform": platform,
                "instrument_mode": mode,
                "polarizations": available_pols,
                "orbit_direction": orbit_state,
                "vv": vv_path,
                "vh": vh_path,
                "crs": "EPSG:4326",
                "bounds": list(bbox),
                "dimensions": dims,
                "selection_reason": score_info["selection_reason"],
                "metadata": {
                    "item_id": scene_id,
                    "datetime": dt_str,
                    "sar:polarizations": available_pols,
                    "sar:instrument_mode": mode,
                    "sar:product_type": product_type,
                    "sat:orbit_state": orbit_state,
                    "source": "REAL_SENTINEL_1",
                    "coverage_percentage": score_info["coverage_details"]["coverage_percentage"],
                    "temporal_delta_days": score_info["delta_days"],
                    "physical_units": "uncalibrated_linear_dn",
                },
                "errors": [],
            }

        except Sentinel1RetrievalError as exc:
            return {
                "status": "REAL_FAILURE",
                "provider": "sentinel1",
                "error_type": exc.error_type,
                "error": exc.message,
                "details": exc.details,
                "vv": None,
                "vh": None,
                "query": {
                    "time_start": time_start,
                    "time_end": time_end,
                    "aoi": aoi,
                    "bbox": list(bbox),
                },
            }
        except Exception as exc:
            return {
                "status": "REAL_FAILURE",
                "provider": "sentinel1",
                "error_type": Sentinel1ErrorType.STAC_UNAVAILABLE,
                "error": str(exc),
                "vv": None,
                "vh": None,
                "query": {
                    "time_start": time_start,
                    "time_end": time_end,
                    "aoi": aoi,
                    "bbox": list(bbox),
                },
            }


# ============================================================
# STANDALONE CONVENIENCE FUNCTION
# ============================================================

def search_real_sentinel1(
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    aoi: Optional[Any] = None,
    target_date: Optional[Union[datetime, str]] = None,
    prefer_dual_pol: bool = True,
    orbit_direction: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Search and fetch real Sentinel-1 SAR imagery (VV and VH) for an AOI and time range.
    """
    provider = Sentinel1Provider()
    return provider.search_and_fetch(
        time_start=time_start,
        time_end=time_end,
        aoi=aoi,
        target_date=target_date,
        prefer_dual_pol=prefer_dual_pol,
        orbit_direction=orbit_direction,
        **kwargs,
    )
