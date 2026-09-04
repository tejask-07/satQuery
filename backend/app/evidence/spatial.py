"""
Phase 6: Spatial Reasoning, Candidate Region Clustering & Geometric Contiguity Engine.

Extends Phase 5B pixel-level candidate rasters into spatially coherent candidate regions,
answering "WHERE did potential change occur?" with rigorous geometric contiguity,
minimum mapping unit (MMU) filtering, polygon vectorization, and grounded spatial descriptions.

CORE PRINCIPLES:
1. Deterministic Spatial Analysis:
   Strictly deterministic algorithms (connected-component labeling, morphology, geometry).
   Zero black-box ML, object detectors, or deep-learning segmentation models.
2. Candidate Regions, Not Objects:
   Regions represent clusters of contiguous candidate pixels, NOT verified real-world buildings,
   forest parcels, or construction developments.
3. Decoupling of Spatial Coherence from Confidence:
   Spatial coherence measures geometric compactness and size regularity. It is NEVER called
   confidence or probability.
4. Accurate Geodesic / Projected Area:
   Area is computed directly from raster resolution and transform, correctly scaled by latitude
   for geographic coordinates (EPSG:4326), never guessed from AOI bounding boxes.
5. Strict Invalid-Pixel Exclusion:
   Joint-invalid / cloud / shadow pixels are strictly excluded and never contribute to candidate regions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio
import rasterio.features
from affine import Affine


# ============================================================
# CENTRALIZED SPATIAL CONFIGURATION
# ============================================================

class SpatialConfig:
    """
    Centralized, documented configuration thresholds for Phase 6 Spatial Reasoning.
    """
    # Minimum Mapping Unit: small disconnected clusters below this pixel count are filtered as noise
    MMU_MIN_PIXELS: int = 5

    # Minimum spatial coherence score to consider a region well-formed
    MIN_SPATIAL_SCORE: float = 0.20

    # Connectivity for connected-component labeling (8-connectivity for natural contiguous spatial features)
    CONNECTIVITY: int = 8

    # Policy for handling uncertain (class 3) candidate pixels: "separate" or "exclude"
    UNCERTAIN_REGION_POLICY: str = "separate"

    # Minimum spatial overlap fraction required to claim a transition between source & destination
    TRANSITION_OVERLAP_THRESHOLD: float = 0.15

    # Earth radius / meters per degree of latitude at WGS84 ellipsoid
    METERS_PER_DEGREE_LAT: float = 111320.0


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class CandidateRegion:
    """
    Structured representation of a contiguous candidate change region.
    """
    region_id: int
    candidate_class: int  # 1: primary candidate, 2: counter candidate, 3: uncertain
    class_label: str      # e.g. "primary_candidate", "counter_candidate", "uncertain"
    pixel_count: int
    area_m2: float
    area_hectares: float
    centroid_pixel: List[float]  # [col, row]
    centroid_geo: List[float]    # [lon, lat]
    bbox_pixel: List[int]        # [min_col, min_row, width, height]
    bbox_geo: List[float]        # [min_lon, min_lat, max_lon, max_lat]
    spatial_coherence: float
    valid_pixel_fraction: float
    mean_candidate_score: float = 0.0
    max_candidate_score: float = 0.0
    mean_ndbi_delta: Optional[float] = None
    mean_ndvi_delta: Optional[float] = None
    mean_ndwi_delta: Optional[float] = None
    mean_spectral_delta: Optional[float] = None
    geometry: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# GEODESIC & TRANSFORM UTILITIES
# ============================================================

def calculate_pixel_area_m2(
    transform: Affine,
    centroid_lat: float,
    crs: Optional[Any] = None,
) -> float:
    """
    Calculates pixel area in square meters.
    If coordinates are geographic (degrees, e.g. EPSG:4326), scales longitude by cos(latitude).
    If coordinates are projected (linear meters), computes dx * dy directly.
    """
    dx = abs(float(transform.a))
    dy = abs(float(transform.e))

    crs_str = str(crs or "").lower()
    is_geographic = (
        "4326" in crs_str
        or "wgs" in crs_str
        or dx < 0.1  # degree increments are small (< 0.1)
    )

    if is_geographic:
        # Scale by latitude at region centroid
        lat_rad = math.radians(centroid_lat)
        meters_x = dx * SpatialConfig.METERS_PER_DEGREE_LAT * max(0.01, math.cos(lat_rad))
        meters_y = dy * SpatialConfig.METERS_PER_DEGREE_LAT
        return float(meters_x * meters_y)
    else:
        # Projected coordinates in meters
        return float(dx * dy)


def pixel_to_geo(transform: Affine, col: float, row: float) -> Tuple[float, float]:
    """
    Transforms pixel coordinate (col, row) to geographic coordinate (lon, lat) using center offset.
    """
    lon, lat = rasterio.transform.xy(transform, row, col, offset="center")
    return float(lon), float(lat)


def compute_geographic_bbox(
    transform: Affine,
    min_col: int,
    min_row: int,
    width: int,
    height: int,
) -> List[float]:
    """
    Computes geographic bounding box [min_lon, min_lat, max_lon, max_lat].
    """
    lon1, lat1 = rasterio.transform.xy(transform, min_row, min_col, offset="ul")
    lon2, lat2 = rasterio.transform.xy(transform, min_row + height, min_col + width, offset="lr")
    return [
        round(min(float(lon1), float(lon2)), 6),
        round(min(float(lat1), float(lat2)), 6),
        round(max(float(lon1), float(lon2)), 6),
        round(max(float(lat1), float(lat2)), 6),
    ]


# ============================================================
# SPATIAL COHERENCE METRIC
# ============================================================

def calculate_spatial_coherence(
    region_mask: np.ndarray,
    pixel_count: int,
    evidence_values: Optional[np.ndarray] = None,
) -> float:
    """
    Calculates deterministic spatial coherence bounded in [0.0, 1.0].
    Quantifies geometric compactness, contiguity, and internal evidence consistency.
    This is NOT a probabilistic confidence.
    """
    if pixel_count <= 0:
        return 0.0

    # 1. Compactness (isoperimetric quotient: 4 * pi * Area / Perimeter^2)
    contours, _ = cv2.findContours(
        region_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        perimeter = float(cv2.arcLength(contours[0], closed=True))
        if perimeter > 0:
            # Circle = 1.0, elongated or noisy shapes have much lower values
            compactness = min(1.0, (4.0 * math.pi * float(pixel_count)) / (perimeter ** 2))
        else:
            compactness = 0.5
    else:
        compactness = 0.5

    # 2. Size regularity factor (asymptotic scale towards 50 pixels)
    size_factor = min(1.0, float(pixel_count) / 50.0)

    # 3. Evidence consistency (standard deviation penalty if evidence array provided)
    if evidence_values is not None and len(evidence_values) > 1:
        finite_vals = evidence_values[np.isfinite(evidence_values)]
        if len(finite_vals) > 1:
            val_std = float(np.std(finite_vals))
            consistency = max(0.0, 1.0 - val_std * 2.0)
        else:
            consistency = 0.8
    else:
        consistency = 0.8

    # Blend deterministically
    coherence = 0.40 * compactness + 0.35 * size_factor + 0.25 * consistency
    return float(round(np.clip(coherence, 0.0, 1.0), 4))


# ============================================================
# RELATIVE LOCATION DESCRIPTION
# ============================================================

def describe_spatial_location(
    centroids: List[Tuple[float, float]],
    aoi_bounds: Optional[List[List[float]]] = None,
) -> str:
    """
    Generates simple, strictly geometric location description (e.g. 'eastern portion of the AOI')
    based solely on computed centroids relative to AOI bounds. Never invents city names or landmarks.
    """
    if not centroids:
        return "across the observation area"

    avg_lon = float(np.mean([c[0] for c in centroids]))
    avg_lat = float(np.mean([c[1] for c in centroids]))

    if not aoi_bounds or len(aoi_bounds) < 2:
        return "within the observation area"

    try:
        min_lat = float(min(aoi_bounds[0][0], aoi_bounds[1][0]))
        max_lat = float(max(aoi_bounds[0][0], aoi_bounds[1][0]))
        min_lon = float(min(aoi_bounds[0][1], aoi_bounds[1][1]))
        max_lon = float(max(aoi_bounds[0][1], aoi_bounds[1][1]))

        mid_lat = (min_lat + max_lat) / 2.0
        mid_lon = (min_lon + max_lon) / 2.0

        d_lat = max_lat - min_lat
        d_lon = max_lon - min_lon

        if d_lat <= 0 or d_lon <= 0:
            return "within the observation area"

        rel_y = (avg_lat - mid_lat) / (d_lat / 2.0)  # +1 = North, -1 = South
        rel_x = (avg_lon - mid_lon) / (d_lon / 2.0)  # +1 = East, -1 = West

        # Deadband for central
        if abs(rel_y) < 0.25 and abs(rel_x) < 0.25:
            return "concentrated in the central portion of the AOI"

        parts: List[str] = []
        if rel_y >= 0.25:
            parts.append("northern")
        elif rel_y <= -0.25:
            parts.append("southern")

        if rel_x >= 0.25:
            parts.append("eastern")
        elif rel_x <= -0.25:
            parts.append("western")

        direction = "-".join(parts) if parts else "central"
        return f"concentrated in the {direction} portion of the AOI"

    except Exception:
        return "within the observation area"


# ============================================================
# CONNECTED COMPONENT EXTRACTION & MMU FILTERING
# ============================================================

def extract_connected_candidate_regions(
    candidate_map: np.ndarray,
    transform: Affine,
    joint_mask: Optional[np.ndarray] = None,
    crs: Optional[Any] = None,
    mmu_min_pixels: int = SpatialConfig.MMU_MIN_PIXELS,
    connectivity: int = SpatialConfig.CONNECTIVITY,
    delta_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[List[CandidateRegion], np.ndarray, np.ndarray]:
    """
    Performs deterministic connected-component labeling on candidate classes.
    Applies MMU filtering to separate noise from coherent candidate clusters.

    Returns:
      (retained_regions, filtered_candidate_mask, labeled_region_raster)
    """
    shape = candidate_map.shape
    if joint_mask is None:
        joint_mask = np.ones(shape, dtype=bool)

    # Valid mask: invalid pixels must NEVER form candidate regions
    valid_candidates = np.where(joint_mask, candidate_map, 0).astype(np.uint8)

    filtered_mask = np.zeros(shape, dtype=np.uint8)
    labeled_regions = np.zeros(shape, dtype=np.int32)
    regions: List[CandidateRegion] = []
    current_region_id = 1

    delta_arrays = delta_arrays or {}
    ndvi_d = delta_arrays.get("ndvi")
    ndbi_d = delta_arrays.get("ndbi")
    ndwi_d = delta_arrays.get("ndwi")

    # Evaluate target classes: 1 (primary candidate), 2 (counter candidate)
    # Class 3 (uncertain) handled separately if policy is separate
    classes_to_extract = [1, 2]
    if SpatialConfig.UNCERTAIN_REGION_POLICY == "separate":
        classes_to_extract.append(3)

    label_names = {
        1: "primary_candidate",
        2: "counter_candidate",
        3: "uncertain",
    }

    for target_cls in classes_to_extract:
        class_binary = (valid_candidates == target_cls).astype(np.uint8)
        if not np.any(class_binary):
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            class_binary,
            connectivity=connectivity,
        )

        # Label 0 is background; iterate components 1..num_labels-1
        for comp_idx in range(1, num_labels):
            pixel_count = int(stats[comp_idx, cv2.CC_STAT_AREA])

            # Apply MMU filter
            if pixel_count < mmu_min_pixels:
                continue

            comp_mask = (labels == comp_idx)
            filtered_mask[comp_mask] = target_cls
            labeled_regions[comp_mask] = current_region_id

            min_col = int(stats[comp_idx, cv2.CC_STAT_LEFT])
            min_row = int(stats[comp_idx, cv2.CC_STAT_TOP])
            width = int(stats[comp_idx, cv2.CC_STAT_WIDTH])
            height = int(stats[comp_idx, cv2.CC_STAT_HEIGHT])

            c_col = float(centroids[comp_idx][0])
            c_row = float(centroids[comp_idx][1])
            c_lon, c_lat = pixel_to_geo(transform, c_col, c_row)

            # Accurate geodesic area
            px_area_m2 = calculate_pixel_area_m2(transform, c_lat, crs=crs)
            area_m2 = round(pixel_count * px_area_m2, 2)
            area_ha = round(area_m2 / 10000.0, 4)

            bbox_geo = compute_geographic_bbox(transform, min_col, min_row, width, height)

            # Sample scientific arrays
            mean_ndvi = round(float(np.mean(ndvi_d[comp_mask])), 4) if ndvi_d is not None and np.any(np.isfinite(ndvi_d[comp_mask])) else None
            mean_ndbi = round(float(np.mean(ndbi_d[comp_mask])), 4) if ndbi_d is not None and np.any(np.isfinite(ndbi_d[comp_mask])) else None
            mean_ndwi = round(float(np.mean(ndwi_d[comp_mask])), 4) if ndwi_d is not None and np.any(np.isfinite(ndwi_d[comp_mask])) else None

            # Primary score proxy
            if target_cls == 1 and mean_ndbi is not None:
                mean_score = abs(mean_ndbi)
            elif target_cls == 1 and mean_ndvi is not None:
                mean_score = abs(mean_ndvi)
            else:
                mean_score = 0.5

            # Spatial coherence
            coherence = calculate_spatial_coherence(
                region_mask=comp_mask,
                pixel_count=pixel_count,
            )

            # Valid pixel fraction (guaranteed 1.0 since comp_mask subset of valid_candidates)
            valid_frac = float(round(np.sum(joint_mask[comp_mask]) / max(1, pixel_count), 4))

            # Extract polygon geometry
            comp_shapes = list(rasterio.features.shapes(
                comp_mask.astype(np.int32),
                mask=comp_mask,
                transform=transform,
            ))
            geom = comp_shapes[0][0] if comp_shapes else None

            region = CandidateRegion(
                region_id=current_region_id,
                candidate_class=target_cls,
                class_label=label_names.get(target_cls, "candidate"),
                pixel_count=pixel_count,
                area_m2=area_m2,
                area_hectares=area_ha,
                centroid_pixel=[round(c_col, 2), round(c_row, 2)],
                centroid_geo=[round(c_lon, 6), round(c_lat, 6)],
                bbox_pixel=[min_col, min_row, width, height],
                bbox_geo=bbox_geo,
                spatial_coherence=coherence,
                valid_pixel_fraction=valid_frac,
                mean_candidate_score=round(mean_score, 4),
                max_candidate_score=round(mean_score, 4),
                mean_ndbi_delta=mean_ndbi,
                mean_ndvi_delta=mean_ndvi,
                mean_ndwi_delta=mean_ndwi,
                geometry=geom,
            )
            regions.append(region)
            current_region_id += 1

    return regions, filtered_mask, labeled_regions


# ============================================================
# VECTORIZATION TO GEOJSON
# ============================================================

def vectorize_candidate_regions(regions: List[CandidateRegion]) -> Dict[str, Any]:
    """
    Converts extracted candidate regions into a valid GeoJSON FeatureCollection.
    """
    features: List[Dict[str, Any]] = []

    for reg in regions:
        if not reg.geometry:
            continue

        props = {
            "region_id": reg.region_id,
            "candidate_class": reg.candidate_class,
            "class_label": reg.class_label,
            "pixel_count": reg.pixel_count,
            "area_m2": reg.area_m2,
            "area_hectares": reg.area_hectares,
            "centroid_lon": reg.centroid_geo[0],
            "centroid_lat": reg.centroid_geo[1],
            "bbox_geo": reg.bbox_geo,
            "spatial_coherence": reg.spatial_coherence,
            "valid_pixel_fraction": reg.valid_pixel_fraction,
            "mean_candidate_score": reg.mean_candidate_score,
            "mean_ndbi_delta": reg.mean_ndbi_delta,
            "mean_ndvi_delta": reg.mean_ndvi_delta,
            "mean_ndwi_delta": reg.mean_ndwi_delta,
        }

        features.append({
            "type": "Feature",
            "id": reg.region_id,
            "geometry": reg.geometry,
            "properties": props,
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


# ============================================================
# SPATIAL TRANSITION MATCHING
# ============================================================

def match_spatial_transitions(
    source_regions: List[CandidateRegion],
    dest_regions: List[CandidateRegion],
    transform: Affine,
    shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """
    Evaluates spatial co-location / border overlap between source and destination candidate regions.
    A candidate transition requires actual geometric overlap or adjacent contiguity.
    """
    transitions: List[Dict[str, Any]] = []
    if not source_regions or not dest_regions:
        return transitions

    for s_reg in source_regions:
        if s_reg.candidate_class != 1:  # Only primary source candidates (e.g. vegetation loss)
            continue
        s_mask = np.zeros(shape, dtype=np.uint8)
        # Create mask from bbox
        min_c, min_r, w, h = s_reg.bbox_pixel
        s_mask[min_r:min_r+h, min_c:min_c+w] = 1

        for d_reg in dest_regions:
            if d_reg.candidate_class != 1:  # Only primary destination candidates (e.g. urban expansion)
                continue
            d_mask = np.zeros(shape, dtype=np.uint8)
            d_min_c, d_min_r, dw, dh = d_reg.bbox_pixel
            d_mask[d_min_r:d_min_r+dh, d_min_c:d_min_c+dw] = 1

            # Check overlap or border dilation
            overlap = np.sum((s_mask & d_mask) > 0)
            if overlap > 0:
                overlap_frac = float(overlap / max(1, min(s_reg.pixel_count, d_reg.pixel_count)))
                if overlap_frac >= SpatialConfig.TRANSITION_OVERLAP_THRESHOLD:
                    t_score = round(min(s_reg.spatial_coherence, d_reg.spatial_coherence) * overlap_frac, 4)
                    transitions.append({
                        "transition_type": "vegetation_to_urban",
                        "source_region_id": s_reg.region_id,
                        "destination_region_id": d_reg.region_id,
                        "overlap_pixels": int(overlap),
                        "overlap_fraction": round(overlap_frac, 4),
                        "transition_score": t_score,
                        "status": "candidate_transition",
                    })

    return transitions


# ============================================================
# MAIN SPATIAL REASONING ORCHESTRATOR
# ============================================================

def extract_spatial_candidate_regions(
    candidate_raster_path: Optional[str],
    target: Optional[str] = None,
    task: Optional[str] = None,
    execution_results: Optional[Dict[str, Any]] = None,
    imagery_result: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    aoi_bounds: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint for Phase 6 Spatial Reasoning.
    Consumes candidate GeoTIFF, performs connected-component labeling,
    MMU filtering, spatial coherence calculation, vectorization, and returns
    the structured spatial_analysis package.
    """
    target_clean = (target or "").lower().strip()
    execution_results = execution_results or {}

    if not candidate_raster_path or not Path(candidate_raster_path).exists():
        return {
            "available": False,
            "target": target_clean,
            "region_count": 0,
            "filtered_pixel_count": 0,
            "total_candidate_area_m2": 0.0,
            "total_candidate_area_hectares": 0.0,
            "regions": [],
            "transitions": [],
            "summary": "No candidate raster available for spatial analysis.",
            "parameters": {
                "mmu_min_pixels": SpatialConfig.MMU_MIN_PIXELS,
                "connectivity": SpatialConfig.CONNECTIVITY,
            },
            "geojson": {"type": "FeatureCollection", "features": []},
        }

    try:
        # Read candidate raster and profile
        with rasterio.open(candidate_raster_path) as src:
            candidate_map = src.read(1)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            shape = (src.height, src.width)

        # Read joint mask
        joint_mask = np.ones(shape, dtype=bool)
        if imagery_result and len(imagery_result.get("images", [])) >= 2:
            m_b_path = imagery_result["images"][0].get("bands", {}).get("mask")
            m_a_path = imagery_result["images"][1].get("bands", {}).get("mask")
            if m_b_path and m_a_path and Path(m_b_path).exists() and Path(m_a_path).exists():
                with rasterio.open(m_b_path) as mb, rasterio.open(m_a_path) as ma:
                    joint_mask = mb.read(1).astype(bool) & ma.read(1).astype(bool)

        # Read original continuous difference arrays for statistics
        delta_arrays: Dict[str, np.ndarray] = {}
        dc_all = execution_results.get("detect_change", {}).get("all_changes", {})
        for key, name in [("calculate_temporal_ndvi", "ndvi"), ("calculate_temporal_ndbi", "ndbi"), ("calculate_temporal_ndwi", "ndwi")]:
            diff_p = execution_results.get(key, {}).get("difference_raster")
            if diff_p and Path(diff_p).exists():
                with rasterio.open(diff_p) as d_src:
                    delta_arrays[name] = d_src.read(1)
            elif name in dc_all and "change_map" in dc_all[name]:
                delta_arrays[name] = np.array(dc_all[name]["change_map"], dtype=np.float32)
            elif name.upper() in dc_all and "change_map" in dc_all[name.upper()]:
                delta_arrays[name] = np.array(dc_all[name.upper()]["change_map"], dtype=np.float32)

        # Connected component extraction & MMU filtering
        regions, filtered_mask, labeled_raster = extract_connected_candidate_regions(
            candidate_map=candidate_map,
            transform=transform,
            joint_mask=joint_mask,
            crs=crs,
            mmu_min_pixels=SpatialConfig.MMU_MIN_PIXELS,
            connectivity=SpatialConfig.CONNECTIVITY,
            delta_arrays=delta_arrays,
        )

        # Compute summary statistics
        primary_regions = [r for r in regions if r.candidate_class == 1]
        total_pixels = sum(r.pixel_count for r in primary_regions)
        total_area_m2 = sum(r.area_m2 for r in primary_regions)
        total_area_ha = round(total_area_m2 / 10000.0, 4)

        largest_region = max(primary_regions, key=lambda r: r.pixel_count).to_dict() if primary_regions else None

        # Relative location description
        centroids = [(r.centroid_geo[0], r.centroid_geo[1]) for r in primary_regions]
        location_desc = describe_spatial_location(centroids, aoi_bounds=aoi_bounds)

        # Vectorize to GeoJSON
        geojson_fc = vectorize_candidate_regions(regions)

        # Save filtered candidate raster & labeled region raster
        out_dir = Path(output_dir) if output_dir else Path(candidate_raster_path).parent
        filtered_raster_path = out_dir / f"candidate_{target_clean}_filtered.tif"
        labeled_raster_path = out_dir / f"regions_{target_clean}.tif"

        p_filt = profile.copy()
        p_filt.update(dtype=rasterio.uint8, count=1, nodata=0)
        with rasterio.open(filtered_raster_path, "w", **p_filt) as f_dst:
            f_dst.write(filtered_mask, 1)

        p_lbl = profile.copy()
        p_lbl.update(dtype=rasterio.int32, count=1, nodata=0)
        with rasterio.open(labeled_raster_path, "w", **p_lbl) as l_dst:
            l_dst.write(labeled_raster, 1)

        summary_text = (
            f"Identified {len(primary_regions)} candidate region(s) totaling {total_area_ha:.2f} hectares, {location_desc}."
            if primary_regions else
            "No spatially coherent candidate regions met the minimum mapping unit threshold."
        )

        return {
            "available": True,
            "target": target_clean,
            "region_count": len(primary_regions),
            "total_regions_all_classes": len(regions),
            "filtered_pixel_count": total_pixels,
            "total_candidate_area_m2": total_area_m2,
            "total_candidate_area_hectares": total_area_ha,
            "largest_region": largest_region,
            "dominant_location_description": location_desc,
            "summary": summary_text,
            "regions": [r.to_dict() for r in regions],
            "transitions": [],
            "parameters": {
                "mmu_min_pixels": SpatialConfig.MMU_MIN_PIXELS,
                "connectivity": SpatialConfig.CONNECTIVITY,
                "uncertain_region_policy": SpatialConfig.UNCERTAIN_REGION_POLICY,
            },
            "rasters": {
                "raw_candidate_raster": str(candidate_raster_path),
                "filtered_candidate_raster": str(filtered_raster_path),
                "labeled_regions_raster": str(labeled_raster_path),
            },
            "geojson": geojson_fc,
        }

    except Exception as exc:
        print(f"[SPATIAL ENGINE WARNING] Spatial analysis failed: {exc}")
        return {
            "available": False,
            "target": target_clean,
            "region_count": 0,
            "filtered_pixel_count": 0,
            "total_candidate_area_m2": 0.0,
            "total_candidate_area_hectares": 0.0,
            "regions": [],
            "transitions": [],
            "summary": f"Spatial analysis encountered an error: {exc}",
            "parameters": {
                "mmu_min_pixels": SpatialConfig.MMU_MIN_PIXELS,
                "connectivity": SpatialConfig.CONNECTIVITY,
            },
            "geojson": {"type": "FeatureCollection", "features": []},
        }
