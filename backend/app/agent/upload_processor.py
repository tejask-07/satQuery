"""
Upload analysis processor for user-provided satellite and aerial imagery.

Implements the complete P2 (validation, band identification, index/change calculation,
statistics, visualization) to P4 (VLM grounding, BigEarthNet retrieval) handoff.

Follows strict remote-sensing scientific rules:
- True NDVI requires Red + NIR bands.
- True NDWI requires Green + NIR bands.
- True NDBI requires NIR + SWIR bands.
- Standard 3-band RGB uploads never invent or fabricate missing spectral bands.
"""

import io
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds

from app.agent.executor import (
    VISUALIZATION_DIR,
    _get_raster_bounds,
    _save_change_map_visualization,
)
from app.schemas.analysis import AnalysisResult
from app.schemas.query import QueryPlan
from app.tools.change import detect_change
from app.tools.indices import (
    calculate_temporal_ndbi,
    calculate_temporal_ndvi,
    calculate_temporal_ndwi,
)
from app.vlm.bigearthnet.s1_p4 import build_s1_visualization
from app.vlm.model import VLM

logger = logging.getLogger(__name__)


def _to_pil_rgb(img_array: np.ndarray) -> Image.Image:
    """Safely convert a 2D or 3D numpy array to an RGB PIL Image."""
    if img_array.ndim == 2:
        norm = _normalize_display_channel(img_array)
        return Image.fromarray(norm, mode="L").convert("RGB")
    elif img_array.ndim == 3:
        if img_array.shape[0] in (3, 4) and img_array.shape[2] not in (3, 4):
            # CHW -> HWC
            img_array = np.transpose(img_array, (1, 2, 0))
        h, w, c = img_array.shape
        rgb_channels = []
        for i in range(min(3, c)):
            rgb_channels.append(_normalize_display_channel(img_array[:, :, i]))
        while len(rgb_channels) < 3:
            rgb_channels.append(rgb_channels[0])
        stacked = np.stack(rgb_channels[:3], axis=-1)
        return Image.fromarray(stacked, mode="RGB")
    return Image.new("RGB", (256, 256), color=(128, 128, 128))


def _normalize_display_channel(channel: np.ndarray) -> np.ndarray:
    """Normalize a floating point or integer raster band to uint8 [0, 255]."""
    arr = np.asarray(channel, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)

    vals = arr[valid]
    low = float(np.percentile(vals, 2))
    high = float(np.percentile(vals, 98))

    if high <= low:
        low = float(np.min(vals))
        high = float(np.max(vals))

    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
    scaled[~valid] = 0.0
    return (scaled * 255.0).astype(np.uint8)


def inspect_image(
    file_bytes: bytes, filename: str
) -> Dict[str, Any]:
    """
    Inspect an uploaded image file, detecting whether it is a multispectral GeoTIFF
    or a standard visible RGB image (JPG/PNG).

    Returns:
        Dictionary containing metadata, extracted bands, dimensions, and display image.
    """
    is_geotiff = False
    is_multispectral = False
    num_bands = 0
    width = 0
    height = 0
    crs = None
    bounds = None
    band_data: Dict[str, np.ndarray] = {}
    pil_image: Optional[Image.Image] = None

    # First attempt reading with rasterio to check for GeoTIFF / geospatial metadata
    try:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                driver = src.driver or ""
                num_bands = src.count
                width = src.width
                height = src.height
                crs_obj = src.crs
                crs = str(crs_obj) if crs_obj else None

                if driver in ("GTiff", "VRT") or crs_obj is not None or num_bands > 3:
                    is_geotiff = True

                # Determine geographic bounds
                if crs_obj and src.bounds:
                    try:
                        if crs_obj.to_string() == "EPSG:4326":
                            b = src.bounds
                            bounds = [[float(b.bottom), float(b.left)], [float(b.top), float(b.right)]]
                        else:
                            w, s, e, n = transform_bounds(crs_obj, "EPSG:4326", *src.bounds)
                            bounds = [[float(s), float(w)], [float(n), float(e)]]
                    except Exception as b_err:
                        logger.warning(f"Could not reproject GeoTIFF bounds: {b_err}")

                descriptions = [
                    (src.descriptions[i] or "").lower().strip()
                    for i in range(num_bands)
                ]

                # Map bands from descriptions or tags
                b_map: Dict[str, int] = {}
                for idx, desc in enumerate(descriptions, start=1):
                    if "red" in desc or desc in ("b4", "b04"):
                        b_map["red"] = idx
                    elif "green" in desc or desc in ("b3", "b03"):
                        b_map["green"] = idx
                    elif "blue" in desc or desc in ("b2", "b02"):
                        b_map["blue"] = idx
                    elif "nir" in desc or desc in ("b8", "b08", "b8a"):
                        b_map["nir"] = idx
                    elif "swir" in desc or desc in ("b11", "b12", "swir1", "swir2"):
                        b_map["swir"] = idx

                # Heuristic mapping if band descriptions are missing
                if num_bands == 4 and "nir" not in b_map:
                    # Standard 4-band package: Red, Green, Blue, NIR
                    b_map = {"red": 1, "green": 2, "blue": 3, "nir": 4}
                elif num_bands == 2 and not b_map:
                    # 2-band pair (e.g. Red + NIR or SWIR + NIR)
                    b_map = {"red": 1, "nir": 2, "swir": 1, "green": 1}
                elif num_bands >= 5 and "nir" not in b_map:
                    # Sentinel-2 style 12-band stack
                    b_map = {"blue": 2, "green": 3, "red": 4, "nir": 8, "swir": 11}
                elif num_bands == 3 and not b_map:
                    b_map = {"red": 1, "green": 2, "blue": 3}

                # Read identified bands
                for b_name, b_idx in b_map.items():
                    if b_idx <= num_bands:
                        band_data[b_name] = src.read(b_idx).astype(np.float32)

                if "nir" in band_data or "swir" in band_data:
                    is_multispectral = True

                # Generate display image
                if "red" in band_data and "green" in band_data and "blue" in band_data:
                    r = _normalize_display_channel(band_data["red"])
                    g = _normalize_display_channel(band_data["green"])
                    b = _normalize_display_channel(band_data["blue"])
                    pil_image = Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")
                elif "red" in band_data and "green" in band_data and "nir" in band_data:
                    # False color RGB (Red, Green, NIR)
                    r = _normalize_display_channel(band_data["red"])
                    g = _normalize_display_channel(band_data["green"])
                    b = _normalize_display_channel(band_data["nir"])
                    pil_image = Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")
                elif num_bands >= 3:
                    raw_rgb = src.read([1, 2, 3])
                    pil_image = _to_pil_rgb(raw_rgb)
                else:
                    raw_single = src.read(1)
                    pil_image = _to_pil_rgb(raw_single)

    except Exception as exc:
        logger.info(f"File {filename} is not a valid GeoTIFF ({exc}). Falling back to PIL.")

    # Fallback to PIL for standard PNG/JPG if rasterio didn't produce an image
    if pil_image is None:
        try:
            pil_raw = Image.open(io.BytesIO(file_bytes))
            pil_image = pil_raw.convert("RGB")
            width, height = pil_image.size
            num_bands = 3
            is_geotiff = False
            is_multispectral = False

            # Extract visible channels
            np_rgb = np.array(pil_image, dtype=np.float32)
            band_data["red"] = np_rgb[:, :, 0]
            band_data["green"] = np_rgb[:, :, 1]
            band_data["blue"] = np_rgb[:, :, 2]
        except Exception as pil_err:
            raise ValueError(f"Could not open image '{filename}': {pil_err}")

    return {
        "filename": filename,
        "is_geotiff": is_geotiff,
        "is_multispectral": is_multispectral,
        "num_bands": num_bands,
        "width": width,
        "height": height,
        "crs": crs,
        "bounds": bounds,
        "bands": band_data,
        "pil_image": pil_image,
    }


def parse_upload_query(query: str) -> Dict[str, Any]:
    """Parse user query to identify target, task, and requested index."""
    q_lower = (query or "").lower().strip()

    if any(w in q_lower for w in ("vegetation", "ndvi", "forest", "crop", "tree", "plant", "green")):
        return {
            "task": "vegetation_change",
            "target": "vegetation",
            "requested_metric": "NDVI",
        }
    elif any(w in q_lower for w in ("water", "ndwi", "flood", "lake", "river", "reservoir", "wetland")):
        return {
            "task": "water_change",
            "target": "water",
            "requested_metric": "NDWI",
        }
    elif any(w in q_lower for w in ("urban", "ndbi", "building", "city", "settlement", "built-up", "development")):
        return {
            "task": "urban_change",
            "target": "urban",
            "requested_metric": "NDBI",
        }
    else:
        return {
            "task": "image_comparison",
            "target": "imagery",
            "requested_metric": "visual_change",
        }


def process_upload_analysis(
    before_bytes: bytes,
    after_bytes: bytes,
    before_name: str = "before_image",
    after_name: str = "after_image",
    query: str = "Show change",
    threshold: Optional[float] = None,
) -> AnalysisResult:
    """
    Execute full P2 analysis on uploaded images and hand off to P4 VLM.

    Args:
        before_bytes: Raw bytes of the before image.
        after_bytes: Raw bytes of the after image.
        before_name: Original filename of the before image.
        after_name: Original filename of the after image.
        query: User investigation query.
        threshold: Optional custom change threshold.

    Returns:
        AnalysisResult matching the unified SatQuery response contract.
    """
    if not before_bytes or not after_bytes:
        raise ValueError("Two non-empty images (before and after) are required for change analysis.")

    # 1. Inspect both uploaded images
    info_before = inspect_image(before_bytes, before_name)
    info_after = inspect_image(after_bytes, after_name)

    # Align spatial dimensions if they differ
    w_b, h_b = info_before["width"], info_before["height"]
    w_a, h_a = info_after["width"], info_after["height"]

    if (w_b, h_b) != (w_a, h_a):
        # Resize after bands to match before dimensions
        for b_name, b_arr in info_after["bands"].items():
            info_after["bands"][b_name] = cv2.resize(
                b_arr, (w_b, h_b), interpolation=cv2.INTER_LINEAR
            )
        info_after["pil_image"] = info_after["pil_image"].resize(
            (w_b, h_b), Image.Resampling.BILINEAR
        )
        info_after["width"], info_after["height"] = w_b, h_b

    # 2. Parse query intent
    plan_info = parse_upload_query(query)
    task = plan_info["task"]
    target = plan_info["target"]
    requested_metric = plan_info["requested_metric"]

    # 3. Check spectral capability and calculate index or visual change
    bands_b = info_before["bands"]
    bands_a = info_after["bands"]

    is_multispectral_pair = (
        info_before["is_multispectral"] and info_after["is_multispectral"]
    )

    actual_metric: Optional[str] = None
    spectral_warning: Optional[str] = None
    change_result: Dict[str, Any] = {}
    change_map_arr: Optional[np.ndarray] = None

    if requested_metric == "NDVI":
        if "red" in bands_b and "nir" in bands_b and "red" in bands_a and "nir" in bands_a:
            calc = calculate_temporal_ndvi(
                red_before=bands_b["red"],
                nir_before=bands_b["nir"],
                red_after=bands_a["red"],
                nir_after=bands_a["nir"],
            )
            th = threshold if threshold is not None else 0.05
            change_result = detect_change(
                before=calc["ndvi_before"],
                after=calc["ndvi_after"],
                threshold=th,
            )
            actual_metric = "NDVI"
            change_map_arr = change_result.get("change_map")
        else:
            spectral_warning = (
                "The uploaded images lack Near-Infrared (NIR) bands required for quantitative "
                "NDVI calculation. Standard RGB imagery cannot compute true NDVI without fabricating bands. "
                "Performing visual change analysis across the visible spectrum."
            )

    elif requested_metric == "NDWI":
        if "green" in bands_b and "nir" in bands_b and "green" in bands_a and "nir" in bands_a:
            calc = calculate_temporal_ndwi(
                green_before=bands_b["green"],
                nir_before=bands_b["nir"],
                green_after=bands_a["green"],
                nir_after=bands_a["nir"],
            )
            th = threshold if threshold is not None else 0.05
            change_result = detect_change(
                before=calc["ndwi_before"],
                after=calc["ndwi_after"],
                threshold=th,
            )
            actual_metric = "NDWI"
            change_map_arr = change_result.get("change_map")
        else:
            spectral_warning = (
                "The uploaded images lack required spectral bands for quantitative NDWI calculation. "
                "Performing visual change analysis across the visible spectrum."
            )

    elif requested_metric == "NDBI":
        if "swir" in bands_b and "nir" in bands_b and "swir" in bands_a and "nir" in bands_a:
            calc = calculate_temporal_ndbi(
                swir_before=bands_b["swir"],
                nir_before=bands_b["nir"],
                swir_after=bands_a["swir"],
                nir_after=bands_a["nir"],
            )
            th = threshold if threshold is not None else 0.05
            change_result = detect_change(
                before=calc["ndbi_before"],
                after=calc["ndbi_after"],
                threshold=th,
            )
            actual_metric = "NDBI"
            change_map_arr = change_result.get("change_map")
        else:
            spectral_warning = (
                "The uploaded images lack SWIR/NIR bands required for quantitative NDBI calculation. "
                "Performing visual change analysis across the visible spectrum."
            )

    # Fallback to visual comparison if spectral index unavailable or requested
    if actual_metric is None:
        actual_metric = "visual_change"
        # Compute normalized luminance change: (-1.0 to +1.0)
        lum_b = (
            0.299 * bands_b.get("red", 0)
            + 0.587 * bands_b.get("green", 0)
            + 0.114 * bands_b.get("blue", 0)
        ) / 255.0
        lum_a = (
            0.299 * bands_a.get("red", 0)
            + 0.587 * bands_a.get("green", 0)
            + 0.114 * bands_a.get("blue", 0)
        ) / 255.0

        th = threshold if threshold is not None else 0.08
        change_result = detect_change(
            before=lum_b,
            after=lum_a,
            threshold=th,
        )
        change_map_arr = change_result.get("change_map")

    # 4. Generate visualization PNGs and save to VISUALIZATION_DIR
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    vis_url: Optional[str] = None
    classified_vis_url: Optional[str] = None
    bounds = info_before["bounds"] or info_after["bounds"]

    if change_map_arr is not None:
        try:
            vis_meta = _save_change_map_visualization(
                change_map=change_map_arr,
                prefix=f"upload_{actual_metric.lower()}_{uid}",
                threshold=change_result.get("threshold"),
            )
            vis_url = f"/visualizations/{vis_meta['filename']}"
            classified_vis_url = f"/visualizations/{vis_meta['classified_filename']}"
            if not bounds and vis_meta.get("bounds"):
                bounds = vis_meta["bounds"]
        except Exception as vis_err:
            logger.warning(f"Failed to generate change visualization: {vis_err}")

    # Save display copies of before and after images for UI rendering
    before_img_filename = f"upload_before_{uid}_{timestamp}.png"
    after_img_filename = f"upload_after_{uid}_{timestamp}.png"
    info_before["pil_image"].save(VISUALIZATION_DIR / before_img_filename)
    info_after["pil_image"].save(VISUALIZATION_DIR / after_img_filename)

    before_img_url = f"/visualizations/{before_img_filename}"
    after_img_url = f"/visualizations/{after_img_filename}"

    # 5. Assemble authoritative statistics and layers
    stats = {
        "metric": actual_metric,
        "mean_before": change_result.get("mean_before"),
        "mean_after": change_result.get("mean_after"),
        "mean_change": change_result.get("mean_change"),
        "changed_pixels": change_result.get("changed_pixels"),
        "valid_pixels": change_result.get("valid_pixels"),
        "total_pixels": change_result.get("total_pixels"),
        "change_ratio": change_result.get("change_ratio"),
        "increased_pixels": change_result.get("increased_pixels"),
        "decreased_pixels": change_result.get("decreased_pixels"),
        "change_type": change_result.get("change_type"),
        "threshold": change_result.get("threshold"),
    }
    if spectral_warning:
        stats["spectral_warning"] = spectral_warning

    layer_item = {
        "type": "change_detection",
        "name": f"{actual_metric.upper()} change map",
        "change_ratio": change_result.get("change_ratio"),
        "changed_pixels": change_result.get("changed_pixels"),
        "visualization_url": vis_url,
        "classified_visualization_url": classified_vis_url,
        "bounds": bounds,
    }

    evidence_item = {
        "source": "user_upload",
        "message": (
            f"Analyzed uploaded imagery for {actual_metric}. "
            + (spectral_warning or "Multispectral bands successfully utilized.")
        ),
        "bands_available": {
            "before": list(bands_b.keys()),
            "after": list(bands_a.keys()),
        },
    }

    plan_dict = {
        "task": task,
        "target": target,
        "metric": actual_metric,
        "spectral_capability": "multispectral" if is_multispectral_pair else "rgb_only",
    }

    # 6. Prepare P4 VLM input
    # Format structured evidence for VLM
    evidence_text_lines = [
        f"Task: {task}",
        f"Target: {target}",
        f"Metric: {actual_metric}",
        f"Mean Before: {stats['mean_before']}",
        f"Mean After: {stats['mean_after']}",
        f"Mean Change: {stats['mean_change']}",
        f"Changed Pixels: {stats['changed_pixels']} / {stats['valid_pixels']}",
        f"Change Ratio: {stats['change_ratio'] * 100:.2f}%" if stats.get('change_ratio') is not None else "Change Ratio: N/A",
        f"Change Direction: {stats['change_type']}",
        f"Threshold: {stats['threshold']}",
    ]
    if spectral_warning:
        evidence_text_lines.append(f"SPECTRAL NOTICE: {spectral_warning}")
        evidence_text_lines.append(
            "CRITICAL: Do NOT invent or claim quantitative NDVI/NDWI/NDBI values. "
            "Explain that the uploaded images are 3-band visible RGB, and report visual change observations."
        )

    evidence_text = "\n".join(evidence_text_lines)

    vlm_images: Dict[str, Image.Image] = {
        "before": info_before["pil_image"],
        "after": info_after["pil_image"],
    }

    # If change map PNG exists, attach to VLM images
    if vis_url:
        change_map_filename = vis_url.split("/")[-1]
        cm_path = VISUALIZATION_DIR / change_map_filename
        if cm_path.exists():
            vlm_images["change_map"] = Image.open(cm_path).convert("RGB")

    # Optionally attach demo Sentinel-1 composite if available for radar context
    try:
        s1_comp = build_s1_visualization("S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57")
        if s1_comp is not None:
            vlm_images["s1_composite"] = s1_comp
    except Exception:
        pass

    # 7. Call P4 VLM (with graceful fallback to backend explanation)
    vlm_answer = None
    execution_trace = [
        "Uploaded images validated and inspected",
        f"Identified bands: Before={list(bands_b.keys())}, After={list(bands_a.keys())}",
        f"Computed temporal {actual_metric} change detection",
        "Generated change map visualizations",
    ]

    try:
        vlm = VLM()
        vlm_answer = vlm.generate(
            question=query,
            evidence=evidence_text,
            images=vlm_images,
        )
        if vlm_answer:
            execution_trace.append("P4 VLM multimodal inference completed")
    except Exception as vlm_err:
        logger.info(f"P4 VLM skipped/unavailable: {vlm_err}")
        execution_trace.append(f"P4 VLM unavailable: {vlm_err}")

    # Fallback explanation if VLM answer is None
    if not vlm_answer:
        c_ratio = float(stats.get("change_ratio") or 0.0)
        c_ratio_pct = c_ratio * 100.0
        c_pixels = int(stats.get("changed_pixels") or 0)
        mean_chg = stats.get("mean_change")
        mean_chg_str = f"{float(mean_chg):+.4f}" if mean_chg is not None else "N/A"
        mean_bef = stats.get("mean_before")
        mean_bef_str = f"{float(mean_bef):.4f}" if mean_bef is not None else "N/A"
        mean_aft = stats.get("mean_after")
        mean_aft_str = f"{float(mean_aft):.4f}" if mean_aft is not None else "N/A"
        thresh = stats.get("threshold", 0.05)

        if spectral_warning:
            vlm_answer = (
                f"Analysis completed for uploaded imagery. {spectral_warning} "
                f"Visual change was detected across {c_ratio_pct:.1f}% of pixels "
                f"({c_pixels} changed pixels). "
                f"Mean visual intensity change: {mean_chg_str}."
            )
        else:
            vlm_answer = (
                f"Temporal {actual_metric} analysis of uploaded imagery completed. "
                f"Mean {actual_metric} changed from {mean_bef_str} to {mean_aft_str} "
                f"(net change: {mean_chg_str}). "
                f"Detected change across {c_ratio_pct:.1f}% of the scene "
                f"({c_pixels} changed pixels with threshold {thresh})."
            )

    return AnalysisResult(
        status="success",
        answer=vlm_answer,
        confidence=0.9 if not spectral_warning else 0.8,
        plan=plan_dict,
        statistics=stats,
        layers=[layer_item],
        evidence=[evidence_item],
        execution_trace=execution_trace,
        visualization_url=vis_url,
        classified_visualization_url=classified_vis_url,
        bounds=bounds,
        images={
            "before": before_img_url,
            "after": after_img_url,
            "change_map": vis_url,
        },
    )
