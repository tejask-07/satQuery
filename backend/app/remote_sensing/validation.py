"""
Centralized Input Validation Engine for Remote-Sensing Workflows.

Provides strict, workflow-aware validation for:
- File formats and magic bytes (.tif, .tiff, .png, .jpg, .jpeg)
- Image integrity and corrupt image detection (truncated TIFF/PNG/JPEG, random bytes)
- GeoTIFF metadata presence (CRS, bounds, transform, bands)
- Paired raster compatibility (temporal before/after, optical-SAR)
- Query and input configuration alignment
"""

from __future__ import annotations

import io
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds

from app.schemas.validation import ValidationErrorCode, ValidationResult

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Magic byte signatures
TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def get_file_extension(filename: str) -> str:
    """Return lowercase file extension with leading dot."""
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def validate_file_format(
    filename: str,
    file_bytes: Optional[bytes] = None,
    allowed_extensions: Optional[Set[str]] = None,
) -> ValidationResult:
    """
    Validate that filename has an approved image extension and non-contradictory magic bytes.

    Rejects unsupported file types (e.g. .pdf, .txt, .docx, .exe, .zip) with
    structured error details.
    """
    allowed = allowed_extensions or SUPPORTED_IMAGE_EXTENSIONS
    ext = get_file_extension(filename)

    if not ext or ext not in allowed:
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.UNSUPPORTED_FILE_TYPE,
            message=f"Unsupported image format: '{ext or 'unknown'}'. Uploaded file '{filename}' is not a valid GeoTIFF/raster.",
            details={
                "received": ext or "none",
                "supported": sorted(list(allowed)),
                "filename": filename,
            },
        )

    # Magic byte verification if bytes are provided
    if file_bytes and len(file_bytes) >= 4:
        header = file_bytes[:8]
        is_tiff = any(file_bytes.startswith(m) for m in TIFF_MAGIC)
        is_png = file_bytes.startswith(PNG_MAGIC[:4])
        is_jpeg = file_bytes.startswith(b"\xff\xd8")

        # Explicit non-image magic signatures to reject immediately
        if header.startswith(b"%PDF") or header.startswith(b"PK\x03\x04") or header.startswith(b"MZ"):
            detected = ".pdf" if header.startswith(b"%PDF") else (".zip" if header.startswith(b"PK") else ".exe")
            return ValidationResult(
                valid=False,
                error_code=ValidationErrorCode.UNSUPPORTED_FILE_TYPE,
                message=f"File '{filename}' contains {detected} data, not supported image content.",
                details={
                    "received": detected,
                    "supported": sorted(list(allowed)),
                    "filename": filename,
                },
            )

        # Disguised extension checks (e.g. random text or binary in .tif/.png)
        if ext in (".tif", ".tiff") and not is_tiff:
            # Let rasterio/PIL have a say, but if completely non-image:
            if not is_png and not is_jpeg and all(b >= 32 and b <= 126 for b in file_bytes[:min(64, len(file_bytes))]):
                return ValidationResult(
                    valid=False,
                    error_code=ValidationErrorCode.UNSUPPORTED_FILE_TYPE,
                    message=f"Uploaded file '{filename}' contains plain text, not a valid raster.",
                    details={
                        "received": ".txt",
                        "supported": sorted(list(allowed)),
                        "filename": filename,
                    },
                )

    return ValidationResult(valid=True)


def validate_image_bytes(
    file_bytes: bytes,
    filename: str,
    workflow: str = "general",
) -> ValidationResult:
    """
    Validate raster readability, dimension sanity, and non-corruption.

    Does not allow image decoding exceptions to become unhandled 500 errors.
    """
    if not file_bytes or len(file_bytes) == 0:
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message=f"Uploaded file '{filename}' is empty (0 bytes).",
            details={"filename": filename, "size": 0},
        )

    # 1. First attempt reading with rasterio
    rasterio_exc = None
    try:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                width = src.width
                height = src.height
                count = src.count

                if width <= 0 or height <= 0:
                    return ValidationResult(
                        valid=False,
                        error_code=ValidationErrorCode.INVALID_IMAGE,
                        message=f"Image '{filename}' has invalid pixel dimensions ({width}x{height}).",
                        details={"filename": filename, "width": width, "height": height},
                    )

                if count < 1:
                    return ValidationResult(
                        valid=False,
                        error_code=ValidationErrorCode.INVALID_IMAGE,
                        message=f"Image '{filename}' contains no raster bands (count=0).",
                        details={"filename": filename, "band_count": 0},
                    )

                # Attempt decoding a small block of raster data to verify stream integrity
                try:
                    _ = src.read(1, window=((0, min(height, 8)), (0, min(width, 8))))
                except Exception as read_err:
                    return ValidationResult(
                        valid=False,
                        error_code=ValidationErrorCode.INVALID_IMAGE,
                        message=f"Corrupt or truncated raster data in '{filename}': {read_err}",
                        details={"filename": filename, "error": str(read_err)},
                    )

                # Workflow-aware CRS checking
                if workflow in ("geospatial", "temporal_geospatial", "optical_sar"):
                    if src.crs is None:
                        return ValidationResult(
                            valid=False,
                            error_code=ValidationErrorCode.MISSING_METADATA,
                            message=f"Image '{filename}' lacks Coordinate Reference System (CRS) required for {workflow}.",
                            details={"filename": filename, "missing": "crs", "workflow": workflow},
                        )

                return ValidationResult(
                    valid=True,
                    details={
                        "width": width,
                        "height": height,
                        "count": count,
                        "crs": str(src.crs) if src.crs else None,
                        "driver": src.driver,
                    },
                )
    except Exception as exc:
        rasterio_exc = exc

    # 2. Fallback to PIL for standard PNG/JPEG
    try:
        with Image.open(io.BytesIO(file_bytes)) as pil_img:
            # Force decoding pixel buffer to detect truncated files or corrupted chunks
            pil_img.load()
            w, h = pil_img.size
            if w <= 0 or h <= 0:
                return ValidationResult(
                    valid=False,
                    error_code=ValidationErrorCode.INVALID_IMAGE,
                    message=f"Image '{filename}' has invalid dimensions ({w}x{h}).",
                    details={"filename": filename, "width": w, "height": h},
                )

            # Check if workflow strictly requires georeferencing
            if workflow in ("geospatial", "temporal_geospatial", "optical_sar"):
                return ValidationResult(
                    valid=False,
                    error_code=ValidationErrorCode.MISSING_METADATA,
                    message=f"Image '{filename}' is a standard non-georeferenced format, but {workflow} requires GeoTIFF metadata.",
                    details={"filename": filename, "missing": "georeferencing", "workflow": workflow},
                )

            return ValidationResult(
                valid=True,
                details={"width": w, "height": h, "mode": pil_img.mode, "format": pil_img.format},
            )
    except Exception as pil_exc:
        # Both rasterio and PIL failed: Genuine corrupt/unreadable file
        reason = str(rasterio_exc) if rasterio_exc else str(pil_exc)
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.INVALID_IMAGE,
            message=f"Could not open image '{filename}': corrupt or unreadable image file: {reason}",
            details={"filename": filename, "error": reason},
        )



def validate_temporal_pair(
    before_bytes: Optional[bytes],
    after_bytes: Optional[bytes],
    before_name: str = "before_image",
    after_name: str = "after_image",
    query: str = "",
    require_crs: bool = False,
    workflow: str = "temporal",
) -> ValidationResult:
    """
    Validate that two uploaded images form a compatible temporal pair.

    Checks:
    - Both exist and non-empty
    - Readable and non-corrupt
    - Georeferenced/CRS present if require_crs is True
    - Compatible dimensions or supported spatial overlap
    - Compatible modalities (rejects optical + SAR for standard temporal change)
    """
    # 1. Missing image checks
    if not before_bytes and not after_bytes:
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Temporal analysis requires both before and after images; neither was provided.",
            details={"missing": ["before_image", "after_image"]},
        )
    if not before_bytes:
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Temporal analysis requires both before and after images; missing before image.",
            details={"missing": ["before_image"]},
        )
    if not after_bytes:
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Temporal analysis requires both before and after images; missing after image.",
            details={"missing": ["after_image"]},
        )

    # 2. Format checks
    fmt_b = validate_file_format(before_name, before_bytes)
    if not fmt_b.valid:
        return fmt_b
    fmt_a = validate_file_format(after_name, after_bytes)
    if not fmt_a.valid:
        return fmt_a

    # 3. Integrity checks
    sub_wf = "temporal_geospatial" if (require_crs or workflow == "temporal_geospatial") else "temporal"
    int_b = validate_image_bytes(before_bytes, before_name, workflow=sub_wf)
    if not int_b.valid:
        return int_b
    int_a = validate_image_bytes(after_bytes, after_name, workflow=sub_wf)
    if not int_a.valid:
        return int_a

    details_b = int_b.details or {}
    details_a = int_a.details or {}

    w_b, h_b = details_b.get("width", 0), details_b.get("height", 0)
    w_a, h_a = details_a.get("width", 0), details_a.get("height", 0)

    # 4. Dimension & Spatial Compatibility
    crs_b = details_b.get("crs")
    crs_a = details_a.get("crs")

    if (require_crs or workflow == "temporal_geospatial") and (not crs_b or not crs_a):
        missing_target = before_name if not crs_b else after_name
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_METADATA,
            message=f"Temporal geospatial operation requires georeferencing/CRS; '{missing_target}' lacks CRS.",
            details={"missing": "crs", "workflow": workflow},
        )

    # If both have geospatial CRS, inspect spatial overlap regardless of pixel dimension equality
    if crs_b and crs_a:
        try:
            with MemoryFile(before_bytes) as mb, MemoryFile(after_bytes) as ma:
                with mb.open() as sb, ma.open() as sa:
                    b_crs = CRS.from_user_input(sb.crs)
                    a_crs = CRS.from_user_input(sa.crs)
                    bb = sb.bounds
                    ab = sa.bounds

                    if b_crs == a_crs:
                        left = max(bb.left, ab.left)
                        bottom = max(bb.bottom, ab.bottom)
                        right = min(bb.right, ab.right)
                        top = min(bb.top, ab.top)
                        overlap = (right > left) and (top > bottom)
                    else:
                        ab_in_b = transform_bounds(a_crs, b_crs, ab.left, ab.bottom, ab.right, ab.top)
                        left = max(bb.left, ab_in_b[0])
                        bottom = max(bb.bottom, ab_in_b[1])
                        right = min(bb.right, ab_in_b[2])
                        top = min(bb.top, ab_in_b[3])
                        overlap = (right > left) and (top > bottom)

                    if not overlap:
                        return ValidationResult(
                            valid=False,
                            error_code=ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR,
                            message=f"Before and after images have zero geographic overlap.",
                            details={
                                "before": before_name,
                                "after": after_name,
                                "reason": "no_spatial_overlap",
                            },
                        )
        except Exception as geo_err:
            logger.warning(f"Spatial overlap inspection failed: {geo_err}")

    # Exact dimension match is always acceptable
    if (w_b, h_b) != (w_a, h_a):
        # If non-georeferenced standard images with drastically different scales/aspect ratios
        ratio_w = w_b / max(w_a, 1)
        ratio_h = h_b / max(h_a, 1)
        aspect_b = w_b / max(h_b, 1)
        aspect_a = w_a / max(h_a, 1)

        if ratio_w > 2.5 or ratio_w < 0.4 or ratio_h > 2.5 or ratio_h < 0.4 or abs(aspect_b - aspect_a) > 0.4:
            return ValidationResult(
                valid=False,
                error_code=ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR,
                message=(
                    f"The before and after images have incompatible dimensions ({w_b}x{h_b} vs {w_a}x{h_a}) "
                    f"and cannot be aligned for temporal change analysis."
                ),
                details={
                    "before_dimensions": f"{w_b}x{h_b}",
                    "after_dimensions": f"{w_a}x{h_a}",
                    "reason": "dimension_mismatch",
                },
            )

    # 5. Modality Consistency Check
    # Reject cross-modality comparison (optical vs SAR) as standard temporal engine cannot compare them
    mod_b = _inspect_stream_modality(before_bytes)
    mod_a = _inspect_stream_modality(after_bytes)
    if (mod_b == "optical" and mod_a == "sar") or (mod_b == "sar" and mod_a == "optical"):
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR,
            message=(
                f"Temporal change detection requires matching sensor modalities; cannot perform standard "
                f"temporal change between an {mod_b} image ('{before_name}') and a {mod_a} image ('{after_name}')."
            ),
            details={
                "before_modality": mod_b,
                "after_modality": mod_a,
                "reason": "sensor_modality_mismatch",
            },
        )

    return ValidationResult(
        valid=True,
        details={
            "before": {"filename": before_name, "width": w_b, "height": h_b, "modality": mod_b},
            "after": {"filename": after_name, "width": w_a, "height": h_a, "modality": mod_a},
        },
    )


def _inspect_stream_modality(file_bytes: bytes) -> str:
    """Inspect raster stream headers and descriptions to detect optical vs SAR modality."""
    try:
        with MemoryFile(file_bytes) as mem:
            with mem.open() as src:
                descriptions = [str(d).upper() for d in (src.descriptions or ()) if d]
                tags_dict = src.tags()
                tag_text = " ".join(f"{k}={v}" for k, v in tags_dict.items()).upper()
                desc_text = " ".join(descriptions)
                combined = f"{desc_text} {tag_text}"

                has_sar = any(
                    k in combined
                    for k in ("VV", "VH", "HH", "HV", "SIGMA0", "GAMMA0", "BETA0", "BACKSCATTER", "POLARIZ")
                )
                has_opt = any(
                    k in combined
                    for k in ("RED", "GREEN", "BLUE", "NIR", "SWIR", "REFLECTANCE", "B01", "B02", "B03", "B04", "B08")
                )

                if has_sar and not has_opt:
                    return "sar"
                if has_opt and not has_sar:
                    return "optical"
                if src.count >= 3 and not has_sar:
                    return "optical"
    except Exception:
        pass
    return "unknown"


def validate_optical_sar_pair(
    optical_bytes: Optional[bytes],
    sar_bytes: Optional[bytes],
    optical_name: str = "optical.tif",
    sar_name: str = "sar.tif",
    require_crs: bool = True,
) -> ValidationResult:
    """
    Validate that two uploaded rasters form a valid Optical-SAR multimodal pair.

    Enforces:
    - Exactly two images (optical + SAR)
    - Rejects optical + optical
    - Rejects SAR + SAR
    - Rejects missing modality metadata / unknown modality rather than guessing
    - Rejects swapped assignments
    - Requires georeferencing/CRS if require_crs is True
    - Inspects geographic overlap when both rasters have CRS
    """
    if not optical_bytes and not sar_bytes:
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Optical-SAR analysis requires both optical and SAR images; neither was provided.",
            details={"missing": ["optical_image", "sar_image"]},
        )
    if not optical_bytes:
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Optical-SAR analysis requires both optical and SAR images; missing optical image (SAR input provided, optical missing).",
            details={"missing": ["optical_image"]},
        )
    if not sar_bytes:
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Optical-SAR analysis requires both optical and SAR images; missing SAR image (Optical input provided, SAR missing).",
            details={"missing": ["sar_image"]},
        )

    # Format verification
    fmt_opt = validate_file_format(optical_name, optical_bytes)
    if not fmt_opt.valid:
        return fmt_opt
    fmt_sar = validate_file_format(sar_name, sar_bytes)
    if not fmt_sar.valid:
        return fmt_sar

    # Workflow-aware image integrity verification
    wf = "optical_sar" if require_crs else "general"
    int_opt = validate_image_bytes(optical_bytes, optical_name, workflow=wf)
    if not int_opt.valid:
        return int_opt
    int_sar = validate_image_bytes(sar_bytes, sar_name, workflow=wf)
    if not int_sar.valid:
        return int_sar

    # Modality verification
    opt_mod = _inspect_stream_modality(optical_bytes)
    sar_mod = _inspect_stream_modality(sar_bytes)

    if opt_mod == "optical" and sar_mod == "optical":
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR,
            message=(
                "Invalid Optical-SAR pair: Both inputs were identified as Optical imagery. "
                "A valid Optical-SAR pair requires one optical raster and one SAR raster."
            ),
            details={"optical_input": opt_mod, "sar_input": sar_mod, "reason": "both_optical"},
        )

    if opt_mod == "sar" and sar_mod == "sar":
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR,
            message=(
                "Invalid Optical-SAR pair: Both inputs were identified as SAR radar imagery. "
                "A valid Optical-SAR pair requires one optical raster and one SAR raster."
            ),
            details={"optical_input": opt_mod, "sar_input": sar_mod, "reason": "both_sar"},
        )

    if opt_mod == "sar" and sar_mod == "optical":
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR,
            message=(
                f"Mismatched modalities: The file provided for optical input ('{optical_name}') "
                f"was identified as SAR, and the file provided for SAR input ('{sar_name}') "
                "was identified as Optical. Please assign them to their respective fields."
            ),
            details={"optical_input": opt_mod, "sar_input": sar_mod, "reason": "modalities_swapped"},
        )

    # Geographic overlap check if both have CRS
    details_o = int_opt.details or {}
    details_s = int_sar.details or {}
    crs_o = details_o.get("crs")
    crs_s = details_s.get("crs")

    if crs_o and crs_s:
        try:
            with MemoryFile(optical_bytes) as mo, MemoryFile(sar_bytes) as ms:
                with mo.open() as so, ms.open() as ss:
                    o_crs = CRS.from_user_input(so.crs)
                    s_crs = CRS.from_user_input(ss.crs)
                    ob = so.bounds
                    sb = ss.bounds

                    if o_crs == s_crs:
                        left = max(ob.left, sb.left)
                        bottom = max(ob.bottom, sb.bottom)
                        right = min(ob.right, sb.right)
                        top = min(ob.top, sb.top)
                        overlap = (right > left) and (top > bottom)
                    else:
                        sb_in_o = transform_bounds(s_crs, o_crs, sb.left, sb.bottom, sb.right, sb.top)
                        left = max(ob.left, sb_in_o[0])
                        bottom = max(ob.bottom, sb_in_o[1])
                        right = min(ob.right, sb_in_o[2])
                        top = min(ob.top, sb_in_o[3])
                        overlap = (right > left) and (top > bottom)

                    if not overlap:
                        return ValidationResult(
                            valid=False,
                            error_code=ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR,
                            message="Optical and SAR images have zero geographic overlap.",
                            details={"optical": optical_name, "sar": sar_name, "reason": "no_spatial_overlap"},
                        )
        except Exception as geo_err:
            logger.warning(f"Optical-SAR spatial overlap inspection warning: {geo_err}")

    return ValidationResult(
        valid=True,
        details={
            "optical": {"filename": optical_name, "modality": opt_mod},
            "sar": {"filename": sar_name, "modality": sar_mod},
        },
    )


def validate_optical_sar_request(
    optical_ref: Optional[str] = None,
    sar_ref: Optional[str] = None,
    image_ids: Optional[List[str]] = None,
) -> ValidationResult:
    """Validate Optical-SAR reference count and presence."""
    ids = image_ids or []
    count = (1 if optical_ref else 0) + (1 if sar_ref else 0) + len(ids)

    if count == 0:
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Optical-SAR analysis requires one optical image and one SAR image; none provided.",
            details={"received_count": 0, "expected_count": 2},
        )
    if count == 1:
        missing = "optical" if sar_ref else "SAR"
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.WRONG_IMAGE_COUNT,
            message=f"Optical-SAR analysis requires one optical image and one SAR image; received 1 (missing {missing} input).",
            details={"received_count": 1, "expected_count": 2},
        )
    if count > 2:
        return ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.WRONG_IMAGE_COUNT,
            message=f"Optical-SAR analysis requires exactly two images; received {count}.",
            details={"received_count": count, "expected_count": 2},
        )

    return ValidationResult(valid=True)


def validate_query_input_compatibility(
    query: str,
    image_ids: Optional[List[str]] = None,
    optical_image_id: Optional[str] = None,
    sar_image_id: Optional[str] = None,
    aoi: Optional[Any] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    task: Optional[str] = None,
) -> ValidationResult:
    """
    Validate compatibility between natural-language query intent and provided inputs.

    Prevents execution of specialist or VLM tools when required inputs are missing or contradictory.
    """
    q_lower = (query or "").lower().strip()
    ids = list(image_ids or [])
    if optical_image_id and optical_image_id not in ids:
        ids.append(optical_image_id)
    if sar_image_id and sar_image_id not in ids:
        ids.append(sar_image_id)
    count = len(ids)

    # 1. Optical-SAR intent (prioritized over general compare/change)
    is_optical_sar_intent = (
        ("optical" in q_lower and "sar" in q_lower)
        or ("sentinel-1" in q_lower and "sentinel-2" in q_lower)
        or (task == "optical_sar_analysis")
    )

    if is_optical_sar_intent:
        if count == 1:
            missing_text = "SAR input is required" if optical_image_id else ("Optical input is required" if sar_image_id else "missing optical or SAR input")
            return ValidationResult(
                valid=False,
                error_code=ValidationErrorCode.INCOMPATIBLE_QUERY_INPUT,
                message=f"Optical-SAR analysis requires one optical image and one SAR image; {missing_text}.",
                details={"query": query, "image_count": 1, "expected": "1 optical + 1 SAR"},
            )
        elif count > 2:
            return ValidationResult(
                valid=False,
                error_code=ValidationErrorCode.WRONG_IMAGE_COUNT,
                message=f"Optical-SAR analysis requires exactly two images; received {count}.",
                details={"query": query, "image_count": count, "expected": 2},
            )
        elif count == 0 and not aoi:
            return ValidationResult(
                valid=False,
                error_code=ValidationErrorCode.MISSING_IMAGE,
                message="Optical-SAR analysis requires one optical image and one SAR image; neither was provided.",
                details={"query": query, "image_count": 0, "expected": 2},
            )

    # 2. Temporal change intent
    is_temporal_intent = any(
        w in q_lower for w in ("change", "compare", "difference", "between", "temporal", "transition")
    ) or (task in ("temporal_change", "change_detection", "urban_change", "water_change", "vegetation_change"))

    if is_temporal_intent:
        if count == 1:
            return ValidationResult(
                valid=False,
                error_code=ValidationErrorCode.INCOMPATIBLE_QUERY_INPUT,
                message="Temporal change analysis requires a before and after image.",
                details={"query": query, "image_count": 1, "expected": "before and after pair"},
            )
        elif count > 2:
            return ValidationResult(
                valid=False,
                error_code=ValidationErrorCode.WRONG_IMAGE_COUNT,
                message=f"Temporal change analysis requires two images; received {count}.",
                details={"query": query, "image_count": count, "expected": 2},
            )
        elif count == 0 and not (aoi or time_start or time_end or any(c.isdigit() for c in q_lower)):
            return ValidationResult(
                valid=False,
                error_code=ValidationErrorCode.MISSING_IMAGE,
                message="Temporal change analysis requires a before and after image; neither was provided.",
                details={"query": query, "image_count": 0, "expected": 2},
            )

    # 3. Single-image VQA / captioning intent
    is_single_image_intent = any(
        w in q_lower for w in ("caption", "describe this image", "what is visible in this image")
    ) or (task in ("single_image_vqa", "captioning", "single_image_caption"))

    if is_single_image_intent and not aoi:
        if count == 0:
            return ValidationResult(
                valid=False,
                error_code=ValidationErrorCode.MISSING_IMAGE,
                message="Single-image analysis requires an image; none provided.",
                details={"query": query, "image_count": 0, "expected": 1},
            )
        elif count > 1:
            return ValidationResult(
                valid=False,
                error_code=ValidationErrorCode.WRONG_IMAGE_COUNT,
                message=f"Single-image analysis accepts only one image; received {count}.",
                details={"query": query, "image_count": count, "expected": 1},
            )

    return ValidationResult(valid=True)
