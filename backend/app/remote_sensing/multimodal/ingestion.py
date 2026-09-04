"""
Secure ingestion and reference resolution module for Optical-SAR multimodal rasters.

Provides safe file storage, validation, and resolution of user-provided image references
(image IDs or authorized paths) into verified local GeoTIFF file paths.

Security & Modality Integrity:
- Disallows arbitrary server filesystem traversal or access outside permitted directories.
- Strictly validates raster readable headers, CRS, dimension consistency, and modality pairing.
- Prevents invalid pairing such as Optical+Optical or SAR+SAR for joint Optical-SAR analysis.
"""

from __future__ import annotations

import io
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rasterio
from rasterio.io import MemoryFile

# ============================================================
# STORAGE DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_STORAGE_ROOTS = [
    UPLOAD_DIR.resolve(),
    (PROJECT_ROOT / "data" / "samples").resolve(),
    (PROJECT_ROOT / "data" / "s1_cache").resolve(),
    (PROJECT_ROOT / "data" / "cache").resolve(),
]


def _sanitize_filename(filename: str) -> str:
    """Strip dangerous path components and characters."""
    clean = Path(filename).name
    clean = re.sub(r"[^\w\-.]", "_", clean)
    if not clean or clean.startswith("."):
        clean = f"raster_{clean}"
    return clean


def store_uploaded_raster(
    file_bytes: bytes,
    filename: str,
    modality_hint: Optional[str] = None,
) -> str:
    """
    Validate and store an uploaded raster into UPLOAD_DIR along with metadata.

    Returns:
        A secure unique `image_id` string referencing the stored raster.
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    # Validate that rasterio can read it
    band_count = 0
    crs_str = ""
    width = 0
    height = 0
    descriptions: list[str] = []
    try:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                band_count = src.count
                if band_count < 1:
                    raise ValueError("Raster has no bands.")
                width = src.width
                height = src.height
                if width < 1 or height < 1:
                    raise ValueError("Raster has invalid dimensions.")
                if src.crs is None:
                    raise ValueError("Raster has no valid CRS.")
                crs_str = str(src.crs)
                descriptions = [str(d) for d in (src.descriptions or ()) if d]
    except Exception as exc:
        raise ValueError(f"Uploaded file '{filename}' is not a valid GeoTIFF/raster: {exc}")

    prefix = modality_hint.lower() if modality_hint in ("optical", "sar") else "img"
    uid = uuid.uuid4().hex[:8]
    safe_name = _sanitize_filename(filename)
    if not safe_name.lower().endswith((".tif", ".tiff")):
        safe_name = f"{safe_name}.tif"

    image_id = f"{prefix}_{uid}_{safe_name}"
    target_path = UPLOAD_DIR / image_id
    target_path.write_bytes(file_bytes)

    # Persist sidecar metadata
    meta_info = {
        "image_id": image_id,
        "filename": filename,
        "declared_modality": modality_hint.lower() if modality_hint in ("optical", "sar") else None,
        "band_count": band_count,
        "descriptions": descriptions,
        "crs": crs_str,
        "width": width,
        "height": height,
    }
    meta_path = UPLOAD_DIR / f"{image_id}.json"
    meta_path.write_text(json.dumps(meta_info, indent=2))

    return image_id


def get_image_metadata(path: Path) -> Dict[str, Any]:
    """Retrieve sidecar metadata if present, or empty dict."""
    candidates = [
        path.parent / f"{path.name}.json",
        path.with_suffix(".json"),
        UPLOAD_DIR / f"{path.name}.json",
        UPLOAD_DIR / f"{path.stem}.json",
    ]
    for c in candidates:
        if c.is_file():
            try:
                data = json.loads(c.read_text())
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return {}


def is_safe_path(candidate_path: Path) -> bool:
    """Verify that candidate_path resolves within one of the allowed storage roots."""
    try:
        resolved = candidate_path.resolve()
        for root in ALLOWED_STORAGE_ROOTS:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False
    except Exception:
        return False


def resolve_image_reference(ref: str) -> Path:
    """
    Resolve a user-provided image identifier or authorized path to a verified server Path.

    Raises:
        ValueError: If the reference violates security rules or is malformed.
        FileNotFoundError: If the referenced image cannot be located.
    """
    if not ref or not isinstance(ref, str):
        raise ValueError("Image reference must be a non-empty string.")

    ref_str = ref.strip()

    # Reject blatant directory traversal attempts
    if ".." in ref_str:
        raise ValueError(f"Security error: Invalid path traversal in image reference: '{ref_str}'")

    # 1. Check if ref directly matches a file in UPLOAD_DIR
    candidate_upload = UPLOAD_DIR / ref_str
    if candidate_upload.is_file():
        resolved = candidate_upload.resolve()
        if is_safe_path(resolved):
            return resolved

    # Check with .tif appended in UPLOAD_DIR
    candidate_upload_tif = UPLOAD_DIR / f"{ref_str}.tif"
    if candidate_upload_tif.is_file():
        resolved = candidate_upload_tif.resolve()
        if is_safe_path(resolved):
            return resolved

    # 2. Check if ref is a relative path inside PROJECT_ROOT / "data"
    candidate_data = PROJECT_ROOT / "data" / ref_str
    if candidate_data.is_file():
        resolved = candidate_data.resolve()
        if is_safe_path(resolved):
            return resolved

    # 3. Check if ref is a path string
    path_obj = Path(ref_str)
    if path_obj.is_absolute():
        resolved = path_obj.resolve()
        if not is_safe_path(resolved):
            raise ValueError(
                f"Security error: Access to path outside authorized storage roots is forbidden: '{ref_str}'"
            )
        if not resolved.is_file():
            raise FileNotFoundError(f"Referenced image file does not exist: '{ref_str}'")
        return resolved

    # Also check directly against each allowed root
    for root in ALLOWED_STORAGE_ROOTS:
        cand = root / ref_str
        if cand.is_file():
            resolved = cand.resolve()
            if is_safe_path(resolved):
                return resolved

    raise FileNotFoundError(f"Image reference not found in authorized storage: '{ref_str}'")


def determine_raster_modality(
    path: Path,
    declared_modality: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Determine whether a raster is Optical or SAR imagery using headers, tags, band descriptions,
    and trusted declaration metadata.

    Returns:
        (modality, reason): where modality is "optical", "sar", or "unknown"
    """
    meta = get_image_metadata(path)
    dec_mod = declared_modality or meta.get("declared_modality")
    if dec_mod:
        dec_mod = str(dec_mod).lower()

    with rasterio.open(path) as src:
        band_count = src.count
        descriptions = [str(d).upper() for d in (src.descriptions or ()) if d]
        tags_dict = src.tags()
        tag_text = " ".join(f"{k}={v}" for k, v in tags_dict.items()).upper()
        name_upper = path.name.upper()

        # Gather all text signals
        all_desc_text = " ".join(descriptions)
        combined_text = f"{name_upper} {all_desc_text} {tag_text}"

        has_sar_tags = any(
            k in combined_text
            for k in ("VV", "VH", "HH", "HV", "POLARIZ", "POLARIS", "SIGMA0", "GAMMA0", "BETA0", "BACKSCATTER")
        ) or any(
            k in name_upper
            for k in ("SAR", "SENTINEL1", "SENTINEL-1", "S1_", "S1A_", "S1B_", "RADAR")
        )

        has_optical_tags = any(
            k in combined_text
            for k in (
                "RED", "GREEN", "BLUE", "NIR", "SWIR", "RGB", "TRUE COLOR", "TRUE_COLOR",
                "B01", "B02", "B03", "B04", "B08", "REFLECTANCE"
            )
        ) or any(
            k in name_upper
            for k in ("OPTICAL", "SENTINEL2", "SENTINEL-2", "S2_", "S2A_", "S2B_", "RGB")
        )

        # 1. Explicit declaration validation
        if dec_mod == "sar":
            if band_count >= 3 and has_optical_tags and not has_sar_tags:
                raise ValueError(
                    f"Declared modality 'sar' conflicts with optical multi-spectral bands in '{path.name}'."
                )
            return "sar", "explicit_metadata"

        if dec_mod == "optical":
            if band_count <= 2 and has_sar_tags and not has_optical_tags:
                raise ValueError(
                    f"Declared modality 'optical' conflicts with SAR polarization tags in '{path.name}'."
                )
            return "optical", "explicit_metadata"

        # 2. SAR Indicators
        if has_sar_tags and band_count in (1, 2):
            return "sar", "sar_headers_and_bands"

        # 3. Optical Indicators
        if has_optical_tags:
            return "optical", "optical_headers"

        if band_count >= 3 and not has_sar_tags:
            return "optical", "multiband_optical"

        # 4. Single / Dual band without tags (ambiguous)
        if band_count in (1, 2):
            return "unknown", "ambiguous_bands_no_metadata"

        return "unknown", "unrecognized_raster_properties"


def inspect_raster_modality(path: Path) -> str:
    """Backward-compatible helper returning 'optical', 'sar', or 'unknown'."""
    mod, _ = determine_raster_modality(path)
    return mod


def resolve_optical_sar_references(
    optical_ref: Optional[str] = None,
    sar_ref: Optional[str] = None,
    image_ids: Optional[List[str]] = None,
) -> Tuple[Path, Path]:
    """
    Resolve, validate, and verify modality integrity for both optical and SAR inputs.

    Raises:
        ValueError: If either modality is missing, invalid, improperly paired, or fails security checks.
        FileNotFoundError: If a referenced file is not found.
    """
    image_ids = image_ids or []

    resolved_opt: Optional[Path] = None
    resolved_sar: Optional[Path] = None

    # Handle explicit references first
    if optical_ref:
        resolved_opt = resolve_image_reference(optical_ref)

    if sar_ref:
        resolved_sar = resolve_image_reference(sar_ref)

    # Fallback to image_ids if explicit references were not both given
    if (resolved_opt is None or resolved_sar is None) and len(image_ids) >= 2:
        path_0 = resolve_image_reference(image_ids[0])
        path_1 = resolve_image_reference(image_ids[1])

        mod_0, _ = determine_raster_modality(path_0)
        mod_1, _ = determine_raster_modality(path_1)

        if mod_0 == "optical" and mod_1 == "sar":
            resolved_opt = resolved_opt or path_0
            resolved_sar = resolved_sar or path_1
        elif mod_0 == "sar" and mod_1 == "optical":
            # Auto-assign reversed positional pair
            resolved_opt = resolved_opt or path_1
            resolved_sar = resolved_sar or path_0
        else:
            resolved_opt = resolved_opt or path_0
            resolved_sar = resolved_sar or path_1

    # Check for missing modalities
    if resolved_opt is None and resolved_sar is None:
        raise ValueError(
            "Both optical and SAR inputs are required for optical_sar_analysis; neither was provided."
        )
    if resolved_opt is None:
        raise ValueError(
            "Optical input is required for optical_sar_analysis; only SAR input was provided."
        )
    if resolved_sar is None:
        raise ValueError(
            "SAR input is required for optical_sar_analysis; only optical input was provided."
        )

    # Validate that both can be opened by rasterio
    for label, p in [("Optical", resolved_opt), ("SAR", resolved_sar)]:
        try:
            with rasterio.open(p) as src:
                if src.crs is None:
                    raise ValueError(f"{label} raster '{p.name}' lacks a Coordinate Reference System (CRS).")
                if src.width < 1 or src.height < 1:
                    raise ValueError(f"{label} raster '{p.name}' has invalid pixel dimensions.")
        except Exception as exc:
            raise ValueError(f"Failed to open {label} raster '{p.name}': {exc}")

    # Modality Integrity Verification
    opt_mod, _ = determine_raster_modality(resolved_opt)
    sar_mod, _ = determine_raster_modality(resolved_sar)

    if opt_mod == "optical" and sar_mod == "optical":
        raise ValueError(
            "Invalid Optical-SAR pair: Both inputs were identified as Optical imagery. "
            "A valid Optical-SAR pair requires one optical raster and one SAR raster."
        )

    if opt_mod == "sar" and sar_mod == "sar":
        raise ValueError(
            "Invalid Optical-SAR pair: Both inputs were identified as SAR radar imagery. "
            "A valid Optical-SAR pair requires one optical raster and one SAR raster."
        )

    if opt_mod == "sar" and sar_mod == "optical":
        raise ValueError(
            f"Mismatched modalities: The file provided for optical input ('{resolved_opt.name}') "
            f"was identified as SAR, and the file provided for SAR input ('{resolved_sar.name}') "
            "was identified as Optical. Please assign them to their respective fields."
        )

    if opt_mod == "unknown" and sar_mod == "sar":
        raise ValueError(
            f"Invalid Optical-SAR pair: The first input '{resolved_opt.name}' could not be verified as an optical raster. "
            "Please provide a valid optical raster or declare modality='optical'."
        )

    if opt_mod == "optical" and sar_mod == "unknown":
        raise ValueError(
            f"Invalid Optical-SAR pair: The second input '{resolved_sar.name}' could not be verified as a SAR raster. "
            "Please provide a valid SAR raster or declare modality='sar'."
        )

    if opt_mod == "unknown" and sar_mod == "unknown":
        raise ValueError(
            f"Invalid Optical-SAR pair: Neither input ('{resolved_opt.name}', '{resolved_sar.name}') "
            "could be verified as optical or SAR. Please declare modalities for both images."
        )

    return resolved_opt, resolved_sar
