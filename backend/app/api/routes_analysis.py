"""
API endpoints for user image upload analysis.

Connects user-uploaded satellite/aerial imagery directly to P2 remote-sensing analysis
and hands off to P4 multimodal VLM reasoning.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.agent.upload_processor import process_upload_analysis
from app.remote_sensing.multimodal.ingestion import store_uploaded_raster
from app.remote_sensing.validation import (
    validate_file_format,
    validate_image_bytes,
    validate_temporal_pair,
)
from app.schemas.analysis import AnalysisResult
from app.schemas.validation import (
    ValidationError,
    ValidationErrorCode,
    ValidationResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Analysis & Uploads"])


@router.post(
    "/upload/image",
    summary="Upload a single satellite raster (Optical or SAR GeoTIFF) and receive an image ID",
)
async def upload_image(
    file: Optional[UploadFile] = File(None),
    modality: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Upload an optical or SAR GeoTIFF raster to the secure store and receive an image_id."""
    if not file or not file.filename:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="A valid file must be provided.",
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    try:
        content = await file.read()
    except Exception as exc:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.INVALID_IMAGE,
            message=f"Failed to read file: {exc}",
            details={"error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    if len(content) == 0:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Uploaded file cannot be empty.",
            details={"filename": file.filename, "size": 0},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    # Format validation
    fmt_res = validate_file_format(file.filename, content)
    if not fmt_res.valid:
        raise HTTPException(status_code=400, detail=fmt_res.to_http_detail())

    # Image byte integrity validation
    int_res = validate_image_bytes(content, file.filename)
    if not int_res.valid:
        raise HTTPException(status_code=400, detail=int_res.to_http_detail())

    try:
        image_id = store_uploaded_raster(
            file_bytes=content,
            filename=file.filename,
            modality_hint=modality,
        )
        return {
            "status": "success",
            "image_id": image_id,
            "filename": file.filename,
            "modality": modality or "unknown",
        }
    except ValueError as val_err:
        err_msg = str(val_err)
        err_code = ValidationErrorCode.INVALID_IMAGE
        if "crs" in err_msg.lower():
            err_code = ValidationErrorCode.MISSING_METADATA
        elif "unsupported" in err_msg.lower():
            err_code = ValidationErrorCode.UNSUPPORTED_FILE_TYPE
        res = ValidationResult(
            valid=False,
            error_code=err_code,
            message=err_msg,
            details={"filename": file.filename, "error": err_msg},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())
    except Exception as exc:
        logger.exception("Upload error")
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.INVALID_IMAGE,
            message=f"Failed to store image: {exc}",
            details={"error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())


@router.post(
    "/upload/optical-sar",
    response_model=AnalysisResult,
    summary="Direct upload of an Optical GeoTIFF and a SAR GeoTIFF for joint multimodal analysis",
)
async def analyze_uploaded_optical_sar(
    optical_image: Optional[UploadFile] = File(None),
    sar_image: Optional[UploadFile] = File(None),
    query: str = Form("Use the optical and SAR images together to analyze the area."),
) -> AnalysisResult:
    """Direct multipart upload of paired Optical and SAR GeoTIFFs for multimodal reasoning."""
    if not optical_image and not sar_image:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Both optical and SAR images are required; none provided.",
            details={"missing": ["optical_image", "sar_image"]},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    if not optical_image:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Optical image is required; missing optical input.",
            details={"missing": ["optical_image"]},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    if not sar_image:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="SAR image is required; missing SAR input.",
            details={"missing": ["sar_image"]},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    try:
        opt_bytes = await optical_image.read()
        sar_bytes = await sar_image.read()
    except Exception as exc:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.INVALID_IMAGE,
            message=f"Could not read uploaded files: {exc}",
            details={"error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    if len(opt_bytes) == 0:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Optical image file cannot be empty.",
            details={"filename": optical_image.filename, "size": 0},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    if len(sar_bytes) == 0:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="SAR image file cannot be empty.",
            details={"filename": sar_image.filename, "size": 0},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    # Format checks
    fmt_opt = validate_file_format(optical_image.filename or "optical.tif", opt_bytes)
    if not fmt_opt.valid:
        raise HTTPException(status_code=400, detail=fmt_opt.to_http_detail())

    fmt_sar = validate_file_format(sar_image.filename or "sar.tif", sar_bytes)
    if not fmt_sar.valid:
        raise HTTPException(status_code=400, detail=fmt_sar.to_http_detail())

    # Integrity checks
    int_opt = validate_image_bytes(opt_bytes, optical_image.filename or "optical.tif")
    if not int_opt.valid:
        raise HTTPException(status_code=400, detail=int_opt.to_http_detail())

    int_sar = validate_image_bytes(sar_bytes, sar_image.filename or "sar.tif")
    if not int_sar.valid:
        raise HTTPException(status_code=400, detail=int_sar.to_http_detail())

    try:
        opt_id = store_uploaded_raster(opt_bytes, optical_image.filename or "optical.tif", modality_hint="optical")
        sar_id = store_uploaded_raster(sar_bytes, sar_image.filename or "sar.tif", modality_hint="sar")
        from app.remote_sensing.multimodal.ingestion import resolve_optical_sar_references
        _ = resolve_optical_sar_references(optical_ref=opt_id, sar_ref=sar_id)
    except ValueError as val_err:
        err_msg = str(val_err)
        err_code = ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR if "pair" in err_msg.lower() or "conflict" in err_msg.lower() or "modality" in err_msg.lower() else ValidationErrorCode.INVALID_IMAGE
        res = ValidationResult(
            valid=False,
            error_code=err_code,
            message=err_msg,
            details={"error": err_msg},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    from app.api.routes_query import process_query
    from app.schemas.query import QueryRequest

    req = QueryRequest(
        query=query,
        optical_image_id=opt_id,
        sar_image_id=sar_id,
    )
    return process_query(req)



@router.post(
    "/upload/analyze",
    response_model=AnalysisResult,
    summary="Analyze two uploaded satellite/aerial images for temporal change",
)
async def analyze_uploaded_images(
    before_image: Optional[UploadFile] = File(None),
    after_image: Optional[UploadFile] = File(None),
    image_a: Optional[UploadFile] = File(None),
    image_b: Optional[UploadFile] = File(None),
    query: str = Form("Show change"),
    threshold: Optional[float] = Form(None),
) -> AnalysisResult:
    """
    Upload two satellite/aerial images (multispectral GeoTIFF or standard RGB)
    and perform change detection analysis grounded by the P4 VLM.

    - Multispectral GeoTIFFs (Red, NIR, Green, SWIR) allow quantitative NDVI/NDWI/NDBI.
    - Standard visible RGB images undergo visual change detection without fake indices.
    """
    # Support both before/after and image_a/image_b field names
    file_before = before_image or image_a
    file_after = after_image or image_b

    if not file_before and not file_after:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Two valid images (before and after) are required for temporal change detection; neither was provided.",
            details={"missing": ["before_image", "after_image"]},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    if not file_before:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Two valid images (before and after) are required for temporal change detection; missing before image.",
            details={"missing": ["before_image"]},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    if not file_after:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Two valid images (before and after) are required for temporal change detection; missing after image.",
            details={"missing": ["after_image"]},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())


    try:
        before_bytes = await file_before.read()
        after_bytes = await file_after.read()
    except Exception as read_err:
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.INVALID_IMAGE,
            message=f"Could not read uploaded files: {read_err}",
            details={"error": str(read_err)},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    if len(before_bytes) == 0 or len(after_bytes) == 0:
        missing_name = file_before.filename if len(before_bytes) == 0 else file_after.filename
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.MISSING_IMAGE,
            message="Uploaded image files cannot be empty.",
            details={"filename": missing_name, "size": 0},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())

    try:
        result = process_upload_analysis(
            before_bytes=before_bytes,
            after_bytes=after_bytes,
            before_name=file_before.filename or "before_image",
            after_name=file_after.filename or "after_image",
            query=query,
            threshold=threshold,
        )
        return result
    except ValidationError as val_err:
        raise HTTPException(status_code=400, detail=val_err.result.to_http_detail())
    except ValueError as val_err:
        err_msg = str(val_err)
        err_code = ValidationErrorCode.INCOMPATIBLE_IMAGE_PAIR if "incompatible" in err_msg.lower() or "pair" in err_msg.lower() else ValidationErrorCode.INVALID_IMAGE
        res = ValidationResult(
            valid=False,
            error_code=err_code,
            message=err_msg,
            details={"error": err_msg},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())
    except Exception as err:
        logger.exception("Upload analysis error")
        res = ValidationResult(
            valid=False,
            error_code=ValidationErrorCode.INVALID_IMAGE,
            message=f"Internal error analyzing uploaded images: {err}",
            details={"error": str(err)},
        )
        raise HTTPException(status_code=400, detail=res.to_http_detail())



@router.post(
    "/analyze/upload",
    response_model=AnalysisResult,
    include_in_schema=False,
)
async def analyze_uploaded_images_alias(
    before_image: Optional[UploadFile] = File(None),
    after_image: Optional[UploadFile] = File(None),
    image_a: Optional[UploadFile] = File(None),
    image_b: Optional[UploadFile] = File(None),
    query: str = Form("Show change"),
    threshold: Optional[float] = Form(None),
) -> AnalysisResult:
    """Alias for /api/upload/analyze."""
    return await analyze_uploaded_images(
        before_image=before_image,
        after_image=after_image,
        image_a=image_a,
        image_b=image_b,
        query=query,
        threshold=threshold,
    )
