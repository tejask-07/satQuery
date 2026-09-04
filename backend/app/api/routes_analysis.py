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
from app.schemas.analysis import AnalysisResult

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
        raise HTTPException(status_code=400, detail="A valid file must be provided.")
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file cannot be empty.")

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
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        logger.exception("Upload error")
        raise HTTPException(status_code=500, detail=f"Failed to store image: {exc}")


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
    if not optical_image:
        raise HTTPException(status_code=400, detail="Optical image is required; missing optical input.")
    if not sar_image:
        raise HTTPException(status_code=400, detail="SAR image is required; missing SAR input.")

    try:
        opt_bytes = await optical_image.read()
        sar_bytes = await sar_image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded files: {exc}")

    if len(opt_bytes) == 0:
        raise HTTPException(status_code=400, detail="Optical image file cannot be empty.")
    if len(sar_bytes) == 0:
        raise HTTPException(status_code=400, detail="SAR image file cannot be empty.")

    try:
        opt_id = store_uploaded_raster(opt_bytes, optical_image.filename or "optical.tif", modality_hint="optical")
        sar_id = store_uploaded_raster(sar_bytes, sar_image.filename or "sar.tif", modality_hint="sar")
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))

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

    if not file_before or not file_after:
        raise HTTPException(
            status_code=400,
            detail="Two valid images (before and after) are required for change detection.",
        )

    try:
        before_bytes = await file_before.read()
        after_bytes = await file_after.read()
    except Exception as read_err:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded files: {read_err}",
        )

    if len(before_bytes) == 0 or len(after_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image files cannot be empty.",
        )

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
    except ValueError as val_err:
        raise HTTPException(
            status_code=400,
            detail=str(val_err),
        )
    except Exception as err:
        logger.exception("Upload analysis error")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error analyzing uploaded images: {err}",
        )


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
