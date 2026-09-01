"""
API endpoints for user image upload analysis.

Connects user-uploaded satellite/aerial imagery directly to P2 remote-sensing analysis
and hands off to P4 multimodal VLM reasoning.
"""

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.agent.upload_processor import process_upload_analysis
from app.schemas.analysis import AnalysisResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Analysis & Uploads"])


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
