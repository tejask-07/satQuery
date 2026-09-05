import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any,Optional

from app import config
from app.config import BACKEND_DIR, load_env_file

# Ensure latest .env variables are present
load_env_file(BACKEND_DIR / ".env")

from fastapi import APIRouter, HTTPException
from app.evidence.visualizations import save_change_map, VISUALIZATION_DIR
from app.agent.executor import (
    execute_plan,
    _save_change_map_visualization,
    _get_raster_bounds,
)
from app.agent.planner import create_execution_plan
from app.remote_sensing.multimodal.ingestion import resolve_optical_sar_references
from app.schemas.analysis import AnalysisResult
from app.schemas.query import QueryPlan, QueryRequest
from app.evidence.multi_index import calculate_multi_index_evidence
from app.evidence.fusion import fuse_evidence_and_classify_candidates
from app.evidence.interpretation import generate_structured_interpretation
from app.evidence.spatial import extract_spatial_candidate_regions
from app.evidence.temporal import (
    build_temporal_analysis_package,
    TemporalObservation,
)
from app.evidence.calibration import build_calibration_package


def _safe_vis_url(filename: Optional[str]) -> Optional[str]:
    """
    Audit visualization URL safety.
    Never allow placeholders, template strings, or non-existent files to escape.
    """
    if not filename:
        return None
    fname_str = str(filename).strip()
    if "<" in fname_str or ">" in fname_str or "placeholder" in fname_str.lower():
        return None
    clean_name = fname_str.split("/")[-1].split("\\")[-1]
    if not clean_name:
        return None
    file_path = VISUALIZATION_DIR / clean_name
    if file_path.exists() and file_path.is_file():
        return f"/visualizations/{clean_name}"
    return None

# ============================================================
# P4 / VLM INTEGRATION
# ============================================================

from app.vlm.evidence_builder import build_evidence
from app.vlm.model import VLM
from app.vlm.p2_imagery import load_p2_images
from app.vlm.bigearthnet.s1_p4 import build_s1_visualization


router = APIRouter(
    prefix="/api",
    tags=["query"],
)


from app.agent.parser import parse_query


# ============================================================
# QUERY PLANNER
# ============================================================

def build_query_plan(
    request: QueryRequest,
) -> QueryPlan:
    """
    Convert natural-language query into a QueryPlan using the Phase 1 structured analysis planner.
    """
    return parse_query(request)



# ============================================================
# EXPLANATION BUILDER
# ============================================================

def build_explanation(
    plan: QueryPlan,
    statistics: dict,
    execution_results: dict,
) -> str:
    """
    Create a human-readable explanation from
    calculated statistics.
    """

    # ========================================================
    # SINGLE INDEX (NDVI / NDWI / NDBI)
    # ========================================================

    if plan.task in ["vegetation_index", "water_index", "urban_index"]:
        metric = (plan.metric or "NDVI").upper()
        target = plan.target or "region"
        mean_val = statistics.get("mean")
        valid_px = statistics.get("valid_pixels")
        min_val = statistics.get("min_value")
        max_val = statistics.get("max_value")
        if mean_val is not None:
            expl = f"{metric} calculated for {target}: mean value is {mean_val:.4f}"
            if min_val is not None and max_val is not None:
                expl += f" (range: {min_val:.4f} to {max_val:.4f})"
            if valid_px is not None:
                expl += f" across {valid_px} valid pixels."
            return expl
        return f"{metric} calculated for {target}."

    # ========================================================
    # IMAGE COMPARISON
    # ========================================================

    if plan.task == "image_comparison":

        mean_before = statistics.get(
            "mean_before"
        )

        mean_after = statistics.get(
            "mean_after"
        )

        mean_change = statistics.get(
            "mean_change"
        )

        changed_pixels = statistics.get(
            "changed_pixels"
        )

        valid_pixels = statistics.get(
            "valid_pixels"
        )

        change_ratio = statistics.get(
            "change_ratio"
        )

        increased_pixels = statistics.get(
            "increased_pixels",
            0,
        )

        decreased_pixels = statistics.get(
            "decreased_pixels",
            0,
        )

        threshold = statistics.get(
            "threshold",
            0.05,
        )

        if (
            mean_before is None
            or mean_after is None
        ):
            return (
                "The satellite images were compared, "
                "but there were not enough valid pixels "
                "to calculate the change."
            )

        before_text = (
            f"{mean_before:.2f}"
        )

        after_text = (
            f"{mean_after:.2f}"
        )

        change_text = (
            f"{mean_change:+.2f}"
            if mean_change is not None
            else "N/A"
        )

        ratio_text = (
            f"{change_ratio * 100:.2f}%"
            if change_ratio is not None
            else "N/A"
        )

        explanation = (
            f"Between {plan.time_start} and "
            f"{plan.time_end}, the mean satellite "
            f"pixel value changed from {before_text} "
            f"to {after_text}, a change of "
            f"{change_text}. "
        )

        if (
            changed_pixels is not None
            and valid_pixels is not None
        ):
            explanation += (
                f"{changed_pixels} of {valid_pixels} "
                f"valid pixels ({ratio_text}) showed "
                f"a change greater than the threshold "
                f"of {threshold:.2f}. "
            )

        if decreased_pixels > increased_pixels:
            explanation += (
                f"{decreased_pixels} pixels decreased "
                f"while {increased_pixels} pixels "
                f"increased. Overall, the imagery "
                f"indicates a decrease in pixel values."
            )

        elif increased_pixels > decreased_pixels:
            explanation += (
                f"{increased_pixels} pixels increased "
                f"while {decreased_pixels} pixels "
                f"decreased. Overall, the imagery "
                f"indicates an increase in pixel values."
            )

        else:
            explanation += (
                f"{increased_pixels} pixels increased "
                f"and {decreased_pixels} pixels "
                f"decreased. The overall change was "
                f"approximately balanced."
            )

        return explanation

    # ========================================================
    # NORMAL INDEX / CHANGE ANALYSIS
    # ========================================================

    metric = statistics.get(
        "metric"
    )

    mean_before = statistics.get(
        "mean_before"
    )

    mean_after = statistics.get(
        "mean_after"
    )

    mean_change = statistics.get(
        "mean_change"
    )

    changed_pixels = statistics.get(
        "changed_pixels"
    )

    valid_pixels = statistics.get(
        "valid_pixels"
    )

    change_ratio = statistics.get(
        "change_ratio"
    )

    change_type = statistics.get(
        "change_type"
    )

    # --------------------------------------------------------
    # No statistics.
    # --------------------------------------------------------

    if metric is None:

        if plan.task == "image_search":

            imagery_result = execution_results.get(
                "search_imagery",
                {},
            )

            images = imagery_result.get(
                "images",
                [],
            )

            if not images:
                return (
                    "No satellite imagery was found "
                    "for the requested period."
                )

            image_descriptions = []

            for image in images:

                image_id = image.get(
                    "id",
                    "unknown",
                )

                date = image.get(
                    "date",
                    "unknown date",
                )

                cloud_cover = image.get(
                    "cloud_cover"
                )

                if cloud_cover is not None:
                    image_descriptions.append(
                        f"{image_id} dated {date} "
                        f"with {cloud_cover:.1f}% cloud cover"
                    )

                else:
                    image_descriptions.append(
                        f"{image_id} dated {date}"
                    )

            return (
                f"Found {len(images)} satellite images "
                f"for the requested period from "
                f"{plan.time_start} to {plan.time_end}. "
                + "Available imagery: "
                + "; ".join(
                    image_descriptions
                )
                + "."
            )

        return (
            "The requested analysis was completed, "
            "but no index statistics were available."
        )

    # --------------------------------------------------------
    # No valid data.
    # --------------------------------------------------------

    if (
        mean_before is None
        or mean_after is None
    ):
        return (
            f"{metric} analysis was completed, "
            "but there were not enough valid pixels "
            "to calculate a before-and-after comparison."
        )

    # --------------------------------------------------------
    # Format numbers.
    # --------------------------------------------------------

    before_text = (
        f"{mean_before:.4f}"
    )

    after_text = (
        f"{mean_after:.4f}"
    )

    if mean_change is not None:
        change_text = (
            f"{mean_change:+.4f}"
        )
    else:
        change_text = "N/A"

    ratio_text = (
        f"{change_ratio * 100:.2f}%"
        if change_ratio is not None
        else "N/A"
    )

    # --------------------------------------------------------
    # Target.
    # --------------------------------------------------------

    target = (
        plan.target
        or "target"
    )

    # --------------------------------------------------------
    # Main explanation.
    # --------------------------------------------------------

    explanation = (
        f"{metric} for {target} changed from "
        f"{before_text} in {plan.time_start} "
        f"to {after_text} in {plan.time_end}. "
        f"The mean change was {change_text}. "
    )

    # --------------------------------------------------------
    # Pixel statistics.
    # --------------------------------------------------------

    if (
        changed_pixels is not None
        and valid_pixels is not None
    ):
        explanation += (
            f"{changed_pixels} of "
            f"{valid_pixels} valid pixels "
            f"({ratio_text}) exceeded the "
            f"change threshold. "
        )

    # --------------------------------------------------------
    # Direction.
    # --------------------------------------------------------

    if change_type == "increase":

        explanation += (
            f"Overall, the {target} signal "
            "indicates an increase."
        )

    elif change_type == "decrease":

        explanation += (
            f"Overall, the {target} signal "
            "indicates a decrease."
        )

    elif change_type == "no_change":

        explanation += (
            f"Overall, no meaningful {target} "
            "change was detected."
        )

    else:

        explanation += (
            "The overall direction of change "
            "could not be determined."
        )

    return explanation


# ============================================================
# P4 VLM HELPER
# ============================================================

def generate_vlm_answer(
    request: QueryRequest,
    p2_response: dict,
) -> str | None:
    """
    Run the P4 VLM using:

    - P2 structured remote-sensing evidence
    - Sentinel-2 before/after imagery
    - Sentinel-2 change map
    - real Sentinel-1 VV/VH composite

    The current SIH demo uses the validated
    BigEarthNet S1 patch.

    Returns None when multimodal reasoning is unavailable.
    """

    task = (
        p2_response
        .get("plan", {})
        .get("task")
    )

    # --------------------------------------------------------
    # Image search currently has no analytical
    # before/after imagery package.
    # --------------------------------------------------------

    if task == "image_search":
        return None

    try:

        # ====================================================
        # 1. Build structured P2 evidence
        # ====================================================

        evidence = p2_response.get("evidence_package") or build_evidence(
            p2_response
        )

        # ====================================================
        # 2. Load P2 / Sentinel-2 imagery
        #
        # Existing loader provides:
        #
        # before
        # after
        # change_map
        # ====================================================

        images = load_p2_images(
            evidence
        )

        if not images:
            return None

        # ====================================================
        # 3. Load real Sentinel-1 imagery
        #
        # Current validated BigEarthNet demo mapping:
        #
        # S2:
        # S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
        #
        # S1:
        # S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57
        #
        # The S1 loader uses:
        #
        # local disk cache -> RAM cache
        # ====================================================

        s1_name = (
            "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"
        )

        s1_composite = build_s1_visualization(
            s1_name
        )

        images["s1_composite"] = (
            s1_composite
        )

        # ====================================================
        # 4. Run P4 VLM
        # ====================================================

        vlm = VLM()

        return vlm.generate(
            question=request.query,
            evidence=evidence,
            images=images,
        )

    except Exception as exc:

        # ----------------------------------------------------
        # Keep P2 analysis working even when
        # the optional P4 layer fails.
        # ----------------------------------------------------

        print(
            f"[P4 VLM WARNING] "
            f"{type(exc).__name__}: {exc}"
        )

        return None


_PATH_PATTERN = re.compile(
    r"^(?:[a-zA-Z]:[\\/]|/(?:home|Users|etc|tmp|var|private|root)[\\/]|\\\\)",
    re.IGNORECASE,
)


def _is_filesystem_path(val: Any) -> bool:
    if isinstance(val, Path):
        return True
    if isinstance(val, str):
        s = val.strip()
        if _PATH_PATTERN.match(s):
            return True
        if (
            "data/cache" in s
            or "data\\cache" in s
            or "backend/data" in s
            or "backend\\data" in s
            or "data/uploads" in s
            or "data\\uploads" in s
        ):
            return True
        if (s.endswith(".tif") or s.endswith(".tiff")) and ("/" in s or "\\" in s):
            return True
    return False


def sanitize_optical_sar_metadata(data: Any) -> Any:
    """
    Recursively sanitize public metadata by removing local server filesystem paths
    while strictly preserving item_ids, acquisition dates, polarizations, modes,
    spatial coverage, temporal delta, selection reasons, and public URL references.
    """
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            # Skip keys known to store raw server file paths
            if k in (
                "path",
                "sar_path",
                "optical_path",
                "raw_path",
                "filepath",
                "file_path",
                "vh_path",
                "vv_path",
            ):
                continue
            # If key is "vh" or "vv" and value is a path string, omit it
            if k in ("vh", "vv") and _is_filesystem_path(v):
                continue
            # If value is a direct path string or Path object, omit
            if _is_filesystem_path(v):
                continue
            # If key is "bands" and contains path strings, omit
            if k == "bands":
                if isinstance(v, (list, tuple)) and any(_is_filesystem_path(item) for item in v):
                    continue
                if isinstance(v, dict) and any(_is_filesystem_path(item) for item in v.values()):
                    continue
            sanitized[k] = sanitize_optical_sar_metadata(v)
        return sanitized
    elif isinstance(data, list):
        sanitized_list = []
        for item in data:
            if not _is_filesystem_path(item):
                sanitized_list.append(sanitize_optical_sar_metadata(item))
        return sanitized_list
    elif isinstance(data, Path):
        return str(data.name)
    else:
        return data


def _build_optical_sar_api_response(
    plan: QueryPlan,
    query: str,
    sar_res: dict,
    opt_path: Optional[Path] = None,
) -> AnalysisResult:
    """
    Format multimodal Optical-SAR specialist execution output into the unified AnalysisResult schema.
    """
    uid = uuid.uuid4().hex[:8]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    try:
        bounds = _get_raster_bounds(str(opt_path)) if opt_path else None
    except Exception:
        bounds = None

    vis_dict = sar_res.get("visuals", {}) or {}
    layers = []
    images_dict = {}
    primary_vis_url = None

    # 1. Optical RGB
    if vis_dict.get("optical") is not None:
        try:
            opt_fname = f"multimodal_optical_{uid}_{ts}.png"
            vis_dict["optical"].save(VISUALIZATION_DIR / opt_fname)
            opt_url = f"/visualizations/{opt_fname}"
            primary_vis_url = opt_url
            images_dict["optical"] = opt_url
            layers.append({
                "name": "Optical Surface Reflectance",
                "type": "optical_rgb",
                "visualization_url": opt_url,
                "bounds": bounds,
            })
        except Exception as exc:
            print(f"[OPTICAL-SAR VIS WARNING] Optical: {exc}")

    # 2. SAR VV
    if vis_dict.get("s1_vv") is not None:
        try:
            vv_fname = f"multimodal_sar_vv_{uid}_{ts}.png"
            vis_dict["s1_vv"].save(VISUALIZATION_DIR / vv_fname)
            vv_url = f"/visualizations/{vv_fname}"
            images_dict["s1_vv"] = vv_url
            layers.append({
                "name": "Sentinel-1 VV Backscatter",
                "type": "sar_vv",
                "visualization_url": vv_url,
                "bounds": bounds,
            })
        except Exception as exc:
            print(f"[OPTICAL-SAR VIS WARNING] VV: {exc}")

    # 3. SAR VH
    if vis_dict.get("s1_vh") is not None:
        try:
            vh_fname = f"multimodal_sar_vh_{uid}_{ts}.png"
            vis_dict["s1_vh"].save(VISUALIZATION_DIR / vh_fname)
            vh_url = f"/visualizations/{vh_fname}"
            images_dict["s1_vh"] = vh_url
            layers.append({
                "name": "Sentinel-1 VH Backscatter",
                "type": "sar_vh",
                "visualization_url": vh_url,
                "bounds": bounds,
            })
        except Exception as exc:
            print(f"[OPTICAL-SAR VIS WARNING] VH: {exc}")

    # 4. SAR Composite
    if vis_dict.get("s1_composite") is not None:
        try:
            comp_fname = f"multimodal_sar_composite_{uid}_{ts}.png"
            vis_dict["s1_composite"].save(VISUALIZATION_DIR / comp_fname)
            comp_url = f"/visualizations/{comp_fname}"
            images_dict["s1_composite"] = comp_url
            layers.append({
                "name": "Sentinel-1 Polarimetric Composite",
                "type": "sar_composite",
                "visualization_url": comp_url,
                "bounds": bounds,
            })
        except Exception as exc:
            print(f"[OPTICAL-SAR VIS WARNING] Composite: {exc}")

    metadata = sar_res.get("metadata", {}) or {}
    raw_pair_info = sar_res.get("optical_sar_pair") or metadata.get("optical_sar_pair")
    pair_info = sanitize_optical_sar_metadata(raw_pair_info) if raw_pair_info else {"source": "user_supplied"}

    evidence_item = {
        "source": "optical_sar_multimodal",
        "modalities": sar_res.get("modalities", []),
        "grid": metadata.get("grid", {}),
        "optical": metadata.get("optical", {}),
        "sar": metadata.get("sar", {}),
        "evidence_used": sar_res.get("evidence_used", False),
        "optical_sar_pair": pair_info,
    }

    evidence = [sanitize_optical_sar_metadata(evidence_item)]

    if pair_info and pair_info.get("source") == "automatic":
        delta_val = pair_info.get("temporal_delta_days")
        delta_str = f"{delta_val:.2f}" if isinstance(delta_val, (int, float)) else "0.00"
        trace = [
            "Natural-language query classified as optical_sar_analysis",
            "Automatic Optical-SAR mode triggered (AOI and temporal range provided)",
            "Searching Sentinel-2 optical candidates",
            "Searching Sentinel-1 SAR candidates",
            f"Best Optical-SAR pair selected (temporal delta: {delta_str} days)",
            "Verified presence of both Optical and SAR GeoTIFF rasters",
            "Co-registered SAR onto optical reference grid",
            "Generated multi-sensor visual layers for qualitative inspection",
            "Grounding prompt and multimodal inputs evaluated by VLM specialist",
        ]
    else:
        trace = [
            "Natural-language query classified as optical_sar_analysis",
            "Verified presence of both Optical and SAR GeoTIFF rasters",
            "Co-registered SAR onto optical reference grid",
            "Generated multi-sensor visual layers for qualitative inspection",
            "Grounding prompt and multimodal inputs evaluated by VLM specialist",
        ]

    statistics_dict = {
        "task": "optical_sar_analysis",
        "modalities": sar_res.get("modalities", []),
        "grid": metadata.get("grid", {}),
        "optical": metadata.get("optical", {}),
        "sar": metadata.get("sar", {}),
        "optical_sar_pair": pair_info,
    }
    statistics_dict = sanitize_optical_sar_metadata(statistics_dict)

    return AnalysisResult(
        status="success",
        answer=sar_res.get("answer"),
        confidence=0.9,
        plan=plan.model_dump(),
        statistics=statistics_dict,
        layers=layers,
        evidence=evidence,
        execution_trace=trace,
        visualization_url=primary_vis_url,
        bounds=bounds,
        images=images_dict,
    )


def _build_single_image_vqa_api_response(
    plan: QueryPlan,
    query: str,
    vqa_res: dict,
    context: Optional[dict] = None,
) -> AnalysisResult:
    """
    Format single-image VQA specialist execution output into the unified AnalysisResult schema,
    adhering strictly to the standardized evidence, confidence, and execution trace contract.
    """
    context = context or {}
    resolved_modality = (
        vqa_res.get("modality")
        or (context.get("modality") if context.get("modality") and context.get("modality") != "unknown" else None)
        or ("sar" if context.get("sar_image_id") or context.get("sar_path") else None)
        or ("optical" if context.get("optical_image_id") or context.get("optical_path") else None)
        or (plan.modalities[0] if plan.modalities else "unknown")
    )
    if isinstance(resolved_modality, str):
        resolved_modality = resolved_modality.strip().lower()
    else:
        resolved_modality = "unknown"

    # Execution trace reflecting complete agent lifecycle
    trace = [
        "Natural-language query received for visual inspection",
        "Natural-language query classified as single-image VQA",
        "Execution plan created for single-image VQA",
        f"Single image resolved (modality: {resolved_modality})" if resolved_modality != "unknown" else "Single image resolved",
        "VQA specialist executed with grounded prompt",
        "VQA response formatted",
    ]

    stats = {
        "task": "single_image_vqa",
        "question": vqa_res.get("question") or query,
        "answer": vqa_res.get("answer"),
        "modality": resolved_modality,
        "confidence": None,
    }

    # Structured evidence contract (Phase 2)
    raw_evidence = vqa_res.get("evidence") or context.get("evidence")
    if raw_evidence:
        if isinstance(raw_evidence, list):
            sanitized_list = sanitize_optical_sar_metadata(raw_evidence)
            evidence = []
            for item in sanitized_list:
                if isinstance(item, dict):
                    copied = dict(item)
                    if "evidence_used" not in copied:
                        copied["evidence_used"] = True
                    evidence.append(copied)
                else:
                    evidence.append(item)
        elif isinstance(raw_evidence, dict):
            sanitized_dict = sanitize_optical_sar_metadata(raw_evidence)
            evidence_item = {
                "source": sanitized_dict.get("source", "single_image_observation"),
                "modality": sanitized_dict.get("modality", resolved_modality),
                "evidence_used": sanitized_dict.get("evidence_used", True),
                **{k: v for k, v in sanitized_dict.items() if k not in ("source", "modality", "evidence_used")},
            }
            evidence = [evidence_item]
        else:
            evidence = [{
                "source": "single_image_observation",
                "modality": resolved_modality,
                "evidence_used": True,
            }]
    else:
        evidence_item = {
            "source": "single_image_observation",
            "modality": resolved_modality,
            "evidence_used": False,
        }
        evidence = [evidence_item]

    return AnalysisResult(
        status="success",
        answer=vqa_res.get("answer"),
        confidence=None,
        plan=plan.model_dump(),
        statistics=stats,
        layers=[],
        evidence=evidence,
        execution_trace=trace,
    )


# ============================================================
# API ENDPOINT
# ============================================================

@router.post(
    "/query",
    response_model=AnalysisResult,
)
def process_query(
    request: QueryRequest,
):
    """
    Process a natural-language satellite analysis query.

    P2:
        imagery + remote sensing + statistics

    P4:
        multimodal evidence reasoning with VLM
    """

    # ========================================================
    # 1. Build query plan
    # ========================================================

    plan = build_query_plan(
        request
    )

    # ========================================================
    # 2. Create execution plan
    # ========================================================

    tools = create_execution_plan(
        plan
    )

    # --------------------------------------------------------
    # OPTICAL-SAR MULTIMODAL ROUTE
    # --------------------------------------------------------
    if plan.task == "optical_sar_analysis":
        has_explicit = bool(
            getattr(request, "optical_image_id", None)
            or getattr(request, "sar_image_id", None)
            or (request.image_ids and len(request.image_ids) > 0)
        )

        opt_path = None
        sar_path = None

        if has_explicit:
            try:
                opt_path, sar_path = resolve_optical_sar_references(
                    optical_ref=getattr(request, "optical_image_id", None),
                    sar_ref=getattr(request, "sar_image_id", None),
                    image_ids=request.image_ids,
                )
            except (ValueError, FileNotFoundError) as ref_err:
                raise HTTPException(
                    status_code=400,
                    detail=str(ref_err),
                )
        else:
            # Mode B: Automatic Acquisition Validation
            aoi_candidate = plan.aoi or getattr(request, "aoi", None)
            if not aoi_candidate:
                raise HTTPException(
                    status_code=400,
                    detail="Optical-SAR automatic acquisition requires an AOI.",
                )

            time_start_candidate = (
                plan.time_start
                or getattr(request, "time_start", None)
            )
            time_end_candidate = (
                plan.time_end
                or getattr(request, "time_end", None)
            )
            if not time_start_candidate and not time_end_candidate:
                raise HTTPException(
                    status_code=400,
                    detail="Optical-SAR automatic acquisition requires a target date or time range.",
                )

        # Ensure environment is loaded from .env
        load_env_file(BACKEND_DIR / ".env")

        context = {
            "optical_path": str(opt_path) if opt_path else None,
            "sar_path": str(sar_path) if sar_path else None,
            "question": request.query,
            "aoi": plan.aoi or getattr(request, "aoi", None),
            "time_start": plan.time_start or getattr(request, "time_start", None),
            "time_end": plan.time_end or getattr(request, "time_end", None),
            "task": plan.task,
        }

        execution_results = execute_plan(
            tools,
            context=context,
        )

        sar_res = execution_results.get("optical_sar_analysis", {})
        if not sar_res.get("success"):
            err_type = sar_res.get("error_type")
            err_msg = sar_res.get("error") or "Optical-SAR analysis failed."
            if err_type in (
                "NO_OPTICAL_SCENES",
                "NO_SAR_SCENES",
                "NO_TEMPORALLY_COMPATIBLE_PAIR",
                "INSUFFICIENT_SPATIAL_OVERLAP",
            ):
                status_code = 404
            elif err_type in ("MALFORMED_AOI", "INVALID_DATES"):
                status_code = 400
            else:
                status_code = 500
            raise HTTPException(
                status_code=status_code,
                detail=err_msg,
            )

        resolved_opt_path = opt_path or context.get("optical_path")
        return _build_optical_sar_api_response(
            plan=plan,
            query=request.query,
            sar_res=sar_res,
            opt_path=Path(resolved_opt_path) if resolved_opt_path else None,
        )

    # --------------------------------------------------------
    # SINGLE-IMAGE VQA ROUTE
    # --------------------------------------------------------
    if plan.task == "single_image_vqa":
        load_env_file(BACKEND_DIR / ".env")

        context = {
            "query": request.query,
            "question": request.query,
            "aoi": plan.aoi or getattr(request, "aoi", None),
            "task": plan.task,
            "modality": (plan.modalities[0] if plan.modalities else "unknown"),
            "modalities": plan.modalities,
        }

        if request.image_ids and len(request.image_ids) > 0:
            context["image_ids"] = request.image_ids
            context["image_id"] = request.image_ids[0]
        if getattr(request, "optical_image_id", None):
            context["optical_image_id"] = request.optical_image_id
            context["image_id"] = request.optical_image_id
        if getattr(request, "sar_image_id", None):
            context["sar_image_id"] = request.sar_image_id
            context["image_id"] = request.sar_image_id

        execution_results = execute_plan(
            tools,
            context=context,
        )

        vqa_res = execution_results.get("single_image_vqa", {})
        return _build_single_image_vqa_api_response(
            plan=plan,
            query=request.query,
            vqa_res=vqa_res,
            context=context,
        )

    # ========================================================
    # Pass target, metric, task in context so executor tools know primary metric
    target_metric = (
        (plan.metric.upper() if plan.metric else None)
        or ("NDBI" if (plan.target == "urban" or (plan.task and "urban" in plan.task.lower())) else
            "NDWI" if (plan.target == "water" or (plan.task and "water" in plan.task.lower())) else
            "NDVI")
    )

    execution_results = execute_plan(
        tools,
        context={
            "time_start": plan.time_start,
            "time_end": plan.time_end,
            "aoi": plan.aoi,
            "metric": plan.metric or target_metric,
            "explicit_metric": getattr(plan, "explicit_metric", None),
            "target": plan.target,
            "task": plan.task,
            "temporal_mode": getattr(plan, "temporal_mode", "bi_temporal"),
        },
    )

    # ========================================================
    # 4. Build statistics
    # ========================================================

    statistics = {}

    # --------------------------------------------------------
    # Single-image index result
    # --------------------------------------------------------

    index_result = None

    if f"calculate_{target_metric.lower()}" in execution_results:
        index_result = execution_results[f"calculate_{target_metric.lower()}"]
    elif "calculate_ndvi" in execution_results:
        index_result = execution_results["calculate_ndvi"]
    elif "calculate_ndwi" in execution_results:
        index_result = execution_results["calculate_ndwi"]
    elif "calculate_ndbi" in execution_results:
        index_result = execution_results["calculate_ndbi"]

    # --------------------------------------------------------
    # Temporal index result
    # --------------------------------------------------------

    temporal_result = (
        execution_results.get(f"calculate_temporal_{target_metric.lower()}")
        or execution_results.get("calculate_temporal_ndvi")
        or execution_results.get("calculate_temporal_ndwi")
        or execution_results.get("calculate_temporal_ndbi")
    )

    # ========================================================
    # 5. Extract index statistics
    # ========================================================

    if temporal_result:

        index_name = temporal_result.get(
            "index"
        )

        statistics["metric"] = index_name

        if index_name == "NDVI":

            statistics["mean_before"] = (
                temporal_result.get(
                    "mean_ndvi_before"
                )
            )

            statistics["mean_after"] = (
                temporal_result.get(
                    "mean_ndvi_after"
                )
            )

            statistics["mean_change"] = (
                temporal_result.get(
                    "mean_ndvi_change"
                )
            )

        elif index_name == "NDWI":

            statistics["mean_before"] = (
                temporal_result.get(
                    "mean_ndwi_before"
                )
            )

            statistics["mean_after"] = (
                temporal_result.get(
                    "mean_ndwi_after"
                )
            )

            statistics["mean_change"] = (
                temporal_result.get(
                    "mean_ndwi_change"
                )
            )

        elif index_name == "NDBI":

            statistics["mean_before"] = (
                temporal_result.get(
                    "mean_ndbi_before"
                )
            )

            statistics["mean_after"] = (
                temporal_result.get(
                    "mean_ndbi_after"
                )
            )

            statistics["mean_change"] = (
                temporal_result.get(
                    "mean_ndbi_change"
                )
            )

    elif index_result:

        index_name = index_result.get(
            "index"
        )

        statistics["metric"] = index_name

        statistics["mean"] = (
            index_result.get(
                "mean"
            )
        )

        statistics["min_value"] = (
            index_result.get(
                "min_value"
            )
        )

        statistics["max_value"] = (
            index_result.get(
                "max_value"
            )
        )

        statistics["valid_pixels"] = (
            index_result.get(
                "valid_pixels"
            )
        )

        statistics["total_pixels"] = (
            index_result.get(
                "total_pixels"
            )
        )

    # ========================================================
    # 6. Change detection statistics
    # ========================================================

    change_result = execution_results.get(
        "detect_change"
    )

    if change_result:

        statistics["metric"] = change_result.get("metric") or target_metric

        statistics["mean_before"] = (
            change_result.get(
                "mean_before"
            )
        )

        statistics["mean_after"] = (
            change_result.get(
                "mean_after"
            )
        )

        statistics["mean_change"] = (
            change_result.get(
                "mean_change"
            )
        )

        statistics["min_value"] = (
            change_result.get(
                "min_value"
            )
        )

        statistics["max_value"] = (
            change_result.get(
                "max_value"
            )
        )

        statistics["changed_pixels"] = (
            change_result.get(
                "changed_pixels"
            )
        )

        statistics["valid_pixels"] = (
            change_result.get(
                "valid_pixels"
            )
        )

        statistics["total_pixels"] = (
            change_result.get(
                "total_pixels"
            )
        )

        statistics["change_ratio"] = (
            change_result.get(
                "change_ratio"
            )
        )

        statistics["increased_pixels"] = (
            change_result.get(
                "increased_pixels"
            )
        )

        statistics["decreased_pixels"] = (
            change_result.get(
                "decreased_pixels"
            )
        )

        statistics["change_type"] = (
            change_result.get(
                "change_type"
            )
        )

        statistics["threshold"] = (
            change_result.get(
                "threshold"
            )
        )

    # --------------------------------------------------------
    # Extract statistics for all temporal indices
    # --------------------------------------------------------
    indices_stats = {}
    for idx_key, tool_key in [
        ("NDVI", "calculate_temporal_ndvi"),
        ("NDWI", "calculate_temporal_ndwi"),
        ("NDBI", "calculate_temporal_ndbi"),
    ]:
        t_res = execution_results.get(tool_key)
        if t_res:
            ik = idx_key.lower()
            istat = {
                "metric": idx_key,
                "mean_before": t_res.get(f"mean_{ik}_before"),
                "mean_after": t_res.get(f"mean_{ik}_after"),
                "mean_change": t_res.get(f"mean_{ik}_change"),
                "min_before": t_res.get(f"min_{ik}_before"),
                "max_before": t_res.get(f"max_{ik}_before"),
                "min_after": t_res.get(f"min_{ik}_after"),
                "max_after": t_res.get(f"max_{ik}_after"),
                "valid_pixels": t_res.get("valid_pixels"),
                "total_pixels": t_res.get("total_pixels"),
            }
            if change_result and change_result.get("all_changes"):
                chg = change_result["all_changes"].get(idx_key) or change_result["all_changes"].get(idx_key.lower())
                if chg:
                    istat.update({
                        "change_ratio": chg.get("change_ratio"),
                        "changed_pixels": chg.get("changed_pixels"),
                        "increased_pixels": chg.get("increased_pixels"),
                        "decreased_pixels": chg.get("decreased_pixels"),
                        "change_type": chg.get("change_type"),
                        "threshold": chg.get("threshold"),
                    })
            indices_stats[idx_key] = istat
    if indices_stats:
        statistics["indices"] = indices_stats

    # ========================================================
    # 7. Build P2 explanation
    # ========================================================

    explanation = build_explanation(
        plan,
        statistics,
        execution_results,
    )

    statistics["explanation"] = explanation

    # Keep this explicitly available to P4.
    statistics["backend_explanation"] = (
        explanation
    )

    # ========================================================
    # 8. Build imagery evidence
    # ========================================================

    evidence = []

    imagery_result = execution_results.get(
        "search_imagery"
    )

    if imagery_result:

        evidence.append(
            {
                "source": imagery_result.get(
                    "source"
                ),
                "images": imagery_result.get(
                    "images",
                    [],
                ),
            }
        )
    # ========================================================
    # 9. Build layers
    # ========================================================

    layers = []
    visualization_url = None
    classified_visualization_url = None
    visualization_bounds = None

    if change_result:

        change_map = change_result.get(
            "change_map"
        )

        visualization = change_result.get(
            "visualization"
        )

        if (
            visualization
            and visualization.get("status") != "error"
        ):
            visualization_url = _safe_vis_url(
                visualization.get("filename")
            )

            classified_visualization_url = _safe_vis_url(
                visualization.get("classified_filename")
            )

            visualization_bounds = (
                visualization.get("bounds")
            )

        # Fallback: if the executor did not attach
        # visualization metadata, create one here if valid pixels exist.
        elif change_map is not None and change_result.get("valid_pixels", 0) > 0:

            filename = (
                f"{plan.target or 'image'}_"
                f"{plan.time_start}_"
                f"{plan.time_end}_change.png"
            )

            try:
                save_change_map(
                    change_map=change_map,
                    filename=filename,
                )
                visualization_url = _safe_vis_url(
                    filename
                )
            except Exception as exc:
                print(
                    "[VISUALIZATION WARNING] "
                    f"{type(exc).__name__}: {exc}"
                )

        if not visualization_bounds and change_result.get("bounds"):
            visualization_bounds = change_result.get("bounds")

        if not visualization_bounds and plan.aoi:
            try:
                from app.remote_sensing.providers.sentinel2 import normalize_aoi
                w, s, e, n = normalize_aoi(plan.aoi)
                visualization_bounds = [[float(s), float(w)], [float(n), float(e)]]
            except Exception:
                pass

        metric_name = str(statistics.get("metric") or plan.metric or target_metric).upper()
        # Primary change detection layer
        layers.append(
            {
                "id": "change_continuous",
                "category": "change",
                "type": "change_detection",
                "name": f"{metric_name} Change",
                "classified_name": f"{metric_name} Change Categories",
                "change_ratio": change_result.get("change_ratio"),
                "regions_detected": change_result.get("regions_detected"),
                "changed_pixels": change_result.get("changed_pixels"),
                "visualization_url": visualization_url,
                "classified_visualization_url": classified_visualization_url,
                "bounds": visualization_bounds,
            }
        )

        # Specific index change layers: change_ndvi, change_ndwi, change_ndbi
        all_chgs = change_result.get("all_changes", {})
        for idx_k in ["NDVI", "NDWI", "NDBI"]:
            c_info = all_chgs.get(idx_k) or all_chgs.get(idx_k.lower())
            if c_info and c_info.get("visualization"):
                c_vis = c_info["visualization"]
                layers.append({
                    "id": f"change_{idx_k.lower()}",
                    "category": "change",
                    "type": "change_detection",
                    "metric": idx_k,
                    "name": f"{idx_k} Change",
                    "classified_name": f"{idx_k} Change Categories",
                    "change_ratio": c_info.get("change_ratio"),
                    "regions_detected": c_info.get("regions_detected"),
                    "changed_pixels": c_info.get("changed_pixels"),
                    "visualization_url": _safe_vis_url(c_vis.get("filename")),
                    "classified_visualization_url": _safe_vis_url(c_vis.get("classified_filename")),
                    "bounds": c_vis.get("bounds") or visualization_bounds,
                })

    elif index_result:

        visualization = index_result.get(
            "visualization"
        )

        if (
            visualization
            and visualization.get("status") != "error"
        ):
            visualization_url = _safe_vis_url(
                visualization.get("filename")
            )

            classified_visualization_url = _safe_vis_url(
                visualization.get("classified_filename")
            )

            visualization_bounds = (
                visualization.get("bounds")
            )

        if not visualization_bounds and plan.aoi:
            try:
                from app.remote_sensing.providers.sentinel2 import normalize_aoi
                w, s, e, n = normalize_aoi(plan.aoi)
                visualization_bounds = [[float(s), float(w)], [float(n), float(e)]]
            except Exception:
                pass

        metric_name = str(statistics.get("metric") or plan.metric or target_metric).upper()
        layers.append(
            {
                "id": "index_primary",
                "category": "analysis",
                "type": "index_map",
                "name": f"{metric_name} Map",
                "mean": statistics.get("mean"),
                "min_value": statistics.get("min_value"),
                "max_value": statistics.get("max_value"),
                "valid_pixels": statistics.get("valid_pixels"),
                "total_pixels": statistics.get("total_pixels"),
                "visualization_url": visualization_url,
                "classified_visualization_url": classified_visualization_url,
                "bounds": visualization_bounds,
            }
        )

    # --------------------------------------------------------
    # Collect additional base, index, and quality layers
    # --------------------------------------------------------
    images = []
    vis_b = {}
    vis_a = {}
    if imagery_result and imagery_result.get("images"):
        images = imagery_result["images"]
        if len(images) > 0 and images[0].get("visualizations"):
            vis_b = images[0]["visualizations"]
            date_b = images[0].get("date") or plan.time_start or "Before"
            if vis_b.get("true_color"):
                layers.append({
                    "id": "true_color_before",
                    "category": "base",
                    "type": "true_color",
                    "name": f"True Color ({date_b})",
                    "date": date_b,
                    "visualization_url": _safe_vis_url(vis_b["true_color"].get("filename")),
                    "bounds": vis_b["true_color"].get("bounds") or visualization_bounds,
                    "metadata": images[0].get("metadata"),
                })
            if vis_b.get("false_color"):
                layers.append({
                    "id": "false_color_before",
                    "category": "base",
                    "type": "false_color",
                    "name": f"False Color NIR ({date_b})",
                    "date": date_b,
                    "visualization_url": _safe_vis_url(vis_b["false_color"].get("filename")),
                    "bounds": vis_b["false_color"].get("bounds") or visualization_bounds,
                })
            if vis_b.get("quality_mask"):
                layers.append({
                    "id": "quality_mask_before",
                    "category": "quality",
                    "type": "quality_mask",
                    "name": f"Quality Mask ({date_b})",
                    "date": date_b,
                    "visualization_url": _safe_vis_url(vis_b["quality_mask"].get("filename")),
                    "bounds": vis_b["quality_mask"].get("bounds") or visualization_bounds,
                })
        if len(images) > 1 and images[1].get("visualizations"):
            vis_a = images[1]["visualizations"]
            date_a = images[1].get("date") or plan.time_end or "After"
            if vis_a.get("true_color"):
                layers.append({
                    "id": "true_color_after",
                    "category": "base",
                    "type": "true_color",
                    "name": f"True Color ({date_a})",
                    "date": date_a,
                    "visualization_url": _safe_vis_url(vis_a["true_color"].get("filename")),
                    "bounds": vis_a["true_color"].get("bounds") or visualization_bounds,
                    "metadata": images[1].get("metadata"),
                })
            if vis_a.get("false_color"):
                layers.append({
                    "id": "false_color_after",
                    "category": "base",
                    "type": "false_color",
                    "name": f"False Color NIR ({date_a})",
                    "date": date_a,
                    "visualization_url": _safe_vis_url(vis_a["false_color"].get("filename")),
                    "bounds": vis_a["false_color"].get("bounds") or visualization_bounds,
                })
            if vis_a.get("quality_mask"):
                layers.append({
                    "id": "quality_mask_after",
                    "category": "quality",
                    "type": "quality_mask",
                    "name": f"Quality Mask ({date_a})",
                    "date": date_a,
                    "visualization_url": _safe_vis_url(vis_a["quality_mask"].get("filename")),
                    "bounds": vis_a["quality_mask"].get("bounds") or visualization_bounds,
                })

    date_b = plan.time_start or "Before"
    date_a = plan.time_end or "After"
    primary_idx = str(statistics.get("metric") or plan.metric or target_metric).upper()

    for idx_key, tool_key in [
        ("NDVI", "calculate_temporal_ndvi"),
        ("NDWI", "calculate_temporal_ndwi"),
        ("NDBI", "calculate_temporal_ndbi"),
    ]:
        t_res = execution_results.get(tool_key)
        if t_res and t_res.get("visualizations"):
            t_vis = t_res["visualizations"]
            b_vis = t_vis.get("before")
            a_vis = t_vis.get("after")
            b_url = _safe_vis_url(b_vis.get("filename")) if b_vis else None
            a_url = _safe_vis_url(a_vis.get("filename")) if a_vis else None
            b_bounds = (b_vis.get("bounds") if b_vis else None) or visualization_bounds
            a_bounds = (a_vis.get("bounds") if a_vis else None) or visualization_bounds

            if b_url:
                layers.append({
                    "id": f"{idx_key.lower()}_before",
                    "category": "analysis",
                    "type": "index_map",
                    "metric": idx_key,
                    "name": f"{idx_key} ({date_b})",
                    "visualization_url": b_url,
                    "bounds": b_bounds,
                })
            if a_url:
                layers.append({
                    "id": f"{idx_key.lower()}_after",
                    "category": "analysis",
                    "type": "index_map",
                    "metric": idx_key,
                    "name": f"{idx_key} ({date_a})",
                    "visualization_url": a_url,
                    "bounds": a_bounds,
                })

            if idx_key == primary_idx:
                if b_url:
                    layers.append({
                        "id": "index_before",
                        "category": "analysis",
                        "type": "index_map",
                        "name": f"{idx_key} ({date_b})",
                        "visualization_url": b_url,
                        "bounds": b_bounds,
                    })
                if a_url:
                    layers.append({
                        "id": "index_after",
                        "category": "analysis",
                        "type": "index_map",
                        "name": f"{idx_key} ({date_a})",
                        "visualization_url": a_url,
                        "bounds": a_bounds,
                    })

    # ========================================================
    # 9b. Build layer_package
    # ========================================================
    t_ndvi = execution_results.get("calculate_temporal_ndvi", {}).get("visualizations", {})
    t_ndwi = execution_results.get("calculate_temporal_ndwi", {}).get("visualizations", {})
    t_ndbi = execution_results.get("calculate_temporal_ndbi", {}).get("visualizations", {})
    all_chg = change_result.get("all_changes", {}) if change_result else {}

    layer_package = {
        "before": {
            "true_color": {
                "url": _safe_vis_url(vis_b.get("true_color", {}).get("filename")) if vis_b.get("true_color") else None,
                "bounds": vis_b.get("true_color", {}).get("bounds") or visualization_bounds if vis_b.get("true_color") else visualization_bounds,
            },
            "false_color": {
                "url": _safe_vis_url(vis_b.get("false_color", {}).get("filename")) if vis_b.get("false_color") else None,
                "bounds": vis_b.get("false_color", {}).get("bounds") or visualization_bounds if vis_b.get("false_color") else visualization_bounds,
            },
            "ndvi": {
                "url": _safe_vis_url(t_ndvi.get("before", {}).get("filename")) if t_ndvi.get("before") else None,
                "bounds": t_ndvi.get("before", {}).get("bounds") or visualization_bounds if t_ndvi.get("before") else visualization_bounds,
            },
            "ndwi": {
                "url": _safe_vis_url(t_ndwi.get("before", {}).get("filename")) if t_ndwi.get("before") else None,
                "bounds": t_ndwi.get("before", {}).get("bounds") or visualization_bounds if t_ndwi.get("before") else visualization_bounds,
            },
            "ndbi": {
                "url": _safe_vis_url(t_ndbi.get("before", {}).get("filename")) if t_ndbi.get("before") else None,
                "bounds": t_ndbi.get("before", {}).get("bounds") or visualization_bounds if t_ndbi.get("before") else visualization_bounds,
            },
        },
        "after": {
            "true_color": {
                "url": _safe_vis_url(vis_a.get("true_color", {}).get("filename")) if vis_a.get("true_color") else None,
                "bounds": vis_a.get("true_color", {}).get("bounds") or visualization_bounds if vis_a.get("true_color") else visualization_bounds,
            },
            "false_color": {
                "url": _safe_vis_url(vis_a.get("false_color", {}).get("filename")) if vis_a.get("false_color") else None,
                "bounds": vis_a.get("false_color", {}).get("bounds") or visualization_bounds if vis_a.get("false_color") else visualization_bounds,
            },
            "ndvi": {
                "url": _safe_vis_url(t_ndvi.get("after", {}).get("filename")) if t_ndvi.get("after") else None,
                "bounds": t_ndvi.get("after", {}).get("bounds") or visualization_bounds if t_ndvi.get("after") else visualization_bounds,
            },
            "ndwi": {
                "url": _safe_vis_url(t_ndwi.get("after", {}).get("filename")) if t_ndwi.get("after") else None,
                "bounds": t_ndwi.get("after", {}).get("bounds") or visualization_bounds if t_ndwi.get("after") else visualization_bounds,
            },
            "ndbi": {
                "url": _safe_vis_url(t_ndbi.get("after", {}).get("filename")) if t_ndbi.get("after") else None,
                "bounds": t_ndbi.get("after", {}).get("bounds") or visualization_bounds if t_ndbi.get("after") else visualization_bounds,
            },
        },
        "change": {
            "delta_ndvi": {
                "url": _safe_vis_url(((all_chg.get("NDVI") or all_chg.get("ndvi") or {}).get("visualization") or {}).get("filename")),
                "classified_url": _safe_vis_url(((all_chg.get("NDVI") or all_chg.get("ndvi") or {}).get("visualization") or {}).get("classified_filename")),
                "bounds": ((all_chg.get("NDVI") or all_chg.get("ndvi") or {}).get("bounds")) or visualization_bounds,
            },
            "delta_ndwi": {
                "url": _safe_vis_url(((all_chg.get("NDWI") or all_chg.get("ndwi") or {}).get("visualization") or {}).get("filename")),
                "classified_url": _safe_vis_url(((all_chg.get("NDWI") or all_chg.get("ndwi") or {}).get("visualization") or {}).get("classified_filename")),
                "bounds": ((all_chg.get("NDWI") or all_chg.get("ndwi") or {}).get("bounds")) or visualization_bounds,
            },
            "delta_ndbi": {
                "url": _safe_vis_url(((all_chg.get("NDBI") or all_chg.get("ndbi") or {}).get("visualization") or {}).get("filename")),
                "classified_url": _safe_vis_url(((all_chg.get("NDBI") or all_chg.get("ndbi") or {}).get("visualization") or {}).get("classified_filename")),
                "bounds": ((all_chg.get("NDBI") or all_chg.get("ndbi") or {}).get("bounds")) or visualization_bounds,
            },
        },
        "quality": {
            "mask_before": {
                "url": _safe_vis_url(vis_b.get("quality_mask", {}).get("filename")) if vis_b.get("quality_mask") else None,
                "bounds": vis_b.get("quality_mask", {}).get("bounds") or visualization_bounds if vis_b.get("quality_mask") else visualization_bounds,
                "metadata": images[0].get("metadata") if len(images) > 0 else None,
            },
            "mask_after": {
                "url": _safe_vis_url(vis_a.get("quality_mask", {}).get("filename")) if vis_a.get("quality_mask") else None,
                "bounds": vis_a.get("quality_mask", {}).get("bounds") or visualization_bounds if vis_a.get("quality_mask") else visualization_bounds,
                "metadata": images[1].get("metadata") if len(images) > 1 else None,
            },
        }
    }

    # ========================================================
    # 10. Build execution trace
    # ========================================================

    execution_trace = [
        "Query received",
        f"Task identified: {plan.task}",
        (
            "Execution plan created: "
            f"{tools}"
        ),
    ]

    for tool_name in tools:
        execution_trace.append(
            f"Executed: {tool_name}"
        )

    execution_trace.append(
        "Statistics calculated"
    )

    execution_trace.append(
        "Explanation generated"
    )

    # ========================================================
    # 10b. Build Multi-Index Evidence (Phase 5A)
    # ========================================================
    multi_index_evidence = calculate_multi_index_evidence(
        target=plan.target,
        task=plan.task,
        execution_results=execution_results,
        imagery_result=imagery_result,
        change_result=change_result,
    )
    statistics["evidence"] = multi_index_evidence
    execution_trace.append("Multi-index evidence calculated")

    # ========================================================
    # 10c. Evidence Fusion & Candidate Classification (Phase 5B)
    # ========================================================
    candidate_package = fuse_evidence_and_classify_candidates(
        target=plan.target,
        task=plan.task,
        multi_index_evidence=multi_index_evidence,
        execution_results=execution_results,
        imagery_result=imagery_result,
        change_result=change_result,
    )
    candidates = candidate_package.get("candidates", [])
    statistics["candidates"] = candidates
    statistics["fusion_statistics"] = candidate_package.get("statistics", {})
    execution_trace.append("Phase 5B Evidence fusion and candidate classification completed")

    # ========================================================
    # 10d. Spatial Reasoning & Candidate Region Clustering (Phase 6)
    # ========================================================
    cand_raster_path = candidate_package.get("candidate_raster")
    spatial_analysis = extract_spatial_candidate_regions(
        candidate_raster_path=cand_raster_path,
        target=plan.target,
        task=plan.task,
        execution_results=execution_results,
        imagery_result=imagery_result,
        aoi_bounds=visualization_bounds,
    )
    statistics["spatial_analysis"] = spatial_analysis
    execution_trace.append("Phase 6 Spatial candidate clustering completed")

    # Expose spatial layers in layer_package
    if spatial_analysis.get("available"):
        sp_rasters = spatial_analysis.get("rasters", {})
        filt_r = sp_rasters.get("filtered_candidate_raster")
        lbl_r = sp_rasters.get("labeled_regions_raster")
        layer_package["spatial"] = {
            "raw_candidate_url": _safe_vis_url(Path(cand_raster_path).name) if cand_raster_path else None,
            "filtered_candidate_url": _safe_vis_url(Path(filt_r).name) if filt_r else None,
            "labeled_regions_url": _safe_vis_url(Path(lbl_r).name) if lbl_r else None,
            "bounds": visualization_bounds,
            "region_count": spatial_analysis.get("region_count", 0),
            "total_candidate_area_hectares": spatial_analysis.get("total_candidate_area_hectares", 0.0),
            "geojson": spatial_analysis.get("geojson"),
        }

    # ========================================================
    # 10e. Temporal Reasoning & Multi-Observation Analysis (Phase 7)
    # ========================================================
    temporal_obs: list[TemporalObservation] = []
    if imagery_result and imagery_result.get("temporal_observations"):
        for t_dict in imagery_result["temporal_observations"]:
            temporal_obs.append(TemporalObservation(
                observation_id=t_dict.get("observation_id", ""),
                scene_id=t_dict.get("scene_id", ""),
                datetime_iso=t_dict.get("datetime_iso", ""),
                date=t_dict.get("date", ""),
                year=int(t_dict.get("year", 2021)),
                day_of_year=int(t_dict.get("day_of_year", 182)),
                cloud_cover=float(t_dict.get("cloud_cover", 0.0)),
                coverage_fraction=float(t_dict.get("coverage_fraction", 1.0)),
                valid_fraction=float(t_dict.get("valid_fraction", 1.0)),
                quality_state=t_dict.get("quality_state", "high"),
                acquisition_score=float(t_dict.get("acquisition_score", 1.0)),
                provenance=t_dict.get("provenance", {}),
                ndvi_mean=t_dict.get("ndvi_mean"),
                ndvi_median=t_dict.get("ndvi_median"),
                ndvi_std=t_dict.get("ndvi_std"),
                ndwi_mean=t_dict.get("ndwi_mean"),
                ndwi_median=t_dict.get("ndwi_median"),
                ndwi_std=t_dict.get("ndwi_std"),
                ndbi_mean=t_dict.get("ndbi_mean"),
                ndbi_median=t_dict.get("ndbi_median"),
                ndbi_std=t_dict.get("ndbi_std"),
                band_paths=t_dict.get("band_paths", {}),
            ))
    elif imagery_result and imagery_result.get("images") and len(imagery_result["images"]) >= 2:
        imgs = imagery_result["images"]
        for idx, img in enumerate([imgs[0], imgs[-1]]):
            i_date = img.get("date") or (plan.time_start if idx == 0 else plan.time_end) or "2021-06-15"
            i_dt_iso = f"{i_date}T00:00:00Z"
            i_yr = int(i_date[:4]) if len(i_date) >= 4 and i_date[:4].isdigit() else 2021
            try:
                i_doy = datetime.fromisoformat(i_dt_iso.replace("Z", "+00:00")).timetuple().tm_yday
            except Exception:
                i_doy = 182
            q_info = img.get("quality", {})
            v_pct = float(q_info.get("valid_coverage_percentage") or q_info.get("valid_percentage") or 100.0) / 100.0
            
            # Map index means from indices statistics
            idx_stats = statistics.get("indices", {})
            ndvi_s = idx_stats.get("NDVI", {})
            ndwi_s = idx_stats.get("NDWI", {})
            ndbi_s = idx_stats.get("NDBI", {})

            key_m = "mean_before" if idx == 0 else "mean_after"
            v_ndvi = ndvi_s.get(key_m) if ndvi_s.get(key_m) is not None else (statistics.get(key_m) if statistics.get("metric") == "NDVI" else None)
            v_ndwi = ndwi_s.get(key_m) if ndwi_s.get(key_m) is not None else (statistics.get(key_m) if statistics.get("metric") == "NDWI" else None)
            v_ndbi = ndbi_s.get(key_m) if ndbi_s.get(key_m) is not None else (statistics.get(key_m) if statistics.get("metric") == "NDBI" else None)

            temporal_obs.append(TemporalObservation(
                observation_id=img.get("id") or f"obs_{idx}",
                scene_id=img.get("id") or f"obs_{idx}",
                datetime_iso=i_dt_iso,
                date=i_date,
                year=i_yr,
                day_of_year=i_doy,
                cloud_cover=float(img.get("cloud_cover", 0.0)),
                coverage_fraction=1.0,
                valid_fraction=v_pct,
                quality_state=q_info.get("quality_state", "high"),
                acquisition_score=1.0,
                provenance=img.get("metadata", {}),
                ndvi_mean=float(v_ndvi) if v_ndvi is not None else None,
                ndwi_mean=float(v_ndwi) if v_ndwi is not None else None,
                ndbi_mean=float(v_ndbi) if v_ndbi is not None else None,
                band_paths=img.get("bands", {}),
            ))

    temporal_analysis = build_temporal_analysis_package(
        observations=temporal_obs,
        target=plan.target,
        task=plan.task,
        spatial_analysis=spatial_analysis,
    )
    statistics["temporal_analysis"] = temporal_analysis
    execution_trace.append("Phase 7 Temporal reasoning completed")

    # Expose temporal layer in layer_package
    if temporal_analysis.get("available"):
        pix_pers = temporal_analysis.get("pixel_persistence", {})
        if pix_pers.get("available"):
            layer_package["temporal"] = {
                "persistence_raster_url": _safe_vis_url(pix_pers.get("raster_filename")),
                "bounds": visualization_bounds,
                "classes": pix_pers.get("classes"),
                "fractions": pix_pers.get("fractions"),
                "observation_count": temporal_analysis.get("observation_count"),
                "usable_observation_count": temporal_analysis.get("usable_observation_count"),
                "seasonal_comparability": temporal_analysis.get("seasonal_comparability", {}).get("comparability"),
            }

    # ========================================================
    # 10f. Reliability & Confidence Calibration (Phase 8)
    # ========================================================
    calibration = build_calibration_package(
        candidate_package=candidate_package,
        multi_index_evidence=multi_index_evidence,
        spatial_analysis=spatial_analysis,
        temporal_analysis=temporal_analysis,
        imagery_result=imagery_result,
        execution_results=execution_results,
        temporal_observations=temporal_obs,
        target=plan.target,
        task=plan.task,
        temporal_mode=plan.temporal_mode or "bi_temporal",
    )
    statistics["calibration"] = calibration
    execution_trace.append("Phase 8 Reliability and confidence calibration completed")

    # ========================================================
    # 10g. Structured Interpretation & Grounded Explanation (Phase 5C + 6 + 7 + 8)
    # ========================================================
    interpretation = generate_structured_interpretation(
        candidate_package=candidate_package,
        multi_index_evidence=multi_index_evidence,
        target=plan.target,
        task=plan.task,
        spatial_analysis=spatial_analysis,
        temporal_analysis=temporal_analysis,
        calibration=calibration,
    )
    statistics["interpretation"] = interpretation
    execution_trace.append("Phase 5C/6/7/8 Structured interpretation generated")

    if change_result and change_result.get("valid_pixels", 0) == 0:
        grounded_explanation = explanation
    else:
        grounded_explanation = interpretation.get("summary") or explanation

    # ========================================================
    # 11. P4 VLM
    # ========================================================

    p2_response = {
        "status": "success",
        "answer": grounded_explanation,
        "confidence": 0.9,
        "query": request.query,
        "plan": plan.model_dump(),
        "statistics": statistics,
        "layers": layers,
        "layer_package": layer_package,
        "multi_index_evidence": multi_index_evidence,
        "candidates": candidates,
        "candidate_package": candidate_package,
        "interpretation": interpretation,
        "spatial_analysis": spatial_analysis,
        "temporal_analysis": temporal_analysis,
        "calibration": calibration,
        "evidence": evidence,
        "execution_trace": execution_trace,
        "execution_tools": tools,
        "visualization_url": visualization_url,
        "classified_visualization_url": classified_visualization_url,
        "bounds": visualization_bounds,
    }

    evidence_package = build_evidence(p2_response)
    evidence_package["multi_index_evidence"] = multi_index_evidence
    evidence_package["candidates"] = candidates
    evidence_package["candidate_package"] = candidate_package
    evidence_package["interpretation"] = interpretation
    evidence_package["spatial_analysis"] = spatial_analysis
    evidence_package["temporal_analysis"] = temporal_analysis
    evidence_package["calibration"] = calibration
    p2_response["evidence_package"] = evidence_package

    if change_result and change_result.get("valid_pixels", 0) == 0:
        vlm_answer = None
        execution_trace.append("Zero valid pixels: using grounded explanation")
    else:
        vlm_answer = generate_vlm_answer(
            request=request,
            p2_response=p2_response,
        )

        if vlm_answer:
            execution_trace.append(
                "P4 VLM inference completed"
            )
        else:
            execution_trace.append(
                "P4 VLM unavailable; using backend explanation"
            )

    # ========================================================
    # 12. Final API response
    # ========================================================

    final_answer = (
        vlm_answer
        if vlm_answer
        else grounded_explanation
    )

    return AnalysisResult(
        status="success",
        answer=final_answer,
        confidence=0.9,
        plan=plan.model_dump(),
        statistics=sanitize_optical_sar_metadata(statistics),
        layers=layers,
        layer_package=layer_package,
        multi_index_evidence=multi_index_evidence,
        candidates=candidates,
        candidate_package=sanitize_optical_sar_metadata(candidate_package),
        interpretation=interpretation,
        spatial_analysis=sanitize_optical_sar_metadata(spatial_analysis),
        temporal_analysis=sanitize_optical_sar_metadata(temporal_analysis),
        calibration=sanitize_optical_sar_metadata(calibration),
        evidence=sanitize_optical_sar_metadata(evidence),
        execution_trace=execution_trace,
        visualization_url=visualization_url,
        classified_visualization_url=classified_visualization_url,
        bounds=visualization_bounds,
        evidence_package=sanitize_optical_sar_metadata(evidence_package),
    )
