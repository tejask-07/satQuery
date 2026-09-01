import re
from typing import Optional

from fastapi import APIRouter
from app.evidence.visualizations import save_change_map, VISUALIZATION_DIR
from app.agent.executor import (
    execute_plan,
    _save_change_map_visualization,
)
from app.agent.planner import create_execution_plan
from app.schemas.analysis import AnalysisResult
from app.schemas.query import QueryPlan, QueryRequest


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


# ============================================================
# QUERY PLANNER
# ============================================================

def build_query_plan(
    request: QueryRequest,
) -> QueryPlan:
    """
    Convert natural-language query into a QueryPlan.

    Lightweight rule-based planner for the MVP.
    """

    query_str = request.query
    parsed_aoi = None
    import re
    aoi_match = re.search(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', query_str)
    if aoi_match:
        v1, v2, v3, v4 = [float(x) for x in aoi_match.groups()]
        minLon = min(v1, v3)
        maxLon = max(v1, v3)
        minLat = min(v2, v4)
        maxLat = max(v2, v4)
        parsed_aoi = {
            "type": "Polygon",
            "coordinates": [[
                [minLon, minLat],
                [maxLon, minLat],
                [maxLon, maxLat],
                [minLon, maxLat],
                [minLon, minLat]
            ]]
        }
        query_str = query_str[:aoi_match.start()] + query_str[aoi_match.end():]
        query_str = re.sub(r' for\s+aoi\b', '', query_str, flags=re.IGNORECASE).strip()
        query_str = re.sub(r'\baoi\b', '', query_str, flags=re.IGNORECASE).strip()

    final_aoi = request.aoi if request.aoi else parsed_aoi
    query = query_str.lower()

    # --------------------------------------------------------
    # Extract years.
    # --------------------------------------------------------

    years = re.findall(
        r"\b(?:19|20)\d{2}\b",
        query,
    )

    if len(years) >= 2:
        time_start = years[0]
        time_end = years[1]
    elif len(years) == 1:
        time_start = years[0]
        time_end = years[0]
    else:
        time_start = "2021"
        time_end = "2025"

    is_change_query = any(
        word in query
        for word in [
            "change",
            "compare",
            "between",
            "difference",
            "decrease",
            "increase",
            "loss",
            "gain",
        ]
    )

    # --------------------------------------------------------
    # Single index tasks
    # --------------------------------------------------------

    if (
        "ndvi" in query
        and not is_change_query
        and ("single" in query or "calculate" in query or len(years) <= 1)
    ):
        task = "vegetation_index"
        target = "vegetation"
        metric = "ndvi"

    elif (
        "ndwi" in query
        and not is_change_query
        and ("single" in query or "calculate" in query or len(years) <= 1)
    ):
        task = "water_index"
        target = "water"
        metric = "ndwi"

    elif (
        "ndbi" in query
        and not is_change_query
        and ("single" in query or "calculate" in query or len(years) <= 1)
    ):
        task = "urban_index"
        target = "urban"
        metric = "ndbi"

    # --------------------------------------------------------
    # Vegetation.
    # --------------------------------------------------------

    elif any(
        word in query
        for word in [
            "vegetation",
            "ndvi",
            "forest",
            "crop",
        ]
    ):
        task = "change_detection"
        target = "vegetation"
        metric = "ndvi"

    # --------------------------------------------------------
    # Water.
    # --------------------------------------------------------

    elif any(
        word in query
        for word in [
            "water",
            "lake",
            "river",
            "ndwi",
        ]
    ):
        task = "water_change"
        target = "water"
        metric = "ndwi"

    # --------------------------------------------------------
    # Urban.
    # --------------------------------------------------------

    elif any(
        word in query
        for word in [
            "urban",
            "city",
            "building",
            "ndbi",
        ]
    ):
        task = "urban_change"
        target = "urban"
        metric = "ndbi"

    # --------------------------------------------------------
    # Image comparison.
    # --------------------------------------------------------

    elif any(
        word in query
        for word in [
            "compare",
            "comparison",
        ]
    ):
        task = "image_comparison"
        target = None
        metric = None

    # --------------------------------------------------------
    # Image search.
    # --------------------------------------------------------

    else:
        task = "image_search"
        target = None
        metric = None

    # --------------------------------------------------------
    # Analysis list.
    # --------------------------------------------------------

    analysis = []

    if metric:
        analysis.append(metric)

    if task != "image_search":
        analysis.append(
            "change_detection"
        )

    return QueryPlan(
        task=task,
        target=target,
        time_start=time_start,
        time_end=time_end,
        modalities=["optical"],
        metric=metric,
        direction="unknown",
        analysis=analysis,
        output=[
            "map",
            "statistics",
            "explanation",
        ],
        aoi=final_aoi,
    )


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

    # ========================================================
    # 3. Execute tools
    # ========================================================

    execution_results = execute_plan(
        tools,
        context={
            "time_start": plan.time_start,
            "time_end": plan.time_end,
            "aoi": plan.aoi,
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

    if "calculate_ndvi" in execution_results:

        index_result = execution_results[
            "calculate_ndvi"
        ]

    elif "calculate_ndwi" in execution_results:

        index_result = execution_results[
            "calculate_ndwi"
        ]

    elif "calculate_ndbi" in execution_results:

        index_result = execution_results[
            "calculate_ndbi"
        ]

    # --------------------------------------------------------
    # Temporal index result
    # --------------------------------------------------------

    temporal_result = None

    if "calculate_temporal_ndvi" in execution_results:

        temporal_result = execution_results[
            "calculate_temporal_ndvi"
        ]

    elif "calculate_temporal_ndwi" in execution_results:

        temporal_result = execution_results[
            "calculate_temporal_ndwi"
        ]

    elif "calculate_temporal_ndbi" in execution_results:

        temporal_result = execution_results[
            "calculate_temporal_ndbi"
        ]

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
        # visualization metadata, create one here.
        elif change_map is not None:

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

        # Layer metadata
        layers.append(
            {
                "type": "change_detection",
                "name": (
                    f"{plan.target or 'Image'} "
                    f"change map"
                ),
                "change_ratio": (
                    change_result.get(
                        "change_ratio"
                    )
                ),
                "regions_detected": (
                    change_result.get(
                        "regions_detected"
                    )
                ),
                "changed_pixels": (
                    change_result.get(
                        "changed_pixels"
                    )
                ),
                "visualization_url": (
                    visualization_url
                ),
                "classified_visualization_url": (
                    classified_visualization_url
                ),
                "bounds": (
                    visualization_bounds
                ),
            }
        )

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

        layers.append(
            {
                "type": "index_map",
                "name": f"{statistics.get('metric') or 'Index'} map",
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
    # 11. P4 VLM
    # ========================================================

    p2_response = {
        "status": "success",
        "answer": explanation,
        "confidence": 0.9,
        "query": request.query,
        "plan": plan.model_dump(),
        "statistics": statistics,
        "layers": layers,
        "evidence": evidence,
        "execution_trace": execution_trace,
        "execution_tools": tools,
        "visualization_url": visualization_url,
        "classified_visualization_url": classified_visualization_url,
        "bounds": visualization_bounds,
    }

    evidence_package = build_evidence(p2_response)
    p2_response["evidence_package"] = evidence_package

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
        else explanation
    )

    return AnalysisResult(
        status="success",
        answer=final_answer,
        confidence=0.9,
        plan=plan.model_dump(),
        statistics=statistics,
        layers=layers,
        evidence=evidence,
        execution_trace=execution_trace,
        visualization_url=visualization_url,
        classified_visualization_url=classified_visualization_url,
        bounds=visualization_bounds,
        evidence_package=evidence_package,
    )
