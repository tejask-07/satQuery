from fastapi import APIRouter

from app.agent.executor import execute_plan
from app.agent.planner import create_execution_plan
from app.schemas.analysis import AnalysisResult
from app.schemas.query import QueryPlan, QueryRequest


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
    Convert a natural-language query into a QueryPlan.

    Lightweight rule-based planner for the MVP.
    """

    query = request.query.lower()

    # --------------------------------------------------------
    # Vegetation
    # --------------------------------------------------------

    if any(
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
    # Water
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
    # Urban
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
    # Image comparison
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
    # Image search
    # --------------------------------------------------------

    else:
        task = "image_search"
        target = None
        metric = None

    # --------------------------------------------------------
    # Analysis list
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
        time_start="2021",
        time_end="2025",
        modalities=["optical"],
        metric=metric,
        direction="unknown",
        analysis=analysis,
        output=[
            "map",
            "statistics",
            "explanation",
        ],
    )


# ============================================================
# EXPLANATION BUILDER
# ============================================================

def build_explanation(
    plan: QueryPlan,
    statistics: dict,
) -> str:
    """
    Create a human-readable explanation from
    the calculated statistics.
    """

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
    # No statistics
    # --------------------------------------------------------

    if metric is None:
        return (
            "The requested analysis was completed, "
            "but no index statistics were available."
        )

    # --------------------------------------------------------
    # No valid data
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
    # Format numbers
    # --------------------------------------------------------

    before_text = f"{mean_before:.4f}"
    after_text = f"{mean_after:.4f}"

    if mean_change is not None:
        change_text = f"{mean_change:+.4f}"
    else:
        change_text = "N/A"

    ratio_text = (
        f"{change_ratio * 100:.2f}%"
        if change_ratio is not None
        else "N/A"
    )

    # --------------------------------------------------------
    # Target description
    # --------------------------------------------------------

    target = (
        plan.target
        or "target"
    )

    # --------------------------------------------------------
    # Main explanation
    # --------------------------------------------------------

    explanation = (
        f"{metric} for {target} changed from "
        f"{before_text} in {plan.time_start} "
        f"to {after_text} in {plan.time_end}. "
        f"The mean change was {change_text}. "
    )

    # --------------------------------------------------------
    # Pixel statistics
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
    # Direction
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
    Process a natural-language satellite
    analysis query.
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
        tools
    )

    # ========================================================
    # 4. Build statistics
    # ========================================================

    statistics = {}

    # --------------------------------------------------------
    # Single-image index result
    # --------------------------------------------------------

    index_result = None

    if (
        "calculate_ndvi"
        in execution_results
    ):
        index_result = execution_results[
            "calculate_ndvi"
        ]

    elif (
        "calculate_ndwi"
        in execution_results
    ):
        index_result = execution_results[
            "calculate_ndwi"
        ]

    elif (
        "calculate_ndbi"
        in execution_results
    ):
        index_result = execution_results[
            "calculate_ndbi"
        ]

    # --------------------------------------------------------
    # Temporal index result
    # --------------------------------------------------------

    temporal_result = None

    if (
        "calculate_temporal_ndvi"
        in execution_results
    ):
        temporal_result = execution_results[
            "calculate_temporal_ndvi"
        ]

    elif (
        "calculate_temporal_ndwi"
        in execution_results
    ):
        temporal_result = execution_results[
            "calculate_temporal_ndwi"
        ]

    elif (
        "calculate_temporal_ndbi"
        in execution_results
    ):
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

        # ----------------------------------------------------
        # Before
        # ----------------------------------------------------

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
    # 7. Build explanation
    # ========================================================

    explanation = build_explanation(
        plan,
        statistics,
    )

    statistics["explanation"] = (
        explanation
    )

    # ========================================================
    # 8. Build evidence
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

    if change_result:

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
            }
        )

    if index_result:

        layers.append(
            {
                "type": "index",
                "name": (
                    f"{index_result.get('index')} "
                    "analysis"
                ),
                "metric": (
                    index_result.get(
                        "index"
                    )
                ),
                "mean": (
                    index_result.get(
                        "mean"
                    )
                ),
            }
        )

    # ========================================================
    # 10. Execution trace
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

    execution_trace.append(
        "Analysis completed"
    )

    # ========================================================
    # 11. Return response
    # ========================================================

    return AnalysisResult(
        status="success",

        answer=explanation,

        confidence=0.9,

        plan=plan.model_dump(),

        statistics=statistics,

        layers=layers,

        evidence=evidence,

        execution_trace=execution_trace,
    )