from fastapi import APIRouter

from app.schemas.analysis import AnalysisResult
from app.schemas.query import QueryRequest, QueryPlan

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=AnalysisResult)
def process_query(request: QueryRequest):
    # Temporary mock planner.
    # Person 5 will replace this with the actual query engine.

    plan = QueryPlan(
        task="change_detection",
        target="vegetation",
        time_start="2021",
        time_end="2025",
        modalities=["optical"],
        analysis=["ndvi", "change_detection"],
        output=["map", "statistics", "explanation"],
    )

    return AnalysisResult(
        status="planned",
        answer="Query successfully converted into an analysis plan.",
        plan=plan.model_dump(),
        execution_trace=[
            "Query received",
            "Intent identified",
            "Analysis plan generated",
        ],
    )