from fastapi import APIRouter

from app.agent.parser import parse_query
from app.agent.planner import create_execution_plan
from app.agent.executor import execute_plan
from app.schemas.analysis import AnalysisResult
from app.schemas.query import QueryRequest

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=AnalysisResult)
def process_query(request: QueryRequest):

    # 1. Understand the user's natural-language query.
    query_plan = parse_query(request)

    # 2. Convert the interpreted query into an ordered tool plan.
    execution_plan = create_execution_plan(query_plan)

    # 3. Execute the planned tools.
    execution_results = execute_plan(execution_plan)

    # 4. Return the plan and execution results to the frontend.
    return AnalysisResult(
        status="executed",
        answer="Query successfully analyzed.",
        plan=query_plan.model_dump(),
        statistics=execution_results,
        execution_trace=[
            "Query received",
            "Intent identified",
            "Analysis plan generated",
            *[
                f"Executed: {tool}"
                for tool in execution_plan
            ],
        ],
    )