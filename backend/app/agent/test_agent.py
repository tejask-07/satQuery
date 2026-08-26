from app.schemas.query import QueryRequest
from app.agent.parser import parse_query
from app.agent.planner import create_execution_plan
from app.agent.executor import execute_plan


def run_test(query: str):

    print("\n" + "=" * 60)
    print("USER QUERY")
    print("=" * 60)

    print(query)

    # 1. Parse
    request = QueryRequest(query=query)
    query_plan = parse_query(request)

    print("\nQUERY PLAN")
    print(query_plan.model_dump_json(indent=2))

    # 2. Plan
    execution_plan = create_execution_plan(query_plan)

    print("\nEXECUTION PLAN")

    for i, tool in enumerate(execution_plan, 1):
        print(f"{i}. {tool}")

    # 3. Execute
    print("\nEXECUTION")

    results = execute_plan(
        execution_plan,
        context=query_plan.model_dump()
    )

    print("\nRESULTS")

    for tool, result in results.items():
        print(f"\n{tool}:")
        print(result)


if __name__ == "__main__":

    run_test(
        "Show me where vegetation decreased between 2021 and 2025."
    )

    run_test(
        "Which areas experienced urban expansion?"
    )

    run_test(
        "Compare these two satellite images."
    )