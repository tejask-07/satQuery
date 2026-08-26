from app.agent.parser import parse_query
from app.schemas.query import QueryRequest


queries = [
    "Show me where vegetation decreased between 2021 and 2025.",
    "Find areas where water increased after the flood.",
    "Which areas experienced urban expansion?",
    "Compare these two satellite images.",
]


for text in queries:
    request = QueryRequest(query=text)

    plan = parse_query(request)

    print("\nQUERY:")
    print(text)

    print("\nPLAN:")
    print(plan.model_dump_json(indent=2))