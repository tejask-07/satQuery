import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agent.parser import parse_query
from app.schemas.query import QueryRequest


def test_descriptive_visual_feature_questions_route_to_single_image_vqa():
    vqa_queries = [
        "What man-made features can you see?",
        "What natural features are visible?",
        "What objects are visible in the image?",
        "What buildings are visible?",
        "What infrastructure is visible?",
        "What can you see in this satellite image?",
    ]

    for query in vqa_queries:
        plan = parse_query(QueryRequest(query=query))
        assert plan.task == "single_image_vqa", query
        assert plan.intent == "single_image_vqa", query


def test_image_retrieval_questions_remain_image_search():
    search_queries = [
        "Find satellite imagery of Mumbai.",
        "Search satellite imagery for this location.",
    ]

    for query in search_queries:
        plan = parse_query(QueryRequest(query=query))
        assert plan.task == "image_search", query
        assert plan.intent == "image_search", query


queries = [
    # 5 required Phase 1 test cases
    "Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]",
    "Analyze vegetation loss between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]",
    "Analyze water change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]",
    "What changed between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]?",
    "Did vegetation become urban between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]?",


    # Legacy queries
    "Show me where vegetation decreased between 2021 and 2025.",
    "Find areas where water increased after the flood.",
    "Which areas experienced urban expansion?",
    "Compare these two satellite images.",
]


if __name__ == "__main__":
    for text in queries:
        request = QueryRequest(query=text)
        plan = parse_query(request)

        print("\n" + "=" * 60)
        print(f"QUERY: {text}")
        print("=" * 60)
        print("PLAN:")
        print(plan.model_dump_json(indent=2))
