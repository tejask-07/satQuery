from app.api import routes_query
from app.schemas.query import QueryRequest


VQA_QUERIES = [
    "What man-made features can you see?",
    "What natural features are visible?",
    "What objects are visible in the image?",
]

IMAGE_SEARCH_QUERIES = [
    "Find satellite imagery of Mumbai.",
    "Search satellite imagery for this location.",
]


def test_process_query_routes_descriptive_visual_questions_to_vqa(monkeypatch):
    executed_tools = []

    def fake_execute_plan(tools, context):
        executed_tools.append(list(tools))
        return {
            "single_image_vqa": {
                "answer": "Visible features include supported scene content.",
                "confidence": None,
                "model": "test",
                "status": "test",
            }
        }

    monkeypatch.setattr(routes_query, "execute_plan", fake_execute_plan)

    for query in VQA_QUERIES:
        response = routes_query.process_query(QueryRequest(query=query))
        assert response.plan["task"] == "single_image_vqa", query
        assert response.execution_summary["task"] == "single_image_vqa", query
        assert response.execution_trace == [
            "Task identified: single_image_vqa",
            "Executed: single_image_vqa",
        ]

    assert executed_tools == [["single_image_vqa"]] * len(VQA_QUERIES)


def test_process_query_keeps_retrieval_questions_on_image_search_path(monkeypatch):
    executed_tools = []

    def fake_execute_plan(tools, context):
        executed_tools.append(list(tools))
        return {}

    monkeypatch.setattr(routes_query, "execute_plan", fake_execute_plan)

    for query in IMAGE_SEARCH_QUERIES:
        response = routes_query.process_query(QueryRequest(query=query))
        assert response.plan["task"] == "image_search", query
        assert executed_tools[-1] == ["search_imagery"]
