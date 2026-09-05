"""
Agent integration and regression tests for Single-Image VQA (Step 16).

Verifies:
1. Parser detects single_image_vqa
2. Parser detects SAR VQA
3. Parser does NOT misclassify Optical-SAR as VQA
4. Parser does NOT misclassify temporal change as VQA
5. Parser does NOT misclassify NDVI/NDWI/NDBI as VQA
6. Planner returns single_image_vqa
7. Registry contains single_image_vqa
8. Executor calls app.vlm.vqa.run_vqa
9. Optical modality forwarding works
10. SAR modality forwarding works
11. Evidence forwarding works
12. Existing Optical-SAR route remains unchanged
13. Existing scientific routes remain unchanged
"""

from pathlib import Path
import pytest
from PIL import Image

from app.schemas.query import QueryRequest, QueryPlan
from app.agent.parser import parse_query, detect_single_image_vqa_intent
from app.agent.planner import create_execution_plan
from app.agent.registry import get_tool, TOOL_REGISTRY
from app.agent.executor import execute_plan
from app.vlm import vqa
from app.api.routes_query import _build_single_image_vqa_api_response


# ============================================================
# 1. PARSER DETECTS SINGLE-IMAGE VQA
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        "What is visible in this image?",
        "Is there a water body in this image?",
        "Is there water here?",
        "Are there buildings?",
        "What type of land cover dominates?",
        "What type of land cover dominates the image?",
        "Describe the objects in this image",
        "Can you identify the main features?",
    ],
)
def test_parser_detects_single_image_vqa(query: str):
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task == "single_image_vqa"
    assert plan.intent == "single_image_vqa"
    assert "single_image_vqa" in plan.analysis


# ============================================================
# 2. PARSER DETECTS SAR VQA
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        "What structures are visible in this SAR image?",
        "Are there bridges visible in the radar image?",
        "What backscatter features are present in this SAR image?",
    ],
)
def test_parser_detects_sar_vqa(query: str):
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task == "single_image_vqa"
    assert plan.intent == "single_image_vqa"
    assert "sar" in plan.modalities


# ============================================================
# 3. PARSER DOES NOT MISCLASSIFY OPTICAL-SAR AS VQA
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        "Compare optical and SAR for built-up areas",
        "Does the optical and SAR imagery indicate built-up areas?",
        "Use optical and SAR imagery to identify water-covered regions.",
        "Use the optical and SAR images together to identify built-up areas.",
        "Combine optical and SAR to analyze the region",
        "What does SAR and optical show together?",
    ],
)
def test_parser_does_not_misclassify_optical_sar_as_vqa(query: str):
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task == "optical_sar_analysis"
    assert plan.task != "single_image_vqa"


# ============================================================
# 4. PARSER DOES NOT MISCLASSIFY TEMPORAL CHANGE AS VQA
# ============================================================

@pytest.mark.parametrize(
    "query,expected_task",
    [
        (
            "What changed between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]?",
            "general_change_detection",
        ),
        (
            "What changed between the two images?",
            "general_change_detection",
        ),
        (
            "Show me where vegetation decreased between 2021 and 2025.",
            "change_detection",
        ),
        (
            "Did vegetation become urban between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]?",
            "land_cover_transition",
        ),
        (
            "Compare these two satellite images.",
            "image_comparison",
        ),
    ],
)
def test_parser_does_not_misclassify_temporal_change_as_vqa(query: str, expected_task: str):
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task == expected_task
    assert plan.task != "single_image_vqa"


# ============================================================
# 5. PARSER DOES NOT MISCLASSIFY NDVI / NDWI / NDBI AS VQA
# ============================================================

@pytest.mark.parametrize(
    "query,expected_task,expected_metric",
    [
        ("Calculate NDVI", "vegetation_index", "ndvi"),
        ("Calculate NDWI", "water_index", "ndwi"),
        ("Calculate NDBI", "urban_index", "ndbi"),
        ("Calculate NDVI using the optical image.", "vegetation_index", "ndvi"),
        ("Calculate NDVI using the optical and SAR images.", "vegetation_index", "ndvi"),
    ],
)
def test_parser_does_not_misclassify_indices_as_vqa(query: str, expected_task: str, expected_metric: str):
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task == expected_task
    assert plan.task != "single_image_vqa"
    assert plan.metric == expected_metric


# ============================================================
# 6. PLANNER RETURNS SINGLE_IMAGE_VQA
# ============================================================

def test_planner_returns_single_image_vqa():
    plan = QueryPlan(task="single_image_vqa")
    tools = create_execution_plan(plan)

    assert tools == ["single_image_vqa"]


# ============================================================
# 7. REGISTRY CONTAINS SINGLE_IMAGE_VQA
# ============================================================

def test_registry_contains_single_image_vqa():
    assert "single_image_vqa" in TOOL_REGISTRY
    tool = get_tool("single_image_vqa")
    assert callable(tool)
    assert tool is vqa.run_vqa


# ============================================================
# 8. EXECUTOR CALLS APP.VLM.VQA.RUN_VQA
# ============================================================

def test_executor_calls_run_vqa(monkeypatch):
    calls = []

    def mock_run_vqa(image, question, modality="unknown", evidence=None):
        calls.append({
            "image": image,
            "question": question,
            "modality": modality,
            "evidence": evidence,
        })
        return {
            "task": "single_image_vqa",
            "question": question,
            "answer": "Mock VQA answer.",
            "modality": modality,
            "confidence": None,
        }

    monkeypatch.setattr("app.vlm.vqa.run_vqa", mock_run_vqa)

    mock_img = Image.new("RGB", (32, 32), color="green")
    results = execute_plan(
        ["single_image_vqa"],
        context={
            "image": mock_img,
            "query": "Is there a water body in this image?",
        },
    )

    assert "single_image_vqa" in results
    res = results["single_image_vqa"]
    assert res["task"] == "single_image_vqa"
    assert res["answer"] == "Mock VQA answer."
    assert len(calls) == 1
    assert calls[0]["image"] is mock_img
    assert calls[0]["question"] == "Is there a water body in this image?"


# ============================================================
# 9. OPTICAL MODALITY FORWARDING
# ============================================================

def test_executor_optical_modality_forwarding(monkeypatch):
    calls = []

    def mock_run_vqa(image, question, modality="unknown", evidence=None):
        calls.append(modality)
        return {
            "task": "single_image_vqa",
            "question": question,
            "answer": "Optical visible answer.",
            "modality": modality,
            "confidence": None,
        }

    monkeypatch.setattr("app.vlm.vqa.run_vqa", mock_run_vqa)

    mock_img = Image.new("RGB", (32, 32), color="blue")
    results = execute_plan(
        ["single_image_vqa"],
        context={
            "image": mock_img,
            "query": "What is visible in this optical image?",
            "modality": "optical",
        },
    )

    assert calls == ["optical"]
    assert results["single_image_vqa"]["modality"] == "optical"


# ============================================================
# 10. SAR MODALITY FORWARDING
# ============================================================

def test_executor_sar_modality_forwarding(monkeypatch):
    calls = []

    def mock_run_vqa(image, question, modality="unknown", evidence=None):
        calls.append(modality)
        return {
            "task": "single_image_vqa",
            "question": question,
            "answer": "SAR structural backscatter answer.",
            "modality": modality,
            "confidence": None,
        }

    monkeypatch.setattr("app.vlm.vqa.run_vqa", mock_run_vqa)

    mock_img = Image.new("L", (32, 32), color=128)
    results = execute_plan(
        ["single_image_vqa"],
        context={
            "image": mock_img,
            "query": "What structures are visible in this SAR image?",
            "modality": "sar",
        },
    )

    assert calls == ["sar"]
    assert results["single_image_vqa"]["modality"] == "sar"


# ============================================================
# 11. EVIDENCE FORWARDING
# ============================================================

def test_executor_evidence_forwarding(monkeypatch):
    calls = []

    def mock_run_vqa(image, question, modality="unknown", evidence=None):
        calls.append(evidence)
        return {
            "task": "single_image_vqa",
            "question": question,
            "answer": "Evidence grounded answer.",
            "modality": modality,
            "confidence": None,
        }

    monkeypatch.setattr("app.vlm.vqa.run_vqa", mock_run_vqa)

    mock_img = Image.new("RGB", (32, 32))
    expected_evidence = {"mean_reflectance": 0.42, "cloud_cover": 0.01}

    results = execute_plan(
        ["single_image_vqa"],
        context={
            "image": mock_img,
            "query": "What type of land cover dominates?",
            "evidence": expected_evidence,
        },
    )

    assert calls == [expected_evidence]
    assert results["single_image_vqa"]["answer"] == "Evidence grounded answer."


# ============================================================
# 12. EXISTING OPTICAL-SAR ROUTE REMAINS UNCHANGED
# ============================================================

def test_existing_optical_sar_route_remains_unchanged():
    plan = QueryPlan(task="optical_sar_analysis")
    tools = create_execution_plan(plan)
    assert tools == ["optical_sar_analysis"]


# ============================================================
# 13. EXISTING SCIENTIFIC ROUTES REMAIN UNCHANGED
# ============================================================

def test_existing_scientific_routes_remain_unchanged():
    assert create_execution_plan(QueryPlan(task="vegetation_index")) == [
        "search_imagery",
        "calculate_ndvi",
    ]
    assert create_execution_plan(QueryPlan(task="water_index")) == [
        "search_imagery",
        "calculate_ndwi",
    ]
    assert create_execution_plan(QueryPlan(task="urban_index")) == [
        "search_imagery",
        "calculate_ndbi",
    ]


# ============================================================
# 14. EXECUTOR ERROR HANDLING: NO IMAGE PROVIDED
# ============================================================

def test_executor_clean_error_when_no_image():
    with pytest.raises(ValueError, match="single_image_vqa requires exactly one image"):
        execute_plan(
            ["single_image_vqa"],
            context={"query": "What is visible in this image?"},
        )


# ============================================================
# 15. API RESPONSE CONTRACT AND ZERO PATH LEAKAGE
# ============================================================

def test_vqa_api_response_contract_no_path_leakage():
    plan = QueryPlan(task="single_image_vqa")
    vqa_res = {
        "task": "single_image_vqa",
        "question": "What is visible in this image?",
        "answer": "A river flowing through agricultural fields.",
        "modality": "optical",
        "confidence": None,
    }

    api_resp = _build_single_image_vqa_api_response(
        plan=plan,
        query="What is visible in this image?",
        vqa_res=vqa_res,
    )

    assert api_resp.status == "success"
    assert api_resp.answer == "A river flowing through agricultural fields."
    assert api_resp.confidence is None
    assert api_resp.statistics["task"] == "single_image_vqa"
    assert api_resp.statistics["question"] == "What is visible in this image?"
    assert api_resp.statistics["answer"] == "A river flowing through agricultural fields."
    assert api_resp.statistics["modality"] == "optical"
    assert api_resp.statistics["confidence"] is None

    # Phase 2 Evidence Contract
    assert len(api_resp.evidence) == 1
    assert api_resp.evidence[0]["source"] == "single_image_observation"
    assert api_resp.evidence[0]["modality"] == "optical"
    assert api_resp.evidence[0]["evidence_used"] is False

    # Phase 3 Execution Trace Milestones
    assert len(api_resp.execution_trace) == 6
    assert api_resp.execution_trace[0] == "Natural-language query received for visual inspection"
    assert api_resp.execution_trace[1] == "Natural-language query classified as single-image VQA"
    assert api_resp.execution_trace[2] == "Execution plan created for single-image VQA"
    assert api_resp.execution_trace[3] == "Single image resolved (modality: optical)"
    assert api_resp.execution_trace[4] == "VQA specialist executed with grounded prompt"
    assert api_resp.execution_trace[5] == "VQA response formatted"

    # Verify zero server paths leak in serialized dict
    resp_dump = api_resp.model_dump_json()
    assert "C:\\" not in resp_dump
    assert "/home/" not in resp_dump
    assert "data/uploads" not in resp_dump
    assert "/mnt/data" not in resp_dump
    assert "backend/data/cache" not in resp_dump


# ============================================================
# 16. API RESPONSE WITH SUPPLIED STRUCTURED EVIDENCE
# ============================================================

def test_vqa_api_response_with_supplied_evidence():
    plan = QueryPlan(task="single_image_vqa")
    vqa_res = {
        "task": "single_image_vqa",
        "question": "What is visible?",
        "answer": "Industrial buildings with high reflectance.",
        "modality": "optical",
        "confidence": None,
        "evidence": {
            "source": "ground_truth",
            "resolution_m": 10.0,
            "cloud_cover": 0.02,
            "file_path": "C:\\Users\\admin\\cache\\opt.tif",
        },
    }

    api_resp = _build_single_image_vqa_api_response(
        plan=plan,
        query="What is visible?",
        vqa_res=vqa_res,
    )

    assert api_resp.status == "success"
    assert api_resp.confidence is None
    assert len(api_resp.evidence) == 1
    ev = api_resp.evidence[0]
    assert ev["source"] == "ground_truth"
    assert ev["modality"] == "optical"
    assert ev["evidence_used"] is True
    assert ev["resolution_m"] == 10.0
    assert ev["cloud_cover"] == 0.02

    # Sanitization check: raw file path must not leak
    resp_dump = api_resp.model_dump_json()
    assert "C:\\Users\\admin" not in resp_dump
    assert "opt.tif" in resp_dump or "file_path" not in ev or ev.get("file_path") == "opt.tif"


# ============================================================
# 17. API RESPONSE WITH SAR MODALITY AND NO LEAKS
# ============================================================

def test_vqa_api_response_sar_modality():
    plan = QueryPlan(task="single_image_vqa", modalities=["sar"])
    vqa_res = {
        "task": "single_image_vqa",
        "question": "What backscatter features are visible?",
        "answer": "Strong double-bounce reflections indicating urban structures.",
        "modality": "sar",
        "confidence": None,
    }

    api_resp = _build_single_image_vqa_api_response(
        plan=plan,
        query="What backscatter features are visible?",
        vqa_res=vqa_res,
        context={"sar_image_id": "S1A_IW_GRDH_1SDV_20250101"},
    )

    assert api_resp.status == "success"
    assert api_resp.confidence is None
    assert api_resp.statistics["modality"] == "sar"
    assert api_resp.evidence[0]["source"] == "single_image_observation"
    assert api_resp.evidence[0]["modality"] == "sar"
    assert api_resp.evidence[0]["evidence_used"] is False
    assert api_resp.execution_trace[3] == "Single image resolved (modality: sar)"

