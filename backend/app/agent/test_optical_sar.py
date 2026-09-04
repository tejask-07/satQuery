"""
Unit and Integration Tests for Optical-SAR Query Detection and Agent Routing (Step 6).
"""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from PIL import Image

from app.schemas.query import QueryRequest, QueryPlan
from app.agent.parser import parse_query, detect_optical_sar_intent
from app.agent.planner import create_execution_plan
from app.agent.executor import execute_plan
from app.agent.registry import get_tool


class MockSpecialistVLM:
    """Mock VLM for deterministic agent routing test without HF API calls."""

    def __init__(self, answer: str = "Mock agent multimodal response."):
        self.answer = answer
        self.called = False

    def generate(self, *args, **kwargs) -> str:
        self.called = True
        return self.answer


def _create_test_geotiff(
    path: Path,
    bounds: tuple[float, float, float, float] = (73.80, 18.50, 73.86, 18.56),
    crs: str = "EPSG:4326",
    width: int = 50,
    height: int = 50,
    count: int = 3,
    dtype: str = "float32",
    descriptions: list[str] | None = None,
) -> Path:
    """Helper to generate a lightweight test GeoTIFF."""
    transform = from_bounds(*bounds, width, height)
    data = np.ones((count, height, width), dtype=dtype) * 100.0

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
        if descriptions:
            for i, desc in enumerate(descriptions, 1):
                dst.set_band_description(i, desc)

    return path


# ============================================================
# TESTS 1 - 5: QUERY DETECTION & PARSING INTENT
# ============================================================

def test_1_query_optical_sar_built_up():
    """Test 1: 'Use the optical and SAR images together to identify built-up areas.' -> optical_sar_analysis."""
    query = "Use the optical and SAR images together to identify built-up areas."
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task == "optical_sar_analysis"
    assert plan.intent == "optical_sar_analysis"
    assert plan.target == "urban"
    assert "optical" in plan.modalities
    assert "sar" in plan.modalities


def test_2_query_optical_sar_water():
    """Test 2: 'Use optical and SAR imagery to identify water-covered regions.' -> optical_sar_analysis."""
    query = "Use optical and SAR imagery to identify water-covered regions."
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task == "optical_sar_analysis"
    assert plan.intent == "optical_sar_analysis"
    assert plan.target == "water"
    assert "optical" in plan.modalities
    assert "sar" in plan.modalities


def test_3_query_ndvi_using_optical_image():
    """Test 3: 'Calculate NDVI using the optical image.' -> vegetation_index (NOT optical_sar_analysis)."""
    query = "Calculate NDVI using the optical image."
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task == "vegetation_index"
    assert plan.task != "optical_sar_analysis"
    assert plan.metric == "ndvi"


def test_4_query_ndwi():
    """Test 4: 'Calculate NDWI.' -> water_index (NOT optical_sar_analysis)."""
    query = "Calculate NDWI."
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task == "water_index"
    assert plan.task != "optical_sar_analysis"
    assert plan.metric == "ndwi"


def test_5_query_what_changed_between_dates():
    """Test 5: 'What changed between these two dates?' -> general_change_detection (NOT optical_sar_analysis)."""
    query = "What changed between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]?"
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task in ("general_change_detection", "change_detection")
    assert plan.task != "optical_sar_analysis"


def test_priority_ndvi_even_with_both_words():
    """Test Step 6D priority: 'Calculate NDVI using the optical and SAR images.' remains vegetation_index."""
    query = "Calculate NDVI using the optical and SAR images."
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task == "vegetation_index"
    assert plan.task != "optical_sar_analysis"


def test_sar_only_query_not_optical_sar():
    """Single modality query 'Analyze the SAR image.' does not trigger optical_sar_analysis."""
    query = "Analyze the SAR image."
    req = QueryRequest(query=query)
    plan = parse_query(req)

    assert plan.task != "optical_sar_analysis"


# ============================================================
# TESTS 6 - 8: PLANNER & EXECUTOR DISPATCH
# ============================================================

def test_6_optical_sar_task_reaches_specialist(tmp_path):
    """Test 6: optical_sar_analysis task with valid optical and SAR paths reaches answer_optical_sar_question."""
    opt_path = _create_test_geotiff(
        tmp_path / "opt.tif",
        count=3,
        descriptions=["Red", "Green", "Blue"],
    )
    sar_path = _create_test_geotiff(
        tmp_path / "sar.tif",
        count=2,
        descriptions=["VV", "VH"],
    )

    query = "Use the optical and SAR images together to identify built-up areas."
    req = QueryRequest(query=query)
    plan = parse_query(req)
    tools = create_execution_plan(plan)

    assert tools == ["optical_sar_analysis"]

    mock_vlm = MockSpecialistVLM(answer="Co-registered optical and radar observations reveal urban structures.")

    results = execute_plan(
        tools,
        context={
            "optical_path": str(opt_path),
            "sar_path": str(sar_path),
            "query": query,
            "vlm": mock_vlm,
        },
    )

    assert "optical_sar_analysis" in results
    res = results["optical_sar_analysis"]
    assert res["success"] is True
    assert res["answer"] == "Co-registered optical and radar observations reveal urban structures."
    assert "optical" in res["modalities"]
    assert "sar_vv" in res["modalities"]
    assert mock_vlm.called is True


def test_7_optical_sar_only_optical_input_fails_cleanly(tmp_path):
    """Test 7: Optical-SAR task with only optical input must fail cleanly with a clear validation error."""
    opt_path = _create_test_geotiff(tmp_path / "opt_only.tif", count=3)
    tools = ["optical_sar_analysis"]

    with pytest.raises(ValueError, match="only optical was provided"):
        execute_plan(
            tools,
            context={
                "optical_path": str(opt_path),
                "sar_path": None,
                "query": "Use optical and SAR together",
            },
        )


def test_8_optical_sar_only_sar_input_fails_cleanly(tmp_path):
    """Test 8: Optical-SAR task with only SAR input must fail cleanly with a clear validation error."""
    sar_path = _create_test_geotiff(tmp_path / "sar_only.tif", count=2)
    tools = ["optical_sar_analysis"]

    with pytest.raises(ValueError, match="only SAR was provided"):
        execute_plan(
            tools,
            context={
                "optical_path": None,
                "sar_path": str(sar_path),
                "query": "Use optical and SAR together",
            },
        )


def test_optical_sar_neither_path_fails_cleanly():
    """Verify clean rejection when neither path is provided to optical_sar_analysis."""
    tools = ["optical_sar_analysis"]
    with pytest.raises(ValueError, match="requires both optical_path and sar_path"):
        execute_plan(
            tools,
            context={
                "query": "Use optical and SAR together",
            },
        )


# ============================================================
# TESTS 9 - 10: REGRESSION INTEGRITY
# ============================================================

def test_9_existing_planner_execution_plans():
    """Test 9: Verify existing planner output for standard change and index tasks."""
    plan_urban = QueryPlan(task="urban_change", target="urban")
    assert create_execution_plan(plan_urban) == [
        "search_imagery",
        "calculate_temporal_ndbi",
        "calculate_temporal_ndvi",
        "calculate_temporal_ndwi",
        "detect_change",
    ]

    plan_veg = QueryPlan(task="vegetation_index", target="vegetation")
    assert create_execution_plan(plan_veg) == [
        "search_imagery",
        "calculate_ndvi",
    ]

    plan_water = QueryPlan(task="water_index", target="water")
    assert create_execution_plan(plan_water) == [
        "search_imagery",
        "calculate_ndwi",
    ]

    plan_comp = QueryPlan(task="image_comparison")
    assert create_execution_plan(plan_comp) == [
        "search_imagery",
        "compare_images",
        "detect_change",
    ]


def test_10_tool_registry_contains_all_tools():
    """Test 10: Verify TOOL_REGISTRY has all existing tools plus optical_sar_analysis."""
    expected_tools = [
        "search_imagery",
        "calculate_ndvi",
        "calculate_ndwi",
        "calculate_ndbi",
        "calculate_temporal_ndvi",
        "calculate_temporal_ndwi",
        "calculate_temporal_ndbi",
        "detect_change",
        "compare_images",
        "optical_sar_analysis",
    ]

    for tool_name in expected_tools:
        tool_fn = get_tool(tool_name)
        assert callable(tool_fn), f"Tool {tool_name} is not callable"
