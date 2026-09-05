"""
Regression test suite for Temporal Index Change Pipeline.

Covers:
1. Explicit NDVI temporal change query planning and execution (omitting NDWI/NDBI).
2. Valid temporal NDVI data handoff to detect_change.
3. Zero-valid-pixel temporal data handling (graceful evidence-grounded result, no 500 error).
4. Explicit NDWI temporal change query planning and execution.
5. Explicit NDBI temporal change query planning and execution.
6. Generic change query planning and multi-index execution.
"""

from unittest.mock import patch
import numpy as np
import pytest

from app.agent.parser import parse_query
from app.agent.planner import create_execution_plan
from app.agent.executor import execute_plan, _save_change_map_visualization
from app.api.routes_query import process_query
from app.schemas.query import QueryRequest, QueryPlan
from app.tools.change import detect_change
from app.tools.indices import calculate_temporal_ndvi


@pytest.fixture
def mock_vlm_offline():
    """Mock VLM so tests don't require external HuggingFace API calls."""
    with patch("app.api.routes_query.VLM.generate", return_value=None):
        yield


def test_explicit_ndvi_temporal_change_plan():
    """
    Test 1: Explicit NDVI temporal change query.
    Planner should prefer search_imagery -> calculate_temporal_ndvi -> detect_change
    and should not include NDWI or NDBI.
    """
    # 1A. User's exact query with vegetation/NDVI
    query1 = "Compare vegetation/NDVI change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]"
    plan1 = parse_query(QueryRequest(query=query1))
    assert plan1.metric == "ndvi"
    assert plan1.explicit_metric == "ndvi"
    tools1 = create_execution_plan(plan1)
    assert tools1 == [
        "search_imagery",
        "calculate_temporal_ndvi",
        "detect_change",
    ]
    assert "calculate_temporal_ndwi" not in tools1
    assert "calculate_temporal_ndbi" not in tools1

    # 1B. Direct NDVI change query
    query2 = "Compare NDVI change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]"
    plan2 = parse_query(QueryRequest(query=query2))
    assert plan2.metric == "ndvi"
    assert plan2.explicit_metric == "ndvi"
    tools2 = create_execution_plan(plan2)
    assert tools2 == [
        "search_imagery",
        "calculate_temporal_ndvi",
        "detect_change",
    ]


def test_valid_temporal_ndvi_data_handoff():
    """
    Test 2: Valid temporal NDVI data correctly consumed by detect_change.
    """
    np.random.seed(42)
    # Synthetic 50x50 valid reflectance arrays
    red_before = np.full((50, 50), 0.10, dtype=np.float32)
    nir_before = np.full((50, 50), 0.50, dtype=np.float32)
    red_after = np.full((50, 50), 0.20, dtype=np.float32)
    nir_after = np.full((50, 50), 0.40, dtype=np.float32)

    ndvi_res = calculate_temporal_ndvi(
        red_before=red_before,
        nir_before=nir_before,
        red_after=red_after,
        nir_after=nir_after,
    )
    assert ndvi_res["status"] == "success"
    assert ndvi_res["valid_pixels"] == 2500
    assert ndvi_res["mean_ndvi_before"] is not None
    assert ndvi_res["mean_ndvi_after"] is not None

    # Handoff to detect_change
    chg_res = detect_change(
        before=ndvi_res["ndvi_before"],
        after=ndvi_res["ndvi_after"],
        threshold=0.05,
        valid_mask=ndvi_res["valid_mask"],
    )
    assert chg_res["status"] == "success"
    assert chg_res["valid_pixels"] == 2500
    assert chg_res["mean_change"] is not None
    assert chg_res["change_type"] in ["decrease", "increase", "no_change", "mixed"]

    # Verify visualization helper with valid pixels
    vis = _save_change_map_visualization(
        change_map=chg_res["change_map"],
        prefix="test_valid_ndvi",
        threshold=0.05,
    )
    assert vis is not None
    assert vis.get("status") == "success"
    assert vis.get("filename") is not None


def test_zero_valid_pixel_temporal_data(mock_vlm_offline):
    """
    Test 3: Zero-valid-pixel temporal data returns graceful evidence-grounded result.
    Must NOT throw RuntimeError: detect_change requires a temporal index calculation...
    Must NOT raise HTTP 500.
    Must return truthful explanation stating not enough valid pixels.
    """
    nan_arr = np.full((60, 60), np.nan, dtype=np.float32)
    false_mask = np.zeros((60, 60), dtype=bool)

    # 3A. Tool-level behavior with zero valid pixels
    ndvi_res = {
        "status": "success",
        "index": "NDVI",
        "ndvi_before": nan_arr,
        "ndvi_after": nan_arr,
        "valid_mask": false_mask,
        "valid_pixels": 0,
        "total_pixels": 3600,
        "mean_ndvi_before": None,
        "mean_ndvi_after": None,
        "mean_ndvi_change": None,
    }

    chg_res = detect_change(
        before=ndvi_res["ndvi_before"],
        after=ndvi_res["ndvi_after"],
        threshold=0.05,
        valid_mask=ndvi_res["valid_mask"],
    )
    assert chg_res["status"] == "success"
    assert chg_res["valid_pixels"] == 0
    assert chg_res["mean_before"] is None
    assert chg_res["mean_after"] is None
    assert chg_res["mean_change"] is None
    assert chg_res["change_type"] == "no_data"

    # Visualization helper must return None gracefully
    vis = _save_change_map_visualization(
        change_map=chg_res["change_map"],
        prefix="test_zero_valid",
        threshold=0.05,
    )
    assert vis is None

    # 3B. Pipeline execution when imagery produces zero valid pixels
    with patch("app.agent.executor._read_raster", return_value=nan_arr), \
         patch("app.agent.executor.get_tool") as mock_gt:

        def get_tool_side_effect(name):
            if name == "search_imagery":
                return lambda **kwargs: {
                    "images": [
                        {"id": "img1", "date": "2021-01-01", "bands": {"red": "r1.tif", "nir": "n1.tif"}},
                        {"id": "img2", "date": "2024-01-01", "bands": {"red": "r2.tif", "nir": "n2.tif"}},
                    ]
                }
            if name == "calculate_temporal_ndvi":
                return lambda **kwargs: ndvi_res
            if name == "detect_change":
                return detect_change
            from app.agent.registry import get_tool as real_get_tool
            return real_get_tool(name)

        mock_gt.side_effect = get_tool_side_effect

        req = QueryRequest(
            query="Compare vegetation/NDVI change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]"
        )
        res = process_query(req)

        assert res.status == "success"
        assert res.statistics.get("valid_pixels") == 0
        assert res.statistics.get("mean_change") is None
        assert res.visualization_url is None
        assert "not enough valid pixels" in res.answer.lower()


def test_explicit_ndwi_temporal_change():
    """
    Test 4: Explicit NDWI temporal change query.
    Planner should prefer search_imagery -> calculate_temporal_ndwi -> detect_change.
    """
    query = "Compare NDWI change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]"
    plan = parse_query(QueryRequest(query=query))
    assert plan.metric == "ndwi"
    assert plan.explicit_metric == "ndwi"
    tools = create_execution_plan(plan)
    assert tools == [
        "search_imagery",
        "calculate_temporal_ndwi",
        "detect_change",
    ]
    assert "calculate_temporal_ndvi" not in tools
    assert "calculate_temporal_ndbi" not in tools


def test_explicit_ndbi_temporal_change():
    """
    Test 5: Explicit NDBI temporal change query.
    Planner should prefer search_imagery -> calculate_temporal_ndbi -> detect_change.
    """
    query = "Compare NDBI change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]"
    plan = parse_query(QueryRequest(query=query))
    assert plan.metric == "ndbi"
    assert plan.explicit_metric == "ndbi"
    tools = create_execution_plan(plan)
    assert tools == [
        "search_imagery",
        "calculate_temporal_ndbi",
        "detect_change",
    ]
    assert "calculate_temporal_ndvi" not in tools
    assert "calculate_temporal_ndwi" not in tools


def test_generic_change_query():
    """
    Test 6: Generic change query ("What changed?").
    Planner should retain all three indices for broad change analysis.
    """
    query = "What changed between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]?"
    plan = parse_query(QueryRequest(query=query))
    assert plan.explicit_metric is None
    tools = create_execution_plan(plan)
    assert "calculate_temporal_ndvi" in tools
    assert "calculate_temporal_ndwi" in tools
    assert "calculate_temporal_ndbi" in tools
    assert "detect_change" in tools


def test_end_to_end_user_ndvi_query(mock_vlm_offline):
    """
    Test 7: Full end-to-end execution of user's exact NDVI query with actual provider data.
    """
    query = "Compare vegetation/NDVI change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]"
    req = QueryRequest(query=query)
    res = process_query(req)

    assert res.status == "success"
    assert res.statistics.get("metric") == "NDVI"
    assert "mean_before" in res.statistics
    assert "mean_after" in res.statistics
    assert "mean_change" in res.statistics
    assert res.answer is not None
    # Only NDVI was calculated; NDWI and NDBI were not unnecessarily calculated
    indices = res.statistics.get("indices", {})
    assert "NDVI" in indices
    assert "NDWI" not in indices
    assert "NDBI" not in indices
