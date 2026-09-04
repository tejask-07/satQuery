"""
Phase 10: Golden End-to-End Validation Test Suite for SatQuery AI.

Validates the entire pipeline flow:
query
→ parser
→ QueryPlan
→ scene selection
→ quality control
→ indices
→ evidence
→ evidence fusion
→ spatial reasoning
→ temporal reasoning
→ calibration
→ structured interpretation
→ API response
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.golden.manifest import load_golden_manifest, GoldenQuery
from app.golden.validators import validate_golden_result
from app.golden.runner import run_golden_suite
from app.schemas.query import QueryRequest
from app.api.routes_query import process_query
from app.evaluation.paths import resolve_repo_path


# ============================================================
# 1. MANIFEST INTEGRITY & STRUCTURE
# ============================================================

def test_golden_manifest_structure_and_integrity():
    """Manifest must exist, contain at least 12 golden queries, and have valid fields."""
    queries = load_golden_manifest()
    assert len(queries) >= 12

    seen_ids = set()
    for q in queries:
        assert q.id.startswith("GOLDEN-")
        assert q.id not in seen_ids
        seen_ids.add(q.id)

        assert len(q.query.strip()) > 10
        assert len(q.aoi) == 4
        assert q.aoi[0] < q.aoi[2], "minLon must be < maxLon"
        assert q.aoi[1] < q.aoi[3], "minLat must be < maxLat"
        assert q.expected_intent in {
            "change_detection",
            "general_change_detection",
            "land_cover_transition",
            "image_comparison",
            "single_index",
        }
        assert q.expected_min_observations >= 2
        assert len(q.expected_properties) > 0


# ============================================================
# 2. PARAMETERIZED END-TO-END EXECUTION OF ALL GOLDEN QUERIES
# ============================================================

GOLDEN_QUERIES = load_golden_manifest()


@pytest.mark.parametrize("golden_query", GOLDEN_QUERIES, ids=lambda q: q.id)
def test_golden_query_end_to_end_validation(golden_query: GoldenQuery):
    """Executes each golden query through process_query and validates semantic properties."""
    req = QueryRequest(query=golden_query.query)
    res = process_query(req)

    val_res = validate_golden_result(res, golden_query)
    assert val_res.passed is True, (
        f"Query {golden_query.id} failed validation:\n" + "\n".join(f" - {e}" for e in val_res.errors)
    )
    assert len(val_res.checks_passed) >= 5


# ============================================================
# 3. SEMANTIC ASSERTIONS: RECOVERY & MULTI-TEMPORAL MODE
# ============================================================

def test_golden_semantic_recovery_multi_temporal_mode():
    """Vegetation recovery queries must enter multi-temporal reversal mode with >= 3 observations."""
    queries = {q.id: q for q in load_golden_manifest()}
    gq = queries["GOLDEN-06-VEGETATION-RECOVERY"]

    req = QueryRequest(query=gq.query)
    res = process_query(req)

    assert res.plan.get("temporal_mode") == "persistence_reversal"
    assert res.temporal_analysis is not None
    obs_count = res.temporal_analysis.get("observation_count", 0)
    assert obs_count >= 3
    veg_data = res.temporal_analysis.get("domains", {}).get("vegetation", {}) or res.temporal_analysis.get("vegetation", {})
    assert "reversal_detected" in veg_data


# ============================================================
# 4. SEMANTIC ASSERTIONS: BI-TEMPORAL PERSISTENCE SUPPRESSION
# ============================================================

def test_golden_semantic_bitemporal_persistence_suppressed():
    """For N=2 bi-temporal queries, domain persistence must be suppressed (cannot claim trend)."""
    queries = {q.id: q for q in load_golden_manifest()}
    gq = queries["GOLDEN-01-URBAN-CHANGE"]

    req = QueryRequest(query=gq.query)
    res = process_query(req)

    assert res.plan.get("temporal_mode") == "bi_temporal"
    assert res.temporal_analysis is not None
    assert res.temporal_analysis.get("observation_count") == 2

    # In domain statistics, persistence must be None / False for N=2
    doms = res.temporal_analysis.get("domains", {})
    for dom_name, dom_data in doms.items():
        assert dom_data.get("persistent_change") is False
        assert dom_data.get("persistence_fraction") is None
        assert dom_data.get("data_sufficiency") == "limited_bi_temporal"


# ============================================================
# 5. SEMANTIC ASSERTIONS: SPATIAL LOCALIZATION
# ============================================================

def test_golden_semantic_spatial_candidate_localization():
    """Spatial query must execute candidate clustering and expose structured spatial analysis."""
    queries = {q.id: q for q in load_golden_manifest()}
    gq = queries["GOLDEN-08-SPATIAL-LOCALIZATION"]

    req = QueryRequest(query=gq.query)
    res = process_query(req)

    assert res.spatial_analysis is not None
    assert "region_count" in res.spatial_analysis
    assert "total_candidate_area_hectares" in res.spatial_analysis

    # Check that spatial calibration assessment exists
    assert res.calibration is not None
    sp_ass = res.calibration.get("spatial_assessment")
    assert sp_ass is not None
    assert sp_ass.get("state") in {"high", "moderate", "low", "unavailable"}


# ============================================================
# 6. SEMANTIC ASSERTIONS: CALIBRATION VOCABULARY
# ============================================================

def test_golden_semantic_calibration_vocabulary():
    """Calibration outputs must strictly adhere to Phase 8 vocabulary."""
    queries = {q.id: q for q in load_golden_manifest()}
    gq = queries["GOLDEN-09-CALIBRATION-RELIABILITY"]

    req = QueryRequest(query=gq.query)
    res = process_query(req)

    cal = res.calibration
    assert cal is not None

    # Check 5 separated components exist
    assert "observation_reliability" in cal
    assert "semantic_evidence" in cal
    assert "spatial_assessment" in cal
    assert "temporal_consistency" in cal
    assert "interpretation_support" in cal

    # Check state vocabularies
    assert cal["observation_reliability"]["state"] in {"high", "moderate", "low", "unavailable"}
    assert cal["semantic_evidence"]["state"] in {"very_strong", "strong", "moderate", "weak", "none", "unavailable"}
    assert cal["spatial_assessment"]["state"] in {"high", "moderate", "low", "unavailable"}
    assert cal["temporal_consistency"]["state"] in {"high", "moderate", "limited", "bi_temporal_only", "unavailable"}
    assert cal["interpretation_support"]["state"] in {
        "strong_support", "moderate_support", "weak_support",
        "insufficient_support", "contradictory_support", "unavailable"
    }


# ============================================================
# 7. SEMANTIC ASSERTIONS: NO-CHANGE QUIESCENCE SANITY
# ============================================================

def test_golden_semantic_no_change_quiescence():
    """In quiescent regions, SatQuery must not conclude severe or drastic change."""
    queries = {q.id: q for q in load_golden_manifest()}
    gq = queries["GOLDEN-10-NO-CHANGE-SANITY"]

    req = QueryRequest(query=gq.query)
    res = process_query(req)

    assert res.interpretation is not None
    conclusion = res.interpretation.get("conclusion", "").lower()
    summary = res.interpretation.get("summary", "").lower()
    full_text = f"{conclusion} {summary}"

    assert "severe deforestation" not in full_text
    assert "drastic loss" not in full_text
    assert "massive destruction" not in full_text


# ============================================================
# 8. API-LEVEL FASTAPI TESTCLIENT VERIFICATION
# ============================================================

def test_golden_api_endpoint_post_query():
    """POST /api/query returns AnalysisResult with status success and complete packages."""
    client = TestClient(app)
    payload = {
        "query": "Compare urban change between 2020 and 2021 for AOI [16.40, 48.20, 16.41, 48.21]"
    }
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["answer"] is not None
    assert len(data["answer"]) > 20
    assert "calibration" in data
    assert "interpretation" in data
    assert "spatial_analysis" in data
    assert "temporal_analysis" in data
    assert "multi_index_evidence" in data


# ============================================================
# 9. RUNNER & REPORT EXPORT VERIFICATION
# ============================================================

def test_golden_runner_generates_valid_report():
    """Runner CLI executes all queries and exports golden_suite_report.json."""
    results, summary = run_golden_suite(save_report=True)
    assert len(results) >= 12
    assert summary["passed_queries"] == len(results)
    assert summary["failed_queries"] == 0
    assert summary["pass_rate"] == 100.0

    report_path = resolve_repo_path("reports/golden_suite_report.json")
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        rep_data = json.load(f)
    assert rep_data["total_queries"] == len(results)
    assert rep_data["pass_rate"] == 100.0


# ============================================================
# 10. LIVE SENTINEL-2 SMOKE TEST (SEPARATED)
# ============================================================

@pytest.mark.live
def test_live_sentinel2_smoke_query():
    """Optional smoke test for live STAC network retrieval."""
    from app.tools.imagery import search_imagery
    aoi_vienna = {
        "type": "Polygon",
        "coordinates": [[
            [16.40, 48.20],
            [16.41, 48.20],
            [16.41, 48.21],
            [16.40, 48.21],
            [16.40, 48.20]
        ]]
    }
    res = search_imagery(time_start="2021", time_end="2022", aoi=aoi_vienna)
    assert res["status"] in {"success", "error"}
