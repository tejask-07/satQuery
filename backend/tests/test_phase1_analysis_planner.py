"""
Phase 1 Test Suite: Query Understanding + Analysis Planner

Tests:
1. Urban change query parsing & structured plan
2. Vegetation loss query parsing & structured plan
3. Water change query parsing & structured plan
4. General change query ("What changed?") parsing & multi-target structured plan
5. Land cover transition query ("Did vegetation become urban?") parsing & transition plan
6. Natural language variations for urban, vegetation, water, and transitions
7. Deterministic AOI and date extraction
8. Schema compliance and allowed-values validation
"""

import pytest
from app.agent.parser import (
    parse_query,
    ALLOWED_INTENTS,
    ALLOWED_INDICATORS,
    ALLOWED_EVIDENCE,
    ALLOWED_OUTPUTS,
)
from app.schemas.query import QueryRequest, QueryPlan
from app.remote_sensing.providers.sentinel2 import normalize_aoi


# ============================================================
# 1. CORE SCENARIOS FROM PROMPT
# ============================================================

def test_urban_change_plan():
    query = "Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
    req = QueryRequest(query=query)
    plan = parse_query(req)

    # 1. Intent & Targets
    assert plan.intent == "change_detection"
    assert plan.target == "urban"
    assert plan.targets == ["urban"]

    # 2. Dates
    assert plan.time_start == "2021"
    assert plan.time_end == "2025"

    # 3. AOI
    assert plan.aoi is not None
    assert plan.aoi["type"] == "Polygon"
    w, s, e, n = normalize_aoi(plan.aoi)
    assert pytest.approx(w) == 16.40
    assert pytest.approx(s) == 48.20
    assert pytest.approx(e) == 16.41
    assert pytest.approx(n) == 48.21

    # Verify GeoJSON polygon bounds directly
    coords = plan.aoi["coordinates"][0]
    lngs = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    assert pytest.approx(min(lngs)) == 16.40
    assert pytest.approx(max(lngs)) == 16.41
    assert pytest.approx(min(lats)) == 48.20
    assert pytest.approx(max(lats)) == 48.21

    # 4. Indicators
    assert plan.primary_indicators == ["NDBI"]
    assert "NDVI" in plan.supporting_indicators
    assert "spectral_change" in plan.supporting_indicators
    assert "spatial_consistency" in plan.supporting_indicators
    assert "temporal_consistency" in plan.supporting_indicators

    # 5. Evidence requirements
    for req_ev in ["spectral", "spatial", "temporal", "data_quality"]:
        assert req_ev in plan.evidence_requirements

    # 6. Outputs
    for out in ["map", "statistics", "explanation", "confidence"]:
        assert out in plan.outputs

    # 7. Backward compatibility
    assert plan.task == "urban_change"
    assert plan.metric == "ndbi"


def test_vegetation_loss_plan():
    query = "Analyze vegetation loss between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
    req = QueryRequest(query=query)
    plan = parse_query(req)

    # 1. Intent & Targets
    assert plan.intent == "change_detection"
    assert plan.target == "vegetation"
    assert plan.targets == ["vegetation"]
    assert plan.direction == "decrease"

    # 2. Dates
    assert plan.time_start == "2021"
    assert plan.time_end == "2025"

    # 3. AOI
    assert plan.aoi is not None
    w, s, e, n = normalize_aoi(plan.aoi)
    assert pytest.approx(w) == 16.40
    assert pytest.approx(s) == 48.20
    assert pytest.approx(e) == 16.41
    assert pytest.approx(n) == 48.21

    # 4. Indicators
    assert plan.primary_indicators == ["NDVI"]
    assert "NDBI" in plan.supporting_indicators
    assert "spectral_change" in plan.supporting_indicators
    assert "spatial_consistency" in plan.supporting_indicators
    assert "temporal_consistency" in plan.supporting_indicators

    # 5. Evidence requirements
    for req_ev in ["spectral", "spatial", "temporal", "data_quality"]:
        assert req_ev in plan.evidence_requirements

    # 6. Outputs
    for out in ["map", "statistics", "explanation", "confidence"]:
        assert out in plan.outputs

    # 7. Backward compatibility
    assert plan.task == "change_detection"
    assert plan.metric == "ndvi"


def test_water_change_plan():
    query = "Analyze water change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
    req = QueryRequest(query=query)
    plan = parse_query(req)

    # 1. Intent & Targets
    assert plan.intent == "change_detection"
    assert plan.target == "water"
    assert plan.targets == ["water"]

    # 2. Dates
    assert plan.time_start == "2021"
    assert plan.time_end == "2025"

    # 3. AOI
    assert plan.aoi is not None
    w, s, e, n = normalize_aoi(plan.aoi)
    assert pytest.approx(w) == 16.40
    assert pytest.approx(s) == 48.20
    assert pytest.approx(e) == 16.41
    assert pytest.approx(n) == 48.21

    # 4. Indicators
    assert plan.primary_indicators == ["NDWI"]
    assert "NDVI" in plan.supporting_indicators
    assert "spectral_change" in plan.supporting_indicators
    assert "spatial_consistency" in plan.supporting_indicators
    assert "temporal_consistency" in plan.supporting_indicators

    # 5. Evidence requirements
    for req_ev in ["spectral", "spatial", "temporal", "data_quality"]:
        assert req_ev in plan.evidence_requirements

    # 6. Outputs
    for out in ["map", "statistics", "explanation", "confidence"]:
        assert out in plan.outputs

    # 7. Backward compatibility
    assert plan.task == "water_change"
    assert plan.metric == "ndwi"


def test_general_change_plan():
    query = "What changed between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]?"
    req = QueryRequest(query=query)
    plan = parse_query(req)

    # 1. Intent & Targets
    assert plan.intent == "general_change_detection"
    assert plan.target is None
    assert set(plan.targets) == {"urban", "vegetation", "water"}

    # 2. Dates
    assert plan.time_start == "2021"
    assert plan.time_end == "2025"

    # 3. AOI
    assert plan.aoi is not None
    w, s, e, n = normalize_aoi(plan.aoi)
    assert pytest.approx(w) == 16.40
    assert pytest.approx(s) == 48.20
    assert pytest.approx(e) == 16.41
    assert pytest.approx(n) == 48.21

    # 4. Primary Indicators cover all 3 primary domains
    assert "NDBI" in plan.primary_indicators
    assert "NDVI" in plan.primary_indicators
    assert "NDWI" in plan.primary_indicators

    # 5. Supporting indicators
    assert "spectral_change" in plan.supporting_indicators
    assert "spatial_consistency" in plan.supporting_indicators
    assert "temporal_consistency" in plan.supporting_indicators

    # 6. Evidence requirements
    for req_ev in ["spectral", "spatial", "temporal", "data_quality"]:
        assert req_ev in plan.evidence_requirements

    # 7. Outputs
    for out in ["map", "statistics", "explanation", "confidence"]:
        assert out in plan.outputs


def test_land_cover_transition_plan():
    query = "Did vegetation become urban between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]?"
    req = QueryRequest(query=query)
    plan = parse_query(req)

    # 1. Intent & Source / Destination
    assert plan.intent == "land_cover_transition"
    assert plan.source == "vegetation"
    assert plan.destination == "urban"
    assert plan.targets == ["vegetation", "urban"]

    # 2. Dates
    assert plan.time_start == "2021"
    assert plan.time_end == "2025"

    # 3. AOI
    assert plan.aoi is not None
    w, s, e, n = normalize_aoi(plan.aoi)
    assert pytest.approx(w) == 16.40
    assert pytest.approx(s) == 48.20
    assert pytest.approx(e) == 16.41
    assert pytest.approx(n) == 48.21

    # 4. Indicators (NDVI for vegetation, NDBI for urban)
    assert plan.primary_indicators == ["NDVI", "NDBI"]
    assert "spectral_change" in plan.supporting_indicators
    assert "spatial_consistency" in plan.supporting_indicators
    assert "temporal_consistency" in plan.supporting_indicators

    # 5. Evidence requirements
    for req_ev in ["spectral", "spatial", "temporal", "data_quality"]:
        assert req_ev in plan.evidence_requirements

    # 6. Outputs
    for out in ["map", "statistics", "explanation", "confidence"]:
        assert out in plan.outputs

        assert out in plan.outputs


# ============================================================
# 2. NATURAL LANGUAGE VARIATIONS
# ============================================================

@pytest.mark.parametrize(
    "query,expected_target,expected_dir",
    [
        ("Detect urban growth between 2020 and 2024", "urban", "increase"),
        ("Show urban expansion between 2021 and 2025", "urban", "increase"),
        ("Evaluate built-up growth from 2020 to 2023", "urban", "increase"),
        ("Monitor construction growth in the area between 2021 and 2024", "urban", "increase"),
        ("Did the city expand between 2021 and 2025?", "urban", "increase"),
    ],
)
def test_urban_natural_language_variations(query, expected_target, expected_dir):
    req = QueryRequest(query=query)
    plan = parse_query(req)
    assert plan.intent == "change_detection"
    assert plan.target == expected_target
    assert plan.direction == expected_dir
    assert "NDBI" in plan.primary_indicators


@pytest.mark.parametrize(
    "query,expected_target,expected_dir",
    [
        ("Assess forest degradation between 2020 and 2024", "vegetation", "decrease"),
        ("Measure greenness decrease between 2021 and 2025", "vegetation", "decrease"),
        ("Show vegetation increase between 2020 and 2024", "vegetation", "increase"),
        ("Analyze tree cover decline from 2021 to 2025", "vegetation", "decrease"),
        ("Track deforestation between 2021 and 2025", "vegetation", "decrease"),
    ],
)
def test_vegetation_natural_language_variations(query, expected_target, expected_dir):
    req = QueryRequest(query=query)
    plan = parse_query(req)
    assert plan.intent == "change_detection"
    assert plan.target == expected_target
    assert plan.direction == expected_dir
    assert "NDVI" in plan.primary_indicators


@pytest.mark.parametrize(
    "query,expected_target,expected_dir",
    [
        ("Detect lake shrinkage between 2020 and 2024", "water", "decrease"),
        ("Analyze water loss between 2021 and 2025", "water", "decrease"),
        ("Show water increase following 2021 to 2024", "water", "increase"),
        ("Monitor water body change between 2021 and 2025", "water", "unknown"),
        ("Find wetland drying between 2020 and 2025", "water", "decrease"),
    ],
)
def test_water_natural_language_variations(query, expected_target, expected_dir):
    req = QueryRequest(query=query)
    plan = parse_query(req)
    assert plan.intent == "change_detection"
    assert plan.target == expected_target
    assert plan.direction == expected_dir
    assert "NDWI" in plan.primary_indicators


@pytest.mark.parametrize(
    "query,expected_src,expected_dst",
    [
        ("Has forest turned into urban between 2020 and 2025?", "vegetation", "urban"),
        ("Track transition from vegetation to urban between 2021 and 2025", "vegetation", "urban"),
        ("Conversion between water and urban from 2021 to 2025", "water", "urban"),
        ("Did crops become buildings between 2020 and 2024?", "vegetation", "urban"),
    ],
)
def test_transition_natural_language_variations(query, expected_src, expected_dst):
    req = QueryRequest(query=query)
    plan = parse_query(req)
    assert plan.intent == "land_cover_transition"
    assert plan.source == expected_src
    assert plan.destination == expected_dst


# ============================================================
# 3. SCHEMA VALUE REGISTRY COMPLIANCE
# ============================================================

def test_schema_compliance_all_queries():
    test_queries = [
        "Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]",
        "Analyze vegetation loss between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]",
        "Analyze water change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]",
        "What changed between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]?",
        "Did vegetation become urban between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]?",
        "Calculate NDVI for AOI [151.195, -33.885, 151.225, -33.855]",
        "Compare these two satellite images.",
    ]


    for q in test_queries:
        req = QueryRequest(query=q)
        plan = parse_query(req)

        assert plan.intent in ALLOWED_INTENTS

        for ind in plan.primary_indicators:
            assert ind in ALLOWED_INDICATORS

        for ind in plan.supporting_indicators:
            assert ind in ALLOWED_INDICATORS

        for ev in plan.evidence_requirements:
            assert ev in ALLOWED_EVIDENCE

        for out in plan.outputs:
            assert out in ALLOWED_OUTPUTS
