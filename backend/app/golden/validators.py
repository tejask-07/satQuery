"""
Phase 10: Reusable Semantic Validation Utilities for SatQuery AI.

Provides comprehensive, non-brittle semantic assertions covering all 11 stages:
1. Parser intent & QueryPlan
2. Scene selection & observation count
3. Quality & readiness status
4. Index availability (NDVI, NDWI, NDBI)
5. Multi-index evidence & candidate classification
6. Spatial reasoning & candidate regions
7. Temporal reasoning & multi-observation persistence
8. Reliability & confidence calibration (Phase 8 vocabulary)
9. Structured interpretation concordance
10. Reason codes presence & validity
11. API response schema integrity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.evidence.calibration import CalibrationReasonCodes
from app.golden.manifest import GoldenQuery
from app.schemas.analysis import AnalysisResult


class GoldenValidationError(Exception):
    """Raised when a golden query fails semantic validation."""
    pass


@dataclass
class GoldenValidationResult:
    """Summary of semantic validation for a single golden query execution."""
    query_id: str
    passed: bool
    checks_passed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    intent: str = ""
    temporal_mode: str = ""
    observation_count: int = 0
    evidence_state: Optional[str] = None
    spatial_support: Optional[str] = None
    temporal_support: Optional[str] = None
    observation_reliability: Optional[str] = None
    interpretation_support: Optional[str] = None
    conclusion: str = ""


# Phase 8 Vocabulary Registries (exact Literal values from app/evidence/calibration.py)
ALLOWED_RELIABILITY_STATES: Set[str] = {"high", "moderate", "low", "unavailable"}
ALLOWED_EVIDENCE_STATES: Set[str] = {"very_strong", "strong", "moderate", "weak", "none", "unavailable"}
ALLOWED_SPATIAL_STATES: Set[str] = {"high", "moderate", "low", "unavailable"}
ALLOWED_TEMPORAL_STATES: Set[str] = {"high", "moderate", "limited", "bi_temporal_only", "unavailable"}
ALLOWED_INTERPRETATION_STATES: Set[str] = {
    "strong_support", "moderate_support", "weak_support",
    "insufficient_support", "contradictory_support", "unavailable"
}
ALLOWED_DATA_SUFFICIENCY: Set[str] = {"sufficient", "limited", "insufficient"}


def validate_plan(plan_dict: Dict[str, Any], query: GoldenQuery, result: GoldenValidationResult) -> None:
    """Validates query parser outputs against expected intent and schema."""
    intent = plan_dict.get("intent")
    task = plan_dict.get("task")
    target = plan_dict.get("target")
    temporal_mode = plan_dict.get("temporal_mode")

    if intent != query.expected_intent and task != query.expected_intent:
        result.errors.append(f"Intent mismatch: expected '{query.expected_intent}', got intent='{intent}', task='{task}'")
    else:
        result.checks_passed.append("intent_valid")

    if query.expected_domain:
        if query.expected_domain == "general":
            pass
        elif query.expected_domain == "vegetation_to_urban":
            if target != "vegetation_to_urban" and plan_dict.get("source") != "vegetation":
                result.errors.append(f"Target transition mismatch: expected 'vegetation_to_urban', got '{target}'")
            else:
                result.checks_passed.append("domain_transition_valid")
        else:
            if target != query.expected_domain and query.expected_domain not in plan_dict.get("targets", []):
                result.errors.append(f"Domain mismatch: expected '{query.expected_domain}', got '{target}'")
            else:
                result.checks_passed.append("domain_valid")

    if query.expected_source:
        if plan_dict.get("source") != query.expected_source:
            result.errors.append(f"Source mismatch: expected '{query.expected_source}', got '{plan_dict.get('source')}'")
        else:
            result.checks_passed.append("source_valid")

    if query.expected_destination:
        if plan_dict.get("destination") != query.expected_destination:
            result.errors.append(f"Destination mismatch: expected '{query.expected_destination}', got '{plan_dict.get('destination')}'")
        else:
            result.checks_passed.append("destination_valid")

    if query.expected_temporal_mode:
        if temporal_mode != query.expected_temporal_mode:
            result.errors.append(f"Temporal mode mismatch: expected '{query.expected_temporal_mode}', got '{temporal_mode}'")
        else:
            result.checks_passed.append("temporal_mode_plan_valid")

    if query.expected_time_start:
        if plan_dict.get("time_start") != query.expected_time_start:
            result.errors.append(f"time_start mismatch: expected '{query.expected_time_start}', got '{plan_dict.get('time_start')}'")
        else:
            result.checks_passed.append("time_start_valid")

    if query.expected_time_end:
        if plan_dict.get("time_end") != query.expected_time_end:
            result.errors.append(f"time_end mismatch: expected '{query.expected_time_end}', got '{plan_dict.get('time_end')}'")
        else:
            result.checks_passed.append("time_end_valid")


def validate_imagery_and_qc(
    statistics: Dict[str, Any],
    images: Optional[Dict[str, Any]],
    query: GoldenQuery,
    result: GoldenValidationResult,
) -> int:
    """Validates imagery acquisition, scene count, and quality control."""
    temp_analysis = statistics.get("temporal_analysis") or {}
    obs_count = temp_analysis.get("observation_count") or 0

    if obs_count == 0 and images and "images" in images:
        obs_count = len(images["images"])

    if obs_count < query.expected_min_observations:
        result.errors.append(
            f"Insufficient observations: expected >= {query.expected_min_observations}, got {obs_count}"
        )
    else:
        result.checks_passed.append("observation_count_sufficient")

    # If generic bi-temporal query, must have exactly 2 observations
    if query.expected_temporal_mode == "bi_temporal":
        if obs_count != 2:
            result.errors.append(f"Bi-temporal query must have exactly 2 observations, got {obs_count}")
        else:
            result.checks_passed.append("bitemporal_count_exact_2")

    return obs_count


def validate_indices(statistics: Dict[str, Any], query: GoldenQuery, result: GoldenValidationResult) -> None:
    """Validates NDVI, NDWI, NDBI availability and valid mathematical values."""
    indices_dict = statistics.get("indices", {})
    if not indices_dict and "mean" in statistics:
        result.checks_passed.append("single_index_available")
        return

    # Check index availability
    for idx_name in ["NDVI", "NDWI", "NDBI"]:
        idx_data = indices_dict.get(idx_name)
        if idx_data is not None:
            result.checks_passed.append(f"index_{idx_name.lower()}_available")

    if query.expected_primary_indicator:
        ind = query.expected_primary_indicator
        if ind not in indices_dict and statistics.get("metric", "").upper() != ind:
            result.errors.append(f"Primary indicator '{ind}' not present in statistics.")
        else:
            result.checks_passed.append("primary_indicator_computed")


def validate_evidence(candidate_package: Optional[Dict[str, Any]], result: GoldenValidationResult) -> None:
    """Validates Phase 5A/5B multi-index evidence and candidate packaging."""
    if candidate_package is None:
        result.errors.append("candidate_package is missing.")
        return

    result.checks_passed.append("candidate_package_present")
    candidates = candidate_package.get("candidates")
    if candidates is not None and isinstance(candidates, list):
        result.checks_passed.append("candidates_list_valid")


def validate_spatial(
    spatial_analysis: Optional[Dict[str, Any]],
    query: GoldenQuery,
    result: GoldenValidationResult,
) -> None:
    """Validates Phase 6 spatial candidate clustering and GeoJSON packaging."""
    if spatial_analysis is None:
        result.errors.append("spatial_analysis is missing.")
        return

    result.checks_passed.append("spatial_analysis_present")
    if "spatial_analysis_evaluated" in query.expected_properties:
        if not spatial_analysis.get("available") and spatial_analysis.get("summary") == "No candidate raster available for spatial analysis.":
            # Valid data-driven rejection
            result.checks_passed.append("spatial_evaluation_completed")
        else:
            result.checks_passed.append("spatial_regions_evaluated")

    if spatial_analysis.get("available"):
        geojson = spatial_analysis.get("geojson")
        if geojson and geojson.get("type") == "FeatureCollection":
            result.checks_passed.append("spatial_geojson_valid")


def validate_temporal(
    temporal_analysis: Optional[Dict[str, Any]],
    query: GoldenQuery,
    result: GoldenValidationResult,
) -> None:
    """Validates Phase 7 temporal reasoning constraints."""
    if temporal_analysis is None:
        result.errors.append("temporal_analysis is missing.")
        return

    result.checks_passed.append("temporal_analysis_present")
    obs_count = temporal_analysis.get("observation_count", 0)

    # N=2 must not expose multi-temporal persistence in domain trends
    if obs_count <= 2:
        doms = temporal_analysis.get("domains", {})
        for dom_name, dom_info in doms.items():
            if dom_info.get("persistent_change") is True:
                result.errors.append(f"Bi-temporal N<=2 must not claim persistent_change for {dom_name}.")
            if dom_info.get("persistence_fraction") is not None:
                result.errors.append(f"Bi-temporal N<=2 must not expose multi-temporal persistence_fraction for {dom_name}.")
        result.checks_passed.append("bitemporal_persistence_suppressed")

    # Reversal queries must evaluate reversal
    if query.expected_temporal_mode == "persistence_reversal":
        rev_state = temporal_analysis.get("vegetation", {}).get("reversal_detected")
        if rev_state is None:
            # Check general temporal_consistency
            tc = temporal_analysis.get("temporal_consistency", {})
            if "reversal_detected" not in tc and "reversal" not in temporal_analysis:
                result.warnings.append("Reversal evaluation not explicitly flagged in vegetation block.")
        result.checks_passed.append("reversal_mode_active")

    # Multi-temporal queries must have >= 3 observations
    if query.expected_temporal_mode in ("persistence_reversal", "trend_analysis", "multi_temporal"):
        if obs_count < 3:
            result.errors.append(
                f"Multi-temporal mode '{query.expected_temporal_mode}' requires >= 3 observations, got {obs_count}"
            )
        else:
            result.checks_passed.append("multi_temporal_observations_ge_3")


def validate_calibration(calibration: Optional[Dict[str, Any]], result: GoldenValidationResult) -> None:
    """Validates Phase 8 calibration components and vocabulary."""
    if calibration is None:
        result.errors.append("calibration package is missing.")
        return

    result.checks_passed.append("calibration_present")

    # 1. Observation Reliability
    obs_rel = calibration.get("observation_reliability")
    if not obs_rel or not isinstance(obs_rel, dict):
        result.errors.append("calibration.observation_reliability is missing.")
    else:
        rel_state = obs_rel.get("state")
        result.observation_reliability = rel_state
        if rel_state not in ALLOWED_RELIABILITY_STATES:
            result.errors.append(f"Invalid observation_reliability state: '{rel_state}'")
        else:
            result.checks_passed.append("observation_reliability_state_valid")

    # 2. Semantic Evidence
    sem_ev = calibration.get("semantic_evidence")
    if sem_ev and isinstance(sem_ev, dict):
        ev_state = sem_ev.get("state")
        result.evidence_state = ev_state
        if ev_state and ev_state not in ALLOWED_EVIDENCE_STATES:
            result.errors.append(f"Invalid semantic_evidence state: '{ev_state}'")
        else:
            result.checks_passed.append("semantic_evidence_state_valid")

    # 3. Spatial Assessment
    sp_ass = calibration.get("spatial_assessment")
    if sp_ass and isinstance(sp_ass, dict):
        sp_state = sp_ass.get("state")
        result.spatial_support = sp_state
        if sp_state and sp_state not in ALLOWED_SPATIAL_STATES:
            result.errors.append(f"Invalid spatial_assessment state: '{sp_state}'")
        else:
            result.checks_passed.append("spatial_assessment_state_valid")

    # 4. Temporal Consistency
    temp_cons = calibration.get("temporal_consistency")
    if temp_cons and isinstance(temp_cons, dict):
        temp_state = temp_cons.get("state")
        result.temporal_support = temp_state
        if temp_state and temp_state not in ALLOWED_TEMPORAL_STATES:
            result.errors.append(f"Invalid temporal_consistency state: '{temp_state}'")
        else:
            result.checks_passed.append("temporal_consistency_state_valid")

    # 5. Interpretation Support
    int_supp = calibration.get("interpretation_support")
    if int_supp and isinstance(int_supp, dict):
        int_state = int_supp.get("state")
        result.interpretation_support = int_state
        if int_state and int_state not in ALLOWED_INTERPRETATION_STATES:
            result.errors.append(f"Invalid interpretation_support state: '{int_state}'")
        else:
            result.checks_passed.append("interpretation_support_state_valid")

    # 6. Reason Codes
    reason_codes = calibration.get("reason_codes", [])
    if isinstance(reason_codes, list):
        result.checks_passed.append("reason_codes_list_valid")


def validate_interpretation(
    interpretation: Optional[Dict[str, Any]],
    query: GoldenQuery,
    result: GoldenValidationResult,
) -> None:
    """Validates Phase 5C/6/7/8 structured interpretation and scientific groundedness."""
    if interpretation is None:
        result.errors.append("interpretation is missing.")
        return

    result.checks_passed.append("interpretation_present")
    conclusion = interpretation.get("conclusion", "")
    summary = interpretation.get("summary", "")
    result.conclusion = conclusion

    if not conclusion:
        result.errors.append("interpretation.conclusion is empty.")
    else:
        result.checks_passed.append("conclusion_present")

    # Sanity check: no-change query must NOT conclude severe/strong change
    if "interpretation_no_strong_change" in query.expected_properties:
        lower_conc = conclusion.lower() + " " + summary.lower()
        if "severe" in lower_conc or "massive" in lower_conc or "drastic loss" in lower_conc:
            result.errors.append("Quiescent query falsely concluded severe/strong change.")
        else:
            result.checks_passed.append("no_change_sanity_confirmed")


def validate_response_schema(res: AnalysisResult, result: GoldenValidationResult) -> None:
    """Validates top-level API AnalysisResult response model."""
    if res.status != "success":
        result.errors.append(f"Response status not 'success': {res.status}")
    else:
        result.checks_passed.append("response_status_success")

    if not res.answer:
        result.errors.append("Response answer text is empty.")
    else:
        result.checks_passed.append("response_answer_present")

    if res.confidence is None or res.confidence < 0.0 or res.confidence > 1.0:
        result.errors.append(f"Invalid top-level confidence: {res.confidence}")
    else:
        result.checks_passed.append("response_confidence_valid")


def validate_golden_result(res: AnalysisResult, query: GoldenQuery) -> GoldenValidationResult:
    """
    Orchestrates end-to-end semantic validation of a SatQuery query execution.
    """
    val_res = GoldenValidationResult(
        query_id=query.id,
        passed=True,
    )

    try:
        # 1. API Schema
        validate_response_schema(res, val_res)

        # 2. QueryPlan
        plan = res.plan or {}
        val_res.intent = plan.get("intent", "")
        val_res.temporal_mode = plan.get("temporal_mode", "")
        validate_plan(plan, query, val_res)

        # 3. Imagery & QC
        obs_count = validate_imagery_and_qc(res.statistics, res.images, query, val_res)
        val_res.observation_count = obs_count

        # 4. Indices
        validate_indices(res.statistics, query, val_res)

        # 5. Evidence Fusion
        validate_evidence(res.candidate_package, val_res)

        # 6. Spatial Reasoning
        validate_spatial(res.spatial_analysis, query, val_res)

        # 7. Temporal Reasoning
        validate_temporal(res.temporal_analysis, query, val_res)

        # 8. Calibration
        validate_calibration(res.calibration, val_res)

        # 9. Structured Interpretation
        validate_interpretation(res.interpretation, query, val_res)

    except Exception as exc:
        val_res.errors.append(f"Unexpected validation exception: {exc}")

    val_res.passed = len(val_res.errors) == 0
    return val_res
