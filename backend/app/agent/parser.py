import re
from typing import Optional, Tuple, List, Dict, Any
from app.schemas.query import QueryRequest, QueryPlan


# ============================================================
# ALLOWED SCHEMA VALUE REGISTRIES (FOR PHASE 1 VALIDATION)
# ============================================================

ALLOWED_INTENTS = {
    "change_detection",
    "general_change_detection",
    "land_cover_transition",
    "single_index",
    "image_comparison",
    "image_search",
    "unknown",
}

ALLOWED_INDICATORS = {
    "NDVI",
    "NDWI",
    "NDBI",
    "spectral_change",
    "spatial_consistency",
    "temporal_consistency",
    "vegetation_health",
}

ALLOWED_EVIDENCE = {
    "spectral",
    "spatial",
    "temporal",
    "data_quality",
}

ALLOWED_OUTPUTS = {
    "map",
    "statistics",
    "explanation",
    "confidence",
    "comparison",
    "change_map",
}

# Synonyms for land cover classification
CATEGORY_SYNONYMS: Dict[str, List[str]] = {
    "vegetation": [
        "vegetation",
        "forest",
        "crop",
        "crops",
        "tree",
        "trees",
        "greenery",
        "canopy",
        "plant",
        "plants",
        "greenness",
        "deforestation",
    ],
    "urban": [
        "urban",
        "city",
        "building",
        "buildings",
        "built-up",
        "built up",
        "construction",
        "settlement",
        "infrastructure",
        "expansion",
    ],
    "water": [
        "water",
        "flood",
        "lake",
        "river",
        "reservoir",
        "wetland",
        "water body",
        "waterbody",
        "shrinkage",
    ],
}

CATEGORY_PRIMARY_INDEX: Dict[str, str] = {
    "vegetation": "NDVI",
    "water": "NDWI",
    "urban": "NDBI",
}


# ============================================================
# DETERMINISTIC EXTRACTION HELPERS
# ============================================================

def extract_aoi(query_str: str, request_aoi: Optional[dict] = None) -> Tuple[Optional[dict], str]:
    """
    Extract GeoJSON Polygon from query string [minLon, minLat, maxLon, maxLat]
    or retain request_aoi if already provided.
    """
    cleaned_query = query_str
    parsed_aoi = None

    aoi_match = re.search(
        r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]',
        query_str,
    )
    if aoi_match:
        v1, v2, v3, v4 = [float(x) for x in aoi_match.groups()]
        minLon = min(v1, v3)
        maxLon = max(v1, v3)
        minLat = min(v2, v4)
        maxLat = max(v2, v4)
        parsed_aoi = {
            "type": "Polygon",
            "coordinates": [[
                [minLon, minLat],
                [maxLon, minLat],
                [maxLon, maxLat],
                [minLon, maxLat],
                [minLon, minLat]
            ]]
        }
        # Clean query string
        cleaned_query = cleaned_query[:aoi_match.start()] + cleaned_query[aoi_match.end():]
        cleaned_query = re.sub(r'\bfor\s+aoi\b', '', cleaned_query, flags=re.IGNORECASE)
        cleaned_query = re.sub(r'\baoi\b', '', cleaned_query, flags=re.IGNORECASE)
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

    final_aoi = request_aoi if request_aoi else parsed_aoi
    return final_aoi, cleaned_query


def extract_years(query_str: str) -> Tuple[str, str, int]:
    """
    Deterministically extract date (YYYY-MM-DD) or year bounds.
    """
    iso_dates = re.findall(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", query_str)
    if len(iso_dates) >= 2:
        return iso_dates[0], iso_dates[1], len(iso_dates)
    elif len(iso_dates) == 1:
        return iso_dates[0], iso_dates[0], 1

    years = re.findall(r"\b(?:19|20)\d{2}\b", query_str)
    if len(years) >= 2:
        return years[0], years[1], len(years)
    elif len(years) == 1:
        return years[0], years[0], 1
    else:
        return "2021", "2025", 0


def match_category(text: str) -> Optional[str]:
    """
    Resolve text snippet to one of the canonical categories: vegetation, urban, water.
    """
    lower = text.lower()
    for cat, synonyms in CATEGORY_SYNONYMS.items():
        for syn in synonyms:
            if re.search(rf"\b{re.escape(syn)}\b", lower):
                return cat
    return None


def detect_transition(query: str) -> Optional[Tuple[str, str]]:
    """
    Detect land cover transitions (e.g. 'did vegetation become urban').
    Returns (source, destination) or None.
    """
    # Pattern 1: did/has X become/turn into/convert to Y
    m = re.search(
        r"(?:did|has)\s+(.+?)\s+(?:become|turn\s+into|turned\s+into|convert\s+(?:in)?to|converted\s+(?:in)?to|transition\s+(?:in)?to|transitioned\s+(?:in)?to)\s+(.+?)(?:\s+between|\s+for|\s+in|\?|$)",
        query,
        re.IGNORECASE,
    )
    if m:
        src = match_category(m.group(1))
        dst = match_category(m.group(2))
        if src and dst and src != dst:
            return src, dst

    # Pattern 2: transition/conversion from X to Y
    m = re.search(
        r"(?:transition|conversion)\s+(?:from|between)\s+(.+?)\s+(?:to|into|and)\s+(.+?)(?:\s+between|\s+for|\s+in|\?|$)",
        query,
        re.IGNORECASE,
    )
    if m:
        src = match_category(m.group(1))
        dst = match_category(m.group(2))
        if src and dst and src != dst:
            return src, dst

    # Pattern 3: X became/turned into/replaced by Y
    m = re.search(
        r"(\b\w+\b)\s+(?:became|turn(?:ed)?\s+into|convert(?:ed)?\s+(?:in)?to|transition(?:ed)?\s+(?:in)?to|replaced\s+by)\s+(\b\w+\b)",
        query,
        re.IGNORECASE,
    )
    if m:
        src = match_category(m.group(1))
        dst = match_category(m.group(2))
        if src and dst and src != dst:
            return src, dst

    return None


# ============================================================
# MAIN QUERY PARSER & ANALYSIS PLANNER
# ============================================================

def _parse_query_impl(request: QueryRequest) -> QueryPlan:
    """
    Internal conversion of natural-language query into a structured Analysis Plan (QueryPlan).
    """
    raw_query = request.query
    aoi, cleaned_query = extract_aoi(raw_query, request.aoi)
    query = cleaned_query.lower()

    time_start, time_end, year_count = extract_years(query)

    is_change_query = any(
        w in query
        for w in [
            "change",
            "compare",
            "between",
            "difference",
            "decrease",
            "increase",
            "loss",
            "gain",
            "growth",
            "expansion",
            "shrinkage",
            "degradation",
            "decline",
            "reduced",
            "expanded",
        ]
    )

    # --------------------------------------------------------
    # 1. LAND COVER TRANSITION
    # --------------------------------------------------------
    transition = detect_transition(query)
    if transition:
        src, dst = transition
        src_idx = CATEGORY_PRIMARY_INDEX.get(src, "NDVI")
        dst_idx = CATEGORY_PRIMARY_INDEX.get(dst, "NDBI")
        primary_inds = [src_idx, dst_idx]

        return QueryPlan(
            task="land_cover_transition",
            intent="land_cover_transition",
            source=src,
            destination=dst,
            target=f"{src}_to_{dst}",
            targets=[src, dst],
            time_start=time_start,
            time_end=time_end,
            aoi=aoi,
            modalities=["multispectral"],
            metric=f"{src_idx}_{dst_idx}",
            primary_indicators=primary_inds,
            supporting_indicators=[
                "spectral_change",
                "spatial_consistency",
                "temporal_consistency",
            ],
            evidence_requirements=[
                "spectral",
                "spatial",
                "temporal",
                "data_quality",
            ],
            direction="both",
            analysis=[
                "search_imagery",
                f"calculate_temporal_{src_idx.lower()}",
                f"calculate_temporal_{dst_idx.lower()}",
                "detect_change",
            ],
            outputs=[
                "map",
                "statistics",
                "explanation",
                "confidence",
            ],
        )

    # --------------------------------------------------------
    # 2. GENERAL CHANGE DETECTION ("What changed?")
    # --------------------------------------------------------
    # Look for questions/requests asking what changed without specifying a single class
    is_general_change = (
        re.search(r"\bwhat\s+(?:has\s+)?changed\b", query) is not None
        or re.search(r"\bdetect\s+changes?\b", query) is not None
        or re.search(r"\bshow\s+changes?\b", query) is not None
        or re.search(r"\ball\s+changes?\b", query) is not None
        or re.search(r"\boverall\s+change\b", query) is not None
        or re.search(r"\bgeneral\s+change\b", query) is not None
        or ("land" in query and "change" in query and not any(k in query for k in ["vegetation", "water", "urban", "forest", "crop", "lake", "city"]))
    )

    # If general change query is detected:
    if is_general_change:
        return QueryPlan(
            task="general_change_detection",
            intent="general_change_detection",
            target=None,
            targets=["urban", "vegetation", "water"],
            time_start=time_start,
            time_end=time_end,
            aoi=aoi,
            modalities=["multispectral"],
            metric=None,
            primary_indicators=["NDBI", "NDVI", "NDWI"],
            supporting_indicators=[
                "spectral_change",
                "spatial_consistency",
                "temporal_consistency",
            ],
            evidence_requirements=[
                "spectral",
                "spatial",
                "temporal",
                "data_quality",
            ],
            direction="both",
            analysis=[
                "search_imagery",
                "calculate_temporal_ndbi",
                "calculate_temporal_ndvi",
                "calculate_temporal_ndwi",
                "detect_change",
            ],
            outputs=[
                "map",
                "statistics",
                "explanation",
                "confidence",
            ],
        )

    # --------------------------------------------------------
    # 3. SINGLE INDEX COMPUTATION (Non-change query)
    # --------------------------------------------------------
    if not is_change_query and ("single" in query or "calculate" in query or year_count <= 1):
        if "ndvi" in query or "vegetation" in query or "forest" in query:
            return QueryPlan(
                task="vegetation_index",
                intent="single_index",
                target="vegetation",
                targets=["vegetation"],
                metric="ndvi",
                time_start=time_start,
                time_end=time_end,
                aoi=aoi,
                modalities=["optical"],
                primary_indicators=["NDVI"],
                supporting_indicators=[],
                evidence_requirements=["spectral", "data_quality"],
                direction="unknown",
                analysis=["calculate_ndvi"],
                outputs=["map", "statistics", "explanation", "confidence"],
            )
        elif "ndwi" in query or "water" in query:
            return QueryPlan(
                task="water_index",
                intent="single_index",
                target="water",
                targets=["water"],
                metric="ndwi",
                time_start=time_start,
                time_end=time_end,
                aoi=aoi,
                modalities=["optical"],
                primary_indicators=["NDWI"],
                supporting_indicators=[],
                evidence_requirements=["spectral", "data_quality"],
                direction="unknown",
                analysis=["calculate_ndwi"],
                outputs=["map", "statistics", "explanation", "confidence"],
            )
        elif "ndbi" in query or "urban" in query:
            return QueryPlan(
                task="urban_index",
                intent="single_index",
                target="urban",
                targets=["urban"],
                metric="ndbi",
                time_start=time_start,
                time_end=time_end,
                aoi=aoi,
                modalities=["optical"],
                primary_indicators=["NDBI"],
                supporting_indicators=[],
                evidence_requirements=["spectral", "data_quality"],
                direction="unknown",
                analysis=["calculate_ndbi"],
                outputs=["map", "statistics", "explanation", "confidence"],
            )

    # --------------------------------------------------------
    # 4. CORE TARGET: URBAN / BUILT-UP
    # --------------------------------------------------------
    has_urban = any(
        w in query
        for w in [
            "urban",
            "city",
            "building",
            "built-up",
            "built up",
            "construction",
            "ndbi",
            "expansion",
            "settlement",
            "infrastructure",
        ]
    )

    # --------------------------------------------------------
    # 5. CORE TARGET: VEGETATION
    # --------------------------------------------------------
    has_vegetation = any(
        w in query
        for w in [
            "vegetation",
            "ndvi",
            "forest",
            "crop",
            "crops",
            "tree",
            "trees",
            "greenery",
            "plant",
            "plants",
            "greenness",
            "deforestation",
            "canopy",
        ]
    )

    # --------------------------------------------------------
    # 6. CORE TARGET: WATER
    # --------------------------------------------------------
    has_water = any(
        w in query
        for w in [
            "water",
            "flood",
            "lake",
            "river",
            "ndwi",
            "reservoir",
            "wetland",
            "shrinkage",
            "water body",
            "waterbody",
        ]
    )

    # Check direction indicators
    is_decrease = any(d in query for d in ["decrease", "loss", "decline", "reduced", "shrink", "shrinkage", "degradation", "dry", "drop", "deforestation"])
    is_increase = any(d in query for d in ["increase", "growth", "expansion", "expanded", "expand", "gain", "bloom", "rise", "grow"])

    if has_urban and not has_vegetation and not has_water:
        return QueryPlan(
            task="urban_change",
            intent="change_detection",
            target="urban",
            targets=["urban"],
            metric="ndbi",
            direction="increase" if is_increase else ("decrease" if is_decrease else "unknown"),
            modalities=["multispectral"],
            time_start=time_start,
            time_end=time_end,
            aoi=aoi,
            primary_indicators=["NDBI"],
            supporting_indicators=[
                "NDVI",
                "spectral_change",
                "spatial_consistency",
                "temporal_consistency",
            ],
            evidence_requirements=[
                "spectral",
                "spatial",
                "temporal",
                "data_quality",
            ],
            analysis=[
                "search_imagery",
                "calculate_temporal_ndbi",
                "detect_change",
            ],
            outputs=[
                "map",
                "statistics",
                "explanation",
                "confidence",
            ],
        )

    if has_vegetation and not has_urban and not has_water:
        return QueryPlan(
            task="change_detection",
            intent="change_detection",
            target="vegetation",
            targets=["vegetation"],
            metric="ndvi",
            direction="decrease" if is_decrease else ("increase" if is_increase else "unknown"),
            modalities=["multispectral"],
            time_start=time_start,
            time_end=time_end,
            aoi=aoi,
            primary_indicators=["NDVI"],
            supporting_indicators=[
                "NDBI",
                "spectral_change",
                "spatial_consistency",
                "temporal_consistency",
            ],
            evidence_requirements=[
                "spectral",
                "spatial",
                "temporal",
                "data_quality",
            ],
            analysis=[
                "search_imagery",
                "calculate_temporal_ndvi",
                "detect_change",
            ],
            outputs=[
                "map",
                "statistics",
                "explanation",
                "confidence",
            ],
        )

    if has_water and not has_vegetation and not has_urban:
        return QueryPlan(
            task="water_change",
            intent="change_detection",
            target="water",
            targets=["water"],
            metric="ndwi",
            direction="decrease" if is_decrease else ("increase" if is_increase else "unknown"),
            modalities=["multispectral"],
            time_start=time_start,
            time_end=time_end,
            aoi=aoi,
            primary_indicators=["NDWI"],
            supporting_indicators=[
                "NDVI",
                "spectral_change",
                "spatial_consistency",
                "temporal_consistency",
            ],
            evidence_requirements=[
                "spectral",
                "spatial",
                "temporal",
                "data_quality",
            ],
            analysis=[
                "search_imagery",
                "calculate_temporal_ndwi",
                "detect_change",
            ],
            outputs=[
                "map",
                "statistics",
                "explanation",
                "confidence",
            ],
        )

    # If multiple targets matched or generic change phrasing
    if (has_urban + has_vegetation + has_water) > 1:
        matched_targets = []
        if has_urban:
            matched_targets.append("urban")
        if has_vegetation:
            matched_targets.append("vegetation")
        if has_water:
            matched_targets.append("water")

        primary_inds = [CATEGORY_PRIMARY_INDEX[t] for t in matched_targets if t in CATEGORY_PRIMARY_INDEX]

        return QueryPlan(
            task="general_change_detection",
            intent="general_change_detection",
            target=matched_targets[0] if len(matched_targets) == 1 else None,
            targets=matched_targets,
            time_start=time_start,
            time_end=time_end,
            aoi=aoi,
            modalities=["multispectral"],
            metric=primary_inds[0] if len(primary_inds) == 1 else None,
            primary_indicators=primary_inds,
            supporting_indicators=[
                "spectral_change",
                "spatial_consistency",
                "temporal_consistency",
            ],
            evidence_requirements=[
                "spectral",
                "spatial",
                "temporal",
                "data_quality",
            ],
            direction="both",
            analysis=[
                "search_imagery",
                *[f"calculate_temporal_{ind.lower()}" for ind in primary_inds],
                "detect_change",
            ],
            outputs=[
                "map",
                "statistics",
                "explanation",
                "confidence",
            ],
        )

    # --------------------------------------------------------
    # 7. IMAGE COMPARISON
    # --------------------------------------------------------
    if "compare" in query or "comparison" in query:
        return QueryPlan(
            task="image_comparison",
            intent="image_comparison",
            target=None,
            targets=[],
            time_start=time_start,
            time_end=time_end,
            aoi=aoi,
            modalities=["optical"],
            primary_indicators=["spectral_change"],
            supporting_indicators=["spatial_consistency"],
            evidence_requirements=["spectral", "spatial", "data_quality"],
            analysis=[
                "search_imagery",
                "compare_images",
                "detect_change",
            ],
            outputs=[
                "comparison",
                "change_map",
                "explanation",
                "confidence",
            ],
        )

    # --------------------------------------------------------
    # 8. FALLBACK / IMAGE SEARCH
    # --------------------------------------------------------
    return QueryPlan(
        task="image_search",
        intent="image_search",
        target=None,
        targets=[],
        time_start=time_start,
        time_end=time_end,
        aoi=aoi,
        modalities=["optical"],
        primary_indicators=[],
        supporting_indicators=[],
        evidence_requirements=["data_quality"],
        analysis=["search_imagery"],
        outputs=["explanation", "confidence"],
    )


# ============================================================
# TEMPORAL INTENT DETECTOR & PUBLIC PARSER
# ============================================================

def detect_temporal_mode(query: str, time_start: Optional[str] = None, time_end: Optional[str] = None) -> str:
    """
    Detect user's temporal intent from natural-language query:
    - 'gradual' or 'sudden' or 'rate' -> 'trend_analysis'
    - 'recover', 'recovery', 'bounce back', 'reversal' -> 'persistence_reversal'
    - 'accelerat', 'decelerat', 'speeding up', 'slowing down' -> 'acceleration'
    - 'how has * changed' or 'across 2021 to 2025' -> 'multi_temporal'
    - Default / simple 'compare 2021 and 2025' -> 'bi_temporal'
    """
    q = query.lower()

    if any(k in q for k in ["gradual", "sudden", "rate of change", "rate of growth", "trend"]):
        return "trend_analysis"
    if any(k in q for k in ["recover", "recovery", "reversal", "bounce back", "reverse"]):
        return "persistence_reversal"
    if any(k in q for k in ["accelerat", "decelerat", "speeding up", "slowing down"]):
        return "acceleration"
    if any(k in q for k in ["continuously", "continuous", "persistently", "persistent"]):
        return "trend_analysis"
    if re.search(r"\bhow\s+(?:has|did|have)\b.+?\bchange(?:d|s)?\b", q):
        return "multi_temporal"
    if (("from 20" in q or "across 20" in q) and "to 20" in q and "compare" not in q):
        return "multi_temporal"

    return "bi_temporal"


def parse_query(request: QueryRequest) -> QueryPlan:
    """
    Public entry point: parse natural-language query into a QueryPlan
    and attach deterministic temporal_mode metadata.
    """
    plan = _parse_query_impl(request)
    t_mode = detect_temporal_mode(request.query, plan.time_start, plan.time_end)
    plan.temporal_mode = t_mode
    return plan
