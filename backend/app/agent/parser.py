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
    "optical_sar_analysis",
    "single_image_vqa",
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

def extract_aoi(query_str: str, request_aoi: Optional[Any] = None) -> Tuple[Optional[Any], str]:
    """
    Extract GeoJSON Polygon from query string [minLon, minLat, maxLon, maxLat]
    or retain request_aoi if already provided.
    """
    cleaned_query = query_str
    parsed_aoi = None

    if isinstance(request_aoi, (list, tuple)) and len(request_aoi) == 4 and all(isinstance(x, (int, float)) for x in request_aoi):
        v1, v2, v3, v4 = [float(x) for x in request_aoi]
        minLon = min(v1, v3)
        maxLon = max(v1, v3)
        minLat = min(v2, v4)
        maxLat = max(v2, v4)
        request_aoi = {
            "type": "Polygon",
            "coordinates": [[
                [minLon, minLat],
                [maxLon, minLat],
                [maxLon, maxLat],
                [minLon, maxLat],
                [minLon, minLat]
            ]]
        }

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


def detect_optical_sar_intent(query: str) -> bool:
    """
    Detect whether a query represents a joint Optical + SAR multimodal analysis request.
    Requires clear reference to BOTH optical and radar/SAR modalities.
    Does NOT trigger for single-modality queries or explicit index calculations (Step 6D).
    """
    q = query.lower()

    # If the user is specifically requesting a spectral index calculation (NDVI/NDWI/NDBI),
    # the existing scientific index calculation has priority (Step 6D).
    if any(k in q for k in ["calculate ndvi", "calculate ndwi", "calculate ndbi", "compute ndvi", "compute ndwi", "compute ndbi"]):
        return False

    has_optical = any(k in q for k in ["optical", "sentinel-2", "sentinel 2", "s2", "multispectral"])
    has_sar = any(k in q for k in ["sar", "radar", "sentinel-1", "sentinel 1", "s1", "backscatter"])

    if not (has_optical and has_sar):
        return False

    # Check for joint / multimodal intent phrasing
    joint_phrases = [
        "optical and sar",
        "optical + sar",
        "optical & sar",
        "sar and optical",
        "sar + optical",
        "sar & optical",
        "optical imagery and radar",
        "radar and optical",
        "optical and radar",
        "multimodal optical sar",
        "multimodal",
        "use both",
        "using both",
        "together",
        "combine",
        "combined",
        "joint",
        "jointly",
    ]

    has_joint_cue = any(p in q for p in joint_phrases)
    has_analysis_cue = any(
        a in q
        for a in [
            "identify",
            "analyze",
            "analyse",
            "describe",
            "features",
            "regions",
            "patterns",
            "land-cover",
            "land cover",
            "urban",
            "built-up",
            "built up",
            "water",
            "wetland",
        ]
    )

    return has_joint_cue or has_analysis_cue


def detect_single_image_vqa_intent(query: str, request: Optional[QueryRequest] = None) -> bool:
    """
    Detect whether a query represents a single-image Visual Question Answering (VQA) request.
    Strict priority rules:
    - Optical-SAR multimodal requests MUST win over generic VQA.
    - Temporal change requests MUST win over generic VQA.
    - Explicit index requests (NDVI/NDWI/NDBI) MUST win over generic VQA.
    - Does not classify every query containing 'image' as VQA.
    """
    q = query.lower()

    # Rule 1: Optical-SAR queries must NOT be classified as generic VQA
    if detect_optical_sar_intent(query):
        return False

    # Rule 2: Land cover transition must NOT be classified as VQA
    if detect_transition(query):
        return False

    # Rule 3: Temporal change detection must NOT be classified as VQA
    if (
        re.search(r"\bwhat\s+(?:has\s+)?changed\b", q) is not None
        or re.search(r"\bdetect\s+changes?\b", q) is not None
        or re.search(r"\bshow\s+changes?\b", q) is not None
        or re.search(r"\ball\s+changes?\b", q) is not None
        or re.search(r"\boverall\s+change\b", q) is not None
        or re.search(r"\bgeneral\s+change\b", q) is not None
        or "before and after" in q
        or "over time" in q
        or "across years" in q
        or ("change" in q and any(k in q for k in ["between", "from", "since", "to 20", "growth", "loss", "decline", "increase", "decrease"]))
    ):
        return False

    # Rule 4: Explicit spectral index calculation must NOT be classified as VQA
    if (
        any(k in q for k in [
            "calculate ndvi", "calculate ndwi", "calculate ndbi",
            "compute ndvi", "compute ndwi", "compute ndbi",
            "measure ndvi", "measure ndwi", "measure ndbi",
            "vegetation index", "water index", "urban index",
            "what is the ndvi", "what is the ndwi", "what is the ndbi",
            "what's the ndvi", "what's the ndwi", "what's the ndbi",
        ])
        or re.search(r"\b(?:ndvi|ndwi|ndbi)\b", q) is not None
        or q.strip() in ["ndvi", "ndwi", "ndbi"]
    ):
        return False

    # Rule 5: Compare two images must remain image_comparison
    if (
        ("compare these two" in q or "compare two" in q or "compare the two" in q or "between the two" in q)
        and not any(k in q for k in ["what is", "is there", "are there"])
    ):
        return False

    # Positive VQA patterns:
    # 1. "What is visible ...", "What structures are visible ...", "What objects are visible ..."
    if re.search(r"\bwhat\s+(?:structures?|features?|objects?|elements?|details?|regions?|patterns?|type\s+of\s+land\s*cover|land\s*cover)?\s*(?:is|are)?\s*(?:visible|present|seen|shown|dominates?|dominant)\b", q):
        return True

    # 2. "What is in this image", "What does this image show", "What does the SAR image show"
    if re.search(r"\bwhat\s+(?:is\s+in|does\s+(?:this|the)?\s*(?:sar\s+|optical\s+)?image\s+show|do\s+you\s+see)\b", q):
        return True

    # 3. "Is there water here?", "Is there a water body in this image?", "Is there [a] <feature> ..."
    if re.search(r"\bis\s+there\s+(?:a\s+|an\s+)?(?:\w+\s+)?(?:water(?:\s*body)?|lake|river|reservoir|wetland|flood|forest|vegetation|trees?|greenery|buildings?|structures?|construction|urban|settlement|roads?|clouds?|snow|ice)\b", q):
        return True

    # 4. "Are there buildings?", "Are there structures?", "Are there [any] <features> ..."
    if re.search(r"\bare\s+there\s+(?:any\s+)?(?:\w+\s+)?(?:buildings?|structures?|bridges?|houses?|settlements?|trees?|roads?|features?|objects?|water\s*bodies?|water|crops?|fields?)\b", q):
        return True

    # 5. "What type of land cover dominates?", "What land cover is dominant?", "Which land cover dominates"
    if re.search(r"\b(?:what|which)\s+(?:type\s+of\s+)?land\s*cover\s+(?:dominates?|is\s+dominant)\b", q) or re.search(r"\bland\s*cover\s+dominates\b", q):
        return True

    # 6. "Describe the objects in this image", "Describe the features...", "Describe the land cover..."
    if re.search(r"\bdescribe\s+(?:the\s+)?(?:objects?|features?|structures?|land\s*cover|main\s+features?|visible\s+features?)\b", q):
        return True

    # 7. "Can you identify the main features?", "Can you identify...", "Can you describe..."
    if re.search(r"\b(?:can|could)\s+you\s+(?:identify|describe|find|detect|see)\s+(?:the\s+)?(?:main\s+|dominant\s+|visible\s+)?(?:features?|objects?|structures?|land\s*cover|characteristics?)\b", q):
        return True

    # 8. Single image reference with interrogative intent
    if any(k in q for k in [
        "this image", "the image", "this sar image", "the sar image",
        "this radar image", "the radar image", "this optical image", "the optical image",
        "the satellite image", "this satellite image",
    ]):
        if any(q.startswith(w) for w in ["what", "is ", "are ", "can ", "could ", "describe ", "identify "]) or q.endswith("?"):
            return True

    # 9. Request with explicit single image identifier and a question
    if request is not None:
        has_single_id = (
            (len(request.image_ids) == 1)
            or (request.optical_image_id and not request.sar_image_id)
            or (request.sar_image_id and not request.optical_image_id)
        )
        if has_single_id and (q.endswith("?") or any(q.startswith(w) for w in ["what", "is ", "are ", "can ", "describe ", "identify "])):
            return True

    return False


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
    req_t_start = getattr(request, "time_start", None)
    req_t_end = getattr(request, "time_end", None)
    if req_t_start or req_t_end:
        time_start = req_t_start or time_start
        time_end = req_t_end or time_end
        has_temporal_input = True
    else:
        has_temporal_input = (year_count > 0)

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
            "transition",
            "conversion",
            "evolution",
            "dynamic",
            "trend",
            "before and after",
            "over time",
            "temporal",
            "across years",
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
    # 1B. OPTICAL-SAR MULTIMODAL ANALYSIS
    # --------------------------------------------------------
    if detect_optical_sar_intent(query):
        target_val = (
            "urban"
            if any(w in query for w in ["urban", "built-up", "built up", "city", "building"])
            else (
                "water"
                if any(w in query for w in ["water", "river", "lake", "flood", "wetland"])
                else ("vegetation" if any(w in query for w in ["vegetation", "forest", "crop"]) else None)
            )
        )
        targets_list = [target_val] if target_val else ["urban", "water"]

        return QueryPlan(
            task="optical_sar_analysis",
            intent="optical_sar_analysis",
            target=target_val,
            targets=targets_list,
            time_start=time_start if has_temporal_input else None,
            time_end=time_end if has_temporal_input else None,
            aoi=aoi,
            modalities=["optical", "sar"],
            primary_indicators=["optical_reflectance", "radar_backscatter"],
            supporting_indicators=["spatial_consistency"],
            evidence_requirements=["multimodal"],
            direction="unknown",
            analysis=["optical_sar_analysis"],
            outputs=["explanation", "confidence"],
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
    # 2B. SINGLE-IMAGE VISUAL QUESTION ANSWERING (VQA)
    # --------------------------------------------------------
    if detect_single_image_vqa_intent(query, request):
        is_sar = any(k in query for k in ["sar", "radar", "sentinel-1", "sentinel 1", "s1", "backscatter"]) or (
            getattr(request, "sar_image_id", None) and not getattr(request, "optical_image_id", None)
        )
        is_optical = any(k in query for k in ["optical", "sentinel-2", "sentinel 2", "s2", "multispectral", "rgb", "true color"]) or (
            getattr(request, "optical_image_id", None) and not getattr(request, "sar_image_id", None)
        )

        if is_sar and not is_optical:
            vqa_modalities = ["sar"]
        elif is_optical and not is_sar:
            vqa_modalities = ["optical"]
        else:
            vqa_modalities = ["unknown"]

        target_val = (
            "water"
            if any(w in query for w in ["water", "river", "lake", "flood", "wetland"])
            else (
                "urban"
                if any(w in query for w in ["urban", "built-up", "built up", "city", "building", "buildings", "structure", "structures"])
                else ("vegetation" if any(w in query for w in ["vegetation", "forest", "crop", "crops", "tree", "trees"]) else None)
            )
        )
        targets_list = [target_val] if target_val else []

        return QueryPlan(
            task="single_image_vqa",
            intent="single_image_vqa",
            target=target_val,
            targets=targets_list,
            time_start=time_start if has_temporal_input else None,
            time_end=time_end if has_temporal_input else None,
            aoi=aoi,
            modalities=vqa_modalities,
            primary_indicators=[],
            supporting_indicators=[],
            evidence_requirements=["visual"],
            direction="unknown",
            analysis=["single_image_vqa"],
            outputs=["explanation", "confidence"],
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

    has_explicit_ndvi = bool(re.search(r"\bndvi\b", query, re.IGNORECASE))
    has_explicit_ndwi = bool(re.search(r"\bndwi\b", query, re.IGNORECASE))
    has_explicit_ndbi = bool(re.search(r"\bndbi\b", query, re.IGNORECASE))

    if has_urban and not has_vegetation and not has_water:
        return QueryPlan(
            task="urban_change",
            intent="change_detection",
            target="urban",
            targets=["urban"],
            metric="ndbi",
            explicit_metric="ndbi" if has_explicit_ndbi else None,
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
            explicit_metric="ndvi" if has_explicit_ndvi else None,
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
            explicit_metric="ndwi" if has_explicit_ndwi else None,
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
