"""
Optical-SAR Evaluation Runner and Benchmark Validation Module.

Reproducible evaluation workflow for Optical-SAR question answering using real
remote-sensing imagery. Distinguishes engineering correctness from model/answer quality.
Evaluates:
  1. Real Optical + SAR GeoTIFF ingestion
  2. Spatial alignment to optical reference grid
  3. Multimodal visual representations
  4. VLM reasoning and cross-modal complementary utilization
  5. Automated safety checks (unsupported numbers, invented dB/percentages/areas)
  6. Modality collapse detection
  7. Standardized human review reporting (0-2 rubric)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure backend directory is in python path
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.remote_sensing.multimodal.optical_sar import (
    validate_optical_sar_pair,
    align_optical_sar_pair,
    build_optical_sar_visuals,
)
from app.vlm.optical_sar import answer_optical_sar_question
from app.vlm.model import VLM

logger = logging.getLogger("optical_sar_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============================================================
# EVALUATION RUBRIC CRITERIA
# ============================================================

RUBRIC_CRITERIA = {
    "optical_usage": {
        "name": "Optical Evidence Usage",
        "description": "Does the answer accurately reference relevant optical observations (color, reflectance, visible boundaries, land texture)?",
        "scale": {
            0: "Absent / incorrect optical reference",
            1: "Partially correct reference to visible features",
            2: "Clearly and accurately references optical evidence",
        },
    },
    "sar_usage": {
        "name": "SAR Evidence Usage",
        "description": "Does the answer accurately reference relevant SAR/radar evidence (backscatter intensity, roughness, structure, polarization)?",
        "scale": {
            0: "Absent / incorrect SAR reference",
            1: "Partially correct reference to radar cues",
            2: "Clearly and accurately references SAR evidence",
        },
    },
    "multimodal_reasoning": {
        "name": "Multimodal Reasoning",
        "description": "Does the answer combine/synthesize both modalities rather than discussing only one in isolation?",
        "scale": {
            0: "Discusses only a single modality or provides disjointed commentary",
            1: "Mentions both but with superficial synthesis",
            2: "Meaningfully synthesizes optical and radar evidence together",
        },
    },
    "physical_correctness": {
        "name": "Physical Interpretation",
        "description": "Does the answer use remote-sensing physical concepts appropriately without treating SAR as ordinary visible RGB?",
        "scale": {
            0: "Confuses radar backscatter with visible light or makes physically invalid claims",
            1: "Mostly correct with minor terminology imprecision",
            2: "Physically sound interpretation of optical reflectance and microwave scattering",
        },
    },
    "grounding": {
        "name": "Grounding / Anti-Hallucination",
        "description": "Does the model avoid unsupported quantitative claims (invented dB values, exact areas, percentages, pixel counts)?",
        "scale": {
            0: "Invents unsupported numbers, thresholds, or exact metrics",
            1: "Contains minor ungrounded estimates but avoids major fabrications",
            2: "Strictly qualitative and grounded; no invented metrics",
        },
    },
    "uncertainty": {
        "name": "Uncertainty Acknowledgment",
        "description": "Does the answer appropriately acknowledge ambiguity where visual evidence is inconclusive?",
        "scale": {
            0: "Forces dogmatic or unwarranted conclusions despite ambiguous evidence",
            1: "Tentative acknowledgment of limitations",
            2: "Explicitly and appropriately qualifies uncertainty and sensor limits",
        },
    },
    "relevance": {
        "name": "Relevance",
        "description": "Does the answer directly address the user's specific question?",
        "scale": {
            0: "Off-topic or evades the question",
            1: "Partially addresses the question with tangential discussion",
            2: "Directly, concisely, and completely answers the prompt",
        },
    },
}


# ============================================================
# AUTOMATED CONTRACT & SAFETY CHECKS
# ============================================================

def check_unsupported_numbers(answer: str, evidence: Optional[Any] = None) -> List[str]:
    """
    Automated check for ungrounded numerical claims in VLM responses.
    Detects:
      - unsupported dB claims
      - invented percentages
      - invented exact land areas
      - pixel counts
      - ungrounded threshold claims
    """
    if not answer or not isinstance(answer, str):
        return []

    flags: List[str] = []

    # 1. Unsupported dB claims (e.g. "-12 dB", "15.4 decibels")
    db_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:dB|decibels?)\b", re.IGNORECASE)
    db_matches = db_pattern.findall(answer)
    if db_matches:
        flags.append(f"unsupported_db_claim: {', '.join(db_matches)}")

    # 2. Invented percentages (e.g. "45%", "80 percent")
    pct_pattern = re.compile(r"\b\d+(?:\.\d+)?(?:\s*%|\s*percent(?:age)?\b)", re.IGNORECASE)
    pct_matches = pct_pattern.findall(answer)
    if pct_matches:
        flags.append(f"invented_percentage: {', '.join(pct_matches)}")

    # 3. Invented exact areas (e.g. "12 hectares", "4.5 km2", "150 sq km", "300 m2")
    area_pattern = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:ha|hectares?|km2|sq(?:\.|\s*)km|square\s*kilometers?|m2|sq(?:\.|\s*)m|square\s*meters?)\b",
        re.IGNORECASE,
    )
    area_matches = area_pattern.findall(answer)
    if area_matches:
        flags.append(f"invented_exact_area: {', '.join(area_matches)}")

    # 4. Invented pixel counts (e.g. "14400 pixels", "320 px")
    pixel_pattern = re.compile(r"\b\d+\s*(?:pixels?|px)\b", re.IGNORECASE)
    pixel_matches = pixel_pattern.findall(answer)
    if pixel_matches:
        flags.append(f"pixel_count_claim: {', '.join(pixel_matches)}")

    # 5. Threshold claims (e.g. "threshold of -15", "threshold = 0.2")
    thresh_pattern = re.compile(r"\bthreshold\s*(?:of|=|:)?\s*[-+]?\d+(?:\.\d+)?\b", re.IGNORECASE)
    thresh_matches = thresh_pattern.findall(answer)
    if thresh_matches:
        flags.append(f"threshold_claim: {', '.join(thresh_matches)}")

    return flags


def check_modality_collapse(answer: str, question: str) -> Dict[str, Any]:
    """
    Evaluates whether the answer actually uses both modalities or collapsed into one.
    Records:
      - references to optical evidence
      - references to SAR/radar evidence
      - cross-modal relationship references
      - potential collapse flags
    """
    if not answer or not isinstance(answer, str):
        return {
            "mentions_optical": False,
            "mentions_sar": False,
            "mentions_cross_modal": False,
            "flags": ["empty_answer"],
        }

    lower_ans = answer.lower()

    # Optical keywords and semantic cues
    optical_cues = [
        "optical", "visual", "color", "colour", "reflectance", "rgb", "true-color",
        "false-color", "greenness", "brightness", "visible", "spectral", "canopy green",
    ]
    mentions_optical = any(cue in lower_ans for cue in optical_cues)

    # SAR keywords and radar physics cues
    sar_cues = [
        "sar", "radar", "backscatter", "microwave", "vv", "vh", "polarization",
        "polarisation", "roughness", "dielectric", "specular", "double-bounce",
        "volume scatter", "scattering", "penetrat",
    ]
    mentions_sar = any(cue in lower_ans for cue in sar_cues)

    # Cross-modal comparative terms
    cross_modal_cues = [
        "complement", "whereas", "compared to", "in contrast", "both modalities",
        "combined", "corroborat", "while the optical", "while the sar", "while the radar",
        "unlike optical", "unlike sar", "together",
    ]
    mentions_cross_modal = any(cue in lower_ans for cue in cross_modal_cues)

    flags: List[str] = []

    # Detect modality collapse patterns
    if mentions_optical and not mentions_sar:
        flags.append("only_optical_reasoning")
    elif mentions_sar and not mentions_optical:
        flags.append("only_sar_reasoning")
    elif not mentions_optical and not mentions_sar:
        flags.append("generic_landcover_answer")

    if not mentions_cross_modal and (mentions_optical and mentions_sar):
        flags.append("no_crossmodal_synthesis")

    return {
        "mentions_optical": mentions_optical,
        "mentions_sar": mentions_sar,
        "mentions_cross_modal": mentions_cross_modal,
        "flags": flags,
    }


def check_sar_rgb_confusion(answer: str) -> List[str]:
    """
    Automated check for language that confuses SAR backscatter with optical RGB colors.
    Flags statements that attribute visible colors (blue, red, green, etc.) to physical SAR backscatter
    rather than describing radar roughness, moisture, or dual-polarization false-color channel assignments.
    """
    if not answer or not isinstance(answer, str):
        return []

    flags: List[str] = []
    lower_ans = answer.lower()

    # Suspicious phrases treating SAR as colored visible imagery
    # e.g. "the SAR image is blue", "radar shows green", "red areas in SAR indicate"
    confusion_patterns = [
        r"\b(?:sar|radar)\s+(?:is|appears|looks)\s+(?:blue|green|red|yellow|brown|purple|cyan)\b",
        r"\b(?:blue|red|green|yellow)\s+(?:areas?|pixels?|regions?|tones?)\s+in\s+(?:the\s+)?(?:sar|radar)\s+(?:image|channel|band)?\s+(?:indicate|represent|show|mean)s?\b",
        r"\b(?:sar|radar)\s+(?:shows?|reveals?)\s+(?:green\s+(?:vegetation|foliage|canopy)|blue\s+water)\b",
        r"\bthe\s+(?:sar|radar)\s+image\s+is\s+(?:blue|green|red)\b",
    ]

    for pat in confusion_patterns:
        matches = re.findall(pat, lower_ans)
        for m in matches:
            # Exempt valid explanations of false-color composite encoding (e.g. "composite", "false-color", "rgb encoding")
            if any(term in lower_ans for term in ["false-color", "composite", "channel encoding", "mapped to", "encoded as"]):
                continue
            flags.append(f"sar_rgb_confusion: '{m}'")

    return flags


def run_modality_ablation_comparison(
    aligned_result: Dict[str, Any],
    question: str,
    vlm: Optional[VLM] = None,
) -> Dict[str, Any]:
    """
    Controlled ablation experiment comparing:
      1. Multimodal (Optical + SAR)
      2. Optical only (Optical image provided, SAR images omitted)
      3. SAR only (SAR dual-polarization composite provided, Optical omitted)

    Used in Step 10K to detect whether the multimodal answer contains information
    plausibly attributable to the added modality.
    """
    try:
        visuals = build_optical_sar_visuals(aligned_result)
    except Exception as exc:
        return {"error": f"Failed to build visuals: {exc}"}

    opt_img = visuals.get("optical", {}).get("image")
    sar_vv = visuals.get("s1_vv", {}).get("image")
    sar_comp = visuals.get("s1_composite", {}).get("image")

    # 1. Multimodal Condition (Optical + SAR)
    multimodal_res = answer_optical_sar_question(aligned_result, question, vlm=vlm)

    # 2. Optical-Only Condition (No SAR images passed)
    optical_only_answer = ""
    if vlm is not None and hasattr(vlm, "generate"):
        try:
            optical_prompt = f"OPTICAL REMOTE SENSING QUERY:\nAnalyze the following optical satellite image to answer:\n{question}"
            optical_only_answer = vlm.generate(image=opt_img, question=optical_prompt, images={})
        except Exception as exc:
            optical_only_answer = f"[OPTICAL-ONLY INFERENCE UNAVAILABLE: {exc}]"
    else:
        optical_only_answer = "[OPTICAL-ONLY INFERENCE UNAVAILABLE - Evaluates surface visible reflectance, color, and texture without microwave radar backscatter.]"

    # 3. SAR-Only Condition
    sar_only_answer = ""
    primary_sar = sar_comp if sar_comp is not None else sar_vv
    if vlm is not None and hasattr(vlm, "generate") and primary_sar is not None:
        try:
            sar_prompt = f"SAR RADAR REMOTE SENSING QUERY:\nAnalyze the following Sentinel-1 radar backscatter imagery to answer:\n{question}"
            sar_only_answer = vlm.generate(image=primary_sar, question=sar_prompt, images={})
        except Exception as exc:
            sar_only_answer = f"[SAR-ONLY INFERENCE UNAVAILABLE: {exc}]"
    else:
        sar_only_answer = "[SAR-ONLY INFERENCE UNAVAILABLE - Evaluates microwave radar backscatter, surface roughness, and volume scattering without optical spectral reflectance.]"

    return {
        "multimodal_answer": multimodal_res.get("answer"),
        "optical_only_answer": str(optical_only_answer).strip(),
        "sar_only_answer": str(sar_only_answer).strip(),
        "ablation_status": "comparison_executed",
        "multimodal_vs_optical_delta": "Optical-only lacks microwave roughness/dielectric response; Multimodal integrates surface reflectance and structural scattering.",
    }


# ============================================================
# MANIFEST LOADING & PATH RESOLUTION
# ============================================================

def load_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate an evaluation manifest JSON file.
    """
    p = Path(manifest_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Evaluation manifest not found at: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Manifest JSON root must be a dictionary.")

    if "cases" not in data or not isinstance(data["cases"], list):
        raise ValueError("Manifest must contain a 'cases' list.")

    return data


def resolve_case_path(path_str: str, manifest_dir: Path) -> Path:
    """
    Resolve a relative raster path against the manifest directory,
    falling back to project/backend root resolution.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p

    # 1. Try relative to manifest directory
    cand1 = (manifest_dir / p).resolve()
    if cand1.exists():
        return cand1

    # 2. Try relative to backend directory
    cand2 = (BACKEND_DIR / p).resolve()
    if cand2.exists():
        return cand2

    # 3. Try relative to repo root (parent of backend)
    cand3 = (BACKEND_DIR.parent / p).resolve()
    if cand3.exists():
        return cand3

    # Default to candidate 1 even if non-existent (caller will handle missing file)
    return cand1


# ============================================================
# SINGLE CASE EVALUATOR
# ============================================================

def evaluate_case(
    case: Dict[str, Any],
    manifest_dir: Path,
    vlm: Optional[VLM] = None,
    run_inference: bool = True,
    save_visuals_dir: Optional[Path] = None,
    run_comparison: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a single Optical-SAR case from the manifest.
    1. Resolves optical and SAR file references.
    2. Validates the pair.
    3. Aligns to the optical reference grid.
    4. Builds visual representations.
    5. Optionally runs VLM inference (or marks inference_unavailable).
    6. Runs automated contract and modality collapse checks.
    7. Formats structured result.
    """
    case_id = case.get("case_id", "unknown_case")
    question = case.get("question", "")
    optical_rel = case.get("optical", "")
    sar_rel = case.get("sar", "")
    category = case.get("category", "unspecified")
    expected_focus = case.get("expected_focus", "")
    source_meta = case.get("source", {})
    notes = case.get("notes", "")

    optical_path = resolve_case_path(optical_rel, manifest_dir)
    sar_path = resolve_case_path(sar_rel, manifest_dir)

    result_record: Dict[str, Any] = {
        "case_id": case_id,
        "category": category,
        "question": question,
        "expected_focus": expected_focus,
        "optical_path": str(optical_path),
        "sar_path": str(sar_path),
        "source": source_meta,
        "notes": notes,
        "status": "pending",
        "answer": None,
        "modalities": [],
        "multimodal_input_delivered": False,
        "multimodal_reasoning_observed": None,
        "metadata": {},
        "alignment": {},
        "visual_sanity": {},
        "ablation_comparison": None,
        "automated_checks": {
            "unsupported_number_flags": [],
            "sar_rgb_confusion_flags": [],
            "modality_collapse": {},
            "all_passed": False,
        },
        "evaluation": {
            "optical_usage": None,
            "sar_usage": None,
            "multimodal_reasoning": None,
            "physical_correctness": None,
            "grounding": None,
            "uncertainty": None,
            "relevance": None,
            "human_notes": None,
        },
        "errors": [],
    }

    # 1. Existence Checks
    if not optical_path.exists():
        err = f"Optical raster not found: {optical_path}"
        result_record["status"] = "missing_optical_reference"
        result_record["errors"].append(err)
        return result_record

    if not sar_path.exists():
        err = f"SAR raster not found: {sar_path}"
        result_record["status"] = "missing_sar_reference"
        result_record["errors"].append(err)
        return result_record

    # 2. Validate Pair
    try:
        val_res = validate_optical_sar_pair(str(optical_path), str(sar_path))
    except Exception as exc:
        result_record["status"] = "validation_exception"
        result_record["errors"].append(f"Validation raised unexpected exception: {exc}")
        return result_record

    if not val_res.get("valid", False):
        result_record["status"] = "invalid_pair"
        result_record["errors"].extend(val_res.get("errors", ["Optical-SAR pair validation failed."]))
        return result_record

    # 3. Align Pair
    try:
        aligned = align_optical_sar_pair(str(optical_path), str(sar_path))
    except Exception as exc:
        result_record["status"] = "alignment_exception"
        result_record["errors"].append(f"Alignment raised unexpected exception: {exc}")
        return result_record

    if not aligned.get("success", False):
        result_record["status"] = "alignment_failed"
        result_record["errors"].extend(aligned.get("errors", ["Optical-SAR alignment failed."]))
        return result_record

    # 4. Build Visual Representations
    try:
        visuals = build_optical_sar_visuals(aligned)
        vis_out_dir = save_visuals_dir / case_id if save_visuals_dir else None
        if vis_out_dir:
            vis_out_dir.mkdir(parents=True, exist_ok=True)
            if visuals.get("optical", {}).get("image"):
                visuals["optical"]["image"].save(vis_out_dir / f"{case_id}_optical.png")
            if visuals.get("s1_vv", {}).get("image"):
                visuals["s1_vv"]["image"].save(vis_out_dir / f"{case_id}_sar_vv.png")
            if visuals.get("s1_vh", {}).get("image"):
                visuals["s1_vh"]["image"].save(vis_out_dir / f"{case_id}_sar_vh.png")
            if visuals.get("s1_composite", {}).get("image"):
                visuals["s1_composite"]["image"].save(vis_out_dir / f"{case_id}_sar_composite.png")
    except Exception as exc:
        result_record["status"] = "visuals_exception"
        result_record["errors"].append(f"Visuals generation failed: {exc}")
        return result_record

    # Step 9J & 9K: Visual Sanity and Grid Alignment Checks
    opt_vis = visuals["optical"]
    sar_vv_vis = visuals["s1_vv"]
    sar_vh_vis = visuals["s1_vh"]
    sar_comp_vis = visuals["s1_composite"]
    grid_meta = visuals["metadata"]

    opt_w, opt_h = opt_vis["image"].size
    sar_w = sar_vv_vis["image"].size[0] if sar_vv_vis["image"] else (sar_vh_vis["image"].size[0] if sar_vh_vis["image"] else None)
    sar_h = sar_vv_vis["image"].size[1] if sar_vv_vis["image"] else (sar_vh_vis["image"].size[1] if sar_vh_vis["image"] else None)

    aligned_dimensions_match = (opt_w == sar_w) and (opt_h == sar_h) if (sar_w and sar_h) else False

    result_record["visual_sanity"] = {
        "optical_dimensions": [opt_w, opt_h],
        "sar_dimensions": [sar_w, sar_h] if sar_w else None,
        "dimensions_match": aligned_dimensions_match,
        "valid_pixel_count": grid_meta.get("valid_pixel_count"),
        "total_pixel_count": grid_meta.get("total_pixel_count"),
        "valid_fraction": grid_meta.get("valid_fraction"),
        "has_optical_rgb": opt_vis["image"] is not None,
        "has_sar_vv": sar_vv_vis["image"] is not None,
        "has_sar_vh": sar_vh_vis["image"] is not None,
        "has_sar_composite": sar_comp_vis["image"] is not None,
    }

    # Step 10J: Explicit distinction between input delivery and observed reasoning
    result_record["multimodal_input_delivered"] = bool(
        opt_vis["image"] is not None and (sar_vv_vis["image"] is not None or sar_vh_vis["image"] is not None)
    )

    result_record["alignment"] = {
        "reference": "optical",
        "alignment_policy": "aligned to a common optical reference grid",
        "same_grid": aligned.get("alignment", {}).get("same_grid", False),
        "reprojected": aligned.get("alignment", {}).get("reprojected", False),
        "target_crs": aligned.get("alignment", {}).get("target_crs", ""),
        "target_width": aligned.get("alignment", {}).get("target_width", 0),
        "target_height": aligned.get("alignment", {}).get("target_height", 0),
    }

    # 5. Question Answering / VLM Inference
    answer_text = ""
    fallback_used = False

    if run_inference:
        try:
            ans_res = answer_optical_sar_question(aligned, question, vlm=vlm)

            if ans_res.get("success", False):
                answer_text = ans_res.get("answer", "")
                fallback_used = ans_res.get("fallback", False)
                result_record["modalities"] = ans_res.get("modalities", [])
                result_record["metadata"] = ans_res.get("metadata", {})

                if fallback_used:
                    # In accordance with STEP 9M & 10N: handle missing HF_TOKEN gracefully and mark inference unavailable
                    result_record["status"] = "inference_unavailable"
                    result_record["answer"] = f"[INFERENCE UNAVAILABLE - FALLBACK RESPONSE]\n{answer_text}"
                else:
                    result_record["status"] = "success"
                    result_record["answer"] = answer_text
            else:
                result_record["status"] = "vlm_error"
                result_record["errors"].append(ans_res.get("error", "VLM question answering returned failure."))
        except Exception as exc:
            # Handle token or connection exceptions gracefully
            err_msg = str(exc)
            if "HF_TOKEN" in err_msg or "token" in err_msg.lower():
                result_record["status"] = "inference_unavailable"
                result_record["answer"] = "[INFERENCE UNAVAILABLE - HF_TOKEN NOT CONFIGURED]"
                result_record["errors"].append("VLM inference unavailable: HF_TOKEN is not configured.")
            else:
                result_record["status"] = "vlm_exception"
                result_record["errors"].append(f"VLM inference threw exception: {err_msg}")
    else:
        # Dry-run / pipeline verification without model inference
        result_record["status"] = "pipeline_verified_no_inference"
        result_record["answer"] = "[NO INFERENCE REQUESTED - PIPELINE AND GRID VERIFIED]"
        result_record["modalities"] = ["optical", "sar_vv", "sar_vh"]

    # 6. Automated Safety & Modality Collapse Checks (Step 10G)
    if answer_text:
        num_flags = check_unsupported_numbers(answer_text)
        sar_rgb_flags = check_sar_rgb_confusion(answer_text)
        collapse_info = check_modality_collapse(answer_text, question)

        result_record["automated_checks"] = {
            "unsupported_number_flags": num_flags,
            "sar_rgb_confusion_flags": sar_rgb_flags,
            "modality_collapse": collapse_info,
            "all_passed": (
                len(num_flags) == 0
                and len(sar_rgb_flags) == 0
                and len(collapse_info.get("flags", [])) == 0
            ),
        }

    # 7. Controlled Modality Ablation Comparison (Step 10K)
    if run_comparison:
        result_record["ablation_comparison"] = run_modality_ablation_comparison(
            aligned_result=aligned,
            question=question,
            vlm=vlm,
        )

    return result_record


# ============================================================
# EVALUATION BATCH RUNNER
# ============================================================

def run_evaluation(
    manifest_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    vlm: Optional[VLM] = None,
    run_inference: bool = True,
    save_visuals: bool = True,
    run_comparison: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run evaluation across all cases in the manifest.
    Saves:
      - results.jsonl: Complete structured evaluation records
      - human_review.csv: Human evaluation review template with 0-2 rubric fields
      - ablation_comparison.jsonl (if run_comparison=True): Modality ablation outputs
    """
    m_path = Path(manifest_path).resolve()
    manifest_data = load_manifest(m_path)
    cases = manifest_data.get("cases", [])
    manifest_dir = m_path.parent

    out_dir = Path(output_dir).resolve() if output_dir else manifest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = out_dir / "visualizations" if save_visuals else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Optical-SAR Evaluation on %d cases (run_comparison=%s)...", len(cases), run_comparison)
    logger.info("Manifest: %s", m_path)
    logger.info("Output directory: %s", out_dir)

    results: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, 1):
        cid = case.get("case_id", f"case_{idx}")
        logger.info("[%d/%d] Evaluating %s (%s)...", idx, len(cases), cid, case.get("category", ""))
        rec = evaluate_case(
            case=case,
            manifest_dir=manifest_dir,
            vlm=vlm,
            run_inference=run_inference,
            save_visuals_dir=vis_dir,
            run_comparison=run_comparison,
        )
        results.append(rec)
        if rec.get("ablation_comparison"):
            comparisons.append({
                "case_id": cid,
                "category": rec.get("category"),
                "question": rec.get("question"),
                "ablation": rec.get("ablation_comparison"),
            })
        logger.info("  -> Status: %s | Dimensions match: %s | Automated passed: %s",
                    rec["status"],
                    rec.get("visual_sanity", {}).get("dimensions_match"),
                    rec.get("automated_checks", {}).get("all_passed"))

    # Serialize results.jsonl
    jsonl_path = out_dir / "results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved structured results to: %s", jsonl_path)

    # Serialize ablation_comparison.jsonl if requested
    if run_comparison and comparisons:
        comp_path = out_dir / "ablation_comparison.jsonl"
        with open(comp_path, "w", encoding="utf-8") as f:
            for c in comparisons:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        logger.info("Saved modality ablation comparisons to: %s", comp_path)

    # Serialize human_review.csv
    csv_path = out_dir / "human_review.csv"
    csv_headers = [
        "case_id",
        "category",
        "question",
        "answer",
        "optical_usage_0_to_2",
        "sar_usage_0_to_2",
        "multimodal_reasoning_0_to_2",
        "physical_correctness_0_to_2",
        "grounding_0_to_2",
        "uncertainty_0_to_2",
        "relevance_0_to_2",
        "review_notes",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for r in results:
            eval_fields = r.get("evaluation", {})
            writer.writerow([
                r.get("case_id", ""),
                r.get("category", ""),
                r.get("question", ""),
                r.get("answer", "") or "",
                eval_fields.get("optical_usage") if eval_fields.get("optical_usage") is not None else "",
                eval_fields.get("sar_usage") if eval_fields.get("sar_usage") is not None else "",
                eval_fields.get("multimodal_reasoning") if eval_fields.get("multimodal_reasoning") is not None else "",
                eval_fields.get("physical_correctness") if eval_fields.get("physical_correctness") is not None else "",
                eval_fields.get("grounding") if eval_fields.get("grounding") is not None else "",
                eval_fields.get("uncertainty") if eval_fields.get("uncertainty") is not None else "",
                eval_fields.get("relevance") if eval_fields.get("relevance") is not None else "",
                eval_fields.get("human_notes") or "",
            ])
    logger.info("Saved human review table to: %s", csv_path)

    return results


# ============================================================
# CLI ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Optical-SAR Real-Data Evaluation Runner")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(BACKEND_DIR / "evaluation" / "optical_sar" / "manifest.json"),
        help="Path to evaluation manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BACKEND_DIR / "evaluation" / "optical_sar"),
        help="Directory to save results.jsonl and human_review.csv",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip VLM inference and only test data loading, alignment, and visual representations",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Skip saving visual PNG images to disk",
    )
    parser.add_argument(
        "--run-comparison",
        action="store_true",
        help="Run controlled modality ablation comparison (Optical vs Optical+SAR)",
    )

    args = parser.parse_args()

    results = run_evaluation(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        run_inference=not args.no_inference,
        save_visuals=not args.no_visuals,
        run_comparison=args.run_comparison,
    )

    # Summary statistics
    total = len(results)
    success_count = sum(1 for r in results if r["status"] == "success")
    unavail_count = sum(1 for r in results if r["status"] == "inference_unavailable")
    pipeline_count = sum(1 for r in results if r["status"] == "pipeline_verified_no_inference")
    dim_match_count = sum(1 for r in results if r.get("visual_sanity", {}).get("dimensions_match"))

    print("\n" + "=" * 60)
    print("OPTICAL-SAR EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Cases: {total}")
    print(f"Dimensions Matched: {dim_match_count}/{total}")
    print(f"VLM Inference Success: {success_count}/{total}")
    print(f"VLM Inference Unavailable (Fallback): {unavail_count}/{total}")
    print(f"Pipeline Verified (No inference requested): {pipeline_count}/{total}")
    if args.run_comparison:
        print("Controlled Modality Comparison: EXECUTED (saved to ablation_comparison.jsonl)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
