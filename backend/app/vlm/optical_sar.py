"""
Optical-SAR Multimodal VLM Specialist Module.

Connects the co-registered Optical + SAR raster visual representations from
app.remote_sensing.multimodal.optical_sar to the existing VLM interface (VLM.generate)
for grounded visual reasoning without altering the existing change-detection workflow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from PIL import Image

from app.remote_sensing.multimodal.optical_sar import (
    align_optical_sar_pair,
    build_optical_sar_visuals,
)

logger = logging.getLogger(__name__)


def build_optical_sar_prompt(
    question: str,
    optical_metadata: Dict[str, Any],
    available_sar_modalities: List[str],
) -> str:
    """
    Construct a grounded specialist prompt guiding the VLM on joint Optical + SAR interpretation.
    Incorporates physical remote-sensing principles:
      - SAR backscatter vs visible optical color
      - Artificial composite display channel encoding (R=norm VV, G=norm VH, B=|norm VV - norm VH|)
      - Multivariate polarization interpretation (lower VH does not unconditionally mean smoother)
      - Double-bounce as a possible urban mechanism (not bright = urban unconditionally)
      - Vegetation volume scattering (VH contributes to volume scattering but is not sufficient alone)
      - Water specular reflection caveats (wind/waves/roughness/incidence angle affect SAR return)
      - Optical representation awareness (true-color vs false-color NIR band mapping)
      - Separation of direct visual observations from inferences
      - Prioritizing scene-specific evidence over generic remote-sensing textbook claims
      - Explicit multimodal synthesis
      - Ban on unsupported numbers, dB values, percentages, areas, pixel counts, or thresholds
      - Normalization is display-only, not calibrated physical radar measurements
    """
    is_false_color = optical_metadata.get("is_false_color", False)
    opt_desc = optical_metadata.get("description", "Optical satellite imagery")
    bands_used = optical_metadata.get("bands_used", [])

    optical_type = "False-color composite" if is_false_color else "True-color RGB"
    bands_str = f" ({', '.join(bands_used)})" if bands_used else ""

    sar_desc_lines = []
    if "sar_vv" in available_sar_modalities:
        sar_desc_lines.append("- Sentinel-1 VV co-polarization grayscale radar backscatter (surface roughness and structural alignment)")
    if "sar_vh" in available_sar_modalities:
        sar_desc_lines.append("- Sentinel-1 VH cross-polarization grayscale radar backscatter (sensitive to volume scattering and depolarization)")
    if "sar_composite" in available_sar_modalities:
        sar_desc_lines.append(
            "- Sentinel-1 VV/VH dual-polarization composite (R=norm VV, G=norm VH, B=|norm VV - norm VH| polarization contrast)"
        )

    sar_summary = "\n".join(sar_desc_lines) if sar_desc_lines else "- No SAR bands available"

    optical_guidance = (
        "NOTE: This optical image is a false-color representation, not true visible light.\n"
        "- Do not describe false-color hues as if they were ordinary visible natural colors.\n"
        "- Use the known band mapping when interpreting visible tones (e.g. high NIR reflectance mapped to red/green)."
        if is_false_color else
        "NOTE: This optical image represents true-color visible reflectance."
    )

    prompt = f"""
OPTICAL + SAR MULTIMODAL CONTEXT:
==================================
You are analyzing a co-registered Optical + SAR remote-sensing pair.
The optical and SAR rasters are co-registered on the exact same spatial pixel grid.

PRIMARY IMAGE:
- Optical imagery: {optical_type} [{opt_desc}]{bands_str}.
{optical_guidance}

ADDITIONAL SAR IMAGES:
{sar_summary}

MULTIMODAL INTERPRETATION & COMPLEMENTARY EVIDENCE:
1. Complementary Evidence:
   - Optical provides surface solar reflectance (e.g. vegetation greenness, water color, visible land-use boundaries).
   - SAR provides microwave backscatter sensitive to surface roughness, geometric structure, and moisture/dielectric constant.
   - Built-up/urban areas typically show strong SAR double-bounce backscatter (bright in VV/VH) through vertical structures and ground interactions, but corroborate with optical visible features (roofs, roads, geometry). Do NOT use "bright = urban" as an unconditional rule.
   - Calm water bodies typically act as specular reflectors in radar (dark in VV/VH), but wind, waves, surface roughness, incidence angle, or emergent vegetation can increase return. Do NOT make "dark SAR = water" an unconditional rule; corroborate with optical appearance.
   - Dense vegetation exhibits volume scattering (elevated cross-polarized VH response), but VH response alone is not sufficient to identify vegetation type. Avoid simplistic rules like "low VH = smooth" or "high VH = forest" without explicit corroborating evidence.

2. Modality Differences & SAR is NOT Optical Color:
   - SAR brightness represents normalized radar backscatter visualization, NOT optical surface color.
   - Do not describe radar channels as ordinary visible light colors.
   - In the dual-polarization composite, colors (red, green, blue, yellow, cyan, purple) are artificial display-channel encodings (R=VV, G=VH, B=|VV-VH| polarization contrast).
   - Do NOT interpret composite hues as physical SAR colors, and do NOT claim that a particular composite color automatically represents a specific land-cover class. Discuss composite colors solely in terms of their underlying VV and VH channel levels.

3. Multivariate Polarization (VV & VH) Interpretation:
   - VV and VH are distinct microwave polarization channels with different physical sensitivities.
   - A lower VH backscatter relative to VV does NOT automatically mean smoother terrain or bare ground. Do NOT infer land-cover type solely from the fact that VH is lower or higher.
   - Radar backscatter is multivariate and depends on surface roughness, vegetation geometry, moisture content / dielectric properties, sensor incidence angle, orientation, and scattering mechanisms. Do NOT make simplistic single-factor interpretations.

4. Observation vs. Inference Separation:
   - Clearly distinguish direct visual observations from inferred semantic land-cover interpretations.
   - Direct observations: State what is directly visible in the rasters (e.g., optical texture, localized bright SAR clusters, VV/VH spatial gradients).
   - Inference: Frame interpretations tentatively (e.g., "This pattern is consistent with...", "This may indicate...", "A likely interpretation is...").
   - When evidence is ambiguous or unresolved between modalities, clearly state the uncertainty. Do NOT force unwarranted certainty.

5. Scene-Specific Evidence Over Generic Claims:
   - Do NOT substitute generic remote-sensing textbook knowledge for observations from the supplied images. Prioritize scene-specific evidence visibly present in the supplied rasters.
   - If evidence is insufficient to draw a conclusion, explicitly state that limitation.

6. Multimodal Synthesis:
   - Explicitly synthesize how optical reflectance and radar backscatter complement, confirm, or qualify each other; do NOT merely produce disconnected paragraphs.

7. Grounding & Anti-Hallucination Rules:
   - The 2% to 98% percentile normalization used to render SAR images is strictly for visual display dynamic range. Do NOT interpret normalized 0-255 display values as calibrated radar measurements.
   - Treat backend numerical measurements (if provided in evidence) as authoritative.
   - Do NOT invent numerical backscatter measurements, exact percentages, areas, pixel counts, thresholds, or dB values unless explicitly provided in evidence.

USER QUESTION:
{question}
""".strip()
    return prompt


def generate_optical_sar_fallback_response(
    question: str,
    metadata: Dict[str, Any],
    evidence: Optional[Any] = None,
) -> str:
    """
    Generate a deterministic qualitative summary when the VLM service is unavailable.
    Adheres strictly to remote-sensing physical safeguards and qualitative transparency.
    """
    opt_type = "false-color composite" if metadata.get("optical_is_false_color") else "true-color RGB"
    sar_mods = ", ".join(metadata.get("sar_modalities", ["SAR"])) or "SAR"
    width = metadata.get("width", "unknown")
    height = metadata.get("height", "unknown")
    crs = metadata.get("crs", "unknown")

    lines = [
        f"Optical imagery ({opt_type}) and Sentinel-1 SAR observations ({sar_mods}) have been co-registered onto the optical reference grid ({width}x{height} pixels, CRS: {crs}).",
        f"Query: \"{question}\"",
    ]

    if evidence:
        lines.append(f"Authoritative backend evidence: {evidence}")
    else:
        lines.append(
            "Qualitative Multimodal Synthesis:\n"
            "- Optical spectral reflectance provides surface solar reflectance, visible land-cover texture, and boundaries.\n"
            "- Sentinel-1 SAR radar backscatter provides complementary physical sensitivity to surface roughness, structural geometry, and dielectric properties.\n"
            "- Display encodings: Grayscale intensity displays normalized radar backscatter (not optical color); the dual-polarization composite maps VV to red, VH to green, and |VV-VH| contrast to blue.\n"
            "- Interpretation caveat: Polarization response depends multivariately on geometry, canopy structure, moisture, and incidence angle rather than single-factor assumptions."
        )

    lines.append(
        "Note: The visual language model (VLM) inference service was unavailable; this deterministic multimodal analysis summary was generated from the co-registered metadata."
    )

    return "\n\n".join(lines)


def answer_optical_sar_question(
    aligned_result: Dict[str, Any],
    question: str,
    evidence: Optional[Any] = None,
    vlm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Analyze an aligned Optical + SAR pair using the shared VLM interface and natural language questions.

    Parameters:
        aligned_result: Output dict from align_optical_sar_pair(...).
        question: Natural language question regarding the multimodal pair.
        evidence: Optional structured evidence or numerical analysis from backend.
        vlm: Optional VLM instance with a .generate(...) method. If None, instantiates app.vlm.model.VLM().

    Returns:
        Structured dict with:
            - success: bool
            - answer: str (grounded multimodal answer or deterministic fallback)
            - modalities: list of str (e.g. ["optical", "sar_vv", "sar_vh", "sar_composite"])
            - metadata: dict of spatial and modality properties
            - evidence_used: bool
            - visuals: dict of PIL images passed to VLM
            - error: Optional[str]
            - fallback: bool
    """
    # 1. Question validation
    if not question or not isinstance(question, str) or not question.strip():
        return {
            "success": False,
            "answer": None,
            "error": "A valid non-empty question string must be provided.",
            "modalities": [],
            "metadata": {},
            "evidence_used": False,
            "visuals": {},
            "fallback": False,
        }

    # 2. Aligned result validation
    if not isinstance(aligned_result, dict) or not aligned_result.get("success", False):
        err = "Invalid or unsuccessful aligned_result provided."
        if isinstance(aligned_result, dict) and aligned_result.get("errors"):
            err += f" Errors: {aligned_result['errors']}"
        return {
            "success": False,
            "answer": None,
            "error": err,
            "modalities": [],
            "metadata": {},
            "evidence_used": False,
            "visuals": {},
            "fallback": False,
        }

    optical_info = aligned_result.get("optical") or {}
    if optical_info.get("data") is None:
        return {
            "success": False,
            "answer": None,
            "error": "Missing optical data in aligned_result.",
            "modalities": [],
            "metadata": {},
            "evidence_used": False,
            "visuals": {},
            "fallback": False,
        }

    sar_info = aligned_result.get("sar") or {}
    has_vv = sar_info.get("vv") is not None
    has_vh = sar_info.get("vh") is not None
    if not has_vv and not has_vh:
        return {
            "success": False,
            "answer": None,
            "error": "No valid SAR polarization band (VV or VH) available in aligned_result.",
            "modalities": ["optical"],
            "metadata": {},
            "evidence_used": False,
            "visuals": {},
            "fallback": False,
        }

    # 3. Build visual representations
    try:
        visuals = build_optical_sar_visuals(aligned_result)
    except Exception as exc:
        return {
            "success": False,
            "answer": None,
            "error": f"Failed to build Optical-SAR visuals: {exc}",
            "modalities": [],
            "metadata": {},
            "evidence_used": False,
            "visuals": {},
            "fallback": False,
        }

    # 4. Prepare modalities and images for VLM
    opt_image = visuals["optical"]["image"]
    sar_images: Dict[str, Image.Image] = {}
    modalities = ["optical"]

    if visuals.get("s1_vv", {}).get("image") is not None:
        sar_images["s1_vv"] = visuals["s1_vv"]["image"]
        modalities.append("sar_vv")

    if visuals.get("s1_vh", {}).get("image") is not None:
        sar_images["s1_vh"] = visuals["s1_vh"]["image"]
        modalities.append("sar_vh")

    if visuals.get("s1_composite", {}).get("image") is not None:
        sar_images["s1_composite"] = visuals["s1_composite"]["image"]
        modalities.append("sar_composite")

    # 5. Metadata extraction
    opt_meta = visuals.get("optical", {})
    vis_meta = visuals.get("metadata", {})

    metadata = {
        "reference": "optical",
        "width": vis_meta.get("width"),
        "height": vis_meta.get("height"),
        "crs": vis_meta.get("crs"),
        "optical_is_false_color": opt_meta.get("is_false_color", False),
        "optical_description": opt_meta.get("description", ""),
        "optical_bands_used": opt_meta.get("bands_used", []),
        "sar_modalities": [m for m in modalities if m != "optical"],
        "valid_pixel_count": vis_meta.get("valid_pixel_count"),
        "valid_fraction": vis_meta.get("valid_fraction"),
    }

    # 6. Specialist prompt construction
    grounded_prompt = build_optical_sar_prompt(
        question=question.strip(),
        optical_metadata=opt_meta,
        available_sar_modalities=modalities,
    )

    # 7. VLM invocation with graceful error handling
    vlm_instance = vlm
    if vlm_instance is None:
        try:
            from app.vlm.model import VLM
            vlm_instance = VLM()
        except Exception as init_err:
            logger.warning(f"[OPTICAL-SAR VLM] Could not initialize VLM: {init_err}")
            fallback_answer = generate_optical_sar_fallback_response(
                question=question.strip(),
                metadata=metadata,
                evidence=evidence,
            )
            return {
                "success": True,
                "answer": fallback_answer,
                "error": f"VLM unavailable ({init_err}); provided deterministic interpretation.",
                "fallback": True,
                "modalities": modalities,
                "metadata": metadata,
                "evidence_used": evidence is not None,
                "visuals": {
                    "optical": opt_image,
                    "s1_vv": sar_images.get("s1_vv"),
                    "s1_vh": sar_images.get("s1_vh"),
                    "s1_composite": sar_images.get("s1_composite"),
                },
            }

    try:
        raw_answer = vlm_instance.generate(
            image=opt_image,
            question=grounded_prompt,
            evidence=evidence,
            images=sar_images,
        )
        answer = str(raw_answer).strip() if raw_answer else ""
        if not answer:
            raise RuntimeError("VLM returned an empty response.")

        return {
            "success": True,
            "answer": answer,
            "error": None,
            "fallback": False,
            "modalities": modalities,
            "metadata": metadata,
            "evidence_used": evidence is not None,
            "visuals": {
                "optical": opt_image,
                "s1_vv": sar_images.get("s1_vv"),
                "s1_vh": sar_images.get("s1_vh"),
                "s1_composite": sar_images.get("s1_composite"),
            },
        }

    except Exception as gen_err:
        logger.warning(f"[OPTICAL-SAR VLM] Inference failed: {gen_err}")
        fallback_answer = generate_optical_sar_fallback_response(
            question=question.strip(),
            metadata=metadata,
            evidence=evidence,
        )
        return {
            "success": True,
            "answer": fallback_answer,
            "error": f"VLM inference failed ({gen_err}); provided deterministic interpretation.",
            "fallback": True,
            "modalities": modalities,
            "metadata": metadata,
            "evidence_used": evidence is not None,
            "visuals": {
                "optical": opt_image,
                "s1_vv": sar_images.get("s1_vv"),
                "s1_vh": sar_images.get("s1_vh"),
                "s1_composite": sar_images.get("s1_composite"),
            },
        }


def run_optical_sar_analysis(
    optical_path: Optional[str] = None,
    sar_path: Optional[str] = None,
    question: str = "",
    evidence: Optional[Any] = None,
    vlm: Optional[Any] = None,
    sar_vh_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Orchestrate the full Optical-SAR specialist workflow:
      validate & align pair -> build visuals -> grounded VLM interpretation.

    Raises:
        ValueError: If optical_path or sar_path is missing.
    """
    if not optical_path and not sar_path:
        raise ValueError("Optical-SAR analysis requires both optical_path and sar_path inputs.")
    if not optical_path:
        raise ValueError("Optical-SAR analysis requires an optical_path; only SAR was provided.")
    if not sar_path:
        raise ValueError("Optical-SAR analysis requires a sar_path; only optical was provided.")

    effective_vh = sar_vh_path or kwargs.get("sar_vh") or kwargs.get("vh_path")
    aligned = align_optical_sar_pair(optical_path, sar_path, sar_vh_path=effective_vh)
    if not aligned.get("success", False):
        return {
            "success": False,
            "answer": None,
            "error": f"Optical-SAR alignment failed: {aligned.get('errors')}",
            "modalities": [],
            "metadata": {},
            "evidence_used": False,
            "visuals": {},
            "fallback": False,
        }

    return answer_optical_sar_question(
        aligned_result=aligned,
        question=question,
        evidence=evidence,
        vlm=vlm,
    )

