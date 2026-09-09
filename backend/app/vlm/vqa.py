"""Single-image remote-sensing visual question answering."""

from typing import Any, Optional

from app.vlm.model import VLM as _OriginalVLM
from app.vlm.rs_prompts import build_rs_prompt
from app.vlm.rs_vlm import get_rs_vlm

# Exposed for backward compatibility and test fixture monkeypatching
VLM = _OriginalVLM


VQA_PROMPT = """You are performing single-image remote-sensing visual question answering for SatQuery.

Answer the user's question using the supplied image and any supplied evidence.
The image may be optical/multispectral or Sentinel-1 SAR imagery.

Grounding rules:
- Answer directly and concisely.
- Use only information supported by the image and supplied evidence.
- Do not invent facts or scientific measurements.
- Do not invent NDVI, NDWI, or NDBI values.
- Do not invent area, percentages, pixel counts, distances, thresholds, dates, coordinates, sensors, or other quantitative measurements.
- If quantitative information is not supplied as evidence, do not fabricate it.
- Clearly distinguish visual observations from measured evidence.
- Do not perform temporal change analysis from a single image.
- If the answer cannot reliably be determined from the image, say so.

Modality guidance:
- For SAR, treat the image as radar/backscatter imagery, not ordinary RGB photography. Describe structural or spatial backscatter features only when supported by the image.
- For optical or multispectral imagery, describe visible land cover, vegetation, water, structures, and objects only when supported by the image.

USER QUESTION:
{question}

IMAGE MODALITY:
{modality}

Return a concise answer to the user's question."""


def run_vqa(
    image: Any,
    question: str,
    modality: str = "unknown",
    evidence: Optional[Any] = None,
    rs_vlm: Optional[Any] = None,
) -> dict:
    """Answer one question about one image using the RS-VLM runtime abstraction."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    normalized_modality = (
        modality.strip().lower() if isinstance(modality, str) else "unknown"
    )
    prompt = build_rs_prompt(
        VQA_PROMPT.format(
            question=question.strip(),
            modality=normalized_modality,
        )
    )

    # If VLM is monkeypatched in tests or custom legacy caller
    if VLM is not _OriginalVLM:
        vlm = VLM()
        answer = vlm.generate(
            image=image,
            question=prompt,
            evidence=evidence,
        )
        return {
            "task": "single_image_vqa",
            "question": question,
            "answer": answer,
            "modality": normalized_modality,
            "confidence": None,
        }

    # Standard RS-VLM runtime routing
    runtime = rs_vlm or get_rs_vlm()
    structured = runtime.answer(
        image=image,
        question=prompt,
        evidence=evidence,
        task="single_image_vqa",
        metadata={"modality": normalized_modality, "raw_question": question},
    )

    return {
        "task": "single_image_vqa",
        "question": question,
        "answer": structured.get("answer"),
        "modality": normalized_modality,
        "confidence": structured.get("confidence"),
        "model": structured.get("model", "mock-rs-vlm"),
        "status": structured.get("status", "mock"),
        "adapter_loaded": structured.get("adapter_loaded", False),
        "observations": structured.get("observations", []),
        "evidence_used": structured.get("evidence_used", False),
    }

