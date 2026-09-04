"""Single-image satellite-image captioning."""

from typing import Any, Optional

from app.vlm.model import VLM


CAPTION_PROMPT = """You are performing single-image remote-sensing captioning for SatQuery.

Write one concise caption describing the supplied satellite image and any supplied evidence.
The image may be optical/multispectral or Sentinel-1 SAR imagery.

Grounding rules:
- Describe only features visually supported by the image.
- Use supplied evidence only when it supports the description.
- Do not invent facts, measurements, dates, coordinates, sensors, locations, or geographic details.
- Do not invent area, percentages, pixel counts, distances, thresholds, NDVI, NDWI, or NDBI values.
- Do not perform temporal change analysis from a single image.
- If a feature is unclear, use appropriately uncertain language or omit it.
- For SAR, describe the image as radar/backscatter imagery rather than ordinary RGB photography.
- For optical or multispectral imagery, describe visible land cover, vegetation, water, structures, and objects only when supported.
- Return only the concise caption, without labels or analysis notes.

IMAGE MODALITY:
{modality}
"""


def run_caption(
    image: Any,
    modality: str = "unknown",
    evidence: Optional[Any] = None,
) -> dict:
    """Generate one grounded caption for one satellite image."""
    normalized_modality = (
        modality.strip().lower()
        if isinstance(modality, str)
        else "unknown"
    )
    prompt = CAPTION_PROMPT.format(modality=normalized_modality)

    vlm = VLM()
    caption = vlm.generate(
        image=image,
        question=prompt,
        evidence=evidence,
    )

    return {
        "task": "single_image_caption",
        "caption": caption,
        "modality": normalized_modality,
        "confidence": None,
    }
