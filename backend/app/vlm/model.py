import base64
import io
import os
from typing import Dict, Optional

from huggingface_hub import InferenceClient
from PIL import Image

from app.vlm.bigearthnet.text_retriever import (
    retrieve_examples,
    format_examples_for_prompt,
)


MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"


class VLM:
    def __init__(self):
        token = os.getenv("HF_TOKEN")

        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set in the current PowerShell session."
            )

        self.client = InferenceClient(
            provider="auto",
            api_key=token,
        )

    @staticmethod
    def image_to_data_url(
        image: Image.Image,
        max_size: int = 512,
    ) -> str:
        """
        Convert a PIL image into a base64 JPEG data URL,
        resizing if necessary to save bandwidth.
        """
        # Create a copy so we don't modify the original P2 object
        img = image.copy()
        
        # Convert RGBA to RGB for JPEG compatibility
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()

        img.save(
            buffer,
            format="JPEG",
            quality=80,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return (
            "data:image/jpeg;base64,"
            f"{encoded}"
        )

    def generate(
        self,
        image: Optional[Image.Image] = None,
        question: str = "",
        evidence=None,
        images: Optional[
            Dict[str, Image.Image]
        ] = None,
    ) -> str:
        """
        Generate a grounded multimodal answer.

        Supported image roles:

        - satellite image
        - before
        - after
        - change_map
        - s1_vv
        - s1_vh
        - s1_composite
        """

        # =====================================================
        # 0. RETRIEVE BIGEARTHNET CONTEXTUAL EXAMPLES
        # =====================================================

        retrieved_examples = []
        if question:
            try:
                retrieved_examples = retrieve_examples(question, max_examples=3)
            except Exception as exc:
                print(f"[P4 VLM WARNING] Failed to retrieve BigEarthNet examples: {exc}")

        examples_section = ""
        if retrieved_examples:
            examples_section = (
                f"\n{format_examples_for_prompt(retrieved_examples)}\n"
            )

        # =====================================================
        # BUILD IMAGE LIST
        # =====================================================

        image_items = []

        if image is not None:
            image_items.append(
                (
                    "satellite image",
                    image,
                )
            )

        if images:

            if images.get("before") is not None:
                image_items.append(
                    (
                        "Sentinel-2 before image",
                        images["before"],
                    )
                )

            if images.get("after") is not None:
                image_items.append(
                    (
                        "Sentinel-2 after image",
                        images["after"],
                    )
                )

            if images.get("change_map") is not None:
                image_items.append(
                    (
                        "remote-sensing change map",
                        images["change_map"],
                    )
                )

            # -------------------------------------------------
            # Sentinel-1 SAR
            # -------------------------------------------------

            if images.get("s1_vv") is not None:
                image_items.append(
                    (
                        "Sentinel-1 VV SAR image",
                        images["s1_vv"],
                    )
                )

            if images.get("s1_vh") is not None:
                image_items.append(
                    (
                        "Sentinel-1 VH SAR image",
                        images["s1_vh"],
                    )
                )

            if images.get("s1_composite") is not None:
                image_items.append(
                    (
                        "Sentinel-1 VV/VH SAR composite",
                        images["s1_composite"],
                    )
                )

        if not image_items:
            raise ValueError(
                "At least one image must be supplied."
            )

        # =====================================================
        # EVIDENCE
        # =====================================================

        if evidence:

            evidence_text = f"""
REMOTE-SENSING EVIDENCE
=======================

The following information was produced by the
remote-sensing analysis backend.

Treat backend numerical measurements as authoritative.

Do NOT invent, estimate, or modify numerical values
when they are already provided.

Use supplied evidence for quantitative conclusions.

IMPORTANT:
- Do not assume the meaning of visualization colors.
- Only describe change-map colors when their meaning is
  explicitly provided by evidence or visualization metadata.
- Do not infer numerical measurements from image appearance.
- Distinguish measured facts from visual observations.

{evidence}
"""

        else:

            evidence_text = """
REMOTE-SENSING EVIDENCE
=======================

No structured remote-sensing evidence was provided.

Use the supplied imagery for qualitative interpretation only.
"""

        # =====================================================
        # MAIN PROMPT
        # =====================================================

        prompt = f"""
You are the multimodal reasoning layer of SatQuery.

You analyze satellite imagery together with structured
remote-sensing evidence.

Your inputs may include:

1. Sentinel-2 optical imagery.
2. Sentinel-1 SAR imagery.
3. Before/after imagery.
4. Remote-sensing change maps.
5. Structured numerical analysis.
6. Contextual BigEarthNet remote-sensing question/answer examples.

IMPORTANT RULES:

- Treat backend numerical measurements as authoritative.
- BigEarthNet examples are contextual demonstrations only.
- Never copy or use BigEarthNet example values or answers as facts about the current image.
- Answer the user's current query from the current imagery and P2 evidence only.
- Never invent NDVI, NDWI, NDBI, area, percentages,
  thresholds, pixel counts, or change values.
- Never contradict supplied measurements.
- Use supplied statistics for quantitative claims.
- Compare before and after imagery when both are provided.
- Use change maps to understand spatial distribution of
  detected changes.
- Do not assume the meaning of visualization colors.
- Only describe colors when a legend or visualization
  metadata explicitly establishes their meaning.
- Use imagery for qualitative observations only when the
  evidence is visually clear.
- Do not claim that a visual difference exists when it
  cannot reasonably be observed.
- Do not infer precise geographic facts solely from appearance.
- Clearly distinguish measured facts from visual observations.

SENTINEL-1 RULES:

- VV and VH are Sentinel-1 SAR backscatter observations.
- Interpret VV/VH as radar observations, not as optical RGB.
- Do not describe VV/VH as ordinary visible colors.
- Use Sentinel-1 imagery to support qualitative spatial
  interpretation unless quantitative SAR measurements are
  explicitly supplied.
- Do not invent SAR-derived quantities.
- When both Sentinel-1 and Sentinel-2 imagery are supplied,
  use them as complementary modalities rather than treating
  them as identical measurements.

RESPONSE STYLE:

- Answer the user's question directly.
- Put the main conclusion first.
- Use the strongest measured evidence to support the conclusion.
- Clearly separate measured facts from visual observations.
- Keep the answer concise but complete.
- Prefer 1–3 short paragraphs unless additional structure
  is necessary.
- Include important numerical values when supplied.
- Do not repeat the entire evidence block.
- Do not mention internal implementation details.
- Always finish the answer.
- Never end with an incomplete sentence.

USER QUESTION
=============

{question}
{examples_section}
{evidence_text}
"""

        # =====================================================
        # MESSAGE CONTENT
        # =====================================================

        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        prompt_chars = len(prompt)
        evidence_bytes = len(evidence_text.encode('utf-8'))
        before_image_bytes = 0
        after_image_bytes = 0
        change_map_bytes = 0
        s1_bytes = 0

        for label, img in image_items:

            if img is None:
                continue

            # -------------------------------------------------
            # Image label
            # -------------------------------------------------

            content.append(
                {
                    "type": "text",
                    "text": (
                        f"\n--- {label.upper()} ---"
                    ),
                }
            )

            # -------------------------------------------------
            # Image
            # -------------------------------------------------

            data_url = self.image_to_data_url(img)
            url_size = len(data_url)
            
            lbl = label.lower()
            if "before" in lbl:
                before_image_bytes += url_size
            elif "after" in lbl:
                after_image_bytes += url_size
            elif "change map" in lbl:
                change_map_bytes += url_size
            elif "sentinel-1" in lbl or "sar" in lbl or "composite" in lbl:
                s1_bytes += url_size

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                    },
                }
            )

        import json
        total_payload_bytes = len(json.dumps(content).encode('utf-8'))
        print(f"[P4 VLM DIAGNOSTICS]\n"
              f"num_retrieved_examples={len(retrieved_examples)}\n"
              f"prompt_chars={prompt_chars}\n"
              f"evidence_bytes={evidence_bytes}\n"
              f"before_image_bytes={before_image_bytes}\n"
              f"after_image_bytes={after_image_bytes}\n"
              f"change_map_bytes={change_map_bytes}\n"
              f"s1_bytes={s1_bytes}\n"
              f"total_payload_bytes={total_payload_bytes}")

        # =====================================================
        # VLM REQUEST
        # =====================================================

        response = (
            self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                max_tokens=512,
            )
        )

        # =====================================================
        # SAFE ANSWER EXTRACTION
        # =====================================================

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if not answer:
            raise RuntimeError(
                "VLM returned an empty response."
            )

        return answer.strip()