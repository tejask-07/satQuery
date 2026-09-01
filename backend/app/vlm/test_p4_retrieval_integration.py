"""
P4 VLM + BigEarthNet.txt Retrieval Integration Test.

Verifies end-to-end integration:
  User Question
  + Retrieved BigEarthNet.txt Examples
  + Real P2 Numerical Evidence
  + Sentinel-2 Before/After Images + Change Map
  + Sentinel-1 SAR VV/VH Composite
  → Qwen2.5-VL-72B-Instruct Prompt & Generation
  → Grounded Multimodal Answer.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.vlm.model import VLM
from app.vlm.bigearthnet.text_retriever import retrieve_examples


def test_p4_retrieval_integration():
    print("\n" + "=" * 70)
    print("P4 INTEGRATION TEST: Question + BigEarthNet Retrieval + P2 + S1/S2 -> Qwen")
    print("=" * 70)

    question = "Show vegetation change between 2021 and 2025"

    # ---------------------------------------------------------
    # 1. Test retrieval directly for the question
    # ---------------------------------------------------------
    print("\n[Step 1] Retrieving BigEarthNet demonstration examples...")
    examples = retrieve_examples(question, max_examples=3)
    print(f"Retrieved {len(examples)} BigEarthNet examples:")
    assert len(examples) > 0, "Retrieval should return at least 1 example"
    for i, ex in enumerate(examples, 1):
        print(f"  ({i}) Category: {ex['category']} | Type: {ex['type']}")
        print(f"      Q: {ex['input'][:80]}...")
        print(f"      A: {ex['output']}")

    # ---------------------------------------------------------
    # 2. Prepare imagery (S2 before/after, change map, S1 composite)
    # ---------------------------------------------------------
    print("\n[Step 2] Loading S2 optical and S1 SAR imagery...")
    s1_dir = Path(backend_dir) / "data" / "s1_visualizations"
    change_map_path = (
        Path(backend_dir)
        / "app"
        / "evidence"
        / "visualizations"
        / "vegetation_2021_2025_change.png"
    )

    before_path = Path(backend_dir) / "data" / "before_p2.png"
    after_path = Path(backend_dir) / "data" / "after_p2.png"

    # Use available sample imagery or create valid placeholder PIL images
    if before_path.exists():
        before = Image.open(before_path).convert("RGB")
    else:
        before = Image.new("RGB", (100, 100), color=(34, 139, 34))

    if after_path.exists():
        after = Image.open(after_path).convert("RGB")
    else:
        after = Image.new("RGB", (100, 100), color=(160, 82, 45))

    s1_comp_path = s1_dir / "s1_vv_vh_composite.png"
    if s1_comp_path.exists():
        s1_composite = Image.open(s1_comp_path).convert("RGB")
    else:
        s1_composite = Image.new("RGB", (120, 120), color=(100, 100, 200))

    if change_map_path.exists():
        change_map = Image.open(change_map_path).convert("RGB")
    else:
        change_map = Image.new("RGB", (100, 100), color=(255, 215, 0))

    images = {
        "before": before,
        "after": after,
        "change_map": change_map,
        "s1_composite": s1_composite,
    }
    print(f"Images ready: {list(images.keys())}")

    # ---------------------------------------------------------
    # 3. Prepare structured P2 evidence
    # ---------------------------------------------------------
    print("\n[Step 3] Preparing authoritative P2 evidence...")
    evidence = """
Task: vegetation change detection (NDVI)
Time range: 2021 to 2025
Metric: NDVI
Mean NDVI Before: 0.4201
Mean NDVI After: 0.3091
Mean NDVI Change: -0.1110 (decrease)
Changed Pixels: 27806 / 90000
Change Ratio: 30.90%
Threshold: 0.05
Conclusion: Significant vegetation loss detected across 30.90% of the monitored area.
"""

    # ---------------------------------------------------------
    # 4. Run VLM.generate and verify the prompt and output
    # ---------------------------------------------------------
    print("\n[Step 4] Testing VLM prompt building & generation...")

    captured_prompt = None

    def fake_create(model, messages, max_tokens):
        nonlocal captured_prompt
        # The prompt is inside messages[0]['content'][0]['text']
        captured_prompt = messages[0]["content"][0]["text"]
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=(
                        "Between 2021 and 2025, vegetation experienced a significant decrease. "
                        "The mean NDVI dropped from 0.4201 to 0.3091, representing an average change "
                        "of -0.1110 across 30.90% of the monitored area (27,806 changed pixels). "
                        "Sentinel-1 radar backscatter confirms structural canopy loss in the affected sectors."
                    )
                )
            )
        ]
        return mock_resp

    has_real_token = bool(os.getenv("HF_TOKEN"))
    if not has_real_token:
        os.environ["HF_TOKEN"] = "hf_test_dummy_token_for_integration_validation"

    try:
        vlm = VLM()

        # If dummy token, mock the API completion call to inspect the exact prompt
        if not has_real_token or os.environ.get("MOCK_VLM") == "1":
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = fake_create
            vlm.client = mock_client

            answer = vlm.generate(
                question=question,
                evidence=evidence,
                images=images,
            )
        else:
            answer = vlm.generate(
                question=question,
                evidence=evidence,
                images=images,
            )

        print("\n[Step 5] Validating prompt structure and grounded output...")
        print("\nVLM Grounded Answer:")
        print(answer)
        assert len(answer) > 50, "Answer should be substantive"
        assert "vegetation" in answer.lower()
        assert "0.4201" in answer or "-0.1110" in answer or "30.90%" in answer

        if captured_prompt:
            assert (
                "BIGEARTHNET REMOTE-SENSING EXAMPLES" in captured_prompt
            ), "Prompt must include BigEarthNet examples header"
            assert (
                "DEMONSTRATION RULES:" in captured_prompt
                or "IMPORTANT RULES:" in captured_prompt
            )
            assert (
                "Treat backend numerical measurements as authoritative."
                in captured_prompt
            )
            assert (
                "Never copy or use BigEarthNet example values"
                in captured_prompt
            )
            print("\nPrompt validation: PASS (BigEarthNet examples and rules verified)")

        print("\n" + "=" * 70)
        print("P4 RETRIEVAL INTEGRATION TEST: SUCCESS")
        print("=" * 70)

    finally:
        if not has_real_token and os.getenv("HF_TOKEN") == "hf_test_dummy_token_for_integration_validation":
            del os.environ["HF_TOKEN"]


if __name__ == "__main__":
    test_p4_retrieval_integration()
