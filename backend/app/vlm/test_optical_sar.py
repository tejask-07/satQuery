"""
Unit and Integration Tests for Optical-SAR VLM Specialist Module (Step 5).
"""

from typing import Any, Dict, Optional
import inspect
import numpy as np
import pytest
from PIL import Image

from app.vlm.optical_sar import (
    answer_optical_sar_question,
    build_optical_sar_prompt,
    generate_optical_sar_fallback_response,
)
from app.vlm.model import VLM, MODEL_ID


class MockVLM:
    """Mock VLM for deterministic testing without external API or HF_TOKEN dependencies."""

    def __init__(self, answer: str = "Mock grounded multimodal reasoning answer."):
        self.answer = answer
        self.last_call: Dict[str, Any] = {}
        self.call_count: int = 0

    def generate(
        self,
        image: Optional[Image.Image] = None,
        question: str = "",
        evidence: Optional[Any] = None,
        images: Optional[Dict[str, Image.Image]] = None,
    ) -> str:
        self.call_count += 1
        self.last_call = {
            "image": image,
            "question": question,
            "evidence": evidence,
            "images": images,
        }
        return self.answer


class FailingVLM:
    """Mock VLM that simulates inference or network failure."""

    def generate(self, *args, **kwargs):
        raise RuntimeError("Hugging Face API inference timed out.")


def _create_mock_aligned_result(
    width: int = 40,
    height: int = 40,
    optical_bands: int = 4,
    optical_names: Optional[list[str]] = None,
    has_vv: bool = True,
    has_vh: bool = True,
    success: bool = True,
) -> Dict[str, Any]:
    """Helper to synthesize aligned_result structures."""
    if not success:
        return {"success": False, "errors": ["Mock alignment failed: no spatial overlap"]}

    if optical_names is None:
        if optical_bands == 4:
            optical_names = ["Red", "Green", "Blue", "NIR"]
        elif optical_bands == 3:
            optical_names = ["Red", "Green", "NIR"]
        else:
            optical_names = [f"band_{i+1}" for i in range(optical_bands)]

    opt_data = np.ones((optical_bands, height, width), dtype=np.float32)
    for b in range(optical_bands):
        opt_data[b] = (b + 1) * 100.0

    sar_bands = []
    vv_arr = None
    vh_arr = None
    if has_vv:
        vv_arr = np.full((height, width), 2.5, dtype=np.float32)
        sar_bands.append(vv_arr)
    if has_vh:
        vh_arr = np.full((height, width), 0.8, dtype=np.float32)
        sar_bands.append(vh_arr)

    sar_data = np.stack(sar_bands, axis=0) if sar_bands else np.zeros((0, height, width), dtype=np.float32)

    return {
        "success": True,
        "optical": {
            "data": opt_data,
            "metadata": {
                "crs": "EPSG:4326",
                "width": width,
                "height": height,
                "band_names": optical_names,
            },
        },
        "sar": {
            "vv": vv_arr,
            "vh": vh_arr,
            "data": sar_data,
            "metadata": {
                "crs": "EPSG:4326",
                "width": width,
                "height": height,
            },
        },
        "alignment": {
            "reference": "optical",
            "target_crs": "EPSG:4326",
            "target_width": width,
            "target_height": height,
        },
        "valid_mask": np.ones((height, width), dtype=bool),
        "errors": [],
    }


def test_optical_vv_vh_success():
    """Verify that Optical + VV + VH successfully reach the VLM and return structured answer."""
    aligned = _create_mock_aligned_result(has_vv=True, has_vh=True)
    mock_vlm = MockVLM(answer="Both optical and SAR indicate dense urban structures and water.")

    res = answer_optical_sar_question(
        aligned_result=aligned,
        question="What features are clearly visible in both optical and SAR images?",
        vlm=mock_vlm,
    )

    assert res["success"] is True
    assert res["answer"] == "Both optical and SAR indicate dense urban structures and water."
    assert res["error"] is None
    assert res["fallback"] is False
    assert set(res["modalities"]) == {"optical", "sar_vv", "sar_vh", "sar_composite"}
    assert mock_vlm.call_count == 1
    assert "s1_vv" in mock_vlm.last_call["images"]
    assert "s1_vh" in mock_vlm.last_call["images"]
    assert "s1_composite" in mock_vlm.last_call["images"]


def test_optical_vv_only_works():
    """Verify that single-band SAR (VV only) passes s1_vv and safely omits VH and composite."""
    aligned = _create_mock_aligned_result(has_vv=True, has_vh=False)
    mock_vlm = MockVLM(answer="Optical shows open ground, SAR VV shows moderate backscatter.")

    res = answer_optical_sar_question(
        aligned_result=aligned,
        question="Describe the land cover.",
        vlm=mock_vlm,
    )

    assert res["success"] is True
    assert res["answer"] == "Optical shows open ground, SAR VV shows moderate backscatter."
    assert res["modalities"] == ["optical", "sar_vv"]
    assert mock_vlm.call_count == 1
    assert "s1_vv" in mock_vlm.last_call["images"]
    assert "s1_vh" not in mock_vlm.last_call["images"]
    assert "s1_composite" not in mock_vlm.last_call["images"]


def test_missing_sar_rejected_cleanly():
    """Verify that an aligned result with no SAR bands is rejected cleanly without calling VLM."""
    aligned = _create_mock_aligned_result(has_vv=False, has_vh=False)
    mock_vlm = MockVLM()

    res = answer_optical_sar_question(
        aligned_result=aligned,
        question="Where are the buildings?",
        vlm=mock_vlm,
    )

    assert res["success"] is False
    assert res["answer"] is None
    assert "No valid SAR polarization band" in res["error"]
    assert mock_vlm.call_count == 0


def test_failed_alignment_rejected_cleanly():
    """Verify that a failed alignment input is rejected cleanly without calling VLM."""
    aligned = _create_mock_aligned_result(success=False)
    mock_vlm = MockVLM()

    res = answer_optical_sar_question(
        aligned_result=aligned,
        question="Are there built-up areas?",
        vlm=mock_vlm,
    )

    assert res["success"] is False
    assert res["answer"] is None
    assert "Invalid or unsuccessful aligned_result" in res["error"]
    assert mock_vlm.call_count == 0


def test_question_forwarded_correctly():
    """Verify that the user query is forwarded and embedded into the grounded prompt."""
    aligned = _create_mock_aligned_result()
    mock_vlm = MockVLM()
    test_question = "Where does the SAR image provide additional information compared with optical?"

    res = answer_optical_sar_question(
        aligned_result=aligned,
        question=test_question,
        vlm=mock_vlm,
    )

    assert res["success"] is True
    prompt_sent = mock_vlm.last_call["question"]
    assert test_question in prompt_sent
    assert "OPTICAL + SAR MULTIMODAL CONTEXT:" in prompt_sent


def test_evidence_forwarded_correctly():
    """Verify that structured scientific evidence is forwarded directly to the VLM."""
    aligned = _create_mock_aligned_result()
    mock_vlm = MockVLM()
    mock_evidence = {
        "urban_score": 0.88,
        "water_score": 0.05,
        "verified_regions": 3,
    }

    res = answer_optical_sar_question(
        aligned_result=aligned,
        question="Confirm urban classification.",
        evidence=mock_evidence,
        vlm=mock_vlm,
    )

    assert res["success"] is True
    assert res["evidence_used"] is True
    assert mock_vlm.last_call["evidence"] == mock_evidence


def test_vlm_receives_optical_via_image_and_sar_via_images():
    """Verify exact VLM contract: optical via image=, SAR bands via images dict."""
    aligned = _create_mock_aligned_result()
    mock_vlm = MockVLM()

    res = answer_optical_sar_question(
        aligned_result=aligned,
        question="Compare optical and radar responses.",
        vlm=mock_vlm,
    )

    assert res["success"] is True

    # 1. Primary image is optical PIL Image
    primary_img = mock_vlm.last_call["image"]
    assert isinstance(primary_img, Image.Image)
    assert primary_img.mode == "RGB"
    assert primary_img.size == (40, 40)

    # 2. Secondary images dict contains SAR PIL Images
    sar_dict = mock_vlm.last_call["images"]
    assert isinstance(sar_dict, dict)
    assert isinstance(sar_dict["s1_vv"], Image.Image)
    assert sar_dict["s1_vv"].mode == "L"
    assert isinstance(sar_dict["s1_vh"], Image.Image)
    assert sar_dict["s1_vh"].mode == "L"
    assert isinstance(sar_dict["s1_composite"], Image.Image)
    assert sar_dict["s1_composite"].mode == "RGB"


def test_false_color_optical_metadata_preserved():
    """Verify false-color optical products are explicitly flagged in metadata and prompt."""
    # Optical with Red, Green, NIR (no Blue) -> false-color NIR
    aligned_fc = _create_mock_aligned_result(optical_bands=3, optical_names=["Red", "Green", "NIR"])
    mock_vlm = MockVLM()

    res_fc = answer_optical_sar_question(
        aligned_result=aligned_fc,
        question="Describe vegetation.",
        vlm=mock_vlm,
    )

    assert res_fc["success"] is True
    assert res_fc["metadata"]["optical_is_false_color"] is True
    assert "NIR" in res_fc["metadata"]["optical_description"]
    prompt_fc = mock_vlm.last_call["question"]
    assert "False-color composite" in prompt_fc
    assert "not true visible light" in prompt_fc

    # Optical with True RGB -> is_false_color False
    aligned_rgb = _create_mock_aligned_result(optical_bands=4, optical_names=["Red", "Green", "Blue", "NIR"])
    res_rgb = answer_optical_sar_question(
        aligned_result=aligned_rgb,
        question="Describe vegetation.",
        vlm=mock_vlm,
    )

    assert res_rgb["success"] is True
    assert res_rgb["metadata"]["optical_is_false_color"] is False
    prompt_rgb = mock_vlm.last_call["question"]
    assert "True-color RGB" in prompt_rgb


def test_vlm_failure_handled_cleanly():
    """Verify that when VLM inference fails, a deterministic fallback response is returned without crashing."""
    aligned = _create_mock_aligned_result()
    failing_vlm = FailingVLM()

    res = answer_optical_sar_question(
        aligned_result=aligned,
        question="Identify built-up regions.",
        vlm=failing_vlm,
    )

    assert res["success"] is True
    assert res["fallback"] is True
    assert "VLM inference failed" in res["error"]
    assert isinstance(res["answer"], str)
    assert len(res["answer"]) > 0
    assert "Optical imagery" in res["answer"]
    assert "co-registered onto the optical reference grid" in res["answer"]
    assert "Identify built-up regions" in res["answer"]


def test_existing_vlm_interface_untouched():
    """Verify that the existing VLM class contract, parameters, and defaults remain intact."""
    sig = inspect.signature(VLM.generate)
    params = sig.parameters

    assert "image" in params
    assert "question" in params
    assert "evidence" in params
    assert "images" in params

    # Confirm static helper still exists on VLM
    assert hasattr(VLM, "image_to_data_url")
    assert MODEL_ID == "Qwen/Qwen2.5-VL-72B-Instruct"


def test_invalid_or_empty_question_rejected():
    """Verify clean rejection when question is empty, whitespace, or non-string."""
    aligned = _create_mock_aligned_result()
    mock_vlm = MockVLM()

    for bad_q in ("", "   ", None, 123):
        res = answer_optical_sar_question(aligned, bad_q, vlm=mock_vlm)  # type: ignore
        assert res["success"] is False
        assert "valid non-empty question" in res["error"]
    assert mock_vlm.call_count == 0


def test_missing_optical_data_rejected():
    """Verify clean rejection when optical raster data is missing."""
    aligned = _create_mock_aligned_result()
    aligned["optical"]["data"] = None
    mock_vlm = MockVLM()

    res = answer_optical_sar_question(aligned, "Any change?", vlm=mock_vlm)
    assert res["success"] is False
    assert "Missing optical data" in res["error"]
    assert mock_vlm.call_count == 0


# ============================================================
# STEP 11 PROMPT SAFEGUARD CONTRACT TESTS
# ============================================================

def test_step11_prompt_physical_safeguards():
    """Verify presence of all 12 required conceptual safeguards in the specialist prompt."""
    prompt = build_optical_sar_prompt(
        question="Analyze the urban, vegetation, and water characteristics.",
        optical_metadata={"is_false_color": False, "description": "True-color Sentinel-2 RGB", "bands_used": ["B04", "B03", "B02"]},
        available_sar_modalities=["sar_vv", "sar_vh", "sar_composite"],
    )
    lower = prompt.lower()

    # 1. SAR is not visible RGB color
    assert "not optical color" in lower or "not surface color" in lower
    assert "backscatter" in lower

    # 2. VV and VH are distinct polarizations
    assert "polarization" in lower and "distinct" in lower

    # 3. Lower VH does not automatically mean smoother terrain
    assert "lower vh" in lower and ("not automatically mean" in lower or "not infer" in lower or "smoother" in lower)

    # 4. Double-bounce is described as a possible urban scattering mechanism (not bright = urban unconditionally)
    assert "double-bounce" in lower or "double bounce" in lower
    assert "bright" in lower and "urban" in lower

    # 5. VH can contribute to vegetation/volume-scattering interpretation but is not sufficient alone
    assert "volume scattering" in lower
    assert "vegetation" in lower and ("not sufficient" in lower or "alone" in lower)

    # 6. Water can have low SAR backscatter under suitable surface conditions (specular reflection)
    assert "specular" in lower
    assert "water" in lower and ("roughness" in lower or "waves" in lower or "wind" in lower)

    # 7. Composite colors are display encoding, not physical SAR colors
    assert "composite" in lower
    assert "display" in lower or "encoding" in lower
    assert "artificial" in lower or "channel" in lower

    # 8. Step 4 normalization is not physical calibration
    assert "normalization" in lower and ("calibrated" in lower or "display" in lower)

    # 9. Observation vs inference distinction
    assert "observation" in lower and "inference" in lower

    # 10. Scene-specific evidence should take priority over generic knowledge
    assert "scene-specific" in lower or "generic" in lower or "textbook" in lower

    # 11. Unsupported numerical claims remain forbidden
    assert "db" in lower and ("not invent" in lower or "prohibit" in lower or "safeguard" in lower)
    assert "percentages" in lower or "pixel counts" in lower

    # 12. True-color optical imagery is identified
    assert "true-color" in lower


def test_step11_false_color_optical_prompt_safeguards():
    """Verify false-color optical imagery is identified and explicitly warned against treating as visible natural colors."""
    prompt_fc = build_optical_sar_prompt(
        question="Evaluate land cover.",
        optical_metadata={"is_false_color": True, "description": "False-color NIR composite", "bands_used": ["NIR", "Red", "Green"]},
        available_sar_modalities=["sar_vv", "sar_vh"],
    )
    lower_fc = prompt_fc.lower()

    assert "false-color" in lower_fc
    assert "not true visible light" in lower_fc
    assert "band mapping" in lower_fc


def test_step11_fallback_response_safeguards():
    """Verify deterministic fallback response follows physical interpretation caveats and transparency."""
    fallback = generate_optical_sar_fallback_response(
        question="Describe vegetation and water.",
        metadata={
            "optical_is_false_color": False,
            "sar_modalities": ["sar_vv", "sar_vh", "sar_composite"],
            "width": 120,
            "height": 120,
            "crs": "EPSG:32633",
        },
    )
    lower_fb = fallback.lower()

    assert "deterministic multimodal analysis summary" in lower_fb or "qualitative" in lower_fb
    assert "backscatter" in lower_fb
    assert "multivariately" in lower_fb or "polarization" in lower_fb
    assert "display encodings" in lower_fb or "encoding" in lower_fb

