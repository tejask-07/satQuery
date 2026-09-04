import pytest

from app.vlm import caption


class FakeVLM:
    instances = []

    def __init__(self):
        self.calls = []
        self.__class__.instances.append(self)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return "A satellite scene with visible land-cover patterns."


@pytest.fixture(autouse=True)
def reset_fake_vlm(monkeypatch):
    FakeVLM.instances.clear()
    monkeypatch.setattr(caption, "VLM", FakeVLM)


def test_run_caption_forwards_optical_image_and_evidence():
    image = object()
    evidence = {"source": "visual observation"}

    result = caption.run_caption(
        image=image,
        modality="optical",
        evidence=evidence,
    )

    assert result == {
        "task": "single_image_caption",
        "caption": "A satellite scene with visible land-cover patterns.",
        "modality": "optical",
        "confidence": None,
    }
    call = FakeVLM.instances[0].calls[0]
    assert call["image"] is image
    assert call["evidence"] is evidence
    assert "optical" in call["question"]


def test_run_caption_prompt_is_grounded_and_non_quantitative():
    caption.run_caption(object(), modality="multispectral")

    prompt = FakeVLM.instances[0].calls[0]["question"]
    assert "single-image remote-sensing captioning" in prompt
    assert "Describe only features visually supported" in prompt
    assert "Do not invent area, percentages, pixel counts" in prompt
    assert "NDVI, NDWI, or NDBI values" in prompt
    assert "Do not perform temporal change analysis" in prompt
    assert "Return only the concise caption" in prompt


def test_run_caption_supports_sar_as_radar_backscatter():
    image = object()

    result = caption.run_caption(
        image=image,
        modality="SAR",
    )

    prompt = FakeVLM.instances[0].calls[0]["question"]
    assert result["modality"] == "sar"
    assert "radar/backscatter imagery" in prompt
    assert "ordinary RGB photography" in prompt
    assert FakeVLM.instances[0].calls[0]["image"] is image


def test_run_caption_does_not_fake_confidence():
    result = caption.run_caption(object(), modality="UNKNOWN")

    assert result["modality"] == "unknown"
    assert result["confidence"] is None
