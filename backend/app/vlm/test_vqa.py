import pytest

from app.vlm import vqa


QUESTIONS = [
    "What is visible in this satellite image?",
    "What type of land cover dominates the image?",
    "Is there a water body visible?",
    "Are there built-up or urban regions in the image?",
    "What major objects are visible?",
]


class FakeVLM:
    instances = []

    def __init__(self):
        self.calls = []
        self.__class__.instances.append(self)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return "The image supports a qualitative answer."


@pytest.fixture(autouse=True)
def reset_fake_vlm(monkeypatch):
    FakeVLM.instances.clear()
    monkeypatch.setattr(vqa, "VLM", FakeVLM)


@pytest.mark.parametrize("question", QUESTIONS)
def test_run_vqa_supports_optical_questions(question):
    image = object()
    evidence = {"statistics": {"mean": 0.42}}

    result = vqa.run_vqa(image, question, modality="optical", evidence=evidence)

    assert result == {
        "task": "single_image_vqa",
        "question": question,
        "answer": "The image supports a qualitative answer.",
        "modality": "optical",
        "confidence": None,
    }
    call = FakeVLM.instances[0].calls[0]
    assert call["image"] is image
    assert call["evidence"] is evidence
    assert question in call["question"]


def test_run_vqa_builds_grounded_optical_prompt():
    vqa.run_vqa(object(), "Is vegetation visible?", modality="optical")

    prompt = FakeVLM.instances[0].calls[0]["question"]
    assert "single-image remote-sensing visual question answering" in prompt
    assert "Do not invent NDVI, NDWI, or NDBI values" in prompt
    assert "Do not perform temporal change analysis" in prompt
    assert "optical" in prompt


def test_run_vqa_supports_sar_without_treating_it_as_rgb():
    image = object()

    result = vqa.run_vqa(
        image,
        "What structural features are visible in this SAR image?",
        modality="sar",
    )

    prompt = FakeVLM.instances[0].calls[0]["question"]
    assert result["modality"] == "sar"
    assert "radar/backscatter imagery" in prompt
    assert "not ordinary RGB photography" in prompt
    assert FakeVLM.instances[0].calls[0]["image"] is image


def test_run_vqa_preserves_original_question_and_does_not_fake_confidence():
    question = "  What is visible?  "

    result = vqa.run_vqa(object(), question, modality="UNKNOWN")

    assert result["question"] == question
    assert result["modality"] == "unknown"
    assert result["confidence"] is None


def test_run_vqa_rejects_empty_questions():
    with pytest.raises(ValueError, match="non-empty"):
        vqa.run_vqa(object(), "")
