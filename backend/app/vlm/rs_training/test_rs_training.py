from pathlib import Path
import json

import pytest
from PIL import Image

from app.vlm.rs_prompts import build_rs_prompt
from app.vlm.rs_training.dataset import build_messages, load_records, split_records
from app.vlm.rs_training.evaluate import adapter_available
from app.vlm.rs_training.prepare_dataset import prepare_bigearthnet, prepare_rsvqa


def test_rs_prompt_is_shared_and_grounded():
    prompt = build_rs_prompt("Describe the image.")
    assert "remote-sensing" in prompt
    assert "Do not invent coordinates" in prompt


def test_manifest_and_split_use_real_records(tmp_path: Path):
    image = tmp_path / "scene.png"
    image.write_bytes(b"fixture")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image": str(image), "question": "What is visible?", "answer": "Vegetation."}) + "\n",
        encoding="utf-8",
    )
    records = load_records(manifest)
    train, validation, test = split_records(records, validation_fraction=0.5)
    assert len(train) == 0 or len(validation) == 1
    assert len(test) == 0
    messages = build_messages(records[0])
    assert messages[0]["content"][0]["type"] == "image"
    assert messages[1]["content"][0]["text"] == "Vegetation."


def test_invalid_manifest_record_is_rejected(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"image": "missing.png", "question": "Q"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="required"):
        load_records(manifest)


def test_adapter_checkpoint_detection(tmp_path: Path):
    assert not adapter_available(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fixture")
    assert adapter_available(tmp_path)


def test_local_message_builder_preserves_image_order(monkeypatch):
    from app.vlm.model import VLM

    vlm = VLM.__new__(VLM)
    monkeypatch.setattr(vlm, "image_to_data_url", lambda image: image)
    content = vlm._build_local_content(
        [("before", "BEFORE"), ("after", "AFTER"), ("change map", "CHANGE")],
        "question",
    )
    images = [item["image"] for item in content if item["type"] == "image"]
    assert images == ["BEFORE", "AFTER", "CHANGE"]


def test_prepare_fails_when_no_real_images_are_available(monkeypatch, tmp_path: Path):
    class EmptyDatasetModule:
        @staticmethod
        def load_dataset(*args, **kwargs):
            return iter([{"patch_id": "missing", "input": "Q", "output": "A"}])

    monkeypatch.setitem(__import__("sys").modules, "datasets", EmptyDatasetModule)
    with pytest.raises(RuntimeError, match="No usable RS-VLM training examples"):
        prepare_bigearthnet(tmp_path / "manifest.jsonl", tmp_path)


def _write_rsvqa_fixture(root: Path, counts=(3, 2, 2)) -> None:
    image_dir = root / "Images_LR"
    image_dir.mkdir(parents=True)
    split_names = ("train", "val", "test")
    question_id = 0
    for split, count in zip(split_names, counts):
        questions = []
        answers = []
        images = []
        for index in range(count):
            image_id = f"image_{split}_{index}"
            current_question_id = f"question_{question_id}"
            filename = f"{image_id}.png"
            Image.new("RGB", (4, 4), color="green").save(image_dir / filename)
            questions.append({"question_id": current_question_id, "image_id": image_id, "question": f"Question {question_id}?"})
            answers.append({"question_id": current_question_id, "answer": f"Answer {question_id}"})
            images.append({"image_id": image_id, "file_name": f"Images_LR/{filename}"})
            question_id += 1
        (root / f"LR_split_{split}_questions.json").write_text(json.dumps(questions), encoding="utf-8")
        (root / f"LR_split_{split}_answers.json").write_text(json.dumps(answers), encoding="utf-8")
        (root / f"LR_split_{split}_images.json").write_text(json.dumps(images), encoding="utf-8")


def test_rsvqa_parser_preserves_splits_and_answers(tmp_path: Path):
    _write_rsvqa_fixture(tmp_path)
    output = tmp_path / "manifest.jsonl"
    stats = prepare_rsvqa(output, tmp_path)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert stats["valid_records"] == 7
    assert {record["split"] for record in records} == {"train", "validation", "test"}
    assert records[0]["answer"].startswith("Answer ")


def test_rsvqa_missing_and_invalid_images_are_reported(tmp_path: Path):
    _write_rsvqa_fixture(tmp_path, counts=(2, 1, 1))
    questions_path = tmp_path / "LR_split_train_questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    questions.append({"question_id": "missing-q", "image_id": "missing-image", "question": "Missing?"})
    questions_path.write_text(json.dumps(questions), encoding="utf-8")
    answers_path = tmp_path / "LR_split_train_answers.json"
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    answers.append({"question_id": "missing-q", "answer": "no image"})
    answers_path.write_text(json.dumps(answers), encoding="utf-8")
    (tmp_path / "Images_LR" / "image_train_0.png").write_bytes(b"corrupt")
    stats = prepare_rsvqa(tmp_path / "manifest.jsonl", tmp_path)
    assert stats["missing_images"] == 1
    assert stats["invalid_images"] == 1


def test_rsvqa_limits_are_deterministic(tmp_path: Path):
    _write_rsvqa_fixture(tmp_path, counts=(10, 6, 6))
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    prepare_rsvqa(first, tmp_path, 4, 2, 2, seed=17)
    prepare_rsvqa(second, tmp_path, 4, 2, 2, seed=17)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    records = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert {split: sum(record["split"] == split for record in records) for split in ("train", "validation", "test")} == {"train": 4, "validation": 2, "test": 2}


def test_rsvqa_smoke_test_limits_each_split(tmp_path: Path):
    _write_rsvqa_fixture(tmp_path, counts=(21, 6, 6))
    output = tmp_path / "smoke.jsonl"
    prepare_rsvqa(output, tmp_path, smoke_test=True)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {split: sum(record["split"] == split for record in records) for split in ("train", "validation", "test")} == {"train": 20, "validation": 5, "test": 5}


def test_processor_format_masks_prompt_tokens(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from PIL import Image
    from app.vlm.rs_training.dataset import format_with_processor

    image_path = tmp_path / "scene.png"
    Image.new("RGB", (2, 2), color="green").save(image_path)

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt" if kwargs["add_generation_prompt"] else "prompt answer"

        def __call__(self, text, **kwargs):
            full = text[0] == "prompt answer"
            ids = torch.tensor([[1, 2, 3, 4]] if full else [[1, 2]])
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.zeros((1, 3)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            }

    result = format_with_processor(
        FakeProcessor(),
        {"image": str(image_path), "question": "What is visible?", "answer": "Vegetation."},
    )
    assert result["labels"].tolist() == [[-100, -100, 3, 4]]
