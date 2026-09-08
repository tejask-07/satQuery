"""JSONL dataset and Qwen2.5-VL message formatting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from app.vlm.rs_prompts import build_rs_prompt


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not all(record.get(key) for key in ("image", "question", "answer")):
                raise ValueError(f"line {line_number}: image, question, and answer are required")
            records.append(record)
    if not records:
        raise ValueError(f"manifest contains no records: {path}")
    return records


def split_records(
    records: list[dict[str, Any]],
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    import random

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * validation_fraction))
    test_size = max(1, round(len(shuffled) * validation_fraction))
    if validation_size + test_size >= len(shuffled):
        validation_size = 1
        test_size = 0 if len(shuffled) == 1 else 1
    validation = shuffled[:validation_size]
    test = shuffled[validation_size:validation_size + test_size]
    return shuffled[validation_size + test_size:], validation, test


def build_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the official Qwen chat-template message structure."""
    image_path = Path(record["image"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": build_rs_prompt(record["question"])},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": record["answer"]}]},
    ]


def format_with_processor(processor: Any, record: dict[str, Any], max_length: int = 2048) -> dict[str, Any]:
    """Tokenize one record using the selected Qwen processor."""
    from PIL import Image

    messages = build_messages(record)
    prompt_messages = [messages[0]]
    full_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    user_prompt = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    with Image.open(record["image"]) as image:
        image = image.convert("RGB")
        tokenized = processor(
            text=[full_prompt],
            images=[image],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        prompt_tokens = processor(
            text=[user_prompt],
            images=[image],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
    result = {
        key: value.squeeze(0) if key in {"input_ids", "attention_mask"} else value
        for key, value in tokenized.items()
    }
    labels = result["input_ids"].clone()
    prompt_length = prompt_tokens["input_ids"].shape[-1]
    labels[..., :prompt_length] = -100
    if "attention_mask" in result:
        labels[result["attention_mask"] == 0] = -100
    result["labels"] = labels
    return result


class QwenVLMCollator:
    """Tokenize and pad records while masking non-assistant tokens."""

    def __init__(self, processor: Any, max_length: int = 2048) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        import torch

        encoded = [
            format_with_processor(self.processor, record, self.max_length)
            for record in records
        ]
        keys = set().union(*(item.keys() for item in encoded))
        batch: dict[str, Any] = {}
        for key in keys:
            values = [item[key] for item in encoded if key in item]
            if key == "labels":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=-100
                )
            elif key == "input_ids":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    values,
                    batch_first=True,
                    padding_value=self.processor.tokenizer.pad_token_id,
                )
            elif key == "attention_mask":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=0
                )
            elif key == "pixel_values":
                batch[key] = torch.cat(values, dim=0)
            elif key == "image_grid_thw":
                batch[key] = torch.cat(values, dim=0)
            else:
                batch[key] = torch.stack(values)
        return batch
