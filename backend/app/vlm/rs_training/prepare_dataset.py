"""Prepare verified RSVQA or BigEarthNet records for RS-VLM training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from .dataset import split_records

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SPLITS = ("train", "validation", "test")


def _record_from_sample(sample: dict, image_root: Path) -> dict | None:
    """Convert a legacy BigEarthNet.txt row only when its image is present."""
    question = sample.get("input")
    answer = sample.get("output")
    patch_id = sample.get("patch_id")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(answer, str) or not answer.strip() or not isinstance(patch_id, str):
        return None
    candidates = [
        image_root / patch_id,
        *(image_root / f"{patch_id}{suffix}" for suffix in IMAGE_SUFFIXES),
        *(image_root / patch_id / f"{patch_id}{suffix}" for suffix in IMAGE_SUFFIXES),
    ]
    image_path = next((path for path in candidates if path.is_file()), None)
    if image_path is None:
        return None
    return {"image": str(image_path.resolve()), "question": question.strip(), "answer": answer.strip(), "source": "BigEarthNet.txt", "patch_id": patch_id, "category": sample.get("category"), "type": sample.get("type")}


def _json_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("questions", "answers", "images", "annotations", "data", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [{str(key): item} for key, item in value.items()]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _normalise_split(value: str) -> str:
    return "validation" if value.lower() == "val" else value.lower()


def _find_files(root: Path, kind: str, split: str) -> list[Path]:
    found: set[Path] = set()
    split_names = (split, "val") if split == "validation" else (split,)
    for split_name in split_names:
        for pattern in (f"*{split_name}*{kind}*.json", f"*{kind}*{split_name}*.json"):
            found.update(root.rglob(pattern))
    return sorted(found)


def _value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _id(value: Any) -> str | None:
    return None if value is None else str(value)


def _mapping_records(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Expand one-key mapping records such as {"image_id": "file.jpg"}."""
    expanded = []
    for item in records:
        if len(item) == 1:
            key, value = next(iter(item.items()))
            if not isinstance(value, dict):
                expanded.append({keys[0]: key, keys[1]: value})
                continue
        expanded.append(item)
    return expanded


def _annotation_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join RSVQA question/answer/image metadata files by identifiers."""
    joined: list[dict[str, Any]] = []
    diagnostics = {"annotation_records": 0, "annotation_files": 0}
    for split in SPLITS:
        question_files = _find_files(root, "question", split)
        answer_files = _find_files(root, "answer", split)
        image_metadata_files = _find_files(root, "image", split)
        if not question_files or not answer_files:
            continue
        diagnostics["annotation_files"] += len(question_files) + len(answer_files)
        questions = [item for path in question_files for item in _json_records(path)]
        answers = [item for path in answer_files for item in _json_records(path)]
        answers = _mapping_records(answers, ("question_id", "answer"))
        image_metadata = [item for path in image_metadata_files for item in _json_records(path)]
        image_metadata = _mapping_records(image_metadata, ("image_id", "file_name"))
        answer_by_question: dict[str, Any] = {}
        for item in answers:
            key = _id(_value(item, ("question_id", "questionId", "id")))
            answer = _value(item, ("answer", "answers", "label", "value"))
            if key is not None and answer is not None:
                answer_by_question[key] = answer
        image_by_id: dict[str, Any] = {}
        for item in image_metadata:
            key = _id(_value(item, ("image_id", "imageId", "id", "img_id", "filename")))
            image = _value(item, ("file_name", "filename", "image", "path", "name"))
            if key is not None and image is not None:
                image_by_id[key] = image
        for question_item in questions:
            question_id = _id(_value(question_item, ("question_id", "questionId", "id")))
            image_id = _id(_value(question_item, ("image_id", "imageId", "img_id", "image")))
            question = _value(question_item, ("question", "query", "text"))
            answer = _value(question_item, ("answer", "answers", "label"))
            if answer is None and question_id is not None:
                answer = answer_by_question.get(question_id)
            image = image_by_id.get(image_id) if image_id is not None else None
            if image is None:
                image = _value(question_item, ("file_name", "filename", "path", "image"))
            if question is None or answer is None:
                continue
            joined.append({"split": _normalise_split(split), "image_ref": None if image is None else str(image), "question": str(question), "answer": answer, "question_id": question_id, "image_id": image_id})
    diagnostics["annotation_records"] = len(joined)
    return joined, diagnostics


def _resolve_image(root: Path, image_ref: str | None) -> Path | None:
    if not image_ref:
        return None
    reference = Path(image_ref)
    candidates = [reference if reference.is_absolute() else root / reference]
    candidates.extend(root.rglob(reference.name))
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate.resolve()
    return None


def _valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _limit_split(records: list[dict[str, Any]], limit: int | None, rng: random.Random) -> list[dict[str, Any]]:
    if limit is None or len(records) <= limit:
        return records
    indexes = sorted(rng.sample(range(len(records)), limit))
    return [records[index] for index in indexes]


def prepare_rsvqa(output: Path, dataset_root: Path, max_train_samples: int | None = None, max_validation_samples: int | None = None, max_test_samples: int | None = None, seed: int = 42, smoke_test: bool = False) -> dict[str, int]:
    """Discover, validate, and write RSVQA image/question/answer records."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"RSVQA dataset directory not found: {dataset_root}")
    annotations, annotation_stats = _annotation_records(dataset_root)
    valid: list[dict[str, Any]] = []
    missing_images = 0
    invalid_images = 0
    for item in annotations:
        image_path = _resolve_image(dataset_root, item["image_ref"])
        if image_path is None:
            missing_images += 1
            continue
        if not _valid_image(image_path):
            invalid_images += 1
            continue
        valid.append({"image": str(image_path), "question": item["question"].strip(), "answer": item["answer"], "split": item["split"], "source": "RSVQA", "question_id": item["question_id"], "image_id": item["image_id"]})
    if not valid:
        raise RuntimeError("No usable RS-VLM training examples were found. Check --dataset-root and RSVQA image/annotation availability.")
    limits = {"train": max_train_samples, "validation": max_validation_samples, "test": max_test_samples}
    if smoke_test:
        limits = {"train": 20, "validation": 5, "test": 5}
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        split_records = [record for record in valid if record["split"] == split]
        selected.extend(_limit_split(split_records, limits[split], rng))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    stats = {**annotation_stats, "total_annotation_records": len(annotations), "valid_records": len(valid), "missing_images": missing_images, "invalid_images": invalid_images, "train": sum(record["split"] == "train" for record in selected), "validation": sum(record["split"] == "validation" for record in selected), "test": sum(record["split"] == "test" for record in selected), "seed": seed}
    output.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def prepare_bigearthnet(output: Path, image_root: Path, limit: int | None = None) -> dict[str, int]:
    """Preserve the existing optional BigEarthNet preparation path."""
    from datasets import load_dataset
    dataset = load_dataset("BIFOLD-BigEarthNetv2-0/BigEarthNet.txt", split="all_data", streaming=True)
    records: list[dict[str, Any]] = []
    total = 0
    missing_images = 0
    invalid_records = 0
    for sample in dataset:
        total += 1
        record = _record_from_sample(sample, image_root)
        if record is None:
            if isinstance(sample.get("patch_id"), str):
                missing_images += 1
            else:
                invalid_records += 1
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise RuntimeError("No usable RS-VLM training examples were found. Check --image-root and dataset availability.")
    train, validation, test = split_records(records, validation_fraction=0.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record, split in [(r, "train") for r in train] + [(r, "validation") for r in validation] + [(r, "test") for r in test]:
            handle.write(json.dumps({**record, "split": split}, ensure_ascii=True) + "\n")
    stats = {"total_records": total, "valid_records": len(records), "missing_images": missing_images, "invalid_records": invalid_records, "train": len(train), "validation": len(validation), "test": len(test)}
    output.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def validate_manifest(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            missing = [key for key in ("image", "question", "answer") if not record.get(key)]
            image = Path(record.get("image", ""))
            if missing:
                errors.append(f"line {line_number}: missing {', '.join(missing)}")
            elif not image.is_file() or not _valid_image(image):
                errors.append(f"line {line_number}: invalid image: {image}")
            else:
                count += 1
    return count, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("rsvqa", "bigearthnet"), default="rsvqa")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        count, errors = validate_manifest(args.output)
        print(json.dumps({"valid_records": count, "errors": errors}, indent=2))
        raise SystemExit(1 if errors else 0)
    if args.dataset == "rsvqa":
        if args.dataset_root is None:
            parser.error("--dataset-root is required for --dataset rsvqa")
        stats = prepare_rsvqa(args.output, args.dataset_root, args.max_train_samples, args.max_validation_samples, args.max_test_samples, args.seed, args.smoke_test)
    else:
        if args.image_root is None:
            parser.error("--image-root is required for --dataset bigearthnet")
        stats = prepare_bigearthnet(args.output, args.image_root, args.limit)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
