"""Evaluate base Qwen2.5-VL against the same model with an RS adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import load_records


def _resolve_backend_path(path: Path) -> Path:
    if path.is_absolute() or path.is_file():
        return path
    backend_root = Path(__file__).resolve().parents[3]
    relative_parts = path.parts[1:] if path.parts and path.parts[0].lower() == "backend" else path.parts
    return backend_root.joinpath(*relative_parts)


def adapter_available(adapter_path: Path) -> bool:
    return (adapter_path / "adapter_config.json").is_file() and any(
        (adapter_path / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def evaluate_manifest(
    manifest: Path,
    model_name: str,
    adapter_path: Path | None = None,
    limit: int | None = None,
    split: str = "test",
) -> dict[str, Any]:
    manifest = _resolve_backend_path(manifest)
    if adapter_path is not None:
        adapter_path = _resolve_backend_path(adapter_path)
    records = load_records(manifest)
    split_records = [record for record in records if record.get("split") == split]
    if split_records:
        records = split_records
    if limit is not None:
        records = records[:limit]
    if adapter_path is not None and not adapter_available(adapter_path):
        return {
            "status": "not_executed",
            "reason": f"adapter checkpoint not found: {adapter_path}",
            "model": model_name,
            "records": len(records),
        }
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        return {
            "status": "not_executed",
            "reason": f"optional ML stack unavailable: {exc}",
            "model": model_name,
            "records": len(records),
        }

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        dtype = "auto" if torch.cuda.is_available() else torch.float32
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
    except Exception as exc:
        return {
            "status": "not_executed",
            "reason": f"model loading failed: {exc}",
            "model": model_name,
            "records": len(records),
        }
    predictions = []
    for record in records:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(record["image"]).resolve())},
                {"type": "text", "text": record["question"]},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)
        generated = model.generate(**inputs, max_new_tokens=256)
        trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated)]
        answer = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        predictions.append({"question": record["question"], "reference": record["answer"], "prediction": answer})
    exact_matches = sum(
        item["reference"].strip().casefold() == item["prediction"].strip().casefold()
        for item in predictions
    )
    return {
        "status": "executed",
        "model": model_name,
        "adapter_loaded": adapter_path is not None,
        "records": len(predictions),
        "exact_match": exact_matches / len(predictions) if predictions else None,
        "predictions": predictions,
    }


def compare_base_and_adapter(config: dict[str, Any]) -> dict[str, Any]:
    adapter_path = _resolve_backend_path(Path(config["model"]["adapter_path"]))
    model_name = config["model"].get("model_name", config["model"].get("name"))
    manifest = _resolve_backend_path(Path(config["data"].get("manifest_path", config["data"].get("manifest"))))
    limit = config["data"].get("max_test_samples")
    base = evaluate_manifest(manifest, model_name, limit=limit)
    adapted = evaluate_manifest(
        manifest,
        model_name,
        adapter_path,
        limit=limit,
    )
    return {"base": base, "adapted": adapted, "comparison_status": "not_executed" if adapted["status"] != "executed" else "executed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adapter-only", action="store_true")
    args = parser.parse_args()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to read the evaluation config") from exc
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.adapter_only:
        manifest = Path(config["data"].get("manifest_path", config["data"].get("manifest")))
        adapter_path = Path(config["model"]["adapter_path"])
        result = evaluate_manifest(
            manifest,
            config["model"].get("model_name", config["model"].get("name")),
            adapter_path,
            limit=config["data"].get("max_test_samples"),
        )
    else:
        result = compare_base_and_adapter(config)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
