"""Smoke-test Qwen2.5-VL processor and one multimodal forward pass.

This module never applies LoRA, creates an adapter, or starts training.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "data/rs_vlm/manifest.jsonl"
REQUIRED_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "pillow": "PIL",
}
CORE_PACKAGES = {"torch", "transformers", "pillow"}
TRAINING_PACKAGES = {"peft", "datasets", "accelerate"}
OPTIONAL_PACKAGES = {"torchvision": "torchvision", "qwen-vl-utils": "qwen_vl_utils"}


def package_report() -> dict[str, str | None]:
    report: dict[str, str | None] = {}
    for display_name, import_name in {**REQUIRED_PACKAGES, **OPTIONAL_PACKAGES}.items():
        if importlib.util.find_spec(import_name) is None:
            report[display_name] = None
            continue
        distribution_name = "Pillow" if display_name == "pillow" else display_name
        try:
            report[display_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            report[display_name] = "importable (distribution version unavailable)"
    return report


def _load_first_record(manifest: Path) -> dict[str, Any]:
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                break
        else:
            raise RuntimeError(f"Manifest is empty: {manifest}")
    image = Path(record.get("image", ""))
    if not image.is_file():
        raise FileNotFoundError(f"Manifest image does not exist: {image}")
    record["image"] = str(image.resolve())
    return record


def _device_and_dtype(torch: Any) -> tuple[Any, Any]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype


def run_smoke_test(model_name: str, manifest: Path, skip_forward: bool = False) -> int:
    report = package_report()
    print("Package versions:")
    for name, version in report.items():
        print(f"  {name}: {version or 'MISSING'}")
    missing_core = [name for name in CORE_PACKAGES if report[name] is None]
    missing_training = [name for name in TRAINING_PACKAGES if report[name] is None]
    if missing_training:
        print("Training-only packages missing: " + ", ".join(missing_training))
    if missing_core:
        print("\nSmoke test blocked: missing core packages: " + ", ".join(missing_core))
        print("Install backend/app/vlm/rs_training/requirements.txt in a supported Python 3.10-3.12 environment.")
        return 2

    import torch
    from PIL import Image

    device, dtype = _device_and_dtype(torch)
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        print(f"\nQwen Transformers import failed: {type(exc).__name__}: {exc}")
        return 3

    try:
        record = _load_first_record(manifest)
        image_path = Path(record["image"])
        with Image.open(image_path) as source_image:
            source_image.verify()
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            print(f"image: {image_path}")
            print(f"TIFF mode converted to: {image.mode}, size: {image.size}")

        processor = AutoProcessor.from_pretrained(model_name)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": record["question"]},
            ],
        }]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        print(f"processor input_ids shape: {tuple(inputs['input_ids'].shape)}")
        if skip_forward:
            print("Processor smoke test passed; forward pass skipped by request.")
            return 0

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        model.to(device)
        model.eval()
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        print(f"forward logits shape: {tuple(outputs.logits.shape)}")
        print("Qwen2.5-VL processor/model smoke test passed. No training or adapter creation was performed.")
        return 0
    except Exception as exc:
        print(f"\nSmoke test failed: {type(exc).__name__}: {exc}")
        print("No training or adapter creation was performed.")
        return 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_smoke_test(args.model, args.manifest, args.skip_forward))


if __name__ == "__main__":
    main()