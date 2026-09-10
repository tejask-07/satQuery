"""Train one Qwen2.5-VL RS adapter with PEFT LoRA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from .dataset import QwenVLMCollator, load_records, split_records


def _require_training_dependencies() -> tuple[Any, ...]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "RS-VLM training requires the optional ML stack. Install "
            "backend/app/vlm/rs_training/requirements.txt in a compatible Python environment."
        ) from exc
    return LoraConfig, TaskType, get_peft_model, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments


def _limit_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    if limit < 0:
        raise ValueError("dataset sample limits must be non-negative")
    return records[:limit]


def _resolve_manifest(path: Path) -> Path:
    if path.is_absolute() or path.is_file():
        return path
    backend_root = Path(__file__).resolve().parents[3]
    relative_parts = path.parts[1:] if path.parts and path.parts[0].lower() == "backend" else path.parts
    return backend_root.joinpath(*relative_parts)


def _install_diagnostics(model: Any, trainer: Any) -> None:
    original_forward = model.forward

    def diagnostic_forward(*args: Any, **kwargs: Any) -> Any:
        import torch

        input_ids = kwargs.get("input_ids")
        labels = kwargs.get("labels")
        if input_ids is not None:
            print(f"[RS-VLM DIAG] input_ids shape: {tuple(input_ids.shape)}", flush=True)
        if labels is not None:
            valid_labels = labels[labels != -100]
            valid_range = (
                f"{int(valid_labels.min())}..{int(valid_labels.max())}"
                if valid_labels.numel()
                else "none"
            )
            print(
                f"[RS-VLM DIAG] labels shape: {tuple(labels.shape)}, valid labels: {valid_labels.numel()}, "
                f"valid range: {valid_range}, labels finite: {bool(torch.isfinite(labels.float()).all())}",
                flush=True,
            )
        non_finite_inputs = [
            name
            for name, value in kwargs.items()
            if hasattr(value, "is_floating_point")
            and value.is_floating_point()
            and not torch.isfinite(value).all()
        ]
        print(f"[RS-VLM DIAG] non-finite input tensors: {non_finite_inputs or 'none'}", flush=True)
        print("[RS-VLM DIAG] before forward", flush=True)
        outputs = original_forward(*args, **kwargs)
        print("[RS-VLM DIAG] after forward", flush=True)
        logits = getattr(outputs, "logits", None)
        if logits is not None:
            print(
                f"[RS-VLM DIAG] logits shape: {tuple(logits.shape)}, finite: {bool(torch.isfinite(logits).all())}, "
                f"min: {float(logits.amin())}, max: {float(logits.amax())}",
                flush=True,
            )
        print("[RS-VLM DIAG] before loss computation", flush=True)
        loss = getattr(outputs, "loss", None)
        if loss is not None:
            print(
                f"[RS-VLM DIAG] after loss computation: {loss.detach().item():.6f}, "
                f"finite: {bool(torch.isfinite(loss).all())}",
                flush=True,
            )
        else:
            print("[RS-VLM DIAG] after loss computation: no loss returned", flush=True)
        return outputs

    model.forward = diagnostic_forward
    original_backward = trainer.accelerator.backward

    def diagnostic_backward(loss: Any, **kwargs: Any) -> Any:
        print("[RS-VLM DIAG] before backward", flush=True)
        result = original_backward(loss, **kwargs)
        print("[RS-VLM DIAG] after backward", flush=True)
        return result

    trainer.accelerator.backward = diagnostic_backward

    from transformers import TrainerCallback

    class OptimizerDiagnostics(TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            optimizer = kwargs.get("optimizer")
            if optimizer is None:
                return control
            original_step = optimizer.step

            def diagnostic_step(*step_args: Any, **step_kwargs: Any) -> Any:
                print("[RS-VLM DIAG] before optimizer step", flush=True)
                result = original_step(*step_args, **step_kwargs)
                print("[RS-VLM DIAG] after optimizer step", flush=True)
                return result

            optimizer.step = diagnostic_step
            return control

    trainer.add_callback(OptimizerDiagnostics())


def train(config: dict[str, Any]) -> None:
    (
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoProcessor,
        QwenModel,
        Trainer,
        TrainingArguments,
    ) = _require_training_dependencies()
    import torch

    model_name = config["model"].get("model_name", config["model"].get("name"))
    if not model_name:
        raise ValueError("config.model.model_name is required")
    manifest = _resolve_manifest(Path(config["data"].get("manifest_path", config["data"].get("manifest"))))
    if not manifest:
        raise ValueError("config.data.manifest_path is required")
    output_dir = Path(config["training"]["output_dir"])
    records = load_records(manifest)
    if any(record.get("split") for record in records):
        train_records = [record for record in records if record.get("split") == "train"]
        eval_records = [record for record in records if record.get("split") == "validation"]
        test_records = [record for record in records if record.get("split") == "test"]
    else:
        train_records, eval_records, test_records = split_records(
            records,
            config["data"].get("validation_fraction", 0.1),
            config["data"].get("seed", config["training"].get("seed", 42)),
        )
    limits = {
        "train": config["data"].get("max_train_samples"),
        "validation": config["data"].get("max_validation_samples"),
        "test": config["data"].get("max_test_samples"),
    }
    if config["data"].get("smoke_test"):
        smoke_limits = {"train": 20, "validation": 5, "test": 5}
        limits = {
            split: limit if limit is not None else smoke_limits[split]
            for split, limit in limits.items()
        }
    train_records = _limit_records(train_records, limits["train"])
    eval_records = _limit_records(eval_records, limits["validation"])
    test_records = _limit_records(test_records, limits["test"])
    if not train_records or not eval_records:
        raise ValueError("Manifest must contain non-empty train and validation splits")

    processor = AutoProcessor.from_pretrained(model_name)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": "auto" if torch.cuda.is_available() else torch.float32,
    }
    if config["training"].get("quantization_4bit"):
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires a CUDA-enabled PyTorch installation.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install a Transformers build with bitsandbytes support for 4-bit QLoRA.") from exc
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("4-bit QLoRA was requested but bitsandbytes is not installed.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    model = QwenModel.from_pretrained(model_name, **model_kwargs)
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing_targets = [name for name in config["lora"]["target_modules"] if name not in module_names]
    if missing_targets:
        raise ValueError(f"LoRA target modules not found in {model_name}: {missing_targets}")
    lora = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class RecordsDataset:
        def __init__(self, items: list[dict[str, Any]]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.items[index]

    training = config["training"]
    if training["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        warmup_ratio=training["warmup_ratio"],
        optim=training["optimizer"],
        lr_scheduler_type=training["scheduler"],
        logging_steps=training["logging_steps"],
        save_steps=training["save_steps"],
        eval_steps=training["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=training["save_total_limit"],
        gradient_checkpointing=training["gradient_checkpointing"],
        remove_unused_columns=False,
        bf16=training["precision"] == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=training["precision"] == "fp16" and torch.cuda.is_available(),
        seed=training["seed"],
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=RecordsDataset(train_records),
        eval_dataset=RecordsDataset(eval_records),
        data_collator=QwenVLMCollator(
            processor,
            config["data"].get("max_seq_length", config["data"].get("max_length", 2048)),
        ),
    )
    if training.get("diagnostic_logging"):
        _install_diagnostics(model, trainer)
    report_metrics = training.get("report_metrics", False)
    if report_metrics:
        expected_steps = math.ceil(
            len(train_records) / training["per_device_train_batch_size"]
            / training["gradient_accumulation_steps"]
        ) * training["num_train_epochs"]
        print(
            f"[RS-VLM BENCHMARK] train_samples={len(train_records)} "
            f"expected_optimizer_steps={expected_steps}",
            flush=True,
        )
    train_result = trainer.train(
        resume_from_checkpoint=config["training"].get("resume_from_checkpoint")
    )
    if report_metrics:
        eval_metrics = trainer.evaluate()
        print(
            "[RS-VLM BENCHMARK] metrics="
            + json.dumps(
                {
                    "optimizer_steps": trainer.state.global_step,
                    "train_loss": train_result.metrics.get("train_loss"),
                    "train_runtime_seconds": train_result.metrics.get("train_runtime"),
                    "validation_loss": eval_metrics.get("eval_loss"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    (output_dir / "dataset_split.json").write_text(
        json.dumps({"train": len(train_records), "validation": len(eval_records), "test": len(test_records)}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to read the training config") from exc
    train(yaml.safe_load(args.config.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
