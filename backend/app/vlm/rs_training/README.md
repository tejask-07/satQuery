# SatQuery RS-VLM LoRA

This package adapts one Qwen2.5-VL model for remote-sensing VQA, captioning, and general visual reasoning. It uses PEFT LoRA; it does not train a second task-specific model and it does not treat BigEarthNet retrieval as training.

## Hardware and model

The repository's configured hosted model is `Qwen/Qwen2.5-VL-72B-Instruct`. The default training configuration uses `Qwen/Qwen2.5-VL-3B-Instruct`, preserving the Qwen2.5-VL architecture. The 72B checkpoint is not practical in the audited environment: Python 3.14.6 is present, but Torch, CUDA, Transformers, PEFT, and Datasets are unavailable. Use a compatible Python 3.10-3.12 CUDA environment for training. 3B LoRA is a practical starting point; QLoRA requires separately verified bitsandbytes/CUDA compatibility.

Install optional dependencies from the repository root. Use Python 3.10-3.12; the audited Python 3.14 environment is not a supported training environment:

```powershell
python -m pip install -r backend/app/vlm/rs_training/requirements.txt
$env:PYTHONPATH = "backend"
```

## Dataset

RSVQA is the primary initial training dataset. The official project provides low-resolution and high-resolution image/question/answer triplets. Download the dataset manually from the official RSVQA project: https://rsvqa.sylvainlobry.com/ . The project links the official Zenodo RSVQA LR and HR releases. Automatic downloading is intentionally not implemented.

The preparation code discovers split files such as `*_questions.json`, `*_answers.json`, and `*_images.json` recursively. It joins question IDs to answers and image IDs, supports list and mapping JSON forms, preserves train/validation/test labels, and verifies every image with Pillow. RSVQA LR and HR require no separate loader because the image files are passed to Qwen's processor unchanged.

The common manifest contains `image`, `question`, `answer`, and `split`. BigEarthNet remains an optional future source. Its retrieval system remains in `app/vlm/bigearthnet/` and is independent from this training pipeline.

Prepare an RSVQA manifest:

```powershell
python -m app.vlm.rs_training.prepare_dataset `
  --dataset rsvqa `
  --dataset-root D:\datasets\RSVQA-LR `
  --output backend/data/rs_vlm/manifest.jsonl
```

Prepare a small deterministic smoke manifest:

```powershell
python -m app.vlm.rs_training.prepare_dataset `
  --dataset rsvqa `
  --dataset-root D:\datasets\RSVQA-LR `
  --output backend/data/rs_vlm/manifest_smoke.jsonl `
  --smoke-test `
  --seed 42
```

Or set explicit per-split limits:

```powershell
python -m app.vlm.rs_training.prepare_dataset `
  --dataset rsvqa `
  --dataset-root D:\datasets\RSVQA-LR `
  --output backend/data/rs_vlm/manifest.jsonl `
  --max-train-samples 1000 `
  --max-validation-samples 200 `
  --max-test-samples 200 `
  --seed 42
```

The command prints total annotation records, valid records, missing images, invalid images, and selected train/validation/test counts. Zero valid examples fails without writing a manifest. Preparing the dataset does not train the model.

## Train and resume

Edit `config.yaml` for the selected hardware, then run:

```powershell
python -m app.vlm.rs_training.train_lora --config backend/app/vlm/rs_training/config.yaml
```

To resume, set `training.resume_from_checkpoint` to an actual Trainer checkpoint path in a local override config and rerun the same command. The adapter is saved under `models/satquery-rs-vlm` with the PEFT files produced by Transformers/PEFT.

Starting values are rank 16, alpha 32, dropout 0.05, learning rate 2e-4, two epochs, batch size 1, gradient accumulation 8, bf16, cosine schedule, 3% warmup, AdamW, maximum sequence length 2048, and the Qwen attention/MLP projection modules listed in `config.yaml`. Verify target modules against the selected checkpoint before a real run.

The collator passes image tensors to Qwen2.5-VL and masks user/image prompt tokens with `-100`; only assistant answer tokens contribute to the supervised loss. `quantization_4bit` is disabled by default. Enable it only on a verified CUDA/bitsandbytes environment.

## Evaluation

Base and adapted generations use the same manifest:

```powershell
python -m app.vlm.rs_training.evaluate `
  --config backend/app/vlm/rs_training/config.yaml `
  --output backend/data/rs_vlm/base_vs_adapter.json
```

The evaluator reports `not_executed` when optional dependencies, images, model weights, or the adapter checkpoint are unavailable. It writes question, reference answer, prediction, and an exact-match metric only after actual generation. No training or evaluation was executed in the audited environment.

## Inference

The existing public API remains available. Set the adapter path to activate local RS-VLM inference:

```powershell
$env:SATQUERY_RS_VLM_ADAPTER_PATH = "models/satquery-rs-vlm"
$env:SATQUERY_RS_VLM_BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
```

Then existing callers continue to use:

```python
from app.vlm.model import VLM
answer = VLM().generate(image=image, question="Describe this remote-sensing image.")
```

Without `SATQUERY_RS_VLM_ADAPTER_PATH`, the existing hosted Hugging Face inference path is used. VQA and captioning continue to instantiate the same shared `VLM` abstraction.

## Current status

The repository contains an RS-VLM training and evaluation pipeline. No adapter is considered trained until the user runs the training command successfully. No training or model evaluation has been executed in the audited environment.
