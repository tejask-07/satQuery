from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.vlm.rs_vlm import LocalRSVLM


def test_local_runtime_is_lazy_and_uses_defaults():
    runtime = LocalRSVLM()

    assert runtime._model is None
    assert runtime._processor is None
    info = runtime.get_model_info()
    assert info["backend"] == "local"
    assert info["status"] == "not_loaded"
    assert info["adapter_loaded"] is False
    assert info["base_model"] == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_local_adapter_path_resolves_from_project_root(monkeypatch, tmp_path):
    adapter = tmp_path / "models" / "satquery-rs-vlm-30k"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "app.vlm.rs_vlm.LocalRSVLM._project_paths",
        staticmethod(lambda: (tmp_path / "backend", tmp_path)),
    )

    runtime = LocalRSVLM(adapter_path="models/satquery-rs-vlm-30k")

    assert runtime._resolve_adapter_path() == adapter.resolve()


def test_local_response_contract_and_multimodal_inputs_without_model(monkeypatch):
    runtime = LocalRSVLM(adapter_path="models/satquery-rs-vlm-30k")
    captured = {}

    def fake_generate(image, question, evidence, images, task):
        captured.update(
            image=image,
            question=question,
            evidence=evidence,
            images=images,
            task=task,
        )
        return "grounded local answer"

    monkeypatch.setattr(runtime, "_generate", fake_generate)
    evidence = {"metric": "NDVI", "mean_change": -0.12}
    before = Image.new("RGB", (8, 8), "green")
    after = Image.new("RGB", (8, 8), "brown")
    change = Image.new("L", (8, 8), 128)

    result = runtime.explain_change(
        before_image=before,
        after_image=after,
        change_map=change,
        evidence=evidence,
    )

    assert captured["task"] == "temporal_change"
    assert set(captured["images"]) == {"before", "after", "change_map"}
    assert captured["evidence"] == evidence
    assert result == {
        "answer": "grounded local answer",
        "task": "temporal_change",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct+LoRA(models/satquery-rs-vlm-30k)",
        "status": "available",
        "adapter_loaded": True,
        "confidence": None,
        "observations": [],
        "evidence_used": True,
        "metadata": {},
    }


def test_local_caption_preserves_supplied_prompt(monkeypatch):
    runtime = LocalRSVLM()
    captured = {}
    monkeypatch.setattr(
        runtime,
        "_generate",
        lambda image, question, evidence, images, task: captured.update(question=question) or "caption",
    )

    runtime.caption(
        Image.new("RGB", (8, 8)),
        metadata={"modality": "optical", "prompt": "existing wrapped caption prompt"},
    )

    assert captured["question"] == "existing wrapped caption prompt"