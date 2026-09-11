"""
Remote Sensing Vision-Language Model (RS-VLM) Runtime Abstraction.

Defines the core interface and factory for the SatQuery RS-VLM layer.
Decouples agent planning, specialist analysis, and API orchestration from
the specific underlying model implementation or checkpoint format.
"""

from __future__ import annotations

import logging
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from app import config

logger = logging.getLogger(__name__)


class RSVLM(ABC):
    """
    Abstract runtime interface for Remote Sensing Vision-Language Models.

    Every implementation must return structured dictionaries adhering to the
    standard SatQuery model output schema rather than raw strings.
    """

    @abstractmethod
    def answer(
        self,
        image: Optional[Any] = None,
        question: str = "",
        evidence: Optional[Any] = None,
        images: Optional[Dict[str, Any]] = None,
        task: str = "vqa",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute visual question answering on single or multi-modal imagery.
        """
        pass

    @abstractmethod
    def caption(
        self,
        image: Any,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate grounded descriptive caption for a single remote-sensing image.
        """
        pass

    @abstractmethod
    def explain_change(
        self,
        before_image: Optional[Any] = None,
        after_image: Optional[Any] = None,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        change_map: Optional[Any] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize natural-language explanation of detected temporal changes
        grounded in authoritative deterministic measurements and change maps.
        """
        pass

    @abstractmethod
    def explain_optical_sar(
        self,
        optical_image: Optional[Any] = None,
        sar_image: Optional[Any] = None,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sar_images: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Explain complementary optical reflectance and Sentinel-1 SAR backscatter
        features from co-registered raster pairs.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return runtime status and adapter metadata.
        """
        pass


_global_rs_vlm_instance: Optional[RSVLM] = None


def get_rs_vlm(backend: Optional[str] = None, reload: bool = False) -> RSVLM:
    """
    Resolve and return an RSVLM runtime instance according to backend configuration.

    Supported backends:
    - 'mock': Development stub (default, safe for testing without checkpoint or GPU)
    - 'local': Integration point for future trained checkpoint / LoRA adapter
    - 'hf': Legacy cloud Qwen VLM inference client (requires HF_TOKEN)
    """
    global _global_rs_vlm_instance

    if _global_rs_vlm_instance is not None and not reload and backend is None:
        return _global_rs_vlm_instance

    resolved_backend = (
        backend or getattr(config, "RS_VLM_BACKEND", "mock") or "mock"
    ).lower().strip()
    is_enabled = getattr(config, "RS_VLM_ENABLED", True)

    if not is_enabled:
        from app.vlm.mock_rs_vlm import MockRSVLM

        logger.info("[RS-VLM] Model is disabled by configuration (RS_VLM_ENABLED=False).")
        return MockRSVLM(disabled=True)

    if resolved_backend == "mock":
        from app.vlm.mock_rs_vlm import MockRSVLM

        instance = MockRSVLM()
    elif resolved_backend == "local":
        instance = LocalRSVLM(
            adapter_path=getattr(config, "RS_VLM_ADAPTER_PATH", ""),
            base_model=getattr(config, "RS_VLM_BASE_MODEL", ""),
        )
    elif resolved_backend in ("hf", "qwen"):
        try:
            from app.vlm.model import VLM
            # Legacy wrapper adapting existing VLM to RSVLM interface
            instance = _LegacyVLMAdapter()
        except Exception as exc:
            logger.warning(f"[RS-VLM] Failed to load legacy VLM adapter ({exc}); using MockRSVLM.")
            from app.vlm.mock_rs_vlm import MockRSVLM

            instance = MockRSVLM()
    else:
        logger.warning(f"[RS-VLM] Unknown backend '{resolved_backend}'; defaulting to MockRSVLM.")
        from app.vlm.mock_rs_vlm import MockRSVLM

        instance = MockRSVLM()

    if backend is None:
        _global_rs_vlm_instance = instance

    return instance


class LocalRSVLM(RSVLM):
    """Lazy local Qwen2.5-VL runtime with the trained PEFT LoRA adapter."""

    DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    DEFAULT_ADAPTER_PATH = "models/satquery-rs-vlm-30k"

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        base_model: Optional[str] = None,
    ) -> None:
        self._adapter_setting = (adapter_path or self.DEFAULT_ADAPTER_PATH).strip()
        self._base_model = (base_model or self.DEFAULT_BASE_MODEL).strip()
        self._model: Any = None
        self._processor: Any = None
        self._device: Any = None
        self._adapter_path: Optional[Path] = None
        self._load_error: Optional[str] = None

    @staticmethod
    def _project_paths() -> tuple[Path, Path]:
        backend_dir = Path(config.BACKEND_DIR).resolve()
        return backend_dir, backend_dir.parent

    def _resolve_adapter_path(self) -> Path:
        configured = Path(self._adapter_setting).expanduser()
        if configured.is_absolute():
            candidates = [configured]
        else:
            backend_dir, project_dir = self._project_paths()
            candidates = [backend_dir / configured, project_dir / configured]
            if configured.parts and configured.parts[0].lower() == "backend":
                candidates.insert(0, project_dir / configured)
        for candidate in candidates:
            if (candidate / "adapter_config.json").is_file():
                return candidate.resolve()
        searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
        raise FileNotFoundError(
            f"RS-VLM LoRA adapter was not found. Searched: {searched}"
        )

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from peft import PeftModel
            from accelerate import disk_offload
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            adapter_path = self._resolve_adapter_path()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = (
                torch.bfloat16
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float32
            )
            processor = AutoProcessor.from_pretrained(str(adapter_path))
            if device.type == "cuda":
                base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self._base_model,
                    torch_dtype=dtype,
                )
                base.to(device)
            else:
                offload_folder = Path(config.BACKEND_DIR) / "data" / "cache" / "rs_vlm_offload"
                offload_folder.mkdir(parents=True, exist_ok=True)
                base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self._base_model,
                    torch_dtype=dtype,
                    device_map={"": "cpu"},
                    low_cpu_mem_usage=True,
                    offload_state_dict=True,
                )
            model = PeftModel.from_pretrained(base, str(adapter_path))
            if device.type == "cpu":
                disk_offload(
                    model,
                    offload_dir=str(offload_folder),
                    execution_device=device,
                )
            model.eval()
        except Exception as exc:
            self._load_error = str(exc)
            raise RuntimeError(
                f"Failed to load local RS-VLM base model '{self._base_model}' "
                f"with LoRA adapter '{self._adapter_setting}': {exc}"
            ) from exc

        self._processor = processor
        self._model = model
        self._device = device
        self._adapter_path = adapter_path

    @staticmethod
    def _image_value(image: Any) -> Any:
        from PIL import Image
        import numpy as np

        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            array = np.asarray(image)
            if array.ndim == 2:
                finite = np.isfinite(array)
                values = array[finite]
                if values.size:
                    low, high = np.percentile(values, [2, 98])
                    if high > low:
                        array = np.clip((array - low) / (high - low), 0, 1) * 255
                array = np.nan_to_num(array, nan=0.0).astype(np.uint8)
                return Image.fromarray(array, mode="L").convert("RGB")
            if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (3, 4):
                array = np.transpose(array, (1, 2, 0))
            if array.ndim == 3:
                array = np.nan_to_num(array, nan=0.0)
                if array.dtype != np.uint8:
                    low, high = np.percentile(array, [2, 98])
                    array = np.clip((array - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
                if array.shape[-1] == 1:
                    array = np.repeat(array, 3, axis=-1)
                return Image.fromarray(array[..., :3], mode="RGB")
        raise TypeError(f"Unsupported RS-VLM image type: {type(image).__name__}")

    @staticmethod
    def _evidence_text(evidence: Any) -> str:
        if evidence is None:
            return "No deterministic evidence was supplied."
        if isinstance(evidence, str):
            return evidence
        try:
            return json.dumps(evidence, ensure_ascii=True, default=str)
        except (TypeError, ValueError):
            return str(evidence)

    @staticmethod
    def _prompt(question: str, evidence: Any, task: str) -> str:
        return (
            "You are SatQuery's trained remote-sensing vision-language model.\n"
            "Answer the requested task directly and concisely.\n"
            "Deterministic remote-sensing measurements in EVIDENCE are authoritative. "
            "Explain them faithfully and do not invent measurements, counts, areas, "
            "coordinates, dates, or sensor facts. If evidence is unavailable, say so.\n\n"
            f"TASK: {task}\n"
            f"QUESTION: {question or 'Describe the supplied remote-sensing imagery.'}\n\n"
            f"EVIDENCE:\n{LocalRSVLM._evidence_text(evidence)}"
        )

    @staticmethod
    def _image_items(image: Optional[Any], images: Optional[Dict[str, Any]]) -> list[tuple[str, Any]]:
        items: list[tuple[str, Any]] = []
        if image is not None:
            items.append(("satellite image", image))
        labels = {
            "before": "Sentinel-2 before image",
            "after": "Sentinel-2 after image",
            "change_map": "remote-sensing change map",
            "s1_vv": "Sentinel-1 VV SAR image",
            "s1_vh": "Sentinel-1 VH SAR image",
            "s1_composite": "Sentinel-1 VV/VH SAR composite",
        }
        for key, label in labels.items():
            if images and images.get(key) is not None:
                items.append((label, images[key]))
        return items

    def _generate(
        self,
        image: Optional[Any],
        question: str,
        evidence: Any,
        images: Optional[Dict[str, Any]],
        task: str,
    ) -> str:
        self._load_model()
        from qwen_vl_utils import process_vision_info

        image_items = self._image_items(image, images)
        if not image_items:
            raise ValueError("At least one image must be supplied to local RS-VLM.")
        content: list[dict[str, Any]] = []
        for label, value in image_items:
            content.extend([
                {"type": "text", "text": f"\n--- {label.upper()} ---"},
                {"type": "image", "image": self._image_value(value)},
            ])
        content.append({"type": "text", "text": self._prompt(question, evidence, task)})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        with __import__("torch").inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=512)
        prompt_length = inputs.input_ids.shape[-1]
        generated_trimmed = generated[:, prompt_length:]
        return self._processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": f"{self._base_model}+LoRA",
            "backend": "local",
            "base_model": self._base_model,
            "adapter_path": str(self._adapter_path or self._adapter_setting),
            "status": "available" if self._model is not None else "not_loaded",
            "adapter_loaded": self._model is not None,
            "confidence": None,
            "load_error": self._load_error,
        }

    def _result(self, answer: str, task: str, evidence: Any, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "answer": answer,
            "task": task,
            "model": f"{self._base_model}+LoRA({self._adapter_path or self._adapter_setting})",
            "status": "available",
            "adapter_loaded": True,
            "confidence": None,
            "observations": [],
            "evidence_used": evidence is not None,
            "metadata": metadata or {},
        }

    def answer(self, image: Optional[Any] = None, question: str = "", evidence: Optional[Any] = None, images: Optional[Dict[str, Any]] = None, task: str = "vqa", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._result(self._generate(image, question, evidence, images, task), task, evidence, metadata)

    def caption(self, image: Any, evidence: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from app.vlm.caption import CAPTION_PROMPT

        meta = metadata or {}
        modality = meta.get("modality", "unknown")
        question = meta.get("prompt") or CAPTION_PROMPT.format(modality=modality)
        return self._result(self._generate(image, question, evidence, None, "single_image_caption"), "single_image_caption", evidence, metadata)

    def explain_change(self, before_image: Optional[Any] = None, after_image: Optional[Any] = None, evidence: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None, change_map: Optional[Any] = None, question: Optional[str] = None) -> Dict[str, Any]:
        images = {key: value for key, value in {"before": before_image, "after": after_image, "change_map": change_map}.items() if value is not None}
        q = question or "Explain the observed remote-sensing changes between before and after imagery."
        return self._result(self._generate(None, q, evidence, images, "temporal_change"), "temporal_change", evidence, metadata)

    def explain_optical_sar(self, optical_image: Optional[Any] = None, sar_image: Optional[Any] = None, evidence: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None, sar_images: Optional[Dict[str, Any]] = None, question: Optional[str] = None) -> Dict[str, Any]:
        images = dict(sar_images or {})
        if sar_image is not None and "s1_composite" not in images:
            images["s1_composite"] = sar_image
        q = question or "Analyze the co-registered optical reflectance and SAR backscatter features."
        return self._result(self._generate(optical_image, q, evidence, images, "optical_sar"), "optical_sar", evidence, metadata)


class _LegacyVLMAdapter(RSVLM):
    """
    Adapter wrapping the legacy HuggingFace InferenceClient VLM under the RSVLM interface.
    """

    def __init__(self):
        from app.vlm.model import VLM, MODEL_ID
        self._vlm = VLM()
        self._model_id = MODEL_ID

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self._model_id,
            "backend": "hf",
            "status": "available",
            "adapter_loaded": False,
            "confidence": None,
        }

    def answer(
        self,
        image: Optional[Any] = None,
        question: str = "",
        evidence: Optional[Any] = None,
        images: Optional[Dict[str, Any]] = None,
        task: str = "vqa",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ans = self._vlm.generate(
            image=image,
            question=question,
            evidence=evidence,
            images=images,
        )
        return {
            "answer": ans,
            "task": task,
            "model": self._model_id,
            "status": "available",
            "adapter_loaded": False,
            "confidence": None,
            "observations": [],
            "evidence_used": bool(evidence),
            "metadata": metadata or {},
        }

    def caption(
        self,
        image: Any,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        from app.vlm.caption import CAPTION_PROMPT

        modality = (metadata or {}).get("modality", "unknown")
        prompt = CAPTION_PROMPT.format(modality=modality)
        return self.answer(
            image=image,
            question=prompt,
            evidence=evidence,
            task="single_image_caption",
            metadata=metadata,
        )

    def explain_change(
        self,
        before_image: Optional[Any] = None,
        after_image: Optional[Any] = None,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        change_map: Optional[Any] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        imgs = {}
        if before_image is not None:
            imgs["before"] = before_image
        if after_image is not None:
            imgs["after"] = after_image
        if change_map is not None:
            imgs["change_map"] = change_map

        q = question or "Explain the observed remote-sensing changes between before and after imagery."
        return self.answer(
            question=q,
            evidence=evidence,
            images=imgs,
            task="temporal_change",
            metadata=metadata,
        )

    def explain_optical_sar(
        self,
        optical_image: Optional[Any] = None,
        sar_image: Optional[Any] = None,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sar_images: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        imgs = dict(sar_images or {})
        q = question or "Analyze the co-registered optical reflectance and SAR backscatter features."
        return self.answer(
            image=optical_image,
            question=q,
            evidence=evidence,
            images=imgs,
            task="optical_sar",
            metadata=metadata,
        )
