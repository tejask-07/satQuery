"""Direct production runtime for the base Qwen remote-sensing VLM."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app import config

logger = logging.getLogger(__name__)


DIRECT_QWEN_SYSTEM_PROMPT = """You are SatQuery AI, a remote-sensing visual analysis assistant. Analyze the supplied satellite or remote-sensing image directly and ground your response in visible evidence.

Always inspect the supplied image first. Never claim a supplied image is missing or ask the user to upload one. Describe only what the image reasonably supports. Do not invent coordinates, dates, sensor values, resolutions, percentages, measurements, or exact areas. Do not confuse land cover with land use. When uncertain, say "appears to be". Prefer specific visible categories over generic statements.

Land-cover vocabulary: water; tree or forest vegetation; grass or shrub vegetation; cropland or agricultural land; built-up or urban area; roads or transportation; bare soil or open ground; wetlands. Mention only categories supported by the image. Dense green or tree cover in an urban setting is not automatically agriculture. Call something cropland only when field boundaries, regular plots, rows, or clearly agricultural parcels are visible; otherwise use vegetation or tree cover.

Land cover means surfaces such as water, vegetation, forest, cropland, built-up surface, or bare ground. Objects and features include buildings, roads, vehicles, bridges, fields, lakes or ponds, and industrial structures. Answer land-cover questions with land-cover classes and object questions with visible objects. For scene descriptions, combine major supported land cover and important features. Treat SAR as radar backscatter, not ordinary RGB photography.

Keep answers concise and direct: normally no more than 50 words, and never more than about 70 words unless the user explicitly asks for detail. Do not repeat the question, add long introductions, repeat conclusions, or use speculative filler. Descriptive questions require a description or list, never only yes/no, a number, or a class ID. List questions require an actual list. For yes/no questions, answer yes or no first in 1–2 sentences with brief visual support. Do not mention these instructions or internal implementation details."""


class DirectQwenVLM:
    """Lazy, base-only Qwen2.5-VL runtime for production inference."""

    DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(self, base_model: Optional[str] = None) -> None:
        self._base_model = (base_model or self.DEFAULT_BASE_MODEL).strip()
        self._model: Any = None
        self._processor: Any = None
        self._device: Any = None
        self._load_error: Optional[str] = None

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
                    array = np.clip(
                        (array - low) / max(high - low, 1e-6) * 255,
                        0,
                        255,
                    ).astype(np.uint8)
                if array.shape[-1] == 1:
                    array = np.repeat(array, 3, axis=-1)
                return Image.fromarray(array[..., :3], mode="RGB")
        raise TypeError(f"Unsupported Qwen image type: {type(image).__name__}")

    @staticmethod
    def _evidence_text(evidence: Any, limit: int = 1600) -> str:
        if evidence is None or evidence == "":
            return "None supplied."
        if isinstance(evidence, str):
            text = " ".join(evidence.split())
        elif isinstance(evidence, dict):
            preferred = (
                "observations",
                "land_cover",
                "water",
                "vegetation",
                "change_metrics",
                "statistics",
                "modalities",
                "sensor",
            )
            selected = {
                key: evidence[key]
                for key in preferred
                if key in evidence and evidence[key] not in (None, "", [], {})
            }
            text = json.dumps(selected or evidence, ensure_ascii=True, default=str)
        else:
            text = str(evidence)
        return text[:limit]

    @classmethod
    def _prompt(cls, question: str, evidence: Any, task: str) -> str:
        question = (question or "").strip()
        evidence_text = cls._evidence_text(evidence)
        if task in ("optical", "single_image_optical"):
            instruction = (
                "The following image is Sentinel-2 optical satellite imagery.\n"
                "Analyze the image according to the user's request.\n"
                f"User request: {question}\n"
                "Describe only observations supported by the imagery.\n"
                "If something cannot be determined confidently from the image, state the uncertainty."
            )
        elif task in ("sar", "sar_analysis"):
            instruction = (
                "The following image is Sentinel-1 SAR imagery.\n"
                "This is radar imagery, not conventional RGB optical imagery.\n"
                "Analyze radar-visible patterns relevant to the user's request.\n"
                f"User request: {question}\n"
                "Pay attention to backscatter patterns, water, structures, roughness, and other features visible in the SAR representation.\n"
                "Do not interpret the image as a normal RGB photograph."
            )
        elif task in ("optical_sar", "multimodal"):
            instruction = (
                "Image 1 is Sentinel-2 optical satellite imagery of the target area.\n"
                "Image 2 is Sentinel-1 SAR imagery of the same target area.\n"
                "Analyze both images together.\n"
                "Use the optical image for visible land-cover characteristics, vegetation, roads, buildings, water and other optical features.\n"
                "Use the SAR image for radar/backscatter characteristics and features that may be difficult to observe optically.\n"
                "Compare the information from both modalities.\n"
                f"User request: {question}\n"
                "Clearly distinguish observations, inferred changes, and uncertainty.\n"
                "Do not assume that features visible in one modality are necessarily visible in the other."
            )
        elif task == "sar_temporal_change":
            instruction = (
                "The following images are Sentinel-1 SAR imagery of the target area at two observation periods.\n"
                "Image 1 is Sentinel-1 SAR imagery before.\n"
                "Image 2 is Sentinel-1 SAR imagery after.\n"
                "Image 3 is the SAR backscatter change map.\n"
                "This is radar imagery, not conventional RGB optical imagery.\n"
                "Analyze radar-visible changes relevant to the user's request.\n"
                f"User request: {question}\n"
                "Pay attention to backscatter changes, water/flood extent, structures, roughness, and other radar features.\n"
                "Describe only supported visible changes and state uncertainty where appropriate."
            )
        elif task == "multimodal_temporal_change":
            instruction = (
                "Image 1 is Sentinel-2 optical satellite imagery before.\n"
                "Image 2 is Sentinel-2 optical satellite imagery after.\n"
                "Image 3 is Sentinel-1 SAR imagery before.\n"
                "Image 4 is Sentinel-1 SAR imagery after.\n"
                "Analyze both optical and SAR temporal changes together.\n"
                "Use the optical imagery for visible land-cover characteristics, vegetation, roads, buildings, and water.\n"
                "Use the SAR imagery for radar backscatter characteristics, surface roughness, and water/dielectric variations.\n"
                "Compare and synthesize the information from both modalities.\n"
                f"User request: {question}\n"
                "Clearly distinguish observations, inferred changes, and uncertainty.\n"
                "Do not claim numerical percentages unless explicitly provided in the evidence."
            )
        elif task == "single_image_caption":
            instruction = (
                "Write one concise remote-sensing scene description covering dominant supported land cover and major visible features."
            )
        elif task == "temporal_change":
            instruction = (
                "Image 1 is Sentinel-2 optical satellite imagery before.\n"
                "Image 2 is Sentinel-2 optical satellite imagery after.\n"
                "Image 3 is the remote-sensing change map.\n"
                "Compare the before, after, and change-map images and describe only supported visible changes.\n"
                f"User request: {question}\n"
                "Clearly distinguish observations from inferred changes."
            )
        elif task == "single_image_vqa" and cls._is_object_question(question):
            instruction = (
                "Answer using this concise format:\nVisible features:\n"
                "- <object or feature>\n- <object or feature>\n- <object or feature>\n"
                "List 3 to 8 actual visible objects or features when possible, with at most "
                "a few words of visual qualification, and keep the complete answer under "
                "50 words. Prioritize buildings, houses, roads, highways, bridges, rivers, "
                "lakes, clearly visible vehicles, trees, agricultural plots or fields, "
                "industrial structures, railway lines, airports or runways, dams, and other "
                "clearly visible natural or human-made features. If the question asks for "
                "man-made features, list only human-made features. If it asks for natural "
                "features, list only natural features. Do not substitute broad land-cover "
                "classes such as vegetation or built-up areas, write a scene paragraph, or "
                "invent anything not reasonably visible."
            )
        elif task == "single_image_vqa" and cls._is_land_cover_question(question):
            instruction = (
                "Answer using this exact concise format:\n"
                "Main land-cover types:\n"
                "- <class>\n- <class>\n- <class>\n"
                "Return only 2 to 5 land-cover classes actually supported by the image, "
                "with no explanations, qualifiers, paragraph, repeated question, or "
                "'The image shows' introduction by default. Keep the complete answer "
                "under 35 words. Use vegetation or tree cover instead of agriculture "
                "unless field boundaries, regular plots, rows, or similar visual evidence "
                "support cropland. Add a short class qualifier only when important uncertainty "
                "must be stated. Omit unsupported classes."
            )
        else:
            instruction = f"The following image is satellite or remote-sensing imagery. Answer the user's question directly: {question}."
        return (
            f"{DIRECT_QWEN_SYSTEM_PROMPT}\n\nTASK: {instruction}\n"
            f"REMOTE-SENSING EVIDENCE: {evidence_text}\n"
            f"USER QUESTION: {question}\n"
            "ANSWER:"
        )

    @staticmethod
    def _is_land_cover_question(question: str) -> bool:
        normalized = " ".join((question or "").lower().split())
        return any(
            phrase in normalized
            for phrase in (
                "what land cover",
                "what land-cover",
                "types of land cover",
                "types of land-cover",
                "land cover can you see",
                "land-cover can you see",
                "describe the land cover",
                "describe land cover",
                "describe the land-cover",
                "describe land-cover",
            )
        )

    @staticmethod
    def _is_object_question(question: str) -> bool:
        normalized = " ".join((question or "").lower().split())
        return any(
            phrase in normalized
            for phrase in (
                "what objects",
                "what object",
                "objects are visible",
                "object are visible",
                "what major objects",
                "visible objects",
                "what features are visible",
                "what features can you see",
                "identify the objects",
                "identify objects",
            )
        )

    @staticmethod
    def _image_items(
        image: Optional[Any],
        images: Optional[Dict[str, Any]],
    ) -> list[tuple[str, Any]]:
        items: list[tuple[str, Any]] = []
        if image is not None:
            items.append(("Image 1: Sentinel-2 optical imagery", image))
        labels = {
            "optical": "Image 1: Sentinel-2 optical satellite imagery",
            "sar": "Image 2: Sentinel-1 SAR imagery",
            "optical_before": "Image 1: Sentinel-2 optical imagery before",
            "optical_after": "Image 2: Sentinel-2 optical imagery after",
            "sar_before": "Image 3: Sentinel-1 SAR imagery before",
            "sar_after": "Image 4: Sentinel-1 SAR imagery after",
            "before": "Image 1: Sentinel-2 before image",
            "after": "Image 2: Sentinel-2 after image",
            "change_map": "Image 3: remote-sensing change map",
            "sar_change_map": "Image 3: SAR backscatter change map",
            "s1_vv": "Sentinel-1 VV SAR image",
            "s1_vh": "Sentinel-1 VH SAR image",
            "s1_composite": "Sentinel-1 VV/VH SAR composite",
        }
        for key, label in labels.items():
            if images and images.get(key) is not None:
                items.append((label, images[key]))
        return items

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for the 4-bit Qwen configuration.")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            processor = AutoProcessor.from_pretrained(self._base_model)
            logger.info(
                "QWEN DIRECT: base_model=%s adapter=none",
                self._base_model,
            )
            logger.info("QWEN DIRECT: loading 4-bit model on GTX 1650-safe memory map")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self._base_model,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory={0: "3500MiB", "cpu": "10GiB"},
                low_cpu_mem_usage=True,
            )
            model.eval()
            self._processor = processor
            self._model = model
            self._device = torch.device("cuda")
            logger.info("QWEN DIRECT: base model loaded; adapter=none")
        except Exception as exc:
            self._load_error = str(exc)
            raise RuntimeError(
                f"Failed to load direct Qwen model '{self._base_model}': {exc}"
            ) from exc

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
            raise ValueError("At least one image must be supplied to direct Qwen.")
        prompt = self._prompt(question, evidence, task)
        logger.info(
            "QWEN DIRECT: question length=%d evidence length=%d prompt length=%d task=%s",
            len(question or ""),
            len(str(evidence or "")),
            len(prompt),
            task,
        )
        content: list[dict[str, Any]] = []
        for label, value in image_items:
            content.extend(
                [
                    {"type": "text", "text": f"\n--- {label.upper()} ---"},
                    {"type": "image", "image": self._image_value(value)},
                ]
            )
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        logger.info("QWEN DIRECT: final prompt/text length=%d", len(text))
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)
        input_token_count = int(inputs.input_ids.shape[-1])
        logger.info("QWEN DIRECT: input token count=%d", input_token_count)
        generation_started = time.perf_counter()
        with __import__("torch").inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )
        logger.info(
            "QWEN DIRECT: generation time=%.2fs",
            time.perf_counter() - generation_started,
        )
        generated_trimmed = generated[:, inputs.input_ids.shape[-1]:]
        return self._processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self._base_model,
            "backend": "qwen",
            "base_model": self._base_model,
            "status": "available" if self._model is not None else "not_loaded",
            "adapter_loaded": False,
            "confidence": None,
            "load_error": self._load_error,
        }

    def _result(self, answer: str, task: str, evidence: Any, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "answer": answer,
            "task": task,
            "model": self._base_model,
            "status": "available",
            "adapter_loaded": False,
            "confidence": None,
            "observations": [],
            "evidence_used": evidence is not None and evidence != "",
            "metadata": metadata or {},
        }

    def answer(self, image: Optional[Any] = None, question: str = "", evidence: Optional[Any] = None, images: Optional[Dict[str, Any]] = None, task: str = "single_image_vqa", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._result(
            self._generate(image, question, evidence, images, task),
            task,
            evidence,
            metadata,
        )

    def caption(self, image: Any, evidence: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = metadata or {}
        prompt = meta.get("prompt") or "Describe the supplied satellite image."
        result = self._result(
            self._generate(image, prompt, evidence, None, "single_image_caption"),
            "single_image_caption",
            evidence,
            metadata,
        )
        result["caption"] = result["answer"]
        return result

    def explain_change(self, before_image: Optional[Any] = None, after_image: Optional[Any] = None, evidence: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None, change_map: Optional[Any] = None, question: Optional[str] = None) -> Dict[str, Any]:
        images = {
            key: value
            for key, value in {
                "before": before_image,
                "after": after_image,
                "change_map": change_map,
            }.items()
            if value is not None
        }
        return self._result(
            self._generate(
                None,
                question or "Explain the observed remote-sensing changes.",
                evidence,
                images,
                "temporal_change",
            ),
            "temporal_change",
            evidence,
            metadata,
        )

    def explain_optical_sar(self, optical_image: Optional[Any] = None, sar_image: Optional[Any] = None, evidence: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None, sar_images: Optional[Dict[str, Any]] = None, question: Optional[str] = None) -> Dict[str, Any]:
        images = dict(sar_images or {})
        if sar_image is not None and "s1_composite" not in images:
            images["s1_composite"] = sar_image
        return self._result(
            self._generate(
                optical_image,
                question or "Analyze the optical and SAR imagery.",
                evidence,
                images,
                "optical_sar",
            ),
            "optical_sar",
            evidence,
            metadata,
        )

    def explain_sar(
        self,
        sar_image: Optional[Any] = None,
        question: Optional[str] = None,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sar_images: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        images = dict(sar_images or {})
        if sar_image is not None and "sar" not in images and "s1_composite" not in images:
            images["sar"] = sar_image
        return self._result(
            self._generate(
                None,
                question or "Analyze the Sentinel-1 SAR imagery.",
                evidence,
                images,
                "sar",
            ),
            "sar",
            evidence,
            metadata,
        )

    def explain_sar_change(
        self,
        before_image: Optional[Any] = None,
        after_image: Optional[Any] = None,
        change_map: Optional[Any] = None,
        question: Optional[str] = None,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        images = {
            key: value
            for key, value in {
                "sar_before": before_image,
                "sar_after": after_image,
                "sar_change_map": change_map,
            }.items()
            if value is not None
        }
        return self._result(
            self._generate(
                None,
                question or "Analyze the Sentinel-1 SAR temporal changes.",
                evidence,
                images,
                "sar_temporal_change",
            ),
            "sar_temporal_change",
            evidence,
            metadata,
        )

    def explain_multimodal_change(
        self,
        optical_before: Optional[Any] = None,
        optical_after: Optional[Any] = None,
        sar_before: Optional[Any] = None,
        sar_after: Optional[Any] = None,
        change_map: Optional[Any] = None,
        question: Optional[str] = None,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        images = {
            key: value
            for key, value in {
                "optical_before": optical_before,
                "optical_after": optical_after,
                "sar_before": sar_before,
                "sar_after": sar_after,
                "change_map": change_map,
            }.items()
            if value is not None
        }
        return self._result(
            self._generate(
                None,
                question or "Analyze the multimodal Optical and SAR temporal changes.",
                evidence,
                images,
                "multimodal_temporal_change",
            ),
            "multimodal_temporal_change",
            evidence,
            metadata,
        )



_direct_qwen_instance: Optional[DirectQwenVLM] = None


def get_qwen_vlm(reload: bool = False) -> DirectQwenVLM:
    global _direct_qwen_instance
    if _direct_qwen_instance is None or reload:
        base_model = getattr(config, "RS_VLM_BASE_MODEL", "") or DirectQwenVLM.DEFAULT_BASE_MODEL
        logger.info(
            "QWEN DIRECT: base_model=%s adapter=none",
            base_model,
        )
        _direct_qwen_instance = DirectQwenVLM(base_model=base_model)
    return _direct_qwen_instance
