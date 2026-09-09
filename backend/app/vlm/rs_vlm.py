"""
Remote Sensing Vision-Language Model (RS-VLM) Runtime Abstraction.

Defines the core interface and factory for the SatQuery RS-VLM layer.
Decouples agent planning, specialist analysis, and API orchestration from
the specific underlying model implementation or checkpoint format.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
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
        # Integration point for future trained checkpoint
        adapter_path = getattr(config, "RS_VLM_ADAPTER_PATH", "")
        base_model = getattr(config, "RS_VLM_BASE_MODEL", "")
        logger.warning(
            f"[RS-VLM] 'local' backend requested with adapter_path='{adapter_path}', base_model='{base_model}', "
            "but real RS-VLM weights are pending training completion. Falling back to MockRSVLM with status indicator."
        )
        from app.vlm.mock_rs_vlm import MockRSVLM

        instance = MockRSVLM(
            name="satquery-rs-vlm-pending",
            status_note="Real trained checkpoint is pending integration.",
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
