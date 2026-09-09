"""
VRSBench (Visual Remote Sensing Benchmark) Evaluator.

Evaluates high-resolution satellite image captioning, visual grounding,
and complex multi-task reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.evaluation.benchmarks.interfaces import (
    BaseBenchmarkEvaluator,
    BenchmarkResult,
    BenchmarkSample,
)
from app.vlm.rs_vlm import RSVLM, get_rs_vlm


class VRSBenchEvaluator(BaseBenchmarkEvaluator):
    """Evaluates multi-task captioning and visual reasoning on VRSBench."""

    def __init__(self):
        super().__init__(benchmark_name="VRSBench", task_name="captioning_and_reasoning")

    def load_samples(self, max_samples: Optional[int] = None) -> List[BenchmarkSample]:
        demo_samples = [
            BenchmarkSample(
                sample_id="VRS_CAP_001",
                benchmark="VRSBench",
                task="captioning",
                ground_truth="A commercial airport with several aircraft parked on the apron next to runways.",
                metadata={"type": "captioning", "split": "val"},
            ),
            BenchmarkSample(
                sample_id="VRS_VQA_002",
                benchmark="VRSBench",
                task="reasoning",
                question="What is the spatial relationship between the industrial facility and the coastal boundary?",
                ground_truth="The industrial facility is situated adjacent to the eastern coastline.",
                metadata={"type": "spatial_reasoning", "split": "val"},
            ),
        ]
        return demo_samples[:max_samples] if max_samples else demo_samples

    def evaluate(
        self,
        rsvlm: Optional[RSVLM] = None,
        max_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        vlm = rsvlm or get_rs_vlm()
        model_info = vlm.get_model_info()
        samples = self.load_samples(max_samples=max_samples)

        is_mock = model_info.get("status") in ("mock", "disabled")
        status = "mock_dry_run" if is_mock else "ready_for_evaluation"

        sample_outputs = []
        for sample in samples:
            if sample.task == "captioning":
                resp = vlm.caption(
                    image=object(),
                    metadata={"benchmark": self.benchmark_name, "sample_id": sample.sample_id},
                )
                output_text = resp.get("caption") or resp.get("answer")
            else:
                resp = vlm.answer(
                    question=sample.question or "",
                    task="vqa",
                    metadata={"benchmark": self.benchmark_name, "sample_id": sample.sample_id},
                )
                output_text = resp.get("answer")

            sample_outputs.append({
                "sample_id": sample.sample_id,
                "task": sample.task,
                "ground_truth": sample.ground_truth,
                "model_output": output_text,
                "status": resp.get("status"),
            })

        metrics = {
            "bleu_4": None,
            "meteor": None,
            "rouge_l": None,
            "cider": None,
            "vqa_accuracy": None,
            "status": "deferred_awaiting_trained_checkpoint",
        }

        return BenchmarkResult(
            benchmark=self.benchmark_name,
            task=self.task_name,
            status=status,
            num_samples_evaluated=len(samples),
            metrics=metrics,
            model_info=model_info,
            is_mock=is_mock,
            notes=(
                "VRSBench evaluation protocol is ready. NLP generation metrics (BLEU/CIDEr) "
                "deferred until model training checkpoint is provided."
            ),
            sample_outputs=sample_outputs,
        )
