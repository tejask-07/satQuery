"""
RSVQA (Remote Sensing Visual Question Answering) Benchmark Evaluator.

Evaluates single-image visual question answering across land-cover, object presence,
and count questions on Sentinel-2 and aerial platforms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.evaluation.benchmarks.interfaces import (
    BaseBenchmarkEvaluator,
    BenchmarkResult,
    BenchmarkSample,
)
from app.vlm.rs_vlm import RSVLM, get_rs_vlm


class RSVQAEvaluator(BaseBenchmarkEvaluator):
    """Evaluates question-answering accuracy on RSVQA dataset splits."""

    def __init__(self):
        super().__init__(benchmark_name="RSVQA", task_name="remote_sensing_vqa")

    def load_samples(self, max_samples: Optional[int] = None) -> List[BenchmarkSample]:
        demo_samples = [
            BenchmarkSample(
                sample_id="RSVQA_LR_001",
                benchmark="RSVQA",
                task="remote_sensing_vqa",
                question="Is there a river visible in this image?",
                ground_truth="yes",
                metadata={"type": "presence", "split": "test"},
            ),
            BenchmarkSample(
                sample_id="RSVQA_LR_002",
                benchmark="RSVQA",
                task="remote_sensing_vqa",
                question="What is the dominant land cover?",
                ground_truth="forest",
                metadata={"type": "land_cover", "split": "test"},
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
            resp = vlm.answer(
                question=sample.question or "",
                task="vqa",
                metadata={"benchmark": self.benchmark_name, "sample_id": sample.sample_id},
            )
            sample_outputs.append({
                "sample_id": sample.sample_id,
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "model_output": resp.get("answer"),
                "status": resp.get("status"),
            })

        metrics = {
            "overall_accuracy": None,
            "presence_accuracy": None,
            "count_accuracy": None,
            "comparison_accuracy": None,
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
                "RSVQA benchmark protocol is ready. Exact evaluation accuracy deferred "
                "until real checkpoint is evaluated."
            ),
            sample_outputs=sample_outputs,
        )
