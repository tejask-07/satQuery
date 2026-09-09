"""
CDVQA (Change Detection Visual Question Answering) Benchmark Evaluator.

Evaluates multimodal temporal change question answering using bi-temporal
remote-sensing imagery pairs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.evaluation.benchmarks.interfaces import (
    BaseBenchmarkEvaluator,
    BenchmarkResult,
    BenchmarkSample,
)
from app.vlm.rs_vlm import RSVLM, get_rs_vlm


class CDVQAEvaluator(BaseBenchmarkEvaluator):
    """Evaluates change detection QA reasoning on CDVQA dataset splits."""

    def __init__(self):
        super().__init__(benchmark_name="CDVQA", task_name="change_detection_vqa")

    def load_samples(self, max_samples: Optional[int] = None) -> List[BenchmarkSample]:
        demo_samples = [
            BenchmarkSample(
                sample_id="CDVQA_001",
                benchmark="CDVQA",
                task="change_detection_vqa",
                question="What kind of change occurred between the two acquisitions?",
                ground_truth="New residential buildings were constructed on previously bare land.",
                metadata={"change_type": "urban_expansion", "split": "test"},
            ),
            BenchmarkSample(
                sample_id="CDVQA_002",
                benchmark="CDVQA",
                task="change_detection_vqa",
                question="Did the water reservoir area decrease?",
                ground_truth="yes",
                metadata={"change_type": "water_shrinkage", "split": "test"},
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
            resp = vlm.explain_change(
                question=sample.question,
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
            "exact_match": None,
            "f1_score": None,
            "change_type_accuracy": None,
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
                "CDVQA benchmark evaluation protocol is ready. Quantitative evaluation "
                "deferred until real model checkpoint is available."
            ),
            sample_outputs=sample_outputs,
        )
