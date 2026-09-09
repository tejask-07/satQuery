"""
BigEarthNet Multi-label Land Cover Benchmark Evaluator.

Evaluates land-cover classification and retrieval verification capabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.evaluation.benchmarks.interfaces import (
    BaseBenchmarkEvaluator,
    BenchmarkResult,
    BenchmarkSample,
)
from app.vlm.rs_vlm import RSVLM, get_rs_vlm


class BigEarthNetEvaluator(BaseBenchmarkEvaluator):
    """Evaluates multi-label classification accuracy on BigEarthNet-S2 / BigEarthNet-MM."""

    def __init__(self):
        super().__init__(benchmark_name="BigEarthNet", task_name="multi_label_classification")

    def load_samples(self, max_samples: Optional[int] = None) -> List[BenchmarkSample]:
        # Reference dataset samples for dry run / protocol verification
        demo_samples = [
            BenchmarkSample(
                sample_id="BEN_S2_sample_01",
                benchmark="BigEarthNet",
                task="multi_label_classification",
                question="List the prominent land-cover classes visible in this Sentinel-2 patch.",
                ground_truth=["Coniferous forest", "Broad-leaved forest"],
                metadata={"split": "test", "patch_name": "S2A_MSIL2A_demo_01"},
            ),
            BenchmarkSample(
                sample_id="BEN_S2_sample_02",
                benchmark="BigEarthNet",
                task="multi_label_classification",
                question="List the prominent land-cover classes visible in this Sentinel-2 patch.",
                ground_truth=["Discontinuous urban fabric", "Industrial or commercial units"],
                metadata={"split": "test", "patch_name": "S2A_MSIL2A_demo_02"},
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
                question=sample.question or "What land cover classes are visible?",
                task="vqa",
                metadata={"benchmark": self.benchmark_name, "sample_id": sample.sample_id},
            )
            sample_outputs.append({
                "sample_id": sample.sample_id,
                "ground_truth": sample.ground_truth,
                "model_output": resp.get("answer"),
                "status": resp.get("status"),
            })

        # Do NOT fabricate metrics if real model or dataset is not evaluated
        metrics = {
            "macro_f1": None,
            "micro_f1": None,
            "mean_average_precision": None,
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
                "BigEarthNet benchmark interface is ready. Quantitative metrics deferred until "
                "the production RS-VLM checkpoint and test split rasters are mounted."
            ),
            sample_outputs=sample_outputs,
        )
