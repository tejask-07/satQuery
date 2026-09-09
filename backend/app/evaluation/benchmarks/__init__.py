"""
Benchmark readiness task interfaces for Remote Sensing Vision-Language Models.
"""

from app.evaluation.benchmarks.bigearthnet_benchmark import BigEarthNetEvaluator
from app.evaluation.benchmarks.cdvqa_benchmark import CDVQAEvaluator
from app.evaluation.benchmarks.interfaces import (
    BaseBenchmarkEvaluator,
    BenchmarkResult,
    BenchmarkSample,
)
from app.evaluation.benchmarks.rsvqa_benchmark import RSVQAEvaluator
from app.evaluation.benchmarks.vrsbench_benchmark import VRSBenchEvaluator

__all__ = [
    "BaseBenchmarkEvaluator",
    "BenchmarkResult",
    "BenchmarkSample",
    "BigEarthNetEvaluator",
    "RSVQAEvaluator",
    "VRSBenchEvaluator",
    "CDVQAEvaluator",
]
