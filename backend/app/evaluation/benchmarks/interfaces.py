"""
Benchmark task interfaces and schemas for Remote Sensing Vision-Language Models.

Provides standard evaluation protocols for:
- BigEarthNet (multi-label land-cover classification and verification)
- RSVQA (remote sensing visual question answering)
- VRSBench (remote sensing visual reasoning, grounding, and captioning)
- CDVQA (change detection visual question answering)

Adheres strictly to scientific transparency:
- Does not fabricate benchmark scores.
- Clearly states status ('not_run', 'ready_for_evaluation', or 'mock_dry_run').
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.vlm.rs_vlm import RSVLM, get_rs_vlm


@dataclass
class BenchmarkSample:
    """Standard container for a single benchmark evaluation item."""
    sample_id: str
    benchmark: str
    task: str
    question: Optional[str] = None
    image_path: Optional[str] = None
    before_image_path: Optional[str] = None
    after_image_path: Optional[str] = None
    sar_image_path: Optional[str] = None
    ground_truth: Any = None
    modality: str = "optical"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Structured evaluation report for a benchmark evaluation pass."""
    benchmark: str
    task: str
    status: str  # "ready_for_evaluation", "mock_dry_run", "not_run", "completed"
    num_samples_evaluated: int
    metrics: Dict[str, Any]
    model_info: Dict[str, Any]
    is_mock: bool = True
    notes: str = ""
    sample_outputs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "task": self.task,
            "status": self.status,
            "num_samples_evaluated": self.num_samples_evaluated,
            "metrics": self.metrics,
            "model_info": self.model_info,
            "is_mock": self.is_mock,
            "notes": self.notes,
            "sample_outputs": self.sample_outputs,
        }


class BaseBenchmarkEvaluator(ABC):
    """Abstract base class for remote sensing benchmark evaluators."""

    def __init__(self, benchmark_name: str, task_name: str):
        self.benchmark_name = benchmark_name
        self.task_name = task_name

    @abstractmethod
    def load_samples(self, max_samples: Optional[int] = None) -> List[BenchmarkSample]:
        """Load benchmark evaluation samples."""
        pass

    @abstractmethod
    def evaluate(
        self,
        rsvlm: Optional[RSVLM] = None,
        max_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run evaluation of the benchmark against the provided RSVLM runtime."""
        pass

    def dry_run(self, rsvlm: Optional[RSVLM] = None) -> BenchmarkResult:
        """Perform a dry-run with a minimal sample subset to verify runtime readiness."""
        vlm = rsvlm or get_rs_vlm()
        return self.evaluate(rsvlm=vlm, max_samples=2)
