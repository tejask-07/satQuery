"""
Unit and integration tests for the RS-VLM runtime abstraction and MockRSVLM.

Validates:
1. Mock status reporting (status='mock', adapter_loaded=False, confidence=None)
2. Single-image VQA routing (query -> planner -> RSVLM -> structured result)
3. Single-image Captioning routing (image -> caption -> RSVLM)
4. Temporal change explanation with evidence preservation
5. Optical-SAR specialist reasoning with RS-VLM synthesis
6. Graceful failure / disabled behavior (deterministic fallback preserved)
7. Benchmark readiness interfaces (BigEarthNet, RSVQA, VRSBench, CDVQA)
"""

import pytest
from PIL import Image

from app.agent.executor import execute_plan
from app.agent.parser import parse_query
from app.agent.planner import create_execution_plan
from app.agent.registry import get_tool
from app.evaluation.benchmarks import (
    BigEarthNetEvaluator,
    CDVQAEvaluator,
    RSVQAEvaluator,
    VRSBenchEvaluator,
)
from app.schemas.query import QueryPlan, QueryRequest
from app.vlm.caption import run_caption
from app.vlm.mock_rs_vlm import MockRSVLM
from app.vlm.optical_sar import answer_optical_sar_question
from app.vlm.rs_vlm import RSVLM, get_rs_vlm
from app.vlm.test_optical_sar import _create_mock_aligned_result
from app.vlm.vqa import run_vqa


def _dummy_aligned_result() -> dict:
    """Helper creating a minimal valid aligned result for optical-sar specialist testing."""
    return _create_mock_aligned_result(has_vv=True, has_vh=True)



def test_mock_rs_vlm_status_and_contracts():
    """Verify MockRSVLM reports explicit mock status, adapter_loaded=False, and null confidence."""
    mock = MockRSVLM()
    info = mock.get_model_info()

    assert info["name"] == "mock-rs-vlm"
    assert info["status"] == "mock"
    assert info["adapter_loaded"] is False
    assert info["confidence"] is None

    ans = mock.answer(question="Is there vegetation?")
    assert ans["status"] == "mock"
    assert ans["model"] == "mock-rs-vlm"
    assert ans["adapter_loaded"] is False
    assert ans["confidence"] is None
    assert "[MOCK RS-VLM]" in ans["answer"]
    assert "production remote-sensing VLM is not available yet" in ans["answer"]


def test_single_image_vqa_routing_and_evidence():
    """Verify single-image VQA routes through RSVLM and incorporates backend evidence."""
    mock_img = Image.new("RGB", (64, 64), color="green")
    ev = {"primary_metric": "NDVI", "delta_mean": 0.45}

    result = run_vqa(
        image=mock_img,
        question="What is the dominant land cover?",
        modality="optical",
        evidence=ev,
    )

    assert result["task"] == "single_image_vqa"
    assert result["status"] == "mock"
    assert result["adapter_loaded"] is False
    assert result["confidence"] is None
    assert result["evidence_used"] is True
    assert "[MOCK RS-VLM]" in result["answer"]
    assert "OPTICAL" in result["answer"]


def test_captioning_routing():
    """Verify captioning routes through RSVLM abstraction and handles modalities."""
    mock_img = Image.new("L", (64, 64), color=128)

    sar_res = run_caption(
        image=mock_img,
        modality="sar",
        evidence={"sensor": "Sentinel-1"},
    )

    assert sar_res["task"] == "single_image_caption"
    assert sar_res["status"] == "mock"
    assert sar_res["adapter_loaded"] is False
    assert sar_res["confidence"] is None
    assert sar_res["evidence_used"] is True
    assert "[MOCK RS-VLM]" in sar_res["caption"]
    assert "radar backscatter" in sar_res["caption"].lower() or "sar" in sar_res["caption"].lower()


def test_planner_distinguishes_core_tasks():
    """Verify planner distinguishes single_image_vqa, captioning, temporal_change, and optical_sar."""
    plan_vqa = QueryPlan(task="single_image_vqa")
    plan_caption = QueryPlan(task="captioning")
    plan_temporal = QueryPlan(task="temporal_change")
    plan_sar = QueryPlan(task="optical_sar_analysis")

    assert create_execution_plan(plan_vqa) == ["single_image_vqa"]
    assert create_execution_plan(plan_caption) == ["captioning"]
    assert "rs_vlm" in create_execution_plan(plan_temporal)
    assert create_execution_plan(plan_sar) == ["optical_sar_analysis"]


def test_parser_detects_vqa_and_caption_intents():
    """Verify natural-language query parser detects caption and VQA intents."""
    req_caption = QueryRequest(query="Generate a caption for this satellite image")
    plan_caption = parse_query(req_caption)
    assert plan_caption.task == "captioning"

    req_vqa = QueryRequest(query="What is visible in this satellite image?")
    plan_vqa = parse_query(req_vqa)
    assert plan_vqa.task == "single_image_vqa"


def test_executor_executes_vqa_and_captioning_tools():
    """Verify agent executor executes single_image_vqa and captioning registered tools."""
    mock_img = Image.new("RGB", (32, 32), color="blue")
    vqa_tool = get_tool("single_image_vqa")
    caption_tool = get_tool("captioning")

    assert callable(vqa_tool)
    assert callable(caption_tool)

    context = {
        "image": mock_img,
        "query": "Is there a water body?",
        "modality": "optical",
    }
    exec_res = execute_plan(["single_image_vqa"], context=context)
    assert "single_image_vqa" in exec_res
    assert exec_res["single_image_vqa"]["status"] == "mock"


def test_temporal_change_explanation_evidence_preservation():
    """Verify temporal change explanation preserves numerical evidence without fabrication."""
    mock = MockRSVLM()
    ev = {
        "primary_metric": "NDVI",
        "net_change_pct": -14.2,
        "delta_mean": -0.158,
        "significant_change_fraction": 0.22,
    }

    resp = mock.explain_change(
        before_image=object(),
        after_image=object(),
        evidence=ev,
        question="Explain the observed changes.",
    )

    assert resp["task"] == "temporal_change"
    assert resp["status"] == "mock"
    assert resp["confidence"] is None
    assert resp["evidence_used"] is True
    # The response text must mention the evidence without hallucinating fake numbers
    assert "net_change_pct" in resp["answer"] or "-14.2" in resp["answer"] or "NDVI" in resp["answer"]
    assert "[MOCK RS-VLM]" in resp["answer"]


def test_optical_sar_specialist_with_rs_vlm():
    """Verify Optical-SAR specialist routes to RS-VLM runtime and preserves deterministic evidence."""
    aligned = _dummy_aligned_result()
    ev = {"backscatter_roughness": "elevated", "urban_candidate": True}

    res = answer_optical_sar_question(
        aligned_result=aligned,
        question="How do optical and radar backscatter complement each other in this scene?",
        evidence=ev,
    )

    assert res["success"] is True
    assert res["status"] == "mock"
    assert res["adapter_loaded"] is False
    assert res["evidence_used"] is True
    assert "[MOCK RS-VLM]" in res["answer"]
    assert "Optical-SAR Multimodal Synthesis (Stub)" in res["answer"]


def test_rsvlm_disabled_fallback_behavior():
    """Verify deterministic fallback when RS-VLM is disabled."""
    disabled_vlm = MockRSVLM(disabled=True)
    info = disabled_vlm.get_model_info()
    assert info["status"] == "disabled"

    ans = disabled_vlm.answer(question="Is water present?")
    assert ans["status"] == "disabled"
    assert ans["confidence"] is None
    assert "DISABLED" in ans["answer"]

    # When used in optical-sar, deterministic fallback is produced
    aligned = _dummy_aligned_result()
    res = answer_optical_sar_question(
        aligned_result=aligned,
        question="Describe the multimodal features.",
        vlm=disabled_vlm,
    )
    assert res["success"] is True
    assert "[MOCK RS-VLM]" in res["answer"] or "Optical" in res["answer"]


def test_benchmark_task_interfaces():
    """Verify benchmark evaluators initialize and execute dry-run with no fabricated metrics."""
    evaluators = [
        BigEarthNetEvaluator(),
        RSVQAEvaluator(),
        VRSBenchEvaluator(),
        CDVQAEvaluator(),
    ]

    for ev in evaluators:
        res = ev.dry_run()
        assert res.is_mock is True
        assert res.status == "mock_dry_run"
        assert res.num_samples_evaluated > 0
        assert res.metrics["status"] == "deferred_awaiting_trained_checkpoint"
        # Verify no fabricated accuracy or F1
        for k, v in res.metrics.items():
            if k != "status":
                assert v is None
