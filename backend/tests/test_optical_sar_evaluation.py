"""
Test Suite for Optical-SAR Real-Data Evaluation and Benchmark Validation (Step 9).

Covers:
  1. Valid manifest parsing
  2. Missing optical reference handling
  3. Missing SAR reference handling
  4. Invalid case handling
  5. Mocked evaluation runner execution
  6. Structured result serialization (results.jsonl, human_review.csv)
  7. Evaluation rubric criteria definition (0-2 scale)
  8. Automated safety checks (unsupported dB, percentages, areas, pixel counts)
  9. Modality collapse detection flags
  10. Real-data spatial alignment sanity checks
"""

import csv
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.optical_sar_eval import (
    RUBRIC_CRITERIA,
    check_modality_collapse,
    check_sar_rgb_confusion,
    check_unsupported_numbers,
    evaluate_case,
    load_manifest,
    resolve_case_path,
    run_evaluation,
    run_modality_ablation_comparison,
)


@pytest.fixture
def manifest_path() -> Path:
    backend_dir = Path(__file__).resolve().parents[1]
    p = backend_dir / "evaluation" / "optical_sar" / "manifest.json"
    if not p.exists():
        pytest.fail(f"Required manifest not found at: {p}")
    return p


@pytest.fixture
def manifest_dir(manifest_path: Path) -> Path:
    return manifest_path.parent


# ============================================================
# 1. MANIFEST PARSING
# ============================================================

def test_manifest_parsing_valid(manifest_path: Path):
    """Verify that the production manifest parses cleanly with all required fields."""
    data = load_manifest(manifest_path)

    assert "manifest_version" in data
    assert "evaluation_set" in data
    assert "dataset_provenance" in data
    assert "cases" in data
    assert len(data["cases"]) >= 6

    # Verify every case structure
    for case in data["cases"]:
        assert "case_id" in case
        assert "category" in case
        assert "optical" in case
        assert "sar" in case
        assert "question" in case
        assert "expected_focus" in case
        assert "source" in case
        assert isinstance(case["source"], dict)
        assert "dataset" in case["source"]


# ============================================================
# 2. MISSING FILE REFERENCES
# ============================================================

def test_missing_optical_reference(manifest_dir: Path):
    """Verify evaluator fails gracefully when optical raster reference is missing."""
    case = {
        "case_id": "test_missing_opt",
        "category": "vegetation",
        "optical": "cases/non_existent_optical_file_123.tif",
        "sar": "cases/austria_s1_sar.tif",
        "question": "Assess vegetation",
    }
    result = evaluate_case(case, manifest_dir, run_inference=False)

    assert result["status"] == "missing_optical_reference"
    assert len(result["errors"]) > 0
    assert any("Optical raster not found" in e for e in result["errors"])
    assert result["answer"] is None


def test_missing_sar_reference(manifest_dir: Path):
    """Verify evaluator fails gracefully when SAR raster reference is missing."""
    case = {
        "case_id": "test_missing_sar",
        "category": "vegetation",
        "optical": "cases/austria_s2_optical.tif",
        "sar": "cases/non_existent_sar_file_123.tif",
        "question": "Assess vegetation",
    }
    result = evaluate_case(case, manifest_dir, run_inference=False)

    assert result["status"] == "missing_sar_reference"
    assert len(result["errors"]) > 0
    assert any("SAR raster not found" in e for e in result["errors"])
    assert result["answer"] is None


# ============================================================
# 3. INVALID CASE HANDLING
# ============================================================

def test_invalid_case(tmp_path: Path, manifest_dir: Path):
    """Verify evaluator handles corrupt or non-raster files gracefully."""
    bad_opt = tmp_path / "corrupt_optical.tif"
    bad_opt.write_text("not a geotiff")
    sar_path = manifest_dir / "cases" / "austria_s1_sar.tif"

    case = {
        "case_id": "test_corrupt",
        "category": "urban",
        "optical": str(bad_opt),
        "sar": str(sar_path),
        "question": "Identify buildings",
    }
    result = evaluate_case(case, manifest_dir, run_inference=False)

    assert result["status"] == "invalid_pair"
    assert len(result["errors"]) > 0


# ============================================================
# 4. MOCKED EVALUATION RUNNER
# ============================================================

def test_mocked_evaluation_runner(manifest_dir: Path):
    """Verify evaluation runner with a mocked VLM produces full multimodal outputs."""
    mock_vlm = MagicMock()
    mock_vlm.is_available.return_value = True
    mock_vlm.generate.return_value = (
        "In the optical image, broad-leaved forest stands appear dark green, whereas the SAR cross-polarized "
        "VH channel exhibits strong volume scattering corresponding to canopy structure. Built-up double-bounce "
        "is absent, confirming rural forest and pasture land cover."
    )

    case = {
        "case_id": "mock_case_001",
        "category": "vegetation",
        "optical": "cases/austria_s2_optical.tif",
        "sar": "cases/austria_s1_sar.tif",
        "question": "Describe dominant vegetation patterns using both optical and SAR evidence.",
        "source": {"dataset": "BigEarthNet-MM"},
    }

    result = evaluate_case(case, manifest_dir, vlm=mock_vlm, run_inference=True)

    assert result["status"] == "success"
    assert "In the optical image" in result["answer"]
    assert "optical" in result["modalities"]
    assert "sar_vv" in result["modalities"]
    assert "sar_vh" in result["modalities"]
    assert result["visual_sanity"]["dimensions_match"] is True
    assert result["automated_checks"]["all_passed"] is True


# ============================================================
# 5. STRUCTURED RESULT SERIALIZATION
# ============================================================

def test_structured_result_serialization(tmp_path: Path, manifest_path: Path):
    """Verify that results.jsonl and human_review.csv serialize correctly without secrets."""
    mock_vlm = MagicMock()
    mock_vlm.is_available.return_value = True
    mock_vlm.generate.return_value = "Optical and SAR observations confirm mixed forest."

    run_evaluation(
        manifest_path=manifest_path,
        output_dir=tmp_path,
        vlm=mock_vlm,
        run_inference=True,
        save_visuals=False,
    )

    jsonl_file = tmp_path / "results.jsonl"
    csv_file = tmp_path / "human_review.csv"

    assert jsonl_file.exists()
    assert csv_file.exists()

    # Verify JSONL lines
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 6

    for line in lines:
        entry = json.loads(line)
        assert "case_id" in entry
        assert "question" in entry
        assert "answer" in entry
        assert "modalities" in entry
        assert "visual_sanity" in entry
        # Check no secret token escaped
        assert "hf_" not in line.lower()
        assert "token" not in entry.get("metadata", {})

    # Verify CSV headers and structure
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert "case_id" in headers
        assert "optical_usage_0_to_2" in headers
        assert "sar_usage_0_to_2" in headers
        assert "multimodal_reasoning_0_to_2" in headers
        assert "physical_correctness_0_to_2" in headers
        assert "grounding_0_to_2" in headers
        assert "uncertainty_0_to_2" in headers
        assert "relevance_0_to_2" in headers
        assert "review_notes" in headers


# ============================================================
# 6. RUBRIC FIELDS DEFINITION
# ============================================================

def test_rubric_fields():
    """Verify all 7 required rubric dimensions are defined with a 0-2 scale."""
    expected_criteria = [
        "optical_usage",
        "sar_usage",
        "multimodal_reasoning",
        "physical_correctness",
        "grounding",
        "uncertainty",
        "relevance",
    ]
    for crit in expected_criteria:
        assert crit in RUBRIC_CRITERIA
        item = RUBRIC_CRITERIA[crit]
        assert "name" in item
        assert "description" in item
        assert "scale" in item
        assert 0 in item["scale"]
        assert 1 in item["scale"]
        assert 2 in item["scale"]


# ============================================================
# 7. AUTOMATED SAFETY CHECKS (UNSUPPORTED NUMBERS)
# ============================================================

def test_safety_checks_unsupported_numbers():
    """Verify detection of hallucinated decibel values, exact areas, percentages, and pixel counts."""
    # Bad answer with fabricated metrics
    bad_answer = (
        "The SAR backscatter is -14.5 dB in the urban core. Approximately 45% of the scene "
        "is vegetated, covering 12 hectares (14400 pixels). The threshold of -10 separates water."
    )
    flags = check_unsupported_numbers(bad_answer)

    assert any("unsupported_db_claim" in f for f in flags)
    assert any("invented_percentage" in f for f in flags)
    assert any("invented_exact_area" in f for f in flags)
    assert any("pixel_count_claim" in f for f in flags)
    assert any("threshold_claim" in f for f in flags)

    # Clean qualitative answer should trigger zero flags
    clean_answer = (
        "Optical imagery exhibits high reflectance in agricultural clearings and deep green tones in forested areas. "
        "In the SAR channels, elevated cross-polarized VH return indicates volumetric canopy scattering, while co-polarized "
        "VV response shows moderate surface scattering across pastures."
    )
    clean_flags = check_unsupported_numbers(clean_answer)
    assert len(clean_flags) == 0


# ============================================================
# 8. MODALITY COLLAPSE DETECTION
# ============================================================

def test_modality_collapse_flags():
    """Verify accurate detection of optical-only, SAR-only, and collapsed responses."""
    question = "Use optical and SAR imagery together to identify land cover."

    # 1. Optical-only answer
    opt_only = "In the optical image, the visual colors show green vegetation and bright soil."
    res_opt = check_modality_collapse(opt_only, question)
    assert res_opt["mentions_optical"] is True
    assert res_opt["mentions_sar"] is False
    assert "only_optical_reasoning" in res_opt["flags"]

    # 2. SAR-only answer
    sar_only = "The radar backscatter shows strong microwave returns and high VV roughness."
    res_sar = check_modality_collapse(sar_only, question)
    assert res_sar["mentions_optical"] is False
    assert res_sar["mentions_sar"] is True
    assert "only_sar_reasoning" in res_sar["flags"]

    # 3. Generic answer mentioning neither
    generic = "This landscape features undulating terrain with scattered agricultural parcels."
    res_gen = check_modality_collapse(generic, question)
    assert "generic_landcover_answer" in res_gen["flags"]

    # 4. Multimodal answer with synthesis
    good_multimodal = (
        "Optical imagery displays green agricultural parcels, whereas the SAR cross-polarization (VH) "
        "shows elevated backscatter in the woodland. Combined, both modalities confirm mixed agricultural and forest cover."
    )
    res_good = check_modality_collapse(good_multimodal, question)
    assert res_good["mentions_optical"] is True
    assert res_good["mentions_sar"] is True
    assert res_good["mentions_cross_modal"] is True
    assert len(res_good["flags"]) == 0


# ============================================================
# 9. REAL DATA SPATIAL ALIGNMENT SANITY
# ============================================================

def test_real_data_alignment_sanity(manifest_dir: Path):
    """
    Verify spatial alignment on real BigEarthNet Sentinel-1 and Sentinel-2 data:
      optical width == SAR width
      optical height == SAR height
    and verify common reference grid metadata.
    """
    opt_file = manifest_dir / "cases" / "austria_s2_optical.tif"
    sar_file = manifest_dir / "cases" / "austria_s1_sar.tif"

    assert opt_file.exists(), f"Missing real optical file: {opt_file}"
    assert sar_file.exists(), f"Missing real SAR file: {sar_file}"

    case = {
        "case_id": "test_alignment_sanity",
        "category": "vegetation",
        "optical": str(opt_file),
        "sar": str(sar_file),
        "question": "Verify grid alignment",
    }
    result = evaluate_case(case, manifest_dir, run_inference=False)

    assert result["status"] == "pipeline_verified_no_inference"
    sanity = result["visual_sanity"]

    assert sanity["dimensions_match"] is True
    assert sanity["optical_dimensions"] == [120, 120]
    assert sanity["sar_dimensions"] == [120, 120]
    assert sanity["has_optical_rgb"] is True
    assert sanity["has_sar_vv"] is True
    assert sanity["has_sar_vh"] is True
    assert sanity["has_sar_composite"] is True

    # Check alignment metadata
    align_meta = result["alignment"]
    assert align_meta["reference"] == "optical"
    assert align_meta["alignment_policy"] == "aligned to a common optical reference grid"
    assert "32633" in align_meta["target_crs"]


# ============================================================
# 10. CROSS-CRS REAL-DATA ALIGNMENT (REPROJECTION)
# ============================================================

def test_cross_crs_real_data_alignment(manifest_dir: Path):
    """Verify real data reprojection when Optical is EPSG:4326 and SAR is EPSG:32633."""
    opt_reproj = manifest_dir / "cases" / "austria_s2_reprojected.tif"
    sar_file = manifest_dir / "cases" / "austria_s1_sar.tif"

    assert opt_reproj.exists()
    assert sar_file.exists()

    case = {
        "case_id": "test_cross_crs",
        "category": "cross_crs_alignment",
        "optical": str(opt_reproj),
        "sar": str(sar_file),
        "question": "Verify reprojection",
    }
    result = evaluate_case(case, manifest_dir, run_inference=False)

    assert result["status"] == "pipeline_verified_no_inference"
    sanity = result["visual_sanity"]
    assert sanity["dimensions_match"] is True
    assert sanity["optical_dimensions"] == [143, 96]
    assert sanity["sar_dimensions"] == [143, 96]
    assert result["alignment"]["reprojected"] is True
    assert "4326" in result["alignment"]["target_crs"]


# ============================================================
# 11. SAR / RGB CONFUSION DETECTION (STEP 10G)
# ============================================================

def test_sar_rgb_confusion_detection():
    """Verify detection of phrases confusing SAR backscatter with visible colors, and exemption of composites."""
    # Bad answer directly assigning visible colors to physical SAR backscatter
    confused_answer = (
        "In this scene, the SAR image is blue where water sits, and radar shows green vegetation in the fields. "
        "The red areas in SAR indicate built-up zones."
    )
    flags = check_sar_rgb_confusion(confused_answer)
    assert len(flags) >= 2
    assert any("the SAR image is blue" in f or "sar image is blue" in f.lower() for f in flags)
    assert any("radar shows green" in f or "sar shows green" in f.lower() for f in flags)

    # Valid answer discussing dual-polarization false-color composite encoding
    composite_answer = (
        "In the dual-polarization false-color composite, VV backscatter is mapped to the red channel and VH to green, "
        "revealing volumetric canopy scattering as distinct tones while open water shows low specular return."
    )
    exempt_flags = check_sar_rgb_confusion(composite_answer)
    assert len(exempt_flags) == 0

    # Clean qualitative answer without confusion
    clean_answer = (
        "Optical imagery shows high visible reflectance and distinct green hues in agricultural clearings. "
        "The SAR channel exhibits strong microwave backscatter corresponding to surface roughness and double bounce."
    )
    assert len(check_sar_rgb_confusion(clean_answer)) == 0


# ============================================================
# 12. CONTROLLED MODALITY ABLATION RUNNER (STEP 10K)
# ============================================================

def test_controlled_modality_ablation(manifest_dir: Path):
    """Verify run_modality_ablation_comparison isolates Optical vs SAR vs Multimodal with mocked VLM."""
    opt_file = manifest_dir / "cases" / "austria_s2_optical.tif"
    sar_file = manifest_dir / "cases" / "austria_s1_sar.tif"

    case = {
        "case_id": "test_ablation",
        "category": "vegetation",
        "optical": str(opt_file),
        "sar": str(sar_file),
        "question": "Assess vegetation canopy and structure.",
    }

    mock_vlm = MagicMock()
    mock_vlm.is_available.return_value = True

    def mock_generate(*args, **kwargs):
        question = kwargs.get("question", "")
        if not question and len(args) > 1:
            question = args[1]
        q = str(question).lower()
        if "optical remote sensing query" in q:
            return "Optical answer: Rich green canopy in visible bands, high NDVI signature."
        elif "sar radar remote sensing query" in q:
            return "SAR answer: Strong volumetric scattering in VH polarization, surface roughness in VV."
        else:
            return "Multimodal answer: Optical greenness correlates directly with elevated SAR VH canopy scattering."

    mock_vlm.generate.side_effect = mock_generate

    result = evaluate_case(case, manifest_dir, vlm=mock_vlm, run_inference=True, run_comparison=True)

    assert result["status"] == "success"
    ablation = result.get("ablation_comparison")
    assert ablation is not None
    assert "Optical answer" in ablation["optical_only_answer"]
    assert "SAR answer" in ablation["sar_only_answer"]
    assert "Multimodal answer" in ablation["multimodal_answer"]
    assert ablation["ablation_status"] == "comparison_executed"


# ============================================================
# 13. SEPARATION OF FALLBACK VS GENUINE VLM (STEP 10B/10J)
# ============================================================

def test_separation_of_fallback_vs_genuine_vlm(manifest_dir: Path):
    """
    Verify that when VLM is unavailable or falls back:
      - multimodal_input_delivered is True (rasters aligned and passed)
      - status is explicitly 'inference_unavailable'
      - fallback answer is prefixed clearly
    """
    opt_file = manifest_dir / "cases" / "austria_s2_optical.tif"
    sar_file = manifest_dir / "cases" / "austria_s1_sar.tif"

    case = {
        "case_id": "test_fallback_separation",
        "category": "urban",
        "optical": str(opt_file),
        "sar": str(sar_file),
        "question": "Detect built-up structures.",
    }

    # Pass failing VLM mock to trigger deterministic fallback
    mock_failing_vlm = MagicMock()
    mock_failing_vlm.generate.side_effect = RuntimeError("VLM service unavailable")
    result = evaluate_case(case, manifest_dir, vlm=mock_failing_vlm, run_inference=True)

    assert result["multimodal_input_delivered"] is True
    assert result["status"] == "inference_unavailable"
    assert "[INFERENCE UNAVAILABLE - FALLBACK RESPONSE]" in result["answer"]
    # Model should not claim genuine multimodal reasoning when inference was unavailable
    assert result["multimodal_reasoning_observed"] is None


# ============================================================
# 14. TOKEN SAFETY (STEP 10N)
# ============================================================

def test_token_safety(tmp_path: Path, manifest_path: Path, monkeypatch):
    """Ensure HF_TOKEN is never printed, logged, or serialized into evaluation outputs."""
    fake_token = "hf_SUPER_SECRET_TOKEN_DO_NOT_LEAK_12345"
    monkeypatch.setenv("HF_TOKEN", fake_token)

    mock_vlm = MagicMock()
    mock_vlm.is_available.return_value = True
    mock_vlm.generate.return_value = "Verified multimodal analysis."

    results = run_evaluation(
        manifest_path=manifest_path,
        output_dir=tmp_path,
        vlm=mock_vlm,
        run_inference=True,
        save_visuals=False,
    )

    # Check memory structures
    dumped_str = json.dumps(results)
    assert fake_token not in dumped_str

    # Check files on disk
    jsonl_text = (tmp_path / "results.jsonl").read_text(encoding="utf-8")
    assert fake_token not in jsonl_text

    csv_text = (tmp_path / "human_review.csv").read_text(encoding="utf-8")
    assert fake_token not in csv_text

