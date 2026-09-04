# Optical–SAR Real-Data Evaluation and Benchmark Framework

## 1. Executive Summary

This framework provides a reproducible evaluation suite for assessing **Optical + SAR multimodal query answering** on **real remote-sensing imagery**, distinguishing **engineering correctness** from **model/answer quality**.

Rather than relying purely on synthetic unit-test arrays or ungrounded synthetic benchmarks, this evaluation directly tests real satellite acquisitions from the European Space Agency Copernicus constellation via the open **BigEarthNet-MM** benchmark.

---

## 2. Dataset Selection and Provenance

### Source Overview
- **Dataset**: BigEarthNet-MM (Multi-Modal BigEarthNet Benchmark; Sumbul et al., 2021)
- **License**: Community Data License Agreement – Permissive (CDLA-Permissive) / Copernicus Open Access Policy
- **Geographic Coverage**: Upper Austria / Bavaria border region (Sentinel-2 Tile `33UUP`)
- **Spatial Grid**: WGS 84 / UTM Zone 33N (`EPSG:32633`), native 10-meter pixel resolution (120 × 120 pixels per patch, covering 1.44 km²)

### Sensors & Acquisitions
1. **Optical**:
   - **Satellite**: Sentinel-2A MSI (Multi-Spectral Instrument)
   - **Product**: Level-2A Bottom-Of-Atmosphere (BOA) surface reflectance
   - **Scene ID**: `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57`
   - **Acquisition Timestamp**: `2017-06-13T10:10:31Z`
   - **Bands**: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)
2. **SAR**:
   - **Satellite**: Sentinel-1B C-band SAR
   - **Product**: Interferometric Wide (IW) swath Ground Range Detected High resolution (GRDH)
   - **Scene ID**: `S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57`
   - **Acquisition Timestamp**: `2017-06-12T16:58:09Z`
   - **Polarizations**: VV (vertical transmit / vertical receive) and VH (vertical transmit / horizontal receive)
   - **Temporal Delta**: 17.2 hours between radar and optical acquisitions

### Scientific Grounding & Alignment Policy
> [!IMPORTANT]
> **Spatial Alignment Definition**:
> In accordance with remote-sensing rigorous practice, the imagery is **"aligned to a common optical reference grid"**. Reprojection maps the rasters to identical bounding coordinates, CRS, and pixel dimensions. We do **NOT** claim sub-pixel co-registration without an empirical point-scatterer cross-correlation test.

---

## 3. Evaluation Cases and Categories

The manifest (`manifest.json`) defines 7 curated evaluation cases addressing core remote-sensing multimodal queries:

| Case ID | Category | Primary Focus |
| :--- | :--- | :--- |
| `case_001_urban` | Built-up / urban | Optical texture vs. SAR double-bounce backscatter; ambiguity between farm structures and dense tree clumps |
| `case_002_water` | Water | Optical water appearance vs. SAR specular reflection / low return |
| `case_003_vegetation` | Vegetation | Optical greenness vs. SAR volume scattering in canopy (cross-polarized VH) |
| `case_004_complementary` | Cross-modal complementary | Explicit comparative analysis of optical reflectance vs. SAR roughness/dielectric cues |
| `case_005_vv_vh_comparison` | VV/VH comparison | Analysis of co-polarization (VV surface return) vs. cross-polarization (VH volume return) |
| `case_006_ambiguous` | Ambiguous scene | Boundary uncertainty between pasture and forest; penalizes dogmatic answers and rewards epistemic humility |
| `case_007_cross_crs_alignment` | Cross-CRS alignment | Verifies real-data reprojection across EPSG:4326 and EPSG:32633 |

---

## 4. Evaluation Rubric (0–2 Scale)

Human evaluation scores each case from 0 to 2 across 7 criteria:

| Criterion | 0 (Absent / Incorrect) | 1 (Partially Correct) | 2 (Clearly Correct) |
| :--- | :--- | :--- | :--- |
| **A. Optical evidence usage** | Does not reference optical data or misidentifies features | Mentions optical features superficially | Accurately and clearly references optical reflectance, color, or visible textures |
| **B. SAR evidence usage** | Does not reference radar data or misinterprets SAR | Mentions radar backscatter superficially | Accurately references radar backscatter, roughness, or structure |
| **C. Multimodal reasoning** | Discusses only one modality in isolation | Mentions both but lacks comparative synthesis | Meaningfully combines optical and SAR evidence to answer the question |
| **D. Physical interpretation** | Treats radar as visible light or makes physically invalid claims | Mostly correct with minor conceptual imprecision | Physically sound interpretation of optical reflectance and microwave scattering |
| **E. Grounding** | Invents unsupported dB values, exact areas, or percentages | Minor ungrounded estimates without major fabrication | Strictly qualitative and grounded; no invented metrics |
| **F. Uncertainty** | Forces dogmatic or ungrounded conclusions | Tentative acknowledgment of limitations | Explicitly acknowledges visual ambiguity and sensor limits |
| **G. Relevance** | Off-topic or ignores question | Partially answers with tangential discussion | Directly, concisely, and completely answers the prompt |

### Detailed Human Review Scoring Guidelines

1. **Optical Evidence Usage (0–2)**:
   - **Score 0**: Optical imagery is ignored, or the answer makes false claims about visible appearance (e.g. claims a dark green forest is bright bare soil).
   - **Score 1**: Mentions optical appearance in generic terms (e.g., "the optical image shows green areas") without distinguishing textures, parcels, or specific reflectance patterns.
   - **Score 2**: Precisely references optical features (e.g. geometric field boundaries, visible tonal differences, canopy texture, or absence of cloud cover).

2. **SAR Evidence Usage (0–2)**:
   - **Score 0**: Radar imagery is ignored, or SAR return is treated as optical photography (e.g. "SAR shows a blue lake").
   - **Score 1**: Mentions backscatter or radar generally without distinguishing polarization or scattering mechanisms (e.g. "SAR is bright in some spots").
   - **Score 2**: Correctly identifies radar physical phenomena: specular reflection over flat water, volumetric scattering in vegetation canopy (elevated VH), double-bounce from vertical structures, or surface roughness in VV.

3. **Multimodal Reasoning (0–2)**:
   - **Score 0**: Answer discusses only one modality in isolation or offers completely disjoint paragraphs without cross-referencing.
   - **Score 1**: Mentions both modalities, but one merely repeats the other without highlighting complementary strengths or differences.
   - **Score 2**: Explicitly synthesizes modalities (e.g. "While optical greenness could indicate either crops or pasture, the high SAR VH volume scattering confirms a mature standing crop").

4. **Physical Interpretation (0–2)**:
   - **Score 0**: Confuses radar microwave backscatter with visible light spectrum or asserts physically impossible sensor capabilities (e.g. optical penetration through dense forest floor).
   - **Score 1**: Uses mostly correct physical concepts but displays minor terminology looseness (e.g., calling radar backscatter "reflection").
   - **Score 2**: Rigorous remote sensing physics: differentiates surface vs volume vs double-bounce scattering; correctly attributes brightness to roughness or dielectric properties.

5. **Grounding / Anti-Hallucination (0–2)**:
   - **Score 0**: Invents specific numerical metrics (e.g. "-14.2 dB", "63% coverage", "12.4 hectares", "5,200 pixels").
   - **Score 1**: Uses approximate qualifiers with minor unwarranted numeric precision (e.g., "around 50%").
   - **Score 2**: Strictly qualitative, grounded descriptions based solely on visible raster evidence without hallucinated quantitative values.

6. **Uncertainty Acknowledgment (0–2)**:
   - **Score 0**: Makes dogmatic, unwarranted assertions on ambiguous features (e.g., claiming a single bright pixel is definitely a residential house).
   - **Score 1**: Shows mild hesitation without explaining why the feature is ambiguous.
   - **Score 2**: Explicitly acknowledges limitations (e.g. "A isolated high backscatter point could represent either an agricultural outbuilding or an electrical transmission mast; resolution limits conclusive identification").

7. **Relevance (0–2)**:
   - **Score 0**: Off-topic, evades the question, or outputs boilerplate unrelated to the prompt.
   - **Score 1**: Answers part of the prompt but omits a key component (e.g. describes vegetation but ignores the requested water boundary).
   - **Score 2**: Directly, concisely, and fully addresses the question asked.

> [!CAUTION]
> **No Fabricated Benchmarks**: We strictly avoid reporting fabricated accuracy percentages (e.g. "92% accuracy"). Benchmark metrics must reflect actual human ratings or verified ground-truth tests.

---

## 5. Automated Safety and Quality Checks

The evaluation runner automatically scans generated answers for common failure modes:

1. **Unsupported Number Checks**:
   - Hallucinated dB backscatter values (e.g., `-14.5 dB` when not provided in evidence)
   - Invented percentages (e.g., `45% covered by trees`)
   - Invented exact land areas (e.g., `12 hectares`, `3.2 sq km`)
   - Hallucinated pixel counts (e.g., `14,400 pixels`)
   - Fabricated classification thresholds (e.g., `threshold of -12`)

2. **SAR / RGB Confusion Check**:
   - Detects attribution of visible colors directly to physical SAR backscatter (e.g. *"the SAR image is blue"*, *"radar reveals green vegetation"*).
   - Correctly exempts valid descriptions of false-color dual-polarization composite channel assignments (e.g. *"dual-polarization composite maps VV to red and VH to green"*).

3. **Modality Collapse Detection**:
   - `only_optical_reasoning`: Answer ignores SAR imagery completely.
   - `only_sar_reasoning`: Answer ignores Optical imagery completely.
   - `generic_landcover_answer`: Generic high-level answer lacking specific evidence from either sensor.
   - `no_crossmodal_synthesis`: Both modalities are mentioned, but without comparative or corroborative synthesis.

---

## 6. Controlled Modality Ablation Experiment

To empirically determine whether the VLM benefits from the multimodal pair or merely relies on optical imagery, the evaluator supports controlled modality ablation (`--run-comparison`):

1. **Multimodal Condition (Optical + SAR)**:
   - Optical RGB + SAR VV + SAR VH + Dual-pol false-color composite are delivered.
   - Measures cross-modal reasoning and complementary utilization.
2. **Optical-Only Condition**:
   - Optical RGB only is provided; radar imagery is omitted.
   - Evaluates surface reflectance, visible color, and visible texture without microwave backscatter.
3. **SAR-Only Condition**:
   - Radar backscatter (dual-polarization composite / VV) is provided; optical imagery is omitted.
   - Evaluates microwave roughness, canopy scattering, and dielectric properties without visible color.

Ablation outputs are serialized to `ablation_comparison.jsonl` for side-by-side inspection of modality delta.

---

## 7. How to Run the Evaluation

### Complete Evaluation with Modality Ablation
```powershell
python backend/evaluation/optical_sar_eval.py --manifest backend/evaluation/optical_sar/manifest.json --output-dir backend/evaluation/optical_sar/ --run-comparison
```

### Pipeline-Only Verification (No API / No Token Required)
```powershell
python backend/evaluation/optical_sar_eval.py --manifest backend/evaluation/optical_sar/manifest.json --output-dir backend/evaluation/optical_sar/ --no-inference
```

### Generated Artifacts
- `results.jsonl`: Structured evaluation records per case, including metadata, alignment details, and automated check results.
- `human_review.csv`: Review table populated with questions, answers, and empty rubric columns for manual review.
- `ablation_comparison.jsonl`: Side-by-side ablation outputs (Multimodal vs Optical-only vs SAR-only).
- `visualizations/`: Generated PNG representations (Optical RGB, SAR VV, SAR VH, and SAR dual-pol composite).

---

## 8. Human Review Workflow

1. Run the evaluation runner to generate `results.jsonl` and `human_review.csv`.
2. Inspect the corresponding visualizations in `backend/evaluation/optical_sar/visualizations/<case_id>/`.
3. Open `human_review.csv` in a spreadsheet editor or text editor.
4. For each case, grade criteria A through G on the 0–2 scale according to Section 4.
5. Compute average scores per criterion across all cases.
6. Summarize qualitative findings without reporting fabricated percentage accuracy figures.

---

## 9. Limitations and Known Constraints

1. **Single-Region Initial Benchmark**: The curated set currently represents Upper Austria (`33UUP`). Future iterations will expand to urban (e.g. Vienna) and coastal (e.g. Mumbai) regions as co-registered SAR archives become available.
2. **Alignment Policy**: Spatial alignment maps rasters to an identical optical reference grid (CRS, bbox, pixel dimensions). We do not claim sub-pixel co-registration without an empirical scatterer cross-correlation test.
3. **Qualitative Evaluation**: Natural-language multimodal QA inherently requires expert human review; automated checks are diagnostic flags, not absolute proof of correctness.
4. **Credential Availability**: When `HF_TOKEN` is not configured, the evaluator records `status = "inference_unavailable"` and records deterministic fallback summaries, explicitly distinguishing pipeline readiness from live model outputs.

