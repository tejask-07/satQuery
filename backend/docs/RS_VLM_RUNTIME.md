# Remote Sensing Vision-Language Model (RS-VLM) Runtime Documentation

> [!IMPORTANT]
> **The production RS-VLM checkpoint is not available yet.**
> The current backend uses a mock runtime so that agent orchestration, specialist analysis, evidence grounding, execution tracing, and tests can be developed independently of model training.

---

## 1. Overview & Architecture

The SatQuery backend decouples agent planning, deterministic remote sensing analyses, and multimodal reasoning from underlying model weights through the **RS-VLM Runtime Abstraction**.

```text
User Query / Upload Request
          ↓
     Query Parser (detects tasks: single_image_vqa, captioning, temporal_change, optical_sar)
          ↓
       Planner (generates ordered tool pipeline)
          ↓
  Deterministic Specialist Tools (NDVI/NDWI/NDBI, change detection, optical-SAR co-registration)
          ↓
  Authoritative Remote-Sensing Evidence (authoritative measurements, rasters, statistics)
          ↓
    RS-VLM Runtime Abstraction (`app.vlm.rs_vlm.RSVLM`)
          ↓
  MockRSVLM / Future Fine-Tuned Checkpoint (`app.vlm.mock_rs_vlm.MockRSVLM`)
          ↓
  Structured Response (auditable execution trace, model metadata, confidence=null)
```

---

## 2. RS-VLM Runtime Interface (`RSVLM`)

Defined in `app/vlm/rs_vlm.py`, the `RSVLM` base class standardizes multimodal remote-sensing inference methods:

```python
class RSVLM(ABC):
    @abstractmethod
    def answer(
        self,
        image: Optional[Any] = None,
        question: str = "",
        evidence: Optional[Any] = None,
        images: Optional[Dict[str, Any]] = None,
        task: str = "vqa",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def caption(
        self,
        image: Any,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def explain_change(
        self,
        before_image: Optional[Any] = None,
        after_image: Optional[Any] = None,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        change_map: Optional[Any] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def explain_optical_sar(
        self,
        optical_image: Optional[Any] = None,
        sar_image: Optional[Any] = None,
        evidence: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sar_images: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]: ...
```

### Standard Return Contract

Every method returns a structured dictionary:

```json
{
  "answer": "[MOCK RS-VLM] ...",
  "task": "single_image_vqa",
  "model": "mock-rs-vlm",
  "status": "mock",
  "adapter_loaded": false,
  "confidence": null,
  "observations": [
    "Observed surface optical reflectance representing visible land cover."
  ],
  "evidence_used": true,
  "metadata": {}
}
```

---

## 3. MockRSVLM Development Stub

Located at `app/vlm/mock_rs_vlm.py`:
- Clearly prefixes all output text with `[MOCK RS-VLM]`.
- Consumes real backend evidence (NDVI/NDWI/NDBI indices, pixel counts, area ratios, SAR polarizations) without inventing figures.
- Explicitly reports `status='mock'`, `adapter_loaded=False`, and `confidence=None`.
- Supports single-image VQA, image captioning, bi-temporal change explanation, and optical-SAR multimodal synthesis.

---

## 4. Evidence Grounding & Physical Safeguards

SatQuery enforces a strict physical remote-sensing evidence contract:
1. **Backend measurements are authoritative**: The model never fabricates NDVI, NDWI, NDBI, area, percentages, or dB values.
2. **Observation vs Inference**: Direct raster observations are strictly distinguished from tentative land-cover inferences.
3. **Radar vs Optical**: Sentinel-1 SAR is treated as microwave radar backscatter, not visible color photography. Display false-color composites are acknowledged as artificial display encodings rather than physical radar hues.
4. **No confidence fabrication**: Model confidence remains `null` until calibrated on actual trained checkpoint validation loss / log-probabilities.

---

## 5. Agent, Planner & Registry Integration

- **Query Parser (`app/agent/parser.py`)**: Detects `captioning`, `single_image_vqa`, and `temporal_change` intents in addition to existing change detection, transition, and index queries.
- **Planner (`app/agent/planner.py`)**:
  - `single_image_vqa` -> `["single_image_vqa"]`
  - `captioning` -> `["captioning"]`
  - `temporal_change` -> `["search_imagery", "calculate_temporal_ndvi", "calculate_temporal_ndwi", "calculate_temporal_ndbi", "detect_change", "rs_vlm"]`
  - `optical_sar_analysis` -> `["optical_sar_analysis"]`
- **Registry (`app/agent/registry.py`)**: Registers `single_image_vqa`, `captioning`, and `rs_vlm` in `TOOL_REGISTRY`.
- **Executor (`app/agent/executor.py`)**: Executes the tools sequentially, passing image rasters and evidence payloads to `RSVLM`.

---

## 6. Execution Trace & Auditable Summary

Every API response extends `AnalysisResult` with structured execution metadata:

```json
{
  "status": "success",
  "answer": "...",
  "confidence": null,
  "execution_trace": [
    "Identified bands: Before=['red', 'nir'], After=['red', 'nir']",
    "Computed temporal NDVI change detection",
    "Generated change map visualizations",
    "RS-VLM multimodal inference completed (status: mock)"
  ],
  "execution_summary": {
    "query": "Show vegetation change between 2021 and 2025",
    "task": "temporal_change",
    "steps": [
      {"tool": "calculate_temporal_ndvi", "status": "success"},
      {"tool": "detect_change", "status": "success"},
      {"tool": "rs_vlm", "status": "mock"}
    ],
    "evidence": [...],
    "model": {
      "name": "mock-rs-vlm",
      "status": "mock",
      "adapter_loaded": false
    }
  },
  "model": {
    "name": "mock-rs-vlm",
    "status": "mock",
    "adapter_loaded": false
  }
}
```

---

## 7. Configuration Reference

In `app/config.py` or `.env`:

| Variable | Default | Options | Description |
|---|---|---|---|
| `RS_VLM_BACKEND` | `mock` | `mock`, `local`, `hf` | Runtime backend selector |
| `RS_VLM_ENABLED` | `true` | `true`, `false` | Enable/disable RS-VLM layer |
| `RS_VLM_ADAPTER_PATH` | `""` | Path to weights | Directory containing future LoRA adapter weights |
| `RS_VLM_BASE_MODEL` | `""` | Model ID/Path | Base model identifier for future trained adapter |

---

## 8. Future Trained RS-VLM Integration Point

The training pipeline and weights are managed separately by the training team under:
- Training Pipeline: `backend/app/vlm/rs_training/`
- Checkpoint Storage: `models/satquery-rs-vlm/`

### How the Future Checkpoint Will Be Plugged In:
1. When weights are ready, place them in `models/satquery-rs-vlm/` or set `RS_VLM_ADAPTER_PATH=/path/to/adapter`.
2. Implement a local model loader class (e.g. `LocalRSVLM(RSVLM)` in `app/vlm/local_rs_vlm.py`) implementing `RSVLM`.
3. In `get_rs_vlm()`, instantiate `LocalRSVLM` when `RS_VLM_BACKEND="local"` and `adapter_loaded=True`.
4. Zero changes will be needed to query parsing, agent execution, evidence pipelines, API endpoints, or test contracts.

---

## 9. Benchmark Readiness

Standard benchmark task harnesses are prepared in `app/evaluation/benchmarks/`:
- **BigEarthNet (`BigEarthNetEvaluator`)**: Multi-label classification and land-cover verification.
- **RSVQA (`RSVQAEvaluator`)**: Remote-sensing visual question answering (low & high resolution).
- **VRSBench (`VRSBenchEvaluator`)**: Satellite image captioning and spatial reasoning.
- **CDVQA (`CDVQAEvaluator`)**: Change detection visual question answering on bi-temporal pairs.

Each evaluator provides `.evaluate()` and `.dry_run()` methods, reporting `status='mock_dry_run'` with all quantitative metrics as `None` (deferred) until the real trained model is evaluated.
