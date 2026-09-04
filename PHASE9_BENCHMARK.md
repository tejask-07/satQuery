# SatQuery AI: Phase 9 Scientific Benchmark, Discovery & Evaluation Report

**Document Version**: 2.0.0  
**Phase**: Phase 9 (Final Core Phase)  
**Status**: **INFRASTRUCTURE COMPLETE — BENCHMARKING PENDING DATASET**  
**Core Scientific Question**:  
> *"How accurately can SatQuery detect and localize urban, vegetation, and water change compared with independent ground truth?"*

---

## 1. Epistemic Grounding & Scientific Boundary

> [!IMPORTANT]
> **Definitive Epistemic Disclosure**:
> **"Benchmark metrics measure agreement with selected reference labels. They do not establish absolute semantic truth."**
> 
> Remote sensing change analysis at 10m–20m pixel resolution inherently involves complex sub-pixel mixed signatures, phenological fluctuations, and sensor calibration variations. In strict adherence to Phase 9 scientific safeguards:
> 1. Candidate geographic regions/tiles are **locations only**, NOT ground truth.
> 2. No pseudo-ground-truth or synthetic labels are reported as real empirical performance.
> 3. Unmaterialized benchmark scenes are explicitly marked `status: "pending_reference_label"` with `ground_truth_path: null`.
> 4. Numerical metrics (precision, recall, F1, IoU, confusion matrices) are **not fabricated**.
> 5. Machine learning models are **DEFERRED** until an independently sourced, validated reference dataset is ingested.

---

## 2. Dataset Discovery Matrix

Prior to manifest creation, an exhaustive investigation of independent remote-sensing change-detection datasets was conducted:

| Dataset Name | Authoritative Source | Primary URL | License | Native Spatial Resolution | Temporal Coverage | Label Semantics | Supported CRS | Label Categorization | Suitable Classes | SatQuery Evaluation Suitability |
|:---|:---|:---|:---|:---:|:---:|:---|:---|:---:|:---|:---|
| **Dynamic World** | WRI & Google Cloud | [dynamicworld.app](https://dynamicworld.app/) | CC-BY-4.0 | 10m | 2015–present (NRT global) | 9 land-cover classes (water, trees, grass, flooded veg, crops, shrub, built, bare, snow) | UTM / EPSG:4326 | Reference / Derived (Deep ensemble on 24k sub-scenes) | Urban expansion/reduction, vegetation loss/gain, water loss/gain | **Very High** (Direct 10m Sentinel-2 alignment, global coverage) |
| **OSCD** | ONERA / IEEE GRSS | [ieee-dataport.org](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) | CC-BY-NC-SA-4.0 | 10m / 20m | 2015–2018 (24 global cities) | Binary change (0: no change, 1: urban/construction change) | Local UTM zones | **Authoritative** (Manual specialist delineation) | Urban expansion / construction | **High for Urban** (Specific to Sentinel-2; limited to binary urban) |
| **ESA WorldCover** | ESA & VITO | [esa-worldcover.org](https://esa-worldcover.org/) | CC-BY-4.0 | 10m | 2020 & 2021 (annual) | 11 land-cover classes | EPSG:4326 | Authoritative Reference Product | Annual macro land-cover transitions | **High for Annual Change** (Covers 2020–2021 only; lacks multi-month cadence) |
| **CLMS Urban Atlas Change** | EEA / Copernicus | [land.copernicus.eu](https://land.copernicus.eu/) | Copernicus Open Access | 10m / vector (0.25ha MMU) | 2012–2018, 2018–2021 | Cadastral land-use polygons | EPSG:3035 (ETRS89 / LAEA) | **Authoritative** (Official European cadastral mapping) | Urban expansion, densification | **High for European AOIs** (Vienna; requires reprojection to EPSG:4326) |
| **Hi-UCD** | Wuhan University | [github.com/ggsDing/Hi-UCD](https://github.com/ggsDing/Hi-UCD) | Academic Research | 0.1m–0.5m | 2017–2019 | 9 semantic urban change classes | Local Cartesian | Authoritative | Urban building/road change | **Unsuitable** (Aerial sensor; lacks Sentinel-2 multispectral/SWIR bands) |
| **SECOND** | Wuhan University | [captain-whu.com](http://www.captain-whu.com/project/SCD/) | Non-commercial | 0.5m–3.0m | Multi-temporal aerial | 6 land-cover categories | Local Cartesian | Authoritative | Land cover change | **Unsuitable** (Lacks geographic CRS georeferencing and SWIR bands) |
| **SpaceNet 7** | SpaceNet / Radiant Earth | [spacenet.ai](https://spacenet.ai/sn7-challenge/) | CC-BY-SA-4.0 | 4m (PlanetScope) | 2018–2020 (24 months) | Building footprint tracking | UTM WGS 84 | Authoritative | Building construction | **Moderate/Low** (PlanetScope 4-band; lacks Sentinel-2 SWIR band) |

---

## 3. Two-Stage Benchmark Engine Architecture

```
STAGE A: INFRASTRUCTURE (COMPLETE)
┌────────────────────────────────────────────────────────┐
│  • Manifest Schema & Split Leakage Validator          │
│  • Metric Engine (Pixel Confusion Matrix, F1, IoU)     │
│  • Region Matching Engine (Configurable IoU, MMU)      │
│  • Diagnostic Error Analysis Engine (8 Categories)     │
│  • Deterministic Baseline Runner & CLI                │
│  • API & Frontend Developer Inspection Interface      │
│  • ML Training Pipeline Architecture                   │
└────────────────────────────────────────────────────────┘
                           ↓
STAGE B: DATASET POPULATION & NUMERICAL BENCHMARKING (PENDING)
┌────────────────────────────────────────────────────────┐
│  • Ingestion of Validated Reference Rasters            │
│    (Dynamic World differencing / OSCD urban change)   │
│  • Materialization of ground_truth_path GeoTIFFs       │
│  • Test Split Evaluation & Metric Generation           │
│  • Empirical Validation-Only Sensitivity Selection     │
│  • ML Baseline Training & Model Serialization          │
└────────────────────────────────────────────────────────┘
```

---

## 4. Dataset-Specific Semantic Class Mappings

Semantic taxonomies from external datasets are **explicitly mapped**; unsupported mappings are rejected:

### A. Dynamic World Bi-Temporal Mapping Schema (`DynamicWorld_to_SatQuery_v1`)
- `(Trees/Grass/Crops/Bare -> Built)` $\rightarrow$ `Class 1: urban_expansion`
- `(Built -> Trees/Grass/Crops/Bare)` $\rightarrow$ `Class 2: urban_reduction`
- `(Trees -> Crops/Bare/Built)` $\rightarrow$ `Class 3: vegetation_loss`
- `(Crops/Bare -> Trees/Grass)` $\rightarrow$ `Class 4: vegetation_gain`
- `(Water -> Bare/Grass/Crops)` $\rightarrow$ `Class 5: water_loss`
- `(Bare/Grass/Crops -> Water)` $\rightarrow$ `Class 6: water_gain`
- `(Any -> Snow/Ice)` or `(Snow/Ice -> Any)` $\rightarrow$ `Class 8: invalid` (or seasonal flag)
- `(Class_t1 == Class_t2)` $\rightarrow$ `Class 0: no_change`

### B. OSCD Binary Mapping Schema (`OSCD_Binary_to_SatQuery_v1`)
- `0` $\rightarrow$ `Class 0: no_change`
- `1` $\rightarrow$ `Class 1: urban_expansion`
- `Classes 2..6`: Explicitly marked **UNSUPPORTED**; evaluated only across active classes `{0, 1}` without penalizing absent domains.

---

## 5. Configurable Region Matching & MMU Policy

- **MMU Filtering**: Removes candidate regions smaller than 4 contiguous pixels ($400\,\text{m}^2$).
- **Configurable Spatial IoU Threshold**: Rather than assuming an arbitrary fixed value, the region matching engine accepts configurable overlap thresholds ($\text{IoU} \in \{0.20, 0.30, 0.40, 0.50\}$).
- **Validation-Only Sensitivity Protocol**: Sensitivity to candidate threshold, deadband, MMU, and region IoU threshold is evaluated exclusively on the `VALIDATION` split prior to testing.

---

## 6. Machine Learning Protocol & Deferred Status

- **Architecture**: `RandomForestClassifier` (100 estimators, max depth 8, seed 42) and `LogisticRegression` on 21 explicit, physically interpretable spectral and spatial features.
- **Scientific Safeguard (Rule 6)**:
  `ML STATUS = DEFERRED`
  The model is deliberately **not trained** on unvalidated or self-derived rasters. ML fitting will execute strictly when real reference labels are materialized, train/val/test geographic isolation is confirmed, and sample size is verified.

---

## 7. Current Verification & Benchmark Status

```powershell
# Run benchmark engine
python -m app.evaluation.run_benchmark
```

**Output**:
```
============================================================
STAGE A INFRASTRUCTURE VERIFICATION COMPLETE
STATUS: Benchmark infrastructure is complete; numerical evaluation is pending validated reference labels.
ML STATUS: DEFERRED (No validated reference-labeled dataset is materialized)
============================================================
```

- **Manifest Validator**: `is_valid = True`, 8 candidate scenes, 0 errors, 0 split leakage.
- **Backend Regression**: **249 passed, 0 failures** in 82.55s.
- **Frontend Production Build**: **Passed (0 errors)** in 356ms.
- **Production Query Isolation**: `/api/query` verified 100% independent and operational with 0 benchmark dependencies.
