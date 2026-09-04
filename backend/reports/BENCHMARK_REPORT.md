# SatQuery AI: Phase 9 Scientific Benchmark Report

**Benchmark Scope**: VALIDATED MULTI-SCENE BENCHMARK  
**Benchmark Status**: PHASE_9C_MULTI_SCENE_BENCHMARK_COMPLETE  
**Evaluation Date**: 2026-09-04 20:19:27  
**Dataset Version**: 1.1.0  
**ML Status**: DEFERRED (No ML training conducted or claimed)  

---

## 1. Scientific Disclosures & Benchmark Scope

> [!IMPORTANT]
> **VALIDATED MULTI-SCENE BENCHMARK: Evaluated N=2 scene pairs across 2 countries (Austria, India) against real ESA WorldCover derived reference rasters. Reference masks are designated strictly as derived_reference, not absolute ground truth.**

- **Reference Label Qualification**: ESA WorldCover 2020 (v100) and 2021 (v200) use different classification algorithms. Observed differences may include algorithm/version effects in addition to physical land-cover change. Reference masks are designated strictly as derived_reference, not absolute ground truth.
- **Status Distinction**: This milestone represents `PHASE_9C_MULTI_SCENE_BENCHMARK_COMPLETE`.
  Full comprehensive benchmark requires post-2021 independent reference data across all splits.
- **Locked Test Split**: Test-set scenes remain locked with zero threshold tuning applied.
- **Production Integrity**: Production thresholds and `/api/query` algorithms remain completely unaltered.

---

## 2. Sample Accounting

| Metric | Count | Description |
|:---|:---:|:---|
| **Candidate Examples** | 10 | Total multi-temporal scene pairs across all splits in manifest |
| **Materialized Examples** | 2 | Examples with ingested reference rasters on disk |
| **Validated Examples** | 2 | Materialized examples passing strict integrity & class checks |
| **Evaluated Examples** | 2 | Examples evaluated by the deterministic benchmark runner |
| **Evaluated Pixel Count** | 370,000 | Total non-invalid pixels evaluated |
| **Ignored Pixel Count** | 0 | Cloud, shadow, and nodata pixels excluded from evaluation |

---

## 3. Pixel-Level Metrics — Deterministic SatQuery

| Metric | Value |
|:---|:---:|
| **Overall Accuracy** | 0.7291 |
| **Balanced Accuracy** | 0.1250 |
| **Macro Precision** | 0.0911 |
| **Macro Recall** | 0.1250 |
| **Macro F1** | 0.1054 |
| **Macro IoU** | 0.0911 |

---

## 4. Region-Level Spatial Matching

Region matching performed at $\text{IoU} \ge 0.30$ with $\text{MMU} = 4$ pixels ($400\,\text{m}^2$).

| Example ID | Target Class | Region Precision | Region Recall | Region F1 | Detection Rate | Mean Centroid Error | Mean Area Error |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BENCH-VIE-WORLDCOVER-TRAIN-01 | urban_expansion | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.00 px | 0.0 m² |
| BENCH-MUM-WORLDCOVER-VAL-01 | urban_expansion | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.00 px | 0.0 m² |

---

## 5. Diagnostic Error Analysis

Primary Failure Mode: **MIXED_PIXEL**

| Diagnostic Category | Pixel Count | Percentage |
|:---|:---:|:---:|
| `URBAN_FALSE_POSITIVE_NDBI` | 0 | 0.0% |
| `VEGETATION_SEASONAL_EFFECT` | 0 | 0.0% |
| `WATER_EPHEMERAL_CHANGE` | 19 | 0.0% |
| `CLOUD_CONTAMINATION` | 0 | 0.0% |
| `SMALL_REGION_FILTERED` | 0 | 0.0% |
| `MIXED_PIXEL` | 90,090 | 89.9% |
| `INSUFFICIENT_TEMPORAL_EVIDENCE` | 8,045 | 8.0% |
| `CONFLICTING_INDICES` | 2,063 | 2.1% |
| `UNCATEGORIZED` | 0 | 0.0% |

---

## 6. Baseline Comparison

| Pipeline | Macro F1 | Note |
|:---|:---:|:---|
| **No-Change Baseline** | 0.1206 | Predicts zero change across all pixels |
| **Deterministic SatQuery** | 0.1054 | Full production multi-index pipeline (Phases 5A–8) |

---

## 7. Next Steps for Full Phase 9 Benchmark

1. Materialize additional post-2021 geographic reference pairs across Queensland and other regions once reference rasters are verified.
2. Keep locked TEST split evaluated only once reference labels are independently validated.
3. Machine Learning remains DEFERRED until a large multi-scene reference dataset is established.