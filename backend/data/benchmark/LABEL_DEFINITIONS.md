# Canonical Benchmark Label Definitions

This document establishes the canonical semantic classes and reference label conventions for **SatQuery AI Phase 9 Benchmarking & Evaluation**.

---

## 1. Class Taxonomy

The SatQuery benchmark operates on **8 canonical semantic classes** plus an explicit **invalid / nodata** class (Class 8).

| Class ID | Label Name | Semantic Category | Description | Primary Physical Drivers |
|:---|:---|:---|:---|:---|
| **0** | `no_change` | Stable / Unchanged | Pixels exhibiting spectral stability within natural sensor/phenological noise bounds. | $\Delta\text{NDVI} \approx 0$, $\Delta\text{NDBI} \approx 0$, $\Delta\text{NDWI} \approx 0$ |
| **1** | `urban_expansion` | Anthropogenic Growth | Conversion of vegetation, bare soil, or agricultural land into built-up structures, impervious surfaces, or dense infrastructure. | Strong $\uparrow\text{NDBI}$, $\downarrow\text{NDVI}$, $\uparrow\text{SWIR/Red}$ |
| **2** | `urban_reduction` | Anthropogenic Demolition / Greening | Demolition of built structures, urban de-paving, brownfield remediation, or conversion of built surface to vegetation. | Strong $\downarrow\text{NDBI}$, $\uparrow\text{NDVI}$ or bare soil transition |
| **3** | `vegetation_loss` | Environmental Degradation | Loss of biomass, tree canopy clearing, deforestation, agricultural harvesting, or severe drought desiccation. | Strong $\downarrow\text{NDVI}$, $\uparrow\text{Red}$, stable/$\uparrow\text{NDBI}$ |
| **4** | `vegetation_gain` | Environmental Greening | Natural reforestation, seasonal crop canopy emergence, wetland revegetation, or ecological restoration. | Strong $\uparrow\text{NDVI}$, $\downarrow\text{Red}$, $\downarrow\text{NDBI}$ |
| **5** | `water_loss` | Hydrological Recession | Shrinkage of open water bodies, reservoir drawdown, wetland drying, or ephemeral pond evaporation. | Strong $\downarrow\text{NDWI}$, $\uparrow\text{NIR}$, $\uparrow\text{SWIR}$ |
| **6** | `water_gain` | Hydrological Inundation | Expansion of lakes/rivers, reservoir filling, coastal storm surge, or severe surface flooding. | Strong $\uparrow\text{NDWI}$, $\downarrow\text{NIR}$, $\downarrow\text{SWIR}$ |
| **7** | `ambiguous` | Discordant / Complex Change | Pixels exhibiting conflicting spectral indices (e.g. concurrent high NDBI and high NDWI) or transient sensor artifacts. | Unresolved index evidence, contradictory trajectory |
| **8** | `invalid` | Unusable / Contaminated | Pixels affected by clouds, cloud shadows, snow/ice, saturated detectors, or nodata areas outside the AOI. | Scene Classification Layer (SCL) masks or nodata |

---

## 2. Invalid Pixel Handling & Masking Rules

1. **Non-Evaluation of Class 8**:
   Pixels labeled `invalid` (Class 8) or flagged by Scene Classification Layer (SCL) masks (`SCL in {0, 1, 3, 8, 9, 10, 11}`) are **strictly excluded** from all pixel-level and region-level evaluation metrics.
2. **Accounting Invariants**:
   $$\text{Total Pixels} = \text{evaluated\_pixel\_count} + \text{ignored\_pixel\_count}$$
   Invalid pixels must never be counted as True Negatives or False Negatives. They are reported transparently in `ignored_pixel_count`.
3. **Joint Masking**:
   If either the "before" acquisition or the "after" acquisition is contaminated by cloud/shadow/nodata at pixel $(x, y)$, the pixel is considered jointly invalid and assigned Class 8.

---

## 3. Epistemic Qualification: Reference vs Ground Truth

> [!IMPORTANT]
> **Scientific Disclosure**:
> Independent authoritative satellite products (e.g. Copernicus Land Monitoring Service, WorldCover, Dynamic World, authoritative cadastres) or derived multi-spectral consensus masks are designated **reference labels** or **weak/derived labels**, not absolute ground truth.
> 
> Remote sensing change analysis at 10m–20m resolution contains intrinsic sub-pixel mixed signatures and phenological nuances. All benchmark metrics measure **agreement with reference labels** rather than establishing infallible physical reality.

---

## 4. Minimum Mapping Unit (MMU) Policy

- Standard pixel evaluation considers individual $10\text{m} \times 10\text{m}$ pixels.
- Region-level evaluation applies an MMU threshold of **4 contiguous pixels** ($400\,\text{m}^2$) to remove salt-and-pepper noise.
- Candidate regions smaller than the MMU are flagged under the error taxonomy as `SMALL_REGION_FILTERED` rather than unaddressed errors.

---

## 5. Dataset-Specific Semantic Class Mappings

To ensure scientific defensibility, datasets with differing semantic taxonomies are **never forced** into the 8-class system without an explicit, documented mapping. Datasets with incompatible semantics are explicitly rejected.

### A. Dynamic World Bi-Temporal Mapping
- **Source Classes**: 0: Water, 1: Trees, 2: Grass, 3: Flooded vegetation, 4: Crops, 5: Shrub, 6: Built, 7: Bare ground, 8: Snow/ice.
- **Bi-Temporal Transition Mapping**:
  - `(Trees/Grass/Crops/Bare -> Built)` $\rightarrow$ `Class 1: urban_expansion`
  - `(Built -> Trees/Grass/Crops/Bare)` $\rightarrow$ `Class 2: urban_reduction`
  - `(Trees -> Crops/Bare/Built)` $\rightarrow$ `Class 3: vegetation_loss`
  - `(Crops/Bare -> Trees/Grass)` $\rightarrow$ `Class 4: vegetation_gain`
  - `(Water -> Bare/Grass/Crops)` $\rightarrow$ `Class 5: water_loss`
  - `(Bare/Grass/Crops -> Water)` $\rightarrow$ `Class 6: water_gain`
  - `(Any -> Snow/Ice)` or `(Snow/Ice -> Any)` $\rightarrow$ `Class 8: invalid` (or seasonal flag)
  - `(Class_t1 == Class_t2)` $\rightarrow$ `Class 0: no_change`

### B. OSCD (Onera Satellite Change Detection) Binary Mapping
- **Source Classes**: 0: No Change, 1: Change (Urban/Construction).
- **Mapping**:
  - `0` $\rightarrow$ `Class 0: no_change`
  - `1` $\rightarrow$ `Class 1: urban_expansion` (or general urban change)
  - **Classes 2–6**: Marked as **UNSUPPORTED** by this dataset. The metric engine will evaluate only active classes (`{0, 1}`) and not penalize SatQuery for absent classes.

### C. Unsupported Taxonomy Policy
- Datasets providing purely land-use economic designations (e.g. residential vs commercial zones without physical surface characteristics) are **strictly rejected**.
- Sub-meter datasets without spectral bands compatible with Sentinel-2 (e.g. RGB-only aerial orthophotos) are **strictly rejected**.

