# Phase 9: Remote Sensing Change Detection Dataset Discovery

This document details the scientific investigation of candidate independent change-reference datasets for benchmarking SatQuery AI, conducted prior to finalizing the benchmark manifest.

---

## 1. Candidate Dataset Evaluation Matrix

### Dataset 1: Onera Satellite Change Detection (OSCD)
- **Dataset Name**: Onera Satellite Change Detection (OSCD)
- **Source**: ONERA (The French Aerospace Lab) / IEEE GRSS
- **URL**: [https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) (DOI: `10.21227/H2NP42`)
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 4.0)
- **Spatial Resolution**: 10m (B2, B3, B4, B8), 20m (B5, B6, B7, B8A, B11, B12), 60m (B1, B9, B10)
- **Temporal Coverage**: 2015–2018 (24 paired Sentinel-2 acquisitions globally)
- **Label Semantics**: Binary change (0 = No Change, 1 = Change). Focus on urban change, construction, earthworks, and infrastructure expansion.
- **CRS**: Local UTM projections corresponding to each city tile (WGS 84 / UTM zones).
- **Label Categorization**: **Authoritative** (hand-annotated and cross-verified by remote sensing specialists).
- **Available Classes**:
  - `0`: `no_change`
  - `1`: `urban_change` (construction/new buildings/demolition/clearing)
- **Suitability for SatQuery Evaluation**:
  - **High** for urban change localization and spatial contiguity validation on Sentinel-2.
  - **Limitations**: Binary labels only; does not separate vegetation loss/gain or hydrological fluctuations without external auxiliary masking.

---

### Dataset 2: Dynamic World
- **Dataset Name**: Dynamic World Near Real-Time Global Land Cover
- **Source**: World Resources Institute (WRI) & Google Cloud
- **URL**: [https://dynamicworld.app/](https://dynamicworld.app/) (DOI: `10.1038/s41597-022-01307-4`)
- **License**: Creative Commons Attribution (CC BY 4.0)
- **Spatial Resolution**: 10m native pixel resolution
- **Temporal Coverage**: 2015–present, global continuous coverage aligned with Sentinel-2 L2A tile grid
- **Label Semantics**: 9-class land cover probabilities and categorical labels:
  - `0`: `water`
  - `1`: `trees`
  - `2`: `grass`
  - `3`: `flooded_vegetation`
  - `4`: `crops`
  - `5`: `shrub_and_scrub`
  - `6`: `built`
  - `7`: `bare`
  - `8`: `snow_and_ice`
- **CRS**: UTM / EPSG:4326 (aligned with S2 L2A grid).
- **Label Categorization**: **Reference / Derived** (deep learning consensus trained on 24,000 hand-annotated 10m sub-scenes, cross-validated with global ground truth).
- **Available Classes for Change Detection (via bi-temporal transition differencing)**:
  - `built` emergence (`trees/grass/crops/bare` $\rightarrow$ `built`): Maps to `urban_expansion`
  - `built` loss (`built` $\rightarrow$ `bare/trees`): Maps to `urban_reduction`
  - `canopy` loss (`trees` $\rightarrow$ `crops/bare`): Maps to `vegetation_loss`
  - `canopy` gain (`crops/bare` $\rightarrow$ `trees`): Maps to `vegetation_gain`
  - `water` loss (`water` $\rightarrow$ `bare/grass`): Maps to `water_loss`
  - `water` gain (`bare/grass` $\rightarrow$ `water`): Maps to `water_gain`
- **Suitability for SatQuery Evaluation**:
  - **Very High**. Matches exact Sentinel-2 10m pixels, provides global coverage including all SatQuery regions (Vienna, Delhi, Queensland, Sundarbans), and directly supports urban, vegetation, and water change semantics.

---

### Dataset 3: ESA WorldCover 10m
- **Dataset Name**: ESA WorldCover (2020 & 2021)
- **Source**: European Space Agency (ESA) & VITO Remote Sensing
- **URL**: [https://esa-worldcover.org/](https://esa-worldcover.org/) (DOI: `10.5281/zenodo.5571936`)
- **License**: Creative Commons Attribution (CC BY 4.0)
- **Spatial Resolution**: 10m native
- **Temporal Coverage**: Annual global land cover products for 2020 and 2021
- **Label Semantics**: 11 land cover classes (10: Tree cover, 20: Shrubland, 30: Grassland, 40: Cropland, 50: Built-up, 60: Bare / sparse vegetation, 70: Snow and ice, 80: Permanent water bodies, 90: Herbaceous wetland, 95: Mangroves, 100: Moss and lichen).
- **CRS**: EPSG:4326 (geographic latitude/longitude).
- **Label Categorization**: **Authoritative Reference Product** (globally validated against statistical point samples with overall accuracy $\sim 74.4\%-76.7\%$).
- **Available Classes for Change Detection**:
  - Annual transition differencing between 2020 and 2021 products.
  - Class 50 built-up transitions, Class 10/20 vegetation transitions, Class 80 water transitions.
- **Suitability for SatQuery Evaluation**:
  - **High** for annual macro-changes between 2020 and 2021.
  - **Limitations**: Only spans 2020–2021 annual epochs; cannot provide monthly or multi-year 2022–2025 change references.

---

### Dataset 4: Copernicus Land Monitoring Service (CLMS) Urban Atlas Change
- **Dataset Name**: Urban Atlas Change (2012–2018) & High Resolution Layers (HRL)
- **Source**: European Environment Agency (EEA) / Copernicus
- **URL**: [https://land.copernicus.eu/en/products/urban-atlas/urban-atlas-change-2012-2018](https://land.copernicus.eu/en/products/urban-atlas/urban-atlas-change-2012-2018)
- **License**: Copernicus Open Access Policy (free, full, and open)
- **Spatial Resolution**: MMU of 0.25 ha for urban change, 1 ha for rural change (vector and 10m raster).
- **Temporal Coverage**: 2012 to 2018; 2018 to 2021 HRL updates.
- **Label Semantics**: Delineated polygon land-use classes: Artificial surfaces (continuous/discontinuous urban fabric, industrial, roads), Green urban areas, Agricultural areas, Forests, Water bodies.
- **CRS**: EPSG:3035 (ETRS89 / LAEA Europe).
- **Label Categorization**: **Authoritative Official Delineation** (cadastral mapping based on VHR satellite imagery and national registers).
- **Suitability for SatQuery Evaluation**:
  - **High** for European cities (e.g. Vienna AOI `T33UXP`), offering rigorous cadastre-level urban change reference.
  - **Limitations**: Restricted to European countries; requires reprojection from EPSG:3035 to EPSG:4326 and temporal alignment with available Sentinel-2 scenes.

---

### Dataset 5: High-Resolution Urban Change Detection (Hi-UCD)
- **Dataset Name**: Hi-UCD
- **Source**: Wuhan University
- **URL**: [https://github.com/ggsDing/Hi-UCD](https://github.com/ggsDing/Hi-UCD)
- **License**: Open for academic and research use
- **Spatial Resolution**: 0.1m–0.5m aerial imagery
- **Temporal Coverage**: 2017–2019
- **Label Semantics**: 9 semantic urban change classes (buildings, greenhouses, roads, water, bare land, etc.).
- **CRS**: Local projected coordinate systems.
- **Label Categorization**: **Authoritative** (manual digitization from aerial orthophotos).
- **Suitability for SatQuery Evaluation**:
  - **Unsuitable** for direct Sentinel-2 evaluation. Sub-meter aerial imagery lacks Sentinel-2 multispectral bands (SWIR-1, SWIR-2, RedEdge) and has a resolution mismatch ($0.1\text{m}$ vs $10\text{m}$).

---

### Dataset 6: Semantic Change Detection Dataset (SECOND)
- **Dataset Name**: SECOND
- **Source**: Wuhan University / State Key Laboratory of LIESMARS
- **URL**: [http://www.captain-whu.com/project/SCD/](http://www.captain-whu.com/project/SCD/)
- **License**: Non-commercial research use
- **Spatial Resolution**: 0.5m–3.0m aerial sensor
- **Temporal Coverage**: Multi-temporal aerial coverage over Hangzhou, Chengdu, Shanghai
- **Label Semantics**: 6 land cover categories (non-vegetated ground surface, tree, low vegetation, water, buildings, playground).
- **CRS**: Pixel-space Cartesian coordinates without EPSG georeferencing.
- **Label Categorization**: **Authoritative**.
- **Suitability for SatQuery Evaluation**:
  - **Unsuitable**. Lacks geographic CRS georeferencing and spectral SWIR bands needed for Sentinel-2 multi-index analysis.

---

### Dataset 7: SpaceNet 7 (Multi-Temporal Urban Development Challenge)
- **Dataset Name**: SpaceNet 7
- **Source**: SpaceNet LLC / Radiant Earth Foundation
- **URL**: [https://spacenet.ai/sn7-challenge/](https://spacenet.ai/sn7-challenge/)
- **License**: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)
- **Spatial Resolution**: 4.0m (PlanetScope RGB-NIR imagery resampled to 3.0m)
- **Temporal Coverage**: 24 monthly mosaics (2018–2020) over 100 global locations
- **Label Semantics**: Building footprint polygons with unique tracking IDs across time.
- **CRS**: UTM projections (WGS 84).
- **Label Categorization**: **Authoritative** (polygon building footprints manually traced and tracked across 24 monthly steps).
- **Suitability for SatQuery Evaluation**:
  - **Moderate/Low**. Excellent for monthly building progression tracking, but uses PlanetScope 4-band imagery (lacking Sentinel-2 SWIR band and 10m calibration).

---

## 2. Recommended Integration Strategy for SatQuery AI

1. **Primary Global Reference Dataset**: **Dynamic World (WRI/Google)**
   - **Why**: Native Sentinel-2 10m grid, globally available across all SatQuery AOIs (Vienna, Delhi, Queensland, Sundarbans), open CC-BY-4.0 license, and provides explicit pixel-level probability distributions over water, trees, grass, crops, built, and bare classes.
   - **Mapping Strategy**: Bi-temporal land cover transition differencing directly yields the 6 active change classes (`urban_expansion`, `urban_reduction`, `vegetation_loss`, `vegetation_gain`, `water_loss`, `water_gain`).
2. **Authoritative Urban Validation Benchmark**: **OSCD (Onera Satellite Change Detection)**
   - **Why**: Expert-annotated binary urban change masks specifically delineated on Sentinel-2 Level-1C/Level-2A imagery.
   - **Mapping Strategy**: Binary mapping (`1: urban_expansion`, `0: no_change`).
3. **Current Materialization Status**:
   - **Benchmark Infrastructure**: STAGE A COMPLETE.
   - **Label Materialization**: STAGE B IN PROGRESS (pending automated or batch download of independent reference rasters from Dynamic World / OSCD APIs).
   - **Scientific Safeguard**: In accordance with Phase 9 scientific boundaries, unmaterialized scenes are explicitly designated `status: "pending_reference_label"` and `ground_truth_path: null`. No synthetic or self-derived rasters are reported as empirical ground truth.
