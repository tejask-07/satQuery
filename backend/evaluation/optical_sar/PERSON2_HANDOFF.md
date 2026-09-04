# Person 2 Backend Handoff & API Contract

**Author**: Person 2: Optical–SAR / Multimodal Engineer  
**Role**: Optical–SAR Ingestion, Modality Validation, Spatial Alignment, Multi-Sensor Visualization, VLM Reasoning Specialist, Sentinel-1 Provider, Scene Pairing, and Automatic Acquisition Pipeline.  
**Repository Branch**: `subham`  
**Status**: Step 2 through Step 14 complete, verified with 151 passing tests.

---

## 1. Overview of Capability

SatQuery now provides unified, cross-sensor **Optical + SAR multimodal intelligence**. The backend automatically pairs Sentinel-2 optical imagery (multispectral surface reflectance) and Sentinel-1 SAR observations (C-band microwave radar backscatter), co-registers them to a common optical reference grid, generates scientific visualizations, and feeds the multimodal package into a specialized VLM reasoning prompt.

There are **two distinct supported operational modes**:
1. **Mode A**: Explicit Upload / Reference Mode (User supplies rasters or image IDs)
2. **Mode B**: Automatic Acquisition Mode (User supplies AOI, target dates, and query)

---

## 2. API Endpoints & Contracts

### A. Automatic Mode (Mode B) — Standard `/api/query`

When no explicit image references are passed, natural language queries requesting joint optical and radar analysis automatically trigger the catalog acquisition and pairing coordinator.

#### Request Contract:
- **Endpoint**: `POST /api/query`
- **Content-Type**: `application/json`
- **Payload Schema**:
  ```json
  {
    "query": "Use the optical and SAR images together to identify built-up areas.",
    "aoi": [13.0, 48.0, 13.02, 48.02],
    "time_start": "2021-06-25",
    "time_end": "2021-06-28"
  }
  ```
  *Note on `aoi`:* Accepts either a 4-element bounding box `[min_lon, min_lat, max_lon, max_lat]` or a standard GeoJSON Polygon geometry dict `{"type": "Polygon", "coordinates": [...]}`.

#### Automatic Pipeline Execution Flow:
```text
POST /api/query
      ↓
parse_query() → Classified as "optical_sar_analysis"
      ↓
plan.task == "optical_sar_analysis"
      ↓
Check explicit inputs: None provided → Trigger Mode B
      ↓
Validate presence of AOI and temporal range:
  - Missing AOI   → HTTP 400 ("Optical-SAR automatic acquisition requires an AOI.")
  - Missing dates → HTTP 400 ("Optical-SAR automatic acquisition requires a target date or time range.")
      ↓
find_optical_sar_pair(aoi, time_start, time_end, fetch_data=True)
      ↓
1. Sentinel2Provider.search_candidate_scenes() → Ranked optical scenes
2. Sentinel1Provider.search_candidate_scenes() → Ranked SAR scenes in temporal window
3. Deterministic multi-factor scoring (cloud, overlap, temporal delta, dual-pol bonus)
4. Cache download: S2 True Color RGB GeoTIFF + S1 VV & VH GeoTIFFs
      ↓
validate_optical_sar_pair(opt_path, sar_path)
      ↓
align_optical_sar_pair(opt_path, sar_path) → Reproject SAR to optical grid
      ↓
build_optical_sar_visuals() → Optical RGB, SAR VV, SAR VH, SAR Composite
      ↓
answer_optical_sar_question() → Evaluated by VLM Specialist with domain grounding
      ↓
Response packaged as AnalysisResult
```

---

### B. Explicit Upload & Reference Mode (Mode A)

Explicitly provided imagery always takes precedence over automatic acquisition.

#### Option 1: Direct Multipart File Upload
- **Endpoint**: `POST /api/upload/optical-sar`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `optical_image`: Binary GeoTIFF file (Optical Surface Reflectance)
  - `sar_image`: Binary GeoTIFF file (SAR backscatter, VV or dual-pol)
  - `query`: Optional question text (Default: `"Use the optical and SAR images together to analyze the area."`)
- **Response**: Immediate `AnalysisResult` JSON object.

#### Option 2: Pre-Uploaded Image IDs via `/api/query`
- **Endpoint**: `POST /api/query`
- **Payload Schema**:
  ```json
  {
    "query": "Use the optical and SAR images together to analyze the area.",
    "optical_image_id": "opt_a1b2c3d4",
    "sar_image_id": "sar_e5f6a7b8"
  }
  ```
  Image IDs are obtained from `POST /api/upload/image`.

---

## 3. Response Schema (`AnalysisResult`)

The response schema strictly adheres to the unified `AnalysisResult` model across all remote-sensing tasks:

```json
{
  "status": "success",
  "answer": "Optical imagery (false-color composite) and Sentinel-1 SAR observations (sar_vv, sar_vh, sar_composite) have been co-registered onto the optical reference grid...",
  "confidence": 0.9,
  "plan": {
    "task": "optical_sar_analysis",
    "target": "AOI",
    "time_start": "2021-06-25",
    "time_end": "2021-06-28",
    "modalities": ["Sentinel-2", "Sentinel-1"],
    "analysis": ["multimodal_reasoning"],
    "output": ["optical_rgb", "sar_vv", "sar_vh", "sar_composite"]
  },
  "statistics": {
    "task": "optical_sar_analysis",
    "modalities": ["optical", "sar_vv", "sar_vh", "sar_composite"],
    "grid": {
      "reference": "optical",
      "crs": "EPSG:4326",
      "width": 200,
      "height": 200
    },
    "optical_sar_pair": {
      "source": "automatic",
      "pair_found": true,
      "optical_item_id": "S2B_MSIL2A_20210627T100559_R022_T32UQU_20210628T005441",
      "sar_item_id": "S1B_IW_GRDH_1SDV_20210627T165835_20210627T165900_027545_0349C1",
      "optical_acquisition_datetime": "2021-06-27T10:05:59.024000+00:00",
      "sar_acquisition_datetime": "2021-06-27T16:58:47.555054+00:00",
      "temporal_delta_days": 0.287,
      "polarizations": ["VV", "VH"],
      "coverage": 1.0,
      "selection_reason": "Selected Optical S2B_... and SAR S1B_... with composite score 0.8980."
    }
  },
  "layers": [
    {
      "name": "Optical Surface Reflectance",
      "type": "optical_rgb",
      "visualization_url": "/visualizations/multimodal_optical_xxxx.png",
      "bounds": [[48.0, 13.0], [48.02, 13.02]]
    },
    {
      "name": "Sentinel-1 VV Backscatter",
      "type": "sar_vv",
      "visualization_url": "/visualizations/multimodal_sar_vv_xxxx.png",
      "bounds": [[48.0, 13.0], [48.02, 13.02]]
    },
    {
      "name": "Sentinel-1 VH Backscatter",
      "type": "sar_vh",
      "visualization_url": "/visualizations/multimodal_sar_vh_xxxx.png",
      "bounds": [[48.0, 13.0], [48.02, 13.02]]
    },
    {
      "name": "Sentinel-1 Polarimetric Composite",
      "type": "sar_composite",
      "visualization_url": "/visualizations/multimodal_sar_composite_xxxx.png",
      "bounds": [[48.0, 13.0], [48.02, 13.02]]
    }
  ],
  "execution_trace": [
    "Natural-language query classified as optical_sar_analysis",
    "Automatic Optical-SAR mode triggered (AOI and temporal range provided)",
    "Searching Sentinel-2 optical candidates",
    "Searching Sentinel-1 SAR candidates",
    "Best Optical-SAR pair selected (temporal delta: 0.29 days)",
    "Verified presence of both Optical and SAR GeoTIFF rasters",
    "Co-registered SAR onto optical reference grid",
    "Generated multi-sensor visual layers for qualitative inspection",
    "Grounding prompt and multimodal inputs evaluated by VLM specialist"
  ],
  "visualization_url": "/visualizations/multimodal_optical_xxxx.png"
}
```

---

## 4. Frontend Layer Presentation Guide

The backend emits up to four co-registered raster visual layers. The recommended frontend presentation is a layer-switcher or tabbed control over the map view:

| Layer `type` | Display Name | Visual Content | Recommended UI Handling |
| :--- | :--- | :--- | :--- |
| `optical_rgb` | Optical Reflectance | True Color RGB (B04/B03/B02) or False-Color (NIR/Red/Green) | Default base layer |
| `sar_vv` | SAR VV Backscatter | Greyscale co-polarized backscatter (surface roughness & double-bounce) | Layer switch / toggle |
| `sar_vh` | SAR VH Backscatter | Greyscale cross-polarized backscatter (volume scattering, canopy) | Layer switch / toggle |
| `sar_composite` | SAR Composite | Dual-pol false color: Red=VV, Green=VH, Blue=VV/VH ratio | Layer switch / toggle |

---

## 5. Important Invariants & System Constraints

1. **Explicit Priority**:
   If explicit image references (`optical_image_id` / `sar_image_id` or uploaded files) are present, automatic acquisition is **never** invoked.
2. **Missing Parameters in Automatic Mode**:
   - Missing AOI returns `HTTP 400`: `"Optical-SAR automatic acquisition requires an AOI."`
   - Missing dates returns `HTTP 400`: `"Optical-SAR automatic acquisition requires a target date or time range."`
3. **Task Isolation**:
   Non-Optical-SAR tasks (`calculate_ndvi`, `calculate_ndwi`, `calculate_ndbi`, `detect_change`) will **never** trigger `find_optical_sar_pair` or Sentinel-1 acquisition.
4. **Physical Calibration Units**:
   Sentinel-1 rasters are preserved in raw physical sensor units (`uint16` linear DN). No lossy conversions to decibels (dB) or 8-bit clipping occur during raster storage or alignment.
5. **No BigEarthNet Fallback**:
   The live automatic acquisition pipeline searches and retrieves real Copernicus assets via Microsoft Planetary Computer STAC. It never silently falls back to BigEarthNet demo scenes.
6. **Path Security**:
   The frontend can never pass arbitrary server paths. All rasters are resolved through secure upload identifiers or sanitized provider caches.
