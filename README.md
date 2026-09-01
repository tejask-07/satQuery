# SatQuery AI

SatQuery AI is an advanced, natural-language-driven satellite imagery analysis platform. It allows users to write intuitive text queries (e.g., "Where did vegetation decrease between 2021 and 2024?") and automatically provisions the correct imagery, routes the analysis pipeline, computes remote sensing indices, and renders geographic visualizations on an interactive map.

## Core Capabilities
- **Natural Language Parsing**: Translates human questions into executable satellite data tasks.
- **Dynamic AOI Selection**: Parses geometric boundaries natively from queries (e.g. `[151.195, -33.885, 151.225, -33.855]`) or via manual map drawing tools.
- **Single-Index Analysis**: Computes spatial distributions of vegetation (NDVI), water (NDWI), and urban footprints (NDBI).
- **Change Detection**: Quantifies pixel-level differences over time (e.g. 2021 vs 2024).
- **Image Comparison**: Direct visual overlay and side-by-side inspection of temporal satellite imagery.
- **VLM / RAG Fallback**: (Optional architecture component) Utilizes visual language models for complex scene interpretations.

## Architecture
- **Frontend Stack**: React 18, TypeScript, Vite, React-Leaflet.
- **Backend Stack**: FastAPI, Python 3.10+, NumPy, OpenCV (cv2) for raster manipulation, Uvicorn.
- **Satellite Data Sources**: Sentinel-2 (Level-2A via Planetary Computer STAC) and BigEarthNet integrations.

## Supported Indices & Scientific Methodology

### 1. Vegetation Index (NDVI)
- **Formula**: `(NIR - Red) / (NIR + Red)`
- **Range**: -1.0 to 1.0
- **Source Bands**: Sentinel-2 Band 8 (NIR), Band 4 (Red).

### 2. Water Index (NDWI)
- **Formula**: `(Green - NIR) / (Green + NIR)`
- **Range**: -1.0 to 1.0
- **Source Bands**: Sentinel-2 Band 3 (Green), Band 8 (NIR).

### 3. Built-Up/Urban Index (NDBI)
- **Formula**: `(SWIR - NIR) / (SWIR + NIR)`
- **Range**: -1.0 to 1.0
- **Source Bands**: Sentinel-2 Band 11 (SWIR1), Band 8 (NIR).

### Change Detection & Masking
- **Diff Algorithm**: `Change = After - Before`
- **Masking**: Pixels with missing or infinite values (e.g., nodata or cloud-masked) in either acquisition are safely ignored.
- **Threshold**: A pixel is classified as "changed" if `abs(After - Before) >= threshold` (default `0.01`).
- **Aggregation**: Final statistics report `valid_pixels` (the intersection of valid data in both scenes) and `changed_pixels`.

## Installation & Running Locally

1. **Clone the repository.**
2. **Backend Setup**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
3. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Production Deployment & Environment Variables

For production, ensure the correct environment variables are set.

### Backend (`backend/.env`)
- `ALLOWED_ORIGINS`: Comma-separated list of allowed frontend origins (e.g. `https://my-app.com`).
- `SENTINEL_STAC_URL`: STAC endpoint URL.

### Frontend (`frontend/.env`)
- `VITE_API_URL`: The full URL to the FastAPI backend (e.g. `https://api.my-app.com`).

*Build the frontend for production using `npm run build`. Serve the generated `/dist` folder through Nginx or a static host.*

## API Overview
- `POST /api/query`: Submits a natural language query and AOI. Returns a structured JSON containing:
  - `plan.task`: The routed task type (e.g., `vegetation_index`, `change_detection`).
  - `layers`: The generated visualization artifacts including `url` (PNG) and geographic `bounds`.
  - `statistics`: Dynamic valid pixel, min/max, and changed pixel calculations.

## Example Queries
- *"Show NDVI for 2024 for AOI [151.195, -33.885, 151.225, -33.855]"* (Renders single NDVI index)
- *"Show NDWI for 2024 for the same AOI"* (Renders single NDWI index)
- *"Compare NDVI change between 2023 and 2024 for this AOI"* (Renders NDVI change detection and metrics)
- *"Compare urban/built-up change between 2023 and 2024"* (Renders NDBI change)

## Testing / QA Status
- **Backend API Regression**: Fully automated via Python scripting (100% PASS).
- **Frontend Verification**: Manually verified via visual QA on running instance (100% PASS).
- **Build Status**: Verified successfully with TypeScript compiler (npx tsc exits Code 0).

## Known Limitations
- The visual language modeling / VLM chat workflow is currently experimental and intended for distinct scene-level reasoning rather than large-scale metric calculation.
- Automated E2E Browser Testing using Playwright is currently disabled due to external CDN restrictions on fetching driver binaries in certain deployment environments. Use manual browser smoke testing instead.
