import { useState, useMemo, useEffect } from "react";

import {
  MapContainer,
  TileLayer,
  Polygon,
  Circle,
  ImageOverlay,
  useMap,
} from "react-leaflet";

import type { QueryResponse } from "../../api/query";

import "leaflet/dist/leaflet.css";
import "./LayerVisualization.css";

function LayersViewportController({
  bounds,
  center,
}: {
  bounds: [[number, number], [number, number]] | null;
  center: [number, number];
}) {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      try {
        map.fitBounds(bounds, { padding: [24, 24], maxZoom: 14 });
      } catch (err) {
        console.warn("fitBounds failed:", err);
      }
    } else {
      map.setView(center, 10);
    }
  }, [map, bounds, center]);
  return null;
}


interface LayersVisualizationProps {
  result: QueryResponse;
  onBack: () => void;
}


type IndexType =
  | "NDVI"
  | "NDWI"
  | "NDBI";


type BaseLayer =
  | "trueColor"
  | "falseColor"
  | "dark";


type VisualizationType =
  | "heatmap"
  | "classified"
  | "gradient";


function LayersVisualization({
  result,
  onBack,
}: LayersVisualizationProps) {
  const metric = (String(result?.statistics?.metric || result?.plan?.metric || "NDVI").toUpperCase()) as IndexType;
  const initialIndex: IndexType = ["NDVI", "NDWI", "NDBI"].includes(metric) ? metric : "NDVI";

  const [selectedIndex, setSelectedIndex] =
    useState<IndexType>(initialIndex);

  const [baseLayer, setBaseLayer] =
    useState<BaseLayer>("trueColor");

  const [visualization, setVisualization] =
    useState<VisualizationType>("heatmap");

  const [opacity, setOpacity] =
    useState(80);

  const [analysisLayers, setAnalysisLayers] =
    useState({
      ndviChange: true,
      ndwiChange: false,
      ndbiChange: false,
      detectedRegions: true,
      aoiBoundary: true,
      confidenceMap: false,
    });

  const [reference, setReference] =
    useState<"before" | "after">("after");

  const dateBefore = result?.plan?.time_start ?? "2021";
  const dateAfter = result?.plan?.time_end ?? "2025";

  /*
   * =========================================================
   * REAL AOI & GEOGRAPHIC BOUNDS
   * =========================================================
   */

  const aoi: [number, number][] = useMemo(() => {
    const rawAoi = result?.plan?.aoi as any;
    if (!rawAoi) {
      return [
        [19.32, 72.70],
        [19.45, 72.92],
        [19.30, 73.08],
        [19.02, 73.12],
        [18.78, 72.98],
        [18.70, 72.78],
        [18.88, 72.66],
        [19.12, 72.61],
      ];
    }
    if (rawAoi.type === "Polygon" && Array.isArray(rawAoi.coordinates?.[0])) {
      return rawAoi.coordinates[0].map((pt: number[]) => [Number(pt[1]), Number(pt[0])] as [number, number]);
    }
    if (Array.isArray(rawAoi) && rawAoi.length > 0 && Array.isArray(rawAoi[0])) {
      return rawAoi.map((pt: number[]) => [Number(pt[0]), Number(pt[1])] as [number, number]);
    }
    return [];
  }, [result?.plan?.aoi]);

  const rawBounds = (result?.layers?.[0] as any)?.bounds ?? result?.bounds ?? null;
  const realBounds: [[number, number], [number, number]] | null = useMemo(() => {
    if (Array.isArray(rawBounds) && rawBounds.length === 2 && Array.isArray(rawBounds[0]) && Array.isArray(rawBounds[1])) {
      const s = Number(rawBounds[0][0]);
      const w = Number(rawBounds[0][1]);
      const n = Number(rawBounds[1][0]);
      const e = Number(rawBounds[1][1]);
      if ([s, w, n, e].every(Number.isFinite)) {
        return [[s, w], [n, e]];
      }
    }
    if (aoi.length >= 3) {
      const lats = aoi.map((p) => p[0]);
      const lngs = aoi.map((p) => p[1]);
      return [
        [Math.min(...lats), Math.min(...lngs)],
        [Math.max(...lats), Math.max(...lngs)],
      ];
    }
    return null;
  }, [rawBounds, aoi]);

  const mapCenter: [number, number] = useMemo(() => {
    if (realBounds) {
      return [
        (realBounds[0][0] + realBounds[1][0]) / 2,
        (realBounds[0][1] + realBounds[1][1]) / 2,
      ];
    }
    if (aoi.length > 0) {
      return [
        aoi.reduce((sum, p) => sum + p[0], 0) / aoi.length,
        aoi.reduce((sum, p) => sum + p[1], 0) / aoi.length,
      ];
    }
    return [19.076, 72.8777];
  }, [realBounds, aoi]);

  const rawVisUrl =
    (result?.layers?.[0] as any)?.visualization_url ??
    (result?.layers?.[0] as any)?.classified_visualization_url ??
    result?.visualization_url ??
    null;

  const fullVisualizationUrl = useMemo(() => {
    if (!rawVisUrl) return null;
    const cleanUrl = String(rawVisUrl).trim();
    if (cleanUrl.startsWith("http://") || cleanUrl.startsWith("https://")) {
      return cleanUrl;
    }
    const baseUrl = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
    const pathUrl = cleanUrl.startsWith("/") ? cleanUrl : `/${cleanUrl}`;
    return `${baseUrl}${pathUrl}`;
  }, [rawVisUrl]);


  /*
   * =========================================================
   * TEMPORARY INDEX VISUALIZATION
   * =========================================================
   */

  const indexAreas: [number, number][] = [
    [19.28, 72.88],
    [19.20, 72.91],
    [19.10, 72.94],
    [19.02, 72.89],
    [18.95, 72.94],
    [19.15, 72.83],
    [18.88, 72.88],
    [19.23, 72.79],
    [19.04, 72.82],
  ];


  /*
   * =========================================================
   * TOGGLE
   * =========================================================
   */

  const toggleLayer = (
    layer: keyof typeof analysisLayers
  ) => {

    setAnalysisLayers((current) => ({
      ...current,
      [layer]: !current[layer],
    }));

  };


  /*
   * =========================================================
   * BASE MAP
   * =========================================================
   */

  const getTileUrl = () => {

    if (baseLayer === "dark") {
      return "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
    }

    if (baseLayer === "falseColor") {
      return "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
    }

    return "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
  };


  /*
   * =========================================================
   * INDEX INFO
   * =========================================================
   */

  const indexInfo = {

    NDVI: {
      name: "NDVI (Normalized Difference Vegetation Index)",
      description:
        "Indicates vegetation health and density using near-infrared and red reflectance.",
      range: "-1 to 1",
      interpretation:
        "High values represent healthy vegetation.",
    },

    NDWI: {
      name: "NDWI (Normalized Difference Water Index)",
      description:
        "Highlights water content and surface water features using spectral reflectance.",
      range: "-1 to 1",
      interpretation:
        "Higher values generally represent water-rich surfaces.",
    },

    NDBI: {
      name: "NDBI (Normalized Difference Built-up Index)",
      description:
        "Highlights built-up and urban surfaces using shortwave infrared and near-infrared reflectance.",
      range: "-1 to 1",
      interpretation:
        "Higher values generally represent built-up surfaces.",
    },

  };


  const currentInfo =
    indexInfo[selectedIndex];


  return (
    <main className="layers-workspace">

      {/* =====================================================
          LEFT — LAYERS
          ===================================================== */}

      <aside className="layers-sidebar">

        <button type="button" className="layers-back-button" onClick={onBack}>
          ← ANALYSIS
        </button>

        <div className="layers-panel-title">
          LAYERS
        </div>


        {/* ===================================================
            BASE LAYERS
            =================================================== */}

        <section className="layers-control-section">

          <div className="layers-control-label">
            BASE LAYERS
          </div>


          <label className="layer-radio-row">

            <input
              type="radio"
              name="base-layer"
              checked={baseLayer === "trueColor"}
              onChange={() =>
                setBaseLayer("trueColor")
              }
            />

            <span>
              Satellite (True Color)
            </span>

          </label>


          <label className="layer-radio-row">

            <input
              type="radio"
              name="base-layer"
              checked={baseLayer === "falseColor"}
              onChange={() =>
                setBaseLayer("falseColor")
              }
            />

            <span>
              Satellite (False Color)
            </span>

          </label>


          <label className="layer-radio-row">

            <input
              type="radio"
              name="base-layer"
              checked={baseLayer === "dark"}
              onChange={() =>
                setBaseLayer("dark")
              }
            />

            <span>
              Dark Basemap
            </span>

          </label>

        </section>


        {/* ===================================================
            ANALYSIS LAYERS
            =================================================== */}

        <section className="layers-control-section">

          <div className="layers-control-label">
            ANALYSIS LAYERS
          </div>


          <label className="layer-check-row">

            <input
              type="checkbox"
              checked={analysisLayers.ndviChange}
              onChange={() =>
                toggleLayer("ndviChange")
              }
            />

            <span>
              NDVI Change
            </span>

          </label>


          <label className="layer-check-row">

            <input
              type="checkbox"
              checked={analysisLayers.ndwiChange}
              onChange={() =>
                toggleLayer("ndwiChange")
              }
            />

            <span>
              NDWI Change
            </span>

          </label>


          <label className="layer-check-row">

            <input
              type="checkbox"
              checked={analysisLayers.ndbiChange}
              onChange={() =>
                toggleLayer("ndbiChange")
              }
            />

            <span>
              NDBI Change
            </span>

          </label>


          <label className="layer-check-row">

            <input
              type="checkbox"
              checked={analysisLayers.detectedRegions}
              onChange={() =>
                toggleLayer("detectedRegions")
              }
            />

            <span>
              Detected Regions
            </span>

          </label>


          <label className="layer-check-row">

            <input
              type="checkbox"
              checked={analysisLayers.aoiBoundary}
              onChange={() =>
                toggleLayer("aoiBoundary")
              }
            />

            <span>
              AOI Boundary
            </span>

          </label>


          <label className="layer-check-row">

            <input
              type="checkbox"
              checked={analysisLayers.confidenceMap}
              onChange={() =>
                toggleLayer("confidenceMap")
              }
            />

            <span>
              Confidence Map
            </span>

          </label>

        </section>


        {/* ===================================================
            REFERENCE
            =================================================== */}

        <section className="layers-control-section">

          <div className="layers-control-label">
            REFERENCE
          </div>


          <label className="layer-check-row">

            <input
              type="radio"
              name="reference"
              checked={reference === "before"}
              onChange={() =>
                setReference("before")
              }
            />

            <span>
              Before ({dateBefore})
            </span>

          </label>


          <label className="layer-check-row">

            <input
              type="radio"
              name="reference"
              checked={reference === "after"}
              onChange={() =>
                setReference("after")
              }
            />

            <span>
              After ({dateAfter})
            </span>

          </label>

        </section>


        <button
          type="button"
          className="layers-clear-button"
          onClick={() =>
            setAnalysisLayers({
              ndviChange: false,
              ndwiChange: false,
              ndbiChange: false,
              detectedRegions: false,
              aoiBoundary: false,
              confidenceMap: false,
            })
          }
        >
          CLEAR ALL
        </button>

      </aside>


      {/* =====================================================
          CENTER — MAP
          ===================================================== */}

      <section className="layers-map-panel">

        <div className="layers-map-header">

          <div className="layers-map-title">
            INDEX
          </div>


          <div className="index-tabs">

            {(["NDVI", "NDWI", "NDBI"] as IndexType[]).map(
              (index) => (

                <button
                  key={index}
                  type="button"
                  className={
                    selectedIndex === index
                      ? "index-tab active"
                      : "index-tab"
                  }
                  onClick={() =>
                    setSelectedIndex(index)
                  }
                >
                  {index}
                </button>

              )
            )}

          </div>

        </div>


        <div className="layers-map-subtitle">

          {selectedIndex} CHANGE
          {" "}
          ({dateBefore} → {dateAfter})

        </div>


        <div className="layers-map">

          <MapContainer
            center={mapCenter}
            zoom={10}
            zoomControl={true}
            attributionControl={true}
            className="index-map"
          >

            <TileLayer
              url={getTileUrl()}
              attribution="Tiles © Esri"
            />

            <LayersViewportController
              bounds={realBounds}
              center={mapCenter}
            />

            {/* REAL RASTER OVERLAY */}
            {fullVisualizationUrl && realBounds && (
              <ImageOverlay
                url={fullVisualizationUrl}
                bounds={realBounds}
                opacity={opacity / 100}
                zIndex={1000}
              />
            )}

            {/* AOI */}

            {analysisLayers.aoiBoundary && aoi.length > 0 && (
              <Polygon
                positions={aoi}
                pathOptions={{
                  color: "#f5f1e9",
                  weight: 2,
                  fillOpacity: 0.02,
                }}
              />
            )}


            {/* Index visualization */}

            {analysisLayers.ndviChange &&
              indexAreas.map((position, index) => (

                <Circle
                  key={`${position[0]}-${position[1]}-${index}`}
                  center={position}
                  radius={700 + (index % 3) * 250}
                  pathOptions={{
                    color:
                      visualization === "classified"
                        ? "#f0c52e"
                        : "#8dbb4f",
                    weight: 0,
                    fillColor:
                      visualization === "gradient"
                        ? "#d7e86b"
                        : "#8dbb4f",
                    fillOpacity:
                      opacity / 100,
                  }}
                />

              ))}


            {/* Detected regions */}

            {analysisLayers.detectedRegions &&
              indexAreas.slice(0, 6).map(
                (position, index) => (

                  <Circle
                    key={`region-${index}`}
                    center={position}
                    radius={260}
                    pathOptions={{
                      color: "#d94228",
                      weight: 0,
                      fillColor: "#d94228",
                      fillOpacity:
                        Math.min(opacity / 100, 0.75),
                    }}
                  />

                )
              )}

          </MapContainer>


          <div className="layers-map-badge">

            <span>
              {selectedIndex}
            </span>

            <span>
              {reference === "after"
                ? "2025"
                : "2021"}
            </span>

          </div>

        </div>


        {/* ===================================================
            COLOR SCALE
            =================================================== */}

        <div className="index-scale">

          <span>
            -1
          </span>

          <div className="index-gradient" />

          <span>
            1
          </span>

        </div>

      </section>


      {/* =====================================================
          RIGHT — INFORMATION
          ===================================================== */}

      <aside className="index-info-panel">

        <section className="index-info-section">

          <div className="layers-panel-title">
            INDEX INFO
          </div>


          <h2>
            {currentInfo.name}
          </h2>


          <p>
            {currentInfo.description}
          </p>


          <p>
            Range: {currentInfo.range}
          </p>


          <p>
            {currentInfo.interpretation}
          </p>


          <button
            type="button"
            className="learn-more-button"
          >
            Learn more →
          </button>

        </section>


        {/* ===================================================
            VISUALIZATION
            =================================================== */}

        <section className="visualization-section">

          <div className="layers-control-label">
            VISUALIZATION
          </div>


          <label className="layer-radio-row">

            <input
              type="radio"
              name="visualization"
              checked={visualization === "heatmap"}
              onChange={() =>
                setVisualization("heatmap")
              }
            />

            <span>
              Heatmap
            </span>

          </label>


          <label className="layer-radio-row">

            <input
              type="radio"
              name="visualization"
              checked={visualization === "classified"}
              onChange={() =>
                setVisualization("classified")
              }
            />

            <span>
              Classified
            </span>

          </label>


          <label className="layer-radio-row">

            <input
              type="radio"
              name="visualization"
              checked={visualization === "gradient"}
              onChange={() =>
                setVisualization("gradient")
              }
            />

            <span>
              Gradient
            </span>

          </label>


          {/* =================================================
              OPACITY
              ================================================= */}

          <div className="opacity-control">

            <div className="opacity-header">

              <span>
                OPACITY
              </span>

              <span>
                {opacity}%
              </span>

            </div>


            <input
              type="range"
              min="0"
              max="100"
              value={opacity}
              onChange={(event) =>
                setOpacity(
                  Number(event.target.value)
                )
              }
            />

          </div>

        </section>

      </aside>

    </main>
  );
}


export default LayersVisualization;