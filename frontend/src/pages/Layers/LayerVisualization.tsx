import { useState, useMemo, useEffect } from "react";

import {
  MapContainer,
  TileLayer,
  Polygon,
  ImageOverlay,
  useMap,
} from "react-leaflet";


import { fetchBenchmarkSummary } from "../../api/query";
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
      ndviChange: initialIndex === "NDVI",
      ndwiChange: initialIndex === "NDWI",
      ndbiChange: initialIndex === "NDBI",
      detectedRegions: true,
      aoiBoundary: true,
      confidenceMap: false,
    });

  const [reference, setReference] =
    useState<"before" | "after">("after");

  const [showBenchmarkModal, setShowBenchmarkModal] = useState(false);
  const [benchmarkData, setBenchmarkData] = useState<Record<string, any> | null>(null);

  useEffect(() => {
    fetchBenchmarkSummary().then((data) => {
      if (data && Object.keys(data).length > 0) {
        setBenchmarkData(data);
      }
    });
  }, []);

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

  const resolveUrl = (rawPath?: string | null) => {
    if (!rawPath) return null;
    const clean = String(rawPath).trim();
    if (!clean) return null;
    if (clean.startsWith("http://") || clean.startsWith("https://")) return clean;
    const baseUrl = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
    const pathUrl = clean.startsWith("/") ? clean : `/${clean}`;
    return `${baseUrl}${pathUrl}`;
  };

  const layersList = useMemo(() => (Array.isArray(result?.layers) ? (result.layers as any[]) : []), [result?.layers]);

  const trueColorBeforeUrl = useMemo(() => {
    const pkgUrl = result?.layer_package?.before?.true_color?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x: any) => x.id === "true_color_before");
    return resolveUrl(l?.visualization_url);
  }, [result?.layer_package, layersList]);

  const trueColorAfterUrl = useMemo(() => {
    const pkgUrl = result?.layer_package?.after?.true_color?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x: any) => x.id === "true_color_after");
    return resolveUrl(l?.visualization_url);
  }, [result?.layer_package, layersList]);

  const falseColorBeforeUrl = useMemo(() => {
    const pkgUrl = result?.layer_package?.before?.false_color?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x: any) => x.id === "false_color_before");
    return resolveUrl(l?.visualization_url);
  }, [result?.layer_package, layersList]);

  const falseColorAfterUrl = useMemo(() => {
    const pkgUrl = result?.layer_package?.after?.false_color?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x: any) => x.id === "false_color_after");
    return resolveUrl(l?.visualization_url);
  }, [result?.layer_package, layersList]);

  const qualityMaskBeforeUrl = useMemo(() => {
    const pkgUrl = result?.layer_package?.quality?.mask_before?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x: any) => x.id === "quality_mask_before");
    return resolveUrl(l?.visualization_url);
  }, [result?.layer_package, layersList]);

  const qualityMaskAfterUrl = useMemo(() => {
    const pkgUrl = result?.layer_package?.quality?.mask_after?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x: any) => x.id === "quality_mask_after");
    return resolveUrl(l?.visualization_url);
  }, [result?.layer_package, layersList]);

  // Index URLs per currently selected index tab
  const currentIndexBeforeUrl = useMemo(() => {
    const idxKey = selectedIndex.toLowerCase();
    const pkgUrl = result?.layer_package?.before?.[idxKey]?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x: any) => x.id === `${idxKey}_before` || (x.id === "index_before" && (x.metric?.toUpperCase() === selectedIndex || !x.metric)));
    return resolveUrl(l?.visualization_url);
  }, [result?.layer_package, layersList, selectedIndex]);

  const currentIndexAfterUrl = useMemo(() => {
    const idxKey = selectedIndex.toLowerCase();
    const pkgUrl = result?.layer_package?.after?.[idxKey]?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x: any) => x.id === `${idxKey}_after` || (x.id === "index_after" && (x.metric?.toUpperCase() === selectedIndex || !x.metric)));
    return resolveUrl(l?.visualization_url);
  }, [result?.layer_package, layersList, selectedIndex]);

  // Delta / Change URLs per index
  const ndviChangeUrl = useMemo(() => {
    if (visualization === "classified") {
      const pkgClassified = result?.layer_package?.change?.delta_ndvi?.classified_url;
      if (pkgClassified) return resolveUrl(pkgClassified);
      const l = layersList.find((x: any) => x.id === "change_ndvi");
      if (l?.classified_visualization_url) return resolveUrl(l.classified_visualization_url);
    }
    const pkgUrl = result?.layer_package?.change?.delta_ndvi?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x: any) => x.id === "change_ndvi");
    return resolveUrl(l?.visualization_url || (result?.statistics?.metric === "NDVI" ? result?.visualization_url : null));
  }, [result?.layer_package, layersList, visualization, result?.statistics?.metric, result?.visualization_url]);

  const ndwiChangeUrl = useMemo(() => {
    if (visualization === "classified") {
      const pkgClassified = result?.layer_package?.change?.delta_ndwi?.classified_url;
      if (pkgClassified) return resolveUrl(pkgClassified);
      const l = layersList.find((x: any) => x.id === "change_ndwi");
      if (l?.classified_visualization_url) return resolveUrl(l.classified_visualization_url);
    }
    const pkgUrl = result?.layer_package?.change?.delta_ndwi?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x: any) => x.id === "change_ndwi");
    return resolveUrl(l?.visualization_url || (result?.statistics?.metric === "NDWI" ? result?.visualization_url : null));
  }, [result?.layer_package, layersList, visualization, result?.statistics?.metric, result?.visualization_url]);

  const ndbiChangeUrl = useMemo(() => {
    if (visualization === "classified") {
      const pkgClassified = result?.layer_package?.change?.delta_ndbi?.classified_url;
      if (pkgClassified) return resolveUrl(pkgClassified);
      const l = layersList.find((x: any) => x.id === "change_ndbi");
      if (l?.classified_visualization_url) return resolveUrl(l.classified_visualization_url);
    }
    const pkgUrl = result?.layer_package?.change?.delta_ndbi?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x: any) => x.id === "change_ndbi");
    return resolveUrl(l?.visualization_url || (result?.statistics?.metric === "NDBI" ? result?.visualization_url : null));
  }, [result?.layer_package, layersList, visualization, result?.statistics?.metric, result?.visualization_url]);

  const currentTrueColorUrl = reference === "before" ? trueColorBeforeUrl : trueColorAfterUrl;
  const currentFalseColorUrl = reference === "before" ? falseColorBeforeUrl : falseColorAfterUrl;
  const currentIndexUrl = reference === "before" ? currentIndexBeforeUrl : currentIndexAfterUrl;
  const currentQualityMaskUrl = reference === "before" ? qualityMaskBeforeUrl : qualityMaskAfterUrl;

  const spatialData = (result?.spatial_analysis || (result as any)?.statistics?.spatial_analysis || (result?.layer_package as any)?.spatial) as any;
  const temporalData = (result?.temporal_analysis || (result as any)?.statistics?.temporal_analysis || (result?.layer_package as any)?.temporal) as any;
  const calibrationData = (result?.calibration || (result as any)?.statistics?.calibration) as any;
  const [showReasonDetails, setShowReasonDetails] = useState<boolean>(false);
  const filteredCandidateUrl = useMemo(() => {
    const pkgUrl = (result?.layer_package as any)?.spatial?.filtered_candidate_url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    return null;
  }, [result?.layer_package]);
  const geojsonFeatures = useMemo(() => {
    return (spatialData?.geojson?.features || []) as any[];
  }, [spatialData]);

  const [splitMode, setSplitMode] = useState<boolean>(false);

  const beforeMetadata = useMemo(() => {
    const ev = (Array.isArray(result?.evidence) ? (result.evidence[0] as any)?.images?.[0] : null) || {};
    const l = layersList.find((x: any) => x.id === "true_color_before");
    return l?.metadata || ev?.metadata || {};
  }, [result?.evidence, layersList]);

  const afterMetadata = useMemo(() => {
    const ev = (Array.isArray(result?.evidence) ? (result.evidence[0] as any)?.images?.[1] : null) || {};
    const l = layersList.find((x: any) => x.id === "true_color_after");
    return l?.metadata || ev?.metadata || {};
  }, [result?.evidence, layersList]);

  const currentMetadata = reference === "before" ? beforeMetadata : afterMetadata;




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

          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <button
              type="button"
              className={!splitMode ? "index-tab active" : "index-tab"}
              onClick={() => setSplitMode(false)}
            >
              Single
            </button>
            <button
              type="button"
              className={splitMode ? "index-tab active" : "index-tab"}
              onClick={() => setSplitMode(true)}
            >
              Side-by-Side
            </button>
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


        <div className="layers-map" style={{ position: "relative", height: "100%" }}>

          {!splitMode ? (
            <MapContainer
              center={mapCenter}
              zoom={10}
              zoomControl={true}
              attributionControl={true}
              className="index-map"
            >

              <TileLayer
                url={getTileUrl()}
                attribution="Tiles © Esri / Carto"
              />

              <LayersViewportController
                bounds={realBounds}
                center={mapCenter}
              />

              {/* REAL BASE LAYER OVERLAY (True-Color or False-Color) */}
              {baseLayer === "trueColor" && currentTrueColorUrl && realBounds && (
                <ImageOverlay
                  url={currentTrueColorUrl}
                  bounds={realBounds}
                  opacity={1.0}
                  zIndex={100}
                />
              )}
              {baseLayer === "falseColor" && currentFalseColorUrl && realBounds && (
                <ImageOverlay
                  url={currentFalseColorUrl}
                  bounds={realBounds}
                  opacity={1.0}
                  zIndex={100}
                />
              )}

              {/* REAL SCIENTIFIC INDEX OVERLAY */}
              {currentIndexUrl && realBounds && (
                <ImageOverlay
                  url={currentIndexUrl}
                  bounds={realBounds}
                  opacity={opacity / 100}
                  zIndex={200}
                />
              )}

              {/* REAL SCIENTIFIC CHANGE OVERLAYS - INDEPENDENTLY RENDERED */}
              {analysisLayers.ndviChange && ndviChangeUrl && realBounds && (
                <ImageOverlay
                  url={ndviChangeUrl}
                  bounds={realBounds}
                  opacity={opacity / 100}
                  zIndex={310}
                />
              )}
              {analysisLayers.ndwiChange && ndwiChangeUrl && realBounds && (
                <ImageOverlay
                  url={ndwiChangeUrl}
                  bounds={realBounds}
                  opacity={opacity / 100}
                  zIndex={320}
                />
              )}
              {analysisLayers.ndbiChange && ndbiChangeUrl && realBounds && (
                <ImageOverlay
                  url={ndbiChangeUrl}
                  bounds={realBounds}
                  opacity={opacity / 100}
                  zIndex={330}
                />
              )}

              {/* QUALITY & CLOUD MASK OVERLAY */}
              {analysisLayers.confidenceMap && currentQualityMaskUrl && realBounds && (
                <ImageOverlay
                  url={currentQualityMaskUrl}
                  bounds={realBounds}
                  opacity={0.65}
                  zIndex={400}
                />
              )}

              {/* DETECTED CANDIDATE REGIONS (PHASE 6) */}
              {analysisLayers.detectedRegions && filteredCandidateUrl && realBounds && (
                <ImageOverlay
                  url={filteredCandidateUrl}
                  bounds={realBounds}
                  opacity={0.85}
                  zIndex={450}
                />
              )}
              {analysisLayers.detectedRegions && geojsonFeatures.map((feat: any, fIdx: number) => {
                if (feat.geometry?.type === "Polygon" && Array.isArray(feat.geometry.coordinates?.[0])) {
                  const positions = feat.geometry.coordinates[0].map((pt: number[]) => [Number(pt[1]), Number(pt[0])] as [number, number]);
                  return (
                    <Polygon
                      key={`feat-${feat.id || fIdx}`}
                      positions={positions}
                      pathOptions={{
                        color: feat.properties?.candidate_class === 1 ? "#ff4d4f" : "#ffa940",
                        weight: 2,
                        fillColor: feat.properties?.candidate_class === 1 ? "#ff4d4f" : "#ffa940",
                        fillOpacity: 0.35,
                      }}
                    />
                  );
                }
                return null;
              })}

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

            </MapContainer>
          ) : (
            <div style={{ display: "flex", width: "100%", height: "100%", gap: "6px" }}>
              {/* LEFT: BEFORE SCENE */}
              <div style={{ flex: 1, position: "relative", height: "100%" }}>
                <div style={{ position: "absolute", top: 12, left: 12, zIndex: 1100, background: "rgba(17,17,15,0.85)", color: "#f5f1e9", padding: "4px 10px", borderRadius: 4, fontSize: 11, fontWeight: 700 }}>
                  BEFORE ({dateBefore})
                </div>
                <MapContainer center={mapCenter} zoom={10} zoomControl={false} className="index-map">
                  <TileLayer url={getTileUrl()} />
                  <LayersViewportController bounds={realBounds} center={mapCenter} />
                  {baseLayer === "trueColor" && trueColorBeforeUrl && realBounds && <ImageOverlay url={trueColorBeforeUrl} bounds={realBounds} opacity={1.0} zIndex={100} />}
                  {baseLayer === "falseColor" && falseColorBeforeUrl && realBounds && <ImageOverlay url={falseColorBeforeUrl} bounds={realBounds} opacity={1.0} zIndex={100} />}
                  {currentIndexBeforeUrl && realBounds && <ImageOverlay url={currentIndexBeforeUrl} bounds={realBounds} opacity={opacity / 100} zIndex={200} />}
                  {analysisLayers.aoiBoundary && aoi.length > 0 && <Polygon positions={aoi} pathOptions={{ color: "#f5f1e9", weight: 2, fillOpacity: 0.02 }} />}
                </MapContainer>
              </div>

              {/* RIGHT: AFTER SCENE */}
              <div style={{ flex: 1, position: "relative", height: "100%" }}>
                <div style={{ position: "absolute", top: 12, left: 12, zIndex: 1100, background: "rgba(17,17,15,0.85)", color: "#f5f1e9", padding: "4px 10px", borderRadius: 4, fontSize: 11, fontWeight: 700 }}>
                  AFTER ({dateAfter})
                </div>
                <MapContainer center={mapCenter} zoom={10} zoomControl={true} className="index-map">
                  <TileLayer url={getTileUrl()} />
                  <LayersViewportController bounds={realBounds} center={mapCenter} />
                  {baseLayer === "trueColor" && trueColorAfterUrl && realBounds && <ImageOverlay url={trueColorAfterUrl} bounds={realBounds} opacity={1.0} zIndex={100} />}
                  {baseLayer === "falseColor" && falseColorAfterUrl && realBounds && <ImageOverlay url={falseColorAfterUrl} bounds={realBounds} opacity={1.0} zIndex={100} />}
                  {currentIndexAfterUrl && realBounds && <ImageOverlay url={currentIndexAfterUrl} bounds={realBounds} opacity={opacity / 100} zIndex={200} />}
                  {analysisLayers.aoiBoundary && aoi.length > 0 && <Polygon positions={aoi} pathOptions={{ color: "#f5f1e9", weight: 2, fillOpacity: 0.02 }} />}
                </MapContainer>
              </div>
            </div>
          )}


          <div className="layers-map-badge">

            <span>
              {selectedIndex}
            </span>

            <span>
              {reference === "after"
                ? dateAfter
                : dateBefore}
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

        </section>

        {/* ===================================================
            SCENE PROVENANCE & QUALITY
            =================================================== */}

        <section className="index-info-section" style={{ borderTop: "1px solid var(--soft-line)", paddingTop: "14px", marginTop: "14px" }}>

          <div className="layers-panel-title">
            PROVENANCE
          </div>

          <p style={{ fontSize: "11px", marginBottom: "4px" }}>
            <strong>Scene Date:</strong> {reference === "before" ? dateBefore : dateAfter}
          </p>

          <p style={{ fontSize: "11px", marginBottom: "4px" }}>
            <strong>Cloud Cover:</strong> {currentMetadata?.cloud_cover != null ? `${(Number(currentMetadata.cloud_cover) * 100).toFixed(2)}%` : "0.0%"}
          </p>

          <p style={{ fontSize: "11px", marginBottom: "4px" }}>
            <strong>AOI Valid:</strong> {currentMetadata?.quality?.valid_percentage != null ? `${currentMetadata.quality.valid_percentage}%` : "100%"}
          </p>

          <p style={{ fontSize: "11px", marginBottom: "4px" }}>
            <strong>Platform:</strong> {currentMetadata?.platform || "Sentinel-2 (ESA)"}
          </p>

          <p style={{ fontSize: "11px", marginBottom: "4px" }}>
            <strong>Resolution:</strong> 10m Ground Sample Distance
          </p>

        </section>

        {/* ===================================================
            SPATIAL REGIONS (PHASE 6)
            =================================================== */}
        {spatialData && (
          <section className="index-info-section" style={{ borderTop: "1px solid var(--soft-line)", paddingTop: "14px", marginTop: "14px" }}>
            <div className="layers-panel-title">
              SPATIAL CANDIDATE REGIONS
            </div>
            <p style={{ fontSize: "11px", marginBottom: "4px" }}>
              <strong>Candidate Regions:</strong> {spatialData.region_count ?? 0}
            </p>
            {spatialData.total_candidate_area_hectares != null && (
              <p style={{ fontSize: "11px", marginBottom: "4px" }}>
                <strong>Total Area:</strong> {Number(spatialData.total_candidate_area_hectares).toFixed(2)} ha
              </p>
            )}
            {spatialData.dominant_location_description && (
              <p style={{ fontSize: "11px", marginBottom: "4px" }}>
                <strong>Location:</strong> {spatialData.dominant_location_description}
              </p>
            )}
          </section>
        )}

        {/* ===================================================
            TEMPORAL ANALYSIS (PHASE 7)
            =================================================== */}
        {temporalData && temporalData.available && (
          <section className="index-info-section" style={{ borderTop: "1px solid var(--soft-line)", paddingTop: "14px", marginTop: "14px" }}>
            <div className="layers-panel-title">
              TEMPORAL ANALYSIS
            </div>
            <p style={{ fontSize: "11px", marginBottom: "4px" }}>
              <strong>Observations:</strong> {temporalData.observation_count ?? 0} (Usable: {temporalData.usable_observation_count ?? 0})
            </p>
            <p style={{ fontSize: "11px", marginBottom: "4px" }}>
              <strong>Temporal Mode:</strong> {temporalData.temporal_mode === "multi_temporal" ? "Multi-Temporal Series" : "Bi-Temporal Comparison"}
            </p>
            {temporalData.seasonal_comparability && (
              <p style={{ fontSize: "11px", marginBottom: "4px" }}>
                <strong>Seasonal Match:</strong> {temporalData.seasonal_comparability.comparability?.toUpperCase()} ({temporalData.seasonal_comparability.max_doy_difference}d diff)
              </p>
            )}
            {temporalData.domains && temporalData.primary_domain && (
              <div style={{ marginTop: "6px", fontSize: "11px", background: "rgba(255,255,255,0.03)", padding: "6px", borderRadius: "4px" }}>
                <p style={{ marginBottom: "2px" }}>
                  <strong>{temporalData.primary_domain.toUpperCase()} Trend:</strong> {temporalData.domains[temporalData.primary_domain]?.direction}
                </p>
                {temporalData.domains[temporalData.primary_domain]?.annualized_slope != null && (
                  <p style={{ marginBottom: "2px" }}>
                    <strong>Slope:</strong> {temporalData.domains[temporalData.primary_domain]?.annualized_slope > 0 ? "+" : ""}{temporalData.domains[temporalData.primary_domain]?.annualized_slope}/yr
                  </p>
                )}
                <p style={{ marginBottom: "2px" }}>
                  <strong>Persistence:</strong> {Math.round((temporalData.domains[temporalData.primary_domain]?.persistence_fraction ?? 0) * 100)}%
                </p>
                <p style={{ marginBottom: "0px" }}>
                  <strong>Evolution:</strong> {temporalData.domains[temporalData.primary_domain]?.change_type}
                </p>
              </div>
            )}
          </section>
        )}

        {/* ===================================================
            EVIDENCE & RELIABILITY (PHASE 8 CALIBRATION)
            =================================================== */}
        {calibrationData && (
          <section className="index-info-section" style={{ borderTop: "1px solid var(--soft-line)", paddingTop: "14px", marginTop: "14px" }}>
            <div className="layers-panel-title">
              EVIDENCE &amp; RELIABILITY
            </div>
            <div style={{ marginTop: "6px", fontSize: "11px", background: "rgba(255,255,255,0.03)", padding: "8px", borderRadius: "4px" }}>
              <div style={{ marginBottom: "6px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span><strong>Overall Support:</strong></span>
                <span style={{
                  padding: "2px 6px",
                  borderRadius: "3px",
                  fontSize: "10px",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  background: calibrationData.interpretation_support?.state === "strong_support" ? "rgba(46, 204, 113, 0.2)" :
                              calibrationData.interpretation_support?.state === "moderate_support" ? "rgba(52, 152, 219, 0.2)" :
                              calibrationData.interpretation_support?.state === "contradictory_support" ? "rgba(231, 76, 60, 0.2)" :
                              "rgba(241, 196, 15, 0.2)",
                  color: calibrationData.interpretation_support?.state === "strong_support" ? "#2ecc71" :
                         calibrationData.interpretation_support?.state === "moderate_support" ? "#3498db" :
                         calibrationData.interpretation_support?.state === "contradictory_support" ? "#e74c3c" :
                         "#f1c40f",
                }}>
                  {calibrationData.interpretation_support?.state ? String(calibrationData.interpretation_support.state).replace(/_/g, " ") : "UNAVAILABLE"}
                </span>
              </div>
              <p style={{ marginBottom: "3px" }}>
                <strong>Observation Reliability:</strong> {calibrationData.observation_reliability?.state ? String(calibrationData.observation_reliability.state).toUpperCase() : "N/A"}
              </p>
              <p style={{ marginBottom: "3px" }}>
                <strong>Evidence Strength:</strong> {calibrationData.semantic_evidence?.state ? String(calibrationData.semantic_evidence.state).replace(/_/g, " ").toUpperCase() : "N/A"}
              </p>
              <p style={{ marginBottom: "3px" }}>
                <strong>Spatial Support:</strong> {calibrationData.spatial_assessment?.state ? String(calibrationData.spatial_assessment.state).toUpperCase() : "N/A"}
              </p>
              <p style={{ marginBottom: "3px" }}>
                <strong>Temporal Support:</strong> {calibrationData.temporal_consistency?.state ? String(calibrationData.temporal_consistency.state).replace(/_/g, " ").toUpperCase() : "N/A"}
              </p>
              <p style={{ marginBottom: "6px" }}>
                <strong>Data Sufficiency:</strong> {calibrationData.data_sufficiency?.state ? String(calibrationData.data_sufficiency.state).toUpperCase() : "N/A"}
              </p>

              {/* Why? Section */}
              {calibrationData.reason_codes && calibrationData.reason_codes.length > 0 && (
                <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "6px", marginTop: "6px" }}>
                  <div
                    onClick={() => setShowReasonDetails(!showReasonDetails)}
                    style={{ cursor: "pointer", color: "var(--accent-color, #38bdf8)", fontSize: "10px", fontWeight: 600, display: "flex", justifyContent: "space-between" }}
                  >
                    <span>Why? ({calibrationData.reason_codes.length} active codes)</span>
                    <span>{showReasonDetails ? "▲" : "▼"}</span>
                  </div>
                  {showReasonDetails && (
                    <div style={{ marginTop: "4px", maxHeight: "120px", overflowY: "auto" }}>
                      {calibrationData.reason_codes.map((rc: string, idx: number) => (
                        <div key={idx} style={{ fontSize: "10px", padding: "2px 0", color: "rgba(255,255,255,0.7)" }}>
                          • {rc.replace(/_/g, " ")}
                        </div>
                      ))}
                      {calibrationData.interpretation_support?.summary && (
                        <div style={{ fontSize: "10px", marginTop: "4px", fontStyle: "italic", color: "rgba(255,255,255,0.6)" }}>
                          {calibrationData.interpretation_support.summary}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>
        )}

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

          {/* =================================================
              PHASE 9 BENCHMARK REFERENCE BADGE
              ================================================= */}
          <div style={{ marginTop: "16px", paddingTop: "12px", borderTop: "1px solid rgba(255,255,255,0.08)" }}>
            <button
              type="button"
              onClick={() => setShowBenchmarkModal(true)}
              style={{
                background: "rgba(59, 130, 246, 0.1)",
                color: "#60a5fa",
                border: "1px solid rgba(59, 130, 246, 0.25)",
                borderRadius: "6px",
                padding: "6px 10px",
                cursor: "pointer",
                width: "100%",
                textAlign: "center",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "6px",
                fontSize: "10px",
                fontWeight: 600,
                letterSpacing: "0.5px"
              }}
            >
              <span>🔬</span> BENCHMARK REFERENCE {benchmarkData?.dataset_version ? `(v${benchmarkData.dataset_version})` : ""}
            </button>
          </div>

        </section>

      </aside>

      {/* =================================================
          DEVELOPER / RESEARCH BENCHMARK MODAL
          ================================================= */}
      {showBenchmarkModal && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100vw",
            height: "100vh",
            backgroundColor: "rgba(0, 0, 0, 0.75)",
            backdropFilter: "blur(4px)",
            zIndex: 9999,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "20px",
          }}
          onClick={() => setShowBenchmarkModal(false)}
        >
          <div
            style={{
              background: "#0f172a",
              border: "1px solid #1e293b",
              borderRadius: "12px",
              padding: "24px",
              maxWidth: "560px",
              width: "100%",
              color: "#f8fafc",
              boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "20px" }}>🔬</span>
                <h3 style={{ margin: 0, fontSize: "16px", fontWeight: 600 }}>SatQuery AI: Phase 9 Benchmark Suite</h3>
              </div>
              <button
                type="button"
                onClick={() => setShowBenchmarkModal(false)}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "#94a3b8",
                  cursor: "pointer",
                  fontSize: "18px",
                  padding: "4px 8px",
                }}
              >
                ✕
              </button>
            </div>

            <p style={{ fontSize: "12px", color: "#94a3b8", lineHeight: 1.5, marginBottom: "16px" }}>
              Reproducible scientific benchmark evaluating deterministic change detection against independent Sentinel-2 L2A reference masks across multi-continental scenes.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "16px" }}>
              <div style={{ background: "#1e293b", padding: "10px", borderRadius: "6px" }}>
                <div style={{ fontSize: "10px", color: "#64748b", textTransform: "uppercase" }}>Dataset Version</div>
                <div style={{ fontSize: "13px", fontWeight: 600, color: "#38bdf8" }}>{benchmarkData?.dataset_version || "1.0.0"}</div>
              </div>
              <div style={{ background: "#1e293b", padding: "10px", borderRadius: "6px" }}>
                <div style={{ fontSize: "10px", color: "#64748b", textTransform: "uppercase" }}>Benchmark Status</div>
                <div style={{ fontSize: "11px", fontWeight: 600, color: "#f59e0b" }}>
                  {benchmarkData?.status_message ? "Infrastructure Ready (Pending Labels)" : (benchmarkData?.benchmark_status || "Pending Reference Labels")}
                </div>
              </div>
            </div>

            <div style={{ background: "rgba(15, 23, 42, 0.6)", border: "1px solid #334155", borderRadius: "8px", padding: "12px", marginBottom: "16px" }}>
              <div style={{ fontSize: "11px", fontWeight: 600, color: "#cbd5e1", marginBottom: "8px" }}>Benchmark Status & Baselines:</div>
              {benchmarkData?.baselines ? (
                <table style={{ width: "100%", fontSize: "11px", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ color: "#94a3b8", borderBottom: "1px solid #334155", textAlign: "left" }}>
                      <th style={{ padding: "4px 0" }}>Pipeline</th>
                      <th style={{ padding: "4px 0" }}>Precision</th>
                      <th style={{ padding: "4px 0" }}>Recall</th>
                      <th style={{ padding: "4px 0" }}>Macro F1</th>
                      <th style={{ padding: "4px 0" }}>IoU</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderBottom: "1px solid rgba(51, 65, 85, 0.4)" }}>
                      <td style={{ padding: "6px 0", color: "#e2e8f0" }}>Deterministic SatQuery</td>
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>{benchmarkData.baselines.deterministic_satquery?.macro_precision ?? "—"}</td>
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>{benchmarkData.baselines.deterministic_satquery?.macro_recall ?? "—"}</td>
                      <td style={{ padding: "6px 0", color: "#38bdf8", fontWeight: 600 }}>{benchmarkData.baselines.deterministic_satquery?.macro_f1 ?? "—"}</td>
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>{benchmarkData.baselines.deterministic_satquery?.macro_iou ?? "—"}</td>
                    </tr>
                    <tr>
                      <td style={{ padding: "6px 0", color: "#e2e8f0" }}>Index Threshold Baseline</td>
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>{benchmarkData.baselines.index_threshold?.macro_precision ?? "—"}</td>
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>{benchmarkData.baselines.index_threshold?.macro_recall ?? "—"}</td>
                      <td style={{ padding: "6px 0", color: "#fbbf24", fontWeight: 600 }}>{benchmarkData.baselines.index_threshold?.macro_f1 ?? "—"}</td>
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>{benchmarkData.baselines.index_threshold?.macro_iou ?? "—"}</td>
                    </tr>
                  </tbody>
                </table>
              ) : (
                <div style={{ fontSize: "11px", color: "#94a3b8", lineHeight: 1.5, padding: "4px 0" }}>
                  <div style={{ color: "#38bdf8", fontWeight: 600, marginBottom: "4px" }}>Stage A: Infrastructure Verified</div>
                  <div>Metric engine, region matching, split leakage checks, and error analysis are fully implemented. Numerical evaluation and ML baseline are strictly held in reserve pending validated reference labels from Dynamic World / OSCD.</div>
                  <div style={{ marginTop: "6px", color: "#64748b", fontSize: "10px" }}>ML Status: <code>DEFERRED</code></div>
                </div>
              )}
            </div>

            <div style={{ fontSize: "10px", color: "#64748b", borderLeft: "2px solid #3b82f6", paddingLeft: "8px", lineHeight: 1.4 }}>
              <strong>Scientific Boundary Notice:</strong> Benchmark metrics quantify empirical agreement with authoritative reference masks and do not establish absolute semantic ground truth. Zero data leakage across splits.
            </div>
          </div>
        </div>
      )}

    </main>
  );
}


export default LayersVisualization;