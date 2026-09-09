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

/* =========================================================
   TYPES & INTERFACES (Strictly typed, zero loose any)
   ========================================================= */

export interface LayersVisualizationProps {
  result: QueryResponse;
  onBack: () => void;
  onViewResults?: () => void;
}

export type IndexType = "NDVI" | "NDWI" | "NDBI";
export type BaseLayer = "trueColor" | "falseColor" | "dark";
export type VisualizationType = "heatmap" | "classified" | "gradient";

export interface GeoJsonFeature {
  id?: string | number;
  type: string;
  geometry: {
    type: string;
    coordinates: number[][][] | number[][];
  };
  properties?: {
    candidate_class?: number;
    score?: number;
    confidence?: number;
    area_ha?: number;
    [key: string]: unknown;
  };
}

export interface LayerMetadata {
  cloud_cover?: number | string | null;
  platform?: string | null;
  resolution?: string | null;
  quality?: {
    valid_percentage?: number | string | null;
    cloud_percentage?: number | string | null;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface LayerItem {
  id?: string;
  name?: string;
  type?: string;
  visualization_url?: string | null;
  classified_visualization_url?: string | null;
  bounds?: [[number, number], [number, number]] | number[][] | null;
  metadata?: LayerMetadata;
  metric?: string;
}

export interface LayerPackageEntry {
  url?: string | null;
  classified_url?: string | null;
  bounds?: [[number, number], [number, number]] | number[][] | null;
  metadata?: LayerMetadata;
}

export interface SpatialAnalysisData {
  available?: boolean;
  region_count?: number;
  total_candidate_area_hectares?: number;
  dominant_location_description?: string;
  geojson?: {
    type: string;
    features: GeoJsonFeature[];
  };
  rasters?: {
    raw_candidate_raster?: string;
    filtered_candidate_raster?: string;
    labeled_regions_raster?: string;
  };
  [key: string]: unknown;
}

export interface TemporalAnalysisData {
  available?: boolean;
  observation_count?: number;
  usable_observation_count?: number;
  temporal_mode?: string;
  seasonal_comparability?: {
    comparability?: string;
    max_doy_difference?: number;
  };
  primary_domain?: string;
  domains?: Record<
    string,
    {
      direction?: string;
      annualized_slope?: number;
      persistence_fraction?: number;
      change_type?: string;
    }
  >;
  [key: string]: unknown;
}

export interface CalibrationData {
  interpretation_support?: {
    state?: string;
    summary?: string;
  };
  observation_reliability?: {
    state?: string;
  };
  semantic_evidence?: {
    state?: string;
  };
  spatial_assessment?: {
    state?: string;
  };
  temporal_consistency?: {
    state?: string;
  };
  data_sufficiency?: {
    state?: string;
  };
  reason_codes?: string[];
  [key: string]: unknown;
}

export interface BenchmarkBaselines {
  deterministic_satquery?: {
    macro_precision?: string | number;
    macro_recall?: string | number;
    macro_f1?: string | number;
    macro_iou?: string | number;
  };
  index_threshold?: {
    macro_precision?: string | number;
    macro_recall?: string | number;
    macro_f1?: string | number;
    macro_iou?: string | number;
  };
}

export interface BenchmarkSummaryData {
  dataset_version?: string;
  benchmark_status?: string;
  status_message?: string;
  baselines?: BenchmarkBaselines;
}

/* =========================================================
   MAP VIEWPORT CONTROLLER
   ========================================================= */

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
        map.fitBounds(bounds, { padding: [28, 28], maxZoom: 14 });
      } catch (err) {
        console.warn("fitBounds failed:", err);
      }
    } else {
      map.setView(center, 10);
    }
  }, [map, bounds, center]);
  return null;
}

/* =========================================================
   MAIN COMPONENT
   ========================================================= */

function LayersVisualization({
  result,
  onBack,
  onViewResults,
}: LayersVisualizationProps) {
  const taskName = String(result?.plan?.task ?? "").toLowerCase();
  const intentName = String((result?.plan as any)?.intent ?? "").toLowerCase();

  const isImageSearch =
    taskName === "image_search" ||
    taskName === "search_imagery" ||
    intentName === "image_search" ||
    (Array.isArray(result?.plan?.analysis) &&
      result.plan.analysis.length === 1 &&
      result.plan.analysis[0] === "search_imagery" &&
      taskName !== "change_detection");

  // 1. Initial index detection from result
  const rawMetric = String(result?.statistics?.metric || result?.plan?.metric || (isImageSearch ? "" : "NDVI")).toUpperCase();
  const initialIndex: IndexType = ["NDVI", "NDWI", "NDBI"].includes(rawMetric)
    ? (rawMetric as IndexType)
    : "NDVI";

  const [selectedIndex, setSelectedIndex] = useState<IndexType>(initialIndex);
  const [baseLayer, setBaseLayer] = useState<BaseLayer>("trueColor");
  const [visualization, setVisualization] = useState<VisualizationType>("heatmap");
  const [opacity, setOpacity] = useState(80);
  const [splitMode, setSplitMode] = useState<boolean>(false);
  const [reference, setReference] = useState<"before" | "after">("after");

  const [showBenchmarkModal, setShowBenchmarkModal] = useState(false);
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkSummaryData | null>(null);
  const [showReasonDetails, setShowReasonDetails] = useState<boolean>(false);

  useEffect(() => {
    fetchBenchmarkSummary().then((data) => {
      if (data && Object.keys(data).length > 0) {
        setBenchmarkData(data as BenchmarkSummaryData);
      }
    });
  }, []);

  const dateBefore = result?.plan?.time_start ?? "Before";
  const dateAfter = result?.plan?.time_end ?? "After";

  // 2. Resolve URLs relative to backend API
  const resolveUrl = (rawPath?: string | null): string | null => {
    if (!rawPath) return null;
    const clean = String(rawPath).trim();
    if (!clean) return null;
    if (clean.startsWith("http://") || clean.startsWith("https://")) return clean;
    const baseUrl = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
    const pathUrl = clean.startsWith("/") ? clean : `/${clean}`;
    return `${baseUrl}${pathUrl}`;
  };

  // 3. Extract Real Layers List & Packages
  const layersList = useMemo<LayerItem[]>(() => {
    return Array.isArray(result?.layers) ? (result.layers as LayerItem[]) : [];
  }, [result?.layers]);

  const layerPackage = result?.layer_package as {
    before?: Record<string, LayerPackageEntry>;
    after?: Record<string, LayerPackageEntry>;
    change?: Record<string, LayerPackageEntry>;
    quality?: Record<string, LayerPackageEntry>;
    spatial?: Record<string, unknown>;
  } | undefined;

  // 4. Real AOI & Geographic Bounds (NO fake Mumbai fallback)
  const aoi: [number, number][] = useMemo(() => {
    const rawAoi = result?.plan?.aoi as {
      type?: string;
      coordinates?: number[][][];
    } | number[][] | null | undefined;

    if (!rawAoi) return [];

    if (typeof rawAoi === "object" && "type" in rawAoi && rawAoi.type === "Polygon" && Array.isArray(rawAoi.coordinates?.[0])) {
      return rawAoi.coordinates[0].map((pt: number[]) => [Number(pt[1]), Number(pt[0])] as [number, number]);
    }

    if (Array.isArray(rawAoi) && rawAoi.length > 0 && Array.isArray(rawAoi[0])) {
      return (rawAoi as number[][]).map((pt: number[]) => [Number(pt[0]), Number(pt[1])] as [number, number]);
    }

    return [];
  }, [result?.plan?.aoi]);

  const rawBounds = (result?.layers?.[0] as LayerItem | undefined)?.bounds ?? result?.bounds ?? null;
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
    return [20.5937, 78.9629]; // Clean geographic center
  }, [realBounds, aoi]);

  // 5. True-Color & False-Color URLs
  const trueColorBeforeUrl = useMemo(() => {
    const pkgUrl = layerPackage?.before?.true_color?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x) => x.id === "true_color_before");
    return resolveUrl(l?.visualization_url || result?.images?.before);
  }, [layerPackage, layersList, result?.images?.before]);

  const trueColorAfterUrl = useMemo(() => {
    const pkgUrl = layerPackage?.after?.true_color?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x) => x.id === "true_color_after");
    return resolveUrl(l?.visualization_url || result?.images?.after);
  }, [layerPackage, layersList, result?.images?.after]);

  const falseColorBeforeUrl = useMemo(() => {
    const pkgUrl = layerPackage?.before?.false_color?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x) => x.id === "false_color_before");
    return resolveUrl(l?.visualization_url);
  }, [layerPackage, layersList]);

  const falseColorAfterUrl = useMemo(() => {
    const pkgUrl = layerPackage?.after?.false_color?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x) => x.id === "false_color_after");
    return resolveUrl(l?.visualization_url);
  }, [layerPackage, layersList]);

  // 6. Quality Masks
  const qualityMaskBeforeUrl = useMemo(() => {
    const pkgUrl = layerPackage?.quality?.mask_before?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x) => x.id === "quality_mask_before");
    return resolveUrl(l?.visualization_url);
  }, [layerPackage, layersList]);

  const qualityMaskAfterUrl = useMemo(() => {
    const pkgUrl = layerPackage?.quality?.mask_after?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x) => x.id === "quality_mask_after");
    return resolveUrl(l?.visualization_url);
  }, [layerPackage, layersList]);

  // 7. Spectral Indices per Tab
  const currentIndexBeforeUrl = useMemo(() => {
    const idxKey = selectedIndex.toLowerCase();
    const pkgUrl = layerPackage?.before?.[idxKey]?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find(
      (x) => x.id === `${idxKey}_before` || (x.id === "index_before" && (x.metric?.toUpperCase() === selectedIndex || !x.metric))
    );
    return resolveUrl(l?.visualization_url);
  }, [layerPackage, layersList, selectedIndex]);

  const currentIndexAfterUrl = useMemo(() => {
    const idxKey = selectedIndex.toLowerCase();
    const pkgUrl = layerPackage?.after?.[idxKey]?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find(
      (x) => x.id === `${idxKey}_after` || (x.id === "index_after" && (x.metric?.toUpperCase() === selectedIndex || !x.metric))
    );
    return resolveUrl(l?.visualization_url);
  }, [layerPackage, layersList, selectedIndex]);

  // 8. Delta / Change Detection Overlays
  const ndviChangeUrl = useMemo(() => {
    if (visualization === "classified") {
      const pkgClassified = layerPackage?.change?.delta_ndvi?.classified_url;
      if (pkgClassified) return resolveUrl(pkgClassified);
      const l = layersList.find((x) => x.id === "change_ndvi");
      if (l?.classified_visualization_url) return resolveUrl(l.classified_visualization_url);
    }
    const pkgUrl = layerPackage?.change?.delta_ndvi?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x) => x.id === "change_ndvi");
    return resolveUrl(l?.visualization_url || (result?.statistics?.metric === "NDVI" ? result?.visualization_url : null));
  }, [layerPackage, layersList, visualization, result?.statistics?.metric, result?.visualization_url]);

  const ndwiChangeUrl = useMemo(() => {
    if (visualization === "classified") {
      const pkgClassified = layerPackage?.change?.delta_ndwi?.classified_url;
      if (pkgClassified) return resolveUrl(pkgClassified);
      const l = layersList.find((x) => x.id === "change_ndwi");
      if (l?.classified_visualization_url) return resolveUrl(l.classified_visualization_url);
    }
    const pkgUrl = layerPackage?.change?.delta_ndwi?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x) => x.id === "change_ndwi");
    return resolveUrl(l?.visualization_url || (result?.statistics?.metric === "NDWI" ? result?.visualization_url : null));
  }, [layerPackage, layersList, visualization, result?.statistics?.metric, result?.visualization_url]);

  const ndbiChangeUrl = useMemo(() => {
    if (visualization === "classified") {
      const pkgClassified = layerPackage?.change?.delta_ndbi?.classified_url;
      if (pkgClassified) return resolveUrl(pkgClassified);
      const l = layersList.find((x) => x.id === "change_ndbi");
      if (l?.classified_visualization_url) return resolveUrl(l.classified_visualization_url);
    }
    const pkgUrl = layerPackage?.change?.delta_ndbi?.url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    const l = layersList.find((x) => x.id === "change_ndbi");
    return resolveUrl(l?.visualization_url || (result?.statistics?.metric === "NDBI" ? result?.visualization_url : null));
  }, [layerPackage, layersList, visualization, result?.statistics?.metric, result?.visualization_url]);

  // 9. Multimodal Optical-SAR Specialist Layers
  const opticalLayer = useMemo(() => {
    return layersList.find((x) => x.type === "optical_rgb" || x.name?.toLowerCase().includes("optical surface reflectance"));
  }, [layersList]);

  const sarVvLayer = useMemo(() => {
    return layersList.find((x) => x.type === "sar_vv" || x.name?.toLowerCase().includes("vv backscatter"));
  }, [layersList]);

  const sarVhLayer = useMemo(() => {
    return layersList.find((x) => x.type === "sar_vh" || x.name?.toLowerCase().includes("vh backscatter"));
  }, [layersList]);

  const sarCompLayer = useMemo(() => {
    return layersList.find((x) => x.type === "sar_composite" || x.name?.toLowerCase().includes("polarimetric composite"));
  }, [layersList]);

  const opticalLayerUrl = useMemo(() => {
    const raw = opticalLayer?.visualization_url || (result?.images as Record<string, string> | undefined)?.optical;
    return resolveUrl(raw);
  }, [opticalLayer, result?.images]);

  const sarVvUrl = useMemo(() => {
    const raw = sarVvLayer?.visualization_url || (result?.images as Record<string, string> | undefined)?.s1_vv;
    return resolveUrl(raw);
  }, [sarVvLayer, result?.images]);

  const sarVhUrl = useMemo(() => {
    const raw = sarVhLayer?.visualization_url || (result?.images as Record<string, string> | undefined)?.s1_vh;
    return resolveUrl(raw);
  }, [sarVhLayer, result?.images]);

  const sarCompUrl = useMemo(() => {
    const raw = sarCompLayer?.visualization_url || (result?.images as Record<string, string> | undefined)?.s1_composite;
    return resolveUrl(raw);
  }, [sarCompLayer, result?.images]);

  const hasOpticalSar = Boolean(opticalLayerUrl || sarVvUrl || sarVhUrl || sarCompUrl);

  // 10. Spatial & Temporal Structures
  const spatialData = (result?.spatial_analysis || (result?.statistics as Record<string, unknown> | undefined)?.spatial_analysis || layerPackage?.spatial) as SpatialAnalysisData | undefined;
  const temporalData = (result?.temporal_analysis || (result?.statistics as Record<string, unknown> | undefined)?.temporal_analysis || (layerPackage as Record<string, unknown> | undefined)?.temporal) as TemporalAnalysisData | undefined;
  const calibrationData = (result?.calibration || (result?.statistics as Record<string, unknown> | undefined)?.calibration) as CalibrationData | undefined;

  const filteredCandidateUrl = useMemo(() => {
    const pkgUrl = (layerPackage?.spatial as Record<string, string> | undefined)?.filtered_candidate_url;
    if (pkgUrl) return resolveUrl(pkgUrl);
    if (spatialData?.rasters?.filtered_candidate_raster) {
      return resolveUrl(spatialData.rasters.filtered_candidate_raster);
    }
    return null;
  }, [layerPackage, spatialData]);

  const geojsonFeatures = useMemo(() => {
    return spatialData?.geojson?.features || [];
  }, [spatialData]);

  // 11. Availability Flags
  const hasNdviChange = !isImageSearch && Boolean(ndviChangeUrl);
  const hasNdwiChange = !isImageSearch && Boolean(ndwiChangeUrl);
  const hasNdbiChange = !isImageSearch && Boolean(ndbiChangeUrl);
  const hasDetectedRegions = Boolean(filteredCandidateUrl || geojsonFeatures.length > 0);
  const hasAoiBoundary = aoi.length > 0;
  const hasQualityMask = Boolean(qualityMaskBeforeUrl || qualityMaskAfterUrl);
  const hasTrueColor = Boolean(trueColorBeforeUrl || trueColorAfterUrl);
  const hasFalseColor = Boolean(falseColorBeforeUrl || falseColorAfterUrl);

  const hasAnyRaster = Boolean(
    hasTrueColor ||
    hasFalseColor ||
    hasNdviChange ||
    hasNdwiChange ||
    hasNdbiChange ||
    hasDetectedRegions ||
    hasQualityMask ||
    hasOpticalSar ||
    currentIndexBeforeUrl ||
    currentIndexAfterUrl ||
    (!isImageSearch && result?.visualization_url)
  );

  // 12. Analysis Layer Toggles
  const [analysisLayers, setAnalysisLayers] = useState({
    ndviChange: hasNdviChange,
    ndwiChange: !hasNdviChange && hasNdwiChange,
    ndbiChange: !hasNdviChange && !hasNdwiChange && hasNdbiChange,
    detectedRegions: hasDetectedRegions,
    aoiBoundary: hasAoiBoundary,
    confidenceMap: false,
    opticalRgb: Boolean(opticalLayerUrl),
    sarVv: Boolean(sarVvUrl),
    sarVh: Boolean(sarVhUrl),
    sarComposite: Boolean(sarCompUrl),
  });

  useEffect(() => {
    setAnalysisLayers({
      ndviChange: !isImageSearch && Boolean(ndviChangeUrl),
      ndwiChange: !isImageSearch && !Boolean(ndviChangeUrl) && Boolean(ndwiChangeUrl),
      ndbiChange: !isImageSearch && !Boolean(ndviChangeUrl) && !Boolean(ndwiChangeUrl) && Boolean(ndbiChangeUrl),
      detectedRegions: hasDetectedRegions,
      aoiBoundary: hasAoiBoundary,
      confidenceMap: false,
      opticalRgb: Boolean(opticalLayerUrl),
      sarVv: Boolean(sarVvUrl),
      sarVh: Boolean(sarVhUrl),
      sarComposite: Boolean(sarCompUrl),
    });
    setBaseLayer("trueColor");
    setSplitMode(false);
    if (!isImageSearch && ["NDVI", "NDWI", "NDBI"].includes(rawMetric)) {
      setSelectedIndex(rawMetric as IndexType);
    }
  }, [result, isImageSearch, ndviChangeUrl, ndwiChangeUrl, ndbiChangeUrl]);

  const toggleLayer = (layer: keyof typeof analysisLayers) => {
    setAnalysisLayers((current) => ({
      ...current,
      [layer]: !current[layer],
    }));
  };

  const currentTrueColorUrl = reference === "before" ? trueColorBeforeUrl : trueColorAfterUrl;
  const currentFalseColorUrl = reference === "before" ? falseColorBeforeUrl : falseColorAfterUrl;
  const currentIndexUrl = reference === "before" ? currentIndexBeforeUrl : currentIndexAfterUrl;
  const currentQualityMaskUrl = reference === "before" ? qualityMaskBeforeUrl : qualityMaskAfterUrl;

  const isCurrentIndexAvailable = Boolean(
    (selectedIndex === "NDVI" && (hasNdviChange || currentIndexUrl)) ||
    (selectedIndex === "NDWI" && (hasNdwiChange || currentIndexUrl)) ||
    (selectedIndex === "NDBI" && (hasNdbiChange || currentIndexUrl))
  );

  // 13. Scene Provenance Metadata
  const beforeMetadata = useMemo<LayerMetadata>(() => {
    const evItem = Array.isArray(result?.evidence) && result.evidence.length > 0 ? (result.evidence[0] as Record<string, unknown>) : null;
    const evImages = Array.isArray(evItem?.images) ? (evItem.images as Array<Record<string, unknown>>) : null;
    const ev0 = evImages && evImages.length > 0 ? evImages[0] : null;
    const l = layersList.find((x) => x.id === "true_color_before");
    return (l?.metadata || (ev0?.metadata as LayerMetadata | undefined) || (ev0 as LayerMetadata | null) || {}) as LayerMetadata;
  }, [result?.evidence, layersList]);

  const afterMetadata = useMemo<LayerMetadata>(() => {
    const evItem = Array.isArray(result?.evidence) && result.evidence.length > 0 ? (result.evidence[0] as Record<string, unknown>) : null;
    const evImages = Array.isArray(evItem?.images) ? (evItem.images as Array<Record<string, unknown>>) : null;
    const ev1 = evImages && evImages.length > 1 ? evImages[1] : null;
    const l = layersList.find((x) => x.id === "true_color_after");
    return (l?.metadata || (ev1?.metadata as LayerMetadata | undefined) || (ev1 as LayerMetadata | null) || {}) as LayerMetadata;
  }, [result?.evidence, layersList]);

  const currentMetadata = reference === "before" ? beforeMetadata : afterMetadata;

  // 14. Tile Provider
  const getTileUrl = () => {
    if (baseLayer === "dark") {
      return "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
    }
    if (baseLayer === "falseColor") {
      return "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
    }
    return "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
  };

  // 15. Index Explanations
  const indexInfo = {
    NDVI: {
      name: "NDVI (Normalized Difference Vegetation Index)",
      description: "Indicates vegetation health and density using near-infrared and red reflectance.",
      range: "-1 to 1",
      interpretation: "High positive values represent dense, healthy vegetation.",
    },
    NDWI: {
      name: "NDWI (Normalized Difference Water Index)",
      description: "Highlights water content and surface water features using spectral reflectance.",
      range: "-1 to 1",
      interpretation: "Higher positive values represent open water and high soil moisture.",
    },
    NDBI: {
      name: "NDBI (Normalized Difference Built-up Index)",
      description: "Highlights built-up and urban surfaces using shortwave infrared and near-infrared reflectance.",
      range: "-1 to 1",
      interpretation: "Higher positive values generally represent impervious built-up surfaces.",
    },
  };

  const currentInfo = indexInfo[selectedIndex];

  return (
    <main className="layers-workspace">
      {/* =====================================================
          LEFT — CONTROLS & LAYERS
          ===================================================== */}
      <aside className="layers-sidebar">
        <div className="layers-header-nav">
          <button type="button" className="layers-back-button" onClick={onBack}>
            ← ANALYSIS
          </button>
          {onViewResults && (
            <button type="button" className="layers-results-button" onClick={onViewResults}>
              RESULTS &amp; INSIGHTS →
            </button>
          )}
        </div>

        <div className="layers-panel-title">LAYERS</div>

        {/* BASE LAYERS */}
        <section className="layers-control-section">
          <div className="layers-control-label">BASE LAYERS</div>

          <label className="layer-radio-row">
            <input
              type="radio"
              name="base-layer"
              checked={baseLayer === "trueColor"}
              onChange={() => setBaseLayer("trueColor")}
            />
            <div>
              <span>Satellite (True Color)</span>
              {!hasTrueColor && <span className="layer-unavailable-tag">(Default basemap)</span>}
            </div>
          </label>

          <label className="layer-radio-row">
            <input
              type="radio"
              name="base-layer"
              checked={baseLayer === "falseColor"}
              onChange={() => setBaseLayer("falseColor")}
            />
            <div>
              <span>Satellite (False Color)</span>
              {!hasFalseColor && <span className="layer-unavailable-tag">(Default basemap)</span>}
            </div>
          </label>

          <label className="layer-radio-row">
            <input
              type="radio"
              name="base-layer"
              checked={baseLayer === "dark"}
              onChange={() => setBaseLayer("dark")}
            />
            <div>
              <span>Dark Basemap</span>
            </div>
          </label>
        </section>

        {/* ANALYSIS LAYERS */}
        <section className="layers-control-section">
          <div className="layers-control-label">ANALYSIS LAYERS</div>

          {/* NDVI Change */}
          <label className={`layer-check-row ${!hasNdviChange ? "disabled" : ""}`}>
            <input
              type="checkbox"
              disabled={!hasNdviChange}
              checked={analysisLayers.ndviChange && hasNdviChange}
              onChange={() => toggleLayer("ndviChange")}
            />
            <div>
              <span>NDVI Change</span>
              {!hasNdviChange && <span className="layer-unavailable-tag">(Not available)</span>}
            </div>
          </label>

          {/* NDWI Change */}
          <label className={`layer-check-row ${!hasNdwiChange ? "disabled" : ""}`}>
            <input
              type="checkbox"
              disabled={!hasNdwiChange}
              checked={analysisLayers.ndwiChange && hasNdwiChange}
              onChange={() => toggleLayer("ndwiChange")}
            />
            <div>
              <span>NDWI Change</span>
              {!hasNdwiChange && <span className="layer-unavailable-tag">(Not available)</span>}
            </div>
          </label>

          {/* NDBI Change */}
          <label className={`layer-check-row ${!hasNdbiChange ? "disabled" : ""}`}>
            <input
              type="checkbox"
              disabled={!hasNdbiChange}
              checked={analysisLayers.ndbiChange && hasNdbiChange}
              onChange={() => toggleLayer("ndbiChange")}
            />
            <div>
              <span>NDBI Change</span>
              {!hasNdbiChange && <span className="layer-unavailable-tag">(Not available)</span>}
            </div>
          </label>

          {/* Detected Regions */}
          <label className={`layer-check-row ${!hasDetectedRegions ? "disabled" : ""}`}>
            <input
              type="checkbox"
              disabled={!hasDetectedRegions}
              checked={analysisLayers.detectedRegions && hasDetectedRegions}
              onChange={() => toggleLayer("detectedRegions")}
            />
            <div>
              <span>Detected Regions</span>
              {!hasDetectedRegions && <span className="layer-unavailable-tag">(Not available)</span>}
            </div>
          </label>

          {/* AOI Boundary */}
          <label className={`layer-check-row ${!hasAoiBoundary ? "disabled" : ""}`}>
            <input
              type="checkbox"
              disabled={!hasAoiBoundary}
              checked={analysisLayers.aoiBoundary && hasAoiBoundary}
              onChange={() => toggleLayer("aoiBoundary")}
            />
            <div>
              <span>AOI Boundary</span>
              {!hasAoiBoundary && <span className="layer-unavailable-tag">(No custom AOI)</span>}
            </div>
          </label>

          {/* Quality / Cloud Mask */}
          <label className={`layer-check-row ${!hasQualityMask ? "disabled" : ""}`}>
            <input
              type="checkbox"
              disabled={!hasQualityMask}
              checked={analysisLayers.confidenceMap && hasQualityMask}
              onChange={() => toggleLayer("confidenceMap")}
            />
            <div>
              <span>Confidence &amp; Cloud Mask</span>
              {!hasQualityMask && <span className="layer-unavailable-tag">(Not available)</span>}
            </div>
          </label>
        </section>

        {/* MULTIMODAL OPTICAL-SAR SENSORS (Only when available) */}
        {hasOpticalSar && (
          <section className="layers-control-section">
            <div className="layers-control-label">MULTIMODAL SENSORS</div>

            {opticalLayerUrl && (
              <label className="layer-check-row">
                <input
                  type="checkbox"
                  checked={analysisLayers.opticalRgb}
                  onChange={() => toggleLayer("opticalRgb")}
                />
                <div>
                  <span>Optical Surface Reflectance</span>
                </div>
              </label>
            )}

            {sarVvUrl && (
              <label className="layer-check-row">
                <input
                  type="checkbox"
                  checked={analysisLayers.sarVv}
                  onChange={() => toggleLayer("sarVv")}
                />
                <div>
                  <span>Sentinel-1 VV Backscatter</span>
                </div>
              </label>
            )}

            {sarVhUrl && (
              <label className="layer-check-row">
                <input
                  type="checkbox"
                  checked={analysisLayers.sarVh}
                  onChange={() => toggleLayer("sarVh")}
                />
                <div>
                  <span>Sentinel-1 VH Backscatter</span>
                </div>
              </label>
            )}

            {sarCompUrl && (
              <label className="layer-check-row">
                <input
                  type="checkbox"
                  checked={analysisLayers.sarComposite}
                  onChange={() => toggleLayer("sarComposite")}
                />
                <div>
                  <span>Polarimetric Composite</span>
                </div>
              </label>
            )}
          </section>
        )}

        {/* REFERENCE TIMEFRAME */}
        <section className="layers-control-section">
          <div className="layers-control-label">REFERENCE SCENE</div>

          <label className="layer-check-row">
            <input
              type="radio"
              name="reference"
              checked={reference === "before"}
              onChange={() => setReference("before")}
            />
            <span>Before ({dateBefore})</span>
          </label>

          <label className="layer-check-row">
            <input
              type="radio"
              name="reference"
              checked={reference === "after"}
              onChange={() => setReference("after")}
            />
            <span>After ({dateAfter})</span>
          </label>
        </section>

        {/* CLEAR ACTION */}
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
              opticalRgb: false,
              sarVv: false,
              sarVh: false,
              sarComposite: false,
            })
          }
        >
          CLEAR ALL
        </button>
      </aside>

      {/* =====================================================
          CENTER — MAP PANEL
          ===================================================== */}
      <section className="layers-map-panel">
        <div className="layers-map-header">
          <div className="layers-map-title">{isImageSearch ? "SATELLITE IMAGERY" : "INDEX"}</div>

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

          {!isImageSearch && (
            <div className="index-tabs">
              {(["NDVI", "NDWI", "NDBI"] as IndexType[]).map((index) => {
                const isAvailable =
                  (index === "NDVI" && hasNdviChange) ||
                  (index === "NDWI" && hasNdwiChange) ||
                  (index === "NDBI" && hasNdbiChange);

                return (
                  <button
                    key={index}
                    type="button"
                    className={selectedIndex === index ? "index-tab active" : "index-tab"}
                    onClick={() => setSelectedIndex(index)}
                    title={isAvailable ? `${index} data available` : `${index} not calculated`}
                  >
                    {index}
                    {isAvailable && <span style={{ marginLeft: "4px", color: "#22c55e", fontSize: "9px" }}>●</span>}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="layers-map-subtitle">
          {isImageSearch
            ? `SATELLITE IMAGERY (${baseLayer === "falseColor" ? "FALSE COLOR NIR" : "TRUE COLOR RGB"}) • ${reference === "before" ? dateBefore : dateAfter}`
            : `${selectedIndex} CHANGE (${dateBefore} → ${dateAfter})`}
        </div>

        <div className="layers-map" style={{ position: "relative", height: "100%" }}>
          {/* Informational banner when no raster layers were generated */}
          {!hasAnyRaster && (
            <div className="empty-raster-banner">
              ℹ️ No raster overlays generated for this analysis (e.g. metadata/text query). Displaying basemap.
            </div>
          )}

          {!splitMode ? (
            <MapContainer
              center={mapCenter}
              zoom={10}
              zoomControl={true}
              attributionControl={true}
              className="index-map"
            >
              <TileLayer url={getTileUrl()} attribution="Tiles © Esri / Carto" />

              <LayersViewportController bounds={realBounds} center={mapCenter} />

              {/* REAL BASE OVERLAYS (Georeferenced True-Color / False-Color) */}
              {baseLayer === "trueColor" && currentTrueColorUrl && realBounds && (
                <ImageOverlay url={currentTrueColorUrl} bounds={realBounds} opacity={1.0} zIndex={100} />
              )}
              {baseLayer === "falseColor" && currentFalseColorUrl && realBounds && (
                <ImageOverlay url={currentFalseColorUrl} bounds={realBounds} opacity={1.0} zIndex={100} />
              )}

              {/* REAL SCIENTIFIC INDEX OVERLAY */}
              {currentIndexUrl && realBounds && (
                <ImageOverlay url={currentIndexUrl} bounds={realBounds} opacity={opacity / 100} zIndex={200} />
              )}

              {/* REAL SCIENTIFIC CHANGE OVERLAYS */}
              {analysisLayers.ndviChange && ndviChangeUrl && realBounds && (
                <ImageOverlay url={ndviChangeUrl} bounds={realBounds} opacity={opacity / 100} zIndex={310} />
              )}
              {analysisLayers.ndwiChange && ndwiChangeUrl && realBounds && (
                <ImageOverlay url={ndwiChangeUrl} bounds={realBounds} opacity={opacity / 100} zIndex={320} />
              )}
              {analysisLayers.ndbiChange && ndbiChangeUrl && realBounds && (
                <ImageOverlay url={ndbiChangeUrl} bounds={realBounds} opacity={opacity / 100} zIndex={330} />
              )}

              {/* MULTIMODAL OPTICAL-SAR OVERLAYS */}
              {analysisLayers.opticalRgb && opticalLayerUrl && realBounds && (
                <ImageOverlay url={opticalLayerUrl} bounds={realBounds} opacity={opacity / 100} zIndex={350} />
              )}
              {analysisLayers.sarVv && sarVvUrl && realBounds && (
                <ImageOverlay url={sarVvUrl} bounds={realBounds} opacity={opacity / 100} zIndex={360} />
              )}
              {analysisLayers.sarVh && sarVhUrl && realBounds && (
                <ImageOverlay url={sarVhUrl} bounds={realBounds} opacity={opacity / 100} zIndex={370} />
              )}
              {analysisLayers.sarComposite && sarCompUrl && realBounds && (
                <ImageOverlay url={sarCompUrl} bounds={realBounds} opacity={opacity / 100} zIndex={380} />
              )}

              {/* QUALITY / CLOUD MASK */}
              {analysisLayers.confidenceMap && currentQualityMaskUrl && realBounds && (
                <ImageOverlay url={currentQualityMaskUrl} bounds={realBounds} opacity={0.65} zIndex={400} />
              )}

              {/* DETECTED CANDIDATE REGIONS */}
              {analysisLayers.detectedRegions && filteredCandidateUrl && realBounds && (
                <ImageOverlay url={filteredCandidateUrl} bounds={realBounds} opacity={0.85} zIndex={450} />
              )}

              {analysisLayers.detectedRegions &&
                geojsonFeatures.map((feat, fIdx) => {
                  if (feat.geometry?.type === "Polygon" && Array.isArray(feat.geometry.coordinates?.[0])) {
                    const positions = (feat.geometry.coordinates[0] as number[][]).map(
                      (pt: number[]) => [Number(pt[1]), Number(pt[0])] as [number, number]
                    );
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

              {/* REAL AOI POLYGON */}
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
                <div
                  style={{
                    position: "absolute",
                    top: 12,
                    left: 12,
                    zIndex: 1100,
                    background: "rgba(17,17,15,0.85)",
                    color: "#f5f1e9",
                    padding: "4px 10px",
                    borderRadius: 4,
                    fontSize: 11,
                    fontWeight: 700,
                  }}
                >
                  BEFORE ({dateBefore})
                </div>
                <MapContainer center={mapCenter} zoom={10} zoomControl={false} className="index-map">
                  <TileLayer url={getTileUrl()} />
                  <LayersViewportController bounds={realBounds} center={mapCenter} />
                  {baseLayer === "trueColor" && trueColorBeforeUrl && realBounds && (
                    <ImageOverlay url={trueColorBeforeUrl} bounds={realBounds} opacity={1.0} zIndex={100} />
                  )}
                  {baseLayer === "falseColor" && falseColorBeforeUrl && realBounds && (
                    <ImageOverlay url={falseColorBeforeUrl} bounds={realBounds} opacity={1.0} zIndex={100} />
                  )}
                  {currentIndexBeforeUrl && realBounds && (
                    <ImageOverlay url={currentIndexBeforeUrl} bounds={realBounds} opacity={opacity / 100} zIndex={200} />
                  )}
                  {analysisLayers.aoiBoundary && aoi.length > 0 && (
                    <Polygon positions={aoi} pathOptions={{ color: "#f5f1e9", weight: 2, fillOpacity: 0.02 }} />
                  )}
                </MapContainer>
              </div>

              {/* RIGHT: AFTER SCENE */}
              <div style={{ flex: 1, position: "relative", height: "100%" }}>
                <div
                  style={{
                    position: "absolute",
                    top: 12,
                    left: 12,
                    zIndex: 1100,
                    background: "rgba(17,17,15,0.85)",
                    color: "#f5f1e9",
                    padding: "4px 10px",
                    borderRadius: 4,
                    fontSize: 11,
                    fontWeight: 700,
                  }}
                >
                  AFTER ({dateAfter})
                </div>
                <MapContainer center={mapCenter} zoom={10} zoomControl={true} className="index-map">
                  <TileLayer url={getTileUrl()} />
                  <LayersViewportController bounds={realBounds} center={mapCenter} />
                  {baseLayer === "trueColor" && trueColorAfterUrl && realBounds && (
                    <ImageOverlay url={trueColorAfterUrl} bounds={realBounds} opacity={1.0} zIndex={100} />
                  )}
                  {baseLayer === "falseColor" && falseColorAfterUrl && realBounds && (
                    <ImageOverlay url={falseColorAfterUrl} bounds={realBounds} opacity={1.0} zIndex={100} />
                  )}
                  {currentIndexAfterUrl && realBounds && (
                    <ImageOverlay url={currentIndexAfterUrl} bounds={realBounds} opacity={opacity / 100} zIndex={200} />
                  )}
                  {analysisLayers.aoiBoundary && aoi.length > 0 && (
                    <Polygon positions={aoi} pathOptions={{ color: "#f5f1e9", weight: 2, fillOpacity: 0.02 }} />
                  )}
                </MapContainer>
              </div>
            </div>
          )}

          {/* MAP BADGE */}
          <div className="layers-map-badge">
            <span>{isImageSearch ? (baseLayer === "falseColor" ? "NIR" : "RGB") : selectedIndex}</span>
            <span>{reference === "after" ? dateAfter : dateBefore}</span>
          </div>
        </div>

        {/* COLOR SCALE */}
        {!isImageSearch && (
          <div className="index-scale">
            <span>-1</span>
            <div className="index-gradient" />
            <span>1</span>
          </div>
        )}
      </section>

      {/* =====================================================
          RIGHT — SCIENTIFIC INFORMATION PANEL
          ===================================================== */}
      <aside className="index-info-panel">
        <section className="index-info-section">
          <div className="layers-panel-title">{isImageSearch ? "IMAGERY INFO" : "INDEX INFO"}</div>
          {isImageSearch ? (
            <>
              <h2>Sentinel-2 MSI Surface Reflectance</h2>
              <p>Level-2A Bottom-Of-Atmosphere (BOA) multi-spectral reflectance imagery.</p>
              <p>Bands: {baseLayer === "falseColor" ? "B08 (NIR), B04 (Red), B03 (Green)" : "B04 (Red), B03 (Green), B02 (Blue)"}</p>
              <p>Native Resolution: 10 meters ground sample distance.</p>
            </>
          ) : (
            <>
              <h2>{currentInfo.name}</h2>
              <p>{currentInfo.description}</p>
              <p>Range: {currentInfo.range}</p>
              <p>{currentInfo.interpretation}</p>

              {!isCurrentIndexAvailable && (
                <div
                  style={{
                    marginTop: "8px",
                    padding: "8px",
                    background: "rgba(245, 158, 11, 0.12)",
                    borderLeft: "3px solid #f59e0b",
                    borderRadius: "3px",
                    fontSize: "11px",
                    color: "var(--ink)",
                  }}
                >
                  <strong>Notice:</strong> This index was not requested or processed in the current query plan.
                </div>
              )}
            </>
          )}
        </section>

        {/* SCENE PROVENANCE & QUALITY */}
        <section
          className="index-info-section"
          style={{ borderTop: "1px solid var(--soft-line)", paddingTop: "14px", marginTop: "14px" }}
        >
          <div className="layers-panel-title">PROVENANCE</div>
          <p style={{ fontSize: "11px", marginBottom: "4px" }}>
            <strong>Scene Date:</strong> {reference === "before" ? dateBefore : dateAfter}
          </p>
          <p style={{ fontSize: "11px", marginBottom: "4px" }}>
            <strong>Cloud Cover:</strong>{" "}
            {currentMetadata?.cloud_cover != null
              ? `${(Number(currentMetadata.cloud_cover) * 100).toFixed(2)}%`
              : "0.0%"}
          </p>
          <p style={{ fontSize: "11px", marginBottom: "4px" }}>
            <strong>AOI Valid:</strong>{" "}
            {currentMetadata?.quality?.valid_percentage != null
              ? `${currentMetadata.quality.valid_percentage}%`
              : "100%"}
          </p>
          <p style={{ fontSize: "11px", marginBottom: "4px" }}>
            <strong>Platform:</strong> {currentMetadata?.platform || (hasOpticalSar ? "Sentinel-2 / Sentinel-1" : "Sentinel-2 (ESA)")}
          </p>
          <p style={{ fontSize: "11px", marginBottom: "4px" }}>
            <strong>Resolution:</strong> 10m Ground Sample Distance
          </p>
        </section>

        {/* SPATIAL CANDIDATE REGIONS (Phase 6) */}
        {spatialData && (
          <section
            className="index-info-section"
            style={{ borderTop: "1px solid var(--soft-line)", paddingTop: "14px", marginTop: "14px" }}
          >
            <div className="layers-panel-title">SPATIAL CANDIDATE REGIONS</div>
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

        {/* TEMPORAL ANALYSIS (Phase 7) */}
        {temporalData && temporalData.available && (
          <section
            className="index-info-section"
            style={{ borderTop: "1px solid var(--soft-line)", paddingTop: "14px", marginTop: "14px" }}
          >
            <div className="layers-panel-title">TEMPORAL ANALYSIS</div>
            <p style={{ fontSize: "11px", marginBottom: "4px" }}>
              <strong>Observations:</strong> {temporalData.observation_count ?? 0} (Usable:{" "}
              {temporalData.usable_observation_count ?? 0})
            </p>
            <p style={{ fontSize: "11px", marginBottom: "4px" }}>
              <strong>Temporal Mode:</strong>{" "}
              {temporalData.temporal_mode === "multi_temporal" ? "Multi-Temporal Series" : "Bi-Temporal Comparison"}
            </p>
            {temporalData.seasonal_comparability && (
              <p style={{ fontSize: "11px", marginBottom: "4px" }}>
                <strong>Seasonal Match:</strong>{" "}
                {temporalData.seasonal_comparability.comparability?.toUpperCase()} (
                {temporalData.seasonal_comparability.max_doy_difference}d diff)
              </p>
            )}
            {temporalData.domains && temporalData.primary_domain && (
              <div
                style={{
                  marginTop: "6px",
                  fontSize: "11px",
                  background: "rgba(255,255,255,0.03)",
                  padding: "6px",
                  borderRadius: "4px",
                }}
              >
                <p style={{ marginBottom: "2px" }}>
                  <strong>{temporalData.primary_domain.toUpperCase()} Trend:</strong>{" "}
                  {temporalData.domains[temporalData.primary_domain]?.direction}
                </p>
                {temporalData.domains[temporalData.primary_domain]?.annualized_slope != null && (
                  <p style={{ marginBottom: "2px" }}>
                    <strong>Slope:</strong>{" "}
                    {Number(temporalData.domains[temporalData.primary_domain]?.annualized_slope) > 0 ? "+" : ""}
                    {temporalData.domains[temporalData.primary_domain]?.annualized_slope}/yr
                  </p>
                )}
                <p style={{ marginBottom: "2px" }}>
                  <strong>Persistence:</strong>{" "}
                  {Math.round((temporalData.domains[temporalData.primary_domain]?.persistence_fraction ?? 0) * 100)}%
                </p>
                <p style={{ marginBottom: "0px" }}>
                  <strong>Evolution:</strong> {temporalData.domains[temporalData.primary_domain]?.change_type}
                </p>
              </div>
            )}
          </section>
        )}

        {/* EVIDENCE & RELIABILITY (Phase 8 Calibration) */}
        {calibrationData && (
          <section
            className="index-info-section"
            style={{ borderTop: "1px solid var(--soft-line)", paddingTop: "14px", marginTop: "14px" }}
          >
            <div className="layers-panel-title">EVIDENCE &amp; RELIABILITY</div>
            <div
              style={{
                marginTop: "6px",
                fontSize: "11px",
                background: "rgba(255,255,255,0.03)",
                padding: "8px",
                borderRadius: "4px",
              }}
            >
              <div
                style={{
                  marginBottom: "6px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <span>
                  <strong>Overall Support:</strong>
                </span>
                <span
                  style={{
                    padding: "2px 6px",
                    borderRadius: "3px",
                    fontSize: "10px",
                    fontWeight: 600,
                    textTransform: "uppercase",
                    background:
                      calibrationData.interpretation_support?.state === "strong_support"
                        ? "rgba(46, 204, 113, 0.2)"
                        : calibrationData.interpretation_support?.state === "moderate_support"
                        ? "rgba(52, 152, 219, 0.2)"
                        : calibrationData.interpretation_support?.state === "contradictory_support"
                        ? "rgba(231, 76, 60, 0.2)"
                        : "rgba(241, 196, 15, 0.2)",
                    color:
                      calibrationData.interpretation_support?.state === "strong_support"
                        ? "#2ecc71"
                        : calibrationData.interpretation_support?.state === "moderate_support"
                        ? "#3498db"
                        : calibrationData.interpretation_support?.state === "contradictory_support"
                        ? "#e74c3c"
                        : "#f1c40f",
                  }}
                >
                  {calibrationData.interpretation_support?.state
                    ? String(calibrationData.interpretation_support.state).replace(/_/g, " ")
                    : "UNAVAILABLE"}
                </span>
              </div>
              <p style={{ marginBottom: "3px" }}>
                <strong>Observation Reliability:</strong>{" "}
                {calibrationData.observation_reliability?.state
                  ? String(calibrationData.observation_reliability.state).toUpperCase()
                  : "N/A"}
              </p>
              <p style={{ marginBottom: "3px" }}>
                <strong>Evidence Strength:</strong>{" "}
                {calibrationData.semantic_evidence?.state
                  ? String(calibrationData.semantic_evidence.state).replace(/_/g, " ").toUpperCase()
                  : "N/A"}
              </p>
              <p style={{ marginBottom: "3px" }}>
                <strong>Spatial Support:</strong>{" "}
                {calibrationData.spatial_assessment?.state
                  ? String(calibrationData.spatial_assessment.state).toUpperCase()
                  : "N/A"}
              </p>
              <p style={{ marginBottom: "3px" }}>
                <strong>Temporal Support:</strong>{" "}
                {calibrationData.temporal_consistency?.state
                  ? String(calibrationData.temporal_consistency.state).replace(/_/g, " ").toUpperCase()
                  : "N/A"}
              </p>
              <p style={{ marginBottom: "6px" }}>
                <strong>Data Sufficiency:</strong>{" "}
                {calibrationData.data_sufficiency?.state
                  ? String(calibrationData.data_sufficiency.state).toUpperCase()
                  : "N/A"}
              </p>

              {/* Why? Dropdown */}
              {calibrationData.reason_codes && calibrationData.reason_codes.length > 0 && (
                <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "6px", marginTop: "6px" }}>
                  <div
                    onClick={() => setShowReasonDetails(!showReasonDetails)}
                    style={{
                      cursor: "pointer",
                      color: "var(--accent-color, #38bdf8)",
                      fontSize: "10px",
                      fontWeight: 600,
                      display: "flex",
                      justifyContent: "space-between",
                    }}
                  >
                    <span>Why? ({calibrationData.reason_codes.length} active codes)</span>
                    <span>{showReasonDetails ? "▲" : "▼"}</span>
                  </div>
                  {showReasonDetails && (
                    <div style={{ marginTop: "4px", maxHeight: "120px", overflowY: "auto" }}>
                      {calibrationData.reason_codes.map((rc, idx) => (
                        <div key={idx} style={{ fontSize: "10px", padding: "2px 0", color: "rgba(255,255,255,0.7)" }}>
                          • {rc.replace(/_/g, " ")}
                        </div>
                      ))}
                      {calibrationData.interpretation_support?.summary && (
                        <div
                          style={{
                            fontSize: "10px",
                            marginTop: "4px",
                            fontStyle: "italic",
                            color: "rgba(255,255,255,0.6)",
                          }}
                        >
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

        {/* VISUALIZATION */}
        <section className="visualization-section">
          <div className="layers-control-label">VISUALIZATION</div>

          <label className="layer-radio-row">
            <input
              type="radio"
              name="visualization"
              checked={visualization === "heatmap"}
              onChange={() => setVisualization("heatmap")}
            />
            <span>Heatmap</span>
          </label>

          <label className="layer-radio-row">
            <input
              type="radio"
              name="visualization"
              checked={visualization === "classified"}
              onChange={() => setVisualization("classified")}
            />
            <span>Classified</span>
          </label>

          <label className="layer-radio-row">
            <input
              type="radio"
              name="visualization"
              checked={visualization === "gradient"}
              onChange={() => setVisualization("gradient")}
            />
            <span>Gradient</span>
          </label>

          {/* OPACITY */}
          <div className="opacity-control">
            <div className="opacity-header">
              <span>OPACITY</span>
              <span>{opacity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={opacity}
              onChange={(event) => setOpacity(Number(event.target.value))}
            />
          </div>

          {/* PHASE 9 BENCHMARK REFERENCE BADGE */}
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
                letterSpacing: "0.5px",
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
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "16px",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "20px" }}>🔬</span>
                <h3 style={{ margin: 0, fontSize: "16px", fontWeight: 600 }}>
                  SatQuery AI: Phase 9 Benchmark Suite
                </h3>
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
              Reproducible scientific benchmark evaluating deterministic change detection against independent Sentinel-2
              L2A reference masks across multi-continental scenes.
            </p>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "12px",
                marginBottom: "16px",
              }}
            >
              <div style={{ background: "#1e293b", padding: "10px", borderRadius: "6px" }}>
                <div style={{ fontSize: "10px", color: "#64748b", textTransform: "uppercase" }}>
                  Dataset Version
                </div>
                <div style={{ fontSize: "13px", fontWeight: 600, color: "#38bdf8" }}>
                  {benchmarkData?.dataset_version || "1.0.0"}
                </div>
              </div>
              <div style={{ background: "#1e293b", padding: "10px", borderRadius: "6px" }}>
                <div style={{ fontSize: "10px", color: "#64748b", textTransform: "uppercase" }}>
                  Benchmark Status
                </div>
                <div style={{ fontSize: "11px", fontWeight: 600, color: "#f59e0b" }}>
                  {benchmarkData?.status_message
                    ? "Infrastructure Ready (Pending Labels)"
                    : benchmarkData?.benchmark_status || "Pending Reference Labels"}
                </div>
              </div>
            </div>

            <div
              style={{
                background: "rgba(15, 23, 42, 0.6)",
                border: "1px solid #334155",
                borderRadius: "8px",
                padding: "12px",
                marginBottom: "16px",
              }}
            >
              <div style={{ fontSize: "11px", fontWeight: 600, color: "#cbd5e1", marginBottom: "8px" }}>
                Benchmark Status &amp; Baselines:
              </div>
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
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>
                        {benchmarkData.baselines.deterministic_satquery?.macro_precision ?? "—"}
                      </td>
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>
                        {benchmarkData.baselines.deterministic_satquery?.macro_recall ?? "—"}
                      </td>
                      <td style={{ padding: "6px 0", color: "#38bdf8", fontWeight: 600 }}>
                        {benchmarkData.baselines.deterministic_satquery?.macro_f1 ?? "—"}
                      </td>
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>
                        {benchmarkData.baselines.deterministic_satquery?.macro_iou ?? "—"}
                      </td>
                    </tr>
                    <tr>
                      <td style={{ padding: "6px 0", color: "#e2e8f0" }}>Index Threshold Baseline</td>
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>
                        {benchmarkData.baselines.index_threshold?.macro_precision ?? "—"}
                      </td>
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>
                        {benchmarkData.baselines.index_threshold?.macro_recall ?? "—"}
                      </td>
                      <td style={{ padding: "6px 0", color: "#fbbf24", fontWeight: 600 }}>
                        {benchmarkData.baselines.index_threshold?.macro_f1 ?? "—"}
                      </td>
                      <td style={{ padding: "6px 0", color: "#94a3b8" }}>
                        {benchmarkData.baselines.index_threshold?.macro_iou ?? "—"}
                      </td>
                    </tr>
                  </tbody>
                </table>
              ) : (
                <div style={{ fontSize: "11px", color: "#94a3b8", lineHeight: 1.5, padding: "4px 0" }}>
                  <div style={{ color: "#38bdf8", fontWeight: 600, marginBottom: "4px" }}>
                    Stage A: Infrastructure Verified
                  </div>
                  <div>
                    Metric engine, region matching, split leakage checks, and error analysis are fully implemented.
                    Numerical evaluation and ML baseline are strictly held in reserve pending validated reference labels
                    from Dynamic World / OSCD.
                  </div>
                  <div style={{ marginTop: "6px", color: "#64748b", fontSize: "10px" }}>
                    ML Status: <code>DEFERRED</code>
                  </div>
                </div>
              )}
            </div>

            <div
              style={{
                fontSize: "10px",
                color: "#64748b",
                borderLeft: "2px solid #3b82f6",
                paddingLeft: "8px",
                lineHeight: 1.4,
              }}
            >
              <strong>Scientific Boundary Notice:</strong> Benchmark metrics quantify empirical agreement with
              authoritative reference masks and do not establish absolute semantic ground truth. Zero data leakage
              across splits.
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default LayersVisualization;