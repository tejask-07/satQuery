import { useState, useEffect, useMemo, useRef } from "react";
import type L from "leaflet";

import type { QueryResponse } from "../../api/query";

import {
  MapContainer,
  TileLayer,
  Polygon,
  Rectangle,
  ImageOverlay,
  useMap,
  useMapEvents,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./AnalysisWorkspace.css";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export interface AoiGeoJson {
  type: "Polygon";
  coordinates: number[][][];
}

interface AnalysisWorkspaceProps {
  result: QueryResponse;
  currentQuery?: string;
  onViewDetails: () => void;
  onViewLayers?: () => void;
  onRequery?: (query: string, aoi?: unknown) => Promise<void>;
  loading?: boolean;
}

type LatLng = [number, number];
type LeafletBounds = [LatLng, LatLng];

/* ============================================================
   MAP VIEWPORT CONTROLLER
   ============================================================ */

function MapViewportController({
  bounds,
  center,
}: {
  bounds: LeafletBounds | null;
  center: LatLng;
}) {
  const map = useMap();

  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, {
        padding: [40, 40],
        maxZoom: 15,
        animate: false,
      });
    } else {
      map.setView(center, 10, {
        animate: false,
      });
    }
  }, [bounds, center, map]);

  return null;
}

/* ============================================================
   AOI DRAW HANDLER
   ============================================================ */

interface AoiDrawHandlerProps {
  isDrawing: boolean;
  onAoiDrawn: (aoi: AoiGeoJson) => void;
  onDrawingCancel: () => void;
}

function AoiDrawHandler({
  isDrawing,
  onAoiDrawn,
  onDrawingCancel,
}: AoiDrawHandlerProps) {
  const map = useMap();

  const [startPoint, setStartPoint] =
    useState<L.LatLng | null>(null);

  const [currentPoint, setCurrentPoint] =
    useState<L.LatLng | null>(null);

  const isShiftDraggingRef = useRef(false);

  // Restore and guarantee map dragging and controls are enabled
  useEffect(() => {
    const container = map.getContainer();

    // Map dragging, touch zoom, and scroll zoom must stay ENABLED so user can pan/zoom freely
    map.dragging.enable();
    map.touchZoom.enable();
    map.scrollWheelZoom.enable();
    map.doubleClickZoom.enable();
    map.boxZoom.enable();

    if (isDrawing) {
      container.style.cursor = "crosshair";
    } else {
      container.style.cursor = "";
      setStartPoint(null);
      setCurrentPoint(null);
      isShiftDraggingRef.current = false;
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isDrawing) {
        setStartPoint(null);
        setCurrentPoint(null);
        isShiftDraggingRef.current = false;
        map.dragging.enable();
        onDrawingCancel();
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      map.dragging.enable();
      map.touchZoom.enable();
      map.scrollWheelZoom.enable();
      map.doubleClickZoom.enable();
      map.boxZoom.enable();
      container.style.cursor = "";
    };
  }, [isDrawing, map, onDrawingCancel]);

  const finishDrawing = (
    pt1: L.LatLng,
    pt2: L.LatLng
  ) => {
    const south = Math.min(pt1.lat, pt2.lat);
    const north = Math.max(pt1.lat, pt2.lat);
    const west = Math.min(pt1.lng, pt2.lng);
    const east = Math.max(pt1.lng, pt2.lng);

    setStartPoint(null);
    setCurrentPoint(null);
    isShiftDraggingRef.current = false;
    map.dragging.enable();

    if (
      north - south >= 0.0005 &&
      east - west >= 0.0005
    ) {
      const westNum = Number(west.toFixed(6));
      const eastNum = Number(east.toFixed(6));
      const southNum = Number(south.toFixed(6));
      const northNum = Number(north.toFixed(6));

      const geojson: AoiGeoJson = {
        type: "Polygon",
        coordinates: [
          [
            [westNum, southNum],
            [eastNum, southNum],
            [eastNum, northNum],
            [westNum, northNum],
            [westNum, southNum],
          ],
        ],
      };

      onAoiDrawn(geojson);
    }
  };

  useMapEvents({
    mousedown(e) {
      if (!isDrawing) return;

      // Right-click or middle-click cancels drawing
      if (e.originalEvent.button !== 0) {
        setStartPoint(null);
        setCurrentPoint(null);
        isShiftDraggingRef.current = false;
        map.dragging.enable();
        onDrawingCancel();
        return;
      }

      // Shift + Drag draws rectangle directly
      if (e.originalEvent.shiftKey) {
        map.dragging.disable();
        isShiftDraggingRef.current = true;
        setStartPoint(e.latlng);
        setCurrentPoint(e.latlng);
      }
    },

    mousemove(e) {
      if (!isDrawing || !startPoint) return;
      setCurrentPoint(e.latlng);
    },

    mouseup(e) {
      if (!isDrawing) return;

      if (isShiftDraggingRef.current && startPoint) {
        map.dragging.enable();
        finishDrawing(startPoint, e.latlng);
        return;
      }
    },

    click(e) {
      if (!isDrawing) return;
      if (isShiftDraggingRef.current) return;

      // First click: place start corner
      if (!startPoint) {
        setStartPoint(e.latlng);
        setCurrentPoint(e.latlng);
        return;
      }

      // Second click: finish AOI rectangle
      finishDrawing(startPoint, e.latlng);
    },
  });

  if (
    isDrawing &&
    startPoint &&
    currentPoint
  ) {
    const bounds: LeafletBounds = [
      [
        Math.min(
          startPoint.lat,
          currentPoint.lat
        ),
        Math.min(
          startPoint.lng,
          currentPoint.lng
        ),
      ],
      [
        Math.max(
          startPoint.lat,
          currentPoint.lat
        ),
        Math.max(
          startPoint.lng,
          currentPoint.lng
        ),
      ],
    ];

    return (
      <Rectangle
        bounds={bounds}
        pathOptions={{
          color: "#f4c43b",
          weight: 2,
          dashArray: "4, 4",
          fillColor: "#f4c43b",
          fillOpacity: 0.15,
        }}
      />
    );
  }

  return null;
}

/* ============================================================
   MAP CONTROL BRIDGE
   ============================================================ */

function MapControlBridge() {
  const map = useMap();

  useEffect(() => {
    const zoomIn = () => {
      map.zoomIn();
    };

    const zoomOut = () => {
      map.zoomOut();
    };

    const locate = () => {
      map.fitBounds(map.getBounds(), {
        padding: [30, 30],
      });
    };

    window.addEventListener(
      "satquery-map-zoom-in",
      zoomIn
    );

    window.addEventListener(
      "satquery-map-zoom-out",
      zoomOut
    );

    window.addEventListener(
      "satquery-map-locate",
      locate
    );

    return () => {
      window.removeEventListener(
        "satquery-map-zoom-in",
        zoomIn
      );

      window.removeEventListener(
        "satquery-map-zoom-out",
        zoomOut
      );

      window.removeEventListener(
        "satquery-map-locate",
        locate
      );
    };
  }, [map]);

  return null;
}

/* ============================================================
   MAIN COMPONENT
   ============================================================ */

function AnalysisWorkspace({
  result,
  currentQuery,
  onViewDetails,
  onViewLayers,
  onRequery,
  loading = false,
}: AnalysisWorkspaceProps) {
  const { plan } = result;

  /* ============================================================
     DRAWING AND LAYER STATE
     ============================================================ */

  const [showChangeLayer, setShowChangeLayer] =
    useState(true);

  const [isDrawing, setIsDrawing] =
    useState(false);

  const [drawnAoi, setDrawnAoi] =
    useState<AoiGeoJson | null>(null);

  const [overlayErrorUrl, setOverlayErrorUrl] =
    useState<string | null>(null);

  useEffect(() => {
    setOverlayErrorUrl(null);
  }, [result]);

  const handleAoiDrawn = (
    aoi: AoiGeoJson
  ) => {
    setDrawnAoi(aoi);
    setIsDrawing(false);

    if (onRequery) {
      const activeQuery =
        currentQuery ||
        (plan.metric === "NDBI"
          ? "compare urban change between 2021 and 2025"
          : plan.metric === "NDWI"
            ? "compare water change between 2021 and 2025"
            : "compare vegetation change between 2021 and 2025");

      onRequery(
        activeQuery,
        aoi
      );
    }
  };

  /* ============================================================
     BASIC HELPERS
     ============================================================ */

  const formatDate = (
    date: string | undefined | null
  ) => {
    if (!date) {
      return "—";
    }

    const parsed = new Date(date);

    if (Number.isNaN(parsed.getTime())) {
      return date;
    }

    return parsed
      .toISOString()
      .split("T")[0];
  };

  const confidence =
    result.confidence != null
      ? `${Math.round(
          result.confidence * 100
        )}%`
      : "—";

  /* ============================================================
     BACKEND STATISTICS
     ============================================================ */

  const statistics =
    result.statistics ?? {};

  const meanBefore =
    statistics.mean_before != null
      ? Number(
          statistics.mean_before
        )
      : null;

  const meanAfter =
    statistics.mean_after != null
      ? Number(
          statistics.mean_after
        )
      : null;

  const meanChange =
    statistics.mean_change != null
      ? Number(
          statistics.mean_change
        )
      : null;

  const changedPixels =
    statistics.changed_pixels != null
      ? Number(
          statistics.changed_pixels
        )
      : 0;

  const validPixels =
    statistics.valid_pixels != null
      ? Number(
          statistics.valid_pixels
        )
      : 0;

  const increasedPixels =
    statistics.increased_pixels != null
      ? Number(
          statistics.increased_pixels
        )
      : 0;

  const decreasedPixels =
    statistics.decreased_pixels != null
      ? Number(
          statistics.decreased_pixels
        )
      : 0;

  const changeRatio =
    statistics.change_ratio != null
      ? Number(
          statistics.change_ratio
        )
      : 0;

  const changeType = String(
    statistics.change_type ??
      "unknown"
  );

  const metric = String(
    statistics.metric ??
      plan.metric ??
      plan.analysis?.[0] ??
      "NDVI"
  ).toUpperCase();

  /* ============================================================
     BACKEND CHANGE VISUALIZATION
     ============================================================ */

  const visualizationLayer =
    Array.isArray(result?.layers)
      ? result.layers.find(
          (layer: any) =>
            layer?.visualization_url || layer?.classified_visualization_url
        )
      : null;

  const isIndexMap =
    (visualizationLayer as any)?.type === "index_map" ||
    plan.task.endsWith("_index");

  const rawVisualizationUrl =
    (visualizationLayer as any)?.visualization_url ??
    (visualizationLayer as any)?.classified_visualization_url ??
    (result as any)?.visualization_url ??
    (result as any)?.visualization?.url ??
    (result as any)?.visualization?.relative_path ??
    null;

  const rawBounds =
    (visualizationLayer as any)?.bounds ??
    (result as any)?.bounds ??
    (result as any)?.visualization?.bounds ??
    null;

  const fullVisualizationUrl = useMemo(() => {
    if (!rawVisualizationUrl) return null;
    const cleanUrl = String(rawVisualizationUrl).trim();
    if (cleanUrl.startsWith("http://") || cleanUrl.startsWith("https://")) {
      return cleanUrl;
    }
    const baseUrl = (API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
    const pathUrl = cleanUrl.startsWith("/") ? cleanUrl : `/${cleanUrl}`;
    return `${baseUrl}${pathUrl}`;
  }, [rawVisualizationUrl]);

  const changeMapBounds: LeafletBounds | null = useMemo(() => {
    if (!rawBounds) return null;

    // Format 1: 2x2 nested array [[south, west], [north, east]]
    if (
      Array.isArray(rawBounds) &&
      rawBounds.length === 2 &&
      Array.isArray(rawBounds[0]) &&
      Array.isArray(rawBounds[1]) &&
      rawBounds[0].length >= 2 &&
      rawBounds[1].length >= 2
    ) {
      const v00 = Number(rawBounds[0][0]);
      const v01 = Number(rawBounds[0][1]);
      const v10 = Number(rawBounds[1][0]);
      const v11 = Number(rawBounds[1][1]);
      if ([v00, v01, v10, v11].every(Number.isFinite)) {
        return [[v00, v01], [v10, v11]];
      }
    }

    // Format 2: Flat 4-element array [minLon, minLat, maxLon, maxLat] or [west, south, east, north]
    if (Array.isArray(rawBounds) && rawBounds.length === 4) {
      const v0 = Number(rawBounds[0]);
      const v1 = Number(rawBounds[1]);
      const v2 = Number(rawBounds[2]);
      const v3 = Number(rawBounds[3]);

      if ([v0, v1, v2, v3].every(Number.isFinite)) {
        // GeoJSON standard: [minLon, minLat, maxLon, maxLat]
        if (Math.abs(v0) > Math.abs(v1) || (Math.abs(v0) > 40 && Math.abs(v1) < 40)) {
          const minLon = Math.min(v0, v2);
          const maxLon = Math.max(v0, v2);
          const minLat = Math.min(v1, v3);
          const maxLat = Math.max(v1, v3);
          return [[minLat, minLon], [maxLat, maxLon]];
        } else {
          // [minLat, minLon, maxLat, maxLon]
          const minLat = Math.min(v0, v2);
          const maxLat = Math.max(v0, v2);
          const minLon = Math.min(v1, v3);
          const maxLon = Math.max(v1, v3);
          return [[minLat, minLon], [maxLat, maxLon]];
        }
      }
    }

    return null;
  }, [rawBounds]);

  /* ============================================================
     AOI HELPERS
     ============================================================ */

  const normalizeCoordinates = (
    value: any
  ): LatLng[] => {
    if (!Array.isArray(value)) {
      return [];
    }

    /*
     * Direct Leaflet format:
     *
     * [
     *   [lat, lng],
     *   [lat, lng]
     * ]
     */

    if (
      value.length > 0 &&
      Array.isArray(value[0]) &&
      value[0].length >= 2 &&
      typeof value[0][0] ===
        "number" &&
      typeof value[0][1] ===
        "number"
    ) {
      return value.map(
        (point: number[]) =>
          [
            Number(point[0]),
            Number(point[1]),
          ] as LatLng
      );
    }

    /*
     * GeoJSON Polygon:
     *
     * [
     *   [
     *     [lng, lat],
     *     [lng, lat]
     *   ]
     * ]
     */

    if (
      value.length > 0 &&
      Array.isArray(value[0]) &&
      Array.isArray(value[0][0])
    ) {
      const ring = value[0];

      if (
        Array.isArray(ring) &&
        ring.length > 0 &&
        Array.isArray(ring[0])
      ) {
        return ring
          .filter(
            (point: any) =>
              Array.isArray(point) &&
              point.length >= 2 &&
              typeof point[0] ===
                "number" &&
              typeof point[1] ===
                "number"
          )
          .map(
            (point: number[]) =>
              [
                Number(point[1]),
                Number(point[0]),
              ] as LatLng
          );
      }
    }

    return [];
  };

  /*
   * IMPORTANT:
   *
   * If the backend doesn't send plan.aoi,
   * look for a bbox directly inside the query.
   *
   * Example:
   *
   * [73.80, 18.50, 73.86, 18.56]
   *
   * = [west, south, east, north]
   */

  const extractBboxFromQuery = (
    query?: string
  ): LatLng[] => {
    if (!query) {
      return [];
    }

    const match =
      query.match(
        /\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]/
      );

    if (!match) {
      return [];
    }

    const west = Number(match[1]);
    const south = Number(match[2]);
    const east = Number(match[3]);
    const north = Number(match[4]);

    if (
      ![
        west,
        south,
        east,
        north,
      ].every(Number.isFinite)
    ) {
      return [];
    }

    return [
      [south, west],
      [south, east],
      [north, east],
      [north, west],
      [south, west],
    ];
  };

  /*
   * Try every possible AOI source.
   *
   * Priority:
   *
   * 1. User-drawn AOI
   * 2. plan.aoi
   * 3. result.aoi
   * 4. query bbox
   */

  const possibleAois = [
    drawnAoi,
    (plan as any)?.aoi,
    (result as any)?.aoi,
    (plan as any)?.geometry,
    (plan as any)?.aoi_geometry,
  ];

  let aoiCoordinates: LatLng[] = [];

  for (
    const candidate of possibleAois
  ) {
    if (!candidate) {
      continue;
    }

    const candidateCoordinates =
      normalizeCoordinates(
        candidate?.coordinates ??
          candidate
      );

    if (
      candidateCoordinates.length >= 3
    ) {
      aoiCoordinates =
        candidateCoordinates;
      break;
    }
  }

  /*
   * FINAL FALLBACK:
   *
   * Extract AOI directly from the user's
   * query text.
   */

  if (
    aoiCoordinates.length < 3
  ) {
    aoiCoordinates =
      extractBboxFromQuery(
        currentQuery ||
          (plan as any)?.task ||
          ""
      );
  }

  const hasAoi =
    aoiCoordinates.length >= 3;

  /* ============================================================
     AOI BOUNDS
     ============================================================ */

  const aoiBounds: LeafletBounds | null =
    hasAoi
      ? ([
          [
            Math.min(
              ...aoiCoordinates.map(
                (point) => point[0]
              )
            ),

            Math.min(
              ...aoiCoordinates.map(
                (point) => point[1]
              )
            ),
          ],

          [
            Math.max(
              ...aoiCoordinates.map(
                (point) => point[0]
              )
            ),

            Math.max(
              ...aoiCoordinates.map(
                (point) => point[1]
              )
            ),
          ],
        ] as LeafletBounds)
      : null;

  /* ============================================================
     MAP CENTER
     ============================================================ */

  const fallbackCenter: LatLng = [
    19.076,
    72.8777,
  ];

  const mapCenter: LatLng =
    hasAoi && aoiCoordinates.length
      ? [
          aoiCoordinates.reduce(
            (sum, point) =>
              sum + point[0],
            0
          ) /
            aoiCoordinates.length,

          aoiCoordinates.reduce(
            (sum, point) =>
              sum + point[1],
            0
          ) /
            aoiCoordinates.length,
        ]
      : changeMapBounds
        ? [
            (
              changeMapBounds[0][0] +
              changeMapBounds[1][0]
            ) / 2,

            (
              changeMapBounds[0][1] +
              changeMapBounds[1][1]
            ) / 2,
          ]
        : fallbackCenter;

  /*
   * AOI gets priority over raster bounds.
   *
   * THIS IS THE IMPORTANT CHANGE.
   */

  const viewportBounds =
    aoiBounds ??
    changeMapBounds;

  const overlayBounds: LeafletBounds | null = useMemo(() => {
    if (hasAoi && aoiBounds) {
      if (changeMapBounds) {
        const latDiff = Math.abs(changeMapBounds[0][0] - aoiBounds[0][0]);
        const lonDiff = Math.abs(changeMapBounds[0][1] - aoiBounds[0][1]);
        if (latDiff < 1.0 && lonDiff < 1.0) {
          return changeMapBounds;
        }
      }
      return aoiBounds;
    }
    return changeMapBounds ?? aoiBounds;
  }, [hasAoi, aoiBounds, changeMapBounds]);

  /* ============================================================
     DISPLAY HELPERS
     ============================================================ */

  const formatNumber = (
    value: number | null,
    digits = 4
  ) => {
    if (
      value == null ||
      Number.isNaN(value)
    ) {
      return "—";
    }

    return value.toFixed(digits);
  };

  const formatSignedNumber = (
    value: number | null,
    digits = 4
  ) => {
    if (
      value == null ||
      Number.isNaN(value)
    ) {
      return "—";
    }

    return `${
      value >= 0 ? "+" : ""
    }${value.toFixed(digits)}`;
  };

  const formatPercentage = (
    value: number | null
  ) => {
    if (
      value == null ||
      Number.isNaN(value)
    ) {
      return "—";
    }

    return `${(
      value * 100
    ).toFixed(2)}%`;
  };

  const readableChangeType =
    changeType === "no_change"
      ? "No significant change"
      : changeType.charAt(0).toUpperCase() +
        changeType.slice(1);

  const dateStart = formatDate(
    plan.time_start
  );

  const dateEnd = formatDate(
    plan.time_end
  );

  const evidenceList =
    Array.isArray(result.evidence)
      ? result.evidence
      : [];

  const firstEvidence =
    (evidenceList[0] || {}) as any;

  const evidenceImages =
    Array.isArray(
      firstEvidence?.images
    )
      ? firstEvidence.images
      : [];

  const realDateBefore =
    evidenceImages[0]?.date
      ? formatDate(
          evidenceImages[0].date
        )
      : dateStart;

  const realDateAfter =
    evidenceImages[1]?.date
      ? formatDate(
          evidenceImages[1].date
        )
      : dateEnd;

  const cloudCovers =
    evidenceImages
      .map(
        (img: any) =>
          img?.cloud_cover
      )
      .filter(
        (v: any) =>
          typeof v === "number" &&
          !Number.isNaN(v)
      );

  const cloudCoverText =
    cloudCovers.length > 0
      ? `${(
          cloudCovers.reduce(
            (
              a: number,
              b: number
            ) => a + b,
            0
          ) /
          cloudCovers.length
        ).toFixed(2)}%`
      : "Cloud cover data unavailable";

  /* ============================================================
     RENDER
     ============================================================ */

  return (
    <main className="analysis-workspace">

      {/* ======================================================
          HEADER
          ====================================================== */}

      <header className="analysis-header">

        <div className="analysis-brand">

          <div className="analysis-brand-name">
            SATQUERY AI
          </div>

          <div className="analysis-brand-subtitle">
            REMOTE SENSING INTELLIGENCE
          </div>

        </div>

        <div className="analysis-context">

          <div className="analysis-context-item">

            <span className="analysis-context-label">
              AOI
            </span>

            <span className="analysis-context-value">
              {plan.target ||
                "IDENTIFYING"}
            </span>

            <span className="analysis-context-arrow">
              ↓
            </span>

          </div>

          <div className="analysis-context-item">

            <span className="analysis-context-label">
              DATE RANGE
            </span>

            <span className="analysis-context-value">
              {isIndexMap ? (
                realDateAfter
              ) : (
                <>
                  {realDateBefore}
                  <span className="date-arrow">
                    →
                  </span>
                  {realDateAfter}
                </>
              )}
            </span>

          </div>

        </div>

        <div className="analysis-header-actions">

          <button
            type="button"
            className="analysis-export"
            onClick={() => {
              window.print();
            }}
          >
            EXPORT
          </button>

          <button type="button" className="analysis-menu" aria-label="Open Menu">
            ☰
          </button>

        </div>

      </header>

      {/* ======================================================
          WORKSPACE
          ====================================================== */}

      <section className="analysis-layout">

        {/* ====================================================
            LEFT SIDEBAR
            ==================================================== */}

        <aside className="analysis-sidebar">

          {/* QUERY */}

          <section className="analysis-section analysis-query-section">

            <div className="analysis-section-label">
              QUERY
            </div>

            <p className="analysis-query">
              {currentQuery ||
                plan.task ||
                "Analysis request"}
            </p>

            {drawnAoi && (
              <div className="aoi-draw-status">

                <span className="aoi-status-tag">
                  ✓ CUSTOM AOI DRAWN
                </span>

                <button
                  type="button"
                  className="aoi-clear-btn"
                  onClick={() => {
                    setDrawnAoi(null);

                    if (onRequery) {
                      const activeQuery =
                        currentQuery ||
                        (plan.metric === "NDBI"
                          ? "compare urban change between 2021 and 2025"
                          : plan.metric === "NDWI"
                            ? "compare water change between 2021 and 2025"
                            : "compare vegetation change between 2021 and 2025");

                      onRequery(
                        activeQuery
                      );
                    }
                  }}
                >
                  RESET AOI
                </button>

              </div>
            )}

          </section>

          {/* DATA SUMMARY */}

          <section className="analysis-section analysis-data">

            <div className="analysis-section-label">
              DATA SUMMARY
            </div>

            <div className="analysis-data-row">

              <span className="analysis-data-label">
                SATELLITE
              </span>

              <span className="analysis-data-value">
                {firstEvidence?.source ===
                "REAL_SENTINEL_2"
                  ? "Sentinel-2 (L2A)"
                  : plan.modalities?.length
                    ? plan.modalities.join(
                        ", "
                      )
                    : "Sentinel-2"}
              </span>

            </div>

            <div className="analysis-data-row">

              <span className="analysis-data-label">
                RESOLUTION
              </span>

              <span className="analysis-data-value">
                10m
              </span>

            </div>

            <div className="analysis-data-row">

              <span className="analysis-data-label">
                ACQUISITION DATES
              </span>

              <span className="analysis-data-value">
                {isIndexMap ? (
                  realDateAfter
                ) : (
                  <>
                    {realDateBefore}
                    {" → "}
                    {realDateAfter}
                  </>
                )}
              </span>

            </div>

            <div className="analysis-data-row">

              <span className="analysis-data-label">
                CLOUD COVER
              </span>

              <span className="analysis-data-value">
                {cloudCoverText}
              </span>

            </div>

          </section>

          {/* INDICATORS */}

          <section className="analysis-section analysis-indicators">

            <div className="analysis-section-label">
              PRIMARY INDICATOR
            </div>

            <div className="finding-analysis large">
              {metric}
            </div>

            {isIndexMap ? (
              <>
                <div className="analysis-section-label indicator-secondary-label">
                  ANALYSIS TYPE
                </div>

                <div className="finding-analysis large">
                  SINGLE INDEX
                </div>
              </>
            ) : (
              <>
                <div className="analysis-section-label indicator-secondary-label">
                  SUPPORTING INDICATOR
                </div>

                <div className="finding-analysis large">
                  CHANGE DETECTION
                </div>

                <div className="analysis-section-label indicator-secondary-label">
                  CHANGE TYPE
                </div>

                <div className="finding-analysis large">
                  {readableChangeType}
                </div>
              </>
            )}

          </section>

        </aside>

        {/* ====================================================
            CENTER — SATELLITE MAP
            ==================================================== */}

        <section className="analysis-map-panel">

          <div className="analysis-map">

            <MapContainer
              center={mapCenter}
              zoom={
                viewportBounds
                  ? 11
                  : 10
              }
              zoomControl={false}
              attributionControl={true}
              className="satellite-map"
            >

              {/* SATELLITE BASEMAP */}

              <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                attribution="Tiles © Esri"
              />

              {/* MAP CONTROL BRIDGE */}

              <MapControlBridge />

              {/* =================================================
                  AOI — THIS NOW CONTROLS THE MAP POSITION
                  ================================================= */}

              {hasAoi && (
                <Polygon
                  positions={
                    aoiCoordinates
                  }
                  pathOptions={{
                    color: "#f5f1e9",
                    weight: 2,
                    fillColor:
                      "#f5f1e9",
                    fillOpacity: 0.03,
                  }}
                />
              )}

              {/* AOI DRAW HANDLER */}

              <AoiDrawHandler
                isDrawing={isDrawing}
                onAoiDrawn={
                  handleAoiDrawn
                }
                onDrawingCancel={() =>
                  setIsDrawing(false)
                }
              />

              {/* =================================================
                  IMPORTANT:
                  FIT MAP TO AOI FIRST.
                  FALL BACK TO RASTER BOUNDS.
                  ================================================= */}

              <MapViewportController
                bounds={
                  viewportBounds
                }
                center={mapCenter}
              />

              {/* =================================================
                  REAL GEOREFERENCED CHANGE RASTER
                  ================================================= */}

              {showChangeLayer &&
                fullVisualizationUrl &&
                overlayBounds &&
                overlayErrorUrl !== fullVisualizationUrl && (
                  <ImageOverlay
                    key={fullVisualizationUrl}
                    url={fullVisualizationUrl}
                    bounds={overlayBounds}
                    opacity={0.8}
                    zIndex={1000}
                    eventHandlers={{
                      load: () => {
                        console.log("CHANGE RASTER LOADED SUCCESS:", fullVisualizationUrl);
                        console.log("BOUNDS:", overlayBounds);
                      },
                      error: (e) => {
                        console.warn("CHANGE RASTER FAILED TO LOAD:", fullVisualizationUrl, e);
                        setOverlayErrorUrl(fullVisualizationUrl);
                      },
                    }}
                  />
                )}

            </MapContainer>

            {/* DRAWING STATUS */}

            {isDrawing && (
              <div className="map-drawing-badge">
                <span>
                  ▱ CLICK 2 POINTS (OR SHIFT+DRAG) TO SELECT AOI • DRAG MAP TO PAN (ESC TO CANCEL)
                </span>
              </div>
            )}

            {/* LOADING */}

            {loading && (
              <div className="analysis-loading-overlay">
                <span>
                  RETRIEVING REAL
                  SENTINEL-2 IMAGERY
                  &amp; ANALYZING...
                </span>
              </div>
            )}

            {/* MAP BADGE */}

            <div className="map-analysis-badge">

              <span>
                {metric} {isIndexMap ? "INDEX" : "CHANGE"}
              </span>

              <span>
                {isIndexMap ? (
                  realDateAfter
                ) : (
                  <>
                    {realDateBefore} → {realDateAfter}
                  </>
                )}
              </span>

            </div>

            {/* MAP CONTROLS */}

            <div className="map-controls">

              <button
                type="button"
                aria-label="Zoom in"
                onClick={() => {
                  window.dispatchEvent(
                    new CustomEvent(
                      "satquery-map-zoom-in"
                    )
                  );
                }}
              >
                +
              </button>

              <button
                type="button"
                aria-label="Zoom out"
                onClick={() => {
                  window.dispatchEvent(
                    new CustomEvent(
                      "satquery-map-zoom-out"
                    )
                  );
                }}
              >
                −
              </button>

              <button
                type="button"
                aria-label="Locate analysis"
                onClick={() => {
                  window.dispatchEvent(
                    new CustomEvent(
                      "satquery-map-locate"
                    )
                  );
                }}
              >
                ⌖
              </button>

              <button
                type="button"
                aria-label="Toggle change layer"
                aria-pressed={
                  showChangeLayer
                }
                title={
                  showChangeLayer
                    ? "Hide change layer"
                    : "Show change layer"
                }
                onClick={() => {
                  setShowChangeLayer(
                    (visible) =>
                      !visible
                  );
                }}
              >
                ⌁
              </button>

              <button
                type="button"
                aria-label={
                  isDrawing
                    ? "Cancel AOI drawing"
                    : "Draw AOI rectangle"
                }
                title={
                  isDrawing
                    ? "Click to cancel drawing"
                    : "Click to draw AOI rectangle on map"
                }
                className={
                  isDrawing
                    ? "active-draw-control"
                    : ""
                }
                aria-pressed={
                  isDrawing
                }
                onClick={() => {
                  setIsDrawing(
                    (drawing) =>
                      !drawing
                  );
                }}
              >
                ▱
              </button>

            </div>

            {/* LEGEND */}

            <div className="map-legend">

              <div className="map-legend-title">
                {metric} {isIndexMap ? "INDEX" : "CHANGE"}
              </div>

              <div
                className="legend-gradient-bar"
                title="Continuous Gradient"
                style={isIndexMap ? {
                  background: metric === "NDWI" ? "linear-gradient(to right, #ffffff, #0055ff)" : metric === "NDBI" ? "linear-gradient(to right, #ffffff, #ff0000)" : "linear-gradient(to right, #ffffff, #00aa00)"
                } : {}}
              />

              {isIndexMap ? (
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#a0a0a0', marginTop: '4px' }}>
                  <span>Low (-1.0)</span>
                  <span>High (1.0)</span>
                </div>
              ) : (
                <>
                  <div className="legend-item">
                    <span className="legend-swatch high" />
                    <span>
                      High decrease
                    </span>
                  </div>

                  <div className="legend-item">
                    <span className="legend-swatch moderate" />
                    <span>
                      Moderate decrease
                    </span>
                  </div>

                  <div className="legend-item">
                    <span className="legend-swatch slight" />
                    <span>
                      Slight decrease
                    </span>
                  </div>

                  <div className="legend-item">
                    <span className="legend-swatch unchanged" />
                    <span>
                      No change
                    </span>
                  </div>

                  <div className="legend-item">
                    <span className="legend-swatch slight-increase" />
                    <span>
                      Slight increase
                    </span>
                  </div>

                  <div className="legend-item">
                    <span className="legend-swatch increase" />
                    <span>
                      Moderate increase
                    </span>
                  </div>

                  <div className="legend-item">
                    <span className="legend-swatch high-increase" />
                    <span>
                      High increase
                    </span>
                  </div>
                </>
              )}

            </div>

            {/* SCALE */}

            <div className="map-scale">

              <span className="map-scale-line" />

              <span>
                2 km
              </span>

            </div>

            {/* COORDINATES */}

            <div className="map-coordinates">

              {mapCenter[0].toFixed(4)}
              ° N,{" "}
              {mapCenter[1].toFixed(4)}
              ° E

            </div>

          </div>

        </section>

        {/* ====================================================
            RIGHT — FINDINGS
            ==================================================== */}

        <aside className="findings-panel">

          {/* FINDINGS */}

          <section className="findings-section">

            <div className="findings-label">
              {isIndexMap ? "INDEX STATISTICS" : "CHANGE FINDINGS"}
            </div>

            {isIndexMap ? (
              <>
                <div className="finding-stat">
                  <div className="finding-stat-value">
                    {Intl.NumberFormat().format((visualizationLayer as any)?.valid_pixels ?? validPixels)}
                  </div>
                  <div className="finding-stat-label">
                    VALID PIXELS
                  </div>
                </div>

                <div className="finding-stat">
                  <div className="finding-stat-value" style={{ whiteSpace: "nowrap" }}>
                    {formatNumber((visualizationLayer as any)?.min_value, 2)} → {formatNumber((visualizationLayer as any)?.max_value, 2)}
                  </div>
                  <div className="finding-stat-label">
                    VALUE RANGE
                  </div>
                </div>

                <div className="finding-stat">
                  <div className="finding-stat-value">
                    {confidence}
                  </div>
                  <div className="finding-stat-label">
                    AI CONFIDENCE
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="finding-stat">
                  <div className="finding-stat-value">
                    {changedPixels} /{" "}
                    {validPixels}
                  </div>
                  <div className="finding-stat-label">
                    CHANGED PIXELS
                  </div>
                </div>

                <div className="finding-stat">
                  <div className="finding-stat-value">
                    {formatSignedNumber(
                      meanChange
                    )}
                  </div>
                  <div className="finding-stat-label">
                    MEAN {metric} CHANGE
                  </div>
                </div>

                <div className="finding-stat">
                  <div className="finding-stat-value">
                    {confidence}
                  </div>
                  <div className="finding-stat-label">
                    AI CONFIDENCE
                  </div>
                </div>

                {/* ADDITIONAL STATISTICS */}

                <div className="finding-stat-row">
                  <div className="finding-stat small">
                    <div className="finding-stat-value">
                      {formatSignedNumber(
                        meanBefore
                      )}
                    </div>
                    <div className="finding-stat-label">
                      BEFORE
                    </div>
                  </div>

                  <div className="finding-stat small">
                    <div className="finding-stat-value">
                      {formatSignedNumber(
                        meanAfter
                      )}
                    </div>
                    <div className="finding-stat-label">
                      AFTER
                    </div>
                  </div>
                </div>

                <div className="finding-stat-row">
                  <div className="finding-stat small increase-stat">
                    <div className="finding-stat-value">
                      {increasedPixels}
                    </div>
                    <div className="finding-stat-label">
                      INCREASED
                    </div>
                  </div>

                  <div className="finding-stat small decrease-stat">
                    <div className="finding-stat-value">
                      {decreasedPixels}
                    </div>
                    <div className="finding-stat-label">
                      DECREASED
                    </div>
                  </div>
                </div>

                <div className="finding-stat-row">
                  <div className="finding-stat small">
                    <div className="finding-stat-value">
                      {formatPercentage(
                        changeRatio
                      )}
                    </div>
                    <div className="finding-stat-label">
                      CHANGE RATIO
                    </div>
                  </div>
                </div>
              </>
            )}

          </section>

          {/* CHANGE SUMMARY */}

          {!isIndexMap && (
            <section className="findings-section">

              <div className="findings-label">
                CHANGE SUMMARY
              </div>

              <div className="analysis-data-row">

                <span className="analysis-data-label">
                  BEFORE
                </span>

                <span className="analysis-data-value">
                  {formatNumber(
                    meanBefore
                  )}
                </span>

              </div>

              <div className="analysis-data-row">

                <span className="analysis-data-label">
                  AFTER
                </span>

                <span className="analysis-data-value">
                  {formatNumber(
                    meanAfter
                  )}
                </span>

              </div>

              <div className="analysis-data-row">

                <span className="analysis-data-label">
                  THRESHOLD
                </span>

                <span className="analysis-data-value">
                  {statistics.threshold !=
                  null
                    ? Number(
                        statistics.threshold
                      ).toFixed(2)
                    : "—"}
                </span>

              </div>

            </section>
          )}

          {/* VISUALIZATION STATUS */}

          {!fullVisualizationUrl && (
            <section className="findings-section">

              <div className="findings-label">
                VISUALIZATION
              </div>

              <div className="analysis-data-value">
                Backend change-map
                visualization
                unavailable.
              </div>

            </section>
          )}

          {fullVisualizationUrl &&
            !changeMapBounds && (
              <section className="findings-section">

                <div className="findings-label">
                  VISUALIZATION
                </div>

                <div className="analysis-data-value">
                  Change raster available,
                  but backend raster
                  bounds are missing.
                </div>

              </section>
            )}

          {/* VIEW DETAILS */}

          <div className="analysis-navigation-button">

            <button type="button" className="view-details-button layers-navigation-button" onClick={onViewLayers}>
              <span>
                LAYERS
              </span>

              <span>
                →
              </span>
            </button>

            <button type="button" className="view-details-button" onClick={onViewDetails}>

              <span>
                VIEW DETAILS
              </span>

              <span>
                →
              </span>
              
            </button>

          </div>

        </aside>

      </section>

    </main>
  );
}

export default AnalysisWorkspace;