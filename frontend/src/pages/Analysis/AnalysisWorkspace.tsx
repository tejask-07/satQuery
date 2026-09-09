import { useEffect, useMemo, useRef, useState } from "react";
import type L from "leaflet";

import type { QueryResponse } from "../../api/query";

import {
  ImageOverlay,
  MapContainer,
  Polygon,
  Rectangle,
  TileLayer,
  useMap,
  useMapEvents,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./AnalysisWorkspace.css";

const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

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
   MAP VIEWPORT
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
        padding: [50, 50],
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
   MAP CONTROLS
   ============================================================ */

function MapControlBridge() {
  const map = useMap();

  useEffect(() => {
    const zoomIn = () => map.zoomIn();
    const zoomOut = () => map.zoomOut();

    const locate = () => {
      map.fitBounds(map.getBounds(), {
        padding: [40, 40],
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

  const shiftDragRef = useRef(false);

  useEffect(() => {
    const container = map.getContainer();

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
      shiftDragRef.current = false;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && isDrawing) {
        setStartPoint(null);
        setCurrentPoint(null);
        shiftDragRef.current = false;
        map.dragging.enable();
        onDrawingCancel();
      }
    };

    window.addEventListener(
      "keydown",
      handleKeyDown
    );

    return () => {
      window.removeEventListener(
        "keydown",
        handleKeyDown
      );

      map.dragging.enable();
      map.touchZoom.enable();
      map.scrollWheelZoom.enable();
      map.doubleClickZoom.enable();
      map.boxZoom.enable();

      container.style.cursor = "";
    };
  }, [isDrawing, map, onDrawingCancel]);

  const finishDrawing = (
    first: L.LatLng,
    second: L.LatLng
  ) => {
    const south = Math.min(
      first.lat,
      second.lat
    );

    const north = Math.max(
      first.lat,
      second.lat
    );

    const west = Math.min(
      first.lng,
      second.lng
    );

    const east = Math.max(
      first.lng,
      second.lng
    );

    setStartPoint(null);
    setCurrentPoint(null);
    shiftDragRef.current = false;

    map.dragging.enable();

    if (
      north - south < 0.0005 ||
      east - west < 0.0005
    ) {
      return;
    }

    const polygon: AoiGeoJson = {
      type: "Polygon",
      coordinates: [
        [
          [
            Number(west.toFixed(6)),
            Number(south.toFixed(6)),
          ],
          [
            Number(east.toFixed(6)),
            Number(south.toFixed(6)),
          ],
          [
            Number(east.toFixed(6)),
            Number(north.toFixed(6)),
          ],
          [
            Number(west.toFixed(6)),
            Number(north.toFixed(6)),
          ],
          [
            Number(west.toFixed(6)),
            Number(south.toFixed(6)),
          ],
        ],
      ],
    };

    onAoiDrawn(polygon);
  };

  useMapEvents({
    mousedown(event) {
      if (!isDrawing) return;

      if (event.originalEvent.button !== 0) {
        setStartPoint(null);
        setCurrentPoint(null);
        shiftDragRef.current = false;
        map.dragging.enable();
        onDrawingCancel();
        return;
      }

      if (event.originalEvent.shiftKey) {
        map.dragging.disable();

        shiftDragRef.current = true;

        setStartPoint(event.latlng);
        setCurrentPoint(event.latlng);
      }
    },

    mousemove(event) {
      if (!isDrawing || !startPoint) return;

      setCurrentPoint(event.latlng);
    },

    mouseup(event) {
      if (!isDrawing) return;

      if (
        shiftDragRef.current &&
        startPoint
      ) {
        finishDrawing(
          startPoint,
          event.latlng
        );
      }
    },

    click(event) {
      if (!isDrawing) return;

      if (shiftDragRef.current) return;

      if (!startPoint) {
        setStartPoint(event.latlng);
        setCurrentPoint(event.latlng);
        return;
      }

      finishDrawing(
        startPoint,
        event.latlng
      );
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
          dashArray: "6 5",
          fillColor: "#f4c43b",
          fillOpacity: 0.12,
        }}
      />
    );
  }

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
  const plan = result.plan as any;
  const statistics = (result.statistics ??
    {}) as any;

  const taskName = String(
    plan?.task ?? ""
  ).toLowerCase();

  const planAny = plan as any;

  const intentName = String(
    planAny?.intent ?? ""
  ).toLowerCase();

  const isVqa =
    taskName === "vqa" ||
    taskName === "visual_qa" ||
    taskName === "visual_question_answering" ||
    taskName.includes("vqa") ||
    taskName.includes("visual_question") ||
    intentName === "vqa" ||
    intentName.includes("visual_question");

  const isOpticalSar =
    taskName.includes("optical_sar") ||
    taskName.includes("optical-sar") ||
    taskName.includes("sar_optical") ||
    intentName.includes("optical_sar") ||
    intentName.includes("optical-sar") ||
    (taskName.includes("sar") && taskName.includes("optical")) ||
    (intentName.includes("sar") && intentName.includes("optical"));

  const isCaption =
    taskName === "caption" ||
    taskName === "image_captioning" ||
    taskName.includes("caption") ||
    intentName === "caption" ||
    intentName.includes("caption");

  const isImageSearch =
    taskName === "image_search" ||
    taskName === "search_imagery" ||
    intentName === "image_search" ||
    (Array.isArray(plan?.analysis) &&
      plan.analysis.length === 1 &&
      plan.analysis[0] === "search_imagery" &&
      taskName !== "change_detection");

  const isIndexMap =
    taskName === "vegetation_index" ||
    taskName === "water_index" ||
    taskName === "urban_index" ||
    Boolean(
      taskName &&
      taskName.includes("_index") &&
      !taskName.includes("change")
    );

  const isChangeTask =
    !isImageSearch &&
    !isVqa &&
    !isCaption &&
    !isOpticalSar &&
    !isIndexMap &&
    (taskName === "change_detection" ||
      taskName.includes("change") ||
      intentName.includes("change") ||
      (Array.isArray(plan?.analysis) && plan.analysis.includes("detect_change")) ||
      (Array.isArray((result as any).layers) &&
        (result as any).layers.some((l: any) => String(l?.id || "").startsWith("change_"))));

  // The workspace is one route; the backend result selects the visual mode.
  const isTemporal = isChangeTask;

  const [showRaster, setShowRaster] =
    useState(true);

  const [isDrawing, setIsDrawing] =
    useState(false);

  const [drawnAoi, setDrawnAoi] =
    useState<AoiGeoJson | null>(null);

  const [overlayError, setOverlayError] =
    useState(false);

  const [imageryType, setImageryType] =
    useState<"true_color" | "false_color">("true_color");

  const [imageryRef, setImageryRef] =
    useState<"after" | "before">("after");

  useEffect(() => {
    setOverlayError(false);
    setDrawnAoi(null);
    setShowRaster(true);
    setImageryType("true_color");
    setImageryRef("after");
  }, [result]);

  /* ============================================================
     BASIC HELPERS
     ============================================================ */

  const formatDate = (
    value: unknown
  ) => {
    if (!value) return "—";

    const parsed = new Date(String(value));

    if (Number.isNaN(parsed.getTime())) {
      return String(value);
    }

    return parsed
      .toISOString()
      .split("T")[0];
  };

  const formatNumber = (
    value: unknown,
    digits = 4
  ) => {
    if (value == null || value === "") {
      return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
      return "—";
    }

    return number.toFixed(digits);
  };

  const formatSigned = (
    value: unknown,
    digits = 4
  ) => {
    if (value == null || value === "") {
      return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
      return "—";
    }

    return `${number >= 0 ? "+" : ""}${number.toFixed(
      digits
    )}`;
  };

  /* ============================================================
     METRIC / STATISTICS
     ============================================================ */

  const rawMetric =
    statistics.metric ??
    plan.metric ??
    null;

  const metric = rawMetric
    ? String(rawMetric).toUpperCase()
    : isTemporal
      ? "NDVI"
      : "";

  const meanBefore =
    statistics.mean_before != null
      ? Number(statistics.mean_before)
      : null;

  const meanAfter =
    statistics.mean_after != null
      ? Number(statistics.mean_after)
      : null;

  const meanChange =
    statistics.mean_change != null
      ? Number(statistics.mean_change)
      : meanBefore != null &&
          meanAfter != null
        ? meanAfter - meanBefore
        : null;

  const changedPixels =
    Number(statistics.changed_pixels) || 0;

  const validPixels =
    Number(statistics.valid_pixels) || 0;

  const increasedPixels =
    Number(statistics.increased_pixels) || 0;

  const decreasedPixels =
    Number(statistics.decreased_pixels) || 0;

  const changeRatio =
    statistics.change_ratio != null
      ? Number(statistics.change_ratio)
      : validPixels > 0
        ? changedPixels / validPixels
        : 0;

  /* ============================================================
     EVIDENCE / DATES
     ============================================================ */

  const evidence = Array.isArray(
    (result as any).evidence
  )
    ? (result as any).evidence
    : [];

  const firstEvidence =
    evidence[0] ?? {};

  const evidenceImages =
    Array.isArray(firstEvidence.images)
      ? firstEvidence.images
      : [];

  const beforeDate = formatDate(
    evidenceImages[0]?.date ??
      plan.time_start
  );

  const afterDate = formatDate(
    evidenceImages[1]?.date ??
      plan.time_end
  );

  const cloudValues =
    evidenceImages
      .map((image: any) =>
        Number(image?.cloud_cover)
      )
      .filter((value: number) =>
        Number.isFinite(value)
      );

  const cloudCover =
    cloudValues.length > 0
      ? `${(
          cloudValues.reduce(
            (sum: number, value: number) =>
              sum + value,
            0
          ) / cloudValues.length
        ).toFixed(1)}%`
      : "Unavailable";

  /* ============================================================
     VISUALIZATION
     ============================================================ */

  const allLayers: any[] = useMemo(() => {
    return Array.isArray((result as any).layers) ? (result as any).layers : [];
  }, [result]);

  const visualizationLayer = useMemo(() => {
    if (isImageSearch) {
      const preferredId = `${imageryType}_${imageryRef}`;
      const altId = `${imageryType}_${imageryRef === "after" ? "before" : "after"}`;

      const layer =
        allLayers.find(
          (l: any) =>
            l?.id === preferredId &&
            (l?.visualization_url || l?.classified_visualization_url)
        ) ||
        allLayers.find(
          (l: any) =>
            l?.id === altId &&
            (l?.visualization_url || l?.classified_visualization_url)
        ) ||
        allLayers.find(
          (l: any) =>
            l?.id?.startsWith(imageryType) &&
            (l?.visualization_url || l?.classified_visualization_url)
        ) ||
        allLayers.find(
          (l: any) =>
            l?.id?.startsWith("true_color") &&
            (l?.visualization_url || l?.classified_visualization_url)
        ) ||
        allLayers.find(
          (l: any) =>
            l?.id?.startsWith("false_color") &&
            (l?.visualization_url || l?.classified_visualization_url)
        ) ||
        allLayers.find(
          (l: any) =>
            l?.visualization_url || l?.classified_visualization_url
        );

      return layer ?? null;
    }

    if (isIndexMap) {
      const targetMetric = (rawMetric || "").toLowerCase();
      const layer =
        allLayers.find(
          (l: any) =>
            (l?.id?.includes(targetMetric) || l?.id?.startsWith("index_")) &&
            (l?.visualization_url || l?.classified_visualization_url)
        ) ||
        allLayers.find(
          (l: any) =>
            l?.id?.startsWith("index_") &&
            (l?.visualization_url || l?.classified_visualization_url)
        ) ||
        allLayers.find(
          (l: any) =>
            l?.visualization_url || l?.classified_visualization_url
        );
      return layer ?? null;
    }

    if (isChangeTask) {
      const layer =
        allLayers.find(
          (l: any) =>
            l?.id === "change_continuous" &&
            (l?.visualization_url || l?.classified_visualization_url)
        ) ||
        allLayers.find(
          (l: any) =>
            l?.id?.startsWith("change_") &&
            (l?.visualization_url || l?.classified_visualization_url)
        ) ||
        allLayers.find(
          (l: any) =>
            l?.visualization_url || l?.classified_visualization_url
        );
      return layer ?? null;
    }

    return (
      allLayers.find(
        (l: any) =>
          l?.visualization_url || l?.classified_visualization_url
      ) ?? null
    );
  }, [
    isImageSearch,
    isIndexMap,
    isChangeTask,
    imageryType,
    imageryRef,
    allLayers,
    rawMetric,
  ]);

  const rawVisualizationUrl =
    visualizationLayer?.visualization_url ??
    visualizationLayer?.classified_visualization_url ??
    (!isImageSearch
      ? ((result as any).visualization_url ??
        (result as any).visualization?.url ??
        (result as any).visualization?.relative_path)
      : null) ??
    null;

  const rawBounds =
    visualizationLayer?.bounds ??
    (result as any).bounds ??
    (result as any).visualization?.bounds ??
    null;

  const visualizationUrl =
    useMemo(() => {
      if (!rawVisualizationUrl) {
        return null;
      }

      const value =
        String(rawVisualizationUrl).trim();

      if (
        value.startsWith("http://") ||
        value.startsWith("https://")
      ) {
        return value;
      }

      const base =
        API_BASE_URL.replace(/\/+$/, "");

      return `${base}${
        value.startsWith("/")
          ? value
          : `/${value}`
      }`;
    }, [rawVisualizationUrl]);

  /* ============================================================
     BOUNDS
     ============================================================ */

  const visualizationBounds =
    useMemo<LeafletBounds | null>(() => {
      if (!rawBounds) {
        return null;
      }

      if (
        Array.isArray(rawBounds) &&
        rawBounds.length === 2 &&
        Array.isArray(rawBounds[0]) &&
        Array.isArray(rawBounds[1])
      ) {
        const a = Number(
          rawBounds[0][0]
        );
        const b = Number(
          rawBounds[0][1]
        );
        const c = Number(
          rawBounds[1][0]
        );
        const d = Number(
          rawBounds[1][1]
        );

        if (
          [a, b, c, d].every(
            Number.isFinite
          )
        ) {
          return [
            [a, b],
            [c, d],
          ];
        }
      }

      if (
        Array.isArray(rawBounds) &&
        rawBounds.length === 4
      ) {
        const v0 = Number(
          rawBounds[0]
        );
        const v1 = Number(
          rawBounds[1]
        );
        const v2 = Number(
          rawBounds[2]
        );
        const v3 = Number(
          rawBounds[3]
        );

        if (
          [v0, v1, v2, v3].every(
            Number.isFinite
          )
        ) {
          return [
            [
              Math.min(v1, v3),
              Math.min(v0, v2),
            ],
            [
              Math.max(v1, v3),
              Math.max(v0, v2),
            ],
          ];
        }
      }

      return null;
    }, [rawBounds]);

  /* ============================================================
     AOI NORMALIZATION
     ============================================================ */

  const normalizeCoordinates = (
    value: any
  ): LatLng[] => {
    if (!Array.isArray(value)) {
      return [];
    }

    /* Leaflet: [[lat,lng], ...] */

    if (
      value.length > 0 &&
      Array.isArray(value[0]) &&
      value[0].length >= 2 &&
      typeof value[0][0] === "number" &&
      typeof value[0][1] === "number"
    ) {
      return value.map(
        (point: number[]) =>
          [
            Number(point[0]),
            Number(point[1]),
          ] as LatLng
      );
    }

    /* GeoJSON: [[[lng,lat], ...]] */

    if (
      value.length > 0 &&
      Array.isArray(value[0]) &&
      Array.isArray(value[0][0])
    ) {
      const ring = value[0];

      return ring
        .filter(
          (point: any) =>
            Array.isArray(point) &&
            point.length >= 2
        )
        .map(
          (point: number[]) =>
            [
              Number(point[1]),
              Number(point[0]),
            ] as LatLng
        );
    }

    return [];
  };

  const extractBbox = (
    query?: string
  ): LatLng[] => {
    if (!query) return [];

    const match = query.match(
      /\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]/
    );

    if (!match) return [];

    const west = Number(match[1]);
    const south = Number(match[2]);
    const east = Number(match[3]);
    const north = Number(match[4]);

    return [
      [south, west],
      [south, east],
      [north, east],
      [north, west],
      [south, west],
    ];
  };

  const aoiCoordinates =
    useMemo<LatLng[]>(() => {
      const candidates = [
        drawnAoi,
        plan?.aoi,
        (result as any)?.aoi,
        planAny?.geometry,
        planAny?.aoi_geometry,
      ];

      for (const candidate of candidates) {
        if (!candidate) continue;

        const coordinates =
          normalizeCoordinates(
            candidate?.coordinates ??
              candidate
          );

        if (coordinates.length >= 3) {
          return coordinates;
        }
      }

      return extractBbox(
        currentQuery ||
          plan?.task ||
          ""
      );
    }, [
      drawnAoi,
      plan,
      result,
      currentQuery,
    ]);

  const hasAoi =
    aoiCoordinates.length >= 3;

  const aoiBounds =
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

  const fallbackCenter: LatLng = [
    19.076,
    72.8777,
  ];

  const mapCenter: LatLng =
    aoiBounds
      ? [
          (aoiBounds[0][0] +
            aoiBounds[1][0]) /
            2,
          (aoiBounds[0][1] +
            aoiBounds[1][1]) /
            2,
        ]
      : visualizationBounds
        ? [
            (visualizationBounds[0][0] +
              visualizationBounds[1][0]) /
              2,
            (visualizationBounds[0][1] +
              visualizationBounds[1][1]) /
              2,
          ]
        : fallbackCenter;

  const viewportBounds =
    aoiBounds ??
    visualizationBounds;

  /* ============================================================
     QUERY / PLAN
     ============================================================ */

  /* ============================================================
     AOI REQUERY
     ============================================================ */

  const handleAoiDrawn = (
    aoi: AoiGeoJson
  ) => {
    setDrawnAoi(aoi);
    setIsDrawing(false);

    if (!onRequery) {
      return;
    }

    const query =
      currentQuery ||
      (metric === "NDBI"
        ? "compare urban change between 2021 and 2025"
        : metric === "NDWI"
          ? "compare water change between 2021 and 2025"
          : "compare vegetation change between 2021 and 2025");

    onRequery(
      query,
      aoi
    );
  };

  /* ============================================================
     FINDINGS DISPLAY DATA
     ============================================================ */

  const realDateBefore = beforeDate;
  const realDateAfter = afterDate;
  const cloudCoverText = cloudCover;

  const formatSignedNumber = (
    value: unknown,
    digits = 4
  ) => formatSigned(value, digits);

  const changeCoverage =
  validPixels > 0
    ? Math.min(
        100,
        Math.max(
          0,
          (changedPixels / validPixels) * 100
        )
      )
    : changeRatio > 0
      ? Math.min(
          100,
          Math.max(
            0,
            changeRatio * 100
          )
        )
      : 0;

const decreaseShare =
  validPixels > 0
    ? Math.min(
        100,
        Math.max(
          0,
          (decreasedPixels / validPixels) * 100
        )
      )
    : 0;

const increaseShare =
  validPixels > 0
    ? Math.min(
        100,
        Math.max(
          0,
          (increasedPixels / validPixels) * 100
        )
      )
    : 0;

const unchangedShare =
  Math.max(
    0,
    100 -
      decreaseShare -
      increaseShare
  );

const beforeAfterValue =
  meanBefore != null &&
  meanAfter != null
    ? `${formatSignedNumber(meanBefore)} → ${formatSignedNumber(meanAfter)}`
    : "—";

const primaryIndicator =
  plan?.primary_indicators?.[0] ??
  (isIndexMap ? metric : `${metric} Difference`);

const supportingIndicator =
  plan?.supporting_indicators?.[0] ??
  (isIndexMap ? "Single index" : "Change detection");

const rawAnswer =
  typeof result.answer === "string" && result.answer.trim()
    ? result.answer.trim()
    : null;

const interpretationText =
  typeof (statistics as any)?.interpretation?.summary === "string"
    ? (statistics as any).interpretation.summary
    : typeof (statistics as any)?.explanation === "string"
      ? (statistics as any).explanation
      : rawAnswer || "Not available for this analysis.";

const vqaAnswer = rawAnswer ?? "Not available for this analysis.";
const captionAnswer = rawAnswer ?? "Not available for this analysis.";
const opticalSarAnswer = rawAnswer ?? "Not available for this analysis.";

const confidencePercent =
  result.confidence !== null && result.confidence !== undefined && Number.isFinite(Number(result.confidence))
    ? Math.round(Number(result.confidence) * 100)
    : null;

const executionTrace = Array.isArray(result.execution_trace)
  ? result.execution_trace
  : [];

const modelInfo =
  result.model?.name ||
  (result as any).interpretation?.model ||
  (isVqa || isCaption || isOpticalSar ? "Remote Sensing VLM" : null);

const analysisStatus = result.status || "COMPLETE";

const vqaSource =
  firstEvidence?.source === "REAL_SENTINEL_2"
    ? "Sentinel-2 (L2A)"
    : plan?.modalities?.length
      ? plan.modalities.join(", ")
      : "Satellite imagery";

const opticalSarModalities =
  Array.isArray(plan?.modalities) && plan.modalities.length
    ? plan.modalities.join(" + ")
    : "Optical + SAR";

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

        <div className="analysis-header-center">
          <div className="header-context">
            <span className="header-context-label">
              INVESTIGATION
            </span>

            <span className="header-context-value">
              {plan?.target ||
                "REMOTE SENSING ANALYSIS"}
            </span>
          </div>

          <div className="header-divider" />

          <div className="header-context">
            <span className="header-context-label">
              {isVqa || isOpticalSar || isCaption || isImageSearch || isIndexMap ? "ANALYSIS TYPE" : "TEMPORAL WINDOW"}
            </span>

            <span className="header-context-value">
              {isVqa
                ? "VISUAL QUESTION ANSWERING"
                : isCaption
                  ? "IMAGE CAPTIONING"
                  : isOpticalSar
                    ? "OPTICAL + SAR ANALYSIS"
                    : isImageSearch
                      ? "SATELLITE IMAGERY SEARCH"
                      : isIndexMap
                        ? `${metric || "SPECTRAL"} INDEX ANALYSIS`
                        : (
                          <>
                            {beforeDate}
                            <span className="header-arrow">
                              →
                            </span>
                            {afterDate}
                          </>
                        )}
            </span>
          </div>
        </div>

        <div className="analysis-header-actions">
          <div className="analysis-complete">
            <span className="complete-dot" />
            {analysisStatus.toUpperCase()}
          </div>

          <button
            type="button"
            className="export-button"
            onClick={() =>
              window.print()
            }
          >
            EXPORT
          </button>
        </div>
      </header>

      {/* ======================================================
          MAIN LAYOUT
          ====================================================== */}

      <section className="analysis-layout">
        {/* ====================================================
            LEFT RAIL
            ==================================================== */}
        
        <aside className="analysis-sidebar">

  {/* ======================================================
      QUERY
      ====================================================== */}

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
                (
                  plan.metric === "NDBI"
                    ? "compare urban change between 2021 and 2025"
                    : plan.metric === "NDWI"
                      ? "compare water change between 2021 and 2025"
                      : "compare vegetation change between 2021 and 2025"
                );

              onRequery(activeQuery);
            }
          }}
        >
          RESET AOI
        </button>

      </div>
    )}

  </section>


  {/* ======================================================
      DATA SUMMARY
      ====================================================== */}

  <section className="analysis-section analysis-data">

    <div className="analysis-section-label">
      DATA SUMMARY
    </div>


    <div className="analysis-data-row">

      <span className="analysis-data-label">
        SATELLITE
      </span>

      <span className="analysis-data-value">
        {firstEvidence?.source === "REAL_SENTINEL_2"
          ? "Sentinel-2 (L2A)"
          : plan.modalities?.length
            ? plan.modalities.join(", ")
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
        {isVqa || isOpticalSar || isCaption || isImageSearch ? "ACQUISITION" : "ACQUISITION DATES"}
      </span>

      <span className="analysis-data-value">

        {isVqa || isOpticalSar || isCaption || isImageSearch ? (
          realDateAfter !== "—"
            ? realDateAfter
            : realDateBefore
        ) : isIndexMap ? (
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


  {/* ======================================================
      MODEL & EXECUTION TRACE
      ====================================================== */}

  {(modelInfo || executionTrace.length > 0) && (
    <section className="analysis-section">

      <div className="analysis-section-label">
        EXECUTION TRACE
      </div>

      {modelInfo && (
        <div className="analysis-data-row" style={{ marginBottom: 8 }}>
          <span className="analysis-data-label">
            MODEL
          </span>
          <span className="analysis-data-value">
            {modelInfo}
          </span>
        </div>
      )}

      {executionTrace.length > 0 ? (
        <ul className="execution-trace-list">
          {executionTrace.map((step: string, index: number) => (
            <li key={`trace-${index}`} className="execution-trace-item">
              <span className="execution-step-num">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span>
                {step}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <div className="analysis-data-value">
          Not available for this analysis.
        </div>
      )}

    </section>
  )}

</aside>
        

        {/* ====================================================
            MAP
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
              attributionControl
              className="satellite-map"
            >
              <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                attribution="Tiles © Esri"
              />

              <MapControlBridge />

              <MapViewportController
                bounds={viewportBounds}
                center={mapCenter}
              />

              {hasAoi && (
                <Polygon
                  positions={
                    aoiCoordinates
                  }
                  pathOptions={{
                    color: "#f7f1e6",
                    weight: 2,
                    fillColor:
                      "#f7f1e6",
                    fillOpacity: 0.035,
                  }}
                />
              )}

              <AoiDrawHandler
                isDrawing={isDrawing}
                onAoiDrawn={
                  handleAoiDrawn
                }
                onDrawingCancel={() =>
                  setIsDrawing(false)
                }
              />

              {showRaster &&
                visualizationUrl &&
                (visualizationBounds ?? aoiBounds) &&
                !overlayError && (
                  <ImageOverlay
                    key={
                      visualizationUrl
                    }
                    url={
                      visualizationUrl
                    }
                    bounds={
                      (visualizationBounds ??
                        aoiBounds)!
                    }
                    opacity={
                      isImageSearch
                        ? 1.0
                        : 0.80
                    }
                    zIndex={1000}
                    eventHandlers={{
                      load: () => {
                        setOverlayError(
                          false
                        );
                      },
                      error: () => {
                        console.warn(
                          "SatQuery raster failed:",
                          visualizationUrl
                        );

                        setOverlayError(
                          true
                        );
                      },
                    }}
                  />
                )}
            </MapContainer>

            {/* MAP TOP LABEL */}

            <div className="map-title">
              <span className="map-title-mode">
                {isVqa || isCaption
                  ? "MULTIMODAL"
                  : isOpticalSar
                    ? "OPTICAL + SAR"
                    : isImageSearch
                      ? imageryType === "false_color"
                        ? "FALSE COLOR (NIR)"
                        : "TRUE COLOR (RGB)"
                      : `${metric || "INDEX"} ${String(
                          plan?.task ?? ""
                        ).includes("index")
                          ? "INDEX"
                          : "CHANGE"}`}
              </span>

              <span className="map-title-date">
                {isVqa || isCaption
                  ? vqaSource
                  : isOpticalSar
                    ? opticalSarModalities
                    : isImageSearch
                      ? imageryRef === "before"
                        ? beforeDate
                        : afterDate !== "—"
                          ? afterDate
                          : beforeDate
                      : <>
                          {beforeDate}
                          <b>→</b>
                          {afterDate}
                        </>}
              </span>
            </div>

            {/* MAP CONTROLS */}

            <div className="map-controls">
              {isImageSearch && (
                <div style={{ display: "flex", gap: "4px", marginRight: "8px" }}>
                  <button
                    type="button"
                    style={{
                      padding: "4px 8px",
                      fontSize: "11px",
                      fontWeight: 600,
                      background: imageryType === "true_color" ? "rgba(56, 189, 248, 0.25)" : "transparent",
                      color: imageryType === "true_color" ? "#38bdf8" : "#94a3b8",
                      border: "1px solid var(--soft-line)",
                      borderRadius: "4px",
                      cursor: "pointer",
                    }}
                    onClick={() => setImageryType("true_color")}
                  >
                    RGB
                  </button>
                  <button
                    type="button"
                    style={{
                      padding: "4px 8px",
                      fontSize: "11px",
                      fontWeight: 600,
                      background: imageryType === "false_color" ? "rgba(56, 189, 248, 0.25)" : "transparent",
                      color: imageryType === "false_color" ? "#38bdf8" : "#94a3b8",
                      border: "1px solid var(--soft-line)",
                      borderRadius: "4px",
                      cursor: "pointer",
                    }}
                    onClick={() => setImageryType("false_color")}
                  >
                    NIR
                  </button>
                </div>
              )}
              <button
                type="button"
                aria-label="Zoom in"
                onClick={() =>
                  window.dispatchEvent(
                    new CustomEvent(
                      "satquery-map-zoom-in"
                    )
                  )
                }
              >
                +
              </button>

              <button
                type="button"
                aria-label="Zoom out"
                onClick={() =>
                  window.dispatchEvent(
                    new CustomEvent(
                      "satquery-map-zoom-out"
                    )
                  )
                }
              >
                −
              </button>

              <button
                type="button"
                aria-label="Locate analysis"
                onClick={() =>
                  window.dispatchEvent(
                    new CustomEvent(
                      "satquery-map-locate"
                    )
                  )
                }
              >
                ⌖
              </button>

              <button
                type="button"
                className={
                  showRaster
                    ? "active"
                    : ""
                }
                aria-label="Toggle raster"
                onClick={() =>
                  setShowRaster(
                    (value) =>
                      !value
                  )
                }
              >
                ◫
              </button>

              <button
                type="button"
                className={
                  isDrawing
                    ? "active"
                    : ""
                }
                aria-label="Draw AOI"
                onClick={() =>
                  setIsDrawing(
                    (value) =>
                      !value
                  )
                }
              >
                ▱
              </button>
            </div>

            {/* DRAWING MESSAGE */}

            {isDrawing && (
              <div className="drawing-notice">
                CLICK TWO POINTS OR
                SHIFT + DRAG TO DEFINE
                AN AOI
                <span>
                  ESC TO CANCEL
                </span>
              </div>
            )}

            {/* LEGEND */}

            {isTemporal ? (
              <div className="map-legend">
                <div className="legend-heading">
                  {metric || "CHANGE"} CHANGE
                </div>

                <div className="legend-gradient" />

                <div className="legend-scale">
                  <span>
                    DECREASE
                  </span>

                  <span>
                    NO CHANGE
                  </span>

                  <span>
                    INCREASE
                  </span>
                </div>
              </div>
            ) : isImageSearch ? (
              <div className="map-legend map-source-legend">
                <div className="legend-heading">
                  SATELLITE IMAGERY
                </div>

                <div className="vqa-source-legend-line">
                  <span className="vqa-source-dot" style={{ background: "#38bdf8" }} />
                  <span>
                    {imageryType === "false_color" ? "False Color (NIR/Red/Green) • 10m" : "True Color (RGB) • 10m"}
                  </span>
                </div>
              </div>
            ) : isVqa || isCaption ? (
              <div className="map-legend map-source-legend">
                <div className="legend-heading">
                  {isCaption ? "CAPTION SOURCE" : "VQA SOURCE"}
                </div>

                <div className="vqa-source-legend-line">
                  <span className="vqa-source-dot" />
                  <span>{vqaSource}</span>
                </div>
              </div>
            ) : (
              <div className="map-legend map-source-legend optical-sar-legend">
                <div className="legend-heading">
                  MULTIMODAL SOURCE
                </div>

                <div className="vqa-source-legend-line">
                  <span className="vqa-source-dot optical-sar-dot" />
                  <span>{opticalSarModalities}</span>
                </div>
              </div>
            )}

            {/* MAP FOOTER */}

            <div className="map-footer">
              <span>
                {mapCenter[0].toFixed(
                  4
                )}
                ° N&nbsp;&nbsp;
                {mapCenter[1].toFixed(
                  4
                )}
                ° E
              </span>

              <span>
                ESRI WORLD IMAGERY
              </span>
            </div>

            {/* LOADING */}

            {loading && (
              <div className="analysis-loading">
                <div className="loading-spinner" />

                <div>
                  <strong>
                    {isVqa
                      ? "ANALYZING IMAGE"
                      : isCaption
                        ? "GENERATING CAPTION"
                        : isOpticalSar
                          ? "ANALYZING MULTIMODAL DATA"
                          : "ANALYZING AOI"}
                  </strong>

                  <span>
                    {isVqa
                      ? "Grounding the visual question in satellite imagery"
                      : isCaption
                        ? "Extracting spatial features and describing scene"
                        : isOpticalSar
                          ? "Aligning optical and SAR evidence for multimodal analysis"
                          : "Retrieving satellite imagery and computing change"}
                  </span>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* ====================================================
            FINDINGS
            ==================================================== */}

            <aside className={`findings-panel${
              isVqa
                ? " findings-panel-vqa"
                : isOpticalSar
                  ? " findings-panel-optical-sar"
                  : ""
            }`}>

              {isVqa && (
                <>
                  <section className="findings-section vqa-answer-section">
                    <div className="findings-label">
                      VQA ANSWER
                    </div>

                    <div className="vqa-answer">
                      {vqaAnswer}
                    </div>
                  </section>

                  <section className="findings-section vqa-confidence-section">
                    <div className="findings-label">
                      MODEL CONFIDENCE
                    </div>

                    {confidencePercent != null ? (
                      <div className="vqa-confidence-row">
                        <div className="vqa-confidence-value">
                          {confidencePercent}%
                        </div>

                        <div className="vqa-confidence-track">
                          <div
                            className="vqa-confidence-fill"
                            style={{
                              width: `${Math.max(0, Math.min(100, confidencePercent))}%`,
                            }}
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="empty-stat-value">
                        Not available for this analysis.
                      </div>
                    )}
                  </section>

                  <section className="findings-section vqa-context-section">
                    <div className="findings-label">
                      QUESTION TYPE
                    </div>

                    <div className="finding-analysis large">
                      Visual Question Answering
                    </div>

                    <div className="findings-label secondary-label">
                      SOURCE
                    </div>

                    <div className="finding-analysis large">
                      {vqaSource}
                    </div>
                  </section>

                  <section className="findings-section interpretation-section vqa-interpretation-section">
                    <div className="findings-label">
                      EVIDENCE CONTEXT
                    </div>

                    <p className="interpretation-text">
                      {interpretationText}
                    </p>
                  </section>
                </>
              )}

              {isCaption && (
                <>
                  <section className="findings-section vqa-answer-section">
                    <div className="findings-label">
                      IMAGE CAPTION
                    </div>

                    <div className="vqa-answer">
                      {captionAnswer}
                    </div>
                  </section>

                  <section className="findings-section vqa-confidence-section">
                    <div className="findings-label">
                      MODEL CONFIDENCE
                    </div>

                    {confidencePercent != null ? (
                      <div className="vqa-confidence-row">
                        <div className="vqa-confidence-value">
                          {confidencePercent}%
                        </div>

                        <div className="vqa-confidence-track">
                          <div
                            className="vqa-confidence-fill"
                            style={{
                              width: `${Math.max(0, Math.min(100, confidencePercent))}%`,
                            }}
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="empty-stat-value">
                        Not available for this analysis.
                      </div>
                    )}
                  </section>

                  <section className="findings-section vqa-context-section">
                    <div className="findings-label">
                      TASK TYPE
                    </div>

                    <div className="finding-analysis large">
                      Remote Sensing Image Captioning
                    </div>

                    <div className="findings-label secondary-label">
                      SOURCE
                    </div>

                    <div className="finding-analysis large">
                      {vqaSource}
                    </div>
                  </section>

                  <section className="findings-section interpretation-section vqa-interpretation-section">
                    <div className="findings-label">
                      SCENE CONTEXT
                    </div>

                    <p className="interpretation-text">
                      {interpretationText}
                    </p>
                  </section>
                </>
              )}

              {isImageSearch && (
                <>
                  <section className="findings-section vqa-answer-section">
                    <div className="findings-label">
                      IMAGERY OVERVIEW
                    </div>

                    <div className="vqa-answer-box" style={{ borderColor: "rgba(56, 189, 248, 0.4)", background: "rgba(56, 189, 248, 0.05)" }}>
                      <div className="vqa-answer-text">
                        {rawAnswer || "Multi-spectral optical surface reflectance imagery retrieved and georeferenced for target location."}
                      </div>
                    </div>
                  </section>

                  <section className="findings-section">
                    <div className="findings-label">
                      SCENE PROPERTIES
                    </div>

                    <div className="finding-stat">
                      <div className="finding-stat-value">
                        {imageryRef === "before" ? beforeDate : afterDate !== "—" ? afterDate : beforeDate}
                      </div>
                      <div className="finding-stat-label">
                        ACQUISITION DATE
                      </div>
                      <div className="finding-stat-description">
                        Satellite capture timestamp
                      </div>
                    </div>

                    <div className="finding-stat">
                      <div className="finding-stat-value">
                        10m
                      </div>
                      <div className="finding-stat-label">
                        SPATIAL RESOLUTION
                      </div>
                      <div className="finding-stat-description">
                        Ground sample distance (10m per pixel)
                      </div>
                    </div>

                    <div className="finding-stat">
                      <div className="finding-stat-value">
                        {cloudCoverText}
                      </div>
                      <div className="finding-stat-label">
                        CLOUD COVERAGE
                      </div>
                      <div className="finding-stat-description">
                        Scene cloud and shadow occlusion
                      </div>
                    </div>
                  </section>

                  <section className="findings-section indicator-section">
                    <div className="findings-label">
                      SENSOR PLATFORM
                    </div>

                    <div className="finding-analysis large">
                      European Space Agency Sentinel-2 MSI
                    </div>

                    <div className="findings-label secondary-label">
                      PROCESSING LEVEL
                    </div>

                    <div className="finding-analysis large">
                      Level-2A Bottom-Of-Atmosphere (BOA) Reflectance
                    </div>
                  </section>

                  <section className="findings-section interpretation-section">
                    <div className="findings-label">
                      SCENE CONTEXT
                    </div>

                    <p className="interpretation-text">
                      {interpretationText}
                    </p>
                  </section>
                </>
              )}

              {isTemporal && (
                <>

  {/* ======================================================
      PRIMARY FINDINGS
      ====================================================== */}

  <section className="findings-section">

    <div className="findings-label">
      {isIndexMap
        ? "INDEX STATISTICS"
        : "FINDINGS"}
    </div>


    {!isIndexMap && (
      <div className="finding-stat">

        <div className="finding-stat-value">
          {changeCoverage.toFixed(1)}%
        </div>

        <div className="finding-stat-label">
          CHANGE COVERAGE
        </div>

        <div className="finding-stat-description">
          of valid pixels show measurable
          change across the selected area.
        </div>

      </div>
    )}


    <div className="finding-stat">

      <div className="finding-stat-value">
        {formatSignedNumber(meanChange)}
      </div>

      <div className="finding-stat-label">
        MEAN {metric} CHANGE
      </div>

    </div>


    {!isIndexMap && (
      <div className="finding-stat">

        <div className="finding-stat-value finding-stat-value-compact">
          {beforeAfterValue}
        </div>

        <div className="finding-stat-label">
          {metric} BEFORE → AFTER
        </div>

      </div>
    )}


    {isIndexMap && (
      <>
        <div className="finding-stat">

          <div className="finding-stat-value">
            {formatNumber(
              (visualizationLayer as any)?.min_value,
              2
            )}
            {" → "}
            {formatNumber(
              (visualizationLayer as any)?.max_value,
              2
            )}
          </div>

          <div className="finding-stat-label">
            VALUE RANGE
          </div>

        </div>

        <div className="finding-stat">

          <div className="finding-stat-value">
            {formatNumber(
              (statistics as any)?.mean,
              4
            )}
          </div>

          <div className="finding-stat-label">
            MEAN {metric}
          </div>

        </div>
      </>
    )}

  </section>


  {/* ======================================================
      CHANGE DISTRIBUTION
      ====================================================== */}

  {!isIndexMap && (
    <section className="findings-section distribution-section">

      <div className="findings-label">
        CHANGE DISTRIBUTION
      </div>


      {validPixels > 0 ? (
        <>
          <div className="distribution-stack">

            <div
              className="distribution-segment distribution-decrease"
              style={{
                width: `${decreaseShare}%`,
              }}
              title={`Decrease: ${decreaseShare.toFixed(1)}%`}
            />

            <div
              className="distribution-segment distribution-stable"
              style={{
                width: `${unchangedShare}%`,
              }}
              title={`Stable: ${unchangedShare.toFixed(1)}%`}
            />

            <div
              className="distribution-segment distribution-increase"
              style={{
                width: `${increaseShare}%`,
              }}
              title={`Increase: ${increaseShare.toFixed(1)}%`}
            />

          </div>


          <div className="distribution-values">

            <div className="distribution-value decrease-value">
              <strong>
                {decreaseShare.toFixed(1)}%
              </strong>

              <span>
                DECREASE
              </span>
            </div>


            <div className="distribution-value stable-value">
              <strong>
                {unchangedShare.toFixed(1)}%
              </strong>

              <span>
                STABLE
              </span>
            </div>


            <div className="distribution-value increase-value">
              <strong>
                {increaseShare.toFixed(1)}%
              </strong>

              <span>
                INCREASE
              </span>
            </div>

          </div>


          <div className="distribution-footnote">
            SHARE OF VALID PIXELS
          </div>
        </>
      ) : (
        <div className="empty-stat-value">
          Not available for this analysis.
        </div>
      )}

    </section>
  )}


  {/* ======================================================
      INDICATORS
      ====================================================== */}

  <section className="findings-section indicator-section">

    <div className="findings-label">
      PRIMARY INDICATOR
    </div>

    <div className="finding-analysis large">
      {primaryIndicator}
    </div>


    <div className="findings-label secondary-label">
      SUPPORTING INDICATOR
    </div>

    <div className="finding-analysis large">
      {supportingIndicator}
    </div>

  </section>


  {/* ======================================================
      INTERPRETATION
      ====================================================== */}

  {!isIndexMap && (
    <section className="findings-section interpretation-section">

      <div className="findings-label">
        INTERPRETATION
      </div>

      <p className="interpretation-text">
        {interpretationText}
      </p>

    </section>
  )}


                </>
              )}

  {isOpticalSar && (
    <>
      <section className="findings-section vqa-answer-section optical-sar-answer-section">
        <div className="findings-label">
          MULTIMODAL FINDING
        </div>

        <div className="vqa-answer">
          {opticalSarAnswer}
        </div>
      </section>

      <section className="findings-section vqa-confidence-section">
        <div className="findings-label">
          MODEL CONFIDENCE
        </div>

        {confidencePercent != null ? (
          <div className="vqa-confidence-row">
            <div className="vqa-confidence-value">
              {confidencePercent}%
            </div>

            <div className="vqa-confidence-track">
              <div
                className="vqa-confidence-fill"
                style={{
                  width: `${Math.max(0, Math.min(100, confidencePercent))}%`,
                }}
              />
            </div>
          </div>
        ) : (
          <div className="empty-stat-value">
            Not available for this analysis.
          </div>
        )}
      </section>

      <section className="findings-section optical-sar-modalities-section">
        <div className="findings-label">
          MODALITIES
        </div>

        <div className="finding-analysis large">
          {opticalSarModalities}
        </div>
      </section>

      <section className="findings-section">
        <div className="findings-label">
          ANALYSIS TYPE
        </div>

        <div className="finding-analysis large">
          Optical + SAR analysis
        </div>

        <div className="findings-label secondary-label">
          EVIDENCE
        </div>

        <div className="finding-analysis large">
          Multimodal satellite evidence
        </div>
      </section>

      <section className="findings-section interpretation-section vqa-interpretation-section">
        <div className="findings-label">
          INTERPRETATION
        </div>

        <p className="interpretation-text">
          The result combines complementary optical and radar observations to interpret the selected scene.
        </p>
      </section>
    </>
  )}

  {/* ======================================================
      NAVIGATION
      ====================================================== */}

  <div className="analysis-navigation-button">

    {onViewLayers && (
      <button
        type="button"
        className="view-details-button layers-navigation-button"
        onClick={onViewLayers}
      >
        <span>
          LAYERS
        </span>

        <span>
          →
        </span>
      </button>
    )}


    <button
      type="button"
      className="view-details-button"
      onClick={onViewDetails}
    >
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