import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";

import {
  MapContainer,
  TileLayer,
  Polygon,
  Circle,
  CircleMarker,
  ScaleControl,
  useMap,
} from "react-leaflet";

import { LatLngBounds } from "leaflet";

import "leaflet/dist/leaflet.css";
import "./AOISelection.css";

/* =========================================================
   TYPES
   ========================================================= */

export type AOIGeometry =
  | {
      type: "polygon";
      coordinates: [number, number][];
    }
  | {
      type: "rectangle";
      coordinates: [number, number][];
    }
  | {
      type: "circle";
      center: [number, number];
      radius: number;
    };

export interface AOIMetadata {
  name: string;
  geometry: AOIGeometry;
  center: [number, number];
  area: string;
  perimeter: string;
}

interface AOISelectionProps {
  initialQuery?: string;

  onAOIChange?: (
    aoi: AOIGeometry | null
  ) => void;

  onRunAnalysis?: (
    query: string,
    aoi: AOIMetadata,
    startDate: string,
    endDate: string
  ) => void;
}

/* =========================================================
   DRAWING TYPES
   ========================================================= */

type DrawingMode =
  | "polygon"
  | "rectangle"
  | "circle";

interface DrawingState {
  points: [number, number][];
  previewPoint: [number, number] | null;
  circleRadius: number;
}

interface DrawingInteractionProps {
  mode: DrawingMode;

  drawingState: DrawingState;

  setDrawingState: Dispatch<
    SetStateAction<DrawingState>
  >;

  onComplete: (
    aoi: AOIGeometry
  ) => void;
}

/* =========================================================
   DRAWING ENGINE
   ========================================================= */

function DrawingInteraction({
  mode,
  drawingState,
  setDrawingState,
  onComplete,
}: DrawingInteractionProps) {
  const map = useMap();

  const modeRef = useRef(mode);
  const stateRef = useRef(drawingState);
  const completeRef = useRef(onComplete);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    stateRef.current = drawingState;
  }, [drawingState]);

  useEffect(() => {
    completeRef.current = onComplete;
  }, [onComplete]);

  useEffect(() => {
    const mapContainer = map.getContainer();

    /*
     * Create a real DOM drawing surface above the Leaflet layers.
     *
     * This deliberately does NOT depend on Leaflet's click
     * event propagation. The overlay receives the pointer events
     * directly, converts them to map coordinates, and updates
     * React state.
     */
    const overlay = document.createElement("div");

    overlay.setAttribute(
      "data-satquery-aoi-drawing",
      "true"
    );

    Object.assign(
      overlay.style,
      {
        position: "absolute",
        inset: "0",
        zIndex: "450",
        background: "transparent",
        cursor: "crosshair",
        pointerEvents: "auto",
        touchAction: "none",
      }
    );

    mapContainer.appendChild(overlay);

    const getPoint = (
      event: MouseEvent
    ): [number, number] => {
      const rect =
        mapContainer.getBoundingClientRect();

      const x =
        event.clientX - rect.left;

      const y =
        event.clientY - rect.top;

      const latLng =
        map.containerPointToLatLng([
          x,
          y,
        ]);

      return [
        latLng.lat,
        latLng.lng,
      ];
    };

    const updateState = (
      next:
        | DrawingState
        | ((
            current: DrawingState
          ) => DrawingState)
    ) => {
      const current =
        stateRef.current;

      const resolved =
        typeof next === "function"
          ? next(current)
          : next;

      stateRef.current =
        resolved;

      setDrawingState(
        resolved
      );
    };

    let lastPolygonClickTime = 0;

    const handlePointerDown = (
      event: PointerEvent
    ) => {
      /*
       * Only primary mouse / touch pointer.
       */
      if (
        event.button !== 0 &&
        event.pointerType !== "touch"
      ) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      const currentMode =
        modeRef.current;

      const currentState =
        stateRef.current;

      const point =
        getPoint(event);

      /*
       * =========================================================
       * RECTANGLE
       * =========================================================
       */

      if (
        currentMode ===
        "rectangle"
      ) {
        if (
          currentState.points.length === 0
        ) {
          updateState({
            points: [point],
            previewPoint: point,
            circleRadius: 0,
          });

          return;
        }

        const start =
          currentState.points[0];

        const coordinates:
          [number, number][] = [
            start,
            [
              start[0],
              point[1],
            ],
            point,
            [
              point[0],
              start[1],
            ],
          ];

        completeRef.current({
          type: "rectangle",
          coordinates,
        });

        return;
      }

      /*
       * =========================================================
       * CIRCLE
       * =========================================================
       */

      if (
        currentMode ===
        "circle"
      ) {
        if (
          currentState.points.length === 0
        ) {
          updateState({
            points: [point],
            previewPoint: point,
            circleRadius: 0,
          });

          return;
        }

        const center =
          currentState.points[0];

        const radius =
          haversineDistance(
            center,
            point
          );

        completeRef.current({
          type: "circle",
          center,
          radius,
        });

        return;
      }

      /*
       * =========================================================
       * POLYGON
       * =========================================================
       */

      if (
        currentMode ===
        "polygon"
      ) {
        /*
         * A browser double-click produces two pointerdown
         * events. Ignore the second one so the finishing
         * double-click does not add a duplicate point.
         */
        const now =
          Date.now();

        if (
          now -
            lastPolygonClickTime <
          250
        ) {
          return;
        }

        lastPolygonClickTime =
          now;

        updateState(
          (current) => ({
            ...current,
            points: [
              ...current.points,
              point,
            ],
            previewPoint: point,
          })
        );
      }
    };

    const handleDoubleClick = (
      event: MouseEvent
    ) => {
      if (
        modeRef.current !==
        "polygon"
      ) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      const currentState =
        stateRef.current;

      if (
        currentState.points.length <
        3
      ) {
        return;
      }

      completeRef.current({
        type: "polygon",
        coordinates: [
          ...currentState.points,
        ],
      });
    };

    const handleContextMenu = (
      event: MouseEvent
    ) => {
      event.preventDefault();
      event.stopPropagation();

      const currentMode =
        modeRef.current;

      const currentState =
        stateRef.current;

      const point =
        getPoint(event);

      /*
       * Right-click finishes a polygon.
       */
      if (
        currentMode ===
          "polygon" &&
        currentState.points.length >=
          3
      ) {
        completeRef.current({
          type: "polygon",
          coordinates: [
            ...currentState.points,
          ],
        });

        return;
      }

      /*
       * Right-click can also complete a
       * rectangle after its first point.
       */
      if (
        currentMode ===
          "rectangle" &&
        currentState.points.length ===
          1
      ) {
        const start =
          currentState.points[0];

        const coordinates:
          [number, number][] = [
            start,
            [
              start[0],
              point[1],
            ],
            point,
            [
              point[0],
              start[1],
            ],
          ];

        completeRef.current({
          type: "rectangle",
          coordinates,
        });

        return;
      }

      /*
       * Right-click can also complete a
       * circle after its center.
       */
      if (
        currentMode ===
          "circle" &&
        currentState.points.length ===
          1
      ) {
        const center =
          currentState.points[0];

        const radius =
          haversineDistance(
            center,
            point
          );

        completeRef.current({
          type: "circle",
          center,
          radius,
        });
      }
    };

    const handleMouseMove = (
      event: MouseEvent
    ) => {
      const currentMode =
        modeRef.current;

      const currentState =
        stateRef.current;

      if (
        currentState.points.length ===
        0
      ) {
        return;
      }

      if (
        currentMode !==
          "rectangle" &&
        currentMode !==
          "circle"
      ) {
        return;
      }

      const point =
        getPoint(event);

      if (
        currentMode ===
        "rectangle"
      ) {
        updateState(
          (current) => ({
            ...current,
            previewPoint: point,
          })
        );

        return;
      }

      const center =
        currentState.points[0];

      const radius =
        haversineDistance(
          center,
          point
        );

      updateState(
        (current) => ({
          ...current,
          previewPoint: point,
          circleRadius: radius,
        })
      );
    };

    overlay.addEventListener(
      "pointerdown",
      handlePointerDown
    );

    overlay.addEventListener(
      "dblclick",
      handleDoubleClick
    );

    overlay.addEventListener(
      "contextmenu",
      handleContextMenu
    );

    overlay.addEventListener(
      "mousemove",
      handleMouseMove
    );

    /*
     * While drawing, normal map dragging would fight
     * with the drawing interaction.
     */
    map.dragging.disable();

    return () => {
      overlay.removeEventListener(
        "pointerdown",
        handlePointerDown
      );

      overlay.removeEventListener(
        "dblclick",
        handleDoubleClick
      );

      overlay.removeEventListener(
        "contextmenu",
        handleContextMenu
      );

      overlay.removeEventListener(
        "mousemove",
        handleMouseMove
      );

      if (
        overlay.parentNode ===
        mapContainer
      ) {
        mapContainer.removeChild(
          overlay
        );
      }

      map.dragging.enable();
    };
  }, [map, setDrawingState]);

  return null;
}

/* =========================================================
   MAP VIEW CONTROLLER
   ========================================================= */

function AOIMapController({
  selectedAOI,
}: {
  selectedAOI: AOIGeometry | null;
}) {
  const map = useMap();

  useEffect(() => {
    if (!selectedAOI) {
      return;
    }

    /*
     * CIRCLE
     */
    if (
      selectedAOI.type ===
      "circle"
    ) {
      const latDelta =
        selectedAOI.radius /
        111320;

      const cosLatitude =
        Math.cos(
          (selectedAOI.center[0] *
            Math.PI) /
            180
        );

      const lngDelta =
        selectedAOI.radius /
        (111320 *
          cosLatitude);

      const bounds =
        new LatLngBounds(
          [
            selectedAOI.center[0] -
              latDelta,

            selectedAOI.center[1] -
              lngDelta,
          ],

          [
            selectedAOI.center[0] +
              latDelta,

            selectedAOI.center[1] +
              lngDelta,
          ]
        );

      map.fitBounds(
        bounds,
        {
          padding: [
            60,
            60,
          ],
          maxZoom: 13,
        }
      );

      return;
    }

    /*
     * RECTANGLE / POLYGON
     */
    const bounds =
      new LatLngBounds(
        selectedAOI.coordinates
      );

    map.fitBounds(
      bounds,
      {
        padding: [
          60,
          60,
        ],
        maxZoom: 13,
      }
    );
  }, [
    map,
    selectedAOI,
  ]);

  return null;
}

/* =========================================================
   GEOMETRY HELPERS
   ========================================================= */

function haversineDistance(
  first: [number, number],
  second: [number, number]
): number {
  const earthRadius =
    6371000;

  const lat1 =
    (first[0] *
      Math.PI) /
    180;

  const lat2 =
    (second[0] *
      Math.PI) /
    180;

  const deltaLat =
    ((second[0] -
      first[0]) *
      Math.PI) /
    180;

  const deltaLng =
    ((second[1] -
      first[1]) *
      Math.PI) /
    180;

  const a =
    Math.sin(
      deltaLat / 2
    ) ** 2 +
    Math.cos(lat1) *
      Math.cos(lat2) *
      Math.sin(
        deltaLng / 2
      ) ** 2;

  return (
    2 *
    earthRadius *
    Math.atan2(
      Math.sqrt(a),
      Math.sqrt(1 - a)
    )
  );
}

/* =========================================================
   POLYGON AREA
   ========================================================= */

function polygonArea(
  coordinates: [number, number][]
): number {
  if (
    coordinates.length < 3
  ) {
    return 0;
  }

  const earthRadius =
    6371000;

  const meanLat =
    coordinates.reduce(
      (sum, point) =>
        sum + point[0],
      0
    ) /
    coordinates.length;

  const latScale =
    (Math.PI / 180) *
    earthRadius;

  const lngScale =
    (Math.PI / 180) *
    earthRadius *
    Math.cos(
      (meanLat *
        Math.PI) /
        180
    );

  let area = 0;

  for (
    let index = 0;
    index <
    coordinates.length;
    index++
  ) {
    const current =
      coordinates[index];

    const next =
      coordinates[
        (index + 1) %
          coordinates.length
      ];

    const x1 =
      current[1] *
      lngScale;

    const y1 =
      current[0] *
      latScale;

    const x2 =
      next[1] *
      lngScale;

    const y2 =
      next[0] *
      latScale;

    area +=
      x1 * y2 -
      x2 * y1;
  }

  return (
    Math.abs(area) / 2
  );
}

/* =========================================================
   POLYGON PERIMETER
   ========================================================= */

function polygonPerimeter(
  coordinates: [number, number][]
): number {
  if (
    coordinates.length < 2
  ) {
    return 0;
  }

  let perimeter = 0;

  for (
    let index = 0;
    index <
    coordinates.length;
    index++
  ) {
    const current =
      coordinates[index];

    const next =
      coordinates[
        (index + 1) %
          coordinates.length
      ];

    perimeter +=
      haversineDistance(
        current,
        next
      );
  }

  return perimeter;
}

/* =========================================================
   CENTROID
   ========================================================= */

function polygonCentroid(
  coordinates: [number, number][]
): [number, number] {
  if (
    coordinates.length === 0
  ) {
    return [0, 0];
  }

  const latitude =
    coordinates.reduce(
      (sum, point) =>
        sum + point[0],
      0
    ) /
    coordinates.length;

  const longitude =
    coordinates.reduce(
      (sum, point) =>
        sum + point[1],
      0
    ) /
    coordinates.length;

  return [
    latitude,
    longitude,
  ];
}

/* =========================================================
   FORMATTERS
   ========================================================= */

function formatNumber(
  value: number,
  decimals = 2
) {
  return value.toLocaleString(
    "en-IN",
    {
      minimumFractionDigits:
        decimals,

      maximumFractionDigits:
        decimals,
    }
  );
}

function formatDistance(
  meters: number
) {
  if (
    meters >= 1000
  ) {
    return `${formatNumber(
      meters / 1000
    )} km`;
  }

  return `${formatNumber(
    meters
  )} m`;
}

function formatCoordinate(
  value: number,
  positive: string,
  negative: string
) {
  return `${Math.abs(
    value
  ).toFixed(4)}° ${
    value >= 0
      ? positive
      : negative
  }`;
}

/* =========================================================
   MAIN COMPONENT
   ========================================================= */

function AOISelection({
  initialQuery = "",
  onAOIChange,
  onRunAnalysis,
}: AOISelectionProps) {
  /* =======================================================
     QUERY
     ======================================================= */

  const [
    query,
    setQuery,
  ] = useState(
    initialQuery
  );

  /* =======================================================
     DATES
     ======================================================= */

  const [
    startDate,
    setStartDate,
  ] = useState(
    "2021-04-17"
  );

  const [
    endDate,
    setEndDate,
  ] = useState(
    "2025-04-17"
  );

  /* =======================================================
     DRAWING MODE
     ======================================================= */

  const [
    drawingMode,
    setDrawingMode,
  ] = useState<DrawingMode>(
    "polygon"
  );

  const [
    drawingState,
    setDrawingState,
  ] = useState<DrawingState>({
    points: [],
    previewPoint: null,
    circleRadius: 0,
  });

  /* =======================================================
     SELECTED AOI
     ======================================================= */

  const [
    selectedAOI,
    setSelectedAOI,
  ] = useState<AOIGeometry | null>(
    null
  );

  /* =======================================================
     COMPLETE AOI
     ======================================================= */

  const handleAOIComplete = (
    aoi: AOIGeometry
  ) => {
    /*
     * THIS is the point where the preview becomes
     * an actual selected AOI.
     */
    setSelectedAOI(aoi);

    /*
     * Remove drawing state.
     */
    setDrawingState({
      points: [],
      previewPoint: null,
      circleRadius: 0,
    });

    /*
     * Tell parent component.
     */
    onAOIChange?.(aoi);
  };

  /* =======================================================
     CLEAR AOI
     ======================================================= */

  const clearAOI = () => {
    setSelectedAOI(null);

    setDrawingState({
      points: [],
      previewPoint: null,
      circleRadius: 0,
    });

    onAOIChange?.(null);
  };

  /* =======================================================
     CHANGE DRAWING MODE
     ======================================================= */

  const changeMode = (
    mode: DrawingMode
  ) => {
    setDrawingMode(mode);

    /*
     * Starting a new shape always clears unfinished
     * drawing state.
     */
    setDrawingState({
      points: [],
      previewPoint: null,
      circleRadius: 0,
    });
  };

  /* =======================================================
     GEOMETRY INFORMATION
     ======================================================= */

  const geometryInfo =
    useMemo(() => {
      if (!selectedAOI) {
        return null;
      }

      /*
       * CIRCLE
       */
      if (
        selectedAOI.type ===
        "circle"
      ) {
        const area =
          Math.PI *
          selectedAOI.radius **
            2;

        const perimeter =
          2 *
          Math.PI *
          selectedAOI.radius;

        return {
          area,
          perimeter,
          center:
            selectedAOI.center,
          type:
            "Circle" as const,
        };
      }

      /*
       * RECTANGLE / POLYGON
       */
      const coordinates =
        selectedAOI.coordinates;

      return {
        area:
          polygonArea(
            coordinates
          ),

        perimeter:
          polygonPerimeter(
            coordinates
          ),

        center:
          polygonCentroid(
            coordinates
          ),

        type:
          selectedAOI.type ===
          "rectangle"
            ? ("Rectangle" as const)
            : ("Polygon" as const),
      };
    }, [
      selectedAOI,
    ]);

  /* =======================================================
     PREVIEW RECTANGLE
     ======================================================= */

  const previewRectangle =
    drawingMode ===
      "rectangle" &&
    drawingState.points
      .length === 1 &&
    drawingState.previewPoint
      ? [
          drawingState.points[0],

          [
            drawingState.points[0][0],
            drawingState.previewPoint[1],
          ],

          drawingState.previewPoint,

          [
            drawingState.previewPoint[0],
            drawingState.points[0][1],
          ],
        ] as [number, number][]
      : null;

  /* =======================================================
     PREVIEW CIRCLE
     ======================================================= */

  const previewCircle =
    drawingMode ===
      "circle" &&
    drawingState.points
      .length === 1
      ? {
          center:
            drawingState.points[0],

          radius:
            drawingState.circleRadius,
        }
      : null;

  /* =======================================================
     AOI METADATA
     ======================================================= */

  const aoiMetadata =
    useMemo<AOIMetadata | null>(
      () => {
        if (
          !selectedAOI ||
          !geometryInfo
        ) {
          return null;
        }

        return {
          name:
            `Custom ${geometryInfo.type}`,

          geometry:
            selectedAOI,

          center:
            geometryInfo.center,

          area:
            `${formatNumber(
              geometryInfo.area /
                1_000_000
            )} km²`,

          perimeter:
            formatDistance(
              geometryInfo.perimeter
            ),
        };
      },
      [
        selectedAOI,
        geometryInfo,
      ]
    );

  /* =======================================================
     RUN ANALYSIS
     ======================================================= */

  const handleRunAnalysis =
    () => {
      if (
        !selectedAOI ||
        !geometryInfo ||
        !aoiMetadata ||
        !query.trim()
      ) {
        return;
      }

      onRunAnalysis?.(
        query.trim(),
        aoiMetadata,
        startDate,
        endDate
      );
    };

  /* =======================================================
     DATE PRESETS
     ======================================================= */

  const setDatePreset = (
    start: string,
    end: string
  ) => {
    setStartDate(start);
    setEndDate(end);
  };

  /* =======================================================
     RENDER
     ======================================================= */

  return (
    <main className="aoi-page">

      {/* ===================================================
          HEADER
          =================================================== */}

      <header className="aoi-header">

        <div className="aoi-brand">

          <div className="aoi-brand-name">
            SATQUERY AI
          </div>

          <div className="aoi-brand-subtitle">
            REMOTE SENSING INTELLIGENCE
          </div>

        </div>

      </header>


      {/* ===================================================
          MAIN LAYOUT
          =================================================== */}

      <section className="aoi-layout">

        {/* =================================================
            LEFT SIDEBAR
            ================================================= */}

        <aside className="aoi-sidebar">

          {/* =================================================
              INTRO
              ================================================= */}

          <div className="aoi-intro">

            <h2>
              DEFINE YOUR
              <br />
              INVESTIGATION
            </h2>

            <p>
              Select an area, choose a
              time range, and describe
              what you want to investigate.
            </p>

          </div>


          {/* =================================================
              SEARCH
              ================================================= */}

          <section className="aoi-section">

            <div className="aoi-section-title">
              SEARCH LOCATION
            </div>

            <div className="aoi-search-box">

              <span className="aoi-search-icon">
                ⌕
              </span>

              <input
                type="text"
                placeholder="Search for a city, region, or place..."
              />

            </div>

            <div className="aoi-help-text">
              e.g. Mumbai, India | California, USA |
              Amazon Rainforest
            </div>

          </section>


          {/* =================================================
              DRAW
              ================================================= */}

          <section className="aoi-section">

            <div className="aoi-section-title">
              DRAW AREA OF INTEREST
            </div>

            <div className="aoi-drawing-buttons">

              <button
                type="button"
                className={
                  drawingMode ===
                  "polygon"
                    ? "aoi-drawing-button active"
                    : "aoi-drawing-button"
                }
                onClick={() =>
                  changeMode(
                    "polygon"
                  )
                }
              >

                <span className="draw-icon polygon-icon">
                  ◇
                </span>

                Polygon

              </button>


              <button
                type="button"
                className={
                  drawingMode ===
                  "rectangle"
                    ? "aoi-drawing-button active"
                    : "aoi-drawing-button"
                }
                onClick={() =>
                  changeMode(
                    "rectangle"
                  )
                }
              >

                <span className="draw-icon">
                  □
                </span>

                Rectangle

              </button>


              <button
                type="button"
                className={
                  drawingMode ===
                  "circle"
                    ? "aoi-drawing-button active"
                    : "aoi-drawing-button"
                }
                onClick={() =>
                  changeMode(
                    "circle"
                  )
                }
              >

                <span className="draw-icon">
                  ○
                </span>

                Circle

              </button>

            </div>


            <div className="aoi-help-text">

              {drawingMode ===
                "polygon" &&
                "Click points on the map. Double click or right click to finish."}

              {drawingMode ===
                "rectangle" &&
                "Click one corner, then click the opposite corner."}

              {drawingMode ===
                "circle" &&
                "Click the center, then click to set the radius."}

            </div>

          </section>


          {/* =================================================
              DATE RANGE
              ================================================= */}

          <section className="aoi-section">

            <div className="aoi-section-title">
              DATE RANGE
            </div>


            <div className="aoi-date-grid">

              <label>

                <span>
                  Start date
                </span>

                <input
                  type="date"
                  value={startDate}
                  onChange={(
                    event
                  ) =>
                    setStartDate(
                      event.target.value
                    )
                  }
                />

              </label>


              <label>

                <span>
                  End date
                </span>

                <input
                  type="date"
                  value={endDate}
                  onChange={(
                    event
                  ) =>
                    setEndDate(
                      event.target.value
                    )
                  }
                />

              </label>

            </div>


            <div className="aoi-presets-title">
              QUICK PRESETS
            </div>


            <div className="aoi-date-presets">

              <button
                type="button"
                className={
                  startDate ===
                    "2024-04-17" &&
                  endDate ===
                    "2025-04-17"
                    ? "active"
                    : ""
                }
                onClick={() =>
                  setDatePreset(
                    "2024-04-17",
                    "2025-04-17"
                  )
                }
              >
                1 Year
              </button>


              <button
                type="button"
                className={
                  startDate ===
                    "2022-04-17" &&
                  endDate ===
                    "2025-04-17"
                    ? "active"
                    : ""
                }
                onClick={() =>
                  setDatePreset(
                    "2022-04-17",
                    "2025-04-17"
                  )
                }
              >
                3 Years
              </button>


              <button
                type="button"
                className={
                  startDate ===
                    "2020-04-17" &&
                  endDate ===
                    "2025-04-17"
                    ? "active"
                    : ""
                }
                onClick={() =>
                  setDatePreset(
                    "2020-04-17",
                    "2025-04-17"
                  )
                }
              >
                5 Years
              </button>


              <button
                type="button"
                className={
                  startDate ===
                    "2015-04-17" &&
                  endDate ===
                    "2025-04-17"
                    ? "active"
                    : ""
                }
                onClick={() =>
                  setDatePreset(
                    "2015-04-17",
                    "2025-04-17"
                  )
                }
              >
                10 Years
              </button>

            </div>

          </section>


          {/* =================================================
              CLEAR
              ================================================= */}

          {selectedAOI && (
            <button
              type="button"
              className="aoi-clear-button"
              onClick={
                clearAOI
              }
            >
              CLEAR AREA
            </button>
          )}

        </aside>


        {/* =================================================
            MAP
            ================================================= */}

        <section className="aoi-map-panel">

          <MapContainer
            center={[
              19.076,
              72.8777,
            ]}
            zoom={10}
            zoomControl={true}
            attributionControl={true}
            doubleClickZoom={false}
            className="aoi-map"
          >

            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution="Tiles © Esri"
            />


            <ScaleControl
              position="bottomleft"
              metric={true}
              imperial={false}
            />


            {/* =================================================
                DRAWING ENGINE
                ================================================= */}

            <DrawingInteraction
              mode={
                drawingMode
              }
              drawingState={
                drawingState
              }
              setDrawingState={
                setDrawingState
              }
              onComplete={
                handleAOIComplete
              }
            />


            {/* =================================================
                MAP VIEW
                ================================================= */}

            <AOIMapController
              selectedAOI={
                selectedAOI
              }
            />


            {/* =================================================
                SELECTED POLYGON
                ================================================= */}

            {selectedAOI?.type ===
              "polygon" && (
              <>
                <Polygon
                  positions={
                    selectedAOI.coordinates
                  }
                  interactive={
                    false
                  }
                  pathOptions={{
                    color:
                      "#ffffff",
                    weight: 3,
                    opacity: 1,
                    fillColor:
                      "#ffffff",
                    fillOpacity:
                      0.08,
                  }}
                />

                {selectedAOI.coordinates.map(
                  (
                    point,
                    index
                  ) => (
                    <CircleMarker
                      key={`polygon-point-${index}`}
                      center={
                        point
                      }
                      radius={
                        5
                      }
                      interactive={
                        false
                      }
                      pathOptions={{
                        color:
                          "#ffffff",
                        weight: 2,
                        fillColor:
                          "#ffffff",
                        fillOpacity:
                          1,
                      }}
                    />
                  )
                )}
              </>
            )}


            {/* =================================================
                SELECTED RECTANGLE
                ================================================= */}

            {selectedAOI?.type ===
              "rectangle" && (
              <>
                <Polygon
                  positions={
                    selectedAOI.coordinates
                  }
                  interactive={
                    false
                  }
                  pathOptions={{
                    color:
                      "#ffffff",
                    weight: 3,
                    opacity: 1,
                    fillColor:
                      "#ffffff",
                    fillOpacity:
                      0.08,
                  }}
                />

                {selectedAOI.coordinates.map(
                  (
                    point,
                    index
                  ) => (
                    <CircleMarker
                      key={`rectangle-point-${index}`}
                      center={
                        point
                      }
                      radius={
                        4
                      }
                      interactive={
                        false
                      }
                      pathOptions={{
                        color:
                          "#ffffff",
                        weight: 2,
                        fillColor:
                          "#ffffff",
                        fillOpacity:
                          1,
                      }}
                    />
                  )
                )}
              </>
            )}


            {/* =================================================
                SELECTED CIRCLE
                ================================================= */}

            {selectedAOI?.type ===
              "circle" && (
              <Circle
                center={
                  selectedAOI.center
                }
                radius={
                  selectedAOI.radius
                }
                interactive={
                  false
                }
                pathOptions={{
                  color:
                    "#ffffff",
                  weight: 3,
                  opacity: 1,
                  fillColor:
                    "#ffffff",
                  fillOpacity:
                    0.08,
                }}
              />
            )}


            {/* =================================================
                POLYGON PREVIEW
                ================================================= */}

            {drawingMode ===
              "polygon" &&
              drawingState.points
                .length >= 2 && (
              <Polygon
                positions={
                  drawingState.points
                }
                interactive={
                  false
                }
                pathOptions={{
                  color:
                    "#ffffff",
                  weight: 2,
                  dashArray:
                    "6 5",
                  fillOpacity:
                    0.03,
                }}
              />
            )}


            {/* =================================================
                RECTANGLE PREVIEW
                ================================================= */}

            {previewRectangle && (
              <Polygon
                positions={
                  previewRectangle
                }
                interactive={
                  false
                }
                pathOptions={{
                  color:
                    "#ffffff",
                  weight: 2,
                  dashArray:
                    "6 5",
                  fillOpacity:
                    0.03,
                }}
              />
            )}


            {/* =================================================
                CIRCLE PREVIEW
                ================================================= */}

            {previewCircle && (
              <Circle
                center={
                  previewCircle.center
                }
                radius={
                  previewCircle.radius
                }
                interactive={
                  false
                }
                pathOptions={{
                  color:
                    "#ffffff",
                  weight: 2,
                  dashArray:
                    "6 5",
                  fillOpacity:
                    0.03,
                }}
              />
            )}

          </MapContainer>

        </section>


        {/* =================================================
            RIGHT INFORMATION PANEL
            ================================================= */}

        <aside className="aoi-info-panel">

          <section className="aoi-info-section">

            <div className="aoi-info-title">
              AOI INFORMATION
            </div>


            {/* AREA */}

            <div className="aoi-info-stat">

              <span>
                AREA
              </span>

              <strong>
                {geometryInfo
                  ? `${formatNumber(
                      geometryInfo.area /
                        1_000_000
                    )} km²`
                  : "—"}
              </strong>

            </div>


            {/* PERIMETER */}

            <div className="aoi-info-stat">

              <span>
                PERIMETER
              </span>

              <strong>
                {geometryInfo
                  ? formatDistance(
                      geometryInfo.perimeter
                    )
                  : "—"}
              </strong>

            </div>


            {/* CENTER */}

            <div className="aoi-info-stat">

              <span>
                CENTER COORDINATES
              </span>

              <strong>
                {geometryInfo
                  ? `${formatCoordinate(
                      geometryInfo.center[0],
                      "N",
                      "S"
                    )}, ${formatCoordinate(
                      geometryInfo.center[1],
                      "E",
                      "W"
                    )}`
                  : "—"}
              </strong>

            </div>


            {/* LOCATION */}

            <div className="aoi-info-stat">

              <span>
                LOCATION
              </span>

              <strong>
                {geometryInfo
                  ? `Custom ${geometryInfo.type}`
                  : "Not selected"}
              </strong>

            </div>


            {/* DATE RANGE */}

            <div className="aoi-info-stat">

              <span>
                DATE RANGE
              </span>

              <strong>
                {startDate} →{" "}
                {endDate}
              </strong>

            </div>


            {/* =================================================
                QUERY
                ================================================= */}

            <div className="aoi-query-section">

              <div className="aoi-query-title">
                WHAT DO YOU WANT TO INVESTIGATE?
              </div>

              <textarea
                value={query}
                onChange={(
                  event
                ) =>
                  setQuery(
                    event.target.value
                  )
                }
                placeholder="Show where vegetation decreased between 2021 and 2025."
              />

              <div className="aoi-query-help">
                e.g. Show urban expansion,
                detect water changes,
                compare vegetation,
                analyze infrastructure growth...
              </div>

            </div>


            {/* =================================================
                RUN ANALYSIS
                ================================================= */}

            <button
              type="button"
              className="aoi-run-button"
              disabled={
                !selectedAOI ||
                !query.trim()
              }
              onClick={
                handleRunAnalysis
              }
            >

              <span>
                RUN ANALYSIS
              </span>

              <span>
                →
              </span>

            </button>

          </section>

        </aside>

      </section>

    </main>
  );
}

export default AOISelection;