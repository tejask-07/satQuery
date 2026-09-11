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

export type AOIGeometry = {
  type: "rectangle";
  coordinates: [number, number][];
};

export interface AOIMetadata {
  name: string;
  geometry: AOIGeometry;
  center: [number, number];
  area: string;
  perimeter: string;
}

export interface AOISelectionProps {
  initialQuery?: string;
  loading?: boolean;

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
   DRAWING STATE
   ========================================================= */

interface DrawingState {
  startPoint: [number, number] | null;
  currentPoint: [number, number] | null;
}


/* =========================================================
   DRAWING ENGINE
   ========================================================= */

interface DrawingInteractionProps {
  drawingState: DrawingState;

  setDrawingState: Dispatch<
    SetStateAction<DrawingState>
  >;

  onComplete: (
    aoi: AOIGeometry
  ) => void;
}


function DrawingInteraction({
  drawingState,
  setDrawingState,
  onComplete,
}: DrawingInteractionProps) {

  const map = useMap();

  const stateRef =
    useRef(drawingState);

  const completeRef =
    useRef(onComplete);

  const isDrawingRef =
    useRef(false);


  useEffect(() => {
    stateRef.current =
      drawingState;
  }, [drawingState]);


  useEffect(() => {
    completeRef.current =
      onComplete;
  }, [onComplete]);


  useEffect(() => {

    const mapContainer =
      map.getContainer();

    /*
     * Transparent drawing surface above
     * the satellite imagery.
     */
    const overlay =
      document.createElement("div");

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

    mapContainer.appendChild(
      overlay
    );


    /* -------------------------------------------------------
       MAP COORDINATE
       ------------------------------------------------------- */

    const getPoint = (
      event: MouseEvent | PointerEvent
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


    /* -------------------------------------------------------
       UPDATE STATE
       ------------------------------------------------------- */

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


    /* -------------------------------------------------------
       POINTER DOWN
       ------------------------------------------------------- */

    const handlePointerDown = (
      event: PointerEvent
    ) => {

      if (
        event.button !== 0 &&
        event.pointerType !== "touch"
      ) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      const point =
        getPoint(event);


      /*
       * Start a new rectangle.
       */
      if (
        !isDrawingRef.current
      ) {

        isDrawingRef.current =
          true;

        updateState({
          startPoint: point,
          currentPoint: point,
        });

        return;
      }
    };


    /* -------------------------------------------------------
       POINTER MOVE
       ------------------------------------------------------- */

    const handlePointerMove = (
      event: PointerEvent
    ) => {

      if (
        !isDrawingRef.current
      ) {
        return;
      }

      event.preventDefault();

      const point =
        getPoint(event);

      updateState(
        (current) => ({
          ...current,
          currentPoint: point,
        })
      );
    };


    /* -------------------------------------------------------
       POINTER UP
       ------------------------------------------------------- */

    const handlePointerUp = (
      event: PointerEvent
    ) => {

      if (
        !isDrawingRef.current
      ) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      const point =
        getPoint(event);

      const start =
        stateRef.current.startPoint;

      if (!start) {
        isDrawingRef.current =
          false;

        return;
      }


      /*
       * Create the rectangle.
       */
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


      /*
       * Ignore extremely tiny accidental
       * clicks instead of creating a tiny AOI.
       */
      const width =
        haversineDistance(
          start,
          [
            start[0],
            point[1],
          ]
        );

      const height =
        haversineDistance(
          start,
          [
            point[0],
            start[1],
          ]
        );


      isDrawingRef.current =
        false;


      if (
        width < 20 ||
        height < 20
      ) {

        updateState({
          startPoint: null,
          currentPoint: null,
        });

        return;
      }


      completeRef.current({
        type: "rectangle",
        coordinates,
      });


      updateState({
        startPoint: null,
        currentPoint: null,
      });
    };


    /* -------------------------------------------------------
       CONTEXT MENU
       ------------------------------------------------------- */

    const handleContextMenu = (
      event: MouseEvent
    ) => {

      event.preventDefault();
      event.stopPropagation();
    };


    /* -------------------------------------------------------
       EVENTS
       ------------------------------------------------------- */

    overlay.addEventListener(
      "pointerdown",
      handlePointerDown
    );

    overlay.addEventListener(
      "pointermove",
      handlePointerMove
    );

    overlay.addEventListener(
      "pointerup",
      handlePointerUp
    );

    overlay.addEventListener(
      "pointercancel",
      handlePointerUp
    );

    overlay.addEventListener(
      "contextmenu",
      handleContextMenu
    );


    /*
     * Normal map dragging would conflict
     * with rectangle drawing.
     */
    map.dragging.disable();


    /* -------------------------------------------------------
       CLEANUP
       ------------------------------------------------------- */

    return () => {

      overlay.removeEventListener(
        "pointerdown",
        handlePointerDown
      );

      overlay.removeEventListener(
        "pointermove",
        handlePointerMove
      );

      overlay.removeEventListener(
        "pointerup",
        handlePointerUp
      );

      overlay.removeEventListener(
        "pointercancel",
        handlePointerUp
      );

      overlay.removeEventListener(
        "contextmenu",
        handleContextMenu
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

  }, [
    map,
    setDrawingState,
  ]);


  return null;
}


/* =========================================================
   MAP VIEW CONTROLLER
   ========================================================= */

function AOIMapController({
  selectedAOI,
  targetLocation,
}: {
  selectedAOI:
    AOIGeometry | null;

  targetLocation?: {
    center: [number, number];
    zoom: number;
  } | null;
}) {

  const map =
    useMap();


  /*
   * Search location navigation.
   */
  useEffect(() => {

    if (
      targetLocation &&
      !selectedAOI
    ) {

      map.setView(
        targetLocation.center,
        targetLocation.zoom,
        {
          animate: true,
        }
      );
    }

  }, [
    map,
    targetLocation,
    selectedAOI,
  ]);


  /*
   * Fit the map around the selected AOI.
   */
  useEffect(() => {

    if (!selectedAOI) {
      return;
    }

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
   RECTANGLE AREA
   ========================================================= */

function polygonArea(
  coordinates:
    [number, number][]
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
   RECTANGLE PERIMETER
   ========================================================= */

function polygonPerimeter(
  coordinates:
    [number, number][]
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
   RECTANGLE CENTER
   ========================================================= */

function rectangleCenter(
  coordinates:
    [number, number][]
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
  loading = false,
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
     SEARCH
     ======================================================= */

  const [
    searchQuery,
    setSearchQuery,
  ] = useState("");


  const [
    targetLocation,
    setTargetLocation,
  ] = useState<{
    center: [number, number];
    zoom: number;
  } | null>(null);


  const LOCATION_PRESETS:
    Record<
      string,
      {
        center: [number, number];
        zoom: number;
      }
    > = {

      mumbai: {
        center: [
          19.076,
          72.8777,
        ],
        zoom: 11,
      },

      california: {
        center: [
          36.7783,
          -119.4179,
        ],
        zoom: 7,
      },

      amazon: {
        center: [
          -3.4653,
          -62.2159,
        ],
        zoom: 7,
      },

      delhi: {
        center: [
          28.6139,
          77.209,
        ],
        zoom: 11,
      },

      bangalore: {
        center: [
          12.9716,
          77.5946,
        ],
        zoom: 11,
      },
    };


  const handleSearchChange = (
    value: string
  ) => {

    setSearchQuery(
      value
    );

    const key =
      value
        .toLowerCase()
        .trim();

    for (
      const [
        preset,
        location,
      ] of Object.entries(
        LOCATION_PRESETS
      )
    ) {

      if (
        key.includes(preset)
      ) {

        setTargetLocation(
          location
        );

        break;
      }
    }
  };


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
     DRAWING
     ======================================================= */

  const [
    drawingState,
    setDrawingState,
  ] = useState<DrawingState>({
    startPoint: null,
    currentPoint: null,
  });


  /* =======================================================
     SELECTED AOI
     ======================================================= */

  const [
    selectedAOI,
    setSelectedAOI,
  ] = useState<
    AOIGeometry | null
  >(null);


  /* =======================================================
     COMPLETE AOI
     ======================================================= */

  const handleAOIComplete = (
    aoi: AOIGeometry
  ) => {

    setSelectedAOI(
      aoi
    );

    setDrawingState({
      startPoint: null,
      currentPoint: null,
    });

    onAOIChange?.(
      aoi
    );
  };


  /* =======================================================
     CLEAR AOI
     ======================================================= */

  const clearAOI = () => {

    setSelectedAOI(
      null
    );

    setDrawingState({
      startPoint: null,
      currentPoint: null,
    });

    onAOIChange?.(
      null
    );
  };


  /* =======================================================
     PREVIEW RECTANGLE
     ======================================================= */

  const previewRectangle =
    drawingState.startPoint &&
    drawingState.currentPoint
      ? [
          drawingState.startPoint,

          [
            drawingState.startPoint[0],
            drawingState.currentPoint[1],
          ],

          drawingState.currentPoint,

          [
            drawingState.currentPoint[0],
            drawingState.startPoint[1],
          ],
        ] as [number, number][]
      : null;


  /* =======================================================
     DISPLAY GEOMETRY
     
     Important:
     While dragging, the preview rectangle is treated
     as the active geometry so the information panel
     updates continuously.
     ======================================================= */

  const displayGeometry =
    previewRectangle ||
    selectedAOI?.coordinates ||
    null;


  /* =======================================================
     LIVE GEOMETRY INFORMATION
     ======================================================= */

  const geometryInfo =
    useMemo(() => {

      if (
        !displayGeometry ||
        displayGeometry.length < 3
      ) {
        return null;
      }

      return {
        area:
          polygonArea(
            displayGeometry
          ),

        perimeter:
          polygonPerimeter(
            displayGeometry
          ),

        center:
          rectangleCenter(
            displayGeometry
          ),

        type:
          "Rectangle" as const,

        isDrawing:
          Boolean(
            previewRectangle
          ),
      };

    }, [
      displayGeometry,
      previewRectangle,
    ]);


  /* =======================================================
     AOI METADATA
     ======================================================= */

  const aoiMetadata =
    useMemo<
      AOIMetadata | null
    >(() => {

      if (
        !selectedAOI ||
        !geometryInfo
      ) {
        return null;
      }

      return {

        name:
          "Custom Rectangle",

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

    }, [
      selectedAOI,
      geometryInfo,
    ]);


  /* =======================================================
     RUN ANALYSIS
     ======================================================= */

  const handleRunAnalysis =
    () => {

      if (
        loading ||
        !selectedAOI ||
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

    setStartDate(
      start
    );

    setEndDate(
      end
    );
  };


  /* =======================================================
     RENDER
     ======================================================= */

  return (

    <main className="aoi-page">

      {/* =================================================
          HEADER
          ================================================= */}

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


      {/* =================================================
          MAIN LAYOUT
          ================================================= */}

      <section className="aoi-layout">


        {/* =================================================
            LEFT SIDEBAR
            ================================================= */}

        <aside className="aoi-sidebar">


          {/* =================================================
              INTRO
              ================================================= */}

          <div className="aoi-intro">

            <h1>
              DEFINE YOUR
              <br />
              INVESTIGATION.
            </h1>

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
                value={searchQuery}
                onChange={(event) =>
                  handleSearchChange(
                    event.target.value
                  )
                }
              />

            </div>

            <div className="aoi-help-text">
              e.g. Mumbai, India | California, USA |
              Amazon Rainforest (presets supported)
            </div>

          </section>


          {/* =================================================
              DRAW AREA
              ================================================= */}

          <section className="aoi-section">

            <div className="aoi-section-title">
              DRAW AREA OF INTEREST
            </div>

            <div className="aoi-drawing-buttons">

              <button
                type="button"
                className="aoi-drawing-button active"
                onClick={() => {
                  setDrawingState({
                    startPoint: null,
                    currentPoint: null,
                  });
                }}
              >

                <span className="draw-icon">
                  □
                </span>

                Rectangle

              </button>

            </div>

            <div className="aoi-help-text">

              {drawingState.startPoint
                ? "Release the mouse to confirm the area."
                : "Click and drag on the map to define your area."}

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
                  onChange={(event) =>
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
                  onChange={(event) =>
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
                MAP CONTROLLER
                ================================================= */}

            <AOIMapController
              selectedAOI={
                selectedAOI
              }

              targetLocation={
                targetLocation
              }
            />


            {/* =================================================
                SELECTED RECTANGLE
                ================================================= */}

            {selectedAOI && (

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

                    weight:
                      3,

                    opacity:
                      1,

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
                      key={
                        `rectangle-point-${index}`
                      }

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

                        weight:
                          2,

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
                LIVE RECTANGLE PREVIEW
                ================================================= */}

            {previewRectangle && (

              <>

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

                    weight:
                      2,

                    dashArray:
                      "6 5",

                    fillColor:
                      "#ffffff",

                    fillOpacity:
                      0.08,
                  }}
                />


                {previewRectangle.map(
                  (
                    point,
                    index
                  ) => (

                    <CircleMarker
                      key={
                        `preview-point-${index}`
                      }

                      center={
                        point
                      }

                      radius={
                        3
                      }

                      interactive={
                        false
                      }

                      pathOptions={{
                        color:
                          "#ffffff",

                        weight:
                          2,

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


            {/* =================================================
                AREA
                ================================================= */}

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


            {/* =================================================
                PERIMETER
                ================================================= */}

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


            {/* =================================================
                CENTER
                ================================================= */}

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


            {/* =================================================
                LOCATION
                ================================================= */}

            <div className="aoi-info-stat">

              <span>
                LOCATION
              </span>

              <strong>
                {geometryInfo
                  ? "Custom Rectangle"
                  : "Not selected"}
              </strong>

            </div>


            {/* =================================================
                DATE RANGE
                ================================================= */}

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
                value={
                  query
                }

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
                loading ||
                !selectedAOI ||
                !query.trim()
              }

              onClick={
                handleRunAnalysis
              }
            >

              <span>
                {loading
                  ? "ANALYZING..."
                  : "RUN ANALYSIS"}
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