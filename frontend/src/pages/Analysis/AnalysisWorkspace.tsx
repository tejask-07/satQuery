import type { QueryResponse } from "../../api/query";
import { MapContainer, TileLayer, Polygon, Circle } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./AnalysisWorkspace.css";

interface AnalysisWorkspaceProps {
  result: QueryResponse;
  onViewDetails: () => void;
  onViewLayers: () => void;
}

function AnalysisWorkspace({
  result,
  onViewDetails,
  onViewLayers,
}: AnalysisWorkspaceProps) {

  const { plan } = result;

  const confidence =
    result.confidence != null
      ? `${Math.round(result.confidence * 100)}%`
      : "—";

  const formatDate = (date: string) => {
    if (!date) return "—";

    const parsed = new Date(date);

    if (Number.isNaN(parsed.getTime())) {
      return date;
    }

    return parsed.toISOString().split("T")[0];
  };

  const statistics = result.statistics ?? {};

  /*
   * TEMPORARY MOCK HISTOGRAM DATA
   *
   * This is still mock data.
   *
   * Later this array will come from the backend's
   * actual NDVI distribution.
   */
  const distribution = [
    8,
    13,
    19,
    27,
    35,
    44,
    55,
    68,
    82,
    100,
    87,
    66,
    49,
    35,
    25,
    18,
    13,
    9,
    5,
    2,
  ];

  /*
   * TEMPORARY AOI
   *
   * Rough Mumbai-region polygon for visualization.
   * Replace with backend-generated geometry later.
   */
  const aoi: [number, number][] = [
    [19.32, 72.70],
    [19.45, 72.92],
    [19.30, 73.08],
    [19.02, 73.12],
    [18.78, 72.98],
    [18.70, 72.78],
    [18.88, 72.66],
    [19.12, 72.61],
  ];

  /*
   * TEMPORARY CHANGE LOCATIONS
   *
   * These are illustrative only.
   * Real detections will come from the backend.
   */
  const changeAreas: [number, number][] = [
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

  return (
    <main className="analysis-workspace">

      {/* =====================================================
          HEADER
          ===================================================== */}

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
              {plan.target || "IDENTIFYING"}
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

              {formatDate(plan.time_start)}

              <span className="date-arrow">
                →
              </span>

              {formatDate(plan.time_end)}

            </span>

          </div>

        </div>

        <div className="analysis-header-actions">

          <button type = "button" className="analysis-export">
            EXPORT
          </button>

          <button type="button" className="analysis-menu" aria-label="Open Menu">
            ☰
          </button>

        </div>

      </header>


      {/* =====================================================
          WORKSPACE
          ===================================================== */}

      <section className="analysis-layout">


        {/* ===================================================
            LEFT
            =================================================== */}

        <aside className="analysis-sidebar">


          {/* ===================================================
              QUERY
              =================================================== */}

          <section className="analysis-section analysis-query-section">

            <div className="analysis-section-label">
              QUERY
            </div>

            <p className="analysis-query">
              {plan.task || "Analysis request"}
            </p>

          </section>


          {/* ===================================================
              DATA SUMMARY
              =================================================== */}

          <section className="analysis-section analysis-data">

            <div className="analysis-section-label">
              DATA SUMMARY
            </div>

            <div className="analysis-data-row">

              <span className="analysis-data-label">
                SATELLITE
              </span>

              <span className="analysis-data-value">
                {plan.modalities.length > 0
                  ? plan.modalities.join(", ")
                  : "—"}
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
                {formatDate(plan.time_start)}
                {" → "}
                {formatDate(plan.time_end)}
              </span>

            </div>


            <div className="analysis-data-row">

              <span className="analysis-data-label">
                CLOUD COVER
              </span>

              <span className="analysis-data-value">
                2021: 6.3%
                <br />
                2025: 4.2%
              </span>

            </div>

          </section>


          {/* ===================================================
              INDICATORS
              =================================================== */}

          <section className="analysis-section analysis-indicators">

            <div className="analysis-section-label">
              PRIMARY INDICATOR
            </div>

            <div className="finding-analysis large">
              {plan.analysis.length > 0
                ? plan.analysis[0]
                : "NDVI Difference"}
            </div>


            <div className="analysis-section-label indicator-secondary-label">
              SUPPORTING INDICATOR
            </div>

            <div className="finding-analysis large">
              NDBI Increase
            </div>

          </section>


        </aside>


        {/* ===================================================
            CENTER — REAL SATELLITE MAP
            =================================================== */}

        <section className="analysis-map-panel">

          <div className="analysis-map">

            <MapContainer
              center={[19.076, 72.8777]}
              zoom={10}
              zoomControl={false}
              attributionControl={true}
              className="satellite-map"
            >

              <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                attribution="Tiles © Esri"
              />


              {/* AOI */}

              <Polygon
                positions={aoi}
                pathOptions={{
                  color: "#f5f1e9",
                  weight: 2,
                  fillColor: "#f5f1e9",
                  fillOpacity: 0.03,
                }}
              />


              {/* Mock change detections */}

              {changeAreas.map((position, index) => (
                <Circle
                  key={`${position[0]}-${position[1]}-${index}`}
                  center={position}
                  radius={900}
                  pathOptions={{
                    color: "#e33420",
                    weight: 0,
                    fillColor: "#e33420",
                    fillOpacity: 0.68,
                  }}
                />
              ))}

            </MapContainer>


            {/* =================================================
                MAP BADGE
                ================================================= */}

            <div className="map-analysis-badge">

              <span>
                NDVI CHANGE
              </span>

              <span>
                2021 → 2025
              </span>

            </div>


            {/* =================================================
                MAP CONTROLS
                ================================================= */}

            <div className="map-controls">

              <button type="button">
                +
              </button>

              <button type="button">
                −
              </button>

              <button type="button">
                ⌖
              </button>

              <button type="button">
                ⌁
              </button>

              <button type="button">
                ▱
              </button>

            </div>


            {/* =================================================
                LEGEND
                ================================================= */}

            <div className="map-legend">

              <div className="map-legend-title">
                NDVI CHANGE
              </div>

              <div className="legend-item">
                <span className="legend-swatch high" />
                <span>High decrease</span>
              </div>

              <div className="legend-item">
                <span className="legend-swatch moderate" />
                <span>Moderate decrease</span>
              </div>

              <div className="legend-item">
                <span className="legend-swatch slight" />
                <span>Slight decrease</span>
              </div>

              <div className="legend-item">
                <span className="legend-swatch unchanged" />
                <span>No change</span>
              </div>

              <div className="legend-item">
                <span className="legend-swatch increase" />
                <span>Increase</span>
              </div>

            </div>


            {/* =================================================
                SCALE
                ================================================= */}

            <div className="map-scale">

              <span className="map-scale-line" />

              <span>
                2 km
              </span>

            </div>


            {/* =================================================
                COORDINATES
                ================================================= */}

            <div className="map-coordinates">
              19.0760° N, 72.8777° E
            </div>

          </div>

        </section>


        {/* ===================================================
            RIGHT — FINDINGS
            =================================================== */}

        <aside className="findings-panel">


          {/* =================================================
              FINDINGS
              ================================================= */}

          <section className="findings-section">

            <div className="findings-label">
              FINDINGS
            </div>


            <div className="finding-stat">

              <div className="finding-stat-value">
                {statistics.area_affected
                  ? String(statistics.area_affected)
                  : "18.4 km²"}
              </div>

              <div className="finding-stat-label">
                AREA AFFECTED
              </div>

            </div>


            <div className="finding-stat">

              <div className="finding-stat-value">
                {statistics.average_ndvi_change
                  ? String(statistics.average_ndvi_change)
                  : "-23.7%"}
              </div>

              <div className="finding-stat-label">
                AVERAGE NDVI CHANGE
              </div>

            </div>


            <div className="finding-stat">

              <div className="finding-stat-value">
                {confidence}
              </div>

              <div className="finding-stat-label">
                CONFIDENCE
              </div>

            </div>

          </section>


          {/* =================================================
              DISTRIBUTION
              ================================================= */}

          <section className="findings-section distribution-section">

            <div className="findings-label">
              CHANGE DISTRIBUTION
            </div>


            <div className="distribution-chart">

              {distribution.map((height, index) => (
                <span
                  key={index}
                  style={{
                    height: `${height}%`,
                  }}
                />
              ))}

            </div>


            <div className="distribution-axis">
              <span>-1</span>
              <span>0</span>
              <span>1</span>
            </div>


            <div className="distribution-label">
              NDVI Change
            </div>

          </section>


          {/* =================================================
              VIEW DETAILS
              ================================================= */}

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