import "./AnalysisProcessing.css";

import type { AOIMetadata } from "../AOI/AOISelection";

interface AnalysisProcessingProps {
  query: string;
  aoi?: AOIMetadata | null;
  startDate?: string;
  endDate?: string;
}

/* =========================================================
   HELPERS
   ========================================================= */

function formatDate(date?: string) {
  if (!date) return "NOT SPECIFIED";

  const parsed = new Date(date);

  if (Number.isNaN(parsed.getTime())) {
    return date;
  }

  return parsed
    .toLocaleDateString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
    })
    .toUpperCase();
}

/*
 * AOIMetadata in the current SatQuery frontend stores geometry
 * points as [latitude, longitude].
 */
function getBounds(aoi?: AOIMetadata | null) {
  const coordinates = aoi?.geometry?.coordinates ?? [];

  if (!coordinates.length) {
    return {
      south: 18.9,
      north: 19.2,
      west: 72.7,
      east: 73.0,
    };
  }

  const lats = coordinates.map(([lat]) => lat);
  const lngs = coordinates.map(([, lng]) => lng);

  return {
    south: Math.min(...lats),
    north: Math.max(...lats),
    west: Math.min(...lngs),
    east: Math.max(...lngs),
  };
}

function getCenter(aoi?: AOIMetadata | null): [number, number] {
  const coordinates = aoi?.geometry?.coordinates ?? [];

  if (!coordinates.length) {
    return aoi?.center || [19.076, 72.8777];
  }

  const lat = coordinates.reduce((sum, point) => sum + point[0], 0) / coordinates.length;
  const lng = coordinates.reduce((sum, point) => sum + point[1], 0) / coordinates.length;

  return [lat, lng];
}

function getSatelliteImageUrl(aoi?: AOIMetadata | null) {
  const bounds = getBounds(aoi);

  const width = 1200;
  const height = 720;

  const latSpan = Math.max(bounds.north - bounds.south, 0.001);
  const lngSpan = Math.max(bounds.east - bounds.west, 0.001);

  const latPadding = latSpan * 0.65;
  const lngPadding = lngSpan * 0.65;

  const bbox = [
    bounds.west - lngPadding,
    bounds.south - latPadding,
    bounds.east + lngPadding,
    bounds.north + latPadding,
  ].join(",");

  const params = new URLSearchParams({
    bbox,
    bboxSR: "4326",
    imageSR: "4326",
    size: `${width},${height}`,
    format: "jpg",
    pixelType: "U8",
    f: "image",
  });

  return (
    "https://server.arcgisonline.com/ArcGIS/rest/services/" +
    "World_Imagery/MapServer/export?" +
    params.toString()
  );
}

/* =========================================================
   AOI OVERLAY
   ========================================================= */

/* =========================================================
   COMPONENT
   ========================================================= */

export default function AnalysisProcessing({
  query,
  aoi,
  startDate,
  endDate,
}: AnalysisProcessingProps) {
  const center = getCenter(aoi);
  const satelliteImage = getSatelliteImageUrl(aoi);

  return (
    <main className="processing-page">
      <header className="processing-header">
        <div className="processing-brand">
          <span className="processing-brand-mark" />

          <div className="processing-brand-copy">
            <div className="processing-brand-name">SATQUERY AI</div>
            <div className="processing-brand-subtitle">
              REMOTE SENSING INTELLIGENCE
            </div>
          </div>
        </div>

        <div className="processing-header-meta">
          <div className="processing-meta-block">
            <span className="processing-meta-label">INVESTIGATION</span>
            <span className="processing-meta-value">
              ANALYSIS INITIALIZING
            </span>
          </div>

          <div className="processing-header-divider" />

          <div className="processing-meta-block">
            <span className="processing-meta-label">ANALYSIS TYPE</span>
            <span className="processing-meta-value">
              REMOTE SENSING QUERY
            </span>
          </div>
        </div>
      </header>

      <div className="processing-layout">
        <aside className="processing-sidebar">
          <div className="processing-steps">
            <div className="processing-step complete">
              <span className="processing-step-number">01</span>
              <div>
                <strong>DEFINE</strong>
                <span>Area, time and query.</span>
              </div>
            </div>

            <div className="processing-step active">
              <span className="processing-step-number">02</span>
              <div>
                <strong>PROCESS</strong>
                <span>SatQuery is analyzing.</span>
              </div>
            </div>

            <div className="processing-step">
              <span className="processing-step-number">03</span>
              <div>
                <strong>UNDERSTAND</strong>
                <span>Insights and results.</span>
              </div>
            </div>
          </div>

          <div className="processing-sidebar-footer">
            <div className="processing-sidebar-footer-top">
              <span>SATQUERY AI</span>
              <span>v0.1.0</span>
            </div>

            <p>
              BRIDGING EARTH
              <br />
              AND UNDERSTANDING
            </p>

            <div className="processing-footer-line" />
          </div>
        </aside>

        <section className="processing-content">
          <div className="processing-hero">
            <div className="processing-kicker">02 / PROCESS</div>

            <h1>
              ANALYSIS
              <br />
              INITIALIZING
            </h1>

            <p>
              PREPARING YOUR SATELLITE INVESTIGATION
              <span className="processing-dots" aria-hidden="true">
                ...
              </span>
            </p>
          </div>

          <div className="processing-main-divider" />

          <section className="investigation-section">
            <div className="investigation-details">
              <div className="section-heading">INVESTIGATION SUMMARY</div>

              <div className="investigation-row">
                <span className="investigation-label">AREA OF INTEREST</span>

                <div className="investigation-value">
                  <strong>{aoi?.name || "SELECTED AREA"}</strong>
                  <span>
                    {center[0].toFixed(4)}° N&nbsp;&nbsp;
                    {center[1].toFixed(4)}° E
                  </span>
                </div>
              </div>

              <div className="investigation-row">
                <span className="investigation-label">AREA</span>

                <div className="investigation-value">
                  <strong>
                    {aoi?.area ? (aoi.area.includes("km²") ? aoi.area : `${aoi.area} km²`) : "SELECTED REGION"}
                  </strong>
                </div>
              </div>

              <div className="investigation-row">
                <span className="investigation-label">DATE RANGE</span>

                <div className="investigation-value date-value">
                  <strong>{formatDate(startDate)}</strong>
                  <span className="date-arrow">→</span>
                  <strong>{formatDate(endDate)}</strong>
                </div>
              </div>

              <div className="investigation-row query-row">
                <span className="investigation-label">QUERY</span>

                <div className="investigation-value">
                  <strong className="query-value">“{query}”</strong>
                </div>
              </div>
            </div>

            <div className="processing-map">
              <img
                src={satelliteImage}
                alt="Satellite imagery of selected area"
                className="processing-satellite-image"
              />

              <div className="processing-map-overlay" />

              <div className="processing-map-label">
                <span>SELECTED AREA</span>
                <strong>SENTINEL-2 · TRUE COLOR</strong>
              </div>

              <div className="processing-map-coordinates">
                {center[0].toFixed(4)}° N&nbsp;&nbsp;
                {Math.abs(center[1]).toFixed(4)}° E
              </div>

              <div className="processing-map-north">
                <span>▲</span>
                <strong>N</strong>
              </div>

              <div className="processing-scale">
                <span>0</span>
                <div className="processing-scale-line">
                  <span />
                  <span />
                  <span />
                </div>
                <span>1 km</span>
              </div>

              <div className="processing-live-indicator">
                <span />
                ANALYSIS IN PROGRESS
              </div>
            </div>
          </section>
        </section>
      </div>
    </main>
  );
}
