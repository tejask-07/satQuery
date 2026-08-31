import "./ResultsInsights.css";

interface ResultsInsightsProps {
  result?: any;
}

function ResultsInsights({ result }: ResultsInsightsProps) {
  const statistics = result?.statistics ?? {};

  const confidence =
    result?.confidence != null
      ? `${Math.round(result.confidence * 100)}%`
      : "91%";

  const areaAffected =
    statistics.area_affected ?? "18.4 km²";

  const averageNdvi =
    statistics.average_ndvi_change ?? "-23.7%";

  return (
    <main className="results-workspace">

      {/* =====================================================
          PAGE HEADER
          ===================================================== */}

      <header className="results-page-header">
        <div className="results-page-number">
          04.
        </div>

        <h1>
          RESULTS &amp; INSIGHTS
        </h1>
      </header>


      {/* =====================================================
          MAIN RESULT CARD
          ===================================================== */}

      <section className="results-card">

        {/* ===================================================
            CARD HEADER
            =================================================== */}

        <div className="results-card-header">
          RESULT SUMMARY
        </div>


        {/* ===================================================
            LEFT COLUMN
            =================================================== */}

        <div className="results-left">

          {/* =================================================
              WHAT CHANGED
              ================================================= */}

          <section className="results-metrics">

            <div className="results-section-label">
              WHAT CHANGED?
            </div>

            <div className="results-metric-grid">

              <div className="results-metric">
                <div className="results-metric-value">
                  {areaAffected}
                </div>

                <div className="results-metric-label">
                  Area Affected
                </div>
              </div>


              <div className="results-metric">
                <div className="results-metric-value">
                  {averageNdvi}
                </div>

                <div className="results-metric-label">
                  Avg NDVI Change
                </div>
              </div>


              <div className="results-metric">
                <div className="results-metric-value">
                  1.24M
                </div>

                <div className="results-metric-label">
                  Pixels Analyzed
                </div>
              </div>


              <div className="results-metric">
                <div className="results-metric-value">
                  {confidence}
                </div>

                <div className="results-metric-label">
                  Confidence
                </div>
              </div>

            </div>

          </section>


          {/* =================================================
              CHANGE HIGHLIGHTS
              ================================================= */}

          <section className="results-highlights">

            <div className="results-section-label">
              CHANGE HIGHLIGHTS
            </div>


            <div className="highlight-item">
              <span className="highlight-dot red" />

              <p>
                Vegetation decline is concentrated in the
                northern and eastern portions of the AOI.
              </p>
            </div>


            <div className="highlight-item">
              <span className="highlight-dot orange" />

              <p>
                Many of these areas coincide with increased
                built-up surface signatures.
              </p>
            </div>


            <div className="highlight-item">
              <span className="highlight-dot yellow" />

              <p>
                Changes are persistent across multiple
                observations in the selected period.
              </p>
            </div>

          </section>


          {/* =================================================
              EVIDENCE
              ================================================= */}

          <section className="results-evidence">

            <div className="results-section-label">
              EVIDENCE
            </div>


            <div className="evidence-grid">

              <div className="evidence-item">
                <div className="evidence-image evidence-ndvi-difference">
                  NDVI
                </div>

                <div className="evidence-label">
                  NDVI Difference
                </div>
              </div>


              <div className="evidence-item">
                <div className="evidence-image evidence-ndvi-2021">
                  NDVI
                </div>

                <div className="evidence-label">
                  NDVI (2021)
                </div>
              </div>


              <div className="evidence-item">
                <div className="evidence-image evidence-ndvi-2025">
                  NDVI
                </div>

                <div className="evidence-label">
                  NDVI (2025)
                </div>
              </div>


              <div className="evidence-item">
                <div className="evidence-image evidence-ndbi">
                  NDBI
                </div>

                <div className="evidence-label">
                  NDBI Change
                </div>
              </div>

            </div>

          </section>

        </div>


        {/* ===================================================
            RIGHT COLUMN
            =================================================== */}

        <div className="results-right">

          {/* =================================================
              CONFIDENCE
              ================================================= */}

          <section className="confidence-section">

            <div className="results-section-label">
              CONFIDENCE
            </div>

            <div className="confidence-ring">

              <div className="confidence-ring-inner">
                <span>
                  {confidence}
                </span>
              </div>

            </div>

            <div className="confidence-status">
              High Confidence
            </div>

          </section>


          {/* =================================================
              BASED ON
              ================================================= */}

          <section className="based-on-section">

            <div className="results-section-label">
              BASED ON
            </div>


            <div className="check-item">
              <span className="check-mark">✓</span>
              <span>Multi-temporal consistency</span>
            </div>


            <div className="check-item">
              <span className="check-mark">✓</span>
              <span>Low cloud cover</span>
            </div>


            <div className="check-item">
              <span className="check-mark">✓</span>
              <span>Strong indicator agreement</span>
            </div>


            <div className="check-item">
              <span className="check-mark">✓</span>
              <span>Spatial coherence</span>
            </div>

          </section>


          {/* =================================================
              AI INTERPRETATION
              ================================================= */}

          <section className="interpretation-section">

            <div className="results-section-label">
              AI INTERPRETATION
            </div>


            <p>
              The analysis shows a significant reduction in
              vegetation health between 2021 and 2025.
            </p>


            <p>
              The strongest declines are aligned with
              expanding urban and infrastructure development.
            </p>


            <p>
              This pattern is supported by a corresponding
              increase in NDBI, indicating surface
              transformation from natural to built-up areas.
            </p>

          </section>


          {/* =================================================
              REPORT BUTTON
              ================================================= */}

          <button
            type="button"
            className="download-report-button"
          >
            <span>
              DOWNLOAD REPORT
            </span>

            <span>
              ↓
            </span>
          </button>

        </div>

      </section>

    </main>
  );
}

export default ResultsInsights;