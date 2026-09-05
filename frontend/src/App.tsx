import { useState } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useNavigate,
} from "react-router-dom";

import LandingPage from "./pages/Landing/LandingPage";

import AOISelection, {
  type AOIGeometry,
} from "./pages/AOI/AOISelection";

import AnalysisWorkspace from "./pages/Analysis/AnalysisWorkspace";
import ResultsInsights from "./pages/Results/ResultsInsights";
import LayersVisualization from "./pages/Layers/LayerVisualization";

import {
  submitQuery,
  type QueryResponse,
} from "./api/query";

import "./index.css";

const USE_MOCK_DATA = false;

const DEFAULT_QUERY =
  "compare vegetation change between 2021 and 2025";

/* =========================================================
   MOCK RESULT
   ========================================================= */

const MOCK_RESULT: QueryResponse = {
  status: "analysis complete",

  answer:
    "Vegetation decreased across several regions of the Mumbai Urban Region between 2021 and 2025.",

  confidence: 0.91,

  plan: {
    task:
      "Show where vegetation decreased between 2021 and 2025.",

    target: "Mumbai Urban Region",

    time_start: "2021-04-17",

    time_end: "2025-04-17",

    modalities: ["Sentinel-2 (L2A)"],

    metric: "NDVI",

    direction: "decrease",

    analysis: [
      "Change detection",
    ],

    output: [
      "NDVI Difference",
      "NDBI Increase",
    ],
  },

  statistics: {
    area_affected: "18.4 km²",
    average_ndvi_change: "-23.7%",
  },

  layers: [],

  evidence: [],

  execution_trace: [
    "Query understood",
    "AOI identified",
    "Imagery selected",
    "NDVI computed (2021)",
    "NDVI computed (2025)",
    "Change detection",
  ],
};

/* =========================================================
   APP CONTENT
   ========================================================= */

function AppContent() {
  const navigate = useNavigate();

  /* =======================================================
     CURRENT QUERY
     ======================================================= */

  const [currentQuery, setCurrentQuery] =
    useState<string>(() => {
      try {
        return (
          sessionStorage.getItem(
            "satquery_last_query"
          ) || DEFAULT_QUERY
        );
      } catch {
        return DEFAULT_QUERY;
      }
    });

  /* =======================================================
     RESULT
     ======================================================= */

  const [result, setResult] =
    useState<QueryResponse | null>(() => {
      try {
        const saved =
          sessionStorage.getItem(
            "satquery_last_result"
          );

        return saved
          ? (JSON.parse(saved) as QueryResponse)
          : null;
      } catch {
        return null;
      }
    });

  /* =======================================================
     LOADING
     ======================================================= */

  const [loading, setLoading] =
    useState(false);

  /* =======================================================
     ERROR
     ======================================================= */

  const [error, setError] =
    useState<string | null>(null);

  /* =========================================================
     PERSIST RESULT
     ========================================================= */

  const persistResult = (
    query: string,
    response: QueryResponse
  ) => {
    setResult(response);

    try {
      sessionStorage.setItem(
        "satquery_last_result",
        JSON.stringify(response)
      );

      sessionStorage.setItem(
        "satquery_last_query",
        query
      );
    } catch {
      // Ignore storage failures.
    }
  };

  /* =========================================================
     LANDING → AOI
     ========================================================= */

  const handleLandingSubmit = (
    query: string
  ) => {
    const nextQuery =
      query.trim() || DEFAULT_QUERY;

    setCurrentQuery(nextQuery);
    setError(null);

    navigate("/aoi");
  };

  /* =========================================================
     AOI → ANALYSIS
     
     IMPORTANT:
     Navigate to /analysis BEFORE waiting for the backend.
     The backend can take several minutes.
     ========================================================= */

  const handleRunAnalysis = async (
    query: string,
    aoi: {
      name: string;
      geometry: AOIGeometry;
      center: [number, number];
      area: string;
      perimeter: string;
    },
    startDate: string,
    endDate: string
  ) => {
    const queryToRun =
      query.trim() || DEFAULT_QUERY;

    setCurrentQuery(queryToRun);
    setLoading(true);
    setError(null);
    setResult(null);

    /*
     * GO TO ANALYSIS IMMEDIATELY.
     *
     * The Analysis page will show a processing state
     * while submitQuery() runs in the background.
     */
    navigate("/analysis");

    try {
      /* ===================================================
         MOCK MODE
         =================================================== */

      if (USE_MOCK_DATA) {
        await new Promise((resolve) =>
          setTimeout(resolve, 700)
        );

        persistResult(
          queryToRun,
          MOCK_RESULT
        );

        return;
      }

      /* ===================================================
         REAL BACKEND
         =================================================== */

      console.log(
        "[SatQuery] Starting analysis..."
      );

      console.log(
        "[SatQuery] Query:",
        queryToRun
      );

      console.log(
        "[SatQuery] AOI:",
        aoi.geometry
      );

      console.log(
        "[SatQuery] Time:",
        startDate,
        "→",
        endDate
      );

      const response =
        await submitQuery(
          queryToRun,
          {
            aoi: aoi.geometry,
            time_start: startDate,
            time_end: endDate,
          }
        );

      console.log(
        "[SatQuery] Analysis complete:",
        response
      );

      /*
       * Store the real backend result.
       */
      persistResult(
        queryToRun,
        response
      );
    } catch (err) {
      console.error(
        "[SatQuery] Analysis failed:",
        err
      );

      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the query."
      );
    } finally {
      setLoading(false);
    }
  };

  /* =========================================================
     LEGACY / REQUERY
     ========================================================= */

  const handleQuery = async (
    query: string,
    aoi?: unknown
  ) => {
    const queryToRun =
      query.trim() ||
      currentQuery ||
      DEFAULT_QUERY;

    setCurrentQuery(queryToRun);
    setLoading(true);
    setError(null);

    /*
     * If the user re-runs an analysis from the
     * Analysis workspace, keep them on Analysis
     * while the new request is processing.
     */
    navigate("/analysis");

    try {
      const response =
        await submitQuery(
          queryToRun,
          {
            aoi:
              aoi as
                | AOIGeometry
                | undefined,
          }
        );

      persistResult(
        queryToRun,
        response
      );
    } catch (err) {
      console.error(
        "Query failed:",
        err
      );

      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the query."
      );
    } finally {
      setLoading(false);
    }
  };

  /* =========================================================
     LANDING
     ========================================================= */

  const Landing = () => {
    return (
      <>
        <LandingPage
          onSubmit={
            handleLandingSubmit
          }
          loading={loading}
          error={error}
        />

        {error && (
          <div className="query-error">
            {error}
          </div>
        )}
      </>
    );
  };

  /* =========================================================
     AOI
     ========================================================= */

  const AOI = () => {
    return (
      <AOISelection
        initialQuery={
          currentQuery
        }
        onRunAnalysis={
          handleRunAnalysis
        }
      />
    );
  };

  /* =========================================================
     ANALYSIS
     ========================================================= */

  const Analysis = () => {

    /*
     * BACKEND IS CURRENTLY PROCESSING.
     *
     * result is intentionally null here.
     * DO NOT redirect.
     */

    if (loading && !result) {
      return (
        <main
          style={{
            minHeight: "100vh",
            background: "#f5f1e8",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily:
              "Lexend, sans-serif",
            padding: "40px",
          }}
        >
          <div
            style={{
              width: "520px",
              maxWidth: "100%",
              textAlign: "center",
            }}
          >
            <div
              style={{
                fontSize: "11px",
                letterSpacing: "0.16em",
                fontWeight: 500,
                marginBottom: "22px",
              }}
            >
              SATQUERY AI
            </div>

            <div
              style={{
                fontSize: "28px",
                fontWeight: 600,
                letterSpacing: "-0.02em",
                marginBottom: "16px",
              }}
            >
              ANALYSIS IN PROGRESS
            </div>

            <div
              style={{
                fontSize: "13px",
                lineHeight: 1.7,
                color: "#68645c",
                maxWidth: "440px",
                margin: "0 auto",
              }}
            >
              Processing satellite imagery,
              calculating spectral indices,
              detecting temporal changes and
              assembling spatial evidence.
            </div>

            <div
              style={{
                marginTop: "34px",
                height: "2px",
                background: "#d8d2c6",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: "38%",
                  height: "100%",
                  background: "#171717",
                  animation:
                    "satquery-analysis-progress 1.5s ease-in-out infinite",
                }}
              />
            </div>

            <div
              style={{
                marginTop: "18px",
                fontSize: "10px",
                lineHeight: 1.6,
                letterSpacing: "0.08em",
                fontFamily:
                  "JetBrains Mono, monospace",
                color: "#8a867d",
              }}
            >
              SEARCHING IMAGERY → COMPUTING
              INDICES → DETECTING CHANGE
            </div>

            <div
              style={{
                marginTop: "30px",
                paddingTop: "18px",
                borderTop:
                  "1px solid #ddd7cc",
                fontSize: "11px",
                color: "#88847b",
                fontFamily:
                  "JetBrains Mono, monospace",
              }}
            >
              {currentQuery}
            </div>
          </div>

          <style>
            {`
              @keyframes satquery-analysis-progress {
                0% {
                  transform: translateX(-150%);
                }

                100% {
                  transform: translateX(380%);
                }
              }
            `}
          </style>
        </main>
      );
    }

    /*
     * BACKEND FAILED BEFORE PRODUCING A RESULT.
     */

    if (error && !result) {
      return (
        <main
          style={{
            minHeight: "100vh",
            background: "#f5f1e8",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily:
              "Lexend, sans-serif",
            padding: "40px",
          }}
        >
          <div
            style={{
              width: "520px",
              maxWidth: "100%",
              padding: "40px",
              border:
                "1px solid #d8d2c6",
              background: "#faf8f2",
            }}
          >
            <div
              style={{
                fontSize: "11px",
                letterSpacing: "0.14em",
                fontWeight: 500,
                marginBottom: "18px",
              }}
            >
              ANALYSIS ERROR
            </div>

            <div
              style={{
                fontSize: "14px",
                lineHeight: 1.7,
                color: "#333",
              }}
            >
              {error}
            </div>

            <button
              type="button"
              onClick={() =>
                navigate("/aoi")
              }
              style={{
                marginTop: "28px",
                padding:
                  "12px 18px",
                border: "none",
                background: "#171717",
                color: "#fff",
                fontFamily:
                  "Lexend, sans-serif",
                fontSize: "10px",
                fontWeight: 500,
                letterSpacing:
                  "0.08em",
                cursor: "pointer",
              }}
            >
              BACK TO AOI
            </button>
          </div>
        </main>
      );
    }

    /*
     * REAL RESULT EXISTS.
     *
     * Render the actual Analysis Workspace.
     */

    if (result) {
      return (
        <AnalysisWorkspace
          result={result}
          currentQuery={
            currentQuery
          }
          onViewDetails={() =>
            navigate("/results")
          }
          onViewLayers={() =>
            navigate("/layers")
          }
          onRequery={
            handleQuery
          }
          loading={loading}
        />
      );
    }

    /*
     * Direct visit to /analysis with no
     * previous analysis.
     */

    return (
      <Navigate
        to="/aoi"
        replace
      />
    );
  };

  /* =========================================================
     RESULTS
     ========================================================= */

  const Results = () => {
    if (!result) {
      return (
        <Navigate
          to="/aoi"
          replace
        />
      );
    }

    return (
      <ResultsInsights
        result={result}
        onBack={() =>
          navigate("/analysis")
        }
      />
    );
  };

  /* =========================================================
     LAYERS
     ========================================================= */

  const Layers = () => {
    if (!result) {
      return (
        <Navigate
          to="/aoi"
          replace
        />
      );
    }

    return (
      <LayersVisualization
        result={result}
        onBack={() =>
          navigate("/analysis")
        }
      />
    );
  };

  /* =========================================================
     ROUTES
     ========================================================= */

  return (
    <Routes>
      <Route
        path="/"
        element={
          <Landing />
        }
      />

      <Route
        path="/aoi"
        element={
          <AOI />
        }
      />

      <Route
        path="/analysis"
        element={
          <Analysis />
        }
      />

      <Route
        path="/layers"
        element={
          <Layers />
        }
      />

      <Route
        path="/results"
        element={
          <Results />
        }
      />

      <Route
        path="*"
        element={
          <Navigate
            to="/"
            replace
          />
        }
      />
    </Routes>
  );
}

/* =========================================================
   ROOT
   ========================================================= */

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App;