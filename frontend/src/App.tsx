import { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from "react-router-dom";

import LandingPage from "./pages/Landing/LandingPage";
import AOISelection, { type AOIMetadata } from "./pages/AOI/AOISelection";
import AnalysisWorkspace from "./pages/Analysis/AnalysisWorkspace";
import ResultsInsights from "./pages/Results/ResultsInsights";
import LayersVisualization from "./pages/Layers/LayerVisualization";

import {
  submitQuery,
  type QueryResponse,
} from "./api/query";

import "./index.css";


const USE_MOCK_DATA = false;


const MOCK_RESULT: QueryResponse = {
  status: "analysis complete",

  answer:
    "Vegetation decreased across several regions of the Mumbai Urban Region between 2021 and 2025.",

  confidence: 0.91,

  plan: {
    task: "Show where vegetation decreased between 2021 and 2025.",

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
   APP STATE
   ========================================================= */

function AppContent() {

  const navigate = useNavigate();

  const [currentQuery, setCurrentQuery] = useState<string>(() => {
    try {
      return (
        sessionStorage.getItem("satquery_last_query") ||
        "compare vegetation change between 2021 and 2025"
      );
    } catch {
      return "compare vegetation change between 2021 and 2025";
    }
  });

  const [result, setResult] = useState<QueryResponse | null>(() => {
    try {
      const saved = sessionStorage.getItem("satquery_last_result");
      return saved ? (JSON.parse(saved) as QueryResponse) : null;
    } catch {
      return null;
    }
  });

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);


  /* =======================================================
     QUERY
     ======================================================= */

  const handleQuery = async (
    query: string,
    aoi?: unknown,
    startDate?: string,
    endDate?: string
  ) => {
    const queryToRun = query || currentQuery || "compare vegetation change between 2021 and 2025";
    setCurrentQuery(queryToRun);
    setLoading(true);
    setError(null);

    try {
      if (USE_MOCK_DATA) {
        await new Promise((resolve) =>
          setTimeout(resolve, 700)
        );

        const mockResult: QueryResponse = {
          ...MOCK_RESULT,
          plan: {
            ...MOCK_RESULT.plan,
            task:
              queryToRun ||
              MOCK_RESULT.plan.task,
          },
        };

        setResult(mockResult);
        try {
          sessionStorage.setItem("satquery_last_result", JSON.stringify(mockResult));
          sessionStorage.setItem("satquery_last_query", queryToRun);
        } catch {
          // ignore
        }

        navigate("/analysis");
        return;
      }

      /* ===================================================
         REAL BACKEND
         =================================================== */

      const response = await submitQuery(queryToRun, aoi, startDate, endDate);

      setResult(response);
      try {
        sessionStorage.setItem("satquery_last_result", JSON.stringify(response));
        sessionStorage.setItem("satquery_last_query", queryToRun);
      } catch {
        // ignore
      }

      navigate("/analysis");

    } catch (err) {

      console.error("Query failed:", err);

      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the query."
      );

    } finally {

      setLoading(false);

    }
  };


  /* =======================================================
     NAVIGATION & ACTIONS
     ======================================================= */

  const handleLandingSubmit = (query: string) => {
    if (query && query.trim()) {
      setCurrentQuery(query.trim());
    }
    setError(null);
    navigate("/aoi");
  };

  const handleRunAnalysis = async (
    query: string,
    aoi: AOIMetadata,
    startDate?: string,
    endDate?: string
  ) => {
    const queryToRun =
      query.trim() ||
      currentQuery ||
      "compare vegetation change between 2021 and 2025";
    setCurrentQuery(queryToRun);

    let geojsonAoi: any = null;
    if (aoi.geometry.type === "polygon" || aoi.geometry.type === "rectangle") {
      const ring = aoi.geometry.coordinates.map(([lat, lng]) => [lng, lat]);
      if (ring.length > 0) {
        const first = ring[0];
        const last = ring[ring.length - 1];
        if (first[0] !== last[0] || first[1] !== last[1]) {
          ring.push([first[0], first[1]]);
        }
      }
      geojsonAoi = {
        type: "Polygon",
        coordinates: [ring],
      };
    } else if (aoi.geometry.type === "circle") {
      const points: [number, number][] = [];
      const [cLat, cLng] = aoi.geometry.center;
      const radiusMeters = aoi.geometry.radius;
      const steps = 32;
      for (let i = 0; i < steps; i++) {
        const angle = (i * 2 * Math.PI) / steps;
        const dLat = (radiusMeters * Math.cos(angle)) / 111320;
        const dLng =
          (radiusMeters * Math.sin(angle)) /
          (111320 * Math.cos((cLat * Math.PI) / 180));
        points.push([cLng + dLng, cLat + dLat]);
      }
      points.push(points[0]);
      geojsonAoi = {
        type: "Polygon",
        coordinates: [points],
      };
    }

    await handleQuery(queryToRun, geojsonAoi, startDate, endDate);
  };


  /* =======================================================
     LANDING
     ======================================================= */

  const Landing = () => {

    return (
      <>
        <LandingPage
          onSubmit={handleLandingSubmit}
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


  /* =======================================================
     AOI
     ======================================================= */

  const AOI = () => {

    return (
      <>
        <AOISelection
          initialQuery={currentQuery}
          onRunAnalysis={handleRunAnalysis}
          loading={loading}
        />

        {error && (
          <div className="query-error">
            {error}
          </div>
        )}
      </>
    );

  };


  /* =======================================================
     ANALYSIS
     ======================================================= */

  const Analysis = () => {

    if (!result) {
      return <Navigate to="/aoi" replace />;
    }

    return (
      <AnalysisWorkspace
        result={result}
        currentQuery={currentQuery}
        onViewDetails={() => navigate("/results")}
        onViewLayers={() => navigate("/layers")}
        onRequery={handleQuery}
        loading={loading}
      />
    );

  };


  /* =======================================================
     RESULTS
     ======================================================= */

  const Results = () => {

    if (!result) {
      return <Navigate to="/aoi" replace />;
    }

    return (
      <ResultsInsights
        result={result}
        onBack={() => navigate("/analysis")}
        onViewLayers={() => navigate("/layers")}
        onNewAnalysis={() => {
          setResult(null);
          try {
            sessionStorage.removeItem("satquery_last_result");
            sessionStorage.removeItem("satquery_last_query");
          } catch {
            // ignore
          }
          navigate("/aoi");
        }}
      />
    );

  };


  /* =======================================================
     LAYERS
     ======================================================= */

  const Layers = () => {

    if (!result) {
      return <Navigate to="/aoi" replace />;
    }

    return (
      <LayersVisualization
        result={result}
        onBack={() => navigate("/analysis")}
        onViewResults={() => navigate("/results")}
      />
    );
  };


  /* =======================================================
     ROUTES
     ======================================================= */

  return (
    <Routes>

      <Route
        path="/"
        element={<Landing />}
      />

      <Route
        path="/aoi"
        element={<AOI />}
      />

      <Route
        path="/analysis"
        element={<Analysis />}
      />

      <Route
        path="/layers"
        element={<Layers />}
      />

      <Route
        path="/results"
        element={<Results />}
      />

      {/* Unknown URL → landing */}
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