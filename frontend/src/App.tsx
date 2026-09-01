import { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from "react-router-dom";

import LandingPage from "./pages/Landing/LandingPage";
import AnalysisWorkspace from "./pages/Analysis/AnalysisWorkspace";
import ResultsInsights from "./pages/Results/ResultsInsights";
import LayersVisualization from "./pages/Layers/LayerVisualization";

import {
  submitQuery,
  type QueryResponse,
} from "./api/query";

import "./index.css";


const USE_MOCK_DATA = true;


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

  const [result, setResult] =
    useState<QueryResponse | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);


  /* =======================================================
     QUERY
     ======================================================= */

  const handleQuery = async (query: string) => {

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
              query ||
              MOCK_RESULT.plan.task,
          },
        };

        setResult(mockResult);

        navigate("/analysis");

        return;
      }


      /* ===================================================
         REAL BACKEND
         =================================================== */

      const response =
        await submitQuery(query);

      setResult(response);

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
     LANDING
     ======================================================= */

  const Landing = () => {

    return (
      <>
        <LandingPage
          onSubmit={handleQuery}
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
      return <Navigate to="/" replace />;
    }

    return (
      <AnalysisWorkspace
        result={result}
        onViewDetails={() => navigate("/results")}
        onViewLayers={() => navigate("/layers")}
      />
    );

  };


  /* =======================================================
     RESULTS
     ======================================================= */

  const Results = () => {

    if (!result) {
      return <Navigate to="/" replace />;
    }

    return (
      <ResultsInsights
        result={result}
        onBack={() => navigate("/analysis")}
      />
    );

  };


  /* =======================================================
     LAYERS
     ======================================================= */

  const Layers = () => {

    if (!result) {
      return <Navigate to="/" replace />;
    }

    return (
      <LayersVisualization
        result={result}
        onBack={() => navigate("/analysis")}
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