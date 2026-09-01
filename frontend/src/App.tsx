import { useState } from "react";

import LandingPage from "./pages/Landing/LandingPage";
import AnalysisWorkspace from "./pages/Analysis/AnalysisWorkspace";
import ResultsInsights from "./pages/Results/ResultsInsights";
import LayersVisualization from "./pages/Layers/LayerVisualization";

import {
  submitQuery,
  type QueryResponse,
} from "./api/query";

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


function App() {

  const [currentPage, setCurrentPage] = useState<
    "landing" | "analysis" | "results" | "layers"
  >("landing");

  const [result, setResult] =
    useState<QueryResponse | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);


  /*
   * =========================================================
   * QUERY
   * =========================================================
   */

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

        setCurrentPage("analysis");

        return;
      }


      const response =
        await submitQuery(query);

      setResult(response);

      setCurrentPage("analysis");

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


  /*
   * =========================================================
   * NAVIGATION
   * =========================================================
   */

  const goToAnalysis = () => {
    setCurrentPage("analysis");
  };


  const goToResults = () => {
    setCurrentPage("results");
  };


  const goToLayers = () => {
    setCurrentPage("layers");
  };


  /*
   * =========================================================
   * LANDING
   * =========================================================
   */

  if (currentPage === "landing") {

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
  }


  /*
   * =========================================================
   * ANALYSIS
   * =========================================================
   */

  if (currentPage === "analysis" && result) {

    return (
      <AnalysisWorkspace
        result={result}
        onViewDetails={goToResults}
        onViewLayers={goToLayers}
      />
    );
  }


  /*
   * =========================================================
   * RESULTS
   * =========================================================
   */

  if (currentPage === "results" && result) {

    return (
      <ResultsInsights
        result={result}
        onBack={goToAnalysis}
      />
    );
  }


  /*
   * =========================================================
   * LAYERS
   * =========================================================
   */

  if (currentPage === "layers" && result) {

    return (
      <LayersVisualization
        result={result}
        onBack={goToAnalysis}
      />
    );
  }


  return null;
}


export default App;