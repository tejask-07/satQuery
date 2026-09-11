import { useState } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useNavigate,
} from "react-router-dom";

import LandingPage from "./pages/Landing/LandingPage";
import AOISelection, { type AOIMetadata } from "./pages/AOI/AOISelection";
import AnalysisWorkspace from "./pages/Analysis/AnalysisWorkspace";
import AnalysisProcessing from "./pages/Processing/AnalysisProcessing";
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


  /* =======================================================
     CURRENT QUERY
     ======================================================= */

  const [currentQuery, setCurrentQuery] =
    useState<string>(() => {

      try {

        return (
          sessionStorage.getItem(
            "satquery_last_query"
          ) ||
          "compare vegetation change between 2021 and 2025"
        );

      } catch {

        return "compare vegetation change between 2021 and 2025";

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


  /* =======================================================
     CURRENT AOI & DATES
     ======================================================= */

  const [currentAOI, setCurrentAOI] =
    useState<AOIMetadata | null>(() => {
      try {
        const saved = sessionStorage.getItem("satquery_last_aoi");
        return saved ? (JSON.parse(saved) as AOIMetadata) : null;
      } catch {
        return null;
      }
    });

  const [startDate, setStartDate] =
    useState<string | undefined>(() => {
      try {
        return (
          sessionStorage.getItem("satquery_last_start_date") ||
          "2021-04-17"
        );
      } catch {
        return "2021-04-17";
      }
    });

  const [endDate, setEndDate] =
    useState<string | undefined>(() => {
      try {
        return (
          sessionStorage.getItem("satquery_last_end_date") ||
          "2025-04-17"
        );
      } catch {
        return "2025-04-17";
      }
    });


  /* =========================================================
     QUERY EXECUTION
     ========================================================= */

  const executeQuery = async (
    query: string,
    aoi?: unknown,
    startDate?: string,
    endDate?: string
  ): Promise<QueryResponse> => {

    /* =====================================================
       MOCK DATA
       ===================================================== */

    if (USE_MOCK_DATA) {

      await new Promise((resolve) =>
        setTimeout(resolve, 3500)
      );

      return {
        ...MOCK_RESULT,

        plan: {
          ...MOCK_RESULT.plan,

          task:
            query ||
            MOCK_RESULT.plan.task,
        },
      };

    }


    /* =====================================================
       REAL BACKEND
       ===================================================== */

    return submitQuery(
      query,
      aoi,
      startDate,
      endDate
    );

  };


  /* =========================================================
     NORMAL QUERY
     
     Used by Analysis → Re-query.
     
     This should NOT show the Processing page because
     the Processing page is specifically for a new
     investigation coming from the AOI page.
     ========================================================= */

  const handleQuery = async (
    query: string,
    aoi?: unknown,
    start?: string,
    end?: string
  ): Promise<void> => {

    const queryToRun =
      query.trim() ||
      currentQuery ||
      "compare vegetation change between 2021 and 2025";

    setCurrentQuery(queryToRun);
    if (start) setStartDate(start);
    if (end) setEndDate(end);
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      sessionStorage.setItem("satquery_last_query", queryToRun);
      sessionStorage.removeItem("satquery_last_result");
    } catch {
      // Ignore session storage errors.
    }

    // Immediately navigate to /analysis (shows Screenshot 2 while loading)
    navigate("/analysis");

    try {
      const response = await executeQuery(queryToRun, aoi, start || startDate, end || endDate);
      setResult(response);
      try {
        sessionStorage.setItem(
          "satquery_last_result",
          JSON.stringify(response)
        );
      } catch {
        // ignore
      }
      setLoading(false);
    } catch (err) {
      console.error("Query failed:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the query."
      );
      setLoading(false);
    }
  };


  /* =========================================================
     LANDING → AOI
     ========================================================= */

  const handleLandingSubmit = (
    query: string
  ) => {

    if (
      query &&
      query.trim()
    ) {

      setCurrentQuery(
        query.trim()
      );

    }


    setError(null);

    navigate("/aoi");

  };


  /* =========================================================
     CONVERT AOI → GEOJSON
     ========================================================= */

  const convertAOIToGeoJSON = (
    aoi: AOIMetadata
  ) => {


    /* =====================================================
       POLYGON / RECTANGLE
       ===================================================== */

    if (
      aoi.geometry.type === "polygon" ||
      aoi.geometry.type === "rectangle"
    ) {

      const ring =
        aoi.geometry.coordinates.map(
          ([lat, lng]) =>
            [lng, lat] as [number, number]
        );


      if (
        ring.length > 0
      ) {

        const first =
          ring[0];

        const last =
          ring[ring.length - 1];


        if (
          first[0] !== last[0] ||
          first[1] !== last[1]
        ) {

          ring.push([
            first[0],
            first[1],
          ]);

        }

      }


      return {
        type: "Polygon",
        coordinates: [
          ring,
        ],
      };

    }


    /* =====================================================
       CIRCLE
       ===================================================== */

    if (
      aoi.geometry.type === "circle"
    ) {

      const points:
        [number, number][] = [];


      const [
        cLat,
        cLng,
      ] =
        aoi.geometry.center;


      const radiusMeters =
        aoi.geometry.radius;


      const steps = 32;


      for (
        let i = 0;
        i < steps;
        i++
      ) {

        const angle =
          (i * 2 * Math.PI) /
          steps;


        const dLat =
          (
            radiusMeters *
            Math.cos(angle)
          ) /
          111320;


        const dLng =
          (
            radiusMeters *
            Math.sin(angle)
          ) /
          (
            111320 *
            Math.cos(
              (
                cLat *
                Math.PI
              ) /
              180
            )
          );


        points.push([
          cLng + dLng,
          cLat + dLat,
        ]);

      }


      points.push(
        points[0]
      );


      return {
        type: "Polygon",
        coordinates: [
          points,
        ],
      };

    }


    return null;

  };


  /* =========================================================
     RUN ANALYSIS FROM AOI
     
     THIS IS THE IMPORTANT PART.
     
     The Processing page is shown immediately.
     The backend request continues in the background.
     ========================================================= */

  const handleRunAnalysis = (
    query: string,
    aoi: AOIMetadata,
    start?: string,
    end?: string
  ) => {

    const queryToRun =
      query.trim() ||
      currentQuery ||
      "compare vegetation change between 2021 and 2025";

    const geojsonAoi =
      convertAOIToGeoJSON(aoi);

    // Save states
    setCurrentQuery(queryToRun);
    setCurrentAOI(aoi);
    if (start) setStartDate(start);
    if (end) setEndDate(end);
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      sessionStorage.setItem("satquery_last_query", queryToRun);
      sessionStorage.setItem("satquery_last_aoi", JSON.stringify(aoi));
      if (start) sessionStorage.setItem("satquery_last_start_date", start);
      if (end) sessionStorage.setItem("satquery_last_end_date", end);
      sessionStorage.removeItem("satquery_last_result");
    } catch {
      // Ignore session storage errors.
    }

    // IMMEDIATELY navigate to /analysis (shows Screenshot 2 while loading)
    navigate("/analysis");

    // Start backend request in background
    executeQuery(queryToRun, geojsonAoi, start, end)
      .then((response) => {
        setResult(response);
        try {
          sessionStorage.setItem(
            "satquery_last_result",
            JSON.stringify(response)
          );
        } catch {
          // ignore
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error("Analysis failed:", err);
        setError(
          err instanceof Error
            ? err.message
            : "Something went wrong while analyzing the investigation."
        );
        setLoading(false);
      });

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

          loading={
            loading
          }

          error={
            error
          }
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
      <>

        <AOISelection

          initialQuery={
            currentQuery
          }

          onRunAnalysis={
            handleRunAnalysis
          }

          loading={
            loading
          }

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
     PROCESSING
     ========================================================= */

  const Processing = () => {
    return (
      <Navigate
        to="/analysis"
        replace
      />
    );
  };


  /* =========================================================
     ANALYSIS
     ========================================================= */

  const Analysis = () => {

    // 1. If actively loading, ALWAYS render Screenshot 2 (AnalysisProcessing)
    if (loading) {
      const activeAoi =
        currentAOI ||
        (() => {
          try {
            const saved = sessionStorage.getItem("satquery_last_aoi");
            return saved ? (JSON.parse(saved) as AOIMetadata) : null;
          } catch {
            return null;
          }
        })();

      return (
        <AnalysisProcessing
          query={currentQuery}
          aoi={activeAoi}
          startDate={startDate}
          endDate={endDate}
        />
      );
    }

    // 2. If finished and result exists, render the Completed AnalysisWorkspace
    const activeResult =
      result ||
      (() => {
        try {
          const saved = sessionStorage.getItem("satquery_last_result");
          return saved ? (JSON.parse(saved) as QueryResponse) : null;
        } catch {
          return null;
        }
      })();

    if (activeResult) {
      return (
        <AnalysisWorkspace
          result={activeResult}
          currentQuery={currentQuery}
          onViewDetails={() =>
            navigate(
              "/results"
            )
          }
          onViewLayers={() =>
            navigate(
              "/layers"
            )
          }
          onRequery={
            handleQuery
          }
          loading={false}
        />
      );
    }

    // 3. If there is an error from the backend request:
    if (error) {
      return (
        <main
          className="analysis-error-page"
          style={{
            padding: "64px 24px",
            textAlign: "center",
            minHeight: "60vh",
            fontFamily: "'Lexend Deca', 'Lexend', system-ui, sans-serif",
          }}
        >
          <h2
            style={{
              fontSize: "20px",
              fontWeight: 700,
              letterSpacing: "0.05em",
              color: "#11110f",
              marginBottom: "16px",
            }}
          >
            ANALYSIS FAILED
          </h2>
          <p
            style={{
              color: "#d94a2f",
              fontSize: "14px",
              maxWidth: "600px",
              margin: "0 auto 24px",
            }}
          >
            {error}
          </p>
          <button
            type="button"
            className="view-details-button"
            onClick={() =>
              navigate("/aoi")
            }
            style={{
              margin: "0 auto",
              display: "inline-flex",
              cursor: "pointer",
            }}
          >
            <span>RETURN TO AOI</span>
            <span>→</span>
          </button>
        </main>
      );
    }

    // 4. If loading is false and result does not exist, redirect to /aoi
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

    const activeResult =
      result ||
      (() => {
        try {
          const saved = sessionStorage.getItem("satquery_last_result");
          return saved ? (JSON.parse(saved) as QueryResponse) : null;
        } catch {
          return null;
        }
      })();

    if (
      !activeResult
    ) {

      return (
        <Navigate
          to="/aoi"
          replace
        />
      );

    }


    return (

      <ResultsInsights

        result={
          activeResult
        }

        onBack={() =>
          navigate(
            "/analysis"
          )
        }

        onViewLayers={() =>
          navigate(
            "/layers"
          )
        }

        onNewAnalysis={() => {
          setResult(null);
          setLoading(false);
          setError(null);
          setCurrentAOI(null);

          try {
            sessionStorage.removeItem("satquery_last_result");
            sessionStorage.removeItem("satquery_last_query");
            sessionStorage.removeItem("satquery_last_aoi");
          } catch {
            // ignore
          }

          navigate("/aoi");
        }}

      />

    );

  };


  /* =========================================================
     LAYERS
     ========================================================= */

  const Layers = () => {

    const activeResult =
      result ||
      (() => {
        try {
          const saved = sessionStorage.getItem("satquery_last_result");
          return saved ? (JSON.parse(saved) as QueryResponse) : null;
        } catch {
          return null;
        }
      })();

    if (
      !activeResult
    ) {

      return (
        <Navigate
          to="/aoi"
          replace
        />
      );

    }


    return (

      <LayersVisualization

        result={
          activeResult
        }

        onBack={() =>
          navigate(
            "/analysis"
          )
        }

        onViewResults={() =>
          navigate(
            "/results"
          )
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


      {/* ===================================================
          NEW PROCESSING PAGE
          =================================================== */}

      <Route
        path="/processing"
        element={
          <Processing />
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