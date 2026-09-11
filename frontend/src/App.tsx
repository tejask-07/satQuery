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
   PROCESSING STATE
   ========================================================= */

interface ProcessingState {
  query: string;
  aoi: AOIMetadata;
  startDate?: string;
  endDate?: string;
}


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
     PROCESSING INVESTIGATION
     ======================================================= */

  const [processingState, setProcessingState] =
    useState<ProcessingState | null>(null);


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
    startDate?: string,
    endDate?: string
  ) => {

    const queryToRun =
      query ||
      currentQuery ||
      "compare vegetation change between 2021 and 2025";


    setCurrentQuery(queryToRun);

    setLoading(true);

    setError(null);


    try {

      const response =
        await executeQuery(
          queryToRun,
          aoi,
          startDate,
          endDate
        );


      setResult(response);


      try {

        sessionStorage.setItem(
          "satquery_last_result",
          JSON.stringify(response)
        );

        sessionStorage.setItem(
          "satquery_last_query",
          queryToRun
        );

      } catch {

        // Ignore session storage errors.

      }


      navigate("/analysis" , {
        state:{
          result: response,
        }
      });


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


    /* =====================================================
       CONVERT AOI
       ===================================================== */

    const geojsonAoi =
      convertAOIToGeoJSON(aoi);


    /* =====================================================
       SAVE CURRENT INVESTIGATION
       ===================================================== */

    setCurrentQuery(
      queryToRun
    );

    setError(null);

    setResult(null);


    setProcessingState({

      query:
        queryToRun,

      aoi,

      startDate,

      endDate,

    });


    /* =====================================================
       START LOADING
       ===================================================== */

    setLoading(true);


    /* =====================================================
       GO TO PROCESSING PAGE IMMEDIATELY
       ===================================================== */

    navigate("/processing");


    /* =====================================================
       RUN REAL BACKEND REQUEST
       ===================================================== */

    try {

      const response =
        await executeQuery(
          queryToRun,
          geojsonAoi,
          startDate,
          endDate
        );


      /* ===================================================
         SAVE RESULT
         =================================================== */

      setResult(
        response
      );


      try {

        sessionStorage.setItem(
          "satquery_last_result",
          JSON.stringify(response)
        );

        sessionStorage.setItem(
          "satquery_last_query",
          queryToRun
        );

      } catch {

        // Ignore session storage errors.

      }


      /* ===================================================
         FINISH PROCESSING
         =================================================== */

      setProcessingState(
        null
      );


      setLoading(
        false
      );


      /* ===================================================
         SHOW ANALYSIS
         =================================================== */

      navigate(
        "/analysis"
      );


    } catch (err) {

      console.error(
        "Analysis failed:",
        err
      );


      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the investigation."
      );


      setProcessingState(
        null
      );


      setLoading(
        false
      );


      /*
       * If the backend fails, return to AOI so the user
       * can see the error and try again.
       */

      navigate(
        "/aoi"
      );

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

    /*
     * If somebody manually visits /processing without
     * starting an investigation, send them back to AOI.
     */

    if (
      !processingState
    ) {

      return (
        <Navigate
          to="/aoi"
          replace
        />
      );

    }


    return (

      <AnalysisProcessing

        query={
          processingState.query
        }

        aoi={
          processingState.aoi
        }

        startDate={
          processingState.startDate
        }

        endDate={
          processingState.endDate
        }

      />

    );

  };


  /* =========================================================
     ANALYSIS
     ========================================================= */

  const Analysis = () => {

    if (
      !result
    ) {

      return (
        <Navigate
          to="/aoi"
          replace
        />
      );

    }


    return (

      <AnalysisWorkspace

        result={
          result
        }

        currentQuery={
          currentQuery
        }

        onViewDetails={() =>
          navigate(
            "/results"
          )
        }

        onRequery={
          handleQuery
        }

        loading={
          loading
        }

      />

    );

  };


  /* =========================================================
     RESULTS
     ========================================================= */

  const Results = () => {

    if (
      !result
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
          result
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

          setResult(
            null
          );

          setProcessingState(
            null
          );


          try {

            sessionStorage.removeItem(
              "satquery_last_result"
            );

            sessionStorage.removeItem(
              "satquery_last_query"
            );

          } catch {

            // ignore

          }


          navigate(
            "/aoi"
          );

        }}

      />

    );

  };


  /* =========================================================
     LAYERS
     ========================================================= */

  const Layers = () => {

    if (
      !result
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
          result
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