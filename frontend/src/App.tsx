import { useState } from "react";

import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useNavigate,
} from "react-router-dom";


import LandingPage from "./pages/Landing/LandingPage";

import AOISelection from "./pages/AOI/AOISelection";

import AnalysisWorkspace from "./pages/Analysis/AnalysisWorkspace";

import ResultsInsights from "./pages/Results/ResultsInsights";

import LayersVisualization from "./pages/Layers/LayerVisualization";


import {
  submitQuery,
  type QueryResponse,
} from "./api/query";


import "./index.css";



/* =========================================================
   CONFIGURATION
   ========================================================= */

const USE_MOCK_DATA = true;



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

    target:
      "Mumbai Urban Region",

    time_start:
      "2021-04-17",

    time_end:
      "2025-04-10",

    modalities:
      ["Sentinel-2 (L2A)"],

    metric:
      "NDVI",

    direction:
      "decrease",

    analysis:
      ["Change detection"],

    output:
      [
        "NDVI Difference",
        "NDBI Increase",
      ],

  },


  statistics: {

    area_affected:
      "18.4 km²",

    average_ndvi_change:
      "-23.7%",

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

  const navigate =
    useNavigate();



  /* =======================================================
     RESULT
     ======================================================= */

  const [
    result,
    setResult,
  ] = useState<QueryResponse | null>(
    null
  );



  /* =======================================================
     LOADING
     ======================================================= */

  const [
    loading,
    setLoading,
  ] = useState(false);



  /* =======================================================
     ERROR
     ======================================================= */

  const [
    error,
    setError,
  ] = useState<string | null>(
    null
  );




  /* =======================================================
     DRAFT QUERY
     =======================================================

     Temporary query storage.

     LandingPage currently accepts the query, then sends
     the user to /aoi.

     The actual analysis query will eventually live entirely
     inside the AOI page.
     */

  const [
    draftQuery,
    setDraftQuery,
  ] = useState("");



  /* =======================================================
     LANDING → AOI
     ======================================================= */

  const handleLandingSubmit = (
    query: string
  ) => {

    setDraftQuery(query);

    setError(null);

    navigate("/aoi");

  };




  /* =======================================================
     AOI → ANALYSIS
     ======================================================= */

  const handleRunAnalysis = async (

    query: string,

    aoi: {
      name: string;
      center: [number, number];
      area: string;
      perimeter: string;
    },

    startDate: string,

    endDate: string

  ) => {

    setLoading(true);

    setError(null);


    try {

      /* ===================================================
         MOCK MODE
         =================================================== */

      if (USE_MOCK_DATA) {

        await new Promise(
          (resolve) =>
            setTimeout(
              resolve,
              700
            )
        );


        const mockResult:
          QueryResponse = {

          ...MOCK_RESULT,


          plan: {

            ...MOCK_RESULT.plan,


            task:
              query.trim()
                ? query
                : MOCK_RESULT.plan.task,


            target:
              aoi.name,


            time_start:
              startDate,


            time_end:
              endDate,

          },

        };


        setResult(
          mockResult
        );


        navigate(
          "/analysis"
        );


        return;

      }



      /* ===================================================
         REAL BACKEND
         ===================================================

         The backend currently accepts only:

         {
           query: string
         }

         AOI geometry + dates will be added when the backend
         endpoint supports them.
         */

      const response =
        await submitQuery(
          query
        );


      setResult(
        response
      );


      navigate(
        "/analysis"
      );

    }


    catch (err) {

      console.error(
        "Analysis failed:",
        err
      );


      setError(

        err instanceof Error

          ? err.message

          : "Something went wrong while analyzing the query."

      );

    }


    finally {

      setLoading(false);

    }

  };



  /* =======================================================
     LANDING PAGE
     ======================================================= */

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
     AOI PAGE
     ======================================================= */

  const AOI = () => {

    return (

      <AOISelection

        initialQuery={
          draftQuery
        }


        onRunAnalysis={
          handleRunAnalysis
        }

      />

    );

  };



  /* =======================================================
     ANALYSIS PAGE
     ======================================================= */

  const Analysis = () => {

    if (!result) {

      return (

        <Navigate
          to="/"
          replace
        />

      );

    }


    return (

      <AnalysisWorkspace

        result={
          result
        }


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

      />

    );

  };



  /* =======================================================
     RESULTS PAGE
     ======================================================= */

  const Results = () => {

    if (!result) {

      return (

        <Navigate
          to="/"
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

      />

    );

  };



  /* =======================================================
     LAYERS PAGE
     ======================================================= */

  const Layers = () => {

    if (!result) {

      return (

        <Navigate
          to="/"
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

      />

    );

  };



  /* =======================================================
     ROUTES
     ======================================================= */

  return (

    <Routes>


      {/* ===================================================
          LANDING
          =================================================== */}

      <Route

        path="/"

        element={
          <Landing />
        }

      />



      {/* ===================================================
          AOI SELECTION
          =================================================== */}

      <Route

        path="/aoi"

        element={
          <AOI />
        }

      />



      {/* ===================================================
          ANALYSIS
          =================================================== */}

      <Route

        path="/analysis"

        element={
          <Analysis />
        }

      />



      {/* ===================================================
          LAYERS
          =================================================== */}

      <Route

        path="/layers"

        element={
          <Layers />
        }

      />



      {/* ===================================================
          RESULTS
          =================================================== */}

      <Route

        path="/results"

        element={
          <Results />
        }

      />



      {/* ===================================================
          UNKNOWN ROUTE
          =================================================== */}

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
   ROOT APP
   ========================================================= */

function App() {

  return (

    <BrowserRouter>

      <AppContent />

    </BrowserRouter>

  );

}



export default App;