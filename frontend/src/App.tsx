import { useState } from "react";

import LandingPage from "./pages/Landing/LandingPage";

import {
  submitQuery,
  type QueryResponse,
} from "./api/query";

function App() {
  const [result, setResult] =
    useState<QueryResponse | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);

  const handleQuery = async (query: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await submitQuery(query);

      setResult(response);
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

  return (
    <>
      <LandingPage
        onSubmit={handleQuery}
        loading={loading}
      />

      {error && (
        <div
          style={{
            position: "fixed",
            left: "50%",
            bottom: "24px",
            transform: "translateX(-50%)",
            padding: "12px 18px",
            border: "1px solid #11110f",
            background: "#f3f0e8",
            fontSize: "12px",
            zIndex: 100,
          }}
        >
          {error}
        </div>
      )}

      {/*
        `result` is intentionally kept here because the
        next step will transition the application from
        the landing page into the analysis workspace.
      */}
      {result && null}
    </>
  );
}

export default App;