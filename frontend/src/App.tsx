import { useState } from "react";

import Header from "./components/Header";
import QueryInput from "./query/QueryInput";
import MapPanel from "./map/MapPanel";
import AnalysisPanel from "./results/AnalysisPanel";
import {
  submitQuery,
  type QueryResponse,
} from "./api/query";

function App() {
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleQuery = async (query: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await submitQuery(query);

      setResult(response);
    } catch (error) {
      console.error("Query failed:", error);

      setError(
        error instanceof Error
          ? error.message
          : "Something went wrong while analyzing the query."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <Header />

      <main>
        <QueryInput
          onSubmit={handleQuery}
          loading={loading}
        />

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <div className="workspace">
          <MapPanel />
          <AnalysisPanel result={result} />
        </div>
      </main>
    </div>
  );
}

export default App;