import { useState } from "react";

import Header from "./components/Header";
import QueryInput from "./query/QueryInput";
import MapPanel from "./map/MapPanel";
import AnalysisPanel from "./results/AnalysisPanel";

function App() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleQuery = async (query: string) => {
    console.log("Query:", query);

    // API integration comes next.
    setLoading(true);

    setTimeout(() => {
      setLoading(false);
    }, 500);
  };

  return (
    <div className="app">
      <Header />

      <main>
        <QueryInput onSubmit={handleQuery} loading={loading} />

        <div className="workspace">
          <MapPanel />
          <AnalysisPanel result={result} />
        </div>
      </main>
    </div>
  );
}

export default App;