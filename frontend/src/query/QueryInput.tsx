import { useState } from "react";

interface QueryInputProps {
  onSubmit: (query: string) => void;
  loading: boolean;
}

function QueryInput({ onSubmit, loading }: QueryInputProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = () => {
    const trimmedQuery = query.trim();

    if (!trimmedQuery || loading) {
      return;
    }

    onSubmit(trimmedQuery);
  };

  return (
    <section className="query-section">
      <div className="section-label">ANALYSIS QUERY</div>

      <textarea
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Ask SatQuery something about your satellite imagery..."
        rows={4}
      />

      <div className="query-footer">
        <span>Natural-language remote sensing analysis</span>

        <button onClick={handleSubmit} disabled={loading || !query.trim()}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>
    </section>
  );
}

export default QueryInput;