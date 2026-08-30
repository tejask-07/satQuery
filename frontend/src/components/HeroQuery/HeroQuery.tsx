import { useState } from "react";
import "./HeroQuery.css";

interface HeroQueryProps {
  onSubmit: (query: string) => void;
  loading: boolean;
}

function HeroQuery({
  onSubmit,
  loading,
}: HeroQueryProps) {
  const [value, setValue] = useState("");

  const handleSubmit = () => {
    const query = value.trim();

    if (!query || loading) {
      return;
    }

    onSubmit(query);
  };

  return (
    <div className="hero-query">

      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder="What do you want to investigate?"
        disabled={loading}
      />

      <div className="hero-query-bottom">

        <div className="hero-query-example">
          e.g. Show where vegetation decreased
          <br />
          between 2021 and 2025.
        </div>

        <button
          type="button"
          className="hero-query-button"
          onClick={handleSubmit}
          disabled={loading || !value.trim()}
        >
          {loading ? "ANALYZING..." : "ANALYZE ↗"}
        </button>

      </div>

    </div>
  );
}

export default HeroQuery;