import { useState } from "react";
import "./HeroQuery.css";

interface HeroQueryProps {
  onSubmit: (query: string) => void;
  loading: boolean;
  error?: string | null;
}

function HeroQuery({
  onSubmit,
  loading,
  error,
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
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleSubmit();
          }
        }}
        placeholder="What do you want to investigate?"
        disabled={loading}
      />

      <div className="hero-query-bottom">

        <div
          className="hero-query-example"
          style={{ cursor: "pointer" }}
          onClick={() =>
            setValue(
              "Compare vegetation/NDVI change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]"
            )
          }
          title="Click to insert example query"
        >
          e.g. Compare vegetation/NDVI change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]
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

      {error && (
        <div className="hero-query-error">
          ⚠️ {error}
        </div>
      )}

    </div>
  );
}

export default HeroQuery;