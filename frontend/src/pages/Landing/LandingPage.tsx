import landingSatellite from "../../assets/landing-satellite.jpg";
import "./LandingPage.css";

interface LandingPageProps {
  onSubmit: (query: string) => void;
  loading?: boolean;
  error?: string | null;
}

export default function LandingPage({ onSubmit, loading = false }: LandingPageProps) {
  return (
    <main className="landing-page">
      <header className="landing-nav">
        <div className="landing-brand">
          <span className="landing-brand-mark" aria-hidden="true" />
          <div>
            <div className="landing-brand-name">SATQUERY AI</div>
            <div className="landing-brand-subtitle">REMOTE SENSING INTELLIGENCE</div>
          </div>
        </div>

        <nav className="landing-nav-links" aria-label="Primary">
          <a href="#product"><span>[01]</span> PRODUCT</a>
          <a href="#about"><span>[02]</span> ABOUT</a>
          <a href="#technology"><span>[03]</span> TECH</a>
          <a href="#contact"><span>[04]</span> CONTACT</a>
        </nav>

        <button
          type="button"
          className="landing-start"
          onClick={() => onSubmit("")}
          disabled={loading}
        >
          {loading ? "ANALYZING..." : <>GET STARTED <span>→</span></>}
        </button>
      </header>

      <section className="landing-hero" id="product">
        <img
          className="landing-hero-image"
          src={landingSatellite}
          alt="Satellite view of an urban coastal region"
        />
        <div className="landing-hero-overlay" />

        <div className="landing-hero-copy">
          <div className="landing-eyebrow">SATELLITE INTELLIGENCE / 01</div>
          <h1>
            FROM SATELLITE DATA
            <br />
            TO <em>REAL INSIGHTS.</em>
          </h1>
          <p>
            Ask natural-language questions about Earth observation data.
            SatQuery AI turns complex remote-sensing imagery into clear,
            evidence-grounded analysis.
          </p>
          <button
            type="button"
            className="landing-explore"
            onClick={() => onSubmit("")}
            disabled={loading}
          >
            EXPLORE THE PLATFORM <span>→</span>
          </button>
        </div>

        <div className="landing-rail" aria-hidden="true">
          <span className="landing-rail-number">01</span>
          <span>OBSERVE</span>
          <span>ANALYSE</span>
          <span>UNDERSTAND</span>
          <span>ACT</span>
        </div>

        <div className="landing-coordinates">19.0760° N&nbsp;&nbsp;72.8777° E</div>
      </section>

      <section className="landing-features" id="about">
        <article>
          <span>01 / OBSERVE</span>
          <h2>See the Earth in context.</h2>
          <p>Define an area, select imagery and establish the observation window.</p>
        </article>
        <article>
          <span>02 / ANALYSE</span>
          <h2>Ask in plain language.</h2>
          <p>Use natural-language queries across temporal, spectral and multimodal workflows.</p>
        </article>
        <article>
          <span>03 / UNDERSTAND</span>
          <h2>Follow the evidence.</h2>
          <p>Move from imagery to interpretable findings, indicators and supporting evidence.</p>
        </article>
      </section>

      <section className="landing-globe" id="technology">
        <div className="landing-globe-copy">
          <span>GLOBAL PERSPECTIVE / LOCAL IMPACT</span>
          <h2>One interface for a changing planet.</h2>
          <p>
            Satellite imagery, remote-sensing indices and vision-language reasoning
            come together in one investigation workflow.
          </p>
        </div>
        <div className="landing-globe-orbit" aria-hidden="true">
          <div className="landing-orbit orbit-a" />
          <div className="landing-orbit orbit-b" />
          <div className="landing-globe-core">EARTH<br /><small>OBSERVATION</small></div>
        </div>
      </section>

      <footer className="landing-footer" id="contact">
        <div>SATQUERY AI</div>
        <div>REMOTE SENSING INTELLIGENCE</div>
        <div>SIH 2026 / ISRO</div>
      </footer>
    </main>
  );
}