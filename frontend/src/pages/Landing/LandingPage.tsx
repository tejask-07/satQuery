import HeroQuery from "../../components/HeroQuery/HeroQuery";
import earthImage from "../../assets/earth.png";
import "./LandingPage.css";

interface LandingPageProps {
  onSubmit: (query: string) => void;
  loading: boolean;
}

function LandingPage({
  onSubmit,
  loading,
}: LandingPageProps) {
  return (
    <main className="landing-page">

      {/* HEADER */}

      <header className="landing-header">

        <div className="landing-brand">
          <div className="landing-brand-name">
            SATQUERY AI
          </div>

          <div className="landing-brand-subtitle">
            REMOTE SENSING INTELLIGENCE
          </div>
        </div>

        <nav className="landing-nav">
          <button type="button">
            ABOUT
          </button>

          <button type="button">
            HOW IT WORKS
          </button>

          <button type="button">
            DATA SOURCES
          </button>
        </nav>

      </header>


      {/* MAIN */}

      <section className="landing-main">

        {/* =================================================
            LEFT — HERO
            ================================================= */}

        <section className="hero-panel">

          <div className="hero-content">

            <div className="hero-kicker">
              REMOTE SENSING INTELLIGENCE
            </div>

            <h1 className="hero-title">
              <span>ASK THE</span>
              <span>EARTH.</span>
            </h1>

            <div className="hero-copy">

              <h2>
                Natural language → spatial evidence.
              </h2>

              <p>
                Ask anything about satellite imagery
                <br />
                and discover what changed.
              </p>

            </div>

            <HeroQuery
              onSubmit={onSubmit}
              loading={loading}
            />

          </div>

        </section>


        {/* =================================================
            CENTER — EARTH
            ================================================= */}

        <section className="earth-panel">

          <div className="earth-grid" />

          <div className="earth-image-wrap">
            <img
              src={earthImage}
              alt=""
              className="earth-image"
            />
          </div>

        </section>


        {/* =================================================
            RIGHT — CAPABILITIES
            ================================================= */}

        <aside className="capabilities-panel">

          <section className="capability-group">

            <h2 className="capability-title">
              ANALYSIS TYPES
            </h2>

            <div className="capability-list">

              <div className="capability-item">
                CHANGE DETECTION
              </div>

              <div className="capability-item">
                INDEX ANALYSIS
              </div>

              <div className="capability-item">
                OBJECT DETECTION
              </div>

              <div className="capability-item">
                COMPARISON
              </div>

            </div>

          </section>


          <section className="capability-group">

            <h2 className="capability-title">
              AI CAPABILITIES
            </h2>

            <div className="capability-list">

              <div className="capability-item">
                VISION-LANGUAGE
              </div>

              <div className="capability-item">
                MULTI-TEMPORAL
              </div>

              <div className="capability-item">
                ANALYSIS
              </div>

            </div>

          </section>

        </aside>

      </section>

    </main>
  );
}

export default LandingPage;