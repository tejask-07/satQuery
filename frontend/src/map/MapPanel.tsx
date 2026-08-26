function MapPanel() {
  return (
    <section className="map-panel">
      <div className="map-placeholder">
        <div className="map-grid" />

        <div className="map-message">
          <span>◈</span>
          <h2>Satellite Analysis Map</h2>
          <p>
            Interactive imagery, analysis layers, and detected regions will
            appear here.
          </p>
        </div>
      </div>

      <div className="map-toolbar">
        <button>Satellite</button>
        <button>Analysis</button>
        <button>Comparison</button>
      </div>
    </section>
  );
}

export default MapPanel;