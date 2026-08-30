import "./Header.css";

function Header() {
  return (
    <header className="site-header">
      <div className="brand">
        <div className="brand-name">SATQUERY AI</div>
        <div className="brand-subtitle">
          REMOTE SENSING INTELLIGENCE
        </div>
      </div>

      <nav className="site-nav">
        <button type="button">ABOUT</button>
        <button type="button">HOW IT WORKS</button>
        <button type="button">DATA SOURCES</button>
      </nav>

      <div className="system-status">
        <span className="status-indicator" />
        <span className="status-label">STATUS</span>
        <span>SYSTEM READY</span>
      </div>
    </header>
  );
}

export default Header;