export default function Header({ title, subtitle, updated, loading, onRefresh, onMenu }) {
  return (
    <header className="header">
      <button className="btn menu-btn" onClick={onMenu} aria-label="Open navigation">
        Menu
      </button>
      <div className="titles">
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <div className="header-actions">
        <span className="updated">{updated ? 'Updated ' + updated.toLocaleTimeString() : 'Not loaded yet'}</span>
        <button className="btn primary" onClick={onRefresh} disabled={loading}>
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
    </header>
  );
}
