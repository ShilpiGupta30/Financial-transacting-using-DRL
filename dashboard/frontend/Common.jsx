export function Loader({ text = 'Loading…' }) {
  return (
    <div className="loader">
      <span className="spinner" />
      {text}
    </div>
  );
}

export function ErrorBanner({ error, onRetry }) {
  if (!error) return null;
  const offline = error.code === 'BACKEND_OFFLINE';
  return (
    <div className="banner bad" role="alert">
      <div>
        <strong>{offline ? 'Backend offline' : error.code || 'Error'}</strong>
        <p>{error.message}</p>
      </div>
      {onRetry && (
        <button className="btn" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
    </div>
  );
}

export function StatCard({ label, value, note, tone = 'neutral' }) {
  return (
    <div className="card stat">
      <span className="label">{label}</span>
      <span className={'value ' + tone}>{value}</span>
      {note && <span className="note">{note}</span>}
    </div>
  );
}

export function StatusBadge({ status }) {
  return <span className={'badge ' + String(status).toLowerCase()}>{String(status).replace('_', ' ')}</span>;
}

export function Tag({ tone = 'neutral', children }) {
  return <span className={'tag ' + tone}>{children}</span>;
}
