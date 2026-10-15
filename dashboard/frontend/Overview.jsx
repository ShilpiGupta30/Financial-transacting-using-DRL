import useFetch from '../hooks/useFetch.js';
import { getSummary } from '../services/api.js';
import { Loader, ErrorBanner, EmptyState, StatCard, StatusBadge } from '../components/Common.jsx';

const ORDER = ['PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'ROLLED_BACK'];

export default function Overview({ health, status, refreshKey }) {
  const caps = status.data && status.data.capabilities;
  const db = status.data && status.data.database;
  const engine = status.data && status.data.cpp_engine;
  const canRead = !!(caps && caps.transactions_read);
  const summary = useFetch(getSummary, [refreshKey], canRead);

  const backendValue = health.data ? 'Online' : health.error ? 'Offline' : 'Checking…';
  const backendTone = health.data ? 'good' : health.error ? 'bad' : 'neutral';

  let banner = null;
  if (status.error) banner = <ErrorBanner error={status.error} onRetry={status.refetch} />;
  else if (status.loading && !status.data) banner = <Loader text="Checking system status…" />;
  else if (db && !db.connected)
    banner = (
      <div className="banner warn">
        <div>
          <strong>Database not connected</strong>
          <p>{db.error || 'Transaction data cannot be read right now.'}</p>
        </div>
      </div>
    );
  else if (db)
    banner = (
      <div className="banner good">
        <div>
          <strong>Backend and database are connected</strong>
          <p>Live transaction data is available from {db.name}.</p>
        </div>
      </div>
    );

  const total = summary.data ? summary.data.total : 0;

  return (
    <>
      {banner}
      <section className="grid stats">
        <StatCard
          label="Backend API"
          value={backendValue}
          tone={backendTone}
          note={health.data ? `v${health.data.version} · ${health.data.service}` : undefined}
        />
        <StatCard
          label="Database"
          value={db ? (db.connected ? 'Connected' : 'Unavailable') : '—'}
          tone={db ? (db.connected ? 'good' : 'bad') : 'neutral'}
          note={db ? db.name : undefined}
        />
        <StatCard
          label="Total transactions"
          value={summary.data ? summary.data.total : 'Unavailable'}
          tone={summary.data ? 'neutral' : 'muted'}
          note={summary.data ? undefined : canRead ? 'Loading…' : 'Needs a database connection'}
        />
        <StatCard
          label="Simulation engine"
          value={engine ? (engine.available ? 'Ready' : 'Not available') : '—'}
          tone={engine ? (engine.available ? 'good' : 'warn') : 'neutral'}
          note={engine && engine.executable_found ? 'Executable found' : undefined}
        />
      </section>

      <section className="card">
        <h2>Transactions by status</h2>
        {summary.error && <ErrorBanner error={summary.error} onRetry={summary.refetch} />}
        {summary.loading && !summary.data && <Loader />}
        {!canRead && !status.loading && (
          <EmptyState title="Transaction data is unavailable">
            {db && db.error ? db.error : 'The backend has not reported a working database connection.'}
          </EmptyState>
        )}
        {summary.data && total === 0 && (
          <EmptyState title="No transactions yet">The table is reachable but contains no rows.</EmptyState>
        )}
        {summary.data && total > 0 && (
          <>
            <div className="bar" role="img" aria-label="Share of transactions by status">
              {ORDER.map((s) => {
                const n = summary.data.by_status[s] || 0;
                return n ? <span key={s} className={'seg ' + s.toLowerCase()} style={{ flexGrow: n }} /> : null;
              })}
            </div>
            <ul className="breakdown">
              {ORDER.map((s) => {
                const n = summary.data.by_status[s] || 0;
                return (
                  <li key={s}>
                    <StatusBadge status={s} />
                    <strong>{n}</strong>
                    <span>{Math.round((n / total) * 100)}%</span>
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </section>
    </>
  );
}
