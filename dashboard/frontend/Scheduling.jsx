import useFetch from '../hooks/useFetch.js';
import { getSchedulers } from '../services/api.js';
import { Loader, ErrorBanner, EmptyState, Tag } from '../components/Common.jsx';

export default function Scheduling({ refreshKey }) {
  const { data, loading, error, refetch } = useFetch(getSchedulers, [refreshKey]);
  const items = (data && data.items) || [];

  return (
    <>
      {error && <ErrorBanner error={error} onRetry={refetch} />}
      {loading && !data && <Loader text="Loading algorithms…" />}
      {data && items.length === 0 && (
        <div className="card">
          <EmptyState title="No algorithms reported">The backend returned an empty scheduler list.</EmptyState>
        </div>
      )}
      {items.length > 0 && (
        <>
          <section className="grid cards">
            {items.map((s) => (
              <article key={s.id} className={'card algo' + (s.available ? '' : ' off')}>
                <div className="algo-head">
                  <h2>{s.name}</h2>
                  <Tag tone={s.available ? 'good' : 'muted'}>{s.available ? 'Available' : 'Unavailable'}</Tag>
                </div>
                <p>{s.description}</p>
                <div className="tags">
                  <Tag>{s.type}</Tag>
                  <Tag>{s.supports_time_quantum ? 'Uses time quantum' : 'No time quantum'}</Tag>
                </div>
                {s.reason && <p className="reason">{s.reason}</p>}
              </article>
            ))}
          </section>

          <section className="card">
            <h2>Capability comparison</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Algorithm</th>
                    <th>Type</th>
                    <th>Time quantum</th>
                    <th>Available</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((s) => (
                    <tr key={s.id}>
                      <td>{s.name}</td>
                      <td>{s.type}</td>
                      <td>{s.supports_time_quantum ? 'Yes' : 'No'}</td>
                      <td>{s.available ? 'Yes' : 'No'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="hint">
              Performance comparisons (waiting time, turnaround time and similar) will appear here only once the
              backend can return real simulation results. It does not provide any yet.
            </p>
          </section>
        </>
      )}
    </>
  );
}
