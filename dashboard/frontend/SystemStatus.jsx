import { Loader, ErrorBanner, Tag } from '../components/Common.jsx';

function Indicator({ title, tone, label, children }) {
  return (
    <article className="card indicator">
      <div className="algo-head">
        <h2>{title}</h2>
        <Tag tone={tone}>{label}</Tag>
      </div>
      {children}
    </article>
  );
}

export default function SystemStatus({ health, status }) {
  if (status.loading && !status.data) return <Loader text="Loading system status…" />;
  if (status.error) return <ErrorBanner error={status.error} onRetry={status.refetch} />;

  const { backend, database, cpp_engine: engine, drl, capabilities } = status.data;
  const engineTone = engine.available ? 'good' : engine.executable_found ? 'warn' : 'bad';
  const engineLabel = engine.available
    ? 'Available'
    : engine.executable_found
      ? 'Executable found, simulation interface missing'
      : 'Executable not found';

  return (
    <>
      <section className="grid cards">
        <Indicator title="API" tone={backend.status === 'ok' ? 'good' : 'bad'} label={backend.status === 'ok' ? 'Online' : backend.status}>
          {health.data && (
            <p>
              {health.data.service} v{health.data.version}
              <br />
              Server time: {new Date(health.data.time_utc).toLocaleString()}
            </p>
          )}
        </Indicator>
        <Indicator title="Database" tone={database.connected ? 'good' : 'bad'} label={database.connected ? 'Connected' : 'Unavailable'}>
          <p>Schema: {database.name}</p>
          {database.error && <p className="reason">{database.error}</p>}
        </Indicator>
        <Indicator title="C++ engine" tone={engineTone} label={engineLabel}>
          <p className="reason">{engine.reason}</p>
        </Indicator>
        <Indicator title="Deep reinforcement learning" tone={drl.available ? 'good' : 'muted'} label={drl.available ? 'Available' : 'Unavailable'}>
          <p className="reason">{drl.reason}</p>
        </Indicator>
      </section>

      <section className="card">
        <h2>Capabilities reported by the backend</h2>
        <ul className="caps">
          {Object.entries(capabilities).map(([name, on]) => (
            <li key={name}>
              <code>{name}</code>
              <Tag tone={on ? 'good' : 'muted'}>{on ? 'Enabled' : 'Disabled'}</Tag>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
