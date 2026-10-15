import { useState } from 'react';
import useFetch from '../hooks/useFetch.js';
import { getSchedulers, runSimulation, getLatestSimulation } from '../services/api.js';
import { Loader, ErrorBanner } from '../components/Common.jsx';

export default function Simulation({ status, refreshKey }) {
  const caps = status.data && status.data.capabilities;
  const canRun = !!(caps && caps.simulation_run);
  const sched = useFetch(getSchedulers, [refreshKey]);
  const latest = useFetch(getLatestSimulation, [refreshKey], !!(caps && caps.simulation_results));

  const [algorithm, setAlgorithm] = useState('fcfs');
  const [quantum, setQuantum] = useState('2');
  const [limit, setLimit] = useState('20');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const options = ((sched.data && sched.data.items) || []).filter((s) => s.available);
  const chosen = options.find((s) => s.id === algorithm);

  async function run(e) {
    e.preventDefault();
    setRunning(true);
    setResult(null);
    setError(null);
    const body = { algorithm, limit: Number(limit) };
    if (chosen && chosen.supports_time_quantum) body.time_quantum = Number(quantum);
    try {
      setResult(await runSimulation(body));
    } catch (err) {
      setError(err);
    } finally {
      setRunning(false);
    }
  }

  return (
    <>
      {status.error && <ErrorBanner error={status.error} onRetry={status.refetch} />}
      {status.loading && !status.data && <Loader text="Checking simulation support…" />}
      {status.data && !canRun && (
        <div className="banner warn">
          <div>
            <strong>Simulation is unavailable</strong>
            <p>{status.data.cpp_engine.reason}</p>
            <p>The run control stays disabled until the backend reports simulation_run as true.</p>
          </div>
        </div>
      )}

      <section className="card">
        <h2>Run a simulation</h2>
        {sched.error && <ErrorBanner error={sched.error} onRetry={sched.refetch} />}
        <form onSubmit={run}>
          <fieldset disabled={!canRun || running}>
            <label>
              Algorithm
              <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
                {options.length === 0 && <option value="fcfs">Algorithms not loaded</option>}
                {options.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Time quantum
              <input
                type="number"
                min="1"
                value={quantum}
                onChange={(e) => setQuantum(e.target.value)}
                disabled={!chosen || !chosen.supports_time_quantum}
              />
            </label>
            <label>
              Transactions to schedule (1–200)
              <input type="number" min="1" max="200" value={limit} onChange={(e) => setLimit(e.target.value)} />
            </label>
            <button className="btn primary" type="submit">
              {running ? 'Running…' : 'Run simulation'}
            </button>
          </fieldset>
        </form>
        {error && <ErrorBanner error={error} />}
        {result && <pre className="json">{JSON.stringify(result, null, 2)}</pre>}
      </section>

      {latest.data && (
        <section className="card">
          <h2>Latest result</h2>
          <pre className="json">{JSON.stringify(latest.data, null, 2)}</pre>
        </section>
      )}
    </>
  );
}
