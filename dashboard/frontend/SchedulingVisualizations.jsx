import { EmptyState } from './Common.jsx';
import { buildGanttData } from './gantt.js';
import { summarizeResourceMonitoring } from './resourceMonitoring.js';

const colorFor = (id) => {
  if (id == null) return '#65718a';
  let hash = 0;
  for (const character of String(id)) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  return `hsl(${(hash * 137.508) % 360} 72% 66%)`;
};

export function GanttChart({ timeline = [], makespan = 0, title = 'CPU execution timeline' }) {
  const data = buildGanttData(timeline, makespan);
  if (data.error) return <div className="gantt-card"><h3>{title}</h3><p className="history-error">{data.error}</p></div>;
  if (!data.segments.length) return <div className="gantt-card"><h3>{title}</h3><EmptyState title="No timeline available">The scheduler did not return execution intervals for this run.</EmptyState></div>;
  const left = 42;
  const unit = 52;
  const width = Math.max(680, left + data.makespan * unit + 20);
  const tickStep = data.makespan <= 20 ? 1 : Math.ceil(data.makespan / 12);
  const ticks = Array.from({ length: Math.floor(data.makespan / tickStep) + 1 }, (_, i) => i * tickStep);
  if (ticks[ticks.length - 1] !== data.makespan) ticks.push(data.makespan);
  const height = 112;
  return <div className="gantt-card">
    <h3>{title}</h3>
    <p className="chart-caption">CPU time · makespan {data.makespan} time units · block order is scheduling order</p>
    <div className="gantt-scroll" role="region" aria-label={`${title}, horizontally scrollable`}>
      <svg className="gantt-svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <text x="0" y="55" className="gantt-axis-title">CPU</text>
        {data.segments.map((segment) => {
          const x = left + segment.startTime * unit;
          const blockWidth = segment.duration * unit;
          const label = segment.isIdle ? 'Idle' : `T${segment.transactionID}`;
          const description = segment.isIdle
            ? `Idle CPU from ${segment.startTime} to ${segment.endTime}, duration ${segment.duration}`
            : `Order ${segment.order}: transaction ${segment.transactionID}, process ${segment.processID}, start ${segment.startTime}, end ${segment.endTime}, duration ${segment.duration}`;
          return <g key={`${segment.order}-${segment.startTime}`}>
            <title>{description}</title>
            <rect x={x} y="32" width={Math.max(1, blockWidth)} height="38" rx="4"
              fill={segment.isIdle ? '#65718a' : colorFor(segment.transactionID)} />
            {blockWidth > 34 && <text x={x + blockWidth / 2} y="55" textAnchor="middle" className="gantt-block-label">
              {blockWidth > 92 ? `#${segment.order} ${label} · ${segment.duration}` : label}
            </text>}
          </g>;
        })}
        <line x1={left} y1="82" x2={left + data.makespan * unit} y2="82" className="gantt-axis" />
        {ticks.map((tick) => {
          const x = left + tick * unit;
          return <g key={tick}><line x1={x} y1="78" x2={x} y2="87" className="gantt-axis" />
            <text x={x} y="103" textAnchor="middle" className="gantt-tick">{tick}</text></g>;
        })}
      </svg>
    </div>
    <div className="gantt-legend">{[...new Set(data.segments.filter((s) => !s.isIdle).map((s) => s.transactionID))].map((id) =>
      <span key={id}><i style={{ backgroundColor: colorFor(id) }} />Transaction {id}</span>)}
      {data.segments.some((s) => s.isIdle) && <span><i style={{ backgroundColor: '#65718a' }} />CPU idle</span>}
    </div>
  </div>;
}

const METRICS = [
  ['waitingTime', 'Waiting time'],
  ['turnaroundTime', 'Turnaround time'],
  ['responseTime', 'Response time'],
];

function MetricBars({ results, field, label }) {
  if (!results?.length) return <div className="metric-chart"><h3>{label}</h3><p className="chart-caption">No transaction metrics available.</p></div>;
  const max = Math.max(1, ...results.map((row) => Number(row[field]) || 0));
  return <section className="metric-chart"><h3>{label} per transaction</h3><div className="metric-bar-list">
    {results.map((row) => <div className="metric-bar-row" key={`${field}-${row.transactionID}`}>
      <span title={`Transaction ${row.transactionID}`}>T{row.transactionID}</span>
      <div className="metric-track"><div className="metric-fill" style={{ width: `${100 * (Number(row[field]) || 0) / max}%`, backgroundColor: colorFor(row.transactionID) }} /></div>
      <strong>{row[field] ?? '—'}</strong>
    </div>)}
  </div></section>;
}

export function MetricCharts({ results }) {
  return <div className="metric-charts">{METRICS.map(([field, label]) =>
    <MetricBars key={field} results={results} field={field} label={label} />)}</div>;
}

export function ResourceMonitor({ summary, results }) {
  const states = {};
  (results || []).forEach((row) => { const state = row.state || 'COMPLETED'; states[state] = (states[state] || 0) + 1; });
  return <section className="card monitor-card">
    <div className="results-heading"><div><h2>Runtime monitoring</h2><p>Only metrics available from this scheduling run are shown.</p></div>
      <span className="tag muted">ResourceManager telemetry unavailable</span></div>
    <div className="grid stats monitor-stats">
      <div className="card stat"><span className="label">CPU utilization</span><span className="value">{summary?.cpuUtilization == null ? '—' : `${Number(summary.cpuUtilization).toFixed(1)}%`}</span><span className="note">Derived from scheduled burst time and makespan</span></div>
      <div className="card stat"><span className="label">Throughput</span><span className="value">{summary?.throughput == null ? '—' : Number(summary.throughput).toFixed(3)}</span><span className="note">Transactions per time unit</span></div>
      <div className="card stat state-stat"><span className="label">Final transaction states</span><span className="state-counts">{Object.entries(states).length ? Object.entries(states).map(([state, count]) => `${state}: ${count}`).join(' · ') : 'No transaction state data'}</span></div>
    </div>
    <p className="monitor-note">Resource allocation/waiting and mutex/semaphore contention are not collected by the current scheduler CLI, so those values are not displayed.</p>
  </section>;
}

export function ResourceScenarioResults({ result }) {
  if (!result) return null;
  const monitoring = summarizeResourceMonitoring(result.resourceMonitoring);
  const deadlock = monitoring.deadlockDetected;
  return <section className="card monitor-card resource-scenario-results">
    <div className="results-heading"><div><h2>Resource and deadlock monitoring</h2>
      <p>Events and availability reported by the C++ simulated resource manager.</p></div>
      <span className={`tag ${deadlock ? 'warn' : 'good'}`}>{deadlock ? 'Deadlock detected' : 'No deadlock detected'}</span>
    </div>
    <div className="grid stats monitor-stats">
      <div className="card stat"><span className="label">Resource wait events</span><span className="value">{monitoring.resourceWaitEvents ?? 0}</span></div>
      <div className="card stat"><span className="label">Transactions that waited</span><span className="value">{monitoring.waitingTransactionIDs.length ? monitoring.waitingTransactionIDs.map((id) => `T${id}`).join(', ') : 'None'}</span></div>
      <div className="card stat"><span className="label">Lock contention</span><span className="value">{monitoring.lockContentionEvents ?? 0}</span></div>
      <div className="card stat"><span className="label">Allocation / release</span><span className="value">{monitoring.allocations ?? 0} / {monitoring.releases ?? 0}</span></div>
      <div className="card stat"><span className="label">Recovery</span><span className="value">{monitoring.deadlockRecovered ? `Recovered · P${monitoring.recoveryVictimProcessID} aborted` : 'Not required'}</span></div>
    </div>
    <h3>Resource availability and utilization</h3>
    {monitoring.resources?.length ? <div className="table-wrap"><table><thead><tr><th>Resource</th><th>Available</th><th>Allocated</th><th>Utilization</th><th>Current owner</th></tr></thead>
      <tbody>{monitoring.resources.map((resource) => <tr key={resource.resourceID}>
        <td>{resource.name}</td><td>{resource.availableUnits} / {resource.totalUnits}</td>
        <td>{resource.totalUnits - resource.availableUnits}</td><td>{Number(resource.utilizationPercent).toFixed(1)}%</td>
        <td>{monitoring.owners.has(resource.name) ? `Transaction ${monitoring.owners.get(resource.name)}` : 'Released'}</td>
      </tr>)}</tbody></table></div> : <p className="chart-caption">No shared-resource snapshot was returned.</p>}
    {deadlock && <p className="resource-deadlock-note">Detected process cycle: {(monitoring.deadlockedProcessIDs || []).map((id) => `P${id}`).join(' → ')}. Recovery status: {monitoring.deadlockRecovered ? 'cycle cleared' : 'not recovered'}.</p>}
    <h3>Resource event trace</h3>
    {monitoring.events?.length ? <div className="table-wrap"><table><thead><tr><th>Time</th><th>Event</th><th>Transaction</th><th>Process</th><th>Resource</th><th>Availability</th><th>Owner after event</th><th>Detail</th></tr></thead>
      <tbody>{monitoring.events.map((event, index) => <tr key={`${event.time}-${event.type}-${index}`}>
        <td>{event.time}</td><td>{event.type}</td><td>{event.transactionID || '—'}</td><td>{event.processID || '—'}</td>
        <td>{event.resource || '—'}</td><td>{event.availableUnits}</td>
        <td>{event.ownerAfterEvent == null ? 'None' : `Transaction ${event.ownerAfterEvent}`}</td><td>{event.detail}</td>
      </tr>)}</tbody></table></div> : <p className="chart-caption">No resource events were reported.</p>}
  </section>;
}
