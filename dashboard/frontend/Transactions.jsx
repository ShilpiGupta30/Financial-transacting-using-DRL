import { useState } from 'react';
import useFetch from '../hooks/useFetch.js';
import { getTransactions, getTransaction } from '../services/api.js';
import { Loader, ErrorBanner, EmptyState, StatusBadge } from '../components/Common.jsx';

const STATUSES = ['PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'ROLLED_BACK'];
const COLUMNS = [
  ['TransactionID', 'ID'],
  ['TransactionType', 'Type'],
  ['SourceAccountID', 'Source'],
  ['DestinationAccountID', 'Destination'],
  ['Amount', 'Amount'],
  ['Status', 'Status'],
  ['Priority', 'Priority'],
  ['CreatedAt', 'Created'],
  ['CompletedAt', 'Completed'],
];

function cell(row, key) {
  const v = row[key];
  if (v === null || v === undefined || v === '') return '—';
  if (key === 'Status') return <StatusBadge status={v} />;
  if (key === 'CreatedAt' || key === 'CompletedAt') return new Date(v).toLocaleString();
  return v; // Amount stays the decimal string from the API
}

function Table({ rows, sort, order, onSort }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {COLUMNS.map(([key, label]) => (
              <th key={key} className={key === 'Amount' ? 'num' : ''}>
                {onSort ? (
                  <button className="sort" onClick={() => onSort(key)}>
                    {label}
                    {sort === key ? (order === 'asc' ? ' ▲' : ' ▼') : ''}
                  </button>
                ) : (
                  label
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.TransactionID}>
              {COLUMNS.map(([key]) => (
                <td key={key} className={key === 'Amount' ? 'num' : ''}>
                  {cell(r, key)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Transactions({ status, refreshKey }) {
  const canRead = !!(status.data && status.data.capabilities.transactions_read);
  const [q, setQ] = useState({ status: '', limit: 25, offset: 0, sort: 'TransactionID', order: 'asc' });
  const [filter, setFilter] = useState('');
  const [idText, setIdText] = useState('');
  const [found, setFound] = useState(null);
  const [idError, setIdError] = useState(null);

  const list = useFetch(() => getTransactions(q), [refreshKey, JSON.stringify(q)], canRead);
  const update = (patch) => setQ((old) => ({ ...old, offset: 0, ...patch }));

  function onSort(key) {
    update({ sort: key, order: q.sort === key && q.order === 'asc' ? 'desc' : 'asc' });
  }

  async function lookup(e) {
    e.preventDefault();
    setFound(null);
    setIdError(null);
    if (!/^\d+$/.test(idText.trim())) return setIdError({ message: 'Enter a numeric transaction ID.' });
    try {
      setFound(await getTransaction(idText.trim()));
    } catch (err) {
      setIdError(err.code === 'NOT_FOUND' ? { message: 'No transaction with that ID.' } : err);
    }
  }

  if (status.loading && !status.data) return <Loader text="Checking database availability…" />;
  if (status.error) return <ErrorBanner error={status.error} onRetry={status.refetch} />;
  if (!canRead) {
    const db = status.data.database;
    return (
      <div className="card">
        <EmptyState title="Transactions are unavailable">
          {db.error || 'The backend reports that it cannot read transactions.'} Configure the database credentials
          in the backend, then refresh.
        </EmptyState>
      </div>
    );
  }

  const rows = ((list.data && list.data.items) || []).filter((r) =>
    JSON.stringify(Object.values(r)).toLowerCase().includes(filter.toLowerCase())
  );
  const total = list.data ? list.data.total : 0;
  const from = total === 0 ? 0 : q.offset + 1;
  const to = Math.min(q.offset + q.limit, total);

  return (
    <>
      <section className="card toolbar">
        <label>
          Status
          <select value={q.status} onChange={(e) => update({ status: e.target.value })}>
            <option value="">All statuses</option>
            {STATUSES.map((s) => (
              <option key={s} value={s}>
                {s.replace('_', ' ')}
              </option>
            ))}
          </select>
        </label>
        <label>
          Rows per page
          <select value={q.limit} onChange={(e) => update({ limit: Number(e.target.value) })}>
            {[10, 25, 50, 100].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <label>
          Filter this page
          <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Type to filter rows" />
        </label>
        <form onSubmit={lookup}>
          <label>
            Go to ID
            <input value={idText} onChange={(e) => setIdText(e.target.value)} placeholder="e.g. 3" inputMode="numeric" />
          </label>
          <button className="btn">Find</button>
        </form>
      </section>

      {idError && <ErrorBanner error={idError} />}
      {found && (
        <section className="card">
          <h2>Transaction {found.TransactionID}</h2>
          <Table rows={[found]} />
        </section>
      )}

      <section className="card">
        {list.error && <ErrorBanner error={list.error} onRetry={list.refetch} />}
        {list.loading && !list.data && <Loader text="Loading transactions…" />}
        {list.data && rows.length === 0 && (
          <EmptyState title="No transactions to show">
            {filter ? 'Nothing on this page matches your filter.' : 'No records match the selected status.'}
          </EmptyState>
        )}
        {rows.length > 0 && <Table rows={rows} sort={q.sort} order={q.order} onSort={onSort} />}
        {list.data && (
          <div className="pager">
            <span>
              Showing {from}–{to} of {total}
            </span>
            <div>
              <button className="btn" disabled={q.offset === 0} onClick={() => setQ({ ...q, offset: Math.max(0, q.offset - q.limit) })}>
                Previous
              </button>
              <button className="btn" disabled={q.offset + q.limit >= total} onClick={() => setQ({ ...q, offset: q.offset + q.limit })}>
                Next
              </button>
            </div>
          </div>
        )}
      </section>
    </>
  );
}
