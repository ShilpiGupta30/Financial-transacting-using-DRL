// All backend calls live here. Paths and fields follow docs/api_contract.md.
const BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(BASE + path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
  } catch {
    throw {
      code: 'BACKEND_OFFLINE',
      status: 0,
      message: `Cannot reach the backend at ${BASE}. Start it, and check that CORS allows this page's origin.`,
    };
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON body; handled below
  }
  if (!res.ok) {
    const e = body && body.error;
    throw {
      code: (e && e.code) || 'HTTP_ERROR',
      status: res.status,
      message: (e && e.message) || `Request failed with status ${res.status}.`,
    };
  }
  return body;
}

export const getHealth = () => request('/api/health');
export const getStatus = () => request('/api/system/status');
export const getSummary = () => request('/api/transactions/summary');
export const getSchedulers = () => request('/api/schedulers');
export const getTransaction = (id) => request(`/api/transactions/${encodeURIComponent(id)}`);
export const getLatestSimulation = () => request('/api/simulations/latest');
export const runSimulation = (body) =>
  request('/api/simulations/run', { method: 'POST', body: JSON.stringify(body) });

export function getTransactions({ status, limit, offset, sort, order }) {
  const p = new URLSearchParams({ limit, offset, sort, order });
  if (status) p.set('status', status);
  return request(`/api/transactions?${p.toString()}`);
}
