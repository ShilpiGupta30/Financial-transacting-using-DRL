import { useEffect, useState } from 'react';
import Sidebar from './components/Sidebar.jsx';
import Header from './components/Header.jsx';
import useFetch from './hooks/useFetch.js';
import { getHealth, getStatus } from './services/api.js';
import Overview from './pages/Overview.jsx';
import Transactions from './pages/Transactions.jsx';
import Scheduling from './pages/Scheduling.jsx';
import Simulation from './pages/Simulation.jsx';
import SystemStatus from './pages/SystemStatus.jsx';

const TITLES = {
  overview: ['Overview', 'Backend health and a summary of transactions'],
  transactions: ['Transactions', 'Browse and filter records from the database'],
  scheduling: ['Scheduling', 'Algorithms implemented by the C++ engine'],
  simulation: ['Simulation', 'Run a scheduling simulation when the backend supports it'],
  status: ['System status', 'API, database, C++ engine and DRL availability'],
};

export default function App() {
  const [page, setPage] = useState('overview');
  const [menuOpen, setMenuOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [updated, setUpdated] = useState(null);

  // Shared by all pages so the capability flags are fetched once per refresh.
  const health = useFetch(getHealth, [refreshKey]);
  const status = useFetch(getStatus, [refreshKey]);

  useEffect(() => {
    if (!health.loading && !status.loading) setUpdated(new Date());
  }, [health.loading, status.loading]);

  const backend = health.loading && !health.data ? 'checking' : health.error ? 'offline' : 'online';
  const shared = { health, status, refreshKey };
  const [title, subtitle] = TITLES[page];

  return (
    <div className="app">
      <Sidebar
        page={page}
        backend={backend}
        open={menuOpen}
        onNavigate={(p) => {
          setPage(p);
          setMenuOpen(false);
        }}
      />
      {menuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} />}
      <div className="main">
        <Header
          title={title}
          subtitle={subtitle}
          updated={updated}
          loading={health.loading || status.loading}
          onRefresh={() => setRefreshKey((k) => k + 1)}
          onMenu={() => setMenuOpen(true)}
        />
        <main className="content">
          {page === 'overview' && <Overview {...shared} />}
          {page === 'transactions' && <Transactions {...shared} />}
          {page === 'scheduling' && <Scheduling {...shared} />}
          {page === 'simulation' && <Simulation {...shared} />}
          {page === 'status' && <SystemStatus {...shared} />}
        </main>
      </div>
    </div>
  );
}
