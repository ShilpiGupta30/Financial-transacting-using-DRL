const ITEMS = [
  ['overview', 'Overview'],
  ['transactions', 'Transactions'],
  ['scheduling', 'Scheduling'],
  ['simulation', 'Simulation'],
  ['status', 'System status'],
];

const BACKEND_TEXT = { online: 'Backend online', offline: 'Backend offline', checking: 'Checking backend…' };

export default function Sidebar({ page, onNavigate, open, backend }) {
  return (
    <aside className={'sidebar' + (open ? ' open' : '')}>
      <div className="brand">
        <span className="logo">IRM</span>
        <div>
          <strong>Resource Manager</strong>
          <small>Financial transactions · DRL</small>
        </div>
      </div>
      <nav>
        {ITEMS.map(([id, label]) => (
          <button
            key={id}
            className={'nav-item' + (page === id ? ' active' : '')}
            onClick={() => onNavigate(id)}
            aria-current={page === id ? 'page' : undefined}
          >
            {label}
          </button>
        ))}
      </nav>
      <div className="conn">
        <span className={'dot ' + backend} />
        {BACKEND_TEXT[backend]}
      </div>
    </aside>
  );
}
