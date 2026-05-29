import { NavLink, Outlet } from 'react-router-dom';
import { useWebSocket } from '../hooks/useWebSocket';

const navItems = [
  { path: '/', label: 'Dashboard' },
  { path: '/performance', label: 'Performance' },
  { path: '/trades', label: 'Trade History' },
  { path: '/risk', label: 'Risk' },
  { path: '/copy-trading', label: 'Copy Trading' },
  { path: '/backtest', label: 'Backtesting' },
  { path: '/news', label: 'News' },
  { path: '/hft', label: 'HFT' },
  { path: '/mistakes', label: 'Mistakes' },
];

export function Layout() {
  const { isConnected, disconnectedWarning } = useWebSocket();

  return (
    <div className="app-layout">
      {disconnectedWarning && (
        <div className="disconnection-banner" role="alert">
          ⚠️ Connection lost. Attempting to reconnect...
        </div>
      )}
      <nav className="sidebar">
        <div className="sidebar-header">
          <h1>IG Trading</h1>
          <span className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '● Live' : '○ Offline'}
          </span>
        </div>
        <ul className="nav-list">
          {navItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
              >
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
