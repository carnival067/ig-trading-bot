import { useState, useEffect } from 'react';
import { dashboardApi } from '../api/endpoints';
import { useWebSocket } from '../hooks/useWebSocket';
import { LoadingSpinner } from '../components/LoadingSpinner';
import type { DashboardMetrics } from '../types';

export function DashboardPage() {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { lastMessage } = useWebSocket();

  useEffect(() => {
    loadMetrics();
  }, []);

  // Update metrics from WebSocket messages
  useEffect(() => {
    if (lastMessage?.type === 'dashboard_update' && lastMessage.data) {
      setMetrics(lastMessage.data as DashboardMetrics);
    }
  }, [lastMessage]);

  const loadMetrics = async () => {
    try {
      const data = await dashboardApi.getMetrics();
      setMetrics(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load metrics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <LoadingSpinner message="Loading dashboard..." />;
  if (error) return <div className="error-state">{error}</div>;
  if (!metrics) return null;

  const formatCurrency = (value: number) =>
    new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP' }).format(value);

  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;

  return (
    <div className="dashboard-page">
      <h2>Dashboard</h2>

      {/* PnL Cards */}
      <section className="metrics-grid">
        <div className="metric-card">
          <h3>Daily PnL</h3>
          <span className={metrics.pnl.daily >= 0 ? 'positive' : 'negative'}>
            {formatCurrency(metrics.pnl.daily)}
          </span>
        </div>
        <div className="metric-card">
          <h3>Weekly PnL</h3>
          <span className={metrics.pnl.weekly >= 0 ? 'positive' : 'negative'}>
            {formatCurrency(metrics.pnl.weekly)}
          </span>
        </div>
        <div className="metric-card">
          <h3>Monthly PnL</h3>
          <span className={metrics.pnl.monthly >= 0 ? 'positive' : 'negative'}>
            {formatCurrency(metrics.pnl.monthly)}
          </span>
        </div>
        <div className="metric-card">
          <h3>All-Time PnL</h3>
          <span className={metrics.pnl.all_time >= 0 ? 'positive' : 'negative'}>
            {formatCurrency(metrics.pnl.all_time)}
          </span>
        </div>
      </section>

      {/* Key Metrics */}
      <section className="metrics-grid">
        <div className="metric-card">
          <h3>Win Rate</h3>
          <span>{formatPercent(metrics.win_rate)}</span>
        </div>
        <div className="metric-card">
          <h3>Drawdown</h3>
          <span className="negative">{formatPercent(metrics.drawdown)}</span>
        </div>
        <div className="metric-card">
          <h3>AI Confidence</h3>
          <span>{formatPercent(metrics.ai_confidence)}</span>
        </div>
        <div className="metric-card">
          <h3>Market Regime</h3>
          <span className={`regime-${metrics.market_regime}`}>
            {metrics.market_regime.replace('_', ' ').toUpperCase()}
          </span>
        </div>
      </section>

      {/* Open Positions */}
      <section className="positions-section">
        <h3>Open Positions</h3>
        {metrics.open_positions.length === 0 ? (
          <p className="empty-state">No open positions</p>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Instrument</th>
                <th>Direction</th>
                <th>Size</th>
                <th>Entry</th>
                <th>Current</th>
                <th>Unrealized PnL</th>
                <th>Strategy</th>
              </tr>
            </thead>
            <tbody>
              {metrics.open_positions.map((pos) => (
                <tr key={pos.id}>
                  <td>{pos.instrument}</td>
                  <td className={pos.direction === 'long' ? 'positive' : 'negative'}>
                    {pos.direction.toUpperCase()}
                  </td>
                  <td>{pos.size}</td>
                  <td>{pos.entry_price.toFixed(4)}</td>
                  <td>{pos.current_price.toFixed(4)}</td>
                  <td className={pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}>
                    {formatCurrency(pos.unrealized_pnl)}
                  </td>
                  <td>{pos.strategy}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
