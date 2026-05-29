import { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { riskApi } from '../api/endpoints';
import { ConfirmDialog } from '../components/ConfirmDialog';
import { LoadingSpinner } from '../components/LoadingSpinner';
import type { RiskMetrics } from '../types';

const COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548'];

export function RiskPage() {
  const [metrics, setMetrics] = useState<RiskMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [showKillSwitch, setShowKillSwitch] = useState(false);
  const [killSwitchLoading, setKillSwitchLoading] = useState(false);

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    try {
      const data = await riskApi.getMetrics();
      setMetrics(data);
    } catch {
      // Handle error
    } finally {
      setLoading(false);
    }
  };

  const handleKillSwitch = async () => {
    setKillSwitchLoading(true);
    try {
      if (metrics?.kill_switch_active) {
        await riskApi.deactivateKillSwitch();
      } else {
        await riskApi.activateKillSwitch();
      }
      await loadMetrics();
    } catch {
      // Handle error
    } finally {
      setKillSwitchLoading(false);
      setShowKillSwitch(false);
    }
  };

  if (loading) return <LoadingSpinner message="Loading risk metrics..." />;
  if (!metrics) return <div className="error-state">Failed to load risk data</div>;

  const pieData = metrics.exposures.map((e) => ({
    name: e.asset_class,
    value: e.exposure_pct,
  }));

  return (
    <div className="risk-page">
      <h2>Risk Management</h2>

      {/* Kill Switch */}
      <section className="kill-switch-section">
        <div className="kill-switch-status">
          <h3>Kill Switch</h3>
          <span className={metrics.kill_switch_active ? 'status-active' : 'status-inactive'}>
            {metrics.kill_switch_active ? '🔴 ACTIVE' : '🟢 Inactive'}
          </span>
        </div>
        <button
          className={`btn ${metrics.kill_switch_active ? 'btn-warning' : 'btn-danger'}`}
          onClick={() => setShowKillSwitch(true)}
          disabled={killSwitchLoading}
        >
          {metrics.kill_switch_active ? 'Deactivate Kill Switch' : 'Activate Kill Switch'}
        </button>
      </section>

      {/* VaR */}
      <section className="metrics-grid">
        <div className="metric-card">
          <h3>Value at Risk (95% 1-day)</h3>
          <span className="negative">£{metrics.var_95_1day.toFixed(2)}</span>
        </div>
      </section>

      {/* Exposure Pie Chart */}
      <section className="chart-section">
        <h3>Exposure by Asset Class</h3>
        <ResponsiveContainer width="100%" height={350}>
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              outerRadius={120}
              dataKey="value"
              label={({ name, value }) => `${name}: ${(value * 100).toFixed(1)}%`}
            >
              {pieData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </section>

      {/* Correlation Matrix */}
      <section className="chart-section">
        <h3>Correlation Matrix</h3>
        {metrics.correlations.length === 0 ? (
          <p className="empty-state">No correlation data available</p>
        ) : (
          <table className="data-table correlation-table">
            <thead>
              <tr>
                <th>Instrument A</th>
                <th>Instrument B</th>
                <th>Correlation</th>
              </tr>
            </thead>
            <tbody>
              {metrics.correlations.map((c, idx) => (
                <tr key={idx}>
                  <td>{c.instrument_a}</td>
                  <td>{c.instrument_b}</td>
                  <td
                    style={{
                      color: Math.abs(c.correlation) > 0.7 ? '#f44336' : 'inherit',
                      fontWeight: Math.abs(c.correlation) > 0.7 ? 'bold' : 'normal',
                    }}
                  >
                    {c.correlation.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* Kill Switch Confirmation Dialog */}
      {showKillSwitch && (
        <ConfirmDialog
          title={metrics.kill_switch_active ? 'Deactivate Kill Switch' : 'Activate Kill Switch'}
          message={
            metrics.kill_switch_active
              ? 'This will re-enable trading. Are you sure the market conditions are safe?'
              : 'This will immediately close ALL open positions and block all new trades. This action cannot be undone for at least 5 minutes.'
          }
          confirmLabel={metrics.kill_switch_active ? 'Deactivate' : 'Activate Kill Switch'}
          variant="danger"
          onConfirm={handleKillSwitch}
          onCancel={() => setShowKillSwitch(false)}
        />
      )}
    </div>
  );
}
