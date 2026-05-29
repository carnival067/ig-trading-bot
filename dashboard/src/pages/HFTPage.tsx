import { useState, useEffect } from 'react';
import { hftApi } from '../api/endpoints';
import { useWebSocket } from '../hooks/useWebSocket';
import { ConfirmDialog } from '../components/ConfirmDialog';
import { LoadingSpinner } from '../components/LoadingSpinner';
import type { HFTMetrics } from '../types';

export function HFTPage() {
  const [metrics, setMetrics] = useState<HFTMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [showReEnable, setShowReEnable] = useState(false);
  const { lastMessage } = useWebSocket();

  useEffect(() => {
    loadMetrics();
  }, []);

  // Real-time HFT updates
  useEffect(() => {
    if (lastMessage?.type === 'hft_update' && lastMessage.data) {
      setMetrics(lastMessage.data as HFTMetrics);
    }
  }, [lastMessage]);

  const loadMetrics = async () => {
    try {
      const data = await hftApi.getMetrics();
      setMetrics(data);
    } catch {
      // Handle error
    } finally {
      setLoading(false);
    }
  };

  const handleReEnable = async () => {
    try {
      await hftApi.reEnableCircuitBreaker();
      await loadMetrics();
    } catch {
      // Handle error
    } finally {
      setShowReEnable(false);
    }
  };

  if (loading) return <LoadingSpinner message="Loading HFT metrics..." />;
  if (!metrics) return <div className="error-state">Failed to load HFT data</div>;

  const getLatencyColor = (ms: number) => {
    if (ms < 5) return '#4CAF50';
    if (ms < 20) return '#FF9800';
    return '#F44336';
  };

  const getGaugeRotation = (value: number, max: number) => {
    return Math.min((value / max) * 180, 180);
  };

  return (
    <div className="hft-page">
      <h2>HFT Dashboard</h2>

      {/* Circuit Breaker Status */}
      <section className="circuit-breaker-section">
        <div className="circuit-breaker-status">
          <h3>Circuit Breaker</h3>
          <span className={metrics.circuit_breaker_active ? 'status-active' : 'status-inactive'}>
            {metrics.circuit_breaker_active ? '🔴 TRIPPED' : '🟢 Normal'}
          </span>
        </div>
        {metrics.circuit_breaker_active && (
          <button className="btn btn-warning" onClick={() => setShowReEnable(true)}>
            Manual Re-Enable
          </button>
        )}
      </section>

      {/* Key Metrics */}
      <section className="metrics-grid">
        {/* Orders/sec Gauge */}
        <div className="metric-card gauge-card">
          <h3>Orders/sec</h3>
          <div className="gauge">
            <div
              className="gauge-needle"
              style={{
                transform: `rotate(${getGaugeRotation(metrics.orders_per_second, 1000)}deg)`,
              }}
            />
            <span className="gauge-value">{metrics.orders_per_second}</span>
          </div>
        </div>

        {/* Avg Latency */}
        <div className="metric-card">
          <h3>Avg Latency</h3>
          <span style={{ color: getLatencyColor(metrics.avg_latency_ms) }}>
            {metrics.avg_latency_ms.toFixed(2)} ms
          </span>
        </div>

        {/* Total HFT Exposure */}
        <div className="metric-card">
          <h3>Total HFT Exposure</h3>
          <span>{(metrics.total_exposure_pct * 100).toFixed(1)}%</span>
        </div>
      </section>

      {/* PnL Metrics */}
      <section className="metrics-grid">
        <div className="metric-card">
          <h3>Net PnL (1 min)</h3>
          <span className={metrics.net_pnl_1min >= 0 ? 'positive' : 'negative'}>
            £{metrics.net_pnl_1min.toFixed(2)}
          </span>
        </div>
        <div className="metric-card">
          <h3>Net PnL (5 min)</h3>
          <span className={metrics.net_pnl_5min >= 0 ? 'positive' : 'negative'}>
            £{metrics.net_pnl_5min.toFixed(2)}
          </span>
        </div>
        <div className="metric-card">
          <h3>Net PnL (Daily)</h3>
          <span className={metrics.net_pnl_daily >= 0 ? 'positive' : 'negative'}>
            £{metrics.net_pnl_daily.toFixed(2)}
          </span>
        </div>
      </section>

      {/* Re-Enable Confirmation */}
      {showReEnable && (
        <ConfirmDialog
          title="Re-Enable HFT Circuit Breaker"
          message="This will reset the circuit breaker and resume HFT operations. Ensure market conditions are stable before proceeding."
          confirmLabel="Re-Enable"
          variant="warning"
          onConfirm={handleReEnable}
          onCancel={() => setShowReEnable(false)}
        />
      )}
    </div>
  );
}
