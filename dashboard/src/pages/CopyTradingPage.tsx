import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { copyTradingApi } from '../api/endpoints';
import { LoadingSpinner } from '../components/LoadingSpinner';
import type { CopyTrader, CopyTraderPerformance } from '../types';

export function CopyTradingPage() {
  const [traders, setTraders] = useState<CopyTrader[]>([]);
  const [selectedTrader, setSelectedTrader] = useState<string | null>(null);
  const [performance, setPerformance] = useState<CopyTraderPerformance | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadTraders();
  }, []);

  useEffect(() => {
    if (selectedTrader) {
      loadPerformance(selectedTrader);
    }
  }, [selectedTrader]);

  const loadTraders = async () => {
    try {
      const data = await copyTradingApi.getTraders();
      setTraders(data);
    } catch {
      // Handle error
    } finally {
      setLoading(false);
    }
  };

  const loadPerformance = async (traderId: string) => {
    try {
      const data = await copyTradingApi.getTraderPerformance(traderId);
      setPerformance(data);
    } catch {
      // Handle error
    }
  };

  const handleAllocationChange = async (traderId: string, value: number) => {
    try {
      await copyTradingApi.updateAllocation(traderId, value);
      setTraders((prev) =>
        prev.map((t) => (t.id === traderId ? { ...t, allocation_pct: value } : t))
      );
    } catch {
      // Handle error
    }
  };

  if (loading) return <LoadingSpinner message="Loading copy trading data..." />;

  return (
    <div className="copy-trading-page">
      <h2>Copy Trading</h2>

      {/* Trader Rankings */}
      <section className="chart-section">
        <h3>Trader Rankings</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Trader</th>
              <th>Win Rate</th>
              <th>Total PnL</th>
              <th>Sharpe</th>
              <th>Max DD</th>
              <th>Followers</th>
              <th>Allocation</th>
            </tr>
          </thead>
          <tbody>
            {traders.map((trader) => (
              <tr
                key={trader.id}
                className={selectedTrader === trader.id ? 'selected-row' : ''}
                onClick={() => setSelectedTrader(trader.id)}
              >
                <td>{trader.name}</td>
                <td>{(trader.win_rate * 100).toFixed(1)}%</td>
                <td className={trader.total_pnl >= 0 ? 'positive' : 'negative'}>
                  £{trader.total_pnl.toFixed(2)}
                </td>
                <td>{trader.sharpe_ratio.toFixed(2)}</td>
                <td className="negative">{(trader.max_drawdown * 100).toFixed(1)}%</td>
                <td>{trader.followers}</td>
                <td>
                  <div className="allocation-slider">
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={trader.allocation_pct}
                      onChange={(e) =>
                        handleAllocationChange(trader.id, Number(e.target.value))
                      }
                      onClick={(e) => e.stopPropagation()}
                      aria-label={`Allocation for ${trader.name}`}
                    />
                    <span>{trader.allocation_pct}%</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* Performance Chart */}
      {performance && (
        <section className="chart-section">
          <h3>Performance Chart</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performance.equity_curve}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="equity" stroke="#4CAF50" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}

      {/* Data Source Configuration */}
      <section className="chart-section">
        <h3>Data Source Configuration</h3>
        <div className="config-form">
          <div className="form-group">
            <label htmlFor="data-source">Data Source</label>
            <select id="data-source" defaultValue="live">
              <option value="live">Live Trading Data</option>
              <option value="demo">Demo Account</option>
              <option value="backtest">Backtest Results</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="update-interval">Update Interval</label>
            <select id="update-interval" defaultValue="5">
              <option value="1">1 second</option>
              <option value="5">5 seconds</option>
              <option value="30">30 seconds</option>
              <option value="60">1 minute</option>
            </select>
          </div>
        </div>
      </section>
    </div>
  );
}
