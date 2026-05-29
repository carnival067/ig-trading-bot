import { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import { performanceApi } from '../api/endpoints';
import { LoadingSpinner } from '../components/LoadingSpinner';
import type { EquityPoint, MonthlyReturn, StrategyComparison } from '../types';

export function PerformancePage() {
  const [equityCurve, setEquityCurve] = useState<EquityPoint[]>([]);
  const [monthlyReturns, setMonthlyReturns] = useState<MonthlyReturn[]>([]);
  const [strategies, setStrategies] = useState<StrategyComparison[]>([]);
  const [loading, setLoading] = useState(true);
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [equity, monthly, strats] = await Promise.all([
        performanceApi.getEquityCurve(dateFrom || undefined, dateTo || undefined),
        performanceApi.getMonthlyReturns(),
        performanceApi.getStrategyComparison(),
      ]);
      setEquityCurve(equity);
      setMonthlyReturns(monthly);
      setStrategies(strats);
    } catch {
      // Handle error silently
    } finally {
      setLoading(false);
    }
  };

  const handleFilterApply = () => {
    setLoading(true);
    loadData();
  };

  if (loading) return <LoadingSpinner message="Loading performance data..." />;

  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const years = [...new Set(monthlyReturns.map((r) => r.year))].sort();

  return (
    <div className="performance-page">
      <h2>Performance Analytics</h2>

      {/* Date Range Filter */}
      <section className="filter-bar">
        <div className="form-group">
          <label htmlFor="date-from">From</label>
          <input
            id="date-from"
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label htmlFor="date-to">To</label>
          <input
            id="date-to"
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
          />
        </div>
        <button className="btn btn-primary" onClick={handleFilterApply}>
          Apply
        </button>
      </section>

      {/* Equity Curve */}
      <section className="chart-section">
        <h3>Equity Curve</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={equityCurve}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="equity" stroke="#2196F3" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      {/* Drawdown Chart */}
      <section className="chart-section">
        <h3>Drawdown</h3>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={equityCurve}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey="drawdown" stroke="#f44336" fill="#ffcdd2" />
          </AreaChart>
        </ResponsiveContainer>
      </section>

      {/* Monthly Returns Heatmap */}
      <section className="chart-section">
        <h3>Monthly Returns</h3>
        <div className="heatmap-table">
          <table className="data-table">
            <thead>
              <tr>
                <th>Year</th>
                {months.map((m) => (
                  <th key={m}>{m}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {years.map((year) => (
                <tr key={year}>
                  <td>{year}</td>
                  {months.map((_, idx) => {
                    const entry = monthlyReturns.find(
                      (r) => r.year === year && r.month === idx + 1
                    );
                    const value = entry?.return_pct ?? null;
                    const bgColor =
                      value === null
                        ? '#f5f5f5'
                        : value >= 0
                        ? `rgba(76, 175, 80, ${Math.min(Math.abs(value) * 5, 1)})`
                        : `rgba(244, 67, 54, ${Math.min(Math.abs(value) * 5, 1)})`;
                    return (
                      <td
                        key={idx}
                        style={{ backgroundColor: bgColor, textAlign: 'center' }}
                      >
                        {value !== null ? `${(value * 100).toFixed(1)}%` : '-'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Strategy Comparison */}
      <section className="chart-section">
        <h3>Strategy Comparison</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Total PnL</th>
              <th>Win Rate</th>
              <th>Sharpe Ratio</th>
              <th>Max Drawdown</th>
              <th>Trades</th>
            </tr>
          </thead>
          <tbody>
            {strategies.map((s) => (
              <tr key={s.strategy}>
                <td>{s.strategy}</td>
                <td className={s.total_pnl >= 0 ? 'positive' : 'negative'}>
                  £{s.total_pnl.toFixed(2)}
                </td>
                <td>{(s.win_rate * 100).toFixed(1)}%</td>
                <td>{s.sharpe_ratio.toFixed(2)}</td>
                <td className="negative">{(s.max_drawdown * 100).toFixed(1)}%</td>
                <td>{s.trade_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
