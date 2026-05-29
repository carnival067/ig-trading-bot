import { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { backtestApi } from '../api/endpoints';
import { LoadingSpinner } from '../components/LoadingSpinner';
import type { BacktestConfig, BacktestResult } from '../types';

export function BacktestPage() {
  const [strategies, setStrategies] = useState<string[]>([]);
  const [config, setConfig] = useState<BacktestConfig>({
    strategy: '',
    parameters: {},
    start_date: '',
    end_date: '',
    initial_capital: 100000,
  });
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStrategies();
  }, []);

  const loadStrategies = async () => {
    try {
      const data = await backtestApi.getStrategies();
      setStrategies(data);
      if (data.length > 0) {
        setConfig((prev) => ({ ...prev, strategy: data[0] }));
      }
    } catch {
      // Handle error
    } finally {
      setLoading(false);
    }
  };

  const handleRun = async () => {
    setRunning(true);
    setResult(null);
    try {
      const data = await backtestApi.run(config);
      setResult(data);
    } catch {
      // Handle error
    } finally {
      setRunning(false);
    }
  };

  if (loading) return <LoadingSpinner message="Loading strategies..." />;

  const monteCarloData = result?.monte_carlo.distribution.map((value, idx) => ({
    bin: idx,
    frequency: value,
  }));

  return (
    <div className="backtest-page">
      <h2>Backtesting</h2>

      {/* Configuration Form */}
      <section className="config-section">
        <h3>Configuration</h3>
        <div className="config-form">
          <div className="form-group">
            <label htmlFor="strategy-select">Strategy</label>
            <select
              id="strategy-select"
              value={config.strategy}
              onChange={(e) => setConfig({ ...config, strategy: e.target.value })}
            >
              {strategies.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="start-date">Start Date</label>
            <input
              id="start-date"
              type="date"
              value={config.start_date}
              onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
            />
          </div>
          <div className="form-group">
            <label htmlFor="end-date">End Date</label>
            <input
              id="end-date"
              type="date"
              value={config.end_date}
              onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
            />
          </div>
          <div className="form-group">
            <label htmlFor="initial-capital">Initial Capital (£)</label>
            <input
              id="initial-capital"
              type="number"
              value={config.initial_capital}
              onChange={(e) =>
                setConfig({ ...config, initial_capital: Number(e.target.value) })
              }
            />
          </div>

          {/* Parameter Configuration */}
          <div className="form-group">
            <label htmlFor="param-risk-pct">Risk Per Trade (%)</label>
            <input
              id="param-risk-pct"
              type="number"
              step="0.1"
              defaultValue="1.0"
              onChange={(e) =>
                setConfig({
                  ...config,
                  parameters: { ...config.parameters, risk_pct: Number(e.target.value) },
                })
              }
            />
          </div>
          <div className="form-group">
            <label htmlFor="param-atr-mult">ATR Multiplier</label>
            <input
              id="param-atr-mult"
              type="number"
              step="0.1"
              defaultValue="1.5"
              onChange={(e) =>
                setConfig({
                  ...config,
                  parameters: { ...config.parameters, atr_multiplier: Number(e.target.value) },
                })
              }
            />
          </div>

          <button
            className="btn btn-primary"
            onClick={handleRun}
            disabled={running || !config.strategy || !config.start_date || !config.end_date}
          >
            {running ? 'Running...' : 'Run Backtest'}
          </button>
        </div>
      </section>

      {running && <LoadingSpinner message="Running backtest..." />}

      {result && (
        <>
          {/* Summary Metrics */}
          <section className="metrics-grid">
            <div className="metric-card">
              <h3>Total Return</h3>
              <span className={result.total_return >= 0 ? 'positive' : 'negative'}>
                {(result.total_return * 100).toFixed(2)}%
              </span>
            </div>
            <div className="metric-card">
              <h3>Sharpe Ratio</h3>
              <span>{result.sharpe_ratio.toFixed(2)}</span>
            </div>
            <div className="metric-card">
              <h3>Max Drawdown</h3>
              <span className="negative">{(result.max_drawdown * 100).toFixed(2)}%</span>
            </div>
            <div className="metric-card">
              <h3>Win Rate</h3>
              <span>{(result.win_rate * 100).toFixed(1)}%</span>
            </div>
            <div className="metric-card">
              <h3>Trade Count</h3>
              <span>{result.trade_count}</span>
            </div>
          </section>

          {/* Equity Curve */}
          <section className="chart-section">
            <h3>Equity Curve</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={result.equity_curve}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="equity" stroke="#2196F3" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </section>

          {/* Monte Carlo Distribution */}
          <section className="chart-section">
            <h3>Monte Carlo Distribution</h3>
            <div className="monte-carlo-stats">
              <span>P5: {(result.monte_carlo.percentiles.p5 * 100).toFixed(1)}%</span>
              <span>P25: {(result.monte_carlo.percentiles.p25 * 100).toFixed(1)}%</span>
              <span>P50: {(result.monte_carlo.percentiles.p50 * 100).toFixed(1)}%</span>
              <span>P75: {(result.monte_carlo.percentiles.p75 * 100).toFixed(1)}%</span>
              <span>P95: {(result.monte_carlo.percentiles.p95 * 100).toFixed(1)}%</span>
            </div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={monteCarloData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="frequency" fill="#9C27B0" />
              </BarChart>
            </ResponsiveContainer>
          </section>

          {/* Walk-Forward Results */}
          <section className="chart-section">
            <h3>Walk-Forward Analysis</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={result.walk_forward}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="in_sample_return" stroke="#2196F3" name="In-Sample" />
                <Line type="monotone" dataKey="out_sample_return" stroke="#FF9800" name="Out-of-Sample" />
              </LineChart>
            </ResponsiveContainer>
          </section>
        </>
      )}
    </div>
  );
}
