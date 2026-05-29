import { useState, useEffect } from 'react';
import { tradesApi } from '../api/endpoints';
import { LoadingSpinner } from '../components/LoadingSpinner';
import type { Trade, TradeFilter, PaginatedResponse } from '../types';

export function TradeHistoryPage() {
  const [trades, setTrades] = useState<PaginatedResponse<Trade> | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<TradeFilter>({
    page: 1,
    page_size: 100,
  });

  useEffect(() => {
    loadTrades();
  }, [filter]);

  const loadTrades = async () => {
    setLoading(true);
    try {
      const data = await tradesApi.getHistory(filter);
      setTrades(data);
    } catch {
      // Handle error
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    return `${Math.floor(seconds / 86400)}d`;
  };

  return (
    <div className="trade-history-page">
      <h2>Trade History</h2>

      {/* Filters */}
      <section className="filter-bar">
        <div className="form-group">
          <label htmlFor="filter-strategy">Strategy</label>
          <input
            id="filter-strategy"
            type="text"
            placeholder="All strategies"
            value={filter.strategy || ''}
            onChange={(e) => setFilter({ ...filter, strategy: e.target.value || undefined, page: 1 })}
          />
        </div>
        <div className="form-group">
          <label htmlFor="filter-instrument">Instrument</label>
          <input
            id="filter-instrument"
            type="text"
            placeholder="All instruments"
            value={filter.instrument || ''}
            onChange={(e) => setFilter({ ...filter, instrument: e.target.value || undefined, page: 1 })}
          />
        </div>
        <div className="form-group">
          <label htmlFor="filter-from">From</label>
          <input
            id="filter-from"
            type="date"
            value={filter.date_from || ''}
            onChange={(e) => setFilter({ ...filter, date_from: e.target.value || undefined, page: 1 })}
          />
        </div>
        <div className="form-group">
          <label htmlFor="filter-to">To</label>
          <input
            id="filter-to"
            type="date"
            value={filter.date_to || ''}
            onChange={(e) => setFilter({ ...filter, date_to: e.target.value || undefined, page: 1 })}
          />
        </div>
        <div className="form-group">
          <label htmlFor="filter-outcome">Outcome</label>
          <select
            id="filter-outcome"
            value={filter.outcome || ''}
            onChange={(e) =>
              setFilter({
                ...filter,
                outcome: (e.target.value || undefined) as TradeFilter['outcome'],
                page: 1,
              })
            }
          >
            <option value="">All</option>
            <option value="win">Win</option>
            <option value="loss">Loss</option>
            <option value="breakeven">Breakeven</option>
          </select>
        </div>
      </section>

      {loading ? (
        <LoadingSpinner message="Loading trades..." />
      ) : trades ? (
        <>
          <div className="table-info">
            Showing {trades.items.length} of {trades.total} trades (Page {trades.page} of{' '}
            {trades.total_pages})
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Instrument</th>
                <th>Direction</th>
                <th>Size</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>PnL</th>
                <th>Strategy</th>
                <th>Confidence</th>
                <th>Duration</th>
                <th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {trades.items.map((trade) => (
                <tr key={trade.id}>
                  <td>{new Date(trade.closed_at).toLocaleDateString()}</td>
                  <td>{trade.instrument}</td>
                  <td className={trade.direction === 'long' ? 'positive' : 'negative'}>
                    {trade.direction.toUpperCase()}
                  </td>
                  <td>{trade.size}</td>
                  <td>{trade.entry_price.toFixed(4)}</td>
                  <td>{trade.exit_price.toFixed(4)}</td>
                  <td className={trade.pnl >= 0 ? 'positive' : 'negative'}>
                    £{trade.pnl.toFixed(2)}
                  </td>
                  <td>{trade.strategy}</td>
                  <td>{(trade.confidence_score * 100).toFixed(0)}%</td>
                  <td>{formatDuration(trade.duration_seconds)}</td>
                  <td className={`outcome-${trade.outcome}`}>{trade.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          <div className="pagination">
            <button
              className="btn btn-secondary"
              disabled={trades.page <= 1}
              onClick={() => setFilter({ ...filter, page: (filter.page || 1) - 1 })}
            >
              Previous
            </button>
            <span>
              Page {trades.page} of {trades.total_pages}
            </span>
            <button
              className="btn btn-secondary"
              disabled={trades.page >= trades.total_pages}
              onClick={() => setFilter({ ...filter, page: (filter.page || 1) + 1 })}
            >
              Next
            </button>
          </div>
        </>
      ) : null}
    </div>
  );
}
