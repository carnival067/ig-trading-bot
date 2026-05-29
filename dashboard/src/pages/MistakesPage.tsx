import { useState, useEffect } from 'react';
import { mistakesApi } from '../api/endpoints';
import { useWebSocket } from '../hooks/useWebSocket';
import { LoadingSpinner } from '../components/LoadingSpinner';
import type { MistakePattern } from '../types';

export function MistakesPage() {
  const [patterns, setPatterns] = useState<MistakePattern[]>([]);
  const [loading, setLoading] = useState(true);
  const { lastMessage } = useWebSocket();

  useEffect(() => {
    loadPatterns();
  }, []);

  // Real-time mistake pattern updates
  useEffect(() => {
    if (lastMessage?.type === 'mistake_update' && lastMessage.data) {
      const updated = lastMessage.data as MistakePattern;
      setPatterns((prev) =>
        prev.map((p) => (p.id === updated.id ? updated : p))
      );
    }
  }, [lastMessage]);

  const loadPatterns = async () => {
    try {
      const data = await mistakesApi.getPatterns();
      setPatterns(data);
    } catch {
      // Handle error
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <LoadingSpinner message="Loading mistake patterns..." />;

  const getPenaltyColor = (level: number) => {
    if (level >= 3) return '#F44336';
    if (level >= 2) return '#FF9800';
    return '#FFC107';
  };

  return (
    <div className="mistakes-page">
      <h2>Mistake Patterns</h2>

      {patterns.length === 0 ? (
        <p className="empty-state">No active mistake patterns detected. Keep trading well!</p>
      ) : (
        <section className="patterns-list">
          <h3>Active Patterns</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>Classification</th>
                <th>Loss Count</th>
                <th>Total Loss</th>
                <th>Last Occurrence</th>
                <th>Penalty Level</th>
                <th>Resolution Progress</th>
              </tr>
            </thead>
            <tbody>
              {patterns.map((pattern) => (
                <tr key={pattern.id}>
                  <td className="pattern-classification">{pattern.classification}</td>
                  <td>{pattern.loss_count}</td>
                  <td className="negative">£{pattern.total_loss.toFixed(2)}</td>
                  <td>{new Date(pattern.last_occurrence).toLocaleString()}</td>
                  <td>
                    <span
                      className="penalty-badge"
                      style={{ backgroundColor: getPenaltyColor(pattern.penalty_level) }}
                    >
                      Level {pattern.penalty_level}
                    </span>
                  </td>
                  <td>
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{
                          width: `${pattern.resolution_progress * 100}%`,
                          backgroundColor:
                            pattern.resolution_progress > 0.7
                              ? '#4CAF50'
                              : pattern.resolution_progress > 0.3
                              ? '#FF9800'
                              : '#F44336',
                        }}
                      />
                      <span className="progress-label">
                        {(pattern.resolution_progress * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
