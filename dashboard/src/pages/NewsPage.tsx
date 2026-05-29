import { useState, useEffect } from 'react';
import { newsApi } from '../api/endpoints';
import { useWebSocket } from '../hooks/useWebSocket';
import { LoadingSpinner } from '../components/LoadingSpinner';
import type { NewsItem, GeopoliticalRisk, EconomicEvent, CrisisAlert } from '../types';

export function NewsPage() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [geoRisks, setGeoRisks] = useState<GeopoliticalRisk[]>([]);
  const [events, setEvents] = useState<EconomicEvent[]>([]);
  const [crisisAlerts, setCrisisAlerts] = useState<CrisisAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const { lastMessage } = useWebSocket();

  useEffect(() => {
    loadData();
  }, []);

  // Real-time news updates
  useEffect(() => {
    if (lastMessage?.type === 'news_update' && lastMessage.data) {
      const newItem = lastMessage.data as NewsItem;
      setNews((prev) => [newItem, ...prev].slice(0, 50));
    }
    if (lastMessage?.type === 'crisis_alert' && lastMessage.data) {
      const alert = lastMessage.data as CrisisAlert;
      setCrisisAlerts((prev) => [alert, ...prev]);
    }
  }, [lastMessage]);

  const loadData = async () => {
    try {
      const [newsData, geoData, eventsData, alertsData] = await Promise.all([
        newsApi.getFeed(50),
        newsApi.getGeopoliticalRisks(),
        newsApi.getEconomicEvents(),
        newsApi.getCrisisAlerts(),
      ]);
      setNews(newsData);
      setGeoRisks(geoData);
      setEvents(eventsData);
      setCrisisAlerts(alertsData);
    } catch {
      // Handle error
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <LoadingSpinner message="Loading news data..." />;

  const getSentimentColor = (score: number) => {
    if (score > 0.3) return '#4CAF50';
    if (score < -0.3) return '#F44336';
    return '#FF9800';
  };

  const getImpactBadge = (impact: string) => {
    const colors: Record<string, string> = { high: '#F44336', medium: '#FF9800', low: '#4CAF50' };
    return (
      <span className="impact-badge" style={{ backgroundColor: colors[impact] || '#999' }}>
        {impact.toUpperCase()}
      </span>
    );
  };

  return (
    <div className="news-page">
      <h2>News & Intelligence</h2>

      {/* Crisis Alerts */}
      {crisisAlerts.length > 0 && (
        <section className="crisis-alerts">
          <h3>🚨 Active Crisis Alerts</h3>
          <div className="alerts-list">
            {crisisAlerts.map((alert) => (
              <div key={alert.id} className={`alert-card severity-${alert.severity}`}>
                <div className="alert-header">
                  <strong>{alert.title}</strong>
                  <span className={`severity-badge ${alert.severity}`}>
                    {alert.severity.toUpperCase()}
                  </span>
                </div>
                <div className="alert-details">
                  <span>Region: {alert.region}</span>
                  <span>Since: {new Date(alert.started_at).toLocaleString()}</span>
                </div>
                <div className="alert-instruments">
                  Affected: {alert.affected_instruments.join(', ')}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <div className="news-grid">
        {/* Live Feed */}
        <section className="news-feed">
          <h3>Live Feed (50 most recent)</h3>
          <div className="feed-list">
            {news.map((item) => (
              <div key={item.id} className="news-item">
                <div className="news-header">
                  <span className="news-source">{item.source}</span>
                  <span className="news-time">
                    {new Date(item.published_at).toLocaleTimeString()}
                  </span>
                </div>
                <h4>{item.title}</h4>
                <div className="news-meta">
                  <span
                    className="sentiment-score"
                    style={{ color: getSentimentColor(item.sentiment_score) }}
                  >
                    Sentiment: {item.sentiment_score.toFixed(2)}
                  </span>
                  {getImpactBadge(item.impact)}
                  {item.correlated_instruments.length > 0 && (
                    <span className="correlated">
                      📈 {item.correlated_instruments.join(', ')}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Sidebar */}
        <aside className="news-sidebar">
          {/* Geopolitical Risk Scores */}
          <section className="geo-risks">
            <h3>Geopolitical Risk by Region</h3>
            <div className="risk-list">
              {geoRisks.map((risk) => (
                <div key={risk.region} className="risk-item">
                  <span className="risk-region">{risk.region}</span>
                  <div className="risk-bar">
                    <div
                      className="risk-fill"
                      style={{
                        width: `${risk.risk_score}%`,
                        backgroundColor:
                          risk.risk_score > 70
                            ? '#F44336'
                            : risk.risk_score > 40
                            ? '#FF9800'
                            : '#4CAF50',
                      }}
                    />
                  </div>
                  <span className="risk-score">{risk.risk_score}</span>
                  <span className={`trend-${risk.trend}`}>
                    {risk.trend === 'rising' ? '↑' : risk.trend === 'falling' ? '↓' : '→'}
                  </span>
                </div>
              ))}
            </div>
          </section>

          {/* Economic Events (24h) */}
          <section className="economic-events">
            <h3>Upcoming Events (24h)</h3>
            <div className="events-list">
              {events.map((event) => (
                <div key={event.id} className="event-item">
                  <div className="event-header">
                    {getImpactBadge(event.impact)}
                    <span className="event-time">
                      {new Date(event.scheduled_at).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="event-title">{event.title}</div>
                  <div className="event-details">
                    <span>{event.currency}</span>
                    <span>Prev: {event.previous}</span>
                    <span>Fcst: {event.forecast}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
