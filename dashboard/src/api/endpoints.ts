import { apiClient } from './client';
import type {
  AuthToken,
  LoginCredentials,
  DashboardMetrics,
  Trade,
  TradeFilter,
  PaginatedResponse,
  EquityPoint,
  MonthlyReturn,
  StrategyComparison,
  RiskMetrics,
  CopyTrader,
  CopyTraderPerformance,
  BacktestConfig,
  BacktestResult,
  NewsItem,
  GeopoliticalRisk,
  EconomicEvent,
  CrisisAlert,
  HFTMetrics,
  MistakePattern,
} from '../types';

// Auth
export const authApi = {
  login: (credentials: LoginCredentials) =>
    apiClient.post<AuthToken>('/auth/login', credentials),
  logout: () => apiClient.post<void>('/auth/logout'),
  refresh: () => apiClient.post<AuthToken>('/auth/refresh'),
};

// Dashboard
export const dashboardApi = {
  getMetrics: () => apiClient.get<DashboardMetrics>('/dashboard/metrics'),
};

// Performance
export const performanceApi = {
  getEquityCurve: (from?: string, to?: string) => {
    const params = new URLSearchParams();
    if (from) params.set('from', from);
    if (to) params.set('to', to);
    const query = params.toString();
    return apiClient.get<EquityPoint[]>(`/dashboard/equity-curve${query ? `?${query}` : ''}`);
  },
  getMonthlyReturns: () =>
    apiClient.get<MonthlyReturn[]>('/dashboard/monthly-returns'),
  getStrategyComparison: () =>
    apiClient.get<StrategyComparison[]>('/dashboard/strategy-comparison'),
};

// Trades
export const tradesApi = {
  getHistory: (filter: TradeFilter) => {
    const params = new URLSearchParams();
    if (filter.strategy) params.set('strategy', filter.strategy);
    if (filter.instrument) params.set('instrument', filter.instrument);
    if (filter.date_from) params.set('date_from', filter.date_from);
    if (filter.date_to) params.set('date_to', filter.date_to);
    if (filter.outcome) params.set('outcome', filter.outcome);
    if (filter.page) params.set('page', String(filter.page));
    if (filter.page_size) params.set('page_size', String(filter.page_size));
    const query = params.toString();
    return apiClient.get<PaginatedResponse<Trade>>(`/trading/history${query ? `?${query}` : ''}`);
  },
};

// Risk
export const riskApi = {
  getMetrics: () => apiClient.get<RiskMetrics>('/risk/metrics'),
  activateKillSwitch: () => apiClient.post<void>('/risk/kill-switch/activate'),
  deactivateKillSwitch: () => apiClient.post<void>('/risk/kill-switch/deactivate'),
};

// Copy Trading
export const copyTradingApi = {
  getTraders: () => apiClient.get<CopyTrader[]>('/copy-trading/traders'),
  getTraderPerformance: (traderId: string) =>
    apiClient.get<CopyTraderPerformance>(`/copy-trading/traders/${traderId}/performance`),
  updateAllocation: (traderId: string, allocation_pct: number) =>
    apiClient.put<void>(`/copy-trading/traders/${traderId}/allocation`, { allocation_pct }),
};

// Backtesting
export const backtestApi = {
  run: (config: BacktestConfig) =>
    apiClient.post<BacktestResult>('/backtest/run', config),
  getStrategies: () => apiClient.get<string[]>('/backtest/strategies'),
};

// News
export const newsApi = {
  getFeed: (limit?: number) =>
    apiClient.get<NewsItem[]>(`/news/feed${limit ? `?limit=${limit}` : ''}`),
  getGeopoliticalRisks: () =>
    apiClient.get<GeopoliticalRisk[]>('/news/geopolitical-risks'),
  getEconomicEvents: () =>
    apiClient.get<EconomicEvent[]>('/news/economic-events'),
  getCrisisAlerts: () =>
    apiClient.get<CrisisAlert[]>('/news/crisis-alerts'),
};

// HFT
export const hftApi = {
  getMetrics: () => apiClient.get<HFTMetrics>('/dashboard/hft-metrics'),
  reEnableCircuitBreaker: () =>
    apiClient.post<void>('/risk/hft/circuit-breaker/reset'),
};

// Mistakes
export const mistakesApi = {
  getPatterns: () => apiClient.get<MistakePattern[]>('/dashboard/mistake-patterns'),
};
