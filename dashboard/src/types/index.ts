// Authentication
export interface LoginCredentials {
  username: string;
  password: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  id: string;
  username: string;
  role: string;
}

// Dashboard
export interface DashboardMetrics {
  pnl: PnLMetrics;
  win_rate: number;
  drawdown: number;
  open_positions: Position[];
  ai_confidence: number;
  market_regime: MarketRegime;
}

export interface PnLMetrics {
  daily: number;
  weekly: number;
  monthly: number;
  all_time: number;
}

export type MarketRegime = 'trending_up' | 'trending_down' | 'ranging' | 'volatile' | 'unknown';

// Positions & Trades
export interface Position {
  id: string;
  instrument: string;
  direction: 'long' | 'short';
  size: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  strategy: string;
  opened_at: string;
}

export interface Trade {
  id: string;
  instrument: string;
  direction: 'long' | 'short';
  size: number;
  entry_price: number;
  exit_price: number;
  pnl: number;
  strategy: string;
  confidence_score: number;
  duration_seconds: number;
  opened_at: string;
  closed_at: string;
  outcome: 'win' | 'loss' | 'breakeven';
}

export interface TradeFilter {
  strategy?: string;
  instrument?: string;
  date_from?: string;
  date_to?: string;
  outcome?: 'win' | 'loss' | 'breakeven';
  page?: number;
  page_size?: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// Performance Analytics
export interface EquityPoint {
  timestamp: string;
  equity: number;
  drawdown: number;
}

export interface MonthlyReturn {
  year: number;
  month: number;
  return_pct: number;
}

export interface StrategyComparison {
  strategy: string;
  total_pnl: number;
  win_rate: number;
  sharpe_ratio: number;
  max_drawdown: number;
  trade_count: number;
}

// Risk Management
export interface ExposureByClass {
  asset_class: string;
  exposure_pct: number;
  position_count: number;
}

export interface CorrelationEntry {
  instrument_a: string;
  instrument_b: string;
  correlation: number;
}

export interface RiskMetrics {
  exposures: ExposureByClass[];
  correlations: CorrelationEntry[];
  var_95_1day: number;
  kill_switch_active: boolean;
}

// Copy Trading
export interface CopyTrader {
  id: string;
  name: string;
  win_rate: number;
  total_pnl: number;
  sharpe_ratio: number;
  max_drawdown: number;
  followers: number;
  allocation_pct: number;
}

export interface CopyTraderPerformance {
  trader_id: string;
  equity_curve: EquityPoint[];
}

// Backtesting
export interface BacktestConfig {
  strategy: string;
  parameters: Record<string, number | string | boolean>;
  start_date: string;
  end_date: string;
  initial_capital: number;
}

export interface BacktestResult {
  id: string;
  strategy: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  trade_count: number;
  equity_curve: EquityPoint[];
  monte_carlo: MonteCarloResult;
  walk_forward: WalkForwardResult[];
}

export interface MonteCarloResult {
  percentiles: { p5: number; p25: number; p50: number; p75: number; p95: number };
  distribution: number[];
}

export interface WalkForwardResult {
  period: string;
  in_sample_return: number;
  out_sample_return: number;
}

// News
export interface NewsItem {
  id: string;
  title: string;
  source: string;
  published_at: string;
  sentiment_score: number;
  impact: 'high' | 'medium' | 'low';
  correlated_instruments: string[];
}

export interface GeopoliticalRisk {
  region: string;
  risk_score: number;
  trend: 'rising' | 'stable' | 'falling';
}

export interface EconomicEvent {
  id: string;
  title: string;
  scheduled_at: string;
  impact: 'high' | 'medium' | 'low';
  currency: string;
  previous: string;
  forecast: string;
}

export interface CrisisAlert {
  id: string;
  title: string;
  severity: 'critical' | 'high' | 'medium';
  region: string;
  started_at: string;
  affected_instruments: string[];
}

// HFT
export interface HFTMetrics {
  orders_per_second: number;
  avg_latency_ms: number;
  net_pnl_1min: number;
  net_pnl_5min: number;
  net_pnl_daily: number;
  circuit_breaker_active: boolean;
  total_exposure_pct: number;
}

// Mistake Patterns
export interface MistakePattern {
  id: string;
  classification: string;
  loss_count: number;
  total_loss: number;
  last_occurrence: string;
  penalty_level: number;
  resolution_progress: number;
}

// WebSocket
export interface WSMessage {
  type: string;
  data: unknown;
  timestamp: string;
}
