import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './hooks/useAuth';
import { Layout } from './components/Layout';
import { LoginPage } from './pages/LoginPage';
import { DashboardPage } from './pages/DashboardPage';
import { PerformancePage } from './pages/PerformancePage';
import { TradeHistoryPage } from './pages/TradeHistoryPage';
import { RiskPage } from './pages/RiskPage';
import { CopyTradingPage } from './pages/CopyTradingPage';
import { BacktestPage } from './pages/BacktestPage';
import { NewsPage } from './pages/NewsPage';
import { HFTPage } from './pages/HFTPage';
import { MistakesPage } from './pages/MistakesPage';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          <Route index element={<DashboardPage />} />
          <Route path="performance" element={<PerformancePage />} />
          <Route path="trades" element={<TradeHistoryPage />} />
          <Route path="risk" element={<RiskPage />} />
          <Route path="copy-trading" element={<CopyTradingPage />} />
          <Route path="backtest" element={<BacktestPage />} />
          <Route path="news" element={<NewsPage />} />
          <Route path="hft" element={<HFTPage />} />
          <Route path="mistakes" element={<MistakesPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
