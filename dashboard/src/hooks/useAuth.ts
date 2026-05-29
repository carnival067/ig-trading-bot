import { useState, useEffect, useCallback, useRef } from 'react';
import { authApi } from '../api/endpoints';
import type { LoginCredentials, User } from '../types';

const INACTIVITY_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
const TOKEN_KEY = 'auth_token';
const EXPIRY_KEY = 'auth_expiry';
const USER_KEY = 'auth_user';

export function useAuth() {
  const [user, setUser] = useState<User | null>(() => {
    const stored = localStorage.getItem(USER_KEY);
    return stored ? JSON.parse(stored) : null;
  });
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(
    () => !!localStorage.getItem(TOKEN_KEY)
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inactivityTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(EXPIRY_KEY);
    localStorage.removeItem(USER_KEY);
    setUser(null);
    setIsAuthenticated(false);
    if (inactivityTimer.current) {
      clearTimeout(inactivityTimer.current);
    }
  }, []);

  const resetInactivityTimer = useCallback(() => {
    if (inactivityTimer.current) {
      clearTimeout(inactivityTimer.current);
    }
    if (isAuthenticated) {
      inactivityTimer.current = setTimeout(() => {
        logout();
      }, INACTIVITY_TIMEOUT_MS);
    }
  }, [isAuthenticated, logout]);

  const login = useCallback(async (credentials: LoginCredentials) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await authApi.login(credentials);
      const expiry = Date.now() + response.expires_in * 1000;
      localStorage.setItem(TOKEN_KEY, response.access_token);
      localStorage.setItem(EXPIRY_KEY, String(expiry));

      // Decode basic user info from token (assumes JWT payload)
      const payload = JSON.parse(atob(response.access_token.split('.')[1]));
      const userData: User = {
        id: payload.sub,
        username: payload.username || credentials.username,
        role: payload.role || 'trader',
      };
      localStorage.setItem(USER_KEY, JSON.stringify(userData));
      setUser(userData);
      setIsAuthenticated(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Check token expiry on mount
  useEffect(() => {
    const expiry = localStorage.getItem(EXPIRY_KEY);
    if (expiry && Date.now() > Number(expiry)) {
      logout();
    }
  }, [logout]);

  // Set up inactivity listeners
  useEffect(() => {
    if (!isAuthenticated) return;

    const events = ['mousedown', 'keydown', 'scroll', 'touchstart'];
    events.forEach((event) =>
      document.addEventListener(event, resetInactivityTimer)
    );
    resetInactivityTimer();

    return () => {
      events.forEach((event) =>
        document.removeEventListener(event, resetInactivityTimer)
      );
      if (inactivityTimer.current) {
        clearTimeout(inactivityTimer.current);
      }
    };
  }, [isAuthenticated, resetInactivityTimer]);

  return {
    user,
    isAuthenticated,
    isLoading,
    error,
    login,
    logout,
  };
}
