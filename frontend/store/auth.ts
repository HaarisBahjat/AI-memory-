/**
 * store/auth.ts — Zustand auth store
 * Persists access_token + refresh_token in localStorage.
 */
import { create } from "zustand";

interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  isLoggedIn: boolean;
  login: (access: string, refresh: string) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  accessToken:  typeof window !== "undefined" ? localStorage.getItem("access_token")  : null,
  refreshToken: typeof window !== "undefined" ? localStorage.getItem("refresh_token") : null,
  isLoggedIn:   typeof window !== "undefined" ? !!localStorage.getItem("access_token") : false,

  login: (access, refresh) => {
    localStorage.setItem("access_token",  access);
    localStorage.setItem("refresh_token", refresh);
    set({ accessToken: access, refreshToken: refresh, isLoggedIn: true });
  },

  logout: () => {
    localStorage.clear();
    set({ accessToken: null, refreshToken: null, isLoggedIn: false });
  },
}));
