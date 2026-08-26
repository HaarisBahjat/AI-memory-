/**
 * lib/api.ts — Axios client wired to FastAPI backend
 * Base URL: http://localhost:8000/api/v1 (from NEXT_PUBLIC_API_URL)
 * Auto-injects JWT Bearer token from Zustand store on every request.
 * Auto-refreshes token on 401 and retries once.
 */
import axios from "axios";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

export const api = axios.create({ baseURL: BASE, timeout: 30_000 });

/** Attach token from localStorage on every request */
api.interceptors.request.use((config: any) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("access_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/** On 401 → try token refresh, then retry once */
api.interceptors.response.use(
  (res: any) => res,
  async (err: any) => {
    const original = err.config;
    if (err.response?.status === 401 && !original._retry) {
      original._retry = true;
      const refresh = localStorage.getItem("refresh_token");
      if (refresh) {
        try {
          const { data } = await axios.post(`${BASE}/auth/refresh`, { refresh_token: refresh });
          localStorage.setItem("access_token", data.access_token);
          original.headers.Authorization = `Bearer ${data.access_token}`;
          return api(original);
        } catch {
          localStorage.clear();
          window.location.href = "/login";
        }
      }
    }
    return Promise.reject(err);
  }
);

/* ── Typed API methods ───────────────────────────────────────── */
export const authApi = {
  register: (email: string, password: string) =>
    api.post("/auth/register", { email, password }),
  login: (email: string, password: string) =>
    api.post("/auth/login", { email, password }),
  logout: (refresh_token: string) =>
    api.post("/auth/logout", { refresh_token }),
};

export const chatApi = {
  send: (message: string) =>
    api.post<{ response: string; memories_used: number; debug: Record<string, number> }>("/chat", { message }),
  endSession: () => api.post("/session/end"),
};

export const memoriesApi = {
  list: () => api.get<{ memories: Memory[] }>("/memories"),
  pin: (id: string, pinned: boolean) => api.patch(`/memories/${id}/pin`, { pinned: pinned }),
  delete: (id: string) => api.delete(`/memories/${id}`),
};

export const episodesApi = {
  list: () => api.get<{ episodes: Episode[] }>("/episodes"),
};

export const adminApi = {
  triggerConsolidate: () => api.post("/system/consolidate"),
  consolidateStatus: () => api.get("/system/consolidation/status"),
  triageEvents: () => api.get("/triage"),
};

/* ── Shared types ────────────────────────────────────────────── */
export interface Memory {
  id: string;
  category: string;
  text: string;
  reinforcement_count: number;
  is_pinned: boolean;
  created_at: string;
  last_reinforced_at: string;
}

export interface Episode {
  id: string;
  timestamp: string;
  session_summary: string;
  extracted_metrics: {
    moodScore?: number;
    primaryStressor?: string;
    sleepHoursLogged?: number;
  };
}
