import type { LaneKey, LiveRun, Showcase } from "./types";

// The backend (deploy/api_app.py) is a separate service; its URL comes from .env.*
const BASE = (import.meta.env.VITE_API_URL as string).replace(/\/$/, "");

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE + path, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((body as { detail?: string }).detail || r.statusText);
  return body as T;
}

export const api = {
  showcase: () => call<Showcase>("/api/showcase"),
  start: (task: string, lanes: LaneKey[]) =>
    call<{ id: string; joined: boolean }>("/api/run", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ task, lanes }),
    }),
  run: (id: string) => call<LiveRun>(`/api/run/${id}`),
  recent: () => call<Pick<LiveRun, "id" | "task" | "status" | "created" | "results">[]>("/api/recent"),
  status: () => call<{ today: number; cap: number; busy: boolean }>("/api/status"),
};
