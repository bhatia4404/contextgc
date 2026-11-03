import type { LaneKey } from "./types";

export const LANES: Record<LaneKey, { name: string; short: string; color: string }> = {
  A1: { name: "Raw Ollama, big window (32k)", short: "Raw · 32k", color: "var(--s1)" },
  A2: { name: "Raw Ollama, small window (8k)", short: "Raw · 8k", color: "var(--s2)" },
  B: { name: "ContextGC, small window (8k)", short: "ContextGC · 8k", color: "var(--s3)" },
};
export const LANE_ORDER: LaneKey[] = ["A1", "A2", "B"];

export const TASKS: Record<string, string> = {
  fix_path_join: "Path builder must handle a trailing slash",
  fix_retry_counter: "Retry counter must reset after success",
  repair_json_reader: "Missing optional JSON field should become an empty list",
  fix_csv_parser: "Commas inside quoted CSV fields stay in the field",
  fix_failing_sum: "Fix a failing sum function",
  fix_cache_key: "Cache keys must not collide across namespaces",
  implement_stable_sort: "Implement a stable sort by timestamp",
  preserve_false_values: "Config lookup must keep false, 0 and empty values",
};

export const fmt = (n: number | null | undefined): string =>
  n == null ? "–" : n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e4 ? Math.round(n / 1e3) + "k"
    : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(Math.round(n));
export const pct = (r: number | null | undefined): string => (r == null || isNaN(r) ? "–" : Math.round(r * 100) + "%");
