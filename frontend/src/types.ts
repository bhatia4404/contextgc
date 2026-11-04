export type LaneKey = "A1" | "A2" | "B";

export interface LaneStats {
  runs: number;
  passed: number;
  rate: number | null;
  prompt_tokens: number | null;
  wall_s: number | null;
  reloads: number | null;
  fidelity_fallbacks: number | null;
  compactions: number | null;
}

export interface Step { step: number; prompt: number | null; tools: string[]; error: boolean }

export interface Campaign {
  id: string;
  model: string;
  gpu: string;
  windows: { big: number | null; small: number | null };
  repeats: number;
  summary: Partial<Record<LaneKey, LaneStats>>;
  per_task: Record<string, Partial<Record<LaneKey, LaneStats>>>;
}

export interface Highlight {
  campaign: string;
  model: string;
  gpu: string;
  task: string;
  lanes: Partial<Record<LaneKey, LaneStats>>;
  traces: Partial<Record<LaneKey, { passed: boolean; repeat: number; window: number; steps: Step[] }>>;
}

export interface CrewStats {
  runs: number; passed: number; prompt_tokens: number; gc_saved_pct_avg: number | null; timeouts: number;
}

export interface HostedLane {
  runs: number; passed: number; prompt_tokens_per_run: number; completion_tokens_per_run: number;
  cost_per_run: number; time_per_run: number; cost_total: number;
}

export interface HostedApi {
  provider: string; model: string; repeats: number; budget: number;
  lanes: { raw: HostedLane; contextgc: HostedLane };
  per_task: Record<string, Partial<Record<"raw" | "contextgc", { passed: number; runs: number }>>>;
}

export interface Showcase {
  generated: string;
  repo: string;
  campaigns: Campaign[];
  highlights: Highlight[];
  ablation: { id: string; model: string; gpu: string; repeats: number; summary: Record<string, LaneStats> } | null;
  crewai: { id: string; model: string; gpu: string; summary: Record<string, CrewStats> } | null;
  hosted_api?: HostedApi | null;
}

export type RunStatus = "queued" | "starting" | "running" | "done" | "error";

export interface LaneResult {
  passed: boolean; steps: number; wall_s: number; prompt_tokens: number | null; compactions: number | null;
}

export interface LiveRun {
  id: string;
  task: string;
  lanes: LaneKey[];
  status: RunStatus;
  created: string;
  message: string;
  steps: Partial<Record<LaneKey, Step[]>>;
  results: Partial<Record<LaneKey, LaneResult>>;
  log: string[];
}
