import { useEffect, useState } from "react";
import { api } from "../api";
import { GroupedBars, HBars, LineChart, Swatch } from "../components/charts";
import { LANES, LANE_ORDER, TASKS, fmt, pct } from "../format";
import type { Campaign, Highlight, LaneKey, LaneStats, Showcase } from "../types";

const passText = (s?: LaneStats) => (s ? `${s.passed}/${s.runs}` : "–");
const lanesOf = (x: { summary: Partial<Record<LaneKey, LaneStats>> }) => LANE_ORDER.filter(k => x.summary[k]);

export default function Results() {
  const [data, setData] = useState<Showcase | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.showcase().then(setData).catch(e => setError(String(e.message || e))); }, []);

  return (
    <main>
      <h1>Small models, small windows, still on task.</h1>
      <p className="lead">ContextGC is a proxy between an AI agent and Ollama. Before every model call it cleans the
        agent's history: duplicates go, huge tool outputs move to disk behind a pointer, and old low-value turns are
        evicted only under pressure. The agent needs no code changes, only a new URL.</p>
      {error && <p className="callout">Could not load results: {error}</p>}
      {!data && !error && <p className="muted">Loading results…</p>}
      {data && <Tiles data={data} />}

      <h2>The problem, and what ContextGC does</h2>
      <div className="steps">
        <div><h3>Agents resend everything</h3>Every step sends the whole history again: file reads, logs, test
          output. It grows past the model's context window within a few steps.</div>
        <div><h3>Ollama cuts silently</h3>When the prompt is too long, the oldest messages are dropped without
          warning. The oldest message is the task itself, so the agent forgets what it was asked to do.</div>
        <div><h3>ContextGC cleans first</h3>It keeps the task and recent work, moves bulky tool output behind a short
          pointer, and drops old filler, so the prompt fits and nothing important is cut.</div>
      </div>

      {data && <>
        <h2>Three highlighted tasks</h2>
        <p className="sub">Picked by a fixed rule, not by hand: the tasks where ContextGC's pass rate beat the raw small
          window by the most, then the raw big window, then token savings. Every other task is in the full grid below,
          including the ones where ContextGC lost.</p>
        {data.highlights.map((h, i) => <HighlightCard key={h.task + h.model} h={h} n={i + 1} />)}

        <h2>All results</h2>
        <p className="sub">The same agent, tasks and settings in three lanes. Each task ran several times per lane; a
          pass means an independent pytest run passed and the tests were not edited.</p>
        <ModelCharts campaigns={data.campaigns} />
        {data.campaigns.map(c => <CampaignTable key={c.id} c={c} />)}

        <h2>Which parts of ContextGC matter</h2>
        <p className="sub">Each variant switches off exactly one stage of the pipeline and reruns every task. A big drop
          means that stage is doing real work.</p>
        <Ablation data={data} />

        <h2>CrewAI, a real agent framework</h2>
        <p className="sub">Two CrewAI crews (incident response, code maintenance) on a demo repository with a 2,000-line
          outage log. CrewAI 1.15 has its own context handling: it parks large tool results in files. We ran it both
          with that feature on and with it off, each against raw Ollama and through ContextGC.</p>
        <Crew data={data} />

        <h2>How to read these numbers</h2>
        <div className="card">
          <ul>
            <li>Same model, prompt, tools, temperature 0 and seed in every lane. Only the context handling differs.</li>
            <li>Even at temperature 0, identical runs on a GPU can end differently. Treat gaps of one or two runs as noise.</li>
            <li>The 8 tasks are small Python repair jobs we wrote, with earlier-session noise added to the history.
              Results on other workloads may differ.</li>
            <li>"Raw, big window" never ran out of room here, so it is what running the model normally looks like.</li>
          </ul>
          <p className="muted">Data generated {data.generated}.</p>
        </div>
      </>}
    </main>
  );
}

function Tiles({ data }: { data: Showcase }) {
  const c = data.campaigns;
  const sum = (k: LaneKey, f: "passed" | "runs" | "prompt_tokens") => c.reduce((a, x) => a + (x.summary[k]?.[f] || 0), 0);
  const all = c.reduce((a, x) => a + Object.values(x.summary).reduce((b, s) => b + (s?.runs || 0), 0), 0);
  const tiles: [string, string, string][] = [
    ["ContextGC, 8k window", pct(sum("B", "passed") / sum("B", "runs")), `${sum("B", "passed")} of ${sum("B", "runs")} runs passed`],
    ["Raw Ollama, same 8k window", pct(sum("A2", "passed") / sum("A2", "runs")), `${sum("A2", "passed")} of ${sum("A2", "runs")} runs passed`],
    ["Raw Ollama, 32k window", pct(sum("A1", "passed") / sum("A1", "runs")), `${sum("A1", "passed")} of ${sum("A1", "runs")} runs passed`],
    ["Tokens vs 32k window", pct(sum("B", "prompt_tokens") / sum("A1", "prompt_tokens")),
      `${fmt(sum("B", "prompt_tokens"))} vs ${fmt(sum("A1", "prompt_tokens"))} prompt tokens`],
  ];
  return <>
    <div className="tiles">
      {tiles.map(([l, v, n]) => <div className="tile" key={l}>
        <div className="label">{l}</div><div className="value">{v}</div><div className="note">{n}</div></div>)}
    </div>
    <p className="muted">Pass rates across {c.length} model{c.length > 1 ? "s" : ""} ({c.map(x => x.model).join(", ")}),
      8 coding tasks, {all} agent runs in total.</p>
  </>;
}

function story(h: Highlight): string {
  const a2 = h.traces.A2, b = h.traces.B;
  const over = a2?.steps.some(s => (s.prompt || 0) >= 7800);
  return (over ? "The raw 8k run hit the window, so Ollama cut the oldest messages, starting with the task. " : "") +
    (b ? `Through ContextGC the prompt stayed under the window and the run ${b.passed ? "passed" : "failed"} in ${b.steps.length} steps.` : "");
}

function HighlightCard({ h, n }: { h: Highlight; n: number }) {
  const lanes = LANE_ORDER.filter(k => h.lanes[k]);
  return (
    <div className="card">
      <h3>{n}. {TASKS[h.task] || h.task}</h3>
      <p className="muted" style={{ margin: "0 0 10px" }}>{h.model} on {h.gpu} · task <code>{h.task}</code></p>
      <div className="grid2">
        <div className="tablewrap">
          <table>
            <thead><tr><th>Lane</th><th>Passed</th><th>Tokens / run</th><th>Steps (shown run)</th></tr></thead>
            <tbody>{lanes.map(k => {
              const s = h.lanes[k]!, tr = h.traces[k];
              return <tr key={k}><td className="l"><Swatch lane={k} />{LANES[k].name}</td><td>{passText(s)}</td>
                <td>{fmt((s.prompt_tokens || 0) / s.runs)}</td><td>{tr ? tr.steps.length : "–"}</td></tr>;
            })}</tbody>
          </table>
          <p className="muted">{story(h)}</p>
        </div>
        <div>
          <div className="muted" style={{ fontSize: 13 }}>Prompt size at each step of one run per lane</div>
          <LineChart label={`Prompt tokens per step for ${h.task}`} xLabel="agent step"
            refs={[{ y: 8192, label: "8k window" }]}
            series={LANE_ORDER.filter(k => h.traces[k]).map(k => ({
              key: k, endLabel: h.traces[k]!.passed ? "pass" : "fail",
              points: h.traces[k]!.steps.map(s => ({ x: s.step, y: s.prompt, note: s.tools.join(", ") || (s.error ? "error" : "reply") })),
            }))} />
        </div>
      </div>
    </div>
  );
}

function ModelCharts({ campaigns }: { campaigns: Campaign[] }) {
  return <>
    <div className="card"><h3>Pass rate by model</h3>
      <GroupedBars label="Pass rate per lane and model" max={1} tick={pct} groups={campaigns.map(x => ({
        label: `${x.model} · ${x.gpu}`,
        bars: lanesOf(x).map(k => ({ key: k, value: x.summary[k]!.rate ?? 0, text: pct(x.summary[k]!.rate),
          tip: `${x.summary[k]!.passed} of ${x.summary[k]!.runs} runs passed` })),
      }))} />
    </div>
    <div className="card"><h3>Prompt tokens processed by model</h3>
      <p className="muted" style={{ margin: 0 }}>All runs together. Fewer tokens means less GPU work per task.</p>
      <GroupedBars label="Prompt tokens per lane and model" groups={campaigns.map(x => ({
        label: `${x.model} · ${x.gpu}`,
        bars: lanesOf(x).map(k => ({ key: k, value: x.summary[k]!.prompt_tokens ?? 0, text: fmt(x.summary[k]!.prompt_tokens),
          tip: `${fmt(x.summary[k]!.prompt_tokens)} prompt tokens over ${x.summary[k]!.runs} runs` })),
      }))} />
    </div>
  </>;
}

function Cell({ s }: { s?: LaneStats }) {
  if (!s) return <td>–</td>;
  const r = s.passed / s.runs;
  const bg = r >= 0.67 ? "rgba(12,163,12,0.16)" : r > 0 ? "rgba(250,178,25,0.20)" : "rgba(208,59,59,0.14)";
  return <td><span className="cell" style={{ background: bg }}>{s.passed}/{s.runs}</span></td>;
}

function CampaignTable({ c }: { c: Campaign }) {
  const lanes = lanesOf(c);
  return (
    <div className="card">
      <h3>{c.model} on {c.gpu}</h3>
      <p className="muted" style={{ margin: "0 0 8px" }}>Windows: big {fmt(c.windows.big)}, small {fmt(c.windows.small)} tokens ·{" "}
        {c.repeats} repeat{c.repeats > 1 ? "s" : ""} per task and lane</p>
      <div className="tablewrap"><table>
        <thead><tr><th>Task</th>{lanes.map(k => <th key={k}><Swatch lane={k} />{LANES[k].short}</th>)}</tr></thead>
        <tbody>
          {Object.entries(c.per_task).map(([t, l]) =>
            <tr key={t}><td className="l">{TASKS[t] || t}</td>{lanes.map(k => <Cell key={k} s={l[k]} />)}</tr>)}
          <tr><td className="l"><b>All tasks</b></td>{lanes.map(k => <td key={k}><b>{passText(c.summary[k])}</b></td>)}</tr>
          <tr><td className="l muted">Prompt tokens</td>{lanes.map(k => <td key={k} className="muted">{fmt(c.summary[k]!.prompt_tokens)}</td>)}</tr>
          <tr><td className="l muted">Model reloads</td>{lanes.map(k => <td key={k} className="muted">{c.summary[k]!.reloads ?? "–"}</td>)}</tr>
        </tbody>
      </table></div>
    </div>
  );
}

const STAGE: Record<string, string> = {
  B: "Full ContextGC", "B:no_semantic": "without semantic merging", "B:no_state": "without the state anchor",
  "B:no_fidelity": "without the loop detector", "B:no_evict": "without eviction", "B:no_offload": "without offload",
  "B:no_dedup": "without dedup", "B:no_reinject": "without re-inject",
};

function Ablation({ data }: { data: Showcase }) {
  const a = data.ablation;
  if (!a) return <div className="card"><p className="muted">Not run yet.</p></div>;
  return (
    <div className="card">
      <HBars label="Pass rate with one stage switched off" max={1} rows={Object.entries(a.summary).map(([k, s]) => ({
        label: STAGE[k] || k, value: s.rate || 0, text: `${pct(s.rate)} · ${s.passed}/${s.runs}`,
        color: k === "B" ? "var(--s3)" : "var(--s1)",
        tip: `${s.passed} of ${s.runs} runs passed · ${fmt(s.prompt_tokens)} prompt tokens`,
      }))} />
      <p className="muted">{a.model} on {a.gpu}, all 8 tasks, {a.repeats} repeats per variant. Green is the full pipeline.</p>
    </div>
  );
}

function Crew({ data }: { data: Showcase }) {
  const c = data.crewai;
  if (!c) return <div className="card"><p className="muted">Not run yet.</p></div>;
  const name = (l: string) => (l.startsWith("raw") ? "Raw Ollama" : "ContextGC") +
    (l.endsWith("-nocrew") ? ", CrewAI's own handling off" : ", CrewAI's own handling on");
  return (
    <div className="card">
      <div className="tablewrap"><table>
        <thead><tr><th>Crew</th><th className="l">Setup</th><th>Passed</th><th>Prompt tokens</th><th>ContextGC saved</th><th>Timeouts</th></tr></thead>
        <tbody>{Object.entries(c.summary).map(([k, s]) => {
          const [agent, lane] = k.split("/");
          return <tr key={k}><td className="l">{agent === "incident" ? "Incident response" : "Code maintenance"}</td>
            <td className="l">{name(lane)}</td><td>{s.passed}/{s.runs}</td><td>{fmt(s.prompt_tokens)}</td>
            <td>{s.gc_saved_pct_avg == null ? "–" : s.gc_saved_pct_avg + "%"}</td><td>{s.timeouts ?? 0}</td></tr>;
        })}</tbody>
      </table></div>
      <p className="muted">{c.model} on {c.gpu}, 8k window in every setup.</p>
    </div>
  );
}
