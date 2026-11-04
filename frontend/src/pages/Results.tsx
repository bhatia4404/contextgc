import { useEffect, useState } from "react";
import { api } from "../api";
import { GroupedBars, HBars, LineChart, Swatch } from "../components/charts";
import { LANES, LANE_ORDER, TASKS, fmt, pct } from "../format";
import type { Campaign, HostedApi, Highlight, LaneKey, LaneStats, Showcase } from "../types";

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

        {data.hosted_api && <HostedApiSection h={data.hosted_api} />}

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
  // Headline: the first campaign (the model ContextGC was tuned on); every model is summarized right below.
  const c = data.campaigns[0], sm = c.summary;
  const rate = (k: LaneKey) => (sm[k] ? sm[k]!.passed / sm[k]!.runs : null);
  const tiles: [string, string, string][] = [
    ["ContextGC, 8k window", pct(rate("B")), `${passText(sm.B)} runs passed`],
    ["Raw Ollama, same 8k window", pct(rate("A2")), `${passText(sm.A2)} runs passed`],
    ["Raw Ollama, 32k window", pct(rate("A1")), `${passText(sm.A1)} runs passed`],
    ["Tokens vs 32k window", pct((sm.B?.prompt_tokens || 0) / (sm.A1?.prompt_tokens || 1)),
      `${fmt(sm.B?.prompt_tokens)} vs ${fmt(sm.A1?.prompt_tokens)} prompt tokens`],
  ];
  const others = data.campaigns.slice(1);
  return <>
    <div className="tiles">
      {tiles.map(([l, v, n]) => <div className="tile" key={l}>
        <div className="label">{l}</div><div className="value">{v}</div><div className="note">{n}</div></div>)}
    </div>
    <p className="muted">{c.model} on {c.gpu}: 8 coding tasks, {c.repeats} runs each per lane. This is the model where
      ContextGC helps most.</p>
    {others.length > 0 && <div className="callout">
      <b>Across models the gain is not universal.</b>{" "}
      {data.campaigns.map(x => `${x.model}: ContextGC ${passText(x.summary.B)}, raw 8k ${passText(x.summary.A2)}, ` +
        `raw 32k ${passText(x.summary.A1)}`).join(" · ")}.{" "}
      On the stronger and the non-Qwen model, simply losing the old history (raw 8k) worked better than ContextGC's
      curated history. ContextGC used about a third of the 32k window's prompt tokens on every model.
    </div>}
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
      <p className="muted">{c.model} on {c.gpu}, 8k window in every setup. Prompts stayed small in every setup: the
        crews' own file tools return at most the last 16k characters, so the window never filled and ContextGC had
        little to remove (0–3%). This test does not stress the context window; the coding benchmark above does.</p>
    </div>
  );
}

function HostedApiSection({ h }: { h: HostedApi }) {
  const { raw, contextgc: gc } = h.lanes;
  const less = (a: number, b: number) => Math.round((1 - b / a) * 100);
  const rows: { label: string; fmtv: (v: number) => string; key: "prompt_tokens_per_run" | "cost_per_run" | "time_per_run" }[] = [
    { label: "Prompt tokens per run", fmtv: fmt, key: "prompt_tokens_per_run" },
    { label: "Cost per run", fmtv: v => "$" + v.toFixed(4), key: "cost_per_run" },
    { label: "Time per run", fmtv: v => v.toFixed(1) + " s", key: "time_per_run" },
  ];
  return <>
    <h2>Hosted model: {h.provider}</h2>
    <p className="sub">The same 8 tasks on Meta's <code>{h.model}</code>, {h.repeats} runs each: straight to the API with the
      whole history, and through ContextGC with an {Math.round(h.budget / 1024)}k-token prompt budget. This model has a 1-million-token
      window, so it never loses the task and ContextGC cannot improve success here. What it changes is how much is sent
      and billed.</p>
    <div className="tiles">
      <div className="tile"><div className="label">Same success</div><div className="value">{gc.passed}/{gc.runs}</div>
        <div className="note">raw API {raw.passed}/{raw.runs}</div></div>
      <div className="tile"><div className="label">Fewer prompt tokens</div><div className="value">−{less(raw.prompt_tokens_per_run, gc.prompt_tokens_per_run)}%</div>
        <div className="note">{fmt(gc.prompt_tokens_per_run)} vs {fmt(raw.prompt_tokens_per_run)} per run</div></div>
      <div className="tile"><div className="label">Lower cost</div><div className="value">−{less(raw.cost_per_run, gc.cost_per_run)}%</div>
        <div className="note">${gc.cost_per_run.toFixed(4)} vs ${raw.cost_per_run.toFixed(4)} per run</div></div>
      <div className="tile"><div className="label">Faster</div><div className="value">−{less(raw.time_per_run, gc.time_per_run)}%</div>
        <div className="note">{gc.time_per_run.toFixed(1)} s vs {raw.time_per_run.toFixed(1)} s per run</div></div>
    </div>
    <div className="card">
      {rows.map(r => <div key={r.key} style={{ marginBottom: 6 }}>
        <h3 style={{ marginBottom: 0 }}>{r.label}</h3>
        <HBars label={r.label} labelWidth={170} max={Math.max(raw[r.key], gc[r.key]) * 1.15} rows={[
          { label: "Raw API", value: raw[r.key], text: r.fmtv(raw[r.key]), color: "var(--s1)" },
          { label: "Through ContextGC", value: gc[r.key], text: r.fmtv(gc[r.key]), color: "var(--s3)" },
        ]} />
      </div>)}
      <p className="muted">The cost saving is smaller than the token saving: Meta bills repeated prompt prefixes at a much
        lower cached rate, and ContextGC's rewritten history loses part of that discount. Measured offline with our own API
        key (the whole comparison cost ${(raw.cost_total + gc.cost_total).toFixed(2)}); this model is not available on the
        Run it live page.</p>
    </div>
  </>;
}
