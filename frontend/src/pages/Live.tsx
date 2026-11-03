import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { LineChart, Swatch } from "../components/charts";
import { LANES, TASKS, fmt } from "../format";
import type { LaneKey, LiveRun, RunStatus } from "../types";

const STATUS: Record<RunStatus, string> = {
  queued: "Waiting for a GPU", starting: "Starting GPU and model", running: "Running", done: "Finished", error: "Failed",
};
const LANE_CHOICES: { key: LaneKey; label: string; on: boolean }[] = [
  { key: "A2", label: "Raw Ollama, 8k window", on: true },
  { key: "B", label: "ContextGC, 8k window", on: true },
  { key: "A1", label: "Raw Ollama, 32k window (slower)", on: false },
];
const finished = (s?: RunStatus) => s === "done" || s === "error";
type Recent = Awaited<ReturnType<typeof api.recent>>;

export default function Live() {
  const [task, setTask] = useState(Object.keys(TASKS)[0]);
  const [lanes, setLanes] = useState<LaneKey[]>(LANE_CHOICES.filter(l => l.on).map(l => l.key));
  const [runId, setRunId] = useState<string | null>(() => new URLSearchParams(location.search).get("run"));
  const [run, setRun] = useState<LiveRun | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [recent, setRecent] = useState<Recent | null>(null);
  const timer = useRef<number>();

  const refreshSide = useCallback(() => {
    api.recent().then(setRecent).catch(() => setRecent([]));
    api.status().then(q => setNote(`${q.today} of ${q.cap} runs used today.` + (q.busy ? " A run is in progress now." : "")))
      .catch(() => { /* the page works without it */ });
  }, []);
  useEffect(refreshSide, [refreshSide]);

  useEffect(() => {
    if (!runId) return;
    history.replaceState(null, "", "?run=" + runId);
    const poll = () => api.run(runId).then(r => {
      setRun(r);
      if (finished(r.status)) { window.clearInterval(timer.current); setBusy(false); refreshSide(); }
    }).catch(e => setNote(String(e.message || e)));
    poll();
    timer.current = window.setInterval(poll, 2500);
    return () => window.clearInterval(timer.current);
  }, [runId, refreshSide]);

  const start = async () => {
    if (!lanes.length) { setNote("Pick at least one lane."); return; }
    setBusy(true);
    try {
      const r = await api.start(task, lanes);
      if (r.joined) setNote("Another run was already going, so you're watching that one.");
      setRun(null);
      setRunId(r.id);
    } catch (e) { setNote(String((e as Error).message)); setBusy(false); }
  };

  return (
    <main>
      <h1>Run it yourself</h1>
      <p className="lead">This starts a real agent on a cloud GPU (NVIDIA T4, model qwen2.5:7b). It works on one coding
        task, once through raw Ollama and once through ContextGC, with the same 8k context window. You can watch every step.</p>
      <p className="callout">The first minute or two is the GPU starting and the model loading. Each lane then takes about
        1–4 minutes. Only one run happens at a time; if someone else's run is going, you'll watch that one.</p>

      <div className="card">
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "end" }}>
          <label style={{ flex: "1 1 280px" }}>Task<br />
            <select style={{ width: "100%" }} value={task} onChange={e => setTask(e.target.value)}>
              {Object.entries(TASKS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </label>
          <button className="primary" disabled={busy} onClick={start}>Start run</button>
        </div>
        <div style={{ marginTop: 12 }}>
          {LANE_CHOICES.map(l => <label key={l.key} className="chk">
            <input type="checkbox" checked={lanes.includes(l.key)}
              onChange={e => setLanes(e.target.checked ? [...lanes, l.key] : lanes.filter(k => k !== l.key))} /> {l.label}
          </label>)}
        </div>
        <p className="muted" style={{ margin: "10px 0 0" }}>{note}</p>
      </div>

      {runId && <RunView run={run} />}

      <h2>Recent runs</h2>
      <div className="card"><div className="tablewrap">
        {recent == null ? <p className="muted">Loading…</p> : !recent.length ? <p className="muted">No runs yet. Start the first one.</p> :
          <table>
            <thead><tr><th>When (UTC)</th><th>Task</th><th>Status</th><th>Raw · 8k</th><th>ContextGC · 8k</th><th>Raw · 32k</th></tr></thead>
            <tbody>{recent.map(r => <tr key={r.id}>
              <td className="l"><a href={"?run=" + r.id} onClick={e => { e.preventDefault(); setRun(null); setRunId(r.id); scrollTo(0, 0); }}>
                {r.created.slice(5, 16).replace("T", " ")}</a></td>
              <td className="l">{TASKS[r.task] || r.task}</td><td>{STATUS[r.status] || r.status}</td>
              {(["A2", "B", "A1"] as LaneKey[]).map(k => <td key={k}>{r.results?.[k]
                ? <span className={r.results[k]!.passed ? "pass" : "fail"}>{r.results[k]!.passed ? "pass" : "fail"}</span> : "–"}</td>)}
            </tr>)}</tbody>
          </table>}
      </div></div>
    </main>
  );
}

function RunView({ run }: { run: LiveRun | null }) {
  if (!run) return <div className="card"><p className="muted">Loading run…</p></div>;
  const withSteps = run.lanes.filter(k => (run.steps[k] || []).length);
  return <>
    <div className="card">
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center", justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }}>{TASKS[run.task] || run.task}</h3>
        <span className="badge">{STATUS[run.status] || run.status}</span>
      </div>
      <p className="muted" style={{ margin: "6px 0 0" }}>{run.message}</p>
    </div>
    <div className="card">
      <h3>Prompt size at each step</h3>
      <p className="muted" style={{ margin: 0 }}>The dashed line is the 8k window. Above it, raw Ollama silently drops the oldest messages.</p>
      {withSteps.length ? <LineChart label="Live prompt tokens per step" xLabel="agent step" minX={12}
        refs={[{ y: 8192, label: "8k window" }]}
        series={withSteps.map(k => ({
          key: k, endLabel: run.results[k] ? (run.results[k]!.passed ? "pass" : "fail") : "",
          points: run.steps[k]!.map(s => ({ x: s.step, y: s.prompt, note: s.tools.join(", ") || (s.error ? "error" : "reply") })),
        }))} /> : <p className="muted">No steps yet.</p>}
    </div>
    <div className="card"><h3>Result</h3><div className="tablewrap">
      {!Object.keys(run.results).length ? <p className="muted">Waiting for the first lane to finish…</p> :
        <table>
          <thead><tr><th>Lane</th><th className="l">Result</th><th>Steps</th><th>Prompt tokens</th><th>Time</th><th>Compactions</th></tr></thead>
          <tbody>{run.lanes.map(k => {
            const r = run.results[k];
            return <tr key={k}><td className="l"><Swatch lane={k} />{LANES[k].name}</td>
              {r ? <><td className="l"><span className={r.passed ? "pass" : "fail"}>{r.passed ? "pass" : "fail"}</span></td>
                <td>{r.steps}</td><td>{fmt(r.prompt_tokens)}</td><td>{r.wall_s}s</td><td>{r.compactions ?? "–"}</td></>
                : <td colSpan={5} className="l muted">{finished(run.status) ? "did not run" : "running…"}</td>}
            </tr>;
          })}</tbody>
        </table>}
    </div></div>
    <div className="card"><details><summary>Step log</summary><div className="log">{run.log.join("\n")}</div></details></div>
  </>;
}
