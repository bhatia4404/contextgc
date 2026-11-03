import { useRef, useState, type ReactNode } from "react";
import { LANES, fmt } from "../format";
import type { LaneKey } from "../types";

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}

export function Legend({ keys }: { keys: LaneKey[] }) {
  return (
    <div className="legend">
      {keys.map(k => <span key={k}><i style={{ background: LANES[k].color }} />{LANES[k].name}</span>)}
    </div>
  );
}

export function Swatch({ lane }: { lane: LaneKey }) {
  return <span className="sw" style={{ background: LANES[lane].color }} />;
}

// Tooltip that follows the pointer inside a chart and stays within its width.
function useTip() {
  const host = useRef<HTMLDivElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; body: ReactNode } | null>(null);
  const show = (ev: React.MouseEvent, body: ReactNode) => {
    const hb = host.current!.getBoundingClientRect();
    setTip({ x: ev.clientX - hb.left, y: ev.clientY - hb.top, body });
  };
  const hide = () => setTip(null);
  const w = host.current?.clientWidth ?? 0, tw = tipRef.current?.offsetWidth ?? 160, th = tipRef.current?.offsetHeight ?? 40;
  const node = tip && (
    <div ref={tipRef} className="tip" style={{
      display: "block", left: Math.max(0, Math.min(w - tw, tip.x + 12)), top: Math.max(0, tip.y - th - 10),
    }}>{tip.body}</div>
  );
  return { host, show, hide, node };
}

export interface LineSeries {
  key: LaneKey;
  endLabel?: string;
  points: { x: number; y: number | null; note?: string }[];
}

// Prompt tokens per agent step, one line per lane, crosshair tooltip.
export function LineChart({ series, refs = [], minX = 1, xLabel = "", label, height = 240 }: {
  series: LineSeries[]; refs?: { y: number; label: string }[]; minX?: number; xLabel?: string; label: string; height?: number;
}) {
  const { host, show, hide, node } = useTip();
  const [cross, setCross] = useState<number | null>(null);
  const W = 640, H = height, L = 46, R = 16, T = 12, B = xLabel ? 46 : 30;
  const maxX = Math.max(minX, ...series.map(s => s.points.length));
  const maxY = niceMax(Math.max(...refs.map(r => r.y), ...series.flatMap(s => s.points.map(p => p.y ?? 0))) * 1.05);
  const x = (i: number) => L + (maxX <= 1 ? 0 : (i - 1) / (maxX - 1)) * (W - L - R);
  const y = (v: number) => T + (1 - v / maxY) * (H - T - B);
  const svgRef = useRef<SVGSVGElement>(null);

  const onMove = (ev: React.MouseEvent) => {
    const box = svgRef.current!.getBoundingClientRect(), sx = (ev.clientX - box.left) * W / box.width;
    const i = Math.max(1, Math.min(maxX, Math.round(1 + (sx - L) / (W - L - R) * (maxX - 1))));
    setCross(i);
    show(ev, <>
      <div className="k">Step {i}</div>
      {series.map(s => {
        const p = s.points.find(p => p.x === i);
        return p ? <div key={s.key}><Swatch lane={s.key} /><span className="k">{LANES[s.key].short}:</span>{" "}
          <b>{p.y == null ? "error" : fmt(p.y) + " tok"}</b>{p.note ? " · " + p.note : ""}</div> : null;
      })}
    </>);
  };

  return (
    <div className="chart" ref={host}>
      <Legend keys={series.map(s => s.key)} />
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
        {[0, 1, 2, 3, 4].map(i => {
          const v = maxY * i / 4;
          return <g key={i}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--grid)" strokeWidth={1} />
            <text x={L - 6} y={y(v) + 4} textAnchor="end">{fmt(v)}</text>
          </g>;
        })}
        {Array.from({ length: maxX }, (_, k) => k + 1).filter(i => maxX <= 14 || i % 2 === 1).map(i =>
          <text key={i} x={x(i)} y={H - B + 18} textAnchor="middle">{i}</text>)}
        {xLabel && <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle">{xLabel}</text>}
        {refs.map(r => <g key={r.label}>
          <line x1={L} x2={W - R} y1={y(r.y)} y2={y(r.y)} stroke="var(--axis)" strokeWidth={1} strokeDasharray="4 4" />
          <text x={W - R} y={y(r.y) - 5} textAnchor="end" className="lbl">{r.label}</text>
        </g>)}
        {series.map(s => {
          const pts = s.points.filter(p => p.y != null) as { x: number; y: number }[];
          if (!pts.length) return null;
          const last = pts[pts.length - 1];
          return <g key={s.key}>
            <path d={pts.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join("")} fill="none"
                  stroke={LANES[s.key].color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
            <circle cx={x(last.x)} cy={y(last.y)} r={4} fill={LANES[s.key].color} stroke="var(--surface)" strokeWidth={2} />
            {s.endLabel && <text x={Math.min(x(last.x) + 8, W - R)} y={y(last.y) + 4} className="lbl">{s.endLabel}</text>}
          </g>;
        })}
        {cross != null && <line x1={x(cross)} x2={x(cross)} y1={T} y2={H - B} stroke="var(--axis)" strokeWidth={1} />}
        <rect x={L} y={T} width={W - L - R} height={H - T - B} fill="transparent"
              onMouseMove={onMove} onMouseLeave={() => { hide(); setCross(null); }} />
      </svg>
      {node}
    </div>
  );
}

export interface BarGroup { label: string; bars: { key: LaneKey; value: number; text: string; tip: string }[] }

// Grouped vertical bars: groups on x (models), one bar per lane.
export function GroupedBars({ groups, max, tick = fmt, label, height = 240 }: {
  groups: BarGroup[]; max?: number; tick?: (v: number) => string; label: string; height?: number;
}) {
  const { host, show, hide, node } = useTip();
  const keys = [...new Set(groups.flatMap(g => g.bars.map(b => b.key)))];
  const W = 640, H = height, L = 42, R = 8, T = 18, B = 34;
  const maxY = max ?? niceMax(Math.max(...groups.flatMap(g => g.bars.map(b => b.value))));
  const y = (v: number) => T + (1 - v / maxY) * (H - T - B);
  const gw = (W - L - R) / Math.max(1, groups.length), bw = Math.min(44, (gw - 24) / keys.length - 2);
  return (
    <div className="chart" ref={host}>
      <Legend keys={keys} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
        {[0, 1, 2, 3, 4].map(i => {
          const v = maxY * i / 4;
          return <g key={i}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--grid)" strokeWidth={1} />
            <text x={L - 6} y={y(v) + 4} textAnchor="end">{tick(v)}</text>
          </g>;
        })}
        {groups.map((g, gi) => {
          const x0 = L + gi * gw + (gw - keys.length * (bw + 2)) / 2;
          return <g key={g.label}>
            <text x={L + gi * gw + gw / 2} y={H - 12} textAnchor="middle" className="lbl">{g.label}</text>
            {g.bars.map(b => {
              const xi = x0 + keys.indexOf(b.key) * (bw + 2), top = y(b.value), h = Math.max(0, y(0) - top);
              const r = Math.min(4, h / 2, bw / 2);
              return <g key={b.key}>
                <path fill={LANES[b.key].color} d={`M${xi},${y(0)}V${top + r}Q${xi},${top} ${xi + r},${top}H${xi + bw - r}` +
                  `Q${xi + bw},${top} ${xi + bw},${top + r}V${y(0)}Z`} />
                <text x={xi + bw / 2} y={top - 5} textAnchor="middle" className="lbl">{b.text}</text>
                <rect x={xi - 1} y={T} width={bw + 2} height={H - T - B} fill="transparent"
                      onMouseMove={ev => show(ev, <><div className="k">{g.label}</div>
                        <div><Swatch lane={b.key} />{LANES[b.key].name}</div><div><b>{b.tip}</b></div></>)}
                      onMouseLeave={hide} />
              </g>;
            })}
          </g>;
        })}
      </svg>
      {node}
    </div>
  );
}

export interface HBarRow { label: string; value: number; text: string; tip?: string; color?: string }

// Horizontal bars for one measure (single series: one hue, labels carry the values).
export function HBars({ rows, max, label, labelWidth = 190 }: { rows: HBarRow[]; max?: number; label: string; labelWidth?: number }) {
  const { host, show, hide, node } = useTip();
  const W = 640, rowH = 30, L = labelWidth, R = 60, H = rows.length * rowH + 8;
  const top = max ?? niceMax(Math.max(...rows.map(r => r.value)));
  return (
    <div className="chart" ref={host}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
        {rows.map((r, i) => {
          const yy = 4 + i * rowH, w = Math.max(2, (r.value / top) * (W - L - R)), bh = rowH - 10;
          return <g key={r.label}>
            <text x={L - 10} y={yy + bh / 2 + 4} textAnchor="end" className="lbl">{r.label}</text>
            <path fill={r.color ?? "var(--s1)"} d={`M${L},${yy}H${L + w - 4}Q${L + w},${yy} ${L + w},${yy + 4}` +
              `V${yy + bh - 4}Q${L + w},${yy + bh} ${L + w - 4},${yy + bh}H${L}Z`} />
            <text x={L + w + 6} y={yy + bh / 2 + 4} className="lbl">{r.text}</text>
            <rect x={0} y={yy - 3} width={W} height={rowH} fill="transparent"
                  onMouseMove={ev => show(ev, <><div className="k">{r.label}</div><div><b>{r.tip ?? r.text}</b></div></>)}
                  onMouseLeave={hide} />
          </g>;
        })}
      </svg>
      {node}
    </div>
  );
}
