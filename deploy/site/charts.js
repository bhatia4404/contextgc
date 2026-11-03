// Small dependency-free SVG charts shared by the showcase and live pages.
const LANES = {
  A1: { name: "Raw Ollama, big window (32k)", short: "Raw · 32k", color: "var(--s1)" },
  A2: { name: "Raw Ollama, small window (8k)", short: "Raw · 8k", color: "var(--s2)" },
  B:  { name: "ContextGC, small window (8k)", short: "ContextGC · 8k", color: "var(--s3)" },
};
const NS = "http://www.w3.org/2000/svg";
const fmt = n => n == null ? "–" : n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e4 ? Math.round(n / 1e3) + "k"
  : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(Math.round(n));
const pct = r => r == null ? "–" : Math.round(r * 100) + "%";
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}
function tipFor(host) {
  let t = host.querySelector(".tip");
  if (!t) { t = document.createElement("div"); t.className = "tip"; host.appendChild(t); }
  return t;
}
function placeTip(host, tip, x, y) {
  const w = host.clientWidth, tw = tip.offsetWidth;
  tip.style.left = Math.max(0, Math.min(w - tw, x + 12)) + "px";
  tip.style.top = Math.max(0, y - tip.offsetHeight - 10) + "px";
}
function legend(host, keys) {
  const d = document.createElement("div");
  d.className = "legend";
  d.innerHTML = keys.map(k => `<span><i style="background:${LANES[k].color}"></i>${LANES[k].name}</span>`).join("");
  host.appendChild(d);
}

// Lines: prompt tokens per agent step, one line per lane, crosshair tooltip.
function lineChart(host, series, opts = {}) {
  host.innerHTML = "";
  host.classList.add("chart");
  legend(host, series.map(s => s.key));
  const W = 640, H = opts.height || 240, L = 46, R = 16, T = 12, B = 30;
  const maxX = Math.max(opts.minX || 1, ...series.map(s => s.points.length));
  const maxY = niceMax(Math.max(...(opts.refs || []).map(r => r.y), ...series.flatMap(s => s.points.map(p => p.y || 0))) * 1.05);
  const x = i => L + (maxX <= 1 ? 0 : (i - 1) / (maxX - 1)) * (W - L - R);
  const y = v => T + (1 - v / maxY) * (H - T - B);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "chart" }, host);
  for (let i = 0; i <= 4; i++) {
    const v = maxY * i / 4;
    el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = fmt(v);
  }
  for (let i = 1; i <= maxX; i++) if (maxX <= 14 || i % 2 === 1)
    el("text", { x: x(i), y: H - 10, "text-anchor": "middle" }, svg).textContent = i;
  el("text", { x: W - R, y: H - 10, "text-anchor": "end" }, svg).textContent = opts.xLabel || "";
  for (const r of opts.refs || []) {
    el("line", { x1: L, x2: W - R, y1: y(r.y), y2: y(r.y), stroke: "var(--axis)", "stroke-width": 1, "stroke-dasharray": "4 4" }, svg);
    el("text", { x: W - R, y: y(r.y) - 5, "text-anchor": "end", class: "lbl" }, svg).textContent = r.label;
  }
  for (const s of series) {
    const pts = s.points.filter(p => p.y != null);
    if (!pts.length) continue;
    el("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join(""), fill: "none",
                 stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    const last = pts[pts.length - 1];
    el("circle", { cx: x(last.x), cy: y(last.y), r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
    if (s.endLabel) el("text", { x: Math.min(x(last.x) + 8, W - R), y: y(last.y) + 4, class: "lbl" }, svg).textContent = s.endLabel;
  }
  const cross = el("line", { y1: T, y2: H - B, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" }, svg);
  const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, svg);
  const tip = tipFor(host);
  hit.addEventListener("mousemove", ev => {
    const box = svg.getBoundingClientRect(), sx = (ev.clientX - box.left) * W / box.width;
    const i = Math.max(1, Math.min(maxX, Math.round(1 + (sx - L) / (W - L - R) * (maxX - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    const rows = series.map(s => {
      const p = s.points.find(p => p.x === i);
      return p ? `<div><span class="sw" style="background:${s.color}"></span><span class="k">${LANES[s.key].short}:</span> ` +
        `<b>${p.y == null ? "error" : fmt(p.y) + " tok"}</b>${p.note ? " · " + esc(p.note) : ""}</div>` : "";
    }).join("");
    tip.innerHTML = `<div class="k">Step ${i}</div>${rows || "<div class='k'>no step</div>"}`;
    tip.style.display = "block";
    const hb = host.getBoundingClientRect();
    placeTip(host, tip, ev.clientX - hb.left, ev.clientY - hb.top);
  });
  hit.addEventListener("mouseleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
}

// Grouped vertical bars: groups on x (e.g. models), one bar per lane.
function groupedBars(host, groups, opts = {}) {
  host.innerHTML = "";
  host.classList.add("chart");
  const keys = [...new Set(groups.flatMap(g => g.bars.map(b => b.key)))];
  legend(host, keys);
  const W = 640, H = opts.height || 240, L = 42, R = 8, T = 18, B = 34;
  const maxY = opts.max || niceMax(Math.max(...groups.flatMap(g => g.bars.map(b => b.value))));
  const y = v => T + (1 - v / maxY) * (H - T - B);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "bar chart" }, host);
  for (let i = 0; i <= 4; i++) {
    const v = maxY * i / 4;
    el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = opts.tick ? opts.tick(v) : fmt(v);
  }
  const gw = (W - L - R) / groups.length, bw = Math.min(44, (gw - 24) / keys.length - 2);
  const tip = tipFor(host);
  groups.forEach((g, gi) => {
    const x0 = L + gi * gw + (gw - keys.length * (bw + 2)) / 2;
    el("text", { x: L + gi * gw + gw / 2, y: H - 12, "text-anchor": "middle", class: "lbl" }, svg).textContent = g.label;
    g.bars.forEach(b => {
      const xi = x0 + keys.indexOf(b.key) * (bw + 2), top = y(b.value), h = Math.max(0, y(0) - top);
      const r = Math.min(4, h / 2, bw / 2);
      el("path", { d: `M${xi},${y(0)}V${top + r}Q${xi},${top} ${xi + r},${top}H${xi + bw - r}Q${xi + bw},${top} ${xi + bw},${top + r}V${y(0)}Z`,
                   fill: LANES[b.key].color }, svg);
      el("text", { x: xi + bw / 2, y: top - 5, "text-anchor": "middle", class: "lbl" }, svg).textContent = b.text;
      const hit = el("rect", { x: xi - 1, y: T, width: bw + 2, height: H - T - B, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", ev => {
        tip.innerHTML = `<div class="k">${esc(g.label)}</div><div><span class="sw" style="background:${LANES[b.key].color}"></span>` +
          `${LANES[b.key].name}</div><div><b>${b.tip}</b></div>`;
        tip.style.display = "block";
        const hb = host.getBoundingClientRect();
        placeTip(host, tip, ev.clientX - hb.left, ev.clientY - hb.top);
      });
      hit.addEventListener("mouseleave", () => tip.style.display = "none");
    });
  });
}

// Horizontal bars for one measure (single series: one hue, labels carry the values).
function hBars(host, rows, opts = {}) {
  host.innerHTML = "";
  host.classList.add("chart");
  const W = 640, rowH = 30, L = opts.labelWidth || 190, R = 60, H = rows.length * rowH + 8;
  const max = opts.max || niceMax(Math.max(...rows.map(r => r.value)));
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "bar chart" }, host);
  const tip = tipFor(host);
  rows.forEach((r, i) => {
    const yy = 4 + i * rowH, w = Math.max(2, (r.value / max) * (W - L - R)), bh = rowH - 10;
    el("text", { x: L - 10, y: yy + bh / 2 + 4, "text-anchor": "end", class: "lbl" }, svg).textContent = r.label;
    el("path", { d: `M${L},${yy}H${L + w - 4}Q${L + w},${yy} ${L + w},${yy + 4}V${yy + bh - 4}Q${L + w},${yy + bh} ${L + w - 4},${yy + bh}H${L}Z`,
                 fill: r.color || "var(--s1)" }, svg);
    el("text", { x: L + w + 6, y: yy + bh / 2 + 4, class: "lbl" }, svg).textContent = r.text;
    const hit = el("rect", { x: 0, y: yy - 3, width: W, height: rowH, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", ev => {
      tip.innerHTML = `<div class="k">${esc(r.label)}</div><div><b>${r.tip || r.text}</b></div>`;
      tip.style.display = "block";
      const hb = host.getBoundingClientRect();
      placeTip(host, tip, ev.clientX - hb.left, ev.clientY - hb.top);
    });
    hit.addEventListener("mouseleave", () => tip.style.display = "none");
  });
}
