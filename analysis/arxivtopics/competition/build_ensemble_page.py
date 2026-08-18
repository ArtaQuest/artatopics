#!/usr/bin/env python3
"""Build docs/ensemble.html + docs/data/ensemble.json — the competition page and its Pyodide lab.

Featured model: STACK v3.1 (recent-regime selection; see the_stack_v31.py for the disclosure).
The page ships the raw share matrix, the stack's parameters and the one heavy member's forecast
(the record receiver), so a browser can rebuild the other members from raw data, reassemble the
exact forecast that sits on the board, re-derive its score from the held-out truth, and re-mix
the ensemble live. The era-shift table — the competition's deepest finding — is printed whole.

  python3 analysis/arxivtopics/competition/build_ensemble_page.py
"""
import os, sys, json, csv, re, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
n = Yv.shape[1]; OUTER = n - 30
tv = af.META["topic_valid"]; J = len(names)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DOCS = os.path.join(REPO, "docs")
assert os.path.isfile(os.path.join(DOCS, "index.html")), f"docs/ not found at {DOCS}"
os.makedirs(os.path.join(DOCS, "data"), exist_ok=True)
META = json.load(open(os.path.expanduser("~/.artaquest-dev/artacomp/outputs/stack31/entry_meta.json")))
MIX = {k: v for k, v in META["mix"].items() if v}
ALPHA = np.array(META["alpha"])
RECORD = np.load(os.path.expanduser("~/.artaquest-dev/artacomp/stack_walls.npz"))[f"w{OUTER}_record"][:, :30]

def carry(): return np.repeat(Yv[:, OUTER - 1:OUTER], 30, 1)
def trend(phi=0.85, K=15):
    P = np.zeros((J, 30))
    for j in range(J):
        idx = np.where(tv[j, OUTER - K:OUTER])[0] + OUTER - K
        L = Yv[j, OUTER - 1]
        if len(idx) < 4: P[j] = L; continue
        m = np.polyfit(idx.astype(float), Yv[j, idx], 1)[0]
        h = np.arange(1, 31)
        P[j] = np.clip(L + m * phi * (1 - phi ** h) / (1 - phi), 0, None)
    return P

TR, C = trend(), carry()
sky = (MIX["trend"] * TR + MIX["record"] * RECORD) / (MIX["trend"] + MIX["record"])
P = np.clip(ALPHA[None, :] * C + (1 - ALPHA[None, :]) * sky, 0, None)

def score(pred):
    sc = []
    for j in range(J):
        t = Yv[j, OUTER:]; mu = t.mean(); ss = ((t - mu) ** 2).sum()
        if ss < 1e-12: continue
        sc.append(1 - ((t - pred[j]) ** 2).sum() / ss)
    return float(np.mean(sc))

# integrity: the page must describe exactly what sits on the board
def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")
sub = {}
with open(os.path.expanduser("~/.artaquest-dev/artacomp/outputs/stack31/submission.csv")) as f:
    for r in csv.DictReader(f): sub[(r["trend"], int(r["date"]))] = float(r["target"])
chk = max(abs(P[j, k] - sub[(slug(names[j]), int(labels[OUTER]) + k)]) for j in range(J) for k in (0, 14, 29))
assert chk < 1e-5, f"page model diverges from the submitted CSV ({chk})"
s_full = score(P)
print(f"reconstructed v3.1 score on truth: {s_full:+.4f} (board: -2.0927)")
assert abs(s_full - (-2.0927)) < 0.002

abl = {
    "stack v3.1 (the deployed model)": s_full,
    "damped trend baseline": score(TR),
    "without the record receiver": score(np.clip(ALPHA[None, :] * C + (1 - ALPHA[None, :]) * TR, 0, None)),
    "without carry (a=0)": score(sky),
    "carry-forward baseline": score(C),
    "record receiver alone": score(RECORD),
}
for k, v in abl.items(): print(f"  {k:>32}: {v:+.4f}")
rec_delta = abl["without the record receiver"] - s_full

BOARD = [
    ("damped linear trend (reference baseline)", "baseline", -2.039975, ""),
    ("THE STACK v3.1 — recent-regime selection", "the stack (this page)", -2.092730, "https://github.com/ArtaQuest/artatopics/blob/main/analysis/arxivtopics/competition/the_stack_v31.py"),
    ("stack v5 — kernel wall members offered; shared-basis took a slice, transferred worse", "the stack", -2.121088, "https://github.com/ArtaQuest/artatopics/blob/main/analysis/arxivtopics/competition/the_stack_v5.py"),
    ("the stack v3 — six-wall selection", "the stack", -2.289728, "https://github.com/ArtaQuest/artatopics/blob/main/analysis/arxivtopics/competition/the_stack.py"),
    ("carry today forward (= every family's round-3 shrink verdict: lam 0)", "baseline", -2.561381, ""),
    ("random sky-feature ridge swarm", "ashranet · GPU", -3.592269, "https://www.kaggle.com/code/ashranet/astro-ensemble-entry-sky-swarm"),
    ("neural shared-basis receiver", "ashraasn · GPU", -5.173915, "https://www.kaggle.com/code/ashraasn/astro-ensemble-entry-shared-basis"),
    ("deep per-field phasor", "arash0ash · GPU", -5.603512, "https://www.kaggle.com/code/arash0ash/astro-ensemble-entry-deep-phasor"),
    ("pooled gradient boosting (round 1)", "artafather · GPU", -62.370848, "https://www.kaggle.com/code/artafather/astro-ensemble-entry-boosted-sky"),
]
ERA = [  # wall year → solo scores (computed from the member cache; see the_stack_v31.py run log)
    (1966, -12.80, -14.98, -4.78, -6.64, -6.08),
    (1971, -5.99, -7.38, -4.00, -4.39, -4.07),
    (1976, -3.83, -6.00, -4.40, -4.67, -5.59),
    (1981, -2.48, -2.61, -3.86, -4.35, -6.78),
    (1986, -3.00, -3.47, -4.15, -4.88, -5.46),
    (1991, -3.92, -3.30, -7.98, -6.87, -8.68),
    (1996, -2.56, -2.04, -3.32, -3.24, None),
]

data = {
    "y0": int(labels[0]), "wall_year": int(labels[OUTER]), "names": names,
    "shares": [[round(float(x), 6) for x in row] for row in Yv],
    "stack": {"mix": MIX, "alpha": [round(float(a), 4) for a in ALPHA],
              "trend_phi": 0.85, "trend_window": 15},
    "record_pred": [[round(float(x), 6) for x in row] for row in RECORD],
    "ablations": {k: round(v, 4) for k, v in abl.items()},
    "board": [{"model": m, "who": w, "score": s, "url": u} for m, w, s, u in BOARD],
}
out = os.path.join(DOCS, "data", "ensemble.json")
json.dump(data, open(out, "w"), separators=(",", ":"))
print(f"data/ensemble.json: {os.path.getsize(out) // 1024}KB")

brows = "\n".join(
    f"<tr><td>{m}{(' · <a href=' + chr(34) + u + chr(34) + '>code</a>') if u else ''}</td>"
    f"<td>{w}</td><td class='n'><b>{s:.4f}</b></td></tr>"
    for m, w, s, u in BOARD)
arows = "\n".join(f"<tr><td>{k}</td><td class='n'>{v:+.4f}</td></tr>"
                  for k, v in sorted(abl.items(), key=lambda kv: -kv[1]))
erows = "\n".join(
    "<tr><td>" + (f"{y} → {min(y + 30, 2025) if y < 1996 else 2025}" if y < 1996 else "<b>1996 → 2025 (judged)</b>") + "</td>"
    + "".join(f"<td class='n'>{v:+.2f}</td>" if v is not None else "<td class='n'>—</td>"
              for v in (c, t, c5, sw, rc)) + "</tr>"
    for y, c, t, c5, sw, rc in ERA)
rec_line = (f"Dropping it would have scored {abl['without the record receiver']:+.4f} — its slice cost "
            f"{abs(rec_delta):.3f} points on the judged years." if rec_delta > 0 else
            f"Dropping it would have scored {abl['without the record receiver']:+.4f} — its slice earned "
            f"{abs(rec_delta):.3f} points on the judged years.")

page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>artatopics — the ensemble stack</title>
<style>
:root { --gold:#8A6D0B; --blue:#1746DC; --bg:#ffffff; --card:#f6f7f9; --ink:#1a2330; --ink2:#4a5568; --ink3:#6b7686; --line:#e3e6ea; }
@media (prefers-color-scheme: dark) {
  :root { --gold:#E8B923; --blue:#6f8dff; --bg:#010C17; --card:#06121E; --ink:#eef2f7; --ink2:#b8c2cf; --ink3:#8b98a8; --line:#12283c; } }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif }
main { max-width:760px; margin:0 auto; padding:36px 20px 70px }
h1 { font-size:26px; margin:0 } h1 b { color:var(--gold) }
h2 { font-size:16px; margin:34px 0 8px }
p { color:var(--ink2); margin:8px 0 }
table { border-collapse:collapse; width:100%; font-size:14px }
th,td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line) }
th { color:var(--ink3); font-size:11px; text-transform:uppercase; letter-spacing:.08em }
td.n { font-variant-numeric:tabular-nums; text-align:right }
select,button { padding:8px 12px; border-radius:10px; border:1px solid var(--line);
  background:var(--card); color:var(--ink); font-size:14px }
button { cursor:pointer } button:hover { border-color:var(--blue) }
svg { width:100%; background:var(--card); border:1px solid var(--line); border-radius:10px }
.cap { font-size:12.5px; color:var(--ink3); margin:4px 0 0 }
.mono { font-family:ui-monospace,Menlo,monospace; background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:10px 14px; overflow-x:auto }
.wrap { overflow-x:auto }
#pyout { white-space:pre-wrap; font:12.5px/1.5 ui-monospace,Menlo,monospace; background:var(--card);
  border:1px solid var(--line); border-radius:10px; padding:10px 14px; min-height:40px }
.row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:8px 0 }
.row label { font-size:13px; color:var(--ink2); min-width:120px }
.row input[type=range] { flex:1; accent-color:var(--gold) }
.row output { font-variant-numeric:tabular-nums; font-size:13px; width:44px; text-align:right }
a { color:var(--blue) }
footer { margin-top:44px; font-size:12.5px; color:var(--ink3) }
</style></head><body><main>

<h1>arta<b>topics</b> · the ensemble stack</h1>
<p>Four Kaggle accounts, four GPU model families, one benchmark: forecast each of 251 research
fields' share of the world's citations for 1996–2025, fitting only on 1700–1995. Scored on the
ArtaQuest platform — per-field R² against the holdout mean, averaged. Zero would mean "as good as
knowing each field's future average"; every point below zero is honest distance from that.
Dataset: <a href="https://www.kaggle.com/datasets/artafather/astro-ensemble-251">astro-ensemble-251</a>.</p>

<h2>1 · The board</h2>
<div class="wrap"><table><tr><th>model</th><th>entrant</th><th>score</th></tr>
__BROWS__</table></div>
<p class="cap">Every single-family GPU model loses to the do-nothing baselines. The stacks are the
point of the competition — and the honest headline is that a plain damped trend edges out even the
best of them. The 0.05 between the trend baseline and stack v3.1 is the price of committing to a
model before the answer was visible.</p>

<h2>2 · The deployed model, in one line</h2>
<p class="mono">forecast(field, h) = a(h) · yesterday + (1−a(h)) · [ 0.875 · trend + 0.125 · receiver ]</p>
<p><b>yesterday</b> — the field's 1995 share, held flat. <b>trend</b> — a straight line through its
last 15 years, with the slope damped away (φ=0.85). <b>receiver</b> — the repository's 9-parameter
per-field sky receiver: level + amplitude + seven phases read against the slow planets, the best
purely astrological model this campaign produced. <b>a(h)</b> — how much "yesterday" matters at
horizon h: 0.14 next year, rising to about 0.55 by year fifteen — and zero beyond, where the
recent walls have no evidence; the six-wall run's long-horizon data independently agrees that
yesterday's exact value stops helping after year thirteen or so. Selection used only data from before 1996,
on the three most recent walls (1981/86/91) — the disclosure of how that window was chosen is
written into <a href="https://github.com/ArtaQuest/artatopics/blob/main/analysis/arxivtopics/competition/the_stack_v31.py">the code's header</a>.</p>

<h2>3 · Every field, forecast against what happened</h2>
<div class="row"><select id="pick" style="flex:1"></select></div>
<svg id="chart" viewBox="0 0 760 280"></svg>
<p class="cap">Blue — the field's real share of each year's citations, 1700–2025. Gold dashes — the
deployed stack's 30-year forecast from the 1995 wall. Thin lines — its members
(<span style="color:#c084fc">damped trend</span>, <span style="color:#f87171">sky receiver</span>,
<span style="color:#34d399">yesterday</span>).</p>

<h2>4 · What each part is worth on the judged years</h2>
<table><tr><th>variant</th><th>score</th></tr>
__AROWS__</table>
<p class="cap">The sky receiver keeps a 12.5% slice because the pre-1996 evidence earned it one.
__RECLINE__ Both facts are on this page because both are true.</p>

<h2>5 · The finding: you learn the era you select in</h2>
<div class="wrap"><table><tr><th>fit ≤ wall, judged after</th><th>yesterday</th><th>trend</th><th>5-yr level</th><th>sky swarm</th><th>receiver</th></tr>
__EROWS__</table></div>
<p>Read down any column. In the 1966 window — thirty years that re-ranked science violently — every
"do nothing" strategy is terrible and the sky models look relatively strong. By the 1980s the
field system has begun to ossify; yesterday's value and a gentle trend dominate, and they keep
dominating through the judged years. A stack selected across all six walls inherits the old era's
tastes and scored −2.29; selected on the recent regime only, −2.09. No planetary configuration
explains this — the calendar does. That is the competition's deepest result, and it is the same
one the <a href="index.html">main page</a> reports for the trending classifier: the sky's
predictive power here is mostly a slow clock.</p>

<h2>6 · Rebuild it in your browser</h2>
<p>This page ships the raw share matrix, the stack's parameters and the receiver's forecast. The
button loads Python (Pyodide + numpy, ~10&nbsp;MB, from a CDN), rebuilds the trend and carry
members from the raw shares, reassembles the stack, checks it against the exact forecast on the
board, and re-scores it on the held-out truth. Then the sliders re-mix the ensemble live. Note
that the two weight sliders only re-mix the bracketed part: yesterday is still folded in by a(h),
so receiver&nbsp;=&nbsp;1 scores about &minus;3.21 rather than the receiver's own &minus;3.58. Pull
the a(h) slider to 0 to remove yesterday entirely and see each member undiluted.</p>
<div class="row"><button id="run">Run the verification</button><span id="pystat" class="cap"></span></div>
<div id="pyout">(not run yet)</div>
<div id="mixer" style="display:none">
<div class="row"><label>damped trend</label><input type="range" id="w_trend" min="0" max="100" value="88"><output id="o_trend">.88</output></div>
<div class="row"><label>sky receiver</label><input type="range" id="w_record" min="0" max="100" value="12"><output id="o_record">.12</output></div>
<div class="row"><label>a(h) scale</label><input type="range" id="w_alpha" min="0" max="200" value="100"><output id="o_alpha">1.0</output></div>
<div class="row"><button id="rescore">Re-score this mix</button><b id="mixscore" style="font-variant-numeric:tabular-nums"></b></div>
<p class="cap">Weights renormalise to sum to one. The re-score runs the same numpy code on the same
data — nothing on this page is taken on faith.</p>
</div>

<footer>No causal claims. Scored by the ArtaQuest platform · code and full history:
<a href="https://github.com/ArtaQuest/artatopics">github.com/ArtaQuest/artatopics</a> ·
<a href="index.html">main results</a></footer>
</main>
<script>
(async () => {
  const D = await (await fetch("data/ensemble.json")).json();
  const J = D.names.length, W = D.wall_year - D.y0, H = 30;
  const pick = document.getElementById("pick"), svg = document.getElementById("chart");
  D.names.map((n, j) => [n, j]).sort((a, b) => a[0].localeCompare(b[0]))
    .forEach(([n, j]) => { const o = document.createElement("option"); o.textContent = n; o.value = j; pick.appendChild(o); });
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  function members(j) {
    const s = D.shares[j], L = s[W - 1], K = D.stack.trend_window, phi = D.stack.trend_phi;
    const xs = [], ys = [];
    for (let t = W - K; t < W; t++) if (s[t] > 0) { xs.push(t); ys.push(s[t]); }
    let m = 0;
    if (xs.length >= 4) {
      const mx = xs.reduce((a, b) => a + b, 0) / xs.length, my = ys.reduce((a, b) => a + b, 0) / ys.length;
      let num = 0, den = 0;
      xs.forEach((x, i) => { num += (x - mx) * (ys[i] - my); den += (x - mx) * (x - mx); });
      m = den ? num / den : 0;
    }
    const tr = [], cy = [];
    for (let h = 1; h <= H; h++) {
      tr.push(Math.max(L + m * phi * (1 - Math.pow(phi, h)) / (1 - phi), 0));
      cy.push(L);
    }
    return { trend: tr, record: D.record_pred[j], carry: cy };
  }
  function stackOf(j) {
    const M = members(j), g = D.stack.mix, a = D.stack.alpha, gt = g.trend + g.record;
    return a.map((ah, k) => Math.max(ah * M.carry[k] + (1 - ah) *
      (g.trend * M.trend[k] + g.record * M.record[k]) / gt, 0));
  }
  function draw(j) {
    const s = D.shares[j].map(x => x * 100), F = stackOf(j).map(x => x * 100), M = members(j);
    const Wd = 760, Hd = 280, Lp = 46, Rp = 8, Tp = 10, Bp = 24, N = s.length;
    const ymax = Math.max(...s, ...F) * 1.08 || 1;
    const x = i => Lp + (Wd - Lp - Rp) * i / (N - 1), y = v => Tp + (Hd - Tp - Bp) * (1 - v / ymax);
    const path = (arr, i0) => arr.map((v, k) => (k ? "L" : "M") + x(i0 + k).toFixed(1) + " " + y(v).toFixed(1)).join("");
    let g = "";
    for (let f = 0; f <= 4; f++) { const yy = Tp + (Hd - Tp - Bp) * f / 4;
      g += `<line x1="${Lp}" y1="${yy}" x2="${Wd - Rp}" y2="${yy}" stroke="${css('--line')}"/>`;
      g += `<text x="${Lp - 6}" y="${yy + 4}" fill="${css('--ink3')}" font-size="10" text-anchor="end">${(ymax * (1 - f / 4)).toFixed(1)}</text>`; }
    for (let yr = 1725; yr <= D.y0 + N; yr += 50) { g += `<text x="${x(yr - D.y0)}" y="${Hd - 8}" fill="${css('--ink3')}" font-size="10" text-anchor="middle">${yr}</text>`; }
    const xw = x(W - 1);
    g += `<line x1="${xw}" y1="${Tp}" x2="${xw}" y2="${Hd - Bp}" stroke="${css('--ink3')}" stroke-dasharray="2 3"/>`;
    const th = (arr, col) => `<path d="${path(arr.map(v => v * 100), W)}" fill="none" stroke="${col}" stroke-width="1" opacity=".65"/>`;
    g += th(M.trend, "#c084fc") + th(M.record, "#f87171") + th(M.carry, "#34d399");
    g += `<path d="${path(s, 0)}" fill="none" stroke="${css('--blue')}" stroke-width="1.6"/>`;
    g += `<path d="${path(F, W)}" fill="none" stroke="${css('--gold')}" stroke-width="2.2" stroke-dasharray="6 4"/>`;
    svg.innerHTML = g;
  }
  pick.onchange = () => draw(+pick.value);
  pick.value = "" + D.names.indexOf("Artificial Intelligence"); draw(+pick.value);

  const stat = document.getElementById("pystat"), out = document.getElementById("pyout");
  let py = null;
  const PYCODE = `
import numpy as np, json
D = json.loads(DATA_JSON)
Y = np.array(D["shares"]); J, N = Y.shape
W = D["wall_year"] - D["y0"]; H = 30
mix, alpha = D["stack"]["mix"], np.array(D["stack"]["alpha"])
phi, K = D["stack"]["trend_phi"], D["stack"]["trend_window"]
carry = np.repeat(Y[:, W-1:W], H, 1)
tr = np.zeros((J, H)); h = np.arange(1, H+1)
for j in range(J):
    win = Y[j, W-K:W]; idx = np.where(win > 0)[0] + (W-K); L = Y[j, W-1]
    if len(idx) < 4: tr[j] = L; continue
    m = np.polyfit(idx.astype(float), Y[j, idx], 1)[0]
    tr[j] = np.clip(L + m*phi*(1-phi**h)/(1-phi), 0, None)
rec = np.array(D["record_pred"])
def assemble(gt, gr, ascale):
    tot = gt + gr
    sky = (gt*tr + gr*rec)/tot if tot > 0 else np.zeros((J, H))
    a = np.clip(alpha*ascale, 0, 1)
    return np.clip(a[None]*carry + (1-a[None])*sky, 0, None)
def score(P):
    sc = []
    for j in range(J):
        t = Y[j, W:W+H]; mu = t.mean(); ss = ((t-mu)**2).sum()
        if ss < 1e-12: continue
        sc.append(1 - ((t-P[j])**2).sum()/ss)
    return float(np.mean(sc))
P = assemble(mix["trend"], mix["record"], 1.0)
print(f"members rebuilt from the raw shares ({J} fields x {N} years)")
print(f"stack reassembled; score on held-out 1996-2025 truth: {score(P):+.4f}")
print("board says -2.0927 -- reproduced" if abs(score(P) - (-2.09273)) < 0.002 else "MISMATCH vs the board")
`;
  document.getElementById("run").onclick = async () => {
    try {
      stat.textContent = "loading Pyodide…";
      if (!py) {
        const mod = await import("https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.mjs");
        py = await mod.loadPyodide(); await py.loadPackage("numpy");
      }
      stat.textContent = "running…";
      py.globals.set("DATA_JSON", JSON.stringify(D));
      let buf = "";
      py.setStdout({ batched: s => { buf += s + "\\n"; out.textContent = buf; } });
      await py.runPythonAsync(PYCODE);
      stat.textContent = "done — live re-mixing enabled";
      document.getElementById("mixer").style.display = "block";
    } catch (e) { stat.textContent = ""; out.textContent = "failed to load/run: " + e; }
  };
  const upd = () => {
    ["trend", "record"].forEach(k =>
      document.getElementById("o_" + k).textContent = (document.getElementById("w_" + k).value / 100).toFixed(2).slice(1));
    document.getElementById("o_alpha").textContent = (document.getElementById("w_alpha").value / 100).toFixed(1);
  };
  ["w_trend", "w_record", "w_alpha"].forEach(id => document.getElementById(id).oninput = upd);
  document.getElementById("rescore").onclick = async () => {
    const v = id => document.getElementById(id).value / 100;
    py.globals.set("GT", v("w_trend")); py.globals.set("GR", v("w_record")); py.globals.set("AS", v("w_alpha"));
    const s = await py.runPythonAsync("score(assemble(GT, GR, AS))");
    document.getElementById("mixscore").textContent = s.toFixed(4) + (s > -2.0398 ? "  — beats every entry on the board" : "");
  };
})();
</script>
</body></html>"""
page = (page.replace("__BROWS__", brows).replace("__AROWS__", arows)
            .replace("__EROWS__", erows).replace("__RECLINE__", rec_line))
open(os.path.join(DOCS, "ensemble.html"), "w").write(page)
print(f"docs/ensemble.html written ({len(page) // 1024}KB)")
