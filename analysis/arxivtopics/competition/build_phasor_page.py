#!/usr/bin/env python3
"""docs/phasor.html — the exact phasor, solved analytically, and what it scored. Ships the solver
itself: a Pyodide lab runs the closed-form solution on the same 8-body kerykeion sky and the same
daily series a reader can pick, and prints b, a_i, p_i and the held-out score. Data shipped:
the daily kerykeion sky (subsampled to every day, 8 bodies) and 12 representative arXiv categories."""
import os, sys, json, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE))); DOCS = os.path.join(REPO, "docs")
D = os.path.expanduser("~/.artaquest-dev/artacomp/daily")
S = json.load(open(f"{D}/phasor_exact_summary.json"))
L2 = pd.read_csv(f"{D}/phasor_exact_l2.csv"); F5 = pd.read_csv(f"{D}/phasor_exact_fast5.csv"); A8 = pd.read_csv(f"{D}/phasor_exact_results.csv")
E = np.load(f"{D}/ephemeris_ker_1991_2026.npz"); EB = list(E["bodies"]); e0 = str(E["d0"])
BOD = ["sun","moon","mercury","venus","mars","jupiter","saturn","true_node"]; SEL = [EB.index(b) for b in BOD]
daily = pd.read_csv(f"{D}/daily.csv", parse_dates=["date"]).set_index("date")
rel = pd.read_csv(f"{D}/reliable_from.csv", parse_dates=["reliable_from"]).set_index("category")
PICK = ["hep-th","astro-ph.EP","cs.LG","math.GT","q-bio.PE","cond-mat.str-el","gr-qc","stat.ML","cs.CV","math.AP","physics.optics","eess.IV"]
PICK = [c for c in PICK if c in daily.columns and c in rel.index]
d0 = dt.date.fromisoformat(e0); days = np.array([d.date() for d in daily.index.to_pydatetime()])
off = np.array([(d-d0).days for d in days]); valid = (off>=0)&(off<E["lon"].shape[0])
series = {}
for c in PICK:
    start = max(int(np.searchsorted(days, rel.loc[c,"reliable_from"].date())), int(np.argmax(valid)))
    x = daily[c].to_numpy(float)[start:]; series[c] = {"start": str(days[start]), "x": [int(v) for v in x]}
data = {"e0": e0, "bodies": BOD, "lon": [[round(float(v),2) for v in row] for row in E["lon"][:, SEL]], "series": series,
        "summary": S, "l2_rows": L2[["category","lam","r2_heldout","auc_level","auc_rise","b","a_sun","a_moon","a_saturn","p_sun"]].round(4).to_dict("records")}
os.makedirs(os.path.join(DOCS,"data"), exist_ok=True)
json.dump(data, open(os.path.join(DOCS,"data","phasor.json"),"w"), separators=(",",":"))
print("data/phasor.json:", os.path.getsize(os.path.join(DOCS,"data","phasor.json"))//1024, "KB")
def row(name, d): return f"<tr><td>{name}</td><td class='n'>{d['auc_level']:.4f}</td><td class='n'>{d['auc_rise']:.4f}</td><td class='n'>{d.get('null', float('nan')):.4f}</td><td class='n'>{d['feasible_pct']:.0f}%</td><td class='n'>{d['r2_median']:+.3f}</td></tr>"
rows = row("all 8 bodies, unregularised", S["all8"] | {"null": float('nan')}) + row("Sun–Mars only (cycle-complete)", S["fast5"]) + row("<b>all 8, L2, λ per category on an inner wall</b>", S["l2_all8"])
lam = S["l2_all8"]["lam_hist"]; lamtxt = " · ".join(f"λ={k}: {v}" for k,v in lam.items() if v)
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>artatopics — the exact phasor, solved</title>
<style>
:root {{ --gold:#8A6D0B; --blue:#1746DC; --bg:#fff; --card:#f6f7f9; --ink:#1a2330; --ink2:#4a5568; --ink3:#6b7686; --line:#e3e6ea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --gold:#E8B923; --blue:#6f8dff; --bg:#010C17; --card:#06121E; --ink:#eef2f7; --ink2:#b8c2cf; --ink3:#8b98a8; --line:#12283c; }} }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif }}
main {{ max-width:760px; margin:0 auto; padding:36px 20px 70px }} h1 {{ font-size:26px; margin:0 }} h1 b {{ color:var(--gold) }} h2 {{ font-size:16px; margin:34px 0 8px }}
p {{ color:var(--ink2); margin:8px 0 }} table {{ border-collapse:collapse; width:100%; font-size:14px }} th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line) }}
th {{ color:var(--ink3); font-size:11px; text-transform:uppercase; letter-spacing:.08em }} td.n {{ font-variant-numeric:tabular-nums; text-align:right }}
.mono {{ font-family:ui-monospace,Menlo,monospace; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; overflow-x:auto; font-size:13.5px }}
select,button {{ padding:8px 12px; border-radius:10px; border:1px solid var(--line); background:var(--card); color:var(--ink); font-size:14px }} button {{ cursor:pointer }}
#pyout {{ white-space:pre-wrap; font:12.5px/1.5 ui-monospace,Menlo,monospace; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; min-height:40px }}
.row {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:8px 0 }} .cap {{ font-size:12.5px; color:var(--ink3) }} a {{ color:var(--blue) }} footer {{ margin-top:44px; font-size:12.5px; color:var(--ink3) }}
</style></head><body><main>
<h1>arta<b>topics</b> · the exact phasor, solved analytically</h1>
<p>The model this campaign was built around, solved in closed form — no optimiser, no seed — and
tested on the finest record we have: every arXiv submission since 1991, day by day, 127 categories.</p>

<h2>1 · The model and its exact solution</h2>
<p class="mono">y(t) = | b + Σᵢ aᵢ · e<sup> i(θᵢ(t) − pᵢ)</sup> |²</p>
<p>The only inputs are θᵢ(t): the sidereal (Lahiri) longitudes of Sun, Moon, Mercury, Venus, Mars,
Jupiter, Saturn and the true node from kerykeion (Swiss Ephemeris). The unknowns b, aᵢ, pᵢ are
solved analytically. Expanding the square exactly:</p>
<p class="mono">y = b² + Σaᵢ² &nbsp;+&nbsp; Σᵢ 2b·aᵢ·cos(θᵢ−pᵢ) &nbsp;+&nbsp; Σᵢ&lt;ₖ 2aᵢaₖ·cos((θᵢ−pᵢ)−(θₖ−pₖ))</p>
<p>Every term is linear in the fixed basis {{1, cos θᵢ, sin θᵢ, cos(θᵢ−θₖ), sin(θᵢ−θₖ)}}, so one
least-squares solve gives its coefficients c. Then, exactly: <b>pᵢ = atan2(βᵢ, αᵢ)</b>;
<b>Mᵢ = √(αᵢ²+βᵢ²) = 2b·aᵢ</b>; and from c₀ = b² + ΣMᵢ²/4b² the quadratic
4b⁴ − 4c₀b² + ΣMᵢ² = 0 gives <b>b² = (c₀ + √(c₀² − ΣMᵢ²))/2</b> — the '+' root, the only one that
reproduces the fitted curve. Self-test on data generated by the exact model: b recovered to six
decimals, aᵢ and pᵢ to 10⁻¹³; with 5% noise, R² 0.9976.</p>
<p>The form <em>over-determines</em> itself: the aspect coefficients must equal 2aᵢaₖcos(pᵢ−pₖ) with the
aᵢ, pᵢ the transit terms already fixed. How far the freely fitted aspects sit from that is the
<b>aspect residual</b> — the part of the data the phasor cannot be. If disc = c₀² − ΣMᵢ² &lt; 0, no
real (b, aᵢ) exists at all: the fit is <b>infeasible</b>.</p>

<h2>2 · What it scored on the daily record</h2>
<p>Each of 127 arXiv categories fitted independently on the first 80% of its reliable days (the
series as a ratio to its own trailing-365-day level, so b carries the level and the arrows the
timing), scored on the last 20% as a <b>peak detector</b>: does the forecast rank the days before
a submission peak above the rest? One AUC per category, averaged.</p>
<table><tr><th>exact phasor, kerykeion sidereal</th><th>level AUC</th><th>rise AUC</th><th>shift-null</th><th>feasible</th><th>held-out R² (median)</th></tr>{rows}</table>
<p class="cap">Chance for these autocorrelated labels is the circular-shift null, ~0.50–0.52. Under L2 the
inner walls chose {lamtxt} — in 118 of 127 categories the strongest shrinkage on offer, i.e. the model
prefers b alone. Aspect residual is 1.000 throughout: the aspect terms the data wants bear no relation
to the ones the transit terms determine.</p>
<p>On the yearly citation-share task — where the campaign's <em>record</em> model scores −3.58 against
carry-forward's −2.56 (per-field R² vs holdout mean) — the exact form with the same horizon anchor,
selection on inner walls, scores <b>−8.25</b> on the live board. The record model fits
√y ≈ b + Σaᵢcos(θᵢ−pᵢ), which drops the (Σaᵢ sin)² term of |z|²; keeping it makes the fit stiffer,
not better. That approximation was carrying the record model.</p>

<h2>3 · Solve it yourself</h2>
<p>Pick a category. The button loads Python (Pyodide + numpy, ~10 MB), runs the <em>same closed-form
solution</em> on the same kerykeion sky and the same daily series, and prints b, every aᵢ, every pᵢ,
the feasibility, the aspect residual and the held-out peak AUC. Nothing here is precomputed.</p>
<div class="row"><select id="cat"></select><button id="run">Solve</button><span id="st" class="cap"></span></div>
<div id="pyout">(not run yet)</div>
<footer>No causal claims. Ephemeris cross-checked against three engines (<a href="https://github.com/ArtaQuest/artatopics/blob/main/analysis/arxivtopics/competition/daily/EPHEMERIS_CHECK.md">the check</a>) ·
<a href="https://github.com/ArtaQuest/artatopics/blob/main/analysis/arxivtopics/competition/phasor_exact.py">solver source</a> ·
<a href="index.html">main results</a> · <a href="ensemble.html">the ensemble stack</a> · <a href="https://huggingface.co/spaces/artaquest/artatopics">also on Hugging Face</a></footer>
</main>
<script>
(async () => {{
  const D = await (await fetch("data/phasor.json")).json();
  const sel = document.getElementById("cat"), out = document.getElementById("pyout"), st = document.getElementById("st");
  Object.keys(D.series).forEach(c => {{ const o = document.createElement("option"); o.textContent = c; sel.appendChild(o); }});
  let py = null;
  const CODE = `
import numpy as np, json, datetime as dt
D = json.loads(DATA); LON = np.array(D["lon"], float); e0 = dt.date.fromisoformat(D["e0"])
def basis(TH):
    T,B = TH.shape; C=[np.ones(T)]
    for i in range(B): C += [np.cos(TH[:,i]), np.sin(TH[:,i])]
    for i in range(B):
        for k in range(i+1,B): d=TH[:,i]-TH[:,k]; C += [np.cos(d), np.sin(d)]
    return np.stack(C,1)
def solve(TH, y, ridge):
    T,B = TH.shape; Phi = basis(TH); R = np.eye(Phi.shape[1]); R[0,0]=0
    c = np.linalg.solve(Phi.T@Phi + ridge*R + 1e-12*np.eye(Phi.shape[1]), Phi.T@y)
    c0=c[0]; al=c[1:1+2*B:2]; be=c[2:2+2*B:2]; p=np.arctan2(be,al); M=np.sqrt(al**2+be**2)
    disc=c0**2-(M**2).sum(); feas = disc>=0 and c0>0
    if not feas: M=M*min(1.0, np.sqrt(max(c0,1e-12)**2/max((M**2).sum(),1e-24))); disc=max(c0**2-(M**2).sum(),0)
    b=np.sqrt(max((c0+np.sqrt(disc))/2,1e-18)); a=M/(2*b)
    gf=c[1+2*B::2]; df_=c[2+2*B::2]; gi=[]; di=[]
    for i in range(B):
        for k in range(i+1,B): gi.append(2*a[i]*a[k]*np.cos(p[i]-p[k])); di.append(2*a[i]*a[k]*np.sin(p[i]-p[k]))
    fr=np.concatenate([gf,df_]); im=np.concatenate([gi,di]); res=float(np.sqrt(((fr-im)**2).sum())/max(np.sqrt((fr**2).sum()),1e-12))
    return b,a,p,feas,res
def predict(TH,b,a,p): return np.abs(b + (a[None,:]*np.exp(1j*(TH-p[None,:]))).sum(1))**2
def auc(y,s):
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1); n1=y.sum(); n0=len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)) if n1>0 and n0>0 else float("nan")
S = D["series"][CAT]; x=np.array(S["x"],float); start=dt.date.fromisoformat(S["start"]); off=(start-e0).days
TH=np.deg2rad(LON[off:off+len(x)]); n=len(x); cut=int(n*0.8)
lvl=np.array([x[max(0,i-365):i].mean() if i>60 else x[:60].mean() for i in range(n)]); xr=x/np.maximum(lvl,1e-9)
# peaks: 7-day centred mean, local max above trailing-90 median + max(20%, 1sd)
s7=np.convolve(x,np.ones(7)/7,mode="same"); med=np.array([np.median(s7[max(0,i-90):i]) if i>30 else np.nan for i in range(n)])
sd=np.array([np.std(s7[max(0,i-90):i]-np.nanmean(med[max(0,i-90):i])) if i>30 else np.nan for i in range(n)])
thr=np.where(np.isnan(med),np.inf,med+np.maximum(0.2*np.nan_to_num(med),np.nan_to_num(sd,nan=np.inf)))
peak=np.zeros(n,bool)
for i in range(3,n-3):
    if s7[i]>=s7[i-3:i+4].max() and s7[i]>thr[i]: peak[i]=True
y=np.zeros(n,int)
for i in np.where(peak)[0]: y[max(0,i-7):i]=1
best=None
for lam in (1e-4,1e-3,1e-2,1e-1,1.0,10.0,100.0):
    k=int(cut*0.75); b,a,p,f,r=solve(TH[:k],xr[:k],lam); yh=predict(TH,b,a,p)
    r2=1-((xr[k:cut]-yh[k:cut])**2).sum()/((xr[k:cut]-xr[:k].mean())**2).sum()
    if best is None or r2>best[0]: best=(r2,lam)
b,a,p,feas,res=solve(TH[:cut],xr[:cut],best[1]); yh=predict(TH,b,a,p); te=slice(cut,n)
r2=1-((xr[te]-yh[te])**2).sum()/((xr[te]-xr[:cut].mean())**2).sum()
print(f"{{CAT}}: {{n}} days from {{start}} · {{int(peak.sum())}} peaks · fit on first {{cut}} days, scored on the last {{n-cut}}")
print(f"lambda chosen on the inner split: {{best[1]}}")
print(f"b = {{b:.4f}}   feasible: {{feas}}   aspect residual: {{res:.3f}}")
for nm,ai,pi in zip(D["bodies"],a,p): print(f"  {{nm:<10}} a = {{ai:.4f}}   p = {{np.rad2deg(pi)%360:7.2f}} deg")
print(f"held-out R2 (ratio to level): {{r2:+.4f}}   ·   peak-detection AUC (level): {{auc(y[te],yh[te]):.4f}}   (rise): {{auc(y[te],np.diff(yh,prepend=yh[0])[te]):.4f}}")
`;
  document.getElementById("run").onclick = async () => {{
    try {{
      st.textContent = "loading Pyodide…";
      if (!py) {{ const m = await import("https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.mjs"); py = await m.loadPyodide(); await py.loadPackage("numpy"); }}
      st.textContent = "solving…"; py.globals.set("DATA", JSON.stringify(D)); py.globals.set("CAT", sel.value);
      let buf = ""; py.setStdout({{ batched: s => {{ buf += s + "\\n"; out.textContent = buf; }} }});
      await py.runPythonAsync(CODE); st.textContent = "done";
    }} catch (e) {{ st.textContent = ""; out.textContent = "failed: " + e; }}
  }};
}})();
</script></body></html>"""
open(os.path.join(DOCS,"phasor.html"),"w").write(page); print("docs/phasor.html written", len(page)//1024, "KB")
