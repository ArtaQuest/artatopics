"""Score the deployed phasor form on the trending-AUC benchmark.

  y_j(t) = | b_j + A_j SUM_i a_i e^{i(theta_i(t) - p_ji)} |^2

Two honest variants, both fitted ONLY on years < 1985 (the benchmark wall):
  (A) the deployed model exactly as it ships — fitted to citation SHARES (global_phasor.fit_wall),
      trending call derived from its own forecast trajectory;
  (B) the same squared-envelope closed form refit to the WORKS series the label is built from
      (like-for-like), both as the 57-feature relaxation (the form's best case) and projected.
Score: predicted relative growth ranked within each year, AUC against the held-out labels.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics"))
from sklearn.metrics import roc_auc_score
import arxiv_fit as af
import global_phasor as GP

BUN = os.path.expanduser("~/.artaquest-dev/artacomp/aucomp")
te = pd.read_csv(f"{BUN}/test.csv"); sol = pd.read_csv(f"{BUN}/solution.csv")
yte = sol.set_index("id").loc[te["id"]]["target"].to_numpy()
usage = sol.set_index("id").loc[te["id"]]["Usage"].to_numpy()
names, Yv, labels, future = af.load_lunar()
years = [int(y) for y in labels]; Y0 = years[0]; n = Yv.shape[1]
WALL_Y = 1985; wall = years.index(WALL_Y)
FI = {nm: i for i, nm in enumerate(names)}
print(f"wall index {wall} ({WALL_Y}) · fields {len(names)}", flush=True)

def score(P, tag):
    """P: (J, n_all) predicted trajectory. Trending score = predicted relative growth."""
    s = []
    for f, t in zip(te["field"], te["year"]):
        j = FI[f]; i = int(t) - Y0
        a, b = P[j, i], P[j, i + 1]
        s.append((b - a) / max(abs(a), 1e-9))
    s = np.asarray(s, float)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    o = roc_auc_score(yte, s)
    pu = roc_auc_score(yte[usage == 'Public'], s[usage == 'Public'])
    pr = roc_auc_score(yte[usage == 'Private'], s[usage == 'Private'])
    print(f"  {tag:<44} overall {o:.4f} · public {pu:.4f} · private {pr:.4f}", flush=True)
    return o

print("\n(A) the deployed model, fitted to citation shares at the 1985 wall:", flush=True)
P_relax, P_exact, P_btopic, P_gain, bg, ag, bj, Aj, Pji = GP.fit_wall(wall)
score(P_gain,  "phasor, per-field level+gain+phases (DEPLOYED)")
score(P_exact, "phasor, exact global projection")
score(P_relax, "the 57-feature relaxation (form's best case)")

print("\n(B) the same form refit to the WORKS series the label is built from:", flush=True)
REPO = os.path.expanduser("~/.artaquest-dev/artatopics")
_w = pd.read_csv(f"{REPO}/analysis/citations/rail_works_yearly.csv")
_c = pd.read_csv(f"{REPO}/analysis/citations/citations_received_yearly.csv")
_w = _w.set_index("subfield_id").loc[_c["subfield_id"]].reset_index()
W = _w[[c for c in _w.columns if c[:1].isdigit()][:n]].to_numpy(float)
F = GP.F; NF = GP.NF; ne = GP.ne
tv = af.META["topic_valid"]
S = np.sqrt(np.maximum(W, 0))                                  # sqrt scale: the envelope's own scale
lvl = np.maximum(S[:, max(0, wall - 5):wall].mean(1), 1e-6)
Wt = tv[:, :wall].astype(float) * (np.clip(af.META["evidence"][:wall], 0, None) ** 0.75)[None]
Wt = Wt / np.maximum(Wt.sum(1, keepdims=True), 1e-9)
Ft = F[:wall]
Rg = np.eye(NF); Rg[0, 0] = 0.0
for ridge in (0.01, 0.1):
    coef = np.zeros((len(names), NF))
    for j in range(len(names)):
        A = Ft.T @ (Ft * Wt[j][:, None]) + ridge * Rg + 1e-8 * np.eye(NF)
        b = Ft.T @ (Wt[j] * (S[j, :wall] / lvl[j]))
        coef[j] = np.linalg.solve(A, b)
    P = (np.clip(coef @ F.T, 0, None) * lvl[:, None]) ** 2
    score(P, f"works-fitted squared envelope (ridge {ridge})")
