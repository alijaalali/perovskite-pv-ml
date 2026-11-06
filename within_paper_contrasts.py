import json, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
PROJ = Path(__file__).resolve().parent; DATA = PROJ / "data"; FIG = PROJ / "figures"; FIG.mkdir(exist_ok=True)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
d = pd.read_parquet(DATA / "stability_table.parquet")
raw = pd.read_csv(DATA / "perovskite_raw.csv", usecols=["Perovskite_additives_compounds", "Perovskite_deposition_quenching_media"], low_memory=False).loc[d.raw_idx.values].reset_index(drop=True)
d["y"] = d.retention_end.clip(0, 1.2); d["yc"] = (d.retention_end >= 0.8).astype(float); d["logt"] = np.log10(d.t_exp_h)
tok = raw.Perovskite_additives_compounds.fillna("Unknown").astype(str).str.split(";").apply(lambda l: [t.strip() for t in l])
d["F_additive"] = tok.apply(lambda l: "unknown" if "Unknown" in l else ("none" if all(t in ("Undoped", "") for t in l) else "additive"))
def simp(s, keep):
    s = s.astype(str).str.strip(); return s.where(s.isin(keep), "other")
def top(s, k): return list(s.value_counts().head(k).index)
d["F_architecture"] = simp(d.Cell_architecture, ["nip", "pin"])
d["F_ETL"] = simp(d.ETL_stack_sequence, top(d.ETL_stack_sequence, 6)); d["F_HTL"] = simp(d.HTL_stack_sequence, top(d.HTL_stack_sequence, 6)); d["F_backcontact"] = simp(d.Backcontact_stack_sequence, top(d.Backcontact_stack_sequence, 4))
d["F_encapsulation"] = simp(d.Encapsulation, ["True", "False"])
d["F_A_site"] = simp(d.a_site, top(d.a_site, 5)); d["F_B_site"] = d.b_site_class
d["F_deposition"] = simp(d.Perovskite_deposition_procedure, top(d.Perovskite_deposition_procedure, 3))
d["F_quench"] = simp(raw.Perovskite_deposition_quenching_media, top(raw.Perovskite_deposition_quenching_media.fillna("Unknown"), 4)) if False else raw.Perovskite_deposition_quenching_media.astype(str).str.strip().replace({"nan": "Unknown"}).map(lambda x: x)
qtop = [t for t in d.F_quench.value_counts().head(4).index]; d["F_quench"] = d.F_quench.where(d.F_quench.isin(qtop), "other")
FACTORS = [c for c in d.columns if c.startswith("F_")]
UNK = {"unknown", "Unknown", "other"}
def contrasts(window):
    rec = []
    for pid, g in d.groupby("paper_id", sort=False):
        if len(g) < 2: continue
        gi = g.reset_index(drop=True); n = len(gi); iu, ju = np.triu_indices(n, 1)
        same = (gi.isos.values[iu] == gi.isos.values[ju]) & (np.abs(gi.logt.values[iu] - gi.logt.values[ju]) < window)
        iu, ju = iu[same], ju[same]
        if len(iu) == 0: continue
        F = gi[FACTORS].values
        for a, b in zip(iu, ju):
            diff = [k for k in range(len(FACTORS)) if F[a, k] != F[b, k]]
            if len(diff) != 1: continue
            k = diff[0]; la, lb = F[a, k], F[b, k]
            if la in UNK or lb in UNK: continue
            rec.append((pid, FACTORS[k][2:], la, lb, gi.y.values[a], gi.y.values[b], gi.yc.values[a], gi.yc.values[b]))
    return pd.DataFrame(rec, columns=["paper", "factor", "la", "lb", "ya", "yb", "ca", "cb"])
def orient(pairs):
    ref = {f: d["F_" + f].value_counts().index[0] for f in [c[2:] for c in FACTORS]}
    out = []
    for r in pairs.itertuples():
        R = ref[r.factor]
        if r.la == R and r.lb != R: out.append((r.paper, r.factor, R, r.lb, r.yb - r.ya, r.cb - r.ca))
        elif r.lb == R and r.la != R: out.append((r.paper, r.factor, R, r.la, r.ya - r.yb, r.ca - r.cb))
    return pd.DataFrame(out, columns=["paper", "factor", "ref", "level", "dy", "dc"])
def summarize(o, nboot=2000, nperm=5000, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for (f, ref, lvl), g in o.groupby(["factor", "ref", "level"]):
        pm = g.groupby("paper").agg(dy=("dy", "mean"), dc=("dc", "mean"), n=("dy", "size")); k = len(pm)
        if k < 3: continue
        est = float(pm.dy.mean()); boots = [pm.dy.values[rng.integers(0, k, k)].mean() for _ in range(nboot)]
        signs = rng.choice([-1, 1], size=(nperm, k)); perm = (signs * pm.dy.values).mean(axis=1); p = float((np.abs(perm) >= abs(est) - 1e-12).mean())
        loo = [np.delete(pm.dy.values, i).mean() for i in range(k)]
        rows.append(dict(factor=f, reference=ref, level=lvl, papers=k, pairs=int(pm.n.sum()), delta_retention_pp=100 * est, ci_lo=100 * float(np.percentile(boots, 2.5)), ci_hi=100 * float(np.percentile(boots, 97.5)),
                         delta_retained80_pp=100 * float(pm.dc.mean()), frac_papers_positive=float((pm.dy > 0).mean()), perm_p=p, loo_min=100 * float(min(loo)), loo_max=100 * float(max(loo))))
    r = pd.DataFrame(rows)
    if len(r):
        o_ = np.argsort(r.perm_p.values); m = len(r); q = np.empty(m); prev = 1.0
        for rank, idx in enumerate(o_[::-1]):
            i = m - rank; prev = min(prev, r.perm_p.values[idx] * m / i); q[idx] = prev
        r["bh_q"] = q
    return r.sort_values("perm_p")
res = {}
for w in [0.05, 0.1, 0.2]:
    pairs = contrasts(w); o = orient(pairs); s = summarize(o); res[str(w)] = s
    log(f"window {w}: matched pairs {len(pairs)}, oriented {len(o)}, contrasts with >=3 papers: {len(s)}")
    s.to_csv(DATA / f"within_paper_contrasts_window{str(w).replace('.', '')}.csv", index=False)
main = res["0.1"]; pd.set_option("display.width", 220); log("\n" + main.round(3).to_string(index=False))
def fe_reg(df, cols):
    X = pd.get_dummies(df[cols], drop_first=False).astype(float); ref = [c for c in X.columns if any(c.endswith("_" + d["F_" + f.split("F_")[1]].value_counts().index[0]) and c.startswith(f) for f in cols)]
    X = X.drop(columns=ref); X["logt"] = df["logt"].values; y = df.y.values; g = df.paper_id.values
    Xd = X.values - pd.DataFrame(X.values).groupby(g).transform("mean").values; yd = y - pd.Series(y).groupby(g).transform("mean").values
    keep = np.abs(Xd).sum(axis=0) > 1e-9; Xd = Xd[:, keep]; names = X.columns[keep]
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None); res_ = yd - Xd @ beta; XtXi = np.linalg.pinv(Xd.T @ Xd)
    meat = np.zeros((Xd.shape[1],) * 2)
    for pid in np.unique(g):
        m = g == pid; s = Xd[m].T @ res_[m]; meat += np.outer(s, s)
    V = XtXi @ meat @ XtXi; se = np.sqrt(np.diag(V))
    return pd.DataFrame(dict(term=names, beta_pp=100 * beta, se_pp=100 * se, t=beta / se)).sort_values("t", key=np.abs, ascending=False)
sub = d[d.groupby("paper_id").paper_id.transform("size") >= 2]
fe = fe_reg(sub, ["F_architecture", "F_encapsulation", "F_additive", "F_B_site", "F_deposition"]); fe.to_csv(DATA / "within_paper_fixed_effects.csv", index=False)
log(f"paper fixed-effects regression on {len(sub)} cells in {sub.paper_id.nunique()} multi-cell papers:\n" + fe.round(2).head(12).to_string(index=False))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
m_ = main[main.papers >= 4].sort_values("delta_retention_pp").reset_index(drop=True)
if len(m_):
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(m_) + 1.2))
    for i, r in m_.iterrows():
        col = "#D55E00" if r.bh_q < 0.10 else "#0072B2"; ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=1.4); ax.plot(r.delta_retention_pp, i, "o", color=col, ms=4)
    ax.axvline(0, color="k", lw=0.7); ax.set_yticks(range(len(m_))); ax.set_yticklabels([f"{r.factor}: {r.level} vs {r.reference} ({r.papers} papers)" for r in m_.itertuples()], fontsize=6.5)
    ax.set_xlabel("Within-paper change in retention at end of test (percentage points)"); plt.tight_layout(); plt.savefig(FIG / "fig_within_paper_contrasts.png", dpi=300); plt.close()
json.dump({k: v.to_dict("records") for k, v in res.items()}, open(DATA / "within_paper_contrasts.json", "w"), indent=1, default=str); log("DONE")
