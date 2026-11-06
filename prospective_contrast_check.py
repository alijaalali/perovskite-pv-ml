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

d_full = d.copy()
def oriented_for(mask, window=0.1):
    global d
    d = d_full[mask].reset_index(drop=True); pairs = contrasts(window); d = d_full; return orient(pairs)
tr_m = (d_full.year <= 2018).values; te_m = (d_full.year >= 2019).values
o_tr = oriented_for(tr_m); o_te = oriented_for(te_m)
log(f"train oriented pairs {len(o_tr)} (papers {o_tr.paper.nunique()}), test oriented pairs {len(o_te)} (papers {o_te.paper.nunique()})")
def pm(o):
    return o.groupby(["factor", "ref", "level", "paper"]).agg(dy=("dy", "mean")).reset_index()
a, b = pm(o_tr), pm(o_te); rows = []
rng = np.random.default_rng(0)
for (f, r, l), g in a.groupby(["factor", "ref", "level"]):
    if len(g) < 3: continue
    h = b[(b.factor == f) & (b.ref == r) & (b.level == l)]
    est_tr = float(g.dy.mean()); row = dict(factor=f, reference=r, level=l, train_papers=len(g), train_pp=100 * est_tr, test_papers=len(h))
    if len(h) >= 1:
        row.update(test_pp=100 * float(h.dy.mean()), frac_test_papers_same_sign=float((np.sign(h.dy) == np.sign(est_tr)).mean()))
        if len(h) >= 2: bs = [h.dy.values[rng.integers(0, len(h), len(h))].mean() for _ in range(2000)]; row.update(test_lo=100 * float(np.percentile(bs, 2.5)), test_hi=100 * float(np.percentile(bs, 97.5)))
    rows.append(row)
pv = pd.DataFrame(rows).sort_values("train_papers", ascending=False); pv.to_csv(DATA / "prospective_contrasts.csv", index=False)
log("PROSPECTIVE (train <=2018 -> test 2019-2020), effects in percentage points of retention:\n" + pv.round(2).to_string(index=False))
t = pv.dropna(subset=["test_pp"]); 
if len(t): log(f"contrasts testable in both periods: {len(t)}; sign agreement train vs test: {int((np.sign(t.train_pp) == np.sign(t.test_pp)).sum())}/{len(t)}; Spearman(train,test)={t[['train_pp','test_pp']].corr(method='spearman').iloc[0,1]:.2f}")
S = {w: pd.read_csv(DATA / f"within_paper_contrasts_window{w}.csv") for w in ["005", "01", "02"]}
key = [("A_site", "FA; MA"), ("B_site", "lead_mixed"), ("encapsulation", "True"), ("deposition", "Spin-coating >> Spin-coating")]
for f, l in key:
    out = []
    for w, s in S.items():
        x = s[(s.factor == f) & (s.level == l)]
        out.append(f"w{w}: " + (f"{x.delta_retention_pp.iloc[0]:+.1f} pp ({int(x.papers.iloc[0])} papers)" if len(x) else "n/a"))
    log(f"window sensitivity {f}/{l}: " + " | ".join(out))
rawb = pd.read_csv(DATA / "perovskite_raw.csv", usecols=["Perovskite_composition_b_ions", "Perovskite_composition_b_ions_coefficients"], low_memory=False).loc[d_full.raw_idx.values].reset_index(drop=True)
lm = d_full.b_site_class.values == "lead_mixed"
log("lead_mixed B-site ions among stability cells:\n" + rawb[lm].Perovskite_composition_b_ions.value_counts().head(8).to_string())
pairs_lm = contrasts(0.1); olm = orient(pairs_lm); olm = olm[(olm.factor == "B_site") & (olm.level == "lead_mixed")]
bmap = dict(zip(d_full.paper_id, rawb.Perovskite_composition_b_ions)); log("papers contributing lead_mixed contrasts and their B-site ions:\n" + olm.groupby("paper").agg(dy=("dy", "mean"), n=("dy", "size")).assign(b=lambda x: [bmap.get(p) for p in x.index]).round(3).to_string())
log("DONE")
