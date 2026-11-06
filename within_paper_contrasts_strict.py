import json, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
PROJ = Path(__file__).resolve().parent; DATA = PROJ / "data"; FIG = PROJ / "figures"; FIG.mkdir(exist_ok=True)
T0 = time.time(); STRICT_OFF = False
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
d = pd.read_parquet(DATA / "stability_table.parquet")
raw = pd.read_csv(DATA / "perovskite_raw.csv", usecols=["Perovskite_additives_compounds", "Perovskite_deposition_quenching_media", "Perovskite_deposition_solvents", "Perovskite_deposition_quenching_media_additives_compounds", "Perovskite_deposition_number_of_deposition_steps", "ETL_additives_compounds", "HTL_additives_compounds", "Perovskite_deposition_synthesis_atmosphere", "Perovskite_deposition_aggregation_state_of_reactants"], low_memory=False).loc[d.raw_idx.values].reset_index(drop=True)
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
STRICT_COLS = ["Perovskite_additives_compounds", "Perovskite_deposition_solvents", "Perovskite_deposition_quenching_media", "Perovskite_deposition_quenching_media_additives_compounds", "Perovskite_deposition_number_of_deposition_steps", "ETL_additives_compounds", "HTL_additives_compounds", "Perovskite_deposition_synthesis_atmosphere", "Perovskite_deposition_aggregation_state_of_reactants"]
for c in STRICT_COLS: d["S_" + c] = raw[c].astype(str).str.strip().values
STRICT = ["S_" + c for c in STRICT_COLS]
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
        F = gi[FACTORS].values; S = gi[STRICT].values
        for a, b in zip(iu, ju):
            diff = [k for k in range(len(FACTORS)) if F[a, k] != F[b, k]]
            if len(diff) != 1: continue
            if not np.all(S[a] == S[b]) and not STRICT_OFF: continue
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
pairs_all = contrasts(0.1); log(f"strict pairs {len(pairs_all)}"); o = orient(pairs_all); s_ = summarize(o); s_.to_csv(DATA / "within_paper_contrasts_strict_window01.csv", index=False)
STRICT_OFF = True; pairs_loose = contrasts(0.1); log(f"loose pairs {len(pairs_loose)}")
pd.set_option("display.width", 220); log("\n" + s_.round(2).to_string(index=False)); log("DONE")
