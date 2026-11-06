import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
PROJ = Path(__file__).resolve().parent; DATA = PROJ / "data"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
d = pd.read_parquet(DATA / "stability_table.parquet")
d["y"] = d.retention_end.clip(0, 1.2); d["yc"] = (d.retention_end >= 0.8).astype(int)
d["tbin"] = np.round(np.log10(d.t_exp_h) / 0.1).astype(int)
d["stack"] = d[["Cell_architecture", "ETL_stack_sequence", "HTL_stack_sequence", "Backcontact_stack_sequence", "Encapsulation"]].astype(str).agg("|".join, axis=1)
LEVELS = {
    "L0 composition": ["comp_group"],
    "L1 composition+ISOS+time": ["comp_group", "isos", "tbin"],
    "L2 L1 within one paper": ["comp_group", "isos", "tbin", "paper_id"],
    "L3 L1+device stack": ["comp_group", "isos", "tbin", "stack"],
    "L4 L3 within one paper": ["comp_group", "isos", "tbin", "stack", "paper_id"],
}
def ceiling(df, keys, y="y", classify=False):
    g = df.groupby(keys, sort=False); n = g[y].transform("size"); m = n >= 2; sub = df[m]
    if len(sub) < 30: return None
    gs = sub.groupby(keys, sort=False)[y]; mean = gs.transform("mean"); cnt = gs.transform("size")
    ssw = float(((sub[y] - mean) ** 2).sum()); dof = len(sub) - gs.ngroups
    within = ssw / dof; tot = float(sub[y].var(ddof=1))
    loo = (mean * cnt - sub[y]) / (cnt - 1)
    out = dict(cells=int(len(sub)), share_of_cells=float(len(sub) / len(df)), groups=int(gs.ngroups), papers=int(sub.paper_id.nunique()), within_var=within, total_var=tot,
               ceiling_r2=float(1 - within / tot), oracle_r2=float(1 - ((sub[y] - loo) ** 2).sum() / ((sub[y] - sub[y].mean()) ** 2).sum()))
    if classify and sub[y].nunique() == 2:
        try: out["oracle_auc"] = float(roc_auc_score(sub[y], loo))
        except Exception: out["oracle_auc"] = np.nan
    return out
res = {"retention": {}, "retained80": {}}
for name, keys in LEVELS.items():
    res["retention"][name] = ceiling(d, keys, "y"); res["retained80"][name] = ceiling(d, keys, "yc", classify=True)
    r = res["retention"][name]; log(f"{name}: " + (f"cells {r['cells']} ({100*r['share_of_cells']:.0f}%), ceiling R2 {r['ceiling_r2']:.3f}, oracle R2 {r['oracle_r2']:.3f}" if r else "too few replicates"))
rng = np.random.default_rng(0); P = d.paper_id.unique(); ints = {k: {"ceiling_r2": [], "oracle_r2": []} for k in LEVELS}
for rep in range(200):
    keep = set(rng.choice(P, int(0.8 * len(P)), replace=False)); s = d[d.paper_id.isin(keep)]
    for name, keys in LEVELS.items():
        r = ceiling(s, keys, "y")
        if r: ints[name]["ceiling_r2"].append(r["ceiling_r2"]); ints[name]["oracle_r2"].append(r["oracle_r2"])
for name in LEVELS:
    for k in ["ceiling_r2", "oracle_r2"]:
        v = ints[name][k]
        if v and res["retention"][name]: res["retention"][name][k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def icc1(df, y):
    g = df.groupby("paper_id")[y]; k = g.size(); df = df[df.paper_id.map(k) >= 2]; g = df.groupby("paper_id")[y]; k = g.size(); N = len(df); a = len(k)
    mean_all = df[y].mean(); msb = float((k * (g.mean() - mean_all) ** 2).sum() / (a - 1)); msw = float(((df[y] - g.transform("mean")) ** 2).sum() / (N - a)); k0 = (N - (k ** 2).sum() / N) / (a - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw)), int(a), int(N)
X = pd.concat([pd.get_dummies(d.isos, drop_first=True).astype(float), pd.DataFrame({"lt": np.log10(d.t_exp_h), "lt2": np.log10(d.t_exp_h) ** 2, "one": 1.0})], axis=1).values
for y in ["y", "yc"]:
    beta, *_ = np.linalg.lstsq(X, d[y].values.astype(float), rcond=None); d[y + "_res"] = d[y] - X @ beta
res["between_paper_icc"] = {y: dict(zip(["icc", "papers", "cells"], icc1(d, y))) for y in ["y", "y_res", "yc", "yc_res"]}
log("ICC (between papers): " + json.dumps({k: round(v["icc"], 3) for k, v in res["between_paper_icc"].items()}))
pdp = pd.read_parquet(DATA / "pdp_featurized.parquet", columns=["formula_str", "PCE"])
pdp["comp_group"] = pdp.formula_str; pdp["paper_id"] = "NA"; pdp["y"] = pdp.PCE
r = ceiling(pdp, ["comp_group"], "y"); res["pce_L0"] = r
log(f"PCE L0 (composition only, all cells): share in replicated groups {100*r['share_of_cells']:.0f}%, ceiling R2 {r['ceiling_r2']:.3f}, oracle R2 {r['oracle_r2']:.3f}")
json.dump(res, open(DATA / "stability_noise_ceiling.json", "w"), indent=1, default=str); log("DONE")
