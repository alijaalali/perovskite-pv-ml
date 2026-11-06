import json, time, re
from pathlib import Path
import numpy as np, pandas as pd
PROJ = Path(__file__).resolve().parent; DATA = PROJ / "data"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
d = pd.read_parquet(DATA / "stability_table.parquet"); d["y"] = d.retention_end.clip(0, 1.2); d["tbin"] = np.round(np.log10(d.t_exp_h) / 0.1).astype(int)
d["stack"] = d[["Cell_architecture", "ETL_stack_sequence", "HTL_stack_sequence", "Backcontact_stack_sequence", "Encapsulation"]].astype(str).agg("|".join, axis=1)
LEVELS = {"L0 composition": ["comp_group"], "L1 composition+ISOS+time": ["comp_group", "isos", "tbin"], "L2 L1 within one paper": ["comp_group", "isos", "tbin", "paper_id"],
          "L3 L1+device stack": ["comp_group", "isos", "tbin", "stack"], "L4 L3 within one paper": ["comp_group", "isos", "tbin", "stack", "paper_id"]}
def ceiling_arrays(y, gid):
    s = pd.Series(y); g = s.groupby(gid); cnt = g.transform("size").values; m = cnt >= 2
    if m.sum() < 30: return None
    ys, gi = y[m], gid[m]; gs = pd.Series(ys).groupby(gi); mean = gs.transform("mean").values; c = gs.transform("size").values
    within = float(((ys - mean) ** 2).sum() / (len(ys) - gs.ngroups)); tot = float(ys.var(ddof=1)); loo = (mean * c - ys) / (c - 1)
    return dict(ceiling=1 - within / tot, oracle=float(1 - ((ys - loo) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()), n=int(m.sum()), groups=int(gs.ngroups))
def keyid(df, keys): return df.groupby(keys, sort=False).ngroup().values
out = {"nested": {}}
for base_name in ["L4 L3 within one paper", "L3 L1+device stack", "L1 composition+ISOS+time"]:
    sub = d[d.groupby(LEVELS[base_name], sort=False).y.transform("size") >= 2]; res = {}
    for name, keys in LEVELS.items():
        r = ceiling_arrays(sub.y.values, keyid(sub, keys))
        if r: res[name] = r
    out["nested"][base_name] = dict(cells=int(len(sub)), levels=res); log(f"nested on cells replicated at '{base_name}' ({len(sub)} cells): " + ", ".join(f"{k.split()[0]} {v['ceiling']:.3f}" for k, v in res.items()))
rng = np.random.default_rng(0); sim = []
for lname in ["L1 composition+ISOS+time", "L3 L1+device stack"]:
    gid_full = keyid(d, LEVELS[lname]); cnt = pd.Series(gid_full).map(pd.Series(gid_full).value_counts()).values; m = cnt >= 2; gid = pd.factorize(gid_full[m])[0]; n_g = len(np.unique(gid))
    for rho in [0.0, 0.05, 0.13, 0.27, 0.5]:
        est = []; orc = []
        for rep in range(300):
            mu = rng.normal(0, np.sqrt(rho), n_g); y = mu[gid] + rng.normal(0, np.sqrt(1 - rho), len(gid)); r = ceiling_arrays(y, gid); est.append(r["ceiling"]); orc.append(r["oracle"])
        sim.append(dict(structure=lname, cells=int(len(gid)), groups=int(n_g), true_share=rho, est_mean=float(np.mean(est)), est_lo=float(np.percentile(est, 2.5)), est_hi=float(np.percentile(est, 97.5)), oracle_mean=float(np.mean(orc)), oracle_lo=float(np.percentile(orc, 2.5)), oracle_hi=float(np.percentile(orc, 97.5)),
                        true_oracle_asymptotic_note="finite-group lookup oracle R2 is below the share rho because group means are estimated from other cells"))
        log(f"sim {lname} rho={rho:.2f}: ceiling {np.mean(est):+.3f} [{np.percentile(est,2.5):+.3f},{np.percentile(est,97.5):+.3f}]; lookup oracle {np.mean(orc):+.3f}")
out["simulation"] = sim
def els(s): return re.findall(r"([A-Z][a-z]?)([0-9.]*)", s)
def frac(s):
    c = {e: float(n) if n else 1.0 for e, n in els(s)}; nC, nN = c.get("C", 0), c.get("N", 0); FA = max(nN - nC, 0); MA = max(2 * nC - nN, 0); A = FA + MA + c.get("Cs", 0) + c.get("Rb", 0) + c.get("K", 0)
    X = c.get("I", 0) + c.get("Br", 0) + c.get("Cl", 0); B = sum(c.get(e, 0) for e in ["Pb", "Sn", "Ge", "Bi", "Sb", "Ag", "In", "Cu", "Ti", "Zn", "Sr", "Eu", "Co", "Mn"])
    return dict(fMA=MA / A if A else 0, fFA=FA / A if A else 0, fCs=c.get("Cs", 0) / A if A else 0, fI=c.get("I", 0) / X if X else 0, fBr=c.get("Br", 0) / X if X else 0, fCl=c.get("Cl", 0) / X if X else 0, fSn=c.get("Sn", 0) / B if B else 0, fPb=c.get("Pb", 0) / B if B else 0)
SF = pd.DataFrame([frac(s) for s in d.formula_str.astype(str)])
RA = {"Cs": 1.88, "MA": 2.17, "FA": 2.53}; RB = {"Pb": 1.19, "Sn": 1.10}; RX = {"I": 2.20, "Br": 1.96, "Cl": 1.81}
rA = SF.fCs * RA["Cs"] + SF.fMA * RA["MA"] + SF.fFA * RA["FA"]; rB = SF.fPb * RB["Pb"] + SF.fSn * RB["Sn"]; rX = SF.fI * RX["I"] + SF.fBr * RX["Br"] + SF.fCl * RX["Cl"]
SF["tolerance_factor"] = np.where(rB > 0, (rA + rX) / (np.sqrt(2) * (rB + rX).replace(0, np.nan)), np.nan); SF["tolerance_factor"] = SF.tolerance_factor.fillna(SF.tolerance_factor.median())
SF.to_parquet(DATA / "stability_site_fractions.parquet")
g = d.paper_id.values; y = d.y.values
def demean(M): return M - pd.DataFrame(M).groupby(g).transform("mean").values
def r2_within(M):
    Xd = demean(M); yd = y - pd.Series(y).groupby(g).transform("mean").values; keep = np.abs(Xd).sum(0) > 1e-9; Xd = Xd[:, keep]
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None); return float(1 - np.sum((yd - Xd @ beta) ** 2) / np.sum(yd ** 2)), int(keep.sum())
oh = lambda c, k=8: pd.get_dummies(d[c].astype(str).where(d[c].astype(str).isin(d[c].astype(str).value_counts().head(k).index), "other"), drop_first=True).astype(float).values
blocks = {"exposure time + ISOS": np.column_stack([np.log10(d.t_exp_h), np.log10(d.t_exp_h) ** 2, oh("isos")]), "site fractions + tolerance factor": SF.values, "A-site and B-site classes": np.column_stack([oh("a_site"), oh("b_site_class")]),
          "device stack (architecture, ETL, HTL, back contact, encapsulation)": np.column_stack([oh("Cell_architecture"), oh("ETL_stack_sequence"), oh("HTL_stack_sequence"), oh("Backcontact_stack_sequence"), oh("Encapsulation")])}
within_share = 1 - (lambda: (lambda a, k, N: 0)(0, 0, 0))() if False else None
dec = {}; base_cols = blocks["exposure time + ISOS"]
dec["within_paper_variance_share_of_total"] = float(np.sum((y - pd.Series(y).groupby(g).transform("mean").values) ** 2) / np.sum((y - y.mean()) ** 2))
for k, M in blocks.items(): dec[k] = r2_within(M)
dec["time+ISOS + site fractions"] = r2_within(np.column_stack([base_cols, SF.values])); dec["time+ISOS + site fractions + A/B-site classes"] = r2_within(np.column_stack([base_cols, SF.values, blocks["A-site and B-site classes"]]))
dec["time+ISOS + device stack"] = r2_within(np.column_stack([base_cols, blocks["device stack (architecture, ETL, HTL, back contact, encapsulation)"]])); dec["all blocks"] = r2_within(np.column_stack([base_cols, SF.values, blocks["A-site and B-site classes"], blocks["device stack (architecture, ETL, HTL, back contact, encapsulation)"]]))
out["within_paper_decomposition"] = dec; log(json.dumps(dec))
json.dump(out, open(DATA / "noise_ceiling_validation.json", "w"), indent=1); log("DONE")
