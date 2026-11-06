import json, sys, time, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb, lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
warnings.filterwarnings("ignore")
P = Path(__file__).resolve().parent; D = P / "data"; sys.path.insert(0, str(Path(__file__).parent))
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"; T0 = time.time(); NJ = 16; SEED = 0
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
from phase1_pipeline import build_formula_dict
from pymatgen.core import Composition
raw = pd.read_csv(D / "perovskite_raw.csv", low_memory=False)
single = raw[raw["Perovskite_dimension_list_of_layers"].astype(str).str.strip() == "3.0"].copy()
single["formula_dict"] = [(lambda r: r[0] if r is not None else None)(build_formula_dict(row)) for _, row in single.iterrows()]
parsed = single[single["formula_dict"].notna()].copy()
parsed["fs"] = parsed["formula_dict"].apply(lambda dd: "".join(f"{el}{amt:.6f}" for el, amt in dd.items()))
def ok(s):
    try: Composition(s); return True
    except Exception: return False
parsed = parsed[parsed.fs.apply(ok)].copy(); parsed["PCE"] = pd.to_numeric(parsed.JV_default_PCE, errors="coerce"); parsed = parsed.dropna(subset=["PCE"]); parsed = parsed[(parsed.PCE >= 0) & (parsed.PCE <= 35)]
feat = pd.read_parquet(D / "pdp_featurized.parquet"); assert len(parsed) == len(feat) and (parsed.fs.values == feat.formula_str.values).all() and np.allclose(parsed.PCE.values, feat.PCE.values); log("alignment verified")
d = feat.reset_index(drop=True); r = parsed.reset_index(drop=True)
if QUICK: keep = np.random.default_rng(0).choice(len(d), 4000, replace=False); d = d.iloc[keep].reset_index(drop=True); r = r.iloc[keep].reset_index(drop=True)
paper = r.Ref_DOI_number.astype(str).str.lower().values; year = pd.to_datetime(r.Ref_publication_date, errors="coerce").dt.year.values; pu, pidx = np.unique(paper, return_inverse=True)
AU = json.load(open(D / "paper_authors_openalex.json")); la = np.array([(AU.get(p) or [{}])[-1].get("aid") or p for p in pu], dtype=object); lau, lidx = np.unique(la, return_inverse=True); last = lidx[pidx]
y = d.PCE.values; mag = [c for c in d.columns if c.startswith("MagpieData")]
def onehot(s, pre, minc=30):
    v = s.astype(str).str.strip().replace({"nan": "Unknown"}); v = v.where(v.map(v.value_counts()) >= minc, "other"); return pd.get_dummies(v, prefix=pre).astype(float)
def numcol(s, name):
    x = pd.to_numeric(s.astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"); return pd.DataFrame({name + "_miss": x.isna().astype(float), name: x.fillna(x.median())})
tok = r.Perovskite_additives_compounds.fillna("Unknown").astype(str).str.split(";").apply(lambda l: [t.strip() for t in l]); cnt = pd.Series([t for l in tok for t in l if t not in ("Unknown", "Undoped", "")]).value_counts(); top = list(cnt[cnt >= 60].index)
add = pd.DataFrame({"add_" + t: tok.apply(lambda l, t=t: float(t in l)) for t in top}); add["additive_known"] = tok.apply(lambda l: float("Unknown" not in l))
stack = pd.concat([onehot(r.Cell_architecture, "arch"), onehot(r.ETL_stack_sequence, "etl"), onehot(r.HTL_stack_sequence, "htl"), onehot(r.Backcontact_stack_sequence, "bc"), numcol(r.Cell_area_measured, "area")], axis=1)
proc = pd.concat([add, onehot(r.Perovskite_deposition_procedure, "dep"), onehot(r.Perovskite_deposition_quenching_media, "quench"), onehot(r.Perovskite_deposition_solvents, "solv"), onehot(r.Perovskite_deposition_synthesis_atmosphere, "atm"),
                  numcol(r.Perovskite_deposition_number_of_deposition_steps, "nsteps"), numcol(r.Perovskite_thickness, "pkthick"), numcol(r.Perovskite_band_gap, "bandgap")], axis=1)
FS = {"composition (Magpie)": d[mag].astype(float), "+ device stack": pd.concat([d[mag].astype(float), stack], axis=1), "+ stack + process": pd.concat([d[mag].astype(float), stack, proc], axis=1)}
log("dims " + str({k: v.shape[1] for k, v in FS.items()}))
rng = np.random.default_rng(1); NB = 40 if QUICK else 200; BW = rng.multinomial(len(pu), np.ones(len(pu)) / len(pu), size=NB)
def wr2(yv, p, w): mu = np.average(yv, weights=w); return 1 - np.sum(w * (yv - p) ** 2) / np.sum(w * (yv - mu) ** 2)
def ci(v): v = np.array(v); v = v[np.isfinite(v)]; return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
fam = d.family_key.values
def splits(proto):
    idx = np.arange(len(d))
    if proto == "random": return list(KFold(5, shuffle=True, random_state=SEED).split(idx))
    if proto == "paper": return list(GroupKFold(5).split(idx, groups=pidx))
    if proto == "last_author": return list(GroupKFold(5).split(idx, groups=last))
    if proto == "family": return list(GroupKFold(5).split(idx, groups=fam))
    if proto == "temporal": return [(np.where(year <= 2018)[0], np.where(year >= 2019)[0])]
def mk(name):
    if name == "XGBoost": return ("raw", xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, tree_method="hist"))
    if name == "LightGBM": return ("raw", lgb.LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, verbose=-1))
    return ("scaled", RidgeCV(alphas=np.logspace(-2, 4, 25)))
def fit_pred(m, Xtr, ytr, Xte):
    kind, mod = m
    if kind == "scaled": sc = StandardScaler().fit(Xtr); Xtr, Xte = np.clip(sc.transform(Xtr), -8, 8), np.clip(sc.transform(Xte), -8, 8)
    mod.fit(Xtr, ytr); return mod.predict(Xte)
MODELS = ["XGBoost", "LightGBM", "Ridge"]; PROT = ["random", "paper", "last_author", "temporal", "family"]
if QUICK: MODELS, PROT = ["XGBoost"], ["paper", "temporal"]
res = []; w1 = np.ones(len(d)); wpap = 1.0 / pd.Series(pidx).map(pd.Series(pidx).value_counts()).values
for proto in PROT:
    sp = splits(proto)
    for mname in MODELS:
        for fs, Xdf in FS.items():
            X = Xdf.values.astype(float); pred = np.full(len(d), np.nan)
            for tr, te in sp: pred[te] = fit_pred(mk(mname), X[tr], y[tr], X[te])
            m = ~np.isnan(pred); bs = [wr2(y[m], pred[m], bw[pidx][m]) for bw in BW]; bsw = [wr2(y[m], pred[m], bw[pidx][m] * wpap[m]) for bw in BW]
            res.append(dict(protocol=proto, model=mname, features=fs, n_test=int(m.sum()), r2=float(wr2(y[m], pred[m], w1[m])), r2_ci=ci(bs), r2_paper_weighted=float(wr2(y[m], pred[m], wpap[m])), r2_paper_weighted_ci=ci(bsw)))
            log(f"{proto:12s} {mname:9s} {fs:22s} R2={res[-1]['r2']:.3f} {res[-1]['r2_ci']}")
    json.dump(res, open(D / ("efficiency_device_features" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1)
def ceiling_arrays(yv, gid):
    s = pd.Series(yv); cnt = s.groupby(gid).transform("size").values; m = cnt >= 2
    if m.sum() < 30: return None
    ys, gi = yv[m], gid[m]; gs = pd.Series(ys).groupby(gi); mean = gs.transform("mean").values; c = gs.transform("size").values
    within = float(((ys - mean) ** 2).sum() / (len(ys) - gs.ngroups)); tot = float(ys.var(ddof=1)); return dict(ceiling=float(1 - within / tot), cells=int(m.sum()), groups=int(gs.ngroups), share=float(m.mean()))
comp = d.formula_str.astype(str); stk = (r.Cell_architecture.astype(str) + "|" + r.ETL_stack_sequence.astype(str) + "|" + r.HTL_stack_sequence.astype(str) + "|" + r.Backcontact_stack_sequence.astype(str))
prc = (r.Perovskite_deposition_procedure.astype(str) + "|" + r.Perovskite_deposition_solvents.astype(str) + "|" + r.Perovskite_deposition_quenching_media.astype(str) + "|" + r.Perovskite_additives_compounds.astype(str) + "|" + r.Perovskite_deposition_synthesis_atmosphere.astype(str))
LV = {"L0 composition": [comp], "L1 + device stack": [comp, stk], "L2 + stack + process": [comp, stk, prc], "L3 L2 within one paper": [comp, stk, prc, pd.Series(paper)]}
def gid_of(parts): return pd.factorize(pd.Series(["\x1f".join(map(str, t)) for t in zip(*parts)]))[0]
ceil = {"own_subset": {}, "nested_on_L3": {}}
for name, parts in LV.items(): ceil["own_subset"][name] = ceiling_arrays(y, gid_of(parts))
cnt3 = pd.Series(gid_of(LV["L3 L2 within one paper"])).map(pd.Series(gid_of(LV["L3 L2 within one paper"])).value_counts()).values; sub = cnt3 >= 2
for name, parts in LV.items(): ceil["nested_on_L3"][name] = ceiling_arrays(y[sub], gid_of([pd.Series(np.asarray(p)[sub]) for p in parts]))
for k, v in ceil.items():
    for n, r_ in v.items(): log(f"ceiling {k} {n}: " + (f"{r_['ceiling']:.3f} ({r_['cells']} cells, {100*r_['share']:.0f}%)" if r_ else "NA"))
json.dump(ceil, open(D / ("efficiency_ceilings" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1)
mp = json.load(open(D / "multipartition_robustness.json")); rows = mp["rows"] if isinstance(mp, dict) and "rows" in mp else mp.get("partitions", mp)
pr = pd.DataFrame(rows); log("partition columns: " + str(list(pr.columns))); pr.to_csv(D / "efficiency_partitions_flat.csv", index=False)
log("DONE")
