import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb, lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score
import torch
warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parent; DATA = PROJ / "data"; RES = PROJ / "data"
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
NJ = 16; SEED = 0
d = pd.read_parquet(DATA / "stability_table.parquet")
if QUICK: d = d.sample(800, random_state=0).reset_index(drop=True)
mag = [c for c in d.columns if c.startswith("MagpieData")]
def onehot(cols, minc=30):
    parts = []
    for c in cols:
        v = d[c].astype(str); vc = v.value_counts(); v = v.where(v.map(vc) >= minc, "other")
        parts.append(pd.get_dummies(v, prefix=c).astype(float))
    return pd.concat(parts, axis=1)
def numfeat(cols):
    out = {}
    for c in cols:
        x = pd.to_numeric(d[c], errors="coerce"); out[c + "_miss"] = x.isna().astype(float); out[c] = x.fillna(x.median())
    return pd.DataFrame(out)
F = {}
F["time"] = pd.DataFrame({"log_t_exp": np.log10(d.t_exp_h)})
F["comp"] = d[mag + ["n_a_cations"]].astype(float)
F["device"] = pd.concat([onehot(["Cell_architecture", "ETL_stack_sequence", "HTL_stack_sequence", "Backcontact_stack_sequence", "Perovskite_deposition_procedure", "Encapsulation"]), numfeat(["cell_area_cm2", "band_gap"])], axis=1)
F["protocol"] = pd.concat([onehot(["isos", "Stability_atmosphere", "Stability_light_load_condition", "Stability_potential_bias_load_condition", "Stability_temperature_load_condition"], 20), numfeat(["stab_temp_C", "stab_RH", "stab_light_intensity"])], axis=1)
sets = {"time": ["time"], "time+comp": ["time", "comp"], "time+device": ["time", "device"], "time+protocol": ["time", "protocol"], "all": ["time", "comp", "device", "protocol"]}
def X_of(name): return pd.concat([F[k] for k in sets[name]], axis=1).values.astype(float)
log("feature dims: " + str({k: v.shape[1] for k, v in F.items()}))
y_reg = d.retention_end.clip(0, 1.2).values
y_clf = (d.retention_end >= 0.8).astype(int).values
L = np.where(d.status == "event", d.t80_h, np.where(d.status == "interval", 1e-3, d.t_exp_h)).astype(float)
U = np.where(d.status == "event", d.t80_h, np.where(d.status == "interval", d.t_exp_h, np.inf)).astype(float)
L = np.maximum(L, 1e-3)
paper = d.paper_id.values; papers_u, pidx = np.unique(paper, return_inverse=True)
def splits(proto, n_rep=1):
    idx = np.arange(len(d)); out = []
    if proto == "random": out = list(KFold(5, shuffle=True, random_state=SEED).split(idx))
    elif proto == "paper": out = list(GroupKFold(5).split(idx, groups=paper))
    elif proto == "composition": out = list(GroupKFold(5).split(idx, groups=d.comp_group.values))
    elif proto == "family": out = list(GroupKFold(5).split(idx, groups=d.family_key.values))
    elif proto == "temporal":
        tr = np.where(d.year <= 2018)[0]; te = np.where(d.year >= 2019)[0]; out = [(tr, te)]
    elif proto == "protocol":
        top = d.isos.value_counts().head(6).index
        for p in top:
            te = np.where(d.isos == p)[0]; tr = np.where(d.isos != p)[0]; out.append((tr, te))
    return out
PROTOS = ["random", "paper", "composition", "family", "temporal", "protocol"]
def make_reg(name):
    if name == "Ridge": return ("scaled", RidgeCV(alphas=np.logspace(-2, 4, 25)))
    if name == "ElasticNet": return ("scaled", ElasticNetCV(l1_ratio=[.2, .6, 1.0], alphas=30, cv=3, n_jobs=NJ, max_iter=3000))
    if name == "RandomForest": return ("raw", RandomForestRegressor(300, min_samples_leaf=3, max_features=0.33, n_jobs=NJ, random_state=SEED))
    if name == "XGBoost": return ("raw", xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, tree_method="hist"))
    if name == "LightGBM": return ("raw", lgb.LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, verbose=-1))
    if name == "MLP": return ("scaled", MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, early_stopping=True, max_iter=300, random_state=SEED))
def make_clf(name):
    if name == "Logistic": return ("scaled", LogisticRegressionCV(Cs=8, cv=3, max_iter=2000, n_jobs=NJ))
    if name == "RandomForest": return ("raw", RandomForestClassifier(300, min_samples_leaf=3, max_features=0.33, n_jobs=NJ, random_state=SEED))
    if name == "XGBoost": return ("raw", xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, tree_method="hist"))
    if name == "LightGBM": return ("raw", lgb.LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, verbose=-1))
    if name == "MLP": return ("scaled", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2, early_stopping=True, max_iter=300, random_state=SEED))
def fit_pred(maker, Xtr, ytr, Xte, proba=False):
    kind, m = maker;
    if kind == "scaled":
        sc = StandardScaler().fit(Xtr); Xtr, Xte = np.clip(sc.transform(Xtr), -8, 8), np.clip(sc.transform(Xte), -8, 8)
    m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1] if proba else m.predict(Xte)
def fit_xgb_aft(Xtr, Ltr, Utr, Xte):
    dtr = xgb.DMatrix(Xtr); dtr.set_float_info("label_lower_bound", Ltr); dtr.set_float_info("label_upper_bound", Utr)
    p = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "extreme", "aft_loss_distribution_scale": 1.0, "tree_method": "hist", "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.5, "nthread": NJ, "seed": SEED}
    b = xgb.train(p, dtr, num_boost_round=300); return np.log(np.clip(b.predict(xgb.DMatrix(Xte)), 1e-3, 1e8))
def fit_weibull_aft(Xtr, Ltr, Utr, Xte, lam=3.0):
    sc = StandardScaler().fit(Xtr); A = torch.tensor(np.clip(sc.transform(Xtr), -8, 8), dtype=torch.float64); B = torch.tensor(np.clip(sc.transform(Xte), -8, 8), dtype=torch.float64)
    lL, lU = torch.tensor(np.log(Ltr)), torch.tensor(np.log(np.where(np.isinf(Utr), 1e9, Utr)))
    ex = torch.tensor((Ltr == Utr)); rc = torch.tensor(np.isinf(Utr)); ic = ~(ex | rc)
    w = torch.zeros(A.shape[1], dtype=torch.float64, requires_grad=True); b0 = torch.tensor([float(np.log(np.median(Ltr)))], dtype=torch.float64, requires_grad=True); ls = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b0, ls], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")
    def logS(z): return -torch.exp(z)
    def nll():
        opt.zero_grad(); mu = A @ w + b0; s = torch.exp(ls); zL = (lL - mu) / s; zU = (lU - mu) / s
        ll_ex = (zL - torch.exp(zL) - torch.log(s) - lL)
        ll_rc = logS(zL)
        SL, SU = torch.exp(logS(zL)), torch.exp(logS(zU)); ll_ic = torch.log(torch.clamp(SL - SU, min=1e-12))
        ll = torch.where(ex, ll_ex, torch.where(rc, ll_rc, ll_ic)).sum()
        loss = -ll / len(Ltr) + lam * (w ** 2).sum() / len(Ltr); loss.backward(); return loss
    opt.step(nll); return (B @ w + b0).detach().numpy()
rng = np.random.default_rng(1)
NB = 40 if QUICK else 200
BW = rng.multinomial(len(papers_u), np.ones(len(papers_u)) / len(papers_u), size=NB)
def wr2(y, p, w):
    mu = np.average(y, weights=w); return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - mu) ** 2)
def wrmse(y, p, w): return float(np.sqrt(np.sum(w * (y - p) ** 2) / np.sum(w)))
def ci(vals): v = np.array(vals); v = v[np.isfinite(v)]; return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]
TEXP = d.t_exp_h.values
def pairs_for(te_list, n=150000, matched=False):
    r = np.random.default_rng(5); out = []
    for te in te_list:
        te = np.asarray(te)
        if len(te) < 2: continue
        m_ = n * (6 if matched else 1); i = r.choice(te, m_); j = r.choice(te, m_); keep = (i != j) & ((U[i] < L[j]) | (U[j] < L[i]))
        if matched: keep &= np.abs(np.log10(TEXP[i]) - np.log10(TEXP[j])) < 0.15
        i, j = i[keep], j[keep]
        first_i = U[i] < L[j]; out.append((i, j, first_i))
    return out
def cindex(pred, prs, w=None):
    num = den = 0.0
    for i, j, fi in prs:
        wt = np.ones(len(i)) if w is None else w[pidx[i]] * w[pidx[j]]
        conc = np.where(fi, pred[i] < pred[j], pred[j] < pred[i]).astype(float); conc = np.where(pred[i] == pred[j], 0.5, conc)
        num += np.sum(wt * conc); den += np.sum(wt)
    return float(num / den) if den > 0 else np.nan

import re
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
RAWP = DATA / "perovskite_raw.csv"
need = ["Perovskite_additives_compounds", "Perovskite_additives_concentrations", "Perovskite_deposition_quenching_media", "Perovskite_deposition_quenching_induced_crystallisation",
        "Perovskite_deposition_solvents", "Perovskite_deposition_synthesis_atmosphere", "Perovskite_deposition_number_of_deposition_steps", "Perovskite_thickness",
        "Perovskite_composition_none_stoichiometry_components_in_excess", "Cell_stack_sequence", "Cell_flexible", "Cell_semitransparent", "Encapsulation_edge_sealing_materials", "Encapsulation_atmosphere_for_encapsulation"]
raw = pd.read_csv(RAWP, usecols=need, low_memory=False).loc[d.raw_idx.values].reset_index(drop=True)
def oh(series, prefix, minc=30):
    v = series.astype(str).str.strip().replace({"nan": "Unknown"}); vc = v.value_counts(); v = v.where(v.map(vc) >= minc, "other"); return pd.get_dummies(v, prefix=prefix).astype(float)
def numcol(series, name):
    x = pd.to_numeric(series.astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"); return pd.DataFrame({name + "_miss": x.isna().astype(float), name: x.fillna(x.median())})
tok = raw.Perovskite_additives_compounds.fillna("Unknown").astype(str).str.split(";").apply(lambda l: [t.strip() for t in l])
cnt = pd.Series([t for l in tok for t in l if t not in ("Unknown", "Undoped", "")]).value_counts(); top = list(cnt[cnt >= 40].index)
A = pd.DataFrame({"add_" + t: tok.apply(lambda l, t=t: float(t in l)) for t in top})
A["additive_known"] = tok.apply(lambda l: float("Unknown" not in l)); A["additive_present"] = tok.apply(lambda l: float(any(t not in ("Unknown", "Undoped", "") for t in l)))
proc = pd.concat([A, numcol(raw.Perovskite_additives_concentrations, "add_conc"), oh(raw.Perovskite_deposition_quenching_media, "quench"), oh(raw.Perovskite_deposition_solvents, "solv"), oh(raw.Perovskite_deposition_synthesis_atmosphere, "synatm"),
                  oh(raw.Perovskite_composition_none_stoichiometry_components_in_excess, "excess"), numcol(raw.Perovskite_deposition_number_of_deposition_steps, "nsteps"), numcol(raw.Perovskite_thickness, "pk_thick"),
                  pd.DataFrame({"quench_induced": raw.Perovskite_deposition_quenching_induced_crystallisation.astype(str).map({"True": 1.0, "False": 0.0}).fillna(0.5)})], axis=1)
stk = raw.Cell_stack_sequence.astype(str).str.split("|").apply(lambda l: [t.strip() for t in l]); c2 = pd.Series([t for l in stk for t in set(l)]).value_counts(); topk = list(c2[c2 >= 60].index)
S = pd.DataFrame({"stk_" + t: stk.apply(lambda l, t=t: float(t in l)) for t in topk}); S["n_layers"] = stk.apply(len).astype(float)
S["flexible"] = raw.Cell_flexible.astype(str).map({"True": 1.0, "False": 0.0}).fillna(0); S["semitransparent"] = raw.Cell_semitransparent.astype(str).map({"True": 1.0, "False": 0.0}).fillna(0)
S["enc_edge_known"] = (raw.Encapsulation_edge_sealing_materials.astype(str).str.strip() != "Unknown").astype(float); S["enc_atm_known"] = (raw.Encapsulation_atmosphere_for_encapsulation.astype(str).str.strip() != "Unknown").astype(float)
F["proc"] = proc; F["stack"] = S; F["pcefresh"] = numfeat(["pce_fresh"])
log("enriched dims: " + str({k: F[k].shape[1] for k in ["proc", "stack", "pcefresh"]}))
BASE = ["time", "comp", "device", "protocol"]
SETS2 = {"base": BASE, "+process": BASE + ["proc"], "+stack tokens": BASE + ["stack"], "+fresh PCE": BASE + ["pcefresh"], "+all enriched": BASE + ["proc", "stack", "pcefresh"]}
def X2(name): return pd.concat([F[k] for k in SETS2[name]], axis=1).values.astype(float)
def mk(name):
    if name == "CatBoost": return ("raw", CatBoostRegressor(iterations=500, depth=6, learning_rate=0.06, verbose=0, thread_count=NJ, random_seed=SEED))
    if name == "Stacking": return ("scaled", StackingRegressor([("rf", RandomForestRegressor(150, min_samples_leaf=3, max_features=0.33, n_jobs=NJ, random_state=SEED)), ("xgb", xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.06, subsample=.8, colsample_bytree=.5, n_jobs=NJ, tree_method="hist", random_state=SEED)), ("lgb", lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.06, subsample=.8, subsample_freq=1, colsample_bytree=.5, n_jobs=NJ, verbose=-1, random_state=SEED)), ("ridge", RidgeCV(alphas=np.logspace(-1, 4, 15)))], final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)), cv=3, n_jobs=1))
    return make_reg(name)
MODELS = ["Ridge", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "Stacking"]
if QUICK: MODELS, SETS2_KEYS, PROTOS = ["Ridge", "XGBoost"], ["base", "+all enriched"], ["paper"]
else: SETS2_KEYS, PROTOS = list(SETS2), ["paper", "temporal", "random"]
results = []; oofs = {}
for proto in PROTOS:
    sp = splits(proto); te_all = [te for _, te in sp]; prs_m = pairs_for(te_all, matched=True); w1 = np.ones(len(d))
    log(f"== protocol {proto}")
    for mname in MODELS:
        for fs in SETS2_KEYS:
            if mname == "Stacking" and fs not in ("base", "+all enriched"): continue
            X = X2(fs); pred = np.full(len(d), np.nan)
            for tr, te in sp: pred[te] = fit_pred(mk(mname), X[tr], y_reg[tr], X[te])
            oofs[(proto, mname, fs)] = pred
            m = ~np.isnan(pred); r2 = float(wr2(y_reg[m], pred[m], w1[m])); bs = [wr2(y_reg[m], pred[m], bw[pidx][m]) for bw in BW]
            results.append(dict(task="retention", protocol=proto, model=mname, features=fs, n_features=int(X.shape[1]), r2=r2, r2_ci=ci(bs)))
            log(f"  {mname:12s} {fs:15s} R2={r2:.3f}")
    for mname in MODELS:
        if (proto, mname, "base") not in oofs: continue
        pb = oofs[(proto, mname, "base")]
        for fs in SETS2_KEYS:
            if fs == "base" or (proto, mname, fs) not in oofs: continue
            pe = oofs[(proto, mname, fs)]; m = ~np.isnan(pb) & ~np.isnan(pe)
            dlt = [wr2(y_reg[m], pe[m], bw[pidx][m]) - wr2(y_reg[m], pb[m], bw[pidx][m]) for bw in BW]
            results.append(dict(task="delta_retention", protocol=proto, model=mname, features=fs, delta_r2=float(wr2(y_reg[m], pe[m], w1[m]) - wr2(y_reg[m], pb[m], w1[m])), delta_ci=ci(dlt)))
    for fs in SETS2_KEYS:
        X = X2(fs); pc_ = np.full(len(d), np.nan); ps = np.full(len(d), np.nan)
        for tr, te in sp:
            pc_[te] = fit_pred(make_clf("XGBoost"), X[tr], y_clf[tr], X[te], proba=True)
            ps[te] = fit_xgb_aft(X[tr][:, 1:], L[tr], U[tr], X[te][:, 1:])
        m = ~np.isnan(pc_); auc = float(roc_auc_score(y_clf[m], pc_[m])); bsa = []
        for bw in BW:
            try: bsa.append(roc_auc_score(y_clf[m], pc_[m], sample_weight=bw[pidx][m]))
            except Exception: pass
        cm = cindex(ps, prs_m); bsc = [cindex(ps, prs_m, bw) for bw in BW[:100]]
        results.append(dict(task="clf_surv", protocol=proto, model="XGBoost", features=fs, auc=auc, auc_ci=ci(bsa), cindex_matched=cm, cindex_matched_ci=ci(bsc)))
        log(f"  clf/surv XGBoost {fs:15s} AUC={auc:.3f} C_matched={cm:.3f}")
    json.dump(results, open(RES / ("stability_enriched_features_quick.json" if QUICK else "stability_enriched_features.json"), "w"), indent=1, default=str)
log("DONE")
