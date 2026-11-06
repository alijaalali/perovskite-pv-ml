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
results = []; oof_store = {}
REG = ["Ridge", "ElasticNet", "RandomForest", "XGBoost", "LightGBM", "MLP"]; CLF = ["Logistic", "RandomForest", "XGBoost", "LightGBM", "MLP"]
if QUICK: PROTOS = ["paper", "temporal"]
Xall = X_of("all")
for proto in PROTOS:
    sp = splits(proto); te_all = [te for _, te in sp]; prs = pairs_for(te_all); prs_m = pairs_for(te_all, matched=True)
    log(f"== protocol {proto}: {len(sp)} folds, test sizes {[len(t) for t in te_all]}")
    for mname in ["null"] + REG:
        pred = np.full(len(d), np.nan)
        for tr, te in sp:
            if mname == "null": pred[te] = y_reg[tr].mean()
            else: pred[te] = fit_pred(make_reg(mname), Xall[tr], y_reg[tr], Xall[te])
        m = ~np.isnan(pred); w1 = np.ones(len(d))
        res = dict(task="retention", protocol=proto, model=mname, features="all", n_test=int(m.sum()), rmse=wrmse(y_reg[m], pred[m], w1[m]), r2=float(wr2(y_reg[m], pred[m], w1[m])))
        bs = [(wrmse(y_reg[m], pred[m], bw[pidx][m]), wr2(y_reg[m], pred[m], bw[pidx][m])) for bw in BW]; res["rmse_ci"] = ci([b[0] for b in bs]); res["r2_ci"] = ci([b[1] for b in bs]); results.append(res); oof_store[("reg", proto, mname)] = pred
        log(f"  reg {mname:12s} R2={res['r2']:.3f} RMSE={res['rmse']:.3f}")
    for mname in ["null"] + CLF:
        pred = np.full(len(d), np.nan)
        for tr, te in sp:
            if mname == "null": pred[te] = y_clf[tr].mean()
            else: pred[te] = fit_pred(make_clf(mname), Xall[tr], y_clf[tr], Xall[te], proba=True)
        m = ~np.isnan(pred)
        try: auc = float(roc_auc_score(y_clf[m], pred[m])) if mname != "null" else 0.5
        except Exception: auc = np.nan
        bs = []
        for bw in BW:
            try: bs.append(roc_auc_score(y_clf[m], pred[m], sample_weight=bw[pidx][m]) if mname != "null" else 0.5)
            except Exception: pass
        results.append(dict(task="retained80", protocol=proto, model=mname, features="all", n_test=int(m.sum()), auc=auc, auc_ci=ci(bs))); oof_store[("clf", proto, mname)] = pred
        log(f"  clf {mname:12s} AUC={auc:.3f}")
    Xs = Xall[:, 1:]
    for mname in ["null", "exposure-time-only (reference)", "Weibull-AFT (ridge)", "XGBoost-AFT"]:
        pred = np.full(len(d), np.nan)
        for tr, te in sp:
            if mname == "null": pred[te] = 0.0
            elif mname.startswith("exposure"): pred[te] = np.log(d.t_exp_h.values[te])
            elif mname.startswith("Weibull"): pred[te] = fit_weibull_aft(Xs[tr], L[tr], U[tr], Xs[te])
            else: pred[te] = fit_xgb_aft(Xs[tr], L[tr], U[tr], Xs[te])
        c = cindex(pred, prs) if mname != "null" else 0.5; bs = [cindex(pred, prs, bw) for bw in BW[:100]] if mname != "null" else [0.5]
        cm = cindex(pred, prs_m) if mname != "null" else 0.5; bm = [cindex(pred, prs_m, bw) for bw in BW[:100]] if mname != "null" else [0.5]
        results.append(dict(task="time_to_80", protocol=proto, model=mname, features="all", n_test=int(np.sum(~np.isnan(pred))), cindex=c, cindex_ci=ci(bs), cindex_matched=cm, cindex_matched_ci=ci(bm), n_pairs_matched=int(sum(len(a[0]) for a in prs_m)))); oof_store[("surv", proto, mname)] = pred
        log(f"  surv {mname:30s} C={c:.3f}  C_matched={cm:.3f}")
    json.dump(results, open(RES / ("stability_benchmark_quick.json" if QUICK else "stability_benchmark.json"), "w"), indent=1, default=str)
if not QUICK:
    for proto in ["paper", "temporal"]:
        sp = splits(proto); te_all = [te for _, te in sp]; prs = pairs_for(te_all); prs_m = pairs_for(te_all, matched=True)
        for fs in sets:
            X = X_of(fs); pr = np.full(len(d), np.nan); ps = np.full(len(d), np.nan)
            for tr, te in sp:
                pr[te] = fit_pred(make_reg("XGBoost"), X[tr], y_reg[tr], X[te])
                ps[te] = fit_xgb_aft(X[tr][:, 1:], L[tr], U[tr], X[te][:, 1:]) if X.shape[1] > 1 else np.log(d.t_exp_h.values[te])
            m = ~np.isnan(pr); w1 = np.ones(len(d))
            bs = [wr2(y_reg[m], pr[m], bw[pidx][m]) for bw in BW]; bc = [cindex(ps, prs, bw) for bw in BW[:100]]; bcm = [cindex(ps, prs_m, bw) for bw in BW[:100]]
            results.append(dict(task="ablation", protocol=proto, model="XGBoost", features=fs, r2=float(wr2(y_reg[m], pr[m], w1[m])), r2_ci=ci(bs), cindex=cindex(ps, prs), cindex_ci=ci(bc), cindex_matched=cindex(ps, prs_m), cindex_matched_ci=ci(bcm)))
            log(f"  ablation {proto} {fs}: R2={results[-1]['r2']:.3f} C={results[-1]['cindex']:.3f}")
        json.dump(results, open(RES / "stability_benchmark.json", "w"), indent=1, default=str)
    pd.to_pickle(oof_store, DATA / "stability_benchmark_oof_predictions.pkl")
log("DONE")
