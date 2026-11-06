import json, sys, time, warnings, itertools
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
SETS2 = {"design only (no fresh PCE)": BASE + ["proc", "stack"], "base": BASE, "+process": BASE + ["proc"], "+stack tokens": BASE + ["stack"], "+fresh PCE": BASE + ["pcefresh"], "+all enriched": BASE + ["proc", "stack", "pcefresh"]}
def X2(name): return pd.concat([F[k] for k in SETS2[name]], axis=1).values.astype(float)
def mk(name):
    if name == "CatBoost": return ("raw", CatBoostRegressor(iterations=500, depth=6, learning_rate=0.06, verbose=0, thread_count=NJ, random_seed=SEED))
    if name == "Stacking": return ("scaled", StackingRegressor([("rf", RandomForestRegressor(150, min_samples_leaf=3, max_features=0.33, n_jobs=NJ, random_state=SEED)), ("xgb", xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.06, subsample=.8, colsample_bytree=.5, n_jobs=NJ, tree_method="hist", random_state=SEED)), ("lgb", lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.06, subsample=.8, subsample_freq=1, colsample_bytree=.5, n_jobs=NJ, verbose=-1, random_state=SEED)), ("ridge", RidgeCV(alphas=np.logspace(-1, 4, 15)))], final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)), cv=3, n_jobs=1))
    return make_reg(name)

from sklearn.linear_model import LogisticRegressionCV
import torch.nn as nn
NOFRESH = "nofresh" in sys.argv
PW_FS = "design only (no fresh PCE)" if NOFRESH else "+all enriched"
Xcell = X2(PW_FS); ycell = d.retention_end.clip(0, 1.2).values; pcode, pnames = pd.factorize(d.paper_id.values)
logt = np.log10(d.t_exp_h.values); isos_ = d.isos.values; yr = d.year.values
paper_year = pd.Series(yr).groupby(pcode).min().values
def simp(s, keep): s = s.astype(str).str.strip(); return s.where(s.isin(keep), "other")
def topk(s, k): return list(s.value_counts().head(k).index)
tokA = raw.Perovskite_additives_compounds.fillna("Unknown").astype(str).str.split(";").apply(lambda l: [t.strip() for t in l])
fac = pd.DataFrame({"arch": simp(d.Cell_architecture, ["nip", "pin"]), "etl": simp(d.ETL_stack_sequence, topk(d.ETL_stack_sequence, 6)), "htl": simp(d.HTL_stack_sequence, topk(d.HTL_stack_sequence, 6)),
                    "back": simp(d.Backcontact_stack_sequence, topk(d.Backcontact_stack_sequence, 4)), "enc": simp(d.Encapsulation, ["True", "False"]), "asite": simp(d.a_site, topk(d.a_site, 5)), "bsite": d.b_site_class.astype(str),
                    "dep": simp(d.Perovskite_deposition_procedure, topk(d.Perovskite_deposition_procedure, 3)),
                    "quench": simp(raw.Perovskite_deposition_quenching_media.fillna("Unknown"), topk(raw.Perovskite_deposition_quenching_media.fillna("Unknown"), 4)),
                    "add": tokA.apply(lambda l: "unknown" if "Unknown" in l else ("none" if all(t in ("Undoped", "") for t in l) else "additive"))}).values
rng0 = np.random.default_rng(0); PA, PB, PL, PD, PP, PN = [], [], [], [], [], []
for pc_, idx in pd.Series(np.arange(len(d))).groupby(pcode):
    idx = idx.values
    if len(idx) < 2: continue
    for a, b in itertools.combinations(idx, 2):
        if isos_[a] != isos_[b] or abs(logt[a] - logt[b]) >= 0.1: continue
        dy = ycell[a] - ycell[b]
        if abs(dy) < 0.01: continue
        if rng0.random() < 0.5: a, b, dy = b, a, -dy
        PA.append(a); PB.append(b); PL.append(int(dy > 0)); PD.append(abs(dy)); PP.append(pc_); PN.append(int((fac[a] != fac[b]).sum()))
PA, PB, PL, PD, PP, PN = map(np.array, (PA, PB, PL, PD, PP, PN))
log(f"pairs: {len(PA)} from {len(np.unique(PP))} papers; label balance {PL.mean():.2f}; |gap|>=0.10: {int((PD >= 0.1).sum())}; differing coded factors 0/1/2+: {[int((PN == k).sum()) if k < 2 else int((PN >= 2).sum()) for k in (0, 1, 2)]}")
def scaled(Xtr, Xte):
    sc = StandardScaler().fit(Xtr); return np.clip(sc.transform(Xtr), -8, 8), np.clip(sc.transform(Xte), -8, 8)
class RankNet:
    def __init__(self, seeds=3, epochs=40): self.seeds, self.epochs = seeds, epochs
    def fit(self, Xc, ia, ib, lab):
        self.nets = []
        for sd in range(self.seeds):
            torch.manual_seed(sd); net = nn.Sequential(nn.Linear(Xc.shape[1], 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
            Xt = torch.tensor(Xc, dtype=torch.float32); la = torch.tensor(lab, dtype=torch.float32); rg = np.random.RandomState(sd)
            for ep in range(self.epochs):
                net.train(); perm = rg.permutation(len(ia))
                for k in range(0, len(perm), 256):
                    b = perm[k:k + 256]; opt.zero_grad(); s = net(Xt[ia[b]]).squeeze(1) - net(Xt[ib[b]]).squeeze(1); nn.functional.binary_cross_entropy_with_logits(s, la[b]).backward(); opt.step()
            net.eval(); self.nets.append(net)
        return self
    def score(self, Xc):
        with torch.no_grad(): return np.mean([n(torch.tensor(Xc, dtype=torch.float32)).squeeze(1).numpy() for n in self.nets], axis=0)
POOLED = ["Ridge", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
DEMEAN = ["Ridge", "XGBoost", "LightGBM"]
if QUICK: POOLED, DEMEAN = ["Ridge", "XGBoost"], ["XGBoost"]
def run_split(tr_cell, te_pairs):
    """return {model: score_diff over te_pairs} (positive = predicts the first cell of the pair lasts longer)"""
    out = {}; tr_idx = np.where(tr_cell)[0]; tr_pair = np.where(tr_cell[PA] & tr_cell[PB])[0]
    cells_te = np.unique(np.r_[PA[te_pairs], PB[te_pairs]])
    for mname in POOLED:
        p = np.full(len(d), np.nan); p[cells_te] = fit_pred(mk(mname), Xcell[tr_idx], ycell[tr_idx], Xcell[cells_te]); out["pooled " + mname] = p[PA[te_pairs]] - p[PB[te_pairs]]
    cnt = pd.Series(pcode[tr_idx]).map(pd.Series(pcode[tr_idx]).value_counts()).values; mt = tr_idx[cnt >= 2]
    ydm = ycell[mt] - pd.Series(ycell[mt]).groupby(pcode[mt]).transform("mean").values
    for mname in DEMEAN:
        p = np.full(len(d), np.nan); p[cells_te] = fit_pred(mk(mname), Xcell[mt], ydm, Xcell[cells_te]); out["demeaned " + mname] = p[PA[te_pairs]] - p[PB[te_pairs]]
    Xs_tr, Xs_all = scaled(Xcell[tr_idx], Xcell); ia, ib, lab = PA[tr_pair], PB[tr_pair], PL[tr_pair]
    D = lambda M, a, b: M[a] - M[b]
    Dtr = np.r_[D(Xs_all, ia, ib), D(Xs_all, ib, ia)]; ltr = np.r_[lab, 1 - lab]; Dte = D(Xs_all, PA[te_pairs], PB[te_pairs])
    bt = LogisticRegressionCV(Cs=8, cv=3, fit_intercept=False, max_iter=3000).fit(Dtr, ltr); out["pairwise logistic (Bradley-Terry)"] = bt.decision_function(Dte)
    kw = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED)
    for nm, cls in [("XGBoost", xgb.XGBClassifier(tree_method="hist", **kw)), ("LightGBM", lgb.LGBMClassifier(verbose=-1, **{k: v for k, v in kw.items() if k != "random_state"}, random_state=SEED))]:
        cls.fit(Dtr, ltr); pr = cls.predict_proba(Dte)[:, 1]; cls2 = cls.predict_proba(-Dte)[:, 1]; out["pairwise " + nm + " (differences)"] = np.log(np.clip(pr, 1e-6, 1)) - np.log(np.clip(cls2, 1e-6, 1))
    Ctr = np.r_[np.c_[D(Xs_all, ia, ib), (Xs_all[ia] + Xs_all[ib]) / 2], np.c_[D(Xs_all, ib, ia), (Xs_all[ia] + Xs_all[ib]) / 2]]
    a_, b_ = PA[te_pairs], PB[te_pairs]; Cte = np.c_[D(Xs_all, a_, b_), (Xs_all[a_] + Xs_all[b_]) / 2]; Cte2 = np.c_[D(Xs_all, b_, a_), (Xs_all[a_] + Xs_all[b_]) / 2]
    cx = xgb.XGBClassifier(tree_method="hist", **kw).fit(Ctr, ltr); out["pairwise XGBoost (differences + context)"] = np.log(np.clip(cx.predict_proba(Cte)[:, 1], 1e-6, 1)) - np.log(np.clip(cx.predict_proba(Cte2)[:, 1], 1e-6, 1))
    if not QUICK:
        rn = RankNet().fit(Xs_all, ia, ib, lab); s = rn.score(Xs_all); out["RankNet (neural, pairwise loss)"] = s[PA[te_pairs]] - s[PB[te_pairs]]
        o = np.argsort(pcode[mt], kind="stable"); rk = xgb.XGBRanker(objective="rank:pairwise", tree_method="hist", **kw).fit(Xcell[mt][o], np.round(ycell[mt][o] * 10).astype(int), qid=pcode[mt][o]); p = np.full(len(d), np.nan); p[cells_te] = rk.predict(Xcell[cells_te]); out["XGBoost ranker (query = paper)"] = p[PA[te_pairs]] - p[PB[te_pairs]]
    return out
splits_pw = {}
for seed in ([0] if QUICK else [0, 1, 2]):
    r_ = np.random.default_rng(100 + seed); fold_of_paper = r_.permutation(len(pnames)) % 5
    splits_pw[f"paper-held-out (seed {seed})"] = [(fold_of_paper[pcode] != k, np.where(fold_of_paper[PP] == k)[0]) for k in range(5)]
splits_pw["temporal (train <=2018, test 2019+)"] = [(paper_year[pcode] <= 2018, np.where(paper_year[PP] >= 2019)[0])]
res = {}
for name, folds in splits_pw.items():
    res[name] = {}
    for tr_cell, te_pairs in folds:
        if len(te_pairs) == 0: continue
        sc = run_split(tr_cell, te_pairs)
        for mname, s in sc.items():
            res[name].setdefault(mname, np.full(len(PA), np.nan))[te_pairs] = s
    log(f"{name}: done ({len(folds)} folds)")
rows = []
subsets = {"all pairs": np.ones(len(PA), bool), "gap >= 0.10": PD >= 0.1, "one coded factor differs": PN == 1, "two or more factors differ": PN >= 2, "no coded factor differs": PN == 0}
Wb = np.random.default_rng(7).multinomial(len(pnames), np.ones(len(pnames)) / len(pnames), size=(100 if QUICK else 500))
def acc_of(s, m, w=None):
    ok = np.where(s[m] == 0, 0.5, (np.sign(s[m]) == np.where(PL[m] == 1, 1, -1)).astype(float)); ww = np.ones(m.sum()) if w is None else w[PP[m]]
    return float((ww * ok).sum() / ww.sum()) if ww.sum() > 0 else np.nan
for proto in ["paper-held-out", "temporal (train <=2018, test 2019+)"]:
    keys = [k for k in res if k.startswith(proto)]
    S_ref = np.mean([res[k]["pooled Ridge"] for k in keys], axis=0) if len(keys) > 1 else res[keys[0]]["pooled Ridge"]
    for mname in res[keys[0]]:
        S = np.mean([res[k][mname] for k in keys], axis=0) if len(keys) > 1 else res[keys[0]][mname]
        for sname, sm in subsets.items():
            m = sm & ~np.isnan(S)
            if m.sum() < 30: continue
            accs = [acc_of(np.nan_to_num(res[k][mname]), m) for k in keys]; boots = [acc_of(S, m, w) for w in Wb]
            rows.append(dict(protocol=proto, model=mname, subset=sname, n_pairs=int(m.sum()), n_papers=int(len(np.unique(PP[m]))), accuracy=float(acc_of(S, m)), acc_seed_mean=float(np.mean(accs)), acc_ci=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                             acc_diff_vs_pooled_ridge=float(acc_of(S, m) - acc_of(np.nan_to_num(S_ref), m)), acc_diff_ci=[float(np.nanpercentile([acc_of(S, m, w) - acc_of(np.nan_to_num(S_ref), m, w) for w in Wb], 2.5)), float(np.nanpercentile([acc_of(S, m, w) - acc_of(np.nan_to_num(S_ref), m, w) for w in Wb], 97.5))],
                             acc_paper_mean=float(pd.Series(np.where(S[m] == 0, 0.5, (np.sign(S[m]) == np.where(PL[m] == 1, 1, -1)).astype(float))).groupby(PP[m]).mean().mean())))
out = pd.DataFrame(rows); out.to_csv(DATA / ("pairwise_learning_quick.csv" if QUICK else ("pairwise_learning_nofresh.csv" if NOFRESH else "pairwise_learning.csv")), index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 50)
log("\n" + out[out["subset"].isin(["all pairs", "gap >= 0.10"])].assign(acc=lambda x: x.accuracy.round(3), lo=lambda x: x.acc_ci.str[0].round(3), hi=lambda x: x.acc_ci.str[1].round(3))[["protocol", "model", "subset", "n_pairs", "acc", "lo", "hi"]].to_string(index=False))
log("DONE")
