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
    if name == "ElasticNet": return ("scaled", ElasticNetCV(l1_ratio=[.2, .6, 1.0], n_alphas=30, cv=3, n_jobs=NJ, max_iter=3000))
    if name == "RandomForest": return ("raw", RandomForestRegressor(300, min_samples_leaf=3, max_features=0.33, n_jobs=NJ, random_state=SEED))
    if name == "XGBoost": return ("raw", xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, tree_method="hist"))
    if name == "LightGBM": return ("raw", lgb.LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, verbose=-1))
    if name == "MLP": return ("scaled", MLPRegressor((128, 64), alpha=1e-2, early_stopping=True, max_iter=300, random_state=SEED))
def make_clf(name):
    if name == "Logistic": return ("scaled", LogisticRegressionCV(Cs=8, cv=3, max_iter=2000, n_jobs=NJ))
    if name == "RandomForest": return ("raw", RandomForestClassifier(300, min_samples_leaf=3, max_features=0.33, n_jobs=NJ, random_state=SEED))
    if name == "XGBoost": return ("raw", xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, tree_method="hist"))
    if name == "LightGBM": return ("raw", lgb.LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, verbose=-1))
    if name == "MLP": return ("scaled", MLPClassifier((128, 64), alpha=1e-2, early_stopping=True, max_iter=300, random_state=SEED))
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

from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier, BaggingRegressor, BaggingClassifier, StackingRegressor, StackingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import BayesianRidge, Ridge, LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostRegressor, CatBoostClassifier
import torch.nn as nn

def _resnet(n_in, width=256, blocks=3, drop=0.1):
    class Block(nn.Module):
        def __init__(s): super().__init__(); s.f = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(drop), nn.Linear(width, width))
        def forward(s, x): return x + s.f(x)
    return nn.Sequential(nn.Linear(n_in, width), *[Block() for _ in range(blocks)], nn.LayerNorm(width), nn.GELU(), nn.Linear(width, 1))
class TorchTab:
    def __init__(self, task="reg", epochs=80, seed=0): self.task, self.epochs, self.seed = task, epochs, seed
    def fit(self, X, y):
        torch.manual_seed(self.seed); np.random.seed(self.seed); n = len(X); idx = np.random.permutation(n); nv = max(int(.1 * n), 20); va, tr = idx[:nv], idx[nv:]
        Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32); self.net = _resnet(X.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-2); lossf = nn.BCEWithLogitsLoss() if self.task == "clf" else nn.MSELoss()
        best, bs, wait = 1e9, None, 0
        for ep in range(self.epochs):
            self.net.train(); perm = tr[np.random.permutation(len(tr))]
            for k in range(0, len(perm), 128):
                b = perm[k:k + 128]; opt.zero_grad(); l = lossf(self.net(Xt[b]).squeeze(1), yt[b]); l.backward(); opt.step()
            self.net.eval()
            with torch.no_grad(): v = lossf(self.net(Xt[va]).squeeze(1), yt[va]).item()
            if v < best - 1e-5: best, wait, bs = v, 0, {k: x.clone() for k, x in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= 12: break
        self.net.load_state_dict(bs); return self
    def predict(self, X):
        self.net.eval()
        with torch.no_grad(): return self.net(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()
    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.predict(X))); return np.c_[1 - p, p]

def make_reg2(name):
    S = np.random.RandomState(SEED)
    if name == "ExtraTrees": return ("raw", ExtraTreesRegressor(400, min_samples_leaf=3, max_features=0.33, n_jobs=NJ, random_state=SEED))
    if name == "HistGradientBoosting": return ("raw", HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, max_iter=300, random_state=SEED))
    if name == "GradientBoosting": return ("raw", GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8, max_features=0.5, random_state=SEED))
    if name == "CatBoost": return ("raw", CatBoostRegressor(iterations=500, depth=6, learning_rate=0.06, verbose=0, thread_count=NJ, random_seed=SEED))
    if name == "AdaBoost": return ("raw", AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=150, learning_rate=0.1, random_state=SEED))
    if name == "Bagging": return ("raw", BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=3), n_estimators=100, max_features=0.5, n_jobs=NJ, random_state=SEED))
    if name == "DecisionTree": return ("raw", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=SEED))
    if name == "kNN": return ("scaled", KNeighborsRegressor(15, weights="distance", n_jobs=NJ))
    if name == "SVR-RBF": return ("scaled", SVR(C=1.0, epsilon=0.05, gamma="scale"))
    if name == "BayesianRidge": return ("scaled", BayesianRidge())
    if name == "KernelRidge-Nystroem": return ("scaled", make_pipeline(Nystroem(n_components=500, gamma=None, random_state=SEED), RidgeCV(alphas=np.logspace(-3, 3, 13))))
    if name == "ResNet-MLP (torch)": return ("scaled", TorchTab("reg", seed=SEED))
    if name == "Stacking (RF+XGB+LGBM+Ridge)": return ("scaled", StackingRegressor([("rf", RandomForestRegressor(150, min_samples_leaf=3, max_features=0.33, n_jobs=NJ, random_state=SEED)), ("xgb", xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.06, subsample=.8, colsample_bytree=.5, n_jobs=NJ, tree_method="hist", random_state=SEED)), ("lgb", lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.06, subsample=.8, subsample_freq=1, colsample_bytree=.5, n_jobs=NJ, verbose=-1, random_state=SEED)), ("ridge", RidgeCV(alphas=np.logspace(-1, 4, 15)))], final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)), cv=3, n_jobs=1))
def make_clf2(name):
    if name == "ExtraTrees": return ("raw", ExtraTreesClassifier(400, min_samples_leaf=3, max_features=0.33, n_jobs=NJ, random_state=SEED))
    if name == "HistGradientBoosting": return ("raw", HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06, max_iter=300, random_state=SEED))
    if name == "GradientBoosting": return ("raw", GradientBoostingClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8, max_features=0.5, random_state=SEED))
    if name == "CatBoost": return ("raw", CatBoostClassifier(iterations=500, depth=6, learning_rate=0.06, verbose=0, thread_count=NJ, random_seed=SEED))
    if name == "AdaBoost": return ("raw", AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=150, learning_rate=0.1, random_state=SEED))
    if name == "Bagging": return ("raw", BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=3), n_estimators=100, max_features=0.5, n_jobs=NJ, random_state=SEED))
    if name == "DecisionTree": return ("raw", DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=SEED))
    if name == "kNN": return ("scaled", KNeighborsClassifier(15, weights="distance", n_jobs=NJ))
    if name == "SVM-RBF": return ("scaled", SVC(C=1.0, gamma="scale", probability=True, random_state=SEED))
    if name == "GaussianNB": return ("scaled", GaussianNB())
    if name == "ResNet-MLP (torch)": return ("scaled", TorchTab("clf", seed=SEED))
    if name == "Stacking (RF+XGB+LGBM+Logistic)": return ("scaled", StackingClassifier([("rf", RandomForestClassifier(150, min_samples_leaf=3, max_features=0.33, n_jobs=NJ, random_state=SEED)), ("xgb", xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.06, subsample=.8, colsample_bytree=.5, n_jobs=NJ, tree_method="hist", random_state=SEED)), ("lgb", lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.06, subsample=.8, subsample_freq=1, colsample_bytree=.5, n_jobs=NJ, verbose=-1, random_state=SEED)), ("lr", LogisticRegression(C=0.1, max_iter=2000))], final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=1))
REG2 = ["ExtraTrees", "HistGradientBoosting", "GradientBoosting", "CatBoost", "AdaBoost", "Bagging", "DecisionTree", "kNN", "SVR-RBF", "BayesianRidge", "KernelRidge-Nystroem", "ResNet-MLP (torch)", "Stacking (RF+XGB+LGBM+Ridge)"]
CLF2 = ["ExtraTrees", "HistGradientBoosting", "GradientBoosting", "CatBoost", "AdaBoost", "Bagging", "DecisionTree", "kNN", "SVM-RBF", "GaussianNB", "ResNet-MLP (torch)", "Stacking (RF+XGB+LGBM+Logistic)"]
def _nll_terms(mu, ls, lL, lU, ex, rc, dist):
    s = torch.exp(ls); zL = (lL - mu) / s; zU = (lU - mu) / s
    if dist == "weibull":
        logS = lambda z: -torch.exp(z); ll_ex = zL - torch.exp(zL) - torch.log(s)
    else:
        N = torch.distributions.Normal(0., 1.); logS = lambda z: torch.log(torch.clamp(1 - N.cdf(z), min=1e-12)); ll_ex = N.log_prob(zL) - torch.log(s)
    ll_rc = logS(zL); ll_ic = torch.log(torch.clamp(torch.exp(logS(zL)) - torch.exp(logS(zU)), min=1e-12))
    return torch.where(ex, ll_ex, torch.where(rc, ll_rc, ll_ic)).sum()
def fit_torch_aft(Xtr, Ltr, Utr, Xte, dist="lognormal", hidden=None, lam=3.0, epochs=120):
    sc = StandardScaler().fit(Xtr); A = torch.tensor(np.clip(sc.transform(Xtr), -8, 8), dtype=torch.float32); B = torch.tensor(np.clip(sc.transform(Xte), -8, 8), dtype=torch.float32)
    lL = torch.tensor(np.log(Ltr), dtype=torch.float32); lU = torch.tensor(np.log(np.where(np.isinf(Utr), 1e9, Utr)), dtype=torch.float32)
    ex = torch.tensor(Ltr == Utr); rc = torch.tensor(np.isinf(Utr)); torch.manual_seed(SEED)
    net = nn.Linear(A.shape[1], 1) if hidden is None else nn.Sequential(nn.Linear(A.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
    with torch.no_grad(): net[-1].bias.fill_(float(np.log(np.median(Ltr)))) if hidden else net.bias.fill_(float(np.log(np.median(Ltr))))
    ls = torch.zeros(1, requires_grad=True); opt = torch.optim.Adam(list(net.parameters()) + [ls], lr=3e-3 if hidden else 2e-2, weight_decay=1e-3 if hidden else 0)
    n = len(Ltr); idx = np.arange(n); np.random.seed(SEED)
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for k in range(0, n, 256):
            b = idx[k:k + 256]; opt.zero_grad(); mu = net(A[b]).squeeze(1)
            loss = -_nll_terms(mu, ls, lL[b], lU[b], ex[b], rc[b], dist) / len(b) + lam * sum((p ** 2).sum() for p in net.parameters() if p.dim() > 1) / n
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(B).squeeze(1).numpy()
def fit_xgb_aft_dist(Xtr, Ltr, Utr, Xte, dist):
    dtr = xgb.DMatrix(Xtr); dtr.set_float_info("label_lower_bound", Ltr); dtr.set_float_info("label_upper_bound", Utr)
    p = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": dist, "aft_loss_distribution_scale": 1.0, "tree_method": "hist", "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.5, "nthread": NJ, "seed": SEED}
    return np.log(np.clip(xgb.train(p, dtr, num_boost_round=300).predict(xgb.DMatrix(Xte)), 1e-3, 1e8))
def fit_cox_ridge(Xtr, Ltr, Utr, Xte, lam=3.0):
    sc = StandardScaler().fit(Xtr); A = torch.tensor(np.clip(sc.transform(Xtr), -8, 8), dtype=torch.float64); B = torch.tensor(np.clip(sc.transform(Xte), -8, 8), dtype=torch.float64)
    t = np.where(np.isinf(Utr), Ltr, Utr); ev = ~np.isinf(Utr); o = np.argsort(-t); A = A[o]; tt = t[o]; e = torch.tensor(ev[o])
    w = torch.zeros(A.shape[1], dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([w], lr=0.5, max_iter=150, line_search_fn="strong_wolfe")
    def nll():
        opt.zero_grad(); eta = A @ w; lse = torch.logcumsumexp(eta, 0); loss = -((eta - lse)[e]).sum() / e.sum() + lam * (w ** 2).sum() / len(tt); loss.backward(); return loss
    opt.step(nll); return -(B @ w).detach().numpy()
SURV2 = ["LogNormal-AFT (ridge)", "XGBoost-AFT (normal)", "XGBoost-AFT (logistic)", "Neural Weibull-AFT (MLP)", "Neural LogNormal-AFT (MLP)", "Cox-PH (ridge, interval~event at upper bound)"]
def fit_surv2(m, Xtr, Ltr, Utr, Xte):
    if m == "LogNormal-AFT (ridge)": return fit_torch_aft(Xtr, Ltr, Utr, Xte, "lognormal")
    if m == "XGBoost-AFT (normal)": return fit_xgb_aft_dist(Xtr, Ltr, Utr, Xte, "normal")
    if m == "XGBoost-AFT (logistic)": return fit_xgb_aft_dist(Xtr, Ltr, Utr, Xte, "logistic")
    if m == "Neural Weibull-AFT (MLP)": return fit_torch_aft(Xtr, Ltr, Utr, Xte, "weibull", hidden=128)
    if m == "Neural LogNormal-AFT (MLP)": return fit_torch_aft(Xtr, Ltr, Utr, Xte, "lognormal", hidden=128)
    if m.startswith("Cox"): return fit_cox_ridge(Xtr, Ltr, Utr, Xte)
def fit_pred2(maker, Xtr, ytr, Xte, proba=False):
    kind, m = maker
    if kind == "scaled":
        sc = StandardScaler().fit(Xtr); Xtr, Xte = np.clip(sc.transform(Xtr), -8, 8), np.clip(sc.transform(Xte), -8, 8)
    m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1] if proba else m.predict(Xte)
results = []
if QUICK: REG2, CLF2, SURV2, PROTOS = REG2[:3] + ["ResNet-MLP (torch)", "Stacking (RF+XGB+LGBM+Ridge)"], CLF2[:2] + ["ResNet-MLP (torch)"], SURV2[:2] + ["Neural Weibull-AFT (MLP)", SURV2[-1]], ["paper"]
Xall = X_of("all")
for proto in PROTOS:
    sp = splits(proto); te_all = [te for _, te in sp]; prs = pairs_for(te_all); prs_m = pairs_for(te_all, matched=True)
    log(f"== protocol {proto}: {len(sp)} folds")
    for mname in REG2:
        pred = np.full(len(d), np.nan)
        for tr, te in sp: pred[te] = fit_pred2(make_reg2(mname), Xall[tr], y_reg[tr], Xall[te])
        m = ~np.isnan(pred); w1 = np.ones(len(d))
        res = dict(task="retention", protocol=proto, model=mname, features="all", n_test=int(m.sum()), rmse=wrmse(y_reg[m], pred[m], w1[m]), r2=float(wr2(y_reg[m], pred[m], w1[m])))
        bs = [(wrmse(y_reg[m], pred[m], bw[pidx][m]), wr2(y_reg[m], pred[m], bw[pidx][m])) for bw in BW]; res["rmse_ci"] = ci([b[0] for b in bs]); res["r2_ci"] = ci([b[1] for b in bs]); results.append(res)
        log(f"  reg {mname:34s} R2={res['r2']:.3f}")
    for mname in CLF2:
        pred = np.full(len(d), np.nan)
        for tr, te in sp: pred[te] = fit_pred2(make_clf2(mname), Xall[tr], y_clf[tr], Xall[te], proba=True)
        m = ~np.isnan(pred)
        try: auc = float(roc_auc_score(y_clf[m], pred[m]))
        except Exception: auc = np.nan
        bs = []
        for bw in BW:
            try: bs.append(roc_auc_score(y_clf[m], pred[m], sample_weight=bw[pidx][m]))
            except Exception: pass
        results.append(dict(task="retained80", protocol=proto, model=mname, features="all", n_test=int(m.sum()), auc=auc, auc_ci=ci(bs))); log(f"  clf {mname:34s} AUC={auc:.3f}")
    Xs = Xall[:, 1:]
    for mname in SURV2:
        pred = np.full(len(d), np.nan)
        for tr, te in sp: pred[te] = fit_surv2(mname, Xs[tr], L[tr], U[tr], Xs[te])
        c = cindex(pred, prs); bs = [cindex(pred, prs, bw) for bw in BW[:100]]; cm = cindex(pred, prs_m); bm = [cindex(pred, prs_m, bw) for bw in BW[:100]]
        results.append(dict(task="time_to_80", protocol=proto, model=mname, features="all", n_test=int(np.sum(~np.isnan(pred))), cindex=c, cindex_ci=ci(bs), cindex_matched=cm, cindex_matched_ci=ci(bm))); log(f"  surv {mname:46s} C={c:.3f} C_matched={cm:.3f}")
    json.dump(results, open(RES / ("stability_benchmark_extra_models_quick.json" if QUICK else "stability_benchmark_extra_models.json"), "w"), indent=1, default=str)
log("DONE")
