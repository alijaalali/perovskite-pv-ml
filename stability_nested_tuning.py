import json, sys
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_enriched_features.py").read()
exec(compile(src[:src.index("MODELS = [")], "enriched_head", "exec"))
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
BASE = ["time", "comp", "device", "protocol"]; SETN = {"base": BASE, "+process+stack": BASE + ["proc", "stack"]}
def Xn(n): return pd.concat([F[k] for k in SETN[n]], axis=1).values.astype(float)
rs = np.random.default_rng(3)
def sample(name):
    if name == "XGBoost": return dict(n_estimators=int(rs.choice([150, 300, 500])), max_depth=int(rs.choice([3, 4, 6, 8])), learning_rate=float(rs.choice([0.02, 0.05, 0.1])), subsample=float(rs.choice([0.6, 0.8, 1.0])), colsample_bytree=float(rs.choice([0.3, 0.5, 0.8])), min_child_weight=int(rs.choice([1, 5, 20])), reg_lambda=float(rs.choice([1, 10, 50])))
    if name == "LightGBM": return dict(n_estimators=int(rs.choice([150, 300, 500])), num_leaves=int(rs.choice([7, 15, 31, 63])), learning_rate=float(rs.choice([0.02, 0.05, 0.1])), subsample=float(rs.choice([0.6, 0.8, 1.0])), colsample_bytree=float(rs.choice([0.3, 0.5, 0.8])), min_child_samples=int(rs.choice([5, 20, 50])), reg_lambda=float(rs.choice([0, 10, 50])))
    if name == "CatBoost": return dict(iterations=int(rs.choice([300, 500, 800])), depth=int(rs.choice([4, 6, 8])), learning_rate=float(rs.choice([0.03, 0.06, 0.1])), l2_leaf_reg=float(rs.choice([1, 3, 10, 30])))
    if name == "RandomForest": return dict(n_estimators=300, min_samples_leaf=int(rs.choice([1, 3, 5, 10, 20])), max_features=float(rs.choice([0.15, 0.33, 0.5, 0.8])), max_depth=rs.choice([None, 10, 20]))
def build(name, par):
    if name == "XGBoost": return xgb.XGBRegressor(**par, n_jobs=NJ, random_state=SEED, tree_method="hist")
    if name == "LightGBM": return lgb.LGBMRegressor(**par, subsample_freq=1, n_jobs=NJ, random_state=SEED, verbose=-1)
    if name == "CatBoost": return CatBoostRegressor(**par, verbose=0, thread_count=NJ, random_seed=SEED)
    if name == "RandomForest": return RandomForestRegressor(**par, n_jobs=NJ, random_state=SEED)
NC = 3 if QUICK else 8; MODELS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]; PROT = ["paper", "temporal"]
if QUICK: MODELS, PROT = ["XGBoost"], ["paper"]
res = []; w1 = np.ones(len(d)); chosen = []
for proto in PROT:
    sp = splits(proto)
    for mname in MODELS:
        grid = [sample(mname) for _ in range(NC)]
        for fs in SETN:
            X = Xn(fs); pred = np.full(len(d), np.nan)
            for fold, (tr, te) in enumerate(sp):
                inner = list(GroupKFold(3).split(tr, groups=paper[tr])); sc = []
                for par in grid:
                    p_ = np.full(len(tr), np.nan)
                    for a, b in inner: m_ = build(mname, par); m_.fit(X[tr][a], y_reg[tr][a]); p_[b] = m_.predict(X[tr][b])
                    sc.append(-np.mean((y_reg[tr] - p_) ** 2))
                best = grid[int(np.argmax(sc))]; chosen.append(dict(protocol=proto, model=mname, features=fs, fold=fold, params={k: (None if v is None else (v.item() if hasattr(v, "item") else v)) for k, v in best.items()}))
                m_ = build(mname, best); m_.fit(X[tr], y_reg[tr]); pred[te] = m_.predict(X[te])
            m = ~np.isnan(pred); res.append(dict(protocol=proto, model=mname, features=fs, tuned=True, r2=float(wr2(y_reg[m], pred[m], w1[m])), r2_ci=ci([wr2(y_reg[m], pred[m], bw[pidx][m]) for bw in BW])))
            log(f"{proto:8s} {mname:12s} {fs:15s} tuned R2={res[-1]['r2']:.3f}")
            json.dump(dict(results=res, chosen=chosen), open(RES / ("stability_nested_tuning" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1)
log("DONE")
