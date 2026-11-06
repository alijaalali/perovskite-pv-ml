import json, sys
mode, arg = sys.argv[1], sys.argv[2]
src_a = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_author_heldout.py").read()
sys.argv = [sys.argv[0]]
exec(compile(src_a[:src_a.index("SETX = {")], "author_head", "exec"))
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
BASE = ["time", "comp", "device", "protocol"]; NOCOMP = ["time", "device", "protocol"]
def Xp(parts): return pd.concat([F[k] for k in parts], axis=1).values.astype(float)
if mode in ("seed", "size"): K = int(arg); SEED = SEED + K
rs = np.random.default_rng(3 + (int(arg) if mode in ("seed", "size") else 0))
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
if mode == "author": sp = split_author(arg); SETS = {"base": BASE, "+process+stack": BASE + ["proc", "stack"]}; proto = arg; MODELS = ["XGBoost", "LightGBM", "RandomForest"]
elif mode == "seed":
    r_ = np.random.default_rng(100 + K); perm = r_.permutation(len(papers_u)); fold_of = np.empty(len(papers_u), int); fold_of[perm] = np.arange(len(papers_u)) % 5
    sp = [(np.where(fold_of[pidx] != f)[0], np.where(fold_of[pidx] == f)[0]) for f in range(5)]; SETS = {"base": BASE, "+process+stack": BASE + ["proc", "stack"]}; proto = f"paper_seed{K}"; MODELS = ["XGBoost", "LightGBM", "RandomForest"]
elif mode == "size":
    sizes = [len(tr_) for tr_, _ in split_author("purged_any_author")]; r_ = np.random.default_rng(500 + K); sp = []
    for (tr_, te_), n_k in zip(splits("paper"), sizes):
        pp = r_.permutation(np.unique(pidx[tr_])); cnt_ = np.bincount(pidx[tr_], minlength=len(papers_u)); keep = []; tot = 0
        for q in pp:
            if tot >= n_k: break
            keep.append(q); tot += cnt_[q]
        sp.append((tr_[np.isin(pidx[tr_], keep)], te_))
    SETS = {"base": BASE, "+process+stack": BASE + ["proc", "stack"]}; proto = f"paper_size{K}"; MODELS = ["XGBoost", "LightGBM", "RandomForest"]
else: sp = splits(arg); SETS = {"base": BASE, "base without composition": NOCOMP}; proto = arg; MODELS = ["XGBoost", "LightGBM", "RandomForest"]
if QUICK: MODELS = MODELS[:1]
NC = 3 if QUICK else 8; w1 = np.ones(len(d)); cnt = pd.Series(paper).map(pd.Series(paper).value_counts()).values; wpap = 1.0 / cnt
res = []; OOF = {}
def summ(mname, fs, pred):
    m = ~np.isnan(pred); return dict(protocol=proto, model=mname, features=fs, r2=float(wr2(y_reg[m], pred[m], w1[m])), r2_ci=ci([wr2(y_reg[m], pred[m], bw[pidx][m]) for bw in BW]), r2_paperweighted=float(wr2(y_reg[m], pred[m], wpap[m])))
for mname in MODELS:
    grid = [sample(mname) for _ in range(NC)]
    for fs, parts in SETS.items():
        X = Xp(parts); pred = np.full(len(d), np.nan)
        for tr, te in sp:
            inner = list(GroupKFold(3).split(tr, groups=paper[tr])); sc = []
            for par in grid:
                p_ = np.full(len(tr), np.nan)
                for a, b in inner: m_ = build(mname, par); m_.fit(X[tr][a], y_reg[tr][a]); p_[b] = m_.predict(X[tr][b])
                sc.append(-np.mean((y_reg[tr] - p_) ** 2))
            m_ = build(mname, grid[int(np.argmax(sc))]); m_.fit(X[tr], y_reg[tr]); pred[te] = m_.predict(X[te])
        OOF[(mname, fs)] = pred; res.append(summ(mname, fs, pred)); json.dump(res, open(RES / f"stability_tuned_robustness_{mode}_{arg}_partial.json", "w")); log(f"{proto} {mname:12s} {fs:28s} tuned R2={res[-1]['r2']:.3f}")
comp = [m for m in ["XGBoost", "LightGBM", "RandomForest"] if m in MODELS]
if len(comp) == 3:
    for fs in SETS: OOF[("Blend (XGB+LGBM+RF)", fs)] = np.mean([OOF[(m, fs)] for m in comp], axis=0); res.append(summ("Blend (XGB+LGBM+RF)", fs, OOF[("Blend (XGB+LGBM+RF)", fs)]))
fs0 = list(SETS)[0]
for mname in MODELS + (["Blend (XGB+LGBM+RF)"] if len(comp) == 3 else []):
    pb = OOF[(mname, fs0)]; m = ~np.isnan(pb)
    for fs in list(SETS)[1:]:
        pe = OOF[(mname, fs)]; g = wr2(y_reg[m], pe[m], w1[m]) - wr2(y_reg[m], pb[m], w1[m]); gb = [wr2(y_reg[m], pe[m], bw[pidx][m]) - wr2(y_reg[m], pb[m], bw[pidx][m]) for bw in BW]
        res.append(dict(protocol=proto, model=mname, features=fs, task="gain", reference=fs0, gain=float(g), gain_ci=ci(gb)))
json.dump(res, open(RES / f"stability_tuned_robustness_{mode}_{arg}{'_quick' if QUICK else ''}.json", "w"), indent=1); log("DONE")
