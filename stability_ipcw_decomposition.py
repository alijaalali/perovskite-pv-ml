import json, sys
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_ipcw_horizons.py").read()
exec(compile(src[:src.index("HOR = [100, 500, 1000]")], "ipcw_head", "exec"))
from sklearn.linear_model import LogisticRegression
Xcens = Xh("+all enriched (no time)")
def cov_weights(known):
    p = np.full(len(d), np.nan)
    for tr, te in GroupKFold(5).split(Xcens, groups=paper):
        sc = StandardScaler().fit(Xcens[tr]); A, B = np.clip(sc.transform(Xcens[tr]), -8, 8), np.clip(sc.transform(Xcens[te]), -8, 8)
        m_ = LogisticRegression(C=0.05, max_iter=2000); m_.fit(A, known[tr]); p[te] = m_.predict_proba(B)[:, 1]
    return 1.0 / np.clip(p, 0.05, 1.0)
HOR = [500, 1000]; PROT = ["paper", "temporal"]; FS = ["base (no time)", "+all enriched (no time)"]
if QUICK: HOR, PROT, FS = [500], ["paper"], ["base (no time)"]
res = []
def fit_pred(X, y, wt, trmask, temask, sp):
    pred = np.full(len(d), np.nan)
    for tr, te in sp:
        trk = tr[trmask[tr]]; tek = te[temask[te]]
        if len(trk) < 50 or len(np.unique(y[trk])) < 2: continue
        mdl = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, tree_method="hist"); mdl.fit(X[trk], y[trk], sample_weight=wt[trk]); pred[tek] = mdl.predict_proba(X[tek])[:, 1]
    return pred
def boot_auc(y, p, w, m):
    out = []
    for bw in BW:
        b = bw[pidx][m] * w[m]
        if b.sum() > 0: out.append(wauc(y[m], p[m], b))
    return ci(out)
for h in HOR:
    known, y, w_km = cohort(h); w_cov = cov_weights(known.astype(int)) * known
    lm = (texp >= h) & ~((st == "interval") & (texp > h)); yl = ((st == "event") & (t80 <= h)).astype(int); ones = np.ones(len(d))
    log(f"h={h}: known {known.sum()}, landmark {lm.sum()}, KM weights {w_km[known].min():.2f}-{w_km[known].max():.2f}, covariate weights {w_cov[known].min():.2f}-{w_cov[known].max():.2f}")
    for proto in PROT:
        sp = splits(proto)
        for fs in FS:
            X = Xh(fs)
            for train_on, trmask, ytr, wtr in [("IPCW-KM", known, y, w_km), ("IPCW-covariate", known, y, w_cov), ("landmark", lm, yl, ones)]:
                pred = fit_pred(X, ytr, wtr, trmask, known | lm, sp)
                for eval_on, emask, ey, ew in [("known cohort, KM weights", known, y, w_km), ("known cohort, covariate weights", known, y, w_cov), ("known cohort, unweighted", known, y, ones), ("landmark subset", lm, yl, ones)]:
                    m = emask & ~np.isnan(pred)
                    if m.sum() < 50 or len(np.unique(ey[m])) < 2: continue
                    res.append(dict(analysis="decomposition", horizon=h, protocol=proto, features=fs, trained_on=train_on, scored_on=eval_on, n=int(m.sum()), auc=float(wauc(ey[m], pred[m], ew[m])), auc_ci=boot_auc(ey, pred, ew, m)))
                    log(f"  h={h} {proto:8s} {fs:24s} train {train_on:15s} score {eval_on:32s} n={m.sum()} AUC={res[-1]['auc']:.3f}")
    json.dump(res, open(RES / ("stability_ipcw_decomposition" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1)

from sklearn.metrics import roc_auc_score as AUC
def simulate(seed, n=7000, h=700.0, signal=1.0, gamma=0.6, sig=0.9, beta_c=0.5):
    r = np.random.default_rng(seed); x = r.normal(size=(n, 4)); b = np.array([0.8, 0.5, -0.4, 0.2]) * signal
    logT = 6.0 + x @ b + 0.8 * r.normal(size=n); T = np.exp(logT)
    logC = logT * 0.5 + 3.0 + gamma + beta_c * x[:, 0] + sig * r.normal(size=n); C = np.exp(logC)
    fail_true = (T <= h).astype(int); score = -(x @ b)
    obs_fail = (T <= h) & (T <= C); obs_surv = (T > h) & (C >= h); known = obs_fail | obs_surv
    auc_true = AUC(fail_true, score)
    G = lambda t: np.mean(C[:, None] > np.array(t)[None, :], axis=0) if False else None
    Tobs = np.minimum(np.minimum(T, C), h); cens = (C < np.minimum(T, h)).astype(int)
    o = np.argsort(Tobs); Ts = Tobs[o]; cs = cens[o]; S = 1.0; ts = []; vs = []
    for u in np.unique(Ts):
        at = Ts == u; dc = cs[at].sum(); nr = (Ts >= u).sum()
        if dc > 0: S *= 1 - dc / nr
        ts.append(u); vs.append(S)
    ts = np.array(ts); vs = np.array(vs); i = np.searchsorted(ts, Tobs, side="left") - 1; Gk = np.where(i < 0, 1.0, vs[np.clip(i, 0, len(vs) - 1)]); w_km = known / np.clip(Gk, 0.05, 1)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(x, known.astype(int)); w_cv = known / np.clip(lr.predict_proba(x)[:, 1], 0.05, 1)
    lm = (C >= h)
    out = dict(true=auc_true, km=AUC(fail_true[known], score[known], sample_weight=w_km[known]), cov=AUC(fail_true[known], score[known], sample_weight=w_cv[known]), unweighted=AUC(fail_true[known], score[known]), landmark=AUC(fail_true[lm], score[lm]) if len(np.unique(fail_true[lm])) == 2 else np.nan, known_share=float(known.mean()), landmark_share=float(lm.mean()))
    return out
sims = []
for name, kw in [("informative test length (gamma=0.6)", dict()), ("shorter tests (gamma=0)", dict(gamma=0.0)), ("weak signal", dict(signal=0.4))]:
    if name.startswith("no dependence"):
        pass
    rows = [simulate(s, **kw) for s in range(60 if QUICK else 200)]
    sims.append(dict(scenario=name, **{k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}, **{k + "_sd": float(np.nanstd([r[k] for r in rows])) for k in ["true", "km", "cov", "unweighted", "landmark"]}))
    log(f"sim {name}: " + ", ".join(f"{k}={sims[-1][k]:.3f}" for k in ["true", "km", "cov", "unweighted", "landmark", "known_share", "landmark_share"]))
json.dump(dict(decomposition=res, simulation=sims), open(RES / ("stability_ipcw_decomposition" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1)
log("DONE")
