import json, sys
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_enriched_features.py").read()
exec(compile(src[:src.index("MODELS = [")], "enriched_head", "exec"))
from sklearn.metrics import roc_auc_score
BASE = ["comp", "device", "protocol"]
SETH = {"protocol only": ["protocol"], "base (no time)": BASE, "+process+stack (no time)": BASE + ["proc", "stack"], "+all enriched (no time)": BASE + ["proc", "stack", "pcefresh"]}
def Xh(n): return pd.concat([F[k] for k in SETH[n]], axis=1).values.astype(float)
isos = d.isos.astype(str).values; texp = d.t_exp_h.values.astype(float); t80 = d.t80_h.values.astype(float); st = d.status.values
Tobs = np.where(st == "event", t80, texp)
is_cens = st == "right"
def km_censoring(T, c):
    o = np.argsort(T); Ts, cs = T[o], c[o]; n = len(T); S = 1.0; times = []; vals = []
    uniq = np.unique(Ts)
    for u in uniq:
        at = (Ts == u); dcen = cs[at].sum(); nr = (Ts >= u).sum()
        if dcen > 0: S *= 1 - dcen / nr
        times.append(u); vals.append(S)
    return np.array(times), np.array(vals)
KM = {}
for p in np.unique(isos):
    m = isos == p
    if m.sum() < 30: continue
    KM[p] = km_censoring(Tobs[m], is_cens[m])
KMall = km_censoring(Tobs, is_cens)
def G(t, p):
    ts, vs = KM.get(p, KMall); i = np.searchsorted(ts, t, side="left") - 1; return np.where(i < 0, 1.0, vs[np.clip(i, 0, len(vs) - 1)])
def cohort(h):
    surv = (st == "right") & (texp >= h) | (st == "event") & (t80 > h) & (texp >= h)
    fail = (st == "event") & (t80 <= h) | (st == "interval") & (texp <= h)
    known = surv | fail; y = fail.astype(int)
    tt = np.minimum(Tobs, h); w = np.array([1.0 / max(float(G(np.array([tt[i]]), isos[i])[0]), 0.05) for i in range(len(d))]) * known
    return known, y, w
def wauc(y, p, w):
    try: return roc_auc_score(y, p, sample_weight=w)
    except Exception: return np.nan
def wbrier(y, p, w): return float(np.sum(w * (y - p) ** 2) / np.sum(w))
HOR = [100, 500, 1000]; PROT = ["paper", "temporal"]; MOD = ["XGBoost", "Logistic"]
if QUICK: HOR, PROT, MOD = [500], ["paper"], ["XGBoost"]
res = []
for h in HOR:
    known, y, w = cohort(h)
    log(f"horizon {h} h: known {known.sum()} (failed {int(y[known].sum())}), weights range {w[known].min():.2f}-{w[known].max():.2f}, papers {len(np.unique(pidx[known]))}")
    for proto in PROT:
        sp = splits(proto)
        for mname in MOD:
            for fs in SETH:
                X = Xh(fs); pred = np.full(len(d), np.nan)
                for tr, te in sp:
                    trk = tr[known[tr]]; tek = te[known[te]]
                    if len(trk) < 50 or len(np.unique(y[trk])) < 2: continue
                    if mname == "XGBoost": mdl = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, tree_method="hist"); mdl.fit(X[trk], y[trk], sample_weight=w[trk]); pred[tek] = mdl.predict_proba(X[tek])[:, 1]
                    else:
                        sc = StandardScaler().fit(X[trk]); A, B = np.clip(sc.transform(X[trk]), -8, 8), np.clip(sc.transform(X[tek]), -8, 8)
                        mdl = LogisticRegressionCV(Cs=6, cv=3, max_iter=2000, n_jobs=NJ); mdl.fit(A, y[trk], sample_weight=w[trk] / w[trk].mean()); pred[tek] = mdl.predict_proba(B)[:, 1]
                m = known & ~np.isnan(pred); ww = w * m
                auc = wauc(y[m], pred[m], w[m]); br = wbrier(y[m], pred[m], w[m])
                null = float(np.sum(w[m] * y[m]) / np.sum(w[m])); br0 = wbrier(y[m], np.full(m.sum(), null), w[m])
                ba, bb = [], []
                for b in range(len(BW)):
                    bw = BW[b][pidx][m] * w[m]
                    if bw.sum() > 0: ba.append(wauc(y[m], pred[m], bw)); bb.append(1 - wbrier(y[m], pred[m], bw) / wbrier(y[m], np.full(m.sum(), np.sum(bw * y[m]) / bw.sum()), bw))
                res.append(dict(analysis="ipcw", horizon=h, protocol=proto, model=mname, features=fs, n=int(m.sum()), auc=float(auc), auc_ci=ci(ba), brier=br, brier_null=br0, brier_skill=float(1 - br / br0), brier_skill_ci=ci(bb)))
                log(f"  IPCW h={h} {proto:8s} {mname:8s} {fs:26s} AUC={auc:.3f} BrierSkill={1-br/br0:+.3f}")
                if proto == "paper" and mname == "XGBoost":
                    for p in ["ISOS-D-1", "ISOS-L-1", "ISOS-D-1I", "ISOS-L-1I"]:
                        mp = m & (isos == p)
                        if mp.sum() > 80 and len(np.unique(y[mp])) == 2: res.append(dict(analysis="per_isos", horizon=h, isos=p, features=fs, n=int(mp.sum()), auc=float(wauc(y[mp], pred[mp], w[mp]))))
    lm = (texp >= h) & ~((st == "interval") & (texp > h)); yl = ((st == "event") & (t80 <= h)).astype(int)
    for proto in PROT:
        sp = splits(proto)
        for fs in SETH:
            X = Xh(fs); pred = np.full(len(d), np.nan)
            for tr, te in sp:
                trk = tr[lm[tr]]; tek = te[lm[te]]
                if len(trk) < 50 or len(np.unique(yl[trk])) < 2: continue
                mdl = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, tree_method="hist"); mdl.fit(X[trk], yl[trk]); pred[tek] = mdl.predict_proba(X[tek])[:, 1]
            m = lm & ~np.isnan(pred)
            if m.sum() < 50 or len(np.unique(yl[m])) < 2: continue
            ba = []
            for b in range(len(BW)):
                bw = BW[b][pidx][m]
                if bw.sum() > 0: ba.append(wauc(yl[m], pred[m], bw))
            res.append(dict(analysis="landmark", horizon=h, protocol=proto, model="XGBoost", features=fs, n=int(m.sum()), failed=int(yl[m].sum()), auc=float(wauc(yl[m], pred[m], np.ones(m.sum()))), auc_ci=ci(ba)))
            log(f"  landmark h={h} {proto:8s} {fs:26s} n={m.sum()} AUC={res[-1]['auc']:.3f}")
    json.dump(res, open(RES / ("stability_ipcw_horizons" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1)
log("DONE")
