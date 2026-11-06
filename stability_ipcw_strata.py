import json, sys
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_ipcw_horizons.py").read()
exec(compile(src[:src.index("HOR = [100, 500, 1000]")], "ipcw_head", "exec"))
X = Xh("+all enriched (no time)"); res = []
for h in [500, 1000]:
    known, y, w_km = cohort(h); lm = (texp >= h) & ~((st == "interval") & (texp > h)); yl = ((st == "event") & (t80 <= h)).astype(int)
    for proto in ["paper", "temporal"]:
        pred = np.full(len(d), np.nan)
        for tr, te in splits(proto):
            trk = tr[known[tr]]; tek = te[known[te] | lm[te]]
            if len(np.unique(y[trk])) < 2: continue
            mdl = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, tree_method="hist"); mdl.fit(X[trk], y[trk], sample_weight=w_km[trk]); pred[tek] = mdl.predict_proba(X[tek])[:, 1]
        cuts = np.quantile(texp[known], [1 / 3, 2 / 3]); tb = np.digitize(texp, cuts)
        strata = [("all", np.ones(len(d), bool))] + [(f"ISOS {p}", isos == p) for p in ["ISOS-D-1", "ISOS-L-1", "ISOS-D-1I", "ISOS-L-1I"]] + [(f"exposure tertile {k+1} ({(cuts[k-1] if k else 0):.0f}-{(cuts[k] if k < 2 else np.inf):.0f} h)", tb == k) for k in range(3)]
        for nm, sm in strata:
            for cohort_nm, cm, ey, ew in [("known cohort, KM-weighted", known, y, w_km), ("landmark subset", lm, yl, np.ones(len(d)))]:
                m = cm & sm & ~np.isnan(pred)
                if m.sum() < 60 or len(np.unique(ey[m])) < 2: continue
                res.append(dict(horizon=h, protocol=proto, stratum=nm, cohort=cohort_nm, n=int(m.sum()), failed=int(ey[m].sum()), auc=float(wauc(ey[m], pred[m], ew[m]))))
                log(f"h={h} {proto:8s} {nm:34s} {cohort_nm:26s} n={m.sum()} AUC={res[-1]['auc']:.3f}")
json.dump(res, open(RES / "stability_ipcw_strata.json", "w"), indent=1); log("DONE")
