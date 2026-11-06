import json, sys
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_enriched_features.py").read()
exec(compile(src[:src.index("MODELS = [")], "enriched_head", "exec"))
BASE = ["time", "comp", "device", "protocol"]; SETS = {"base": BASE, "+process+stack": BASE + ["proc", "stack"]}; ALPHA = 0.10
PAR = dict(n_estimators=500, max_depth=3, learning_rate=0.02, subsample=0.6, colsample_bytree=0.3, min_child_weight=20, reg_lambda=50.0)
rng = np.random.default_rng(11); res = []; texp = d.t_exp_h.values; tb = np.digitize(texp, np.quantile(texp, [1 / 3, 2 / 3]))
def q_of(r): n = len(r); return float(np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))
for proto in ["paper", "temporal"]:
    sp = splits(proto)
    for fs, parts in SETS.items():
        X = pd.concat([F[k] for k in parts], axis=1).values.astype(float)
        for calib in ["paper-grouped", "cell-level"]:
            lo = np.full(len(d), np.nan); hi = np.full(len(d), np.nan); pred = np.full(len(d), np.nan)
            for tr, te in sp:
                pu = np.unique(pidx[tr]); cal_p = rng.choice(pu, max(1, int(0.2 * len(pu))), replace=False)
                if calib == "paper-grouped": cal = tr[np.isin(pidx[tr], cal_p)]
                else: cal = rng.choice(tr, int(0.2 * len(tr)), replace=False)
                fit = np.setdiff1d(tr, cal); m = xgb.XGBRegressor(**PAR, n_jobs=NJ, random_state=SEED, tree_method="hist").fit(X[fit], y_reg[fit])
                q = q_of(np.abs(y_reg[cal] - m.predict(X[cal]))); p = m.predict(X[te]); pred[te] = p; lo[te] = p - q; hi[te] = p + q
            ok = ~np.isnan(pred); cov = (y_reg[ok] >= lo[ok]) & (y_reg[ok] <= hi[ok]); w = hi[ok] - lo[ok]
            pc = pd.Series(cov.astype(float)).groupby(pidx[ok]).mean(); bs = []
            for bw in BW: b = bw[pidx][ok]; bs.append(float(np.sum(b * cov) / np.sum(b)))
            rows = dict(protocol=proto, features=fs, calibration=calib, n=int(ok.sum()), coverage=float(cov.mean()), coverage_ci=ci(bs), mean_width=float(w.mean()), paper_mean_coverage=float(pc.mean()), share_papers_below_80=float((pc < 0.8).mean()),
                        coverage_by_exposure_tertile=[float(cov[tb[ok] == k].mean()) for k in range(3)], r2=float(wr2(y_reg[ok], pred[ok], np.ones(ok.sum()))))
            res.append(rows); log(f"{proto:8s} {fs:15s} {calib:14s} coverage {rows['coverage']:.3f} {rows['coverage_ci']} width {rows['mean_width']:.3f} paper-mean {rows['paper_mean_coverage']:.3f} tertiles {np.round(rows['coverage_by_exposure_tertile'],3)}")
json.dump(res, open(RES / "stability_conformal.json", "w"), indent=1); log("DONE")
