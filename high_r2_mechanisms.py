import json, sys
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_enriched_features.py").read()
exec(compile(src[:src.index("MODELS = [")], "enriched_head", "exec"))
pf = pd.to_numeric(d.pce_fresh, errors="coerce").values; ret = d.retention_end.clip(0, 1.2).values
t80 = d.t80_h.values.astype(float); ev = (d.status.values == "event") & np.isfinite(t80) & (t80 > 0)
TG = {"final efficiency": (pf * ret, np.isfinite(pf * ret)), "log10 T80 (exact-T80 cells)": (np.log10(np.where(ev, t80, 1.0)), ev), "retention": (y_reg, np.ones(len(d), bool))}
BASE = ["comp", "device", "protocol"]
FS = {"base (no time)": BASE, "base + exposure time": ["time"] + BASE, "base + fresh efficiency": BASE + ["pcefresh"], "base + exposure time + fresh efficiency": ["time"] + BASE + ["pcefresh"]}
res = []
for tn, (y, ok) in TG.items():
    for proto in ["random", "paper"]:
        sp = [(tr[ok[tr]], te[ok[te]]) for tr, te in splits(proto)]
        for fn, parts in FS.items():
            X = pd.concat([F[k] for k in parts], axis=1).values.astype(float); pred = np.full(len(d), np.nan)
            for tr, te in sp:
                m_ = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, random_state=SEED, tree_method="hist"); m_.fit(X[tr], y[tr]); pred[te] = m_.predict(X[te])
            m = ~np.isnan(pred) & ok; r2 = float(wr2(y[m], pred[m], np.ones(m.sum()))); res.append(dict(target=tn, protocol=proto, features=fn, n=int(m.sum()), r2=r2)); log(f"{tn:28s} {proto:7s} {fn:42s} n={m.sum()} R2={r2:.3f}")
json.dump(res, open(RES / "high_r2_mechanisms.json", "w"), indent=1); log("DONE")
