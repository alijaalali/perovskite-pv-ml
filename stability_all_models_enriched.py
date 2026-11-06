import json, sys, pickle
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_benchmark_more_models.py").read()
exec(compile(src[:src.index('PROTOS3 = ["paper"')], "more_head", "exec"))
ORIG = ["Ridge", "ElasticNet", "RandomForest", "XGBoost", "LightGBM", "MLP"]
ALL = ORIG + [m for m in REG2] + [m for m in REG3]
def maker(name):
    if name in REG3: return mk3(name)
    if name in REG2: return make_reg2(name)
    return mk(name)
SETY = {"base": BASE, "+process+stack": BASE + ["proc", "stack"], "+all enriched": BASE + ["proc", "stack", "pcefresh"]}
def Xs(n): return pd.concat([F[k] for k in SETY[n]], axis=1).values.astype(float)
PROT = ["paper", "temporal"]
if QUICK: ALL, PROT = ALL[:3] + ["kNN", "Blend (XGB+CatBoost+RF)"], ["paper"]
results = []; oofs = {}; w1 = np.ones(len(d))
for proto in PROT:
    sp = splits(proto); log(f"== {proto}")
    for mname in ALL:
        for fs in SETY:
            X = Xs(fs); pred = np.full(len(d), np.nan)
            try:
                for tr, te in sp: pred[te] = fit_pred2(maker(mname), X[tr], y_reg[tr], X[te])
            except Exception as e:
                log(f"  FAILED {mname} {fs}: {e}"); continue
            oofs[(proto, mname, fs)] = pred; m = ~np.isnan(pred)
            results.append(dict(task="retention", protocol=proto, model=mname, features=fs, r2=float(wr2(y_reg[m], pred[m], w1[m])), r2_ci=ci([wr2(y_reg[m], pred[m], bw[pidx][m]) for bw in BW]), n_features=int(X.shape[1])))
            log(f"  {mname:38s} {fs:16s} R2={results[-1]['r2']:.3f}")
        for fs in ("+process+stack", "+all enriched"):
            if (proto, mname, fs) in oofs and (proto, mname, "base") in oofs:
                pb, pe = oofs[(proto, mname, "base")], oofs[(proto, mname, fs)]; m = ~np.isnan(pb) & ~np.isnan(pe)
                dl = [wr2(y_reg[m], pe[m], bw[pidx][m]) - wr2(y_reg[m], pb[m], bw[pidx][m]) for bw in BW]
                results.append(dict(task="delta_retention", protocol=proto, model=mname, features=fs, delta_r2=float(wr2(y_reg[m], pe[m], w1[m]) - wr2(y_reg[m], pb[m], w1[m])), delta_ci=ci(dl)))
        json.dump(results, open(RES / ("stability_all_models_enriched" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1, default=str)
pickle.dump(oofs, open(RES / ("stability_all_models_enriched_oof" + ("_quick" if QUICK else "") + ".pkl"), "wb"))
log("DONE")
