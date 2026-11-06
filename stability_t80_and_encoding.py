import json, sys
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_enriched_features.py").read()
exec(compile(src[:src.index("MODELS = [")], "enriched_head", "exec"))
from sklearn.metrics import r2_score
SFR = pd.read_parquet(DATA / "stability_site_fractions.parquet")
if QUICK: SFR = SFR.loc[d.index.values] if False else SFR.iloc[:0]
F["sitefrac"] = SFR.reset_index(drop=True).iloc[:len(d)] if not QUICK else pd.DataFrame(np.zeros((len(d), 1)))
res = {"t80": [], "encoding": []}
ev = np.where(d.status.values == "event")[0]; lt80 = np.log10(d.t80_h.values[ev]); log(f"event cells {len(ev)}; corr(log T80, log t_exp) = {np.corrcoef(lt80, np.log10(d.t_exp_h.values[ev]))[0,1]:.3f}")
sets = {"exposure time only": ["time"], "composition + device (no time)": ["comp", "device"], "all features without exposure time": ["comp", "device", "protocol"], "all features with exposure time": ["time", "comp", "device", "protocol"]}
for proto in (["random", "paper"] if not QUICK else ["random"]):
    idx = np.arange(len(ev))
    sp = list(KFold(5, shuffle=True, random_state=SEED).split(idx)) if proto == "random" else list(GroupKFold(5).split(idx, groups=paper[ev]))
    for mname in ["XGBoost", "RandomForest", "Ridge"]:
        for sn, parts in sets.items():
            X = pd.concat([F[k] for k in parts], axis=1).values.astype(float)[ev]; pred = np.full(len(ev), np.nan)
            for tr, te in sp: pred[te] = fit_pred(make_reg(mname), X[tr], lt80[tr], X[te])
            r2 = float(r2_score(lt80, pred)); res["t80"].append(dict(protocol=proto, model=mname, features=sn, n=int(len(ev)), r2=r2)); log(f"T80 {proto:6s} {mname:12s} {sn:38s} R2={r2:.3f}")
json.dump(res, open(RES / ("stability_t80_and_encoding" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1)
encsets = {"no composition": ["time", "device", "protocol"], "Magpie statistics": ["time", "comp", "device", "protocol"], "site fractions + tolerance factor": ["time", "sitefrac", "device", "protocol"], "Magpie + site fractions": ["time", "comp", "sitefrac", "device", "protocol"]}
for proto in (["paper", "temporal"] if not QUICK else ["paper"]):
    sp = splits(proto)
    for mname in (["Ridge", "XGBoost", "LightGBM", "RandomForest"] if not QUICK else ["XGBoost"]):
        for sn, parts in encsets.items():
            X = pd.concat([F[k] for k in parts], axis=1).values.astype(float); pred = np.full(len(d), np.nan)
            for tr, te in sp: pred[te] = fit_pred(make_reg(mname), X[tr], y_reg[tr], X[te])
            m = ~np.isnan(pred); bs = [wr2(y_reg[m], pred[m], bw[pidx][m]) for bw in BW]
            res["encoding"].append(dict(protocol=proto, model=mname, features=sn, n_features=int(X.shape[1]), r2=float(wr2(y_reg[m], pred[m], np.ones(m.sum()))), r2_ci=ci(bs))); log(f"enc {proto:8s} {mname:12s} {sn:36s} R2={res['encoding'][-1]['r2']:.3f}")
    json.dump(res, open(RES / ("stability_t80_and_encoding" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1)
log("DONE")
