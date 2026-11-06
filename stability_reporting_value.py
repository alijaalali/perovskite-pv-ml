import json, sys
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_enriched_features.py").read()
exec(compile(src[:src.index("MODELS = [")], "enriched_head", "exec"))
def cols(prefixes, df): return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
P_ = F["proc"]; S_ = F["stack"]
GROUPS = {"additives (identity, known/present, concentration)": ("proc", ["add_", "additive_"]), "quenching medium and quench-induced crystallisation": ("proc", ["quench"]), "solvents": ("proc", ["solv"]),
          "synthesis atmosphere": ("proc", ["synatm"]), "stoichiometric excess": ("proc", ["excess"]), "number of deposition steps": ("proc", ["nsteps"]), "perovskite thickness": ("proc", ["pk_thick"]),
          "full stack tokens, layer count, flexible/semitransparent": ("stack", ["stk_", "n_layers", "flexible", "semitransparent"]), "encapsulation edge-sealing/atmosphere known": ("stack", ["enc_"])}
parts = {}
for g, (blk, pref) in GROUPS.items(): parts[g] = cols(pref, F[blk])
log({g: len(c) for g, c in parts.items()})
Bdf = pd.concat([F[k] for k in ["time", "comp", "device", "protocol"]], axis=1); full = pd.concat([Bdf, P_, S_], axis=1)
def X_without(g): return full.drop(columns=parts[g]).values.astype(float)
def X_base_plus(g): return pd.concat([Bdf, F[GROUPS[g][0]][parts[g]]], axis=1).values.astype(float)
MODELS = ["XGBoost", "LightGBM"]; PROT = ["paper", "temporal"]
if QUICK: MODELS, PROT = ["XGBoost"], ["paper"]
res = []; w1 = np.ones(len(d))
raw_cov = {"additives (identity, known/present, concentration)": float(P_["additive_known"].mean()), "quenching medium and quench-induced crystallisation": float(1 - P_.get("quench_Unknown", pd.Series(0.0, index=P_.index)).mean()), "solvents": float(1 - P_.get("solv_Unknown", pd.Series(0.0, index=P_.index)).mean()),
           "synthesis atmosphere": float(1 - P_.get("synatm_Unknown", pd.Series(0.0, index=P_.index)).mean()), "perovskite thickness": float(1 - P_["pk_thick_miss"].mean()), "stoichiometric excess": float(1 - P_.get("excess_Unknown", pd.Series(0.0, index=P_.index)).mean()), "number of deposition steps": float(1 - P_["nsteps_miss"].mean()),
           "encapsulation edge-sealing/atmosphere known": float(S_[["enc_edge_known", "enc_atm_known"]].mean().mean()), "full stack tokens, layer count, flexible/semitransparent": 1.0}
for proto in PROT:
    sp = splits(proto)
    for mname in MODELS:
        def run(X):
            pred = np.full(len(d), np.nan)
            for tr, te in sp: pred[te] = fit_pred(make_reg(mname), X[tr], y_reg[tr], X[te])
            return pred
        pf = run(full.values.astype(float)); pb = run(Bdf.values.astype(float)); m0 = ~np.isnan(pf)
        for g in GROUPS:
            if not parts[g]: continue
            pd_ = run(X_without(g)); pa = run(X_base_plus(g)); m = ~np.isnan(pf) & ~np.isnan(pd_)
            drop = [wr2(y_reg[m], pf[m], bw[pidx][m]) - wr2(y_reg[m], pd_[m], bw[pidx][m]) for bw in BW]; add = [wr2(y_reg[m], pa[m], bw[pidx][m]) - wr2(y_reg[m], pb[m], bw[pidx][m]) for bw in BW]
            res.append(dict(protocol=proto, model=mname, group=g, n_cols=len(parts[g]), reported_share=raw_cov.get(g), loss_if_removed=float(wr2(y_reg[m], pf[m], w1[m]) - wr2(y_reg[m], pd_[m], w1[m])), loss_ci=ci(drop), gain_if_added_to_base=float(wr2(y_reg[m], pa[m], w1[m]) - wr2(y_reg[m], pb[m], w1[m])), gain_ci=ci(add)))
            log(f"{proto:8s} {mname:9s} {g:60s} loss if removed {res[-1]['loss_if_removed']:+.4f}  gain if added {res[-1]['gain_if_added_to_base']:+.4f}")
        json.dump(res, open(RES / ("stability_reporting_value" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1)
log("DONE")
