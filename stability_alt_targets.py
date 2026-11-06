import json, sys, re
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_enriched_features.py").read()
exec(compile(src[:src.index("MODELS = [")], "enriched_head", "exec"))
pf = pd.to_numeric(d.pce_fresh, errors="coerce"); ret = d.retention_end.clip(0, 1.2).values
y_pce = (pf * ret).values
loss = np.clip(1 - d.retention_end.values, 0.01, 1.0); y_rate = np.log10(loss / (d.t_exp_h.values / 1000.0))
log(f"fresh PCE coverage {100*pf.notna().mean():.1f}%, final-PCE target range {np.nanmin(y_pce):.2f}-{np.nanmax(y_pce):.2f}")
def fe_effect(mask_free, y, label):
    sub = d[(d.b_site_class == "lead_only") | mask_free].copy(); sub["x"] = mask_free[sub.index].astype(float); sub["yy"] = y[sub.index]; sub["logt"] = np.log10(sub.t_exp_h)
    sub = sub[np.isfinite(sub.yy)]; sub = sub[sub.groupby("paper_id").paper_id.transform("size") >= 2]
    g = sub.paper_id.values; X = sub[["x", "logt"]].values.astype(float)
    Xd = X - pd.DataFrame(X).groupby(g).transform("mean").values; yd = sub.yy.values - pd.Series(sub.yy.values).groupby(g).transform("mean").values
    keep_papers = np.abs(Xd[:, 0]) > 1e-12
    if keep_papers.sum() < 5: return dict(label=label, n_free=int(mask_free[sub.index].sum()), papers_free=0, beta=np.nan)
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None); r = yd - Xd @ beta; XtXi = np.linalg.pinv(Xd.T @ Xd); meat = np.zeros((2, 2))
    for pid in np.unique(g):
        m = g == pid; s = Xd[m].T @ r[m]; meat += np.outer(s, s)
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi)); pw = sub[sub.x == 1].paper_id.nunique(); pmix = len(set(sub[sub.x == 1].paper_id) & set(sub[sub.x == 0].paper_id))
    return dict(label=label, n_free=int(sub.x.sum()), papers_free=int(pw), papers_with_both=int(pmix), beta=float(beta[0]), se=float(se[0]), t=float(beta[0] / se[0]))
form = d.formula_str.astype(str)
def els(s): return set(re.findall(r"([A-Z][a-z]?)", re.sub(r"[0-9.]", "", s)))
free = (d.b_site_class == "lead_free").values; isSn = np.array([("Sn" in els(s)) and not (els(s) & {"Bi", "Sb", "Ag", "Cu", "In", "Fe", "O"}) for s in form]) & free
isDouble = np.array([bool(els(s) & {"Ag", "Cu", "In", "Fe", "O", "Na", "K"}) and bool(els(s) & {"Bi", "Sb"}) or ("Ag" in els(s)) or ("Fe" in els(s)) for s in form]) & free
sets = {"all lead-free": free, "Sn-based only": isSn, "excluding double/oxide (Ag, Cu, In, Fe, O with Bi/Sb)": free & ~isDouble, "Bi/Sb-based (no Sn)": free & ~isSn & ~isDouble}
partA = []
for tname, y in [("retention (pp)", 100 * ret), ("final efficiency (percentage points)", y_pce), ("log10 loss per 1000 h", y_rate)]:
    for lab, mk_ in sets.items():
        r = fe_effect(pd.Series(mk_, index=d.index), pd.Series(y, index=d.index), lab); r["target"] = tname; partA.append(r); log(str(r))
json.dump(partA, open(RES / "stability_lead_free_effect.json", "w"), indent=1)
BASE = ["time", "comp", "device", "protocol"]
SETB = {"time only": ["time"], "base": BASE, "+process+stack": BASE + ["proc", "stack"]}
def Xb(n): return pd.concat([F[k] for k in SETB[n]], axis=1).values.astype(float)
cells = pd.Series(paper).map(pd.Series(paper).value_counts()).values; wp = 1.0 / cells
TARGETS = {"retention": y_reg, "final efficiency": y_pce, "log10 loss rate": y_rate}
MODELSB = ["Ridge", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
PROT = ["paper", "temporal"]
if QUICK: MODELSB, PROT = ["Ridge", "XGBoost"], ["paper"]
out = []
for tname, y in TARGETS.items():
    okc = np.isfinite(y)
    for proto in PROT:
        sp = [(tr[okc[tr]], te[okc[te]]) for tr, te in splits(proto)]
        for mname in MODELSB:
            for fs in SETB:
                X = Xb(fs); pred = np.full(len(d), np.nan)
                for tr, te in sp: pred[te] = fit_pred(mk(mname) if mname == "CatBoost" else make_reg(mname), X[tr], y[tr], X[te])
                m = ~np.isnan(pred) & okc; yy, pp = y[m], pred[m]
                r2 = float(wr2(yy, pp, np.ones(m.sum()))); r2w = float(wr2(yy, pp, wp[m]))
                bs = [wr2(yy, pp, BW[b][pidx][m]) for b in range(len(BW))]; bsw = [wr2(yy, pp, BW[b][pidx][m] * wp[m]) for b in range(len(BW))]
                out.append(dict(target=tname, protocol=proto, model=mname, features=fs, n=int(m.sum()), r2=r2, r2_ci=ci(bs), r2_paper_weighted=r2w, r2_paper_weighted_ci=ci(bsw)))
                log(f"  {tname:16s} {proto:8s} {mname:12s} {fs:15s} R2={r2:.3f} weighted={r2w:.3f}")
        json.dump(out, open(RES / ("stability_alt_targets" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1)
log("DONE")
