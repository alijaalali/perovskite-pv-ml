import json, sys, re
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_enriched_features.py").read()
exec(compile(src[:src.index("MODELS = [")], "enriched_head", "exec"))
pf = pd.to_numeric(d.pce_fresh, errors="coerce"); ret = d.retention_end.clip(0, 1.2).values
Y = {"retention (pp)": 100 * ret, "final efficiency (pp of PCE)": (pf * ret).values}
form = d.formula_str.astype(str)
def els(s): return set(re.findall(r"([A-Z][a-z]?)", re.sub(r"[0-9.]", "", s)))
free = (d.b_site_class == "lead_free").values
isDouble = np.array([bool(els(s) & {"Ag", "Cu", "In", "Fe", "O", "Na", "K"}) or False for s in form]) & free
isSn = np.array(["Sn" in els(s) for s in form]) & free & ~isDouble; isBiSb = np.array([bool(els(s) & {"Bi", "Sb"}) for s in form]) & free & ~isDouble & ~isSn
sets = {"all lead-free": free, "excluding double and oxide perovskites": free & ~isDouble, "Sn-based only": isSn, "Bi/Sb-based only": isBiSb}
def fe(mask_free, y, label, target):
    other = ~free; keep = (mask_free | other) & np.isfinite(y); sub = d[keep].copy(); sub["x"] = mask_free[keep].astype(float); sub["yy"] = y[keep]; sub["logt"] = np.log10(sub.t_exp_h)
    sub = sub[sub.groupby("paper_id").paper_id.transform("size") >= 2]; g = sub.paper_id.values; X = sub[["x", "logt"]].values.astype(float)
    Xd = X - pd.DataFrame(X).groupby(g).transform("mean").values; yd = sub.yy.values - pd.Series(sub.yy.values).groupby(g).transform("mean").values
    contrib = pd.Series(np.abs(Xd[:, 0]) > 1e-12).groupby(g).any(); npap = int(contrib.sum())
    if npap < 3: return dict(label=label, target=target, cells_free=int(sub.x.sum()), papers_contributing=npap, beta=None)
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None); r = yd - Xd @ beta; XtXi = np.linalg.pinv(Xd.T @ Xd); meat = np.zeros((2, 2))
    for pid in np.unique(g):
        m = g == pid; s = Xd[m].T @ r[m]; meat += np.outer(s, s)
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi)); G_ = len(np.unique(g)); se = se * np.sqrt(G_ / (G_ - 1))
    return dict(label=label, target=target, cells_free=int(sub.x.sum()), papers_contributing=npap, beta=float(beta[0]), se=float(se[0]), lo=float(beta[0] - 1.96 * se[0]), hi=float(beta[0] + 1.96 * se[0]))
out = []
for tn, y in Y.items():
    for lab, mk_ in sets.items(): out.append(fe(mk_, y, lab, tn)); log(str(out[-1]))
json.dump(out, open(RES / "lead_free_contrast.json", "w"), indent=1); log("DONE")
