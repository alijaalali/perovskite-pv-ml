import json, sys
from pathlib import Path
src = open(__import__("pathlib").Path(__file__).resolve().parent / "within_paper_contrasts.py").read()
exec(compile(src[:src.index("res = {}")], "wpc_head", "exec"))
from scipy import stats, optimize
pairs = contrasts(0.1); o = orient(pairs)
cov = d.groupby("paper_id").agg(temp=("stab_temp_C", "mean"), rh=("stab_RH", "mean"), light=("stab_light_intensity", "mean"), logt=("logt", "mean"), year=("year", "mean"))
def reml(y, v, X):
    k, p = X.shape
    def nll(t):
        w = 1 / (v + t); XtW = X.T * w; A = XtW @ X
        try: b = np.linalg.solve(A, XtW @ y)
        except np.linalg.LinAlgError: return 1e10
        r = y - X @ b; return 0.5 * (np.sum(np.log(v + t)) + np.linalg.slogdet(A)[1] + np.sum(w * r * r))
    hi = max(10 * np.var(y), 1e-4); t = optimize.minimize_scalar(nll, bounds=(0, hi), method="bounded").x
    return t if nll(t) < nll(0) else 0.0
def hk(y, v, X, tau2=None):
    k, p = X.shape; t = reml(y, v, X) if tau2 is None else tau2; w = 1 / (v + t); A = (X.T * w) @ X; b = np.linalg.solve(A, (X.T * w) @ y)
    r = y - X @ b; q = max(1.0, np.sum(w * r * r) / (k - p)); V = q * np.linalg.inv(A); se = np.sqrt(np.diag(V)); tq = stats.t.ppf(0.975, k - p)
    return t, b, se, tq, w
rows = []; per = {}
for (f, ref, lvl), g in o.groupby(["factor", "ref", "level"]):
    pm = g.groupby("paper").agg(dy=("dy", "mean"), n=("dy", "size"), sd=("dy", "std")); k = len(pm)
    if k < 5: continue
    y = 100 * pm.dy.values; sdp = 100 * pm.sd.dropna().values; s2 = float(np.median(sdp ** 2)) if len(sdp) else float(np.var(y)); n = pm.n.values.astype(float)
    v = np.where(pm.sd.notna().values, np.maximum((100 * pm.sd.fillna(0).values) ** 2, 0.25 * s2), s2) / n
    X = np.ones((k, 1)); t2, b, se, tq, w = hk(y, v, X); mu = float(b[0]); half = tq * float(se[0])
    pi = stats.t.ppf(0.975, max(k - 2, 1)) * np.sqrt(t2 + float(se[0]) ** 2); I2 = float(max(0, (np.sum(w * (y - np.sum(w * y) / w.sum()) ** 2) - (k - 1)) / np.sum(w * (y - np.sum(w * y) / w.sum()) ** 2))) if True else np.nan
    se_i = np.sqrt(v); egg_p = pet = peese = np.nan
    if k >= 10 and np.std(se_i) > 0:
        res_ = stats.linregress(1 / se_i, y / se_i); egg_p = float(2 * stats.t.sf(abs(res_.intercept / res_.intercept_stderr), k - 2))
        pet_r = stats.linregress(se_i, y); pet = float(pet_r.intercept); pet_p = float(2 * stats.t.sf(abs(pet_r.intercept / pet_r.intercept_stderr), k - 2))
        peese = float(stats.linregress(v, y).intercept); final = peese if pet_p < 0.1 and pet > 0 else pet
    else: final = np.nan
    rows.append(dict(factor=f, reference=ref, level=lvl, papers=k, naive_mean=float(y.mean()), re_mean=mu, hk_lo=mu - half, hk_hi=mu + half, tau=float(np.sqrt(t2)), I2=I2, pred_lo=mu - pi, pred_hi=mu + pi, egger_p=egg_p, pet_intercept=pet, peese_intercept=peese, pet_peese=final))
    per[(f, lvl)] = (pm.index.values, y, v)
MA = pd.DataFrame(rows).sort_values("papers", ascending=False); MA.to_csv(DATA / "contrast_meta_analysis.csv", index=False)
log("\n" + MA.round(2).to_string(index=False))
mr = []
for (f, lvl), (pp, y, v) in per.items():
    k = len(y)
    if k < 12: continue
    C = cov.loc[pp]
    for name in ["logt", "temp", "rh", "light", "year"]:
        z = C[name].values.astype(float); ok = ~np.isnan(z)
        if ok.sum() < 10 or np.std(z[ok]) == 0: continue
        z = (z[ok] - z[ok].mean()) / z[ok].std(); X = np.column_stack([np.ones(ok.sum()), z]); t2, b, se, tq, w = hk(y[ok], v[ok], X)
        mr.append(dict(factor=f, level=lvl, covariate=name, k=int(ok.sum()), intercept=float(b[0]), slope_per_sd=float(b[1]), slope_lo=float(b[1] - tq * se[1]), slope_hi=float(b[1] + tq * se[1]), tau=float(np.sqrt(t2))))
MR = pd.DataFrame(mr); MR.to_csv(DATA / "contrast_meta_regression.csv", index=False); log("\n" + MR.round(2).to_string(index=False))
rng = np.random.default_rng(0); rec = []
for pid, g in d.groupby("paper_id", sort=False):
    if len(g) < 2: continue
    gi = g.reset_index(drop=True); n = len(gi); iu, ju = np.triu_indices(n, 1)
    same = (gi.isos.values[iu] == gi.isos.values[ju]) & (np.abs(gi.logt.values[iu] - gi.logt.values[ju]) < 0.1)
    F = gi[FACTORS].values
    for a, b_ in zip(iu[same], ju[same]):
        if all(F[a, k] == F[b_, k] for k in range(len(FACTORS))): rec.append((pid, gi.y.values[b_] - gi.y.values[a]))
PL = pd.DataFrame(rec, columns=["paper", "dy"]).groupby("paper").dy.mean(); pl = 100 * PL.values
plac = dict(n_papers=int(len(pl)), mean_pp=float(pl.mean()), sd_pp=float(pl.std()), share_positive=float((pl > 0).mean()), t=float(stats.ttest_1samp(pl, 0).statistic), p=float(stats.ttest_1samp(pl, 0).pvalue))
log(f"placebo (no coded factor differs, oriented by row order): {plac}")
PR = pd.read_csv(DATA / "prospective_contrasts.csv").dropna(subset=["test_pp"]); agree = int((np.sign(PR.train_pp) == np.sign(PR.test_pp)).sum()); n = len(PR)
p_random = float(stats.binom.sf(agree - 1, n, 0.5)); q_pos = float((PR.train_pp > 0).mean()); share_pos_all = float((100 * o.groupby(["factor", "level", "paper"]).dy.mean() > 0).mean())
p_sel = float(stats.binom.sf(agree - 1, n, share_pos_all))
sign = dict(agree=agree, n=n, p_if_signs_random=p_random, share_positive_all_oriented_paper_effects=share_pos_all, p_if_each_test_contrast_positive_with_that_share=p_sel, all_training_positive=bool((PR.train_pp > 0).all()), shrink_ratio_median=float((PR.test_pp / PR.train_pp).median()))
log(f"sign agreement: {sign}")
json.dump(dict(placebo=plac, sign_agreement=sign), open(DATA / "contrast_meta_diagnostics.json", "w"), indent=1); log("DONE")
