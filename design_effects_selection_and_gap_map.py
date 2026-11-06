import json, time, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize
PROJ = Path(__file__).resolve().parent; DATA = PROJ / "data"
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
d = pd.read_parquet(DATA / "stability_table.parquet")
raw = pd.read_csv(DATA / "perovskite_raw.csv", usecols=["Perovskite_additives_compounds", "Perovskite_deposition_quenching_media"], low_memory=False).loc[d.raw_idx.values].reset_index(drop=True)
d["y"] = d.retention_end.clip(0, 1.2); d["logt"] = np.log10(d.t_exp_h)
tok = raw.Perovskite_additives_compounds.fillna("Unknown").astype(str).str.split(";").apply(lambda l: [t.strip() for t in l])
d["F_additive"] = tok.apply(lambda l: "unknown" if "Unknown" in l else ("none" if all(t in ("Undoped", "") for t in l) else "additive"))
def simp(s, keep): s = s.astype(str).str.strip(); return s.where(s.isin(keep), "other")
def top(s, k): return list(s.value_counts().head(k).index)
d["F_architecture"] = simp(d.Cell_architecture, ["nip", "pin"])
d["F_ETL"] = simp(d.ETL_stack_sequence, top(d.ETL_stack_sequence, 6)); d["F_HTL"] = simp(d.HTL_stack_sequence, top(d.HTL_stack_sequence, 6)); d["F_backcontact"] = simp(d.Backcontact_stack_sequence, top(d.Backcontact_stack_sequence, 4))
d["F_encapsulation"] = simp(d.Encapsulation, ["True", "False"])
d["F_A_site"] = simp(d.a_site, top(d.a_site, 5)); d["F_B_site"] = d.b_site_class.astype(str)
d["F_deposition"] = simp(d.Perovskite_deposition_procedure, top(d.Perovskite_deposition_procedure, 3))
q = raw.Perovskite_deposition_quenching_media.astype(str).str.strip().replace({"nan": "Unknown"}); d["F_quench"] = q.where(q.isin(list(q.value_counts().head(4).index)), "other")
FACTORS = [c for c in d.columns if c.startswith("F_")]; UNK = {"unknown", "Unknown", "other"}
def matched_pairs(window=0.1):
    rec = []
    for pid, g in d.groupby("paper_id", sort=False):
        if len(g) < 2: continue
        gi = g.reset_index(drop=True); n = len(gi); iu, ju = np.triu_indices(n, 1)
        same = (gi.isos.values[iu] == gi.isos.values[ju]) & (np.abs(gi.logt.values[iu] - gi.logt.values[ju]) < window); iu, ju = iu[same], ju[same]
        F = gi[FACTORS].values
        for a, b in zip(iu, ju):
            diff = [k for k in range(len(FACTORS)) if F[a, k] != F[b, k]]
            if len(diff) != 1: continue
            k = diff[0]; la, lb = F[a, k], F[b, k]
            if la in UNK or lb in UNK: continue
            rec.append((pid, FACTORS[k][2:], la, lb, gi.y.values[a], gi.y.values[b]))
    return pd.DataFrame(rec, columns=["paper", "factor", "la", "lb", "ya", "yb"])

def any_pairs(window=0.1):
    codes = {f: pd.factorize(d[f])[0] for f in FACTORS}; levs = {f: pd.factorize(d[f])[1] for f in FACTORS}
    Cm = np.column_stack([codes[f] for f in FACTORS]); rec_p, rec_a, rec_b = [], [], []
    for pid, idx in d.groupby("paper_id", sort=False).indices.items():
        if len(idx) < 2: continue
        iu, ju = np.triu_indices(len(idx), 1); ia, ib = idx[iu], idx[ju]
        ok = (d.isos.values[ia] == d.isos.values[ib]) & (np.abs(d.logt.values[ia] - d.logt.values[ib]) < window)
        ia, ib = ia[ok], ib[ok]; rec_p += [pid] * len(ia); rec_a.append(Cm[ia]); rec_b.append(Cm[ib])
    return np.array(rec_p), np.vstack(rec_a), np.vstack(rec_b), levs
AP, AA, AB, LEVS = any_pairs(); log(f"all matched-protocol pairs (any number of differing factors): {len(AP)}")
def entangle(f, a, b_):
    k = FACTORS.index("F_" + f); la = list(LEVS["F_" + f]).index(a); lb = list(LEVS["F_" + f]).index(b_)
    m = ((AA[:, k] == la) & (AB[:, k] == lb)) | ((AA[:, k] == lb) & (AB[:, k] == la))
    if m.sum() == 0: return 0, 0, ""
    other = np.delete(np.arange(len(FACTORS)), k); dif = (AA[m][:, other] != AB[m][:, other]); share = dif.mean(axis=0); j = int(share.argmax())
    return int(len(np.unique(AP[m]))), float(share[j]), FACTORS[other[j]][2:] if share[j] >= 0.5 else ""
P = matched_pairs(0.1); log(f"matched single-factor pairs: {len(P)}")
ref = {f[2:]: d[f].value_counts().index[0] for f in FACTORS}
rows = []
for r in P.itertuples():
    R = ref[r.factor]
    if r.la == R and r.lb != R: rows.append((r.paper, r.factor, R, r.lb, 100 * (r.yb - r.ya)))
    elif r.lb == R and r.la != R: rows.append((r.paper, r.factor, R, r.la, 100 * (r.ya - r.yb)))
O = pd.DataFrame(rows, columns=["paper", "factor", "ref", "level", "dy"])
C = {}
for (f, rf, lv), g in O.groupby(["factor", "ref", "level"]):
    pm = g.groupby("paper").dy.mean().values
    if len(pm) >= 5: C[(f, rf, lv)] = pm
log(f"contrasts with >=5 papers: {len(C)}")
def nll(par, y, w):
    mu, ls = par; s = np.exp(ls); z = (0 - mu) / s; A = w * stats.norm.cdf(z) + (1 - stats.norm.cdf(z))
    return -(stats.norm.logpdf(y, mu, s) + (y < 0) * np.log(w) - np.log(np.clip(A, 1e-12, None))).sum()
def fit(y, w):
    best = None
    for mu0 in (y.mean(), y.mean() + 10):
        r = optimize.minimize(nll, [mu0, np.log(max(y.std(), 2.0))], args=(y, w), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=2000))
        if best is None or r.fun < best.fun: best = r
    return best.x[0], float(np.exp(best.x[1])), best.fun
OM = np.array([1, .75, .5, .35, .25, .15, .1, .05])
prof = {}
for key, y in C.items(): prof[key] = [fit(y, w) for w in OM]
tot = np.array([sum(prof[k][i][2] for k in C) for i in range(len(OM))])
ll = -tot; ll_max = ll.max(); om_hat = OM[ll.argmax()]
ci_om = OM[(2 * (ll_max - ll)) < 3.84]
log(f"pooled profile over omega: " + ", ".join(f"{w:.2f}:{x:.1f}" for w, x in zip(OM, ll - ll_max)) + f" | omega_hat={om_hat}, 95% LR grid set={ci_om.min()}-{ci_om.max()}")
nb = 60 if QUICK else 400; rng = np.random.default_rng(0); out = []
for key, y in C.items():
    k = len(y); tcrit = stats.t.ppf(0.975, k - 1)
    row = dict(factor=key[0], reference=key[1], level=key[2], papers=k, naive_mean=float(y.mean()), naive_sd=float(y.std(ddof=1)), frac_positive=float((y > 0).mean()),
               naive_pred_lo=float(y.mean() - tcrit * y.std(ddof=1) * np.sqrt(1 + 1 / k)), naive_pred_hi=float(y.mean() + tcrit * y.std(ddof=1) * np.sqrt(1 + 1 / k)))
    for w in (0.5, 0.25, 0.1):
        i = int(np.where(OM == w)[0][0]); mu, s, _ = prof[key][i]; row[f"mu_omega{w}"] = float(mu); row[f"sd_omega{w}"] = float(s)
        bs = [fit(y[rng.integers(0, k, k)], w)[0] for _ in range(nb)]; row[f"mu_omega{w}_lo"] = float(np.percentile(bs, 2.5)); row[f"mu_omega{w}_hi"] = float(np.percentile(bs, 97.5))
    out.append(row)
A = pd.DataFrame(out).sort_values("naive_mean", ascending=False); A.to_csv(DATA / ("design_effects_selection_model_quick.csv" if QUICK else "design_effects_selection_model.csv"), index=False)
json.dump(dict(omega_grid=OM.tolist(), pooled_loglik_rel=(ll - ll_max).tolist(), omega_hat=float(om_hat), omega_ci_grid=[float(ci_om.min()), float(ci_om.max())], n_contrasts=len(C)),
          open(DATA / ("design_effects_selection_pooled_quick.json" if QUICK else "design_effects_selection_pooled.json"), "w"), indent=1)
pd.set_option("display.width", 250); log("\n" + A[["factor", "level", "papers", "frac_positive", "naive_mean", "naive_pred_lo", "naive_pred_hi", "mu_omega0.5", "mu_omega0.5_lo", "mu_omega0.5_hi", "mu_omega0.25", "mu_omega0.1"]].round(1).to_string(index=False))
sub = d[d.groupby("paper_id").paper_id.transform("size") >= 2].reset_index(drop=True)
def design(df):
    cols = {}
    for f in FACTORS:
        for lv in df[f].unique():
            if lv == d[f].value_counts().index[0]: continue
            cols[f"{f[2:]}={lv}"] = (df[f] == lv).astype(float).values
    cols["logt"] = df.logt.values
    for p in d.isos.value_counts().index[1:6]: cols[f"isos={p}"] = (df.isos == p).astype(float).values
    return pd.DataFrame(cols)
def fe_beta(df):
    X = design(df); y = df.y.values; g = df.paper_id.values
    Xd = X.values - pd.DataFrame(X.values).groupby(g).transform("mean").values; yd = y - pd.Series(y).groupby(g).transform("mean").values
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=1e-8); return pd.Series(beta, index=X.columns), Xd
beta0, Xd0 = fe_beta(sub)
informative = {c: int((pd.DataFrame(Xd0[:, [list(beta0.index).index(c)]]).abs().sum(axis=1).groupby(sub.paper_id.values).max() > 1e-9).sum()) for c in beta0.index}
levels = {}
for f in FACTORS:
    vc = d[f].value_counts(); pc = d.groupby(f).paper_id.nunique()
    levels[f[2:]] = [lv for lv in vc.index if lv not in UNK and vc[lv] >= 50 and pc[lv] >= 15]
refmap = {f[2:]: d[f].value_counts().index[0] for f in FACTORS}
def coef(b, f, lv): return 0.0 if lv == refmap[f] else float(b.get(f"{f}={lv}", np.nan))
tested = {}
for r in P.itertuples():
    tested.setdefault((r.factor, frozenset((r.la, r.lb))), set()).add(r.paper)
nbB = 40 if QUICK else 300; pids = sub.paper_id.unique(); groups = {p: g for p, g in sub.groupby("paper_id")}; rngB = np.random.default_rng(1)
boot = []
for b in range(nbB):
    pick = rngB.choice(len(pids), len(pids)); parts = []
    for j, pi in enumerate(pick): gg = groups[pids[pi]].copy(); gg["paper_id"] = f"b{j}"; parts.append(gg)
    boot.append(fe_beta(pd.concat(parts, ignore_index=True))[0])
log(f"bootstrap done ({nbB} reps)")
rowsB = []
for f, lvls in levels.items():
    for a, b_ in itertools.combinations(lvls, 2):
        est = 100 * (coef(beta0, f, b_) - coef(beta0, f, a))
        bs = np.array([100 * (coef(bb, f, b_) - coef(bb, f, a)) for bb in boot])
        info = lambda lv: 99 if lv == refmap[f] else informative.get(f"{f}={lv}", 0)
        rowsB.append(dict(factor=f, level_a=a, level_b=b_, cells_a=int((d["F_" + f] == a).sum()), cells_b=int((d["F_" + f] == b_).sum()), papers_a=int(d.loc[d["F_" + f] == a, "paper_id"].nunique()), papers_b=int(d.loc[d["F_" + f] == b_, "paper_id"].nunique()),
                          informative_papers_min=int(min(info(a), info(b_))), pred_effect_pp=est, ci_lo=float(np.nanpercentile(bs, 2.5)), ci_hi=float(np.nanpercentile(bs, 97.5)),
                          direct_papers=len(tested.get((f, frozenset((a, b_))), set())), **dict(zip(["papers_any_pair", "co_change_share", "co_changed_factor"], entangle(f, a, b_)))))
M = pd.DataFrame(rowsB); M["ci_width"] = M.ci_hi - M.ci_lo; M["ci_excludes_0"] = (M.ci_lo > 0) | (M.ci_hi < 0)
M["status"] = np.where((M.direct_papers == 0) & (M.papers_any_pair == 0), "never co-occur in one paper", np.where((M.direct_papers == 0) & (M.co_changed_factor != ""), "entangled with another factor", np.where(M.direct_papers == 0, "co-occur only with several changes", np.where(M.direct_papers < 3, "1-2 papers", ">=3 papers"))))
M.sort_values(["status", "ci_width"]).to_csv(DATA / ("untested_contrast_map_quick.csv" if QUICK else "untested_contrast_map.csv"), index=False)
log("level pairs by status:\n" + M.status.value_counts().to_string())
chk = M[M.direct_papers >= 3].copy()
dm = {(r.factor, frozenset((r.la, r.lb))): [] for r in P.itertuples()}
direct = []
for r in chk.itertuples():
    g = P[(P.factor == r.factor) & (P.la.isin([r.level_a, r.level_b])) & (P.lb.isin([r.level_a, r.level_b]))]
    v = np.where(g.lb == r.level_b, g.yb - g.ya, g.ya - g.yb) * 100; direct.append(g.assign(v=v).groupby("paper").v.mean().mean())
chk["direct_mean_pp"] = direct
if len(chk) > 2:
    rs = stats.spearmanr(chk.pred_effect_pp, chk.direct_mean_pp); log(f"agreement of fixed-effects prediction with direct matched means ({len(chk)} pairs): Spearman {rs.statistic:.2f} (p={rs.pvalue:.3f}); sign agreement {(np.sign(chk.pred_effect_pp) == np.sign(chk.direct_mean_pp)).mean():.2f}")
    chk.to_csv(DATA / ("untested_contrast_map_check_quick.csv" if QUICK else "untested_contrast_map_check.csv"), index=False)
pd.set_option("display.max_colwidth", 30)
u = M[(M.direct_papers == 0)].sort_values(["status", "ci_width"]); log("pairs with no single-factor comparison:\n" + u.head(40)[["factor", "level_a", "level_b", "status", "papers_any_pair", "co_changed_factor", "pred_effect_pp", "ci_lo", "ci_hi"]].round(1).to_string(index=False))
log("DONE")
