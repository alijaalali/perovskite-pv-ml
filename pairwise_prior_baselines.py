import json
src = open(__import__("pathlib").Path(__file__).resolve().parent / "within_paper_contrasts.py").read()
exec(compile(src[:src.index("res = {}")], "contrast_head", "exec"))
from sklearn.model_selection import GroupKFold
treated = ((d.F_encapsulation == "True").astype(int) + (d.F_additive == "additive").astype(int) + d.F_A_site.astype(str).str.contains(";").astype(int) + (d.F_B_site == "lead_mixed").astype(int) + d.F_deposition.astype(str).str.contains(">>").astype(int)).values
rec = []; Fv = d[FACTORS].values
for pid, g in d.groupby("paper_id", sort=False):
    ix = g.index.values
    if len(ix) < 2: continue
    iu, ju = np.triu_indices(len(ix), 1)
    for a, b in zip(ix[iu], ix[ju]):
        if d.isos.values[a] != d.isos.values[b] or abs(d.logt.values[a] - d.logt.values[b]) >= 0.1 or abs(d.y.values[a] - d.y.values[b]) < 0.01: continue
        diff = [k for k in range(len(FACTORS)) if Fv[a, k] != Fv[b, k]]
        rec.append((pid, a, b, len(diff), FACTORS[diff[0]] if len(diff) == 1 else "", Fv[a, diff[0]] if len(diff) == 1 else "", Fv[b, diff[0]] if len(diff) == 1 else "", d.y.values[b] - d.y.values[a], treated[b] - treated[a]))
P = pd.DataFrame(rec, columns=["paper", "a", "b", "nd", "factor", "la", "lb", "dy", "dtreat"]); P["nd_c"] = P.nd.clip(upper=2)
log(f"pairs {len(P)}; coded factors differing 0/1/2+: {(P.nd==0).sum()}/{(P.nd==1).sum()}/{(P.nd>=2).sum()}")
pap = P.paper.values; up, pi_ = np.unique(pap, return_inverse=True); rng = np.random.default_rng(0)
score_prior = np.zeros(len(P))
for tr, te in GroupKFold(5).split(P, groups=pap):
    T = P.iloc[tr]; T = T[(T.nd == 1) & ~T.la.isin(UNK) & ~T.lb.isin(UNK)]; k = {}
    for r in T.itertuples(): k.setdefault((r.factor, r.la, r.lb), {}).setdefault(r.paper, []).append(r.dy)
    prior = {kk: np.sign(np.mean([np.mean(v) for v in vv.values()])) for kk, vv in k.items()}
    for kk in list(prior): prior.setdefault((kk[0], kk[2], kk[1]), -prior[kk])
    for i in te:
        r = P.iloc[i]
        if r.nd == 1: score_prior[i] = prior.get((r.factor, r.la, r.lb), 0.0)
P["prior"] = score_prior; P["more_treated"] = np.sign(P.dtreat.values)
def acc(score, mask):
    m = mask & (P.dy.values != 0); s = np.sign(score[m]); hit = np.where(s == 0, 0.5, (s == np.sign(P.dy.values[m])).astype(float))
    out = []
    for _ in range(300):
        bi = rng.integers(0, len(up), len(up)); w = np.bincount(bi, minlength=len(up))[pi_[m]]; out.append(float(np.sum(w * hit) / np.sum(w)))
    return dict(acc=float(hit.mean()), lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)), n=int(m.sum()))
res = {}
for nm, sub in [("all pairs", P.nd >= 0), ("one coded factor differs", P.nd == 1), ("no coded factor differs", P.nd == 0)]:
    for rule in ["prior", "more_treated"]: res[f"{rule} | {nm}"] = acc(P[rule].values, sub.values); log(f"{rule:13s} {nm:26s} {res[f'{rule} | {nm}']}")
prop = P[P.nd == 1].dy.gt(0).mean(); res["share of one-factor pairs in which cell b has the higher retention (coin flip check)"] = float(prop)
json.dump(res, open(DATA / "pairwise_prior_baselines.json", "w"), indent=1); log("DONE")
