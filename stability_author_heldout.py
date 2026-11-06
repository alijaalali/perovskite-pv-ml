import json, sys
src = open(__import__("pathlib").Path(__file__).resolve().parent / "stability_enriched_features.py").read()
exec(compile(src[:src.index("MODELS = [")], "enriched_head", "exec"))
import scipy.sparse as sp_, scipy.sparse.csgraph as csg
AU = json.load(open(DATA / "paper_authors_openalex.json"))
doi_of = {p: p.lower() for p in papers_u}
def auth_list(p):
    a = AU.get(doi_of[p]); return [x["aid"] for x in a if x.get("aid")] if a else []
AL = {p: auth_list(p) for p in papers_u}
log("papers with author data: %d of %d" % (sum(bool(v) for v in AL.values()), len(AL)))
def comps(cap):
    from collections import Counter
    c = Counter(a for v in AL.values() for a in set(v)); ids = {}; rows = []; cols = []
    for pi, p in enumerate(papers_u):
        for a in set(AL[p]):
            if c[a] <= cap: rows.append(pi); cols.append(ids.setdefault(a, len(ids)))
    M = sp_.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(papers_u), max(len(ids), 1)))
    n, lab = csg.connected_components(M @ M.T, directed=False); return lab
GRP = {}
GRP["last_author"] = np.array([hash(AL[p][-1]) % (2**31) if AL[p] else hash(p) % (2**31) for p in papers_u])
GRP["first_author"] = np.array([hash(AL[p][0]) % (2**31) if AL[p] else hash(p) % (2**31) for p in papers_u])
for k, v in GRP.items():
    u, c = np.unique(v, return_counts=True); cells = np.bincount(np.searchsorted(u, v)[pidx]); log(f"group {k}: {len(u)} groups; largest {c.max()} papers ({100*cells.max()/len(d):.1f}% of cells)")
def split_author(key):
    if key == "purged_any_author":
        g = GRP["last_author"][pidx]; out = []; A_of = [set(AL[p]) for p in papers_u]
        for tr, te in GroupKFold(5).split(np.arange(len(d)), groups=g):
            tp = np.unique(pidx[te]); bad = set().union(*[A_of[i] for i in tp]); ok = np.array([len(A_of[i] & bad) == 0 for i in range(len(papers_u))])
            tr2 = tr[ok[pidx[tr]]]; out.append((tr2, te)); log(f"   purge: train {len(tr)} -> {len(tr2)} cells ({100*len(tr2)/len(tr):.0f}%)")
        return out
    g = GRP[key][pidx]; return list(GroupKFold(5).split(np.arange(len(d)), groups=g))
def noflag_cols(names):
    return [i for i, n in enumerate(names) if not (n.endswith("_miss") or n.endswith("_known") or "_Unknown" in n or n.endswith("_other") or "_nan" in n or n.endswith("_NA"))]
SETX = {"base": BASE, "+all enriched": BASE + ["proc", "stack", "pcefresh"], "+process+stack (no fresh PCE)": BASE + ["proc", "stack"], "+process": BASE + ["proc"]}
def Xn(name, noflags=False):
    parts = [F[k] for k in SETX[name]]; df = pd.concat(parts, axis=1)
    if noflags: df = df.iloc[:, noflag_cols(list(df.columns))]
    return df.values.astype(float)
MODELS = ["Ridge", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
PROT = ["paper", "last_author", "first_author", "purged_any_author"]
if QUICK: MODELS, PROT = ["Ridge", "XGBoost"], ["paper", "last_author"]
results = []; w1 = np.ones(len(d))
for proto in PROT:
    sp = splits("paper") if proto == "paper" else split_author(proto)
    log(f"== {proto}: fold test sizes {[len(te) for _, te in sp]}")
    for mname in MODELS:
        oo = {}
        for fs in SETX:
            for nf in (False, True):
                if nf and fs == "base" and False: continue
                X = Xn(fs, nf); pred = np.full(len(d), np.nan)
                for tr, te in sp: pred[te] = fit_pred(mk(mname) if mname in ("CatBoost",) else make_reg(mname), X[tr], y_reg[tr], X[te]) if True else None
                oo[(fs, nf)] = pred; m = ~np.isnan(pred)
                results.append(dict(protocol=proto, model=mname, features=fs, noflags=nf, n_features=int(X.shape[1]), r2=float(wr2(y_reg[m], pred[m], w1[m])), r2_ci=ci([wr2(y_reg[m], pred[m], bw[pidx][m]) for bw in BW])))
        for fs in SETX:
            for nf in (False, True):
                if (fs, nf) == ("base", False): continue
                pb = oo[("base", nf)] if nf else oo[("base", False)]; pe = oo[(fs, nf)]; m = ~np.isnan(pb) & ~np.isnan(pe)
                dl = [wr2(y_reg[m], pe[m], bw[pidx][m]) - wr2(y_reg[m], pb[m], bw[pidx][m]) for bw in BW]
                results.append(dict(protocol=proto, model=mname, features=fs, noflags=nf, task="delta", delta_r2=float(wr2(y_reg[m], pe[m], w1[m]) - wr2(y_reg[m], pb[m], w1[m])), delta_ci=ci(dl)))
        log(f"  {proto} {mname}: " + ", ".join(f"{k[0]}{'/nf' if k[1] else ''}={r['r2']:.3f}" for k, r in [(k, next(x for x in results if x['protocol'] == proto and x['model'] == mname and x['features'] == k[0] and x['noflags'] == k[1] and 'r2' in x and 'task' not in x)) for k in oo]))
        json.dump(results, open(RES / ("stability_author_heldout" + ("_quick" if QUICK else "") + ".json"), "w"))
log("done")
