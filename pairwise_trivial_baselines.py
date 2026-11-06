import itertools, json, numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parent / "data"
d = pd.read_parquet(D / "stability_table.parquet")
y = d.retention_end.clip(0, 1.2).values; logt = np.log10(d.t_exp_h.values); pf = pd.to_numeric(d.pce_fresh, errors="coerce").values
pc, _ = pd.factorize(d.paper_id.values); isos = d.isos.values
rng = np.random.default_rng(0); rows = []
for p, idx in pd.Series(np.arange(len(d))).groupby(pc):
    idx = idx.values
    for a, b in itertools.combinations(idx, 2):
        if isos[a] != isos[b] or abs(logt[a] - logt[b]) >= 0.1 or abs(y[a] - y[b]) < 0.01: continue
        rows.append((p, logt[a] - logt[b], pf[a] - pf[b], y[a] - y[b], np.log10(d.t_exp_h.values[a]) == np.log10(d.t_exp_h.values[b])))
r = pd.DataFrame(rows, columns=["p", "dt", "dpf", "dy", "same_t"])
print("pairs", len(r), "same exposure time exactly:", r.same_t.mean().round(3))
def acc(score, sub=None):
    m = np.isfinite(score) & (score != 0) & (r.dy != 0)
    if sub is not None: m &= sub
    return round(float(np.mean(np.sign(score[m]) == np.sign(r.dy[m]))), 3), int(m.sum())
print("shorter test lasts better (sign -dt):", acc(-r.dt.values))
print("same-time pairs only, -dt undefined; n same_t:", int(r.same_t.sum()))
print("higher fresh PCE retains more (sign dpf):", acc(r.dpf.values))
print("higher fresh PCE retains LESS (sign -dpf):", acc(-r.dpf.values))
print("fresh PCE among same-time pairs:", acc(r.dpf.values, r.same_t.values))
print("fresh PCE among different-time pairs:", acc(r.dpf.values, ~r.same_t.values))

json.dump(dict(n_pairs=int(len(r)), frac_same_exposure=float(r.same_t.mean()), shorter_test_acc=acc(-r.dt.values)[0], shorter_test_n=acc(-r.dt.values)[1], fresh_pce_acc=acc(r.dpf.values)[0], fresh_pce_n=acc(r.dpf.values)[1]), open(D / "pairwise_trivial_baselines.json", "w"), indent=1)
