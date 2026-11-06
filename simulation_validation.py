import json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize
D = Path(__file__).resolve().parent / "data"; T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
rng = np.random.default_rng(0); out = {}
def concordance(score, L, U, window=None, texp=None, npairs=300000, rng=rng):
    n = len(score); i = rng.integers(0, n, npairs * 4); j = rng.integers(0, n, npairs * 4); keep = (i != j) & ((U[i] < L[j]) | (U[j] < L[i]))
    if window is not None: keep &= np.abs(np.log10(texp[i]) - np.log10(texp[j])) < window
    i, j = i[keep][:npairs], j[keep][:npairs]; fi = U[i] < L[j]; c = np.where(fi, score[i] < score[j], score[j] < score[i]).astype(float); c = np.where(score[i] == score[j], 0.5, c); return float(c.mean()), int(len(i))
simA = []
for b_inf in [0.0, 0.5, 1.0]:
    for rho in [0.0, 0.3, 0.6]:
        N = 6000; lT = rng.normal(2.3, 0.9, N); x = rho * (lT - 2.3) / 0.9 + np.sqrt(1 - rho ** 2) * rng.normal(size=N)
        ltexp = 2.3 + b_inf * (lT - 2.3) + rng.normal(0, 0.55, N); texp = 10 ** ltexp; T = 10 ** lT; ev = T <= texp
        L = np.where(ev, T, texp); U = np.where(ev, T, np.inf)
        full_L = U_full = T
        truth = concordance(x, T, T)[0]
        row = dict(b_informative=b_inf, rho=rho, truth_all_pairs=truth, event_share=float(ev.mean()))
        for nm, sc in [("model score", x), ("exposure only", ltexp), ("random", rng.normal(size=N))]:
            row[nm + " | all comparable pairs"] = concordance(sc, L, U)[0]
            for w in [0.15, 0.05]: row[nm + f" | matched {w}"] = concordance(sc, L, U, w, texp)[0]
        simA.append(row)
        log(f"A b={b_inf} rho={rho}: truth {truth:.3f}; model all {row['model score | all comparable pairs']:.3f} matched.15 {row['model score | matched 0.15']:.3f}; exposure-only all {row['exposure only | all comparable pairs']:.3f} matched.15 {row['exposure only | matched 0.15']:.3f} matched.05 {row['exposure only | matched 0.05']:.3f}")
out["matched_concordance"] = simA
def nll(par, y, w):
    mu, ls = par; s = np.exp(ls); z = (0 - mu) / s; A = w * stats.norm.cdf(z) + (1 - stats.norm.cdf(z))
    return -(stats.norm.logpdf(y, mu, s) + (y < 0) * np.log(w) - np.log(np.clip(A, 1e-12, None))).sum()
def fit(y, w):
    best = None
    for mu0 in (y.mean(), y.mean() + 10):
        r = optimize.minimize(nll, [mu0, np.log(max(y.std(), 2.0))], args=(y, w), method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-5, maxiter=1000))
        if best is None or r.fun < best.fun: best = r
    return best.x[0], float(np.exp(best.x[1])), best.fun
OM = np.array([1, .75, .5, .35, .25, .15, .1, .05]); SM = pd.read_csv(D / "design_effects_selection_model.csv"); K = SM.papers.values.astype(int); SD = SM.naive_sd.values
def simulate(scenario, mu_true, w_true):
    ys = []
    for k, sd in zip(K, SD):
        mu_c = mu_true + rng.normal(0, 8)
        got = []
        while len(got) < k:
            if scenario == "normal": e = rng.normal(mu_c, sd)
            elif scenario == "heavy_tail": e = mu_c + sd * rng.standard_t(3) / np.sqrt(3)
            elif scenario == "design_selection": e = abs(rng.normal(mu_c * 0.6, sd * 0.9)) * (1 if rng.random() < 0.9 else -1) * 0.8 + mu_c * 0.3
            if e < 0 and rng.random() > w_true: continue
            got.append(e)
        ys.append(np.array(got))
    return ys
def profile(ys):
    tot = np.array([sum(fit(y, w)[2] for y in ys) for w in OM]); ll = -tot; return OM[ll.argmax()], float(2 * (ll.max() - ll[0])), ll
simB = []
NS = 150
for scen, mu_true, w_true in [("normal", 15, 1.0), ("normal", 5, 1.0), ("normal", 15, 0.15), ("normal", 5, 0.15), ("heavy_tail", 15, 1.0), ("design_selection", 15, 1.0)]:
    oh, lr, rej, mu_hat_naive, mu_hat_adj = [], [], 0, [], []
    for s in range(NS):
        ys = simulate(scen, mu_true, w_true); om, lrv, ll = profile(ys); oh.append(om); lr.append(lrv); rej += lrv > 3.84
        mu_hat_naive.append(np.mean([y.mean() for y in ys])); mu_hat_adj.append(np.mean([fit(y, 0.5)[0] for y in ys]))
    row = dict(scenario=scen, mu_true=mu_true, omega_true=w_true, omega_hat_median=float(np.median(oh)), omega_hat_p10=float(np.percentile(oh, 10)), omega_hat_p90=float(np.percentile(oh, 90)), share_rejecting_no_selection=float(rej / NS), naive_mean_bias=float(np.mean(mu_hat_naive) - mu_true), omega05_adjusted_bias=float(np.mean(mu_hat_adj) - mu_true))
    simB.append(row); log(f"B {scen} mu={mu_true} omega_true={w_true}: omega_hat median {row['omega_hat_median']} [{row['omega_hat_p10']},{row['omega_hat_p90']}], rejects omega=1 in {100*rej/NS:.0f}%, naive bias {row['naive_mean_bias']:+.1f}, omega=0.5 adjusted bias {row['omega05_adjusted_bias']:+.1f}")
out["selection_model"] = simB
json.dump(out, open(D / "simulation_validation.json", "w"), indent=1); log("DONE")
