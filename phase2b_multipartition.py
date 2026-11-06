import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

RNG_SEED = 20261029
N_PARTITIONS = 40
GBM_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, random_state=0)

df = pd.read_parquet("data/pdp_featurized.parquet")
magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
X_all = df[magpie_cols].values
y_all = df["PCE"].values
families = df["family_key"].values
uniq_families, fam_counts = np.unique(families, return_counts=True)
fam_size = dict(zip(uniq_families, fam_counts))
n_total = len(df)
log(f"Loaded {n_total} rows, {len(magpie_cols)} features, {len(uniq_families)} families")
log(f"Family sizes: {dict(sorted(fam_size.items(), key=lambda x:-x[1]))}")

rng = np.random.default_rng(RNG_SEED)
results = []
attempts = 0
while len(results) < N_PARTITIONS and attempts < N_PARTITIONS * 10:
    attempts += 1
    k = rng.integers(3, 16)
    held_out = rng.choice(uniq_families, size=k, replace=False)
    test_mask = np.isin(families, held_out)
    n_test = test_mask.sum()
    test_frac = n_test / n_total
    if not (0.05 <= test_frac <= 0.60):
        continue
    train_mask = ~test_mask

    X_train_fd, X_test_fd = X_all[train_mask], X_all[test_mask]
    y_train_fd, y_test_fd = y_all[train_mask], y_all[test_mask]

    model_fd = GradientBoostingRegressor(**GBM_PARAMS)
    model_fd.fit(X_train_fd, y_train_fd)
    pred_fd = model_fd.predict(X_test_fd)
    r2_fd = r2_score(y_test_fd, pred_fd)
    mae_fd = mean_absolute_error(y_test_fd, pred_fd)

    perm = rng.permutation(n_total)
    test_idx_rand = perm[:n_test]
    train_idx_rand = perm[n_test:]
    X_train_r, X_test_r = X_all[train_idx_rand], X_all[test_idx_rand]
    y_train_r, y_test_r = y_all[train_idx_rand], y_all[test_idx_rand]

    model_r = GradientBoostingRegressor(**GBM_PARAMS)
    model_r.fit(X_train_r, y_train_r)
    pred_r = model_r.predict(X_test_r)
    r2_r = r2_score(y_test_r, pred_r)
    mae_r = mean_absolute_error(y_test_r, pred_r)

    dominant_family = held_out[np.argmax([fam_size[f] for f in held_out])]
    dominant_share = fam_size[dominant_family] / n_test

    row = dict(
        partition_id=len(results),
        k_families_held_out=int(k),
        held_out_families=held_out.tolist(),
        n_test=int(n_test),
        test_frac=float(test_frac),
        dominant_family=str(dominant_family),
        dominant_family_share_of_test=float(dominant_share),
        family_disjoint_r2=float(r2_fd),
        family_disjoint_mae=float(mae_fd),
        random_r2=float(r2_r),
        random_mae=float(mae_r),
        fd_beats_random_r2=bool(r2_fd > r2_r),
        mae_diff_random_minus_fd=float(mae_r - mae_fd),
    )
    results.append(row)
    log(f"partition {row['partition_id']:2d}: k={k:2d} n_test={n_test:5d} "
        f"dom_share={dominant_share:.2f} FD_R2={r2_fd:.3f} RAND_R2={r2_r:.3f} "
        f"beats={row['fd_beats_random_r2']}")

log(f"Done {len(results)} partitions ({attempts} attempts)")

r2_diffs = np.array([r["family_disjoint_r2"] - r["random_r2"] for r in results])
mae_diffs = np.array([r["mae_diff_random_minus_fd"] for r in results])
dom_shares = np.array([r["dominant_family_share_of_test"] for r in results])
frac_fd_wins = float(np.mean(r2_diffs > 0))

from scipy import stats
t_stat, t_p = stats.ttest_1samp(r2_diffs, 0.0)
w_stat, w_p = stats.wilcoxon(r2_diffs)

corr_dom_vs_advantage = float(np.corrcoef(dom_shares, r2_diffs)[0, 1])
corr_testfrac_vs_advantage = float(np.corrcoef(
    [r["test_frac"] for r in results], r2_diffs)[0, 1])

summary = dict(
    n_partitions=len(results),
    frac_partitions_fd_beats_random_by_r2=frac_fd_wins,
    mean_r2_diff_fd_minus_random=float(np.mean(r2_diffs)),
    std_r2_diff=float(np.std(r2_diffs)),
    paired_ttest_stat=float(t_stat),
    paired_ttest_pvalue=float(t_p),
    wilcoxon_stat=float(w_stat),
    wilcoxon_pvalue=float(w_p),
    mean_mae_diff_random_minus_fd=float(np.mean(mae_diffs)),
    corr_dominant_family_share_vs_r2_advantage=corr_dom_vs_advantage,
    corr_test_frac_vs_r2_advantage=corr_testfrac_vs_advantage,
)
log(f"SUMMARY: {json.dumps(summary, indent=2)}")

with open("data/multipartition_robustness.json", "w") as f:
    json.dump(dict(summary=summary, partitions=results), f, indent=2)
log("Saved data/multipartition_robustness.json")
