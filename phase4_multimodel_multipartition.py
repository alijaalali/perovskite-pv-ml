import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

df = pd.read_parquet("data/pdp_featurized.parquet")
magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
X_all = df[magpie_cols].values
y_all = df["PCE"].values
families = df["family_key"].values
uniq_families, fam_counts = np.unique(families, return_counts=True)
fam_size = dict(zip(uniq_families, fam_counts))
n_total = len(df)
log(f"Loaded {n_total} rows, {len(magpie_cols)} features")

with open("data/multipartition_robustness.json") as f:
    existing = json.load(f)
partitions_meta = existing["partitions"]
log(f"Loaded {len(partitions_meta)} existing partitions (GBM results reused, not refit)")

MODEL_BUILDERS = {
    "Ridge": lambda: Ridge(alpha=1.0, random_state=0),
    "XGBoost": lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, random_state=0, verbosity=0),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, max_depth=12, random_state=0, n_jobs=4),
    "ElasticNet": lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=0, max_iter=5000),
}

all_results = {name: [] for name in MODEL_BUILDERS}

for pmeta in partitions_meta:
    pid = pmeta["partition_id"]
    held_out = np.array(pmeta["held_out_families"])
    test_mask = np.isin(families, held_out)
    train_mask = ~test_mask
    n_test = int(test_mask.sum())
    assert n_test == pmeta["n_test"], f"partition {pid} n_test mismatch: {n_test} vs {pmeta['n_test']}"

    rng = np.random.default_rng(90000 + pid)
    perm = rng.permutation(n_total)
    test_idx_rand = perm[:n_test]
    train_idx_rand = perm[n_test:]

    X_train_fd, X_test_fd = X_all[train_mask], X_all[test_mask]
    y_train_fd, y_test_fd = y_all[train_mask], y_all[test_mask]
    X_train_r, X_test_r = X_all[train_idx_rand], X_all[test_idx_rand]
    y_train_r, y_test_r = y_all[train_idx_rand], y_all[test_idx_rand]

    for name, build in MODEL_BUILDERS.items():
        m_fd = build()
        m_fd.fit(X_train_fd, y_train_fd)
        pred_fd = m_fd.predict(X_test_fd)
        r2_fd = r2_score(y_test_fd, pred_fd)
        mae_fd = mean_absolute_error(y_test_fd, pred_fd)

        m_r = build()
        m_r.fit(X_train_r, y_train_r)
        pred_r = m_r.predict(X_test_r)
        r2_r = r2_score(y_test_r, pred_r)
        mae_r = mean_absolute_error(y_test_r, pred_r)

        all_results[name].append(dict(
            partition_id=pid,
            dominant_family_share_of_test=pmeta["dominant_family_share_of_test"],
            test_frac=pmeta["test_frac"],
            family_disjoint_r2=float(r2_fd),
            family_disjoint_mae=float(mae_fd),
            random_r2=float(r2_r),
            random_mae=float(mae_r),
            fd_beats_random_r2=bool(r2_fd > r2_r),
        ))
    log(f"partition {pid:2d} done: " +
        " | ".join(f"{n}:FD={all_results[n][-1]['family_disjoint_r2']:.3f}" for n in MODEL_BUILDERS))

summaries = {}
for name, rows in all_results.items():
    r2_diffs = np.array([r["family_disjoint_r2"] - r["random_r2"] for r in rows])
    dom_shares = np.array([r["dominant_family_share_of_test"] for r in rows])
    test_fracs = np.array([r["test_frac"] for r in rows])
    frac_wins = float(np.mean(r2_diffs > 0))
    t_stat, t_p = stats.ttest_1samp(r2_diffs, 0.0)
    w_stat, w_p = stats.wilcoxon(r2_diffs)
    summaries[name] = dict(
        n_partitions=len(rows),
        frac_partitions_fd_beats_random_by_r2=frac_wins,
        mean_r2_diff_fd_minus_random=float(np.mean(r2_diffs)),
        median_r2_diff=float(np.median(r2_diffs)),
        min_r2_diff=float(np.min(r2_diffs)),
        max_r2_diff=float(np.max(r2_diffs)),
        std_r2_diff=float(np.std(r2_diffs)),
        paired_ttest_pvalue=float(t_p),
        wilcoxon_pvalue=float(w_p),
        corr_dominant_family_share_vs_r2_advantage=float(np.corrcoef(dom_shares, r2_diffs)[0, 1]),
        corr_test_frac_vs_r2_advantage=float(np.corrcoef(test_fracs, r2_diffs)[0, 1]),
        frac_partitions_family_disjoint_r2_negative=float(np.mean(
            [r["family_disjoint_r2"] < 0 for r in rows])),
        mean_family_disjoint_r2=float(np.mean([r["family_disjoint_r2"] for r in rows])),
        min_family_disjoint_r2=float(np.min([r["family_disjoint_r2"] for r in rows])),
    )
    log(f"SUMMARY {name}: frac_wins={frac_wins:.3f} mean_diff={summaries[name]['mean_r2_diff_fd_minus_random']:.3f} "
        f"p_ttest={t_p:.3f} p_wilcoxon={w_p:.3f} frac_negative_R2={summaries[name]['frac_partitions_family_disjoint_r2_negative']:.3f}")

gbm_r2_diffs = np.array([r["family_disjoint_r2"] - r["random_r2"] for r in partitions_meta])
summaries["GBM"] = dict(
    n_partitions=len(partitions_meta),
    frac_partitions_fd_beats_random_by_r2=existing["summary"]["frac_partitions_fd_beats_random_by_r2"],
    mean_r2_diff_fd_minus_random=existing["summary"]["mean_r2_diff_fd_minus_random"],
    std_r2_diff=existing["summary"]["std_r2_diff"],
    paired_ttest_pvalue=existing["summary"]["paired_ttest_pvalue"],
    wilcoxon_pvalue=existing["summary"]["wilcoxon_pvalue"],
    corr_dominant_family_share_vs_r2_advantage=existing["summary"]["corr_dominant_family_share_vs_r2_advantage"],
    corr_test_frac_vs_r2_advantage=existing["summary"]["corr_test_frac_vs_r2_advantage"],
    frac_partitions_family_disjoint_r2_negative=float(np.mean(
        [r["family_disjoint_r2"] < 0 for r in partitions_meta])),
    mean_family_disjoint_r2=float(np.mean([r["family_disjoint_r2"] for r in partitions_meta])),
    min_family_disjoint_r2=float(np.min([r["family_disjoint_r2"] for r in partitions_meta])),
    note="reused from data/multipartition_robustness.json (computed by phase2b_multipartition.py)",
)

with open("data/multimodel_multipartition_robustness.json", "w") as f:
    json.dump(dict(summaries=summaries, per_model_partitions=all_results), f, indent=2)
log("Saved data/multimodel_multipartition_robustness.json")
log("DONE")
