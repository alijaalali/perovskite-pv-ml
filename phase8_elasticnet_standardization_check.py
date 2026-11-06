import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
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
n_total = len(df)
log(f"Loaded {n_total} rows, {len(magpie_cols)} features")

with open("data/multipartition_robustness.json") as f:
    existing = json.load(f)
partitions_meta = existing["partitions"]
log(f"Loaded {len(partitions_meta)} existing partitions")

ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

VARIANTS = ["unstandardized_original", "standardized_original_hparams", "standardized_cv_tuned"]
all_results = {v: [] for v in VARIANTS}

for pmeta in partitions_meta:
    pid = pmeta["partition_id"]
    held_out = np.array(pmeta["held_out_families"])
    test_mask = np.isin(families, held_out)
    train_mask = ~test_mask
    n_test = int(test_mask.sum())
    assert n_test == pmeta["n_test"], f"partition {pid} n_test mismatch"

    rng = np.random.default_rng(90000 + pid)
    perm = rng.permutation(n_total)
    test_idx_rand = perm[:n_test]
    train_idx_rand = perm[n_test:]

    X_train_fd, X_test_fd = X_all[train_mask], X_all[test_mask]
    y_train_fd, y_test_fd = y_all[train_mask], y_all[test_mask]
    X_train_r, X_test_r = X_all[train_idx_rand], X_all[test_idx_rand]
    y_train_r, y_test_r = y_all[train_idx_rand], y_all[test_idx_rand]

    def fit_eval(Xtr, ytr, Xte, yte, variant):
        if variant == "unstandardized_original":
            m = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=0, max_iter=5000)
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            best_alpha, best_l1 = 0.01, 0.5
        elif variant == "standardized_original_hparams":
            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            m = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=0, max_iter=5000)
            m.fit(Xtr_s, ytr)
            pred = m.predict(Xte_s)
            best_alpha, best_l1 = 0.01, 0.5
        elif variant == "standardized_cv_tuned":
            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            m = ElasticNetCV(alphas=ALPHA_GRID, l1_ratio=L1_RATIO_GRID, cv=5,
                              max_iter=5000, random_state=0, n_jobs=4)
            m.fit(Xtr_s, ytr)
            pred = m.predict(Xte_s)
            best_alpha, best_l1 = float(m.alpha_), float(m.l1_ratio_)
        r2 = r2_score(yte, pred)
        mae = mean_absolute_error(yte, pred)
        return float(r2), float(mae), best_alpha, best_l1

    for variant in VARIANTS:
        r2_fd, mae_fd, a_fd, l1_fd = fit_eval(X_train_fd, y_train_fd, X_test_fd, y_test_fd, variant)
        r2_r, mae_r, a_r, l1_r = fit_eval(X_train_r, y_train_r, X_test_r, y_test_r, variant)
        all_results[variant].append(dict(
            partition_id=pid,
            dominant_family_share_of_test=pmeta["dominant_family_share_of_test"],
            test_frac=pmeta["test_frac"],
            family_disjoint_r2=r2_fd, family_disjoint_mae=mae_fd,
            chosen_alpha_fd=a_fd, chosen_l1_ratio_fd=l1_fd,
            random_r2=r2_r, random_mae=mae_r,
            chosen_alpha_random=a_r, chosen_l1_ratio_random=l1_r,
            fd_beats_random_r2=bool(r2_fd > r2_r),
        ))
    log(f"partition {pid:2d} done: " +
        " | ".join(f"{v}:FD={all_results[v][-1]['family_disjoint_r2']:.3f}" for v in VARIANTS))

orig_r2s = [r["family_disjoint_r2"] for r in all_results["unstandardized_original"]]
log(f"Check: unstandardized variant mean FD R2 = {np.mean(orig_r2s):.4f} "
    f"(expect to match multimodel_multipartition_robustness.json's ElasticNet mean_family_disjoint_r2 ~= 0.139)")

summaries = {}
for variant, rows in all_results.items():
    r2_diffs = np.array([r["family_disjoint_r2"] - r["random_r2"] for r in rows])
    fd_r2s = np.array([r["family_disjoint_r2"] for r in rows])
    frac_wins = float(np.mean(r2_diffs > 0))
    t_stat, t_p = stats.ttest_1samp(r2_diffs, 0.0)
    w_stat, w_p = stats.wilcoxon(r2_diffs)
    summaries[variant] = dict(
        n_partitions=len(rows),
        frac_partitions_fd_beats_random_by_r2=frac_wins,
        mean_r2_diff_fd_minus_random=float(np.mean(r2_diffs)),
        median_r2_diff=float(np.median(r2_diffs)),
        min_r2_diff=float(np.min(r2_diffs)),
        max_r2_diff=float(np.max(r2_diffs)),
        std_r2_diff=float(np.std(r2_diffs)),
        paired_ttest_pvalue=float(t_p),
        wilcoxon_pvalue=float(w_p),
        frac_partitions_family_disjoint_r2_negative=float(np.mean(fd_r2s < 0)),
        mean_family_disjoint_r2=float(np.mean(fd_r2s)),
        median_family_disjoint_r2=float(np.median(fd_r2s)),
        min_family_disjoint_r2=float(np.min(fd_r2s)),
        max_family_disjoint_r2=float(np.max(fd_r2s)),
    )
    log(f"SUMMARY {variant}: frac_wins={frac_wins:.3f} mean_FD_R2={summaries[variant]['mean_family_disjoint_r2']:.3f} "
        f"min_FD_R2={summaries[variant]['min_family_disjoint_r2']:.3f} "
        f"frac_negative={summaries[variant]['frac_partitions_family_disjoint_r2_negative']:.3f} "
        f"p_ttest={t_p:.4f}")

us = summaries["unstandardized_original"]
sa = summaries["standardized_original_hparams"]
sc_ = summaries["standardized_cv_tuned"]
verdict = (
    f"Original (unstandardized, alpha=0.01/l1=0.5): mean FD R2={us['mean_family_disjoint_r2']:.3f}, "
    f"min={us['min_family_disjoint_r2']:.3f}, {us['frac_partitions_family_disjoint_r2_negative']*100:.1f}% of partitions negative, "
    f"win_rate={us['frac_partitions_fd_beats_random_by_r2']*100:.1f}%. "
    f"Standardized, same fixed hparams: mean FD R2={sa['mean_family_disjoint_r2']:.3f}, "
    f"min={sa['min_family_disjoint_r2']:.3f}, {sa['frac_partitions_family_disjoint_r2_negative']*100:.1f}% negative, "
    f"win_rate={sa['frac_partitions_fd_beats_random_by_r2']*100:.1f}%. "
    f"Standardized + CV-tuned alpha/l1_ratio: mean FD R2={sc_['mean_family_disjoint_r2']:.3f}, "
    f"min={sc_['min_family_disjoint_r2']:.3f}, {sc_['frac_partitions_family_disjoint_r2_negative']*100:.1f}% negative, "
    f"win_rate={sc_['frac_partitions_fd_beats_random_by_r2']*100:.1f}%."
)
log("Verdict: " + verdict)

with open("data/elasticnet_standardization_check.json", "w") as f:
    json.dump(dict(summaries=summaries, per_variant_partitions=all_results, verdict=verdict), f, indent=2)
log("Saved data/elasticnet_standardization_check.json")
log("DONE")
