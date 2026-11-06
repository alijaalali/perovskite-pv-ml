import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

GBM_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                   subsample=0.8, random_state=0)

df = pd.read_parquet("data/pdp_featurized.parquet")
magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
log(f"Loaded {len(df)} rows, {len(magpie_cols)} features, "
    f"{df['formula_str'].nunique()} unique formulas")

dup_counts = df.groupby("formula_str").size()
log(f"Duplicate-count distribution: min={dup_counts.min()}, median={dup_counts.median()}, "
    f"mean={dup_counts.mean():.2f}, max={dup_counts.max()}")
hist = dup_counts.value_counts().sort_index()
hist_summary = {int(k): int(v) for k, v in hist.items() if k <= 20}
hist_summary["21+"] = int((dup_counts > 20).sum())
log(f"Histogram (dup-count -> n_formulas): {hist_summary}")

median_cols = ["PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF", "Perovskite_band_gap"]
for c in median_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
agg_dict = {c: "first" for c in df.columns if c not in median_cols}
for c in median_cols:
    if c in df.columns:
        agg_dict[c] = "median"

dedup = df.groupby("formula_str", as_index=False).agg(agg_dict)
log(f"Deduplicated: {len(dedup)} rows (from {len(df)})")
dedup.to_parquet("data/pdp_deduplicated.parquet")
log("Saved data/pdp_deduplicated.parquet")

def family_disjoint_vs_random(data, label, n_partitions=10, seed=20270101):
    X_all = data[magpie_cols].values
    y_all = data["PCE"].values
    families = data["family_key"].values
    uniq_fams, fam_counts = np.unique(families, return_counts=True)
    fam_size = dict(zip(uniq_fams, fam_counts))
    n_total = len(data)
    rng = np.random.default_rng(seed)
    rows = []
    attempts = 0
    while len(rows) < n_partitions and attempts < n_partitions * 15:
        attempts += 1
        k = rng.integers(3, min(16, len(uniq_fams)))
        held_out = rng.choice(uniq_fams, size=k, replace=False)
        test_mask = np.isin(families, held_out)
        n_test = int(test_mask.sum())
        test_frac = n_test / n_total
        if not (0.05 <= test_frac <= 0.60):
            continue
        train_mask = ~test_mask
        m_fd = GradientBoostingRegressor(**GBM_PARAMS)
        m_fd.fit(X_all[train_mask], y_all[train_mask])
        pred_fd = m_fd.predict(X_all[test_mask])
        r2_fd = r2_score(y_all[test_mask], pred_fd)
        mae_fd = mean_absolute_error(y_all[test_mask], pred_fd)

        perm = rng.permutation(n_total)
        test_idx_r, train_idx_r = perm[:n_test], perm[n_test:]
        m_r = GradientBoostingRegressor(**GBM_PARAMS)
        m_r.fit(X_all[train_idx_r], y_all[train_idx_r])
        pred_r = m_r.predict(X_all[test_idx_r])
        r2_r = r2_score(y_all[test_idx_r], pred_r)
        mae_r = mean_absolute_error(y_all[test_idx_r], pred_r)

        rows.append(dict(partition_id=len(rows), k=int(k), n_test=n_test,
                          test_frac=float(test_frac),
                          family_disjoint_r2=float(r2_fd), family_disjoint_mae=float(mae_fd),
                          random_r2=float(r2_r), random_mae=float(mae_r),
                          fd_beats_random=bool(r2_fd > r2_r)))
        log(f"[{label}] partition {rows[-1]['partition_id']}: FD_R2={r2_fd:.3f} RAND_R2={r2_r:.3f} "
            f"n_test={n_test}")
    r2_diffs = np.array([r["family_disjoint_r2"] - r["random_r2"] for r in rows])
    from scipy import stats
    t_stat, t_p = stats.ttest_1samp(r2_diffs, 0.0) if len(rows) > 1 else (np.nan, np.nan)
    return dict(
        label=label, n_rows=n_total, n_partitions=len(rows),
        frac_fd_beats_random=float(np.mean(r2_diffs > 0)),
        mean_r2_diff=float(np.mean(r2_diffs)),
        std_r2_diff=float(np.std(r2_diffs)),
        paired_ttest_pvalue=float(t_p),
        partitions=rows,
    )

log("Running on original (duplicated) data, 10 partitions for comparison...")
orig_result = family_disjoint_vs_random(df, "original_duplicated", n_partitions=10, seed=30000001)
log("Running on deduplicated data, 10 partitions...")
dedup_result = family_disjoint_vs_random(dedup, "deduplicated", n_partitions=10, seed=30000002)

def single_split_fit(data, label):
    X_all = data[magpie_cols].values
    y_all = data["PCE"].values
    tr_mask = (data["split_family_disjoint"] == "train").values
    te_mask = (data["split_family_disjoint"] == "test").values
    m = GradientBoostingRegressor(**GBM_PARAMS)
    m.fit(X_all[tr_mask], y_all[tr_mask])
    pred = m.predict(X_all[te_mask])
    r2 = r2_score(y_all[te_mask], pred)
    mae = mean_absolute_error(y_all[te_mask], pred)
    rng = np.random.default_rng(40000000)
    n_total = len(data)
    n_test = int(te_mask.sum())
    perm = rng.permutation(n_total)
    test_idx_r, train_idx_r = perm[:n_test], perm[n_test:]
    m2 = GradientBoostingRegressor(**GBM_PARAMS)
    m2.fit(X_all[train_idx_r], y_all[train_idx_r])
    pred_r = m2.predict(X_all[test_idx_r])
    r2_r = r2_score(y_all[test_idx_r], pred_r)
    mae_r = mean_absolute_error(y_all[test_idx_r], pred_r)
    log(f"[{label}] canonical split: FD R2={r2:.3f} MAE={mae:.3f} | random R2={r2_r:.3f} MAE={mae_r:.3f}")
    return dict(family_disjoint={"r2": float(r2), "mae": float(mae)},
                random={"r2": float(r2_r), "mae": float(mae_r)})

orig_canonical = single_split_fit(df, "original_canonical_split")
dedup_canonical = single_split_fit(dedup, "dedup_canonical_split")

out = dict(
    duplicate_histogram=hist_summary,
    duplicate_count_stats=dict(min=int(dup_counts.min()), median=float(dup_counts.median()),
                                mean=float(dup_counts.mean()), max=int(dup_counts.max())),
    n_original=len(df), n_deduplicated=len(dedup),
    canonical_split_comparison=dict(original=orig_canonical, deduplicated=dedup_canonical),
    multipartition_comparison=dict(original=orig_result, deduplicated=dedup_result),
)
with open("data/dedup_benchmark_comparison.json", "w") as f:
    json.dump(out, f, indent=2)
log("Saved data/dedup_benchmark_comparison.json")
log("DONE")
