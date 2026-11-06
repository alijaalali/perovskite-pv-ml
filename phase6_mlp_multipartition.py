import json
import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
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
log(f"Loaded {len(partitions_meta)} existing partitions (reused exactly, not regenerated)")

def build_mlp():
    return MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                         alpha=1e-4, max_iter=2000, early_stopping=True,
                         n_iter_no_change=20, random_state=0)

results = []
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

    sc_fd = StandardScaler().fit(X_train_fd)
    m_fd = build_mlp()
    m_fd.fit(sc_fd.transform(X_train_fd), y_train_fd)
    pred_fd = m_fd.predict(sc_fd.transform(X_test_fd))
    r2_fd = r2_score(y_test_fd, pred_fd)
    mae_fd = mean_absolute_error(y_test_fd, pred_fd)

    sc_r = StandardScaler().fit(X_train_r)
    m_r = build_mlp()
    m_r.fit(sc_r.transform(X_train_r), y_train_r)
    pred_r = m_r.predict(sc_r.transform(X_test_r))
    r2_r = r2_score(y_test_r, pred_r)
    mae_r = mean_absolute_error(y_test_r, pred_r)

    row = dict(partition_id=pid,
               dominant_family_share_of_test=pmeta["dominant_family_share_of_test"],
               test_frac=pmeta["n_test"] / n_total,
               family_disjoint_r2=float(r2_fd), family_disjoint_mae=float(mae_fd),
               random_r2=float(r2_r), random_mae=float(mae_r),
               fd_beats_random=bool(r2_fd > r2_r))
    results.append(row)
    log(f"partition {pid:2d}: FD_R2={r2_fd:7.3f} RAND_R2={r2_r:7.3f} beats={row['fd_beats_random']}")

r2_diffs = np.array([r["family_disjoint_r2"] - r["random_r2"] for r in results])
dom_shares = np.array([r["dominant_family_share_of_test"] for r in results])
test_fracs = np.array([r["test_frac"] for r in results])
t_stat, t_p = stats.ttest_1samp(r2_diffs, 0.0)
w_stat, w_p = stats.wilcoxon(r2_diffs)

summary = dict(
    model="MLP (64,32 hidden, relu, adam, early_stopping, StandardScaler-preprocessed, fit independently per partition/split)",
    n_partitions=len(results),
    frac_partitions_fd_beats_random_by_r2=float(np.mean(r2_diffs > 0)),
    mean_r2_diff_fd_minus_random=float(np.mean(r2_diffs)),
    median_r2_diff=float(np.median(r2_diffs)),
    min_r2_diff=float(np.min(r2_diffs)),
    max_r2_diff=float(np.max(r2_diffs)),
    std_r2_diff=float(np.std(r2_diffs)),
    paired_ttest_pvalue=float(t_p),
    wilcoxon_pvalue=float(w_p),
    corr_dominant_family_share_vs_r2_advantage=float(np.corrcoef(dom_shares, r2_diffs)[0, 1]),
    corr_test_frac_vs_r2_advantage=float(np.corrcoef(test_fracs, r2_diffs)[0, 1]),
    frac_partitions_family_disjoint_r2_negative=float(np.mean(np.array([r["family_disjoint_r2"] for r in results]) < 0)),
    mean_family_disjoint_r2=float(np.mean([r["family_disjoint_r2"] for r in results])),
    min_family_disjoint_r2=float(np.min([r["family_disjoint_r2"] for r in results])),
)
log(f"SUMMARY: {json.dumps(summary, indent=2)}")
with open("data/mlp_multipartition_robustness.json", "w") as f:
    json.dump(dict(summary=summary, partitions=results), f, indent=2)
log("Saved data/mlp_multipartition_robustness.json")
log("DONE")
