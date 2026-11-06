import json
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

df = pd.read_parquet("data/pdp_featurized.parquet")
magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
X_all = df[magpie_cols].values
families = df["family_key"].values
log(f"Loaded {len(df)} rows, {len(magpie_cols)} features")

with open("data/multipartition_robustness.json") as f:
    gbm_data = json.load(f)
with open("data/multimodel_multipartition_robustness.json") as f:
    multi_data = json.load(f)

gbm_partitions = {p["partition_id"]: p for p in gbm_data["partitions"]}
ridge_partitions = {p["partition_id"]: p for p in multi_data["per_model_partitions"]["Ridge"]}
xgb_partitions = {p["partition_id"]: p for p in multi_data["per_model_partitions"]["XGBoost"]}
rf_partitions = {p["partition_id"]: p for p in multi_data["per_model_partitions"]["RandomForest"]}
en_partitions = {p["partition_id"]: p for p in multi_data["per_model_partitions"]["ElasticNet"]}

N_PCA = 15
EPS = 1e-8
distance_rows = []
for pmeta in gbm_data["partitions"]:
    pid = pmeta["partition_id"]
    held_out = np.array(pmeta["held_out_families"])
    test_mask = np.isin(families, held_out)
    train_mask = ~test_mask

    scaler = StandardScaler().fit(X_all[train_mask])
    X_train_s = scaler.transform(X_all[train_mask])
    X_test_s = scaler.transform(X_all[test_mask])

    pca = PCA(n_components=N_PCA, random_state=0).fit(X_train_s)
    train_scores = pca.transform(X_train_s)
    test_scores = pca.transform(X_test_s)
    eigvals = pca.explained_variance_
    eigvals = np.clip(eigvals, EPS, None)

    sq_dist_test = np.sum((test_scores ** 2) / eigvals, axis=1)
    mean_dist_test = float(np.sqrt(sq_dist_test).mean())
    median_dist_test = float(np.median(np.sqrt(sq_dist_test)))

    sq_dist_train = np.sum((train_scores ** 2) / eigvals, axis=1)
    mean_dist_train = float(np.sqrt(sq_dist_train).mean())

    row = dict(
        partition_id=pid,
        mean_mahalanobis_distance_test=mean_dist_test,
        median_mahalanobis_distance_test=median_dist_test,
        mean_mahalanobis_distance_train_insample=mean_dist_train,
        distance_ratio_test_over_train=float(mean_dist_test / max(mean_dist_train, EPS)),
        gbm_family_disjoint_r2=gbm_partitions[pid]["family_disjoint_r2"],
        ridge_family_disjoint_r2=ridge_partitions[pid]["family_disjoint_r2"],
        xgboost_family_disjoint_r2=xgb_partitions[pid]["family_disjoint_r2"],
        randomforest_family_disjoint_r2=rf_partitions[pid]["family_disjoint_r2"],
        elasticnet_family_disjoint_r2=en_partitions[pid]["family_disjoint_r2"],
    )
    distance_rows.append(row)
    log(f"partition {pid:2d}: mean_dist_test={mean_dist_test:7.2f} "
        f"ratio={row['distance_ratio_test_over_train']:5.2f} "
        f"GBM_R2={row['gbm_family_disjoint_r2']:.3f} Ridge_R2={row['ridge_family_disjoint_r2']:.3f}")

dist = np.array([r["mean_mahalanobis_distance_test"] for r in distance_rows])
ratio = np.array([r["distance_ratio_test_over_train"] for r in distance_rows])

def corr_report(model_key):
    r2 = np.array([r[f"{model_key}_family_disjoint_r2"] for r in distance_rows])
    c_dist = float(np.corrcoef(dist, r2)[0, 1])
    c_ratio = float(np.corrcoef(ratio, r2)[0, 1])
    from scipy.stats import spearmanr
    rho_dist, p_dist = spearmanr(dist, r2)
    rho_ratio, p_ratio = spearmanr(ratio, r2)
    return dict(pearson_corr_meandist_vs_r2=c_dist, pearson_corr_distratio_vs_r2=c_ratio,
                spearman_rho_meandist_vs_r2=float(rho_dist), spearman_p=float(p_dist),
                spearman_rho_distratio_vs_r2=float(rho_ratio))

correlations = {m: corr_report(m) for m in ["gbm", "ridge", "xgboost", "randomforest", "elasticnet"]}
log(f"CORRELATIONS: {json.dumps(correlations, indent=2)}")

out = dict(
    method="Mahalanobis distance of held-out test-family compositions from the training-set "
           "centroid, computed in a 15-component PCA-whitened subspace of the 132 Magpie "
           "features (StandardScaler + PCA fit on training data only per partition; full "
           "132-dim covariance not used directly due to near-collinearity among Magpie's "
           "min/max/range/mean/avg_dev/mode sub-statistics per element property, which makes "
           "a full-rank Mahalanobis distance numerically unstable).",
    n_pca_components=N_PCA,
    partitions=distance_rows,
    correlations=correlations,
)
with open("data/extrapolation_distance_mechanism.json", "w") as f:
    json.dump(out, f, indent=2)
log("Saved data/extrapolation_distance_mechanism.json")
log("DONE")
