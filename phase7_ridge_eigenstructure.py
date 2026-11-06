import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

PROJ = Path(__file__).resolve().parent
DATA = PROJ / "data"

feat = pd.read_parquet(DATA / "pdp_featurized.parquet")
magpie_cols = [c for c in feat.columns if c.startswith("MagpieData")]
X_all = feat[magpie_cols].values
y_all = feat["PCE"].values
family_key = feat["family_key"].values
log(f"Loaded {len(feat)} rows, {len(magpie_cols)} features, "
    f"{pd.Series(family_key).nunique()} families")

d1 = json.load(open(DATA / "multimodel_multipartition_robustness.json"))
ridge_r2 = {p["partition_id"]: p["family_disjoint_r2"] for p in d1["per_model_partitions"]["Ridge"]}
d2 = json.load(open(DATA / "multipartition_robustness.json"))
held_out_map = {p["partition_id"]: p["held_out_families"] for p in d2["partitions"]}

ridge_sorted = sorted(ridge_r2.items(), key=lambda kv: kv[1])
worst5 = [pid for pid, _ in ridge_sorted[:5]]
best5 = [pid for pid, _ in ridge_sorted[-5:]]
log(f"Worst-5 Ridge partitions (by R2): {worst5}")
log(f"Best-5 Ridge partitions (by R2): {best5}")

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def analyze_partition(pid):
    held_out = set(held_out_map[pid])
    test_mask = np.isin(family_key, list(held_out))
    train_mask = ~test_mask
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_te, y_te = X_all[test_mask], y_all[test_mask]

    Xc = X_tr - X_tr.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    cond_number = float(eigvals[-1] / eigvals[0])

    n_low = max(1, int(0.10 * len(eigvals)))
    low_dirs = eigvecs[:, :n_low]
    x_test_mean = X_te.mean(axis=0) - X_tr.mean(axis=0)
    proj_low = low_dirs.T @ x_test_mean
    low_dir_magnitude = float(np.linalg.norm(proj_low))
    total_magnitude = float(np.linalg.norm(x_test_mean))
    frac_in_low_subspace = low_dir_magnitude / total_magnitude if total_magnitude > 0 else float("nan")

    inv_sqrt_eig_low = 1.0 / np.sqrt(eigvals[:n_low])
    weighted_proj = float(np.sqrt(np.sum((proj_low * inv_sqrt_eig_low) ** 2)))

    m = Ridge(alpha=1.0, random_state=0)
    m.fit(X_tr, y_tr)
    r2_check = float(r2_score(y_te, m.predict(X_te)))

    return dict(
        partition_id=pid, n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
        condition_number=cond_number,
        frac_test_deviation_in_bottom10pct_eigendirections=float(frac_in_low_subspace),
        weighted_low_eigendirection_projection=weighted_proj,
        ridge_r2_stored=float(ridge_r2[pid]), ridge_r2_recomputed_check=r2_check,
    )

results = []
for pid in worst5 + best5:
    r = analyze_partition(pid)
    r["group"] = "worst5" if pid in worst5 else "best5"
    results.append(r)
    log(f"partition {pid} [{r['group']}]: cond#={r['condition_number']:.2e}, "
        f"frac_low_eig_dev={r['frac_test_deviation_in_bottom10pct_eigendirections']:.3f}, "
        f"weighted_proj={r['weighted_low_eigendirection_projection']:.3e}, "
        f"R2_stored={r['ridge_r2_stored']:.3f}, R2_recheck={r['ridge_r2_recomputed_check']:.3f}")

df = pd.DataFrame(results)
from scipy import stats
worst = df[df.group == "worst5"]
best = df[df.group == "best5"]
log(f"Mean weighted_low_eigendirection_projection: worst5={worst['weighted_low_eigendirection_projection'].mean():.3e}, "
    f"best5={best['weighted_low_eigendirection_projection'].mean():.3e}")
log(f"Mean frac_test_deviation_in_bottom10pct: worst5={worst['frac_test_deviation_in_bottom10pct_eigendirections'].mean():.3f}, "
    f"best5={best['frac_test_deviation_in_bottom10pct_eigendirections'].mean():.3f}")
log(f"Mean condition_number: worst5={worst['condition_number'].mean():.2e}, best5={best['condition_number'].mean():.2e}")

r_corr, p_corr = stats.pearsonr(df["weighted_low_eigendirection_projection"], df["ridge_r2_stored"])
log(f"Correlation (weighted low-eigendirection projection vs. Ridge R2), n=10: "
    f"pearson r={r_corr:.3f}, p={p_corr:.3f}")

verdict = (
    f"Across the 5 worst- and 5 best-performing Ridge family-disjoint partitions, the "
    f"theory-motivated weighted projection of the held-out family's mean feature deviation "
    f"onto the training covariance's low-eigenvalue (under-constrained) subspace is "
    f"{'substantially larger for the worst partitions' if worst['weighted_low_eigendirection_projection'].mean() > 2*best['weighted_low_eigendirection_projection'].mean() else ('modestly larger for the worst partitions' if worst['weighted_low_eigendirection_projection'].mean() > best['weighted_low_eigendirection_projection'].mean() else 'not larger for the worst partitions, giving no support for this mechanism')}"
    f" (worst5 mean={worst['weighted_low_eigendirection_projection'].mean():.3e} vs. "
    f"best5 mean={best['weighted_low_eigendirection_projection'].mean():.3e}), and the "
    f"pearson correlation across all 10 partitions is r={r_corr:.3f} (p={p_corr:.3f}). "
    "An earlier, simpler (unweighted) Mahalanobis-distance test did not hold after "
    "controlling for test-set size; this eigenvalue-weighted projection is a separate test "
    "motivated by ridge regression theory."
)
log(verdict)

out = dict(
    method="Eigendecomposition of the training-set raw-feature covariance matrix (no "
        "StandardScaler, matching the Ridge(alpha=1.0) fit in "
        "phase4_multimodel_multipartition.py). "
        "For each of the 5 worst- and 5 best-Ridge-R2 family-disjoint partitions, the held-out "
        "test family's mean feature deviation from the training mean is projected onto the "
        "bottom-10%-eigenvalue training-covariance subspace, both as a raw magnitude and as an "
        "inverse-sqrt-eigenvalue-weighted magnitude (ridge extrapolation theory predicts that "
        "this weighting matters, not only the raw distance).",
    partitions=results,
    worst5_mean_weighted_projection=float(worst["weighted_low_eigendirection_projection"].mean()),
    best5_mean_weighted_projection=float(best["weighted_low_eigendirection_projection"].mean()),
    worst5_mean_condition_number=float(worst["condition_number"].mean()),
    best5_mean_condition_number=float(best["condition_number"].mean()),
    correlation_weighted_projection_vs_r2=dict(pearson_r=float(r_corr), pearson_p=float(p_corr), n=10),
    note_on_ridge_preprocessing=(
        "Ridge in the main experiments (phase2_pipeline.py, "
        "phase4_multimodel_multipartition.py) is fit on raw, unstandardized Magpie features "
        "with alpha=1.0. Magpie features span very different raw scales (e.g. AtomicWeight "
        "~1-250, Electronegativity ~0.7-4, valence counts ~0-18), so a single fixed alpha "
        "gives uneven effective regularization across features (the effective ridge penalty "
        "scales with 1/variance of the feature in its raw units). This is a second "
        "contributor to the instability of Ridge, separate from the eigenstructure "
        "mechanism tested above; standardizing features before Ridge or ElasticNet addresses it."
    ),
    verdict=verdict,
)
with open(DATA / "ridge_eigenstructure_mechanism.json", "w") as f:
    json.dump(out, f, indent=2)
log("Saved data/ridge_eigenstructure_mechanism.json")
log("DONE")
