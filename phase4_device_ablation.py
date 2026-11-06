import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from phase1_pipeline import build_formula_dict, DATA

GBM_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                   subsample=0.8, random_state=0)

raw = pd.read_csv(DATA / "perovskite_raw.csv", low_memory=False)
log(f"Raw rows: {len(raw)}")
single = raw[raw["Perovskite_dimension_list_of_layers"].astype(str).str.strip() == "3.0"].copy()
formulas = []
for _, row in single.iterrows():
    res = build_formula_dict(row)
    formulas.append(res[0] if res is not None else None)
single["formula_dict"] = formulas
parsed = single[single["formula_dict"].notna()].copy()

from pymatgen.core import Composition
def to_formula_string(d):
    return "".join(f"{el}{amt:.6f}" for el, amt in d.items())
parsed["formula_str_check"] = parsed["formula_dict"].apply(to_formula_string)
def safe_comp(s):
    try:
        Composition(s)
        return True
    except Exception:
        return False
parsed = parsed[parsed["formula_str_check"].apply(safe_comp)].copy()
parsed["PCE"] = pd.to_numeric(parsed["JV_default_PCE"], errors="coerce")
parsed = parsed.dropna(subset=["PCE"])
parsed = parsed[(parsed["PCE"] >= 0) & (parsed["PCE"] <= 35)]
log(f"Replicated-filter row count: {len(parsed)} (raw index preserved, not yet reset)")

feat = pd.read_parquet(DATA / "pdp_featurized.parquet")
log(f"Featurized table rows: {len(feat)}")
assert len(parsed) == len(feat), "Row count mismatch between replicated filter and pdp_featurized.parquet!"
check_match = (parsed["formula_str_check"].values == feat["formula_str"].values).mean()
check_pce = np.allclose(parsed["PCE"].values, feat["PCE"].values)
log(f"Alignment check: formula_str exact match fraction={check_match:.4f}, PCE values allclose={check_pce}")
assert check_match > 0.999 and check_pce, "Row alignment between replicated filter and featurized table failed."
log("Row alignment verified; device columns are joined by position.")

def parse_first_number(s):
    if pd.isna(s):
        return np.nan
    s = str(s).split("|")[0].strip()
    try:
        return float(s)
    except Exception:
        return np.nan

device_raw_cols = ["ETL_thickness", "HTL_thickness_list", "Perovskite_thickness",
                    "Backcontact_thickness_list"]
device_numeric = pd.DataFrame({
    c: parsed[c].apply(parse_first_number).values for c in device_raw_cols
}, index=range(len(parsed)))
for c in device_raw_cols:
    n_valid = device_numeric[c].notna().sum()
    log(f"{c}: {n_valid}/{len(parsed)} parseable ({n_valid/len(parsed)*100:.1f}%)")

def top_n_onehot(series, n=8, prefix=""):
    top = series.value_counts().nlargest(n).index.tolist()
    return pd.DataFrame({
        f"{prefix}_{t}": (series == t).astype(int).values for t in top
    }, index=range(len(series)))

etl_stack_oh = top_n_onehot(parsed["ETL_stack_sequence"].reset_index(drop=True), n=8, prefix="ETL")
htl_stack_oh = top_n_onehot(parsed["HTL_stack_sequence"].reset_index(drop=True), n=8, prefix="HTL")
log(f"ETL stack one-hot: {etl_stack_oh.shape[1]} cols (top-8); HTL stack one-hot: {htl_stack_oh.shape[1]} cols")

device_block = pd.concat([device_numeric.reset_index(drop=True),
                           etl_stack_oh.reset_index(drop=True),
                           htl_stack_oh.reset_index(drop=True)], axis=1)

complete_mask = device_numeric.notna().all(axis=1).values
n_complete = int(complete_mask.sum())
log(f"Rows with all 4 numeric device features present: {n_complete} / {len(parsed)} "
    f"({n_complete/len(parsed)*100:.1f}%)")

magpie_cols = [c for c in feat.columns if c.startswith("MagpieData")]
X_comp = feat[magpie_cols].values
y = feat["PCE"].values
device_block_filled = device_block.fillna(device_block.median(numeric_only=True))
X_comp_plus_device = np.concatenate([X_comp, device_block_filled.values], axis=1)

tr_mask_full = (feat["split_family_disjoint"] == "train").values
te_mask_full = (feat["split_family_disjoint"] == "test").values

def fit_eval(X, train_mask, test_mask, label, y_arr=None):
    yy = y if y_arr is None else y_arr
    m = GradientBoostingRegressor(**GBM_PARAMS)
    m.fit(X[train_mask], yy[train_mask])
    pred = m.predict(X[test_mask])
    r2 = r2_score(yy[test_mask], pred)
    mae = mean_absolute_error(yy[test_mask], pred)
    log(f"[{label}] n_train={train_mask.sum()} n_test={test_mask.sum()} R2={r2:.3f} MAE={mae:.3f}")
    return dict(r2=float(r2), mae=float(mae), n_train=int(train_mask.sum()), n_test=int(test_mask.sum()))

log("=== Full data (device features present or median-imputed), family-disjoint split ===")
comp_only_full = fit_eval(X_comp, tr_mask_full, te_mask_full, "composition-only, full data, family-disjoint")
comp_device_full = fit_eval(X_comp_plus_device, tr_mask_full, te_mask_full,
                             "composition+device(imputed), full data, family-disjoint")

log("=== Complete-device-data subset (no imputation), family-disjoint split ===")
X_comp_c = X_comp[complete_mask]
X_dev_c = device_block.values[complete_mask]
X_comp_device_c = np.concatenate([X_comp_c, X_dev_c], axis=1)
y_c = y[complete_mask]
tr_c = tr_mask_full[complete_mask]
te_c = te_mask_full[complete_mask]
log(f"Complete subset family-disjoint split: n_train={tr_c.sum()}, n_test={te_c.sum()}")
if tr_c.sum() > 50 and te_c.sum() > 50:
    comp_only_complete = fit_eval(X_comp_c, tr_c, te_c, "composition-only, complete-device subset, family-disjoint", y_arr=y_c)
    comp_device_complete = fit_eval(X_comp_device_c, tr_c, te_c,
                                     "composition+device(real), complete-device subset, family-disjoint", y_arr=y_c)
else:
    log("Warning: the family-disjoint split of the complete-device subset is too small or imbalanced; "
        "using a random split on the complete subset instead.")
    rng = np.random.default_rng(50000000)
    idx = rng.permutation(len(y_c))
    n_test_c = max(50, int(0.2 * len(y_c)))
    te_idx, tr_idx = idx[:n_test_c], idx[n_test_c:]
    te_c2 = np.zeros(len(y_c), dtype=bool); te_c2[te_idx] = True
    tr_c2 = ~te_c2
    comp_only_complete = fit_eval(X_comp_c, tr_c2, te_c2, "composition-only, complete-device subset, random split", y_arr=y_c)
    comp_device_complete = fit_eval(X_comp_device_c, tr_c2, te_c2,
                                     "composition+device(real), complete-device subset, random split", y_arr=y_c)

out = dict(
    device_feature_completeness={c: float(device_numeric[c].notna().mean()) for c in device_raw_cols},
    n_complete_device_subset=n_complete,
    n_total=len(feat),
    full_data_imputed=dict(composition_only=comp_only_full, composition_plus_device=comp_device_full),
    complete_subset_no_imputation=dict(composition_only=comp_only_complete,
                                        composition_plus_device=comp_device_complete),
    note=("Device features (ETL/HTL/Perovskite/Backcontact thickness parsed as first numeric "
          "token before any '|' multi-value separator; ETL/HTL stack sequence as top-8 one-hot) "
          "are sparse in PDP (see device_feature_completeness). Results are reported for "
          "the full data with median imputation and for the complete-device subset without "
          "imputation."),
)
with open(DATA / "composition_plus_device_ablation.json", "w") as f:
    json.dump(out, f, indent=2)
log("Saved data/composition_plus_device_ablation.json")
log("DONE")
