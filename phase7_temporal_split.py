import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from phase1_pipeline import build_formula_dict, DATA

GBM_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)

raw = pd.read_csv(DATA / "perovskite_raw.csv", low_memory=False)
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
        Composition(s); return True
    except Exception:
        return False
parsed = parsed[parsed["formula_str_check"].apply(safe_comp)].copy()
parsed["PCE"] = pd.to_numeric(parsed["JV_default_PCE"], errors="coerce")
parsed = parsed.dropna(subset=["PCE"])
parsed = parsed[(parsed["PCE"] >= 0) & (parsed["PCE"] <= 35)].reset_index(drop=True)
log(f"Replicated-filter row count: {len(parsed)}")

feat = pd.read_parquet(DATA / "pdp_featurized.parquet").reset_index(drop=True)
assert len(parsed) == len(feat)
check_match = (parsed["formula_str_check"].values == feat["formula_str"].values).mean()
assert check_match > 0.999
log(f"Alignment verified: match_frac={check_match:.4f}")

pub_date = pd.to_datetime(parsed["Ref_publication_date"], errors="coerce")
pub_year = pub_date.dt.year.values
log(f"Publication year range: {np.nanmin(pub_year)}-{np.nanmax(pub_year)}, "
    f"{pd.isna(pub_year).sum()} missing")
year_counts = pd.Series(pub_year).value_counts().sort_index()
log(f"Year distribution:\n{year_counts.to_string()}")

CUTOFF = 2019
train_mask = pub_year < CUTOFF
test_mask = pub_year >= CUTOFF
log(f"Cutoff year={CUTOFF}: n_train={train_mask.sum()} ({train_mask.mean()*100:.1f}%), "
    f"n_test={test_mask.sum()} ({test_mask.mean()*100:.1f}%)")

magpie_cols = [c for c in feat.columns if c.startswith("MagpieData")]
X = feat[magpie_cols].values
y = feat["PCE"].values
family_key = feat["family_key"].values
formula_str = feat["formula_str"].values

def fit_eval(Xa, ya, tr, te, label):
    m = GradientBoostingRegressor(**GBM_PARAMS)
    m.fit(Xa[tr], ya[tr])
    pred = m.predict(Xa[te])
    r2 = r2_score(ya[te], pred)
    mae = mean_absolute_error(ya[te], pred)
    log(f"[{label}] n_train={tr.sum()} n_test={te.sum()} R2={r2:.3f} MAE={mae:.3f}")
    return dict(r2=float(r2), mae=float(mae), n_train=int(tr.sum()), n_test=int(te.sum()))

log("=== Original (duplicated) data, temporal split ===")
temporal_original = fit_eval(X, y, train_mask, test_mask, "temporal, original data")

fd_mask_te = (feat["split_family_disjoint"] == "test").values
fd_mask_tr = (feat["split_family_disjoint"] == "train").values
log("=== Original data, family-disjoint split (canonical split, for reference) ===")
family_disjoint_original = fit_eval(X, y, fd_mask_tr, fd_mask_te, "family-disjoint, original data (canonical)")

rng = np.random.default_rng(777)
idx = rng.permutation(len(y))
n_test_rand = int(test_mask.sum())
rand_te_idx = idx[:n_test_rand]
rand_te = np.zeros(len(y), dtype=bool); rand_te[rand_te_idx] = True
rand_tr = ~rand_te
log("=== Original data, random split (test size matched to temporal) ===")
random_original = fit_eval(X, y, rand_tr, rand_te, "random, original data (matched size)")

dedup = pd.read_parquet(DATA / "pdp_deduplicated.parquet")
dup_df = pd.DataFrame({"formula_str": formula_str, "pub_year": pub_year})
first_year = dup_df.groupby("formula_str")["pub_year"].min()
dedup["first_pub_year"] = dedup["formula_str"].map(first_year)
log(f"Deduplicated data: {len(dedup)} rows, "
    f"{dedup['first_pub_year'].isna().sum()} missing first-pub-year (dropped)")
dedup2 = dedup.dropna(subset=["first_pub_year"]).copy()
X_dd = dedup2[magpie_cols].values
y_dd = dedup2["PCE"].values
tr_dd = (dedup2["first_pub_year"] < CUTOFF).values
te_dd = (dedup2["first_pub_year"] >= CUTOFF).values
log(f"Dedup temporal split: n_train={tr_dd.sum()}, n_test={te_dd.sum()}")

log("=== Deduplicated data, temporal split (by first-report year) ===")
if tr_dd.sum() > 20 and te_dd.sum() > 20:
    temporal_dedup = fit_eval(X_dd, y_dd, tr_dd, te_dd, "temporal, deduplicated data")
else:
    temporal_dedup = dict(note="insufficient train/test size on dedup temporal split",
                            n_train=int(tr_dd.sum()), n_test=int(te_dd.sum()))
    log(f"WARNING: {temporal_dedup['note']}")

fd_tr_dd = (dedup2["split_family_disjoint"] == "train").values
fd_te_dd = (dedup2["split_family_disjoint"] == "test").values
log("=== Deduplicated data, family-disjoint split (for reference) ===")
family_disjoint_dedup = fit_eval(X_dd, y_dd, fd_tr_dd, fd_te_dd, "family-disjoint, dedup data")

rng2 = np.random.default_rng(778)
idx2 = rng2.permutation(len(y_dd))
n_test_rand_dd = int(te_dd.sum())
rte_idx = idx2[:n_test_rand_dd]
rte_dd = np.zeros(len(y_dd), dtype=bool); rte_dd[rte_idx] = True
rtr_dd = ~rte_dd
log("=== Deduplicated data, random split (matched size) ===")
random_dedup = fit_eval(X_dd, y_dd, rtr_dd, rte_dd, "random, dedup data (matched size)")

out = dict(
    cutoff_year=CUTOFF,
    year_distribution={str(int(k)): int(v) for k, v in year_counts.items() if not pd.isna(k)},
    original_duplicated_data=dict(
        temporal=temporal_original, family_disjoint_canonical=family_disjoint_original,
        random_matched_size=random_original,
    ),
    deduplicated_data=dict(
        temporal_by_first_report_year=temporal_dedup,
        family_disjoint=family_disjoint_dedup, random_matched_size=random_dedup,
    ),
)

ranking_orig = sorted(
    [("temporal", temporal_original["r2"]), ("family_disjoint", family_disjoint_original["r2"]),
     ("random", random_original["r2"])], key=lambda kv: kv[1])
ranking_dedup = sorted(
    [("temporal", temporal_dedup.get("r2", float("nan"))),
     ("family_disjoint", family_disjoint_dedup["r2"]), ("random", random_dedup["r2"])],
    key=lambda kv: kv[1])
out["verdict"] = (
    f"On original (duplicated) data: split difficulty ranking (easiest to hardest R2) = "
    f"{[f'{k}={v:.3f}' for k, v in reversed(ranking_orig)]}. "
    f"On deduplicated data: {[f'{k}={v:.3f}' for k, v in reversed(ranking_dedup)]}. "
    "The temporal split is not family-disjoint (older and newer literature can report the "
    "same or closely related compositions), so a high temporal R2 does not contradict the "
    "family-disjoint and duplication findings. It addresses a different question: whether a "
    "model trained on the literature available at time T predicts the literature that "
    "appears after T, which is relevant to prospective screening."
)
log(out["verdict"])

with open(DATA / "temporal_generalization.json", "w") as f:
    json.dump(out, f, indent=2)
log("Saved data/temporal_generalization.json")
log("DONE")
