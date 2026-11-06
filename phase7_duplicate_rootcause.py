import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from phase1_pipeline import build_formula_dict, DATA

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
assert len(parsed) == len(feat), "row count mismatch"
check_match = (parsed["formula_str_check"].values == feat["formula_str"].values).mean()
check_pce = np.allclose(parsed["PCE"].values, feat["PCE"].values)
assert check_match > 0.999 and check_pce, "alignment check failed"
log(f"Alignment verified: match_frac={check_match:.4f}, pce_allclose={check_pce}")

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
device_numeric = pd.DataFrame({c: parsed[c].apply(parse_first_number).values for c in device_raw_cols})
for c in device_raw_cols:
    log(f"{c}: {device_numeric[c].notna().sum()}/{len(parsed)} parseable "
        f"({device_numeric[c].notna().mean()*100:.1f}%)")

formula_str = feat["formula_str"].values
pce = feat["PCE"].values

groups = pd.DataFrame({"formula_str": formula_str, "PCE": pce})
for c in device_raw_cols:
    groups[c] = device_numeric[c].values
grp = groups.groupby("formula_str")
sizes = grp.size()
dup_formulas = sizes[sizes > 1].index
log(f"Duplicate-composition groups (size>1): {len(dup_formulas)} "
    f"(covering {sizes[sizes>1].sum()} rows out of {len(groups)})")

rows = []
for f in dup_formulas:
    g = groups[groups["formula_str"] == f]
    pce_range = float(g["PCE"].max() - g["PCE"].min())
    pce_std = float(g["PCE"].std())
    n = len(g)
    rec = dict(formula=f, n=n, pce_range=pce_range, pce_std=pce_std)
    n_device_present = 0
    for c in device_raw_cols:
        vals = g[c].dropna()
        if len(vals) >= 2:
            rec[f"{c}_range"] = float(vals.max() - vals.min())
            rec[f"{c}_std"] = float(vals.std())
            n_device_present += 1
        else:
            rec[f"{c}_range"] = np.nan
            rec[f"{c}_std"] = np.nan
    rec["n_device_fields_with_ge2_values"] = n_device_present
    rows.append(rec)

dup_df = pd.DataFrame(rows)
log(f"Duplicate groups with >=1 device field having >=2 real values: "
    f"{(dup_df['n_device_fields_with_ge2_values'] > 0).sum()} / {len(dup_df)}")

any_device_range_cols = [f"{c}_range" for c in device_raw_cols]
dup_df["any_device_range_max"] = dup_df[any_device_range_cols].max(axis=1, skipna=True)
usable = dup_df.dropna(subset=["any_device_range_max"]).copy()
log(f"Groups usable for device-vs-PCE correlation (>=1 device field with range): {len(usable)}")

results = {}
if len(usable) >= 10:
    r, p = stats.pearsonr(usable["any_device_range_max"], usable["pce_range"])
    rho, p_rho = stats.spearmanr(usable["any_device_range_max"], usable["pce_range"])
    results["any_device_vs_pce_range"] = dict(pearson_r=float(r), pearson_p=float(p),
                                                spearman_rho=float(rho), spearman_p=float(p_rho),
                                                n=int(len(usable)))
    log(f"any_device_range_max vs pce_range: pearson r={r:.3f} (p={p:.2e}), "
        f"spearman rho={rho:.3f} (p={p_rho:.2e}), n={len(usable)}")
else:
    results["any_device_vs_pce_range"] = dict(note="insufficient usable groups", n=int(len(usable)))

per_field = {}
for c in device_raw_cols:
    rc, pc_col = f"{c}_range", None
    sub = dup_df.dropna(subset=[rc])
    if len(sub) >= 10:
        r, p = stats.pearsonr(sub[rc], sub["pce_range"])
        per_field[c] = dict(pearson_r=float(r), pearson_p=float(p), n=int(len(sub)))
        log(f"{c}_range vs pce_range: r={r:.3f} (p={p:.2e}), n={len(sub)}")
    else:
        per_field[c] = dict(note="insufficient data", n=int(len(sub)))
results["per_device_field"] = per_field

dup_rows_mask = groups["formula_str"].isin(dup_formulas)
dup_rows = groups[dup_rows_mask]
grand_mean = dup_rows["PCE"].mean()
ss_total = float(((dup_rows["PCE"] - grand_mean) ** 2).sum())
group_means = dup_rows.groupby("formula_str")["PCE"].transform("mean")
ss_within = float(((dup_rows["PCE"] - group_means) ** 2).sum())
ss_between = ss_total - ss_within
frac_within = ss_within / ss_total if ss_total > 0 else float("nan")
results["variance_decomposition"] = dict(
    ss_total=ss_total, ss_between_composition=ss_between, ss_within_composition=ss_within,
    frac_variance_within_duplicate_groups=float(frac_within),
    n_dup_rows=int(dup_rows_mask.sum()),
    note=("frac_variance_within_duplicate_groups is the share of total PCE variance "
          "among duplicate-composition rows that occurs within a fixed composition "
          "(it cannot be explained by composition, whether or not device or process "
          "variables account for it). It bounds the variance any composition-only model "
          "can explain on this subset."),
)
log(f"Variance decomposition: within-composition fraction = {frac_within:.3f} "
    f"(this is the composition-only noise ceiling)")

verdict = (
    "Device/process-feature range within a fixed duplicate composition shows "
    f"{'a statistically significant positive' if results['any_device_vs_pce_range'].get('pearson_p', 1) < 0.05 and results['any_device_vs_pce_range'].get('pearson_r', 0) > 0 else 'no clear'} "
    "correlation with that composition's PCE range, based on the fields available "
    "(ETL/HTL/perovskite/backcontact thickness only; deposition method, annealing "
    "conditions and other process variables are not captured by these four numeric "
    "fields, so this is a partial test). Independently of that correlation test, the "
    "variance decomposition shows "
    f"{frac_within*100:.1f}% of total PCE variance among duplicate-composition rows is "
    "within-composition variance that composition alone cannot explain, whether or not "
    "the available device fields capture its cause. This indicates a composition-only "
    "noise ceiling."
)
results["verdict"] = verdict
log(verdict)

with open(DATA / "duplicate_disagreement_rootcause.json", "w") as f:
    json.dump(results, f, indent=2)
log("Saved data/duplicate_disagreement_rootcause.json")
log("DONE")
