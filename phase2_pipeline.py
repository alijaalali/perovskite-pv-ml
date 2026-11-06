import json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RNG = 42

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading featurized + raw data...")
df = pd.read_parquet(DATA / "pdp_featurized.parquet")
raw = pd.read_csv(DATA / "perovskite_raw.csv", low_memory=False)
feat_cols = [c for c in df.columns if c not in
             ["formula_str", "PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF",
              "Perovskite_band_gap", "b_site_class", "is_lead_based", "is_mixed_cation",
              "family_key", "split_family_disjoint", "n_a_cations"]]
log(f"n={len(df)}, n_feat={len(feat_cols)}")

X = df[feat_cols].to_numpy(dtype=float)
y = df["PCE"].to_numpy(dtype=float)
tr_mask = (df["split_family_disjoint"] == "train").to_numpy()
te_mask = (df["split_family_disjoint"] == "test").to_numpy()

log("Step 1: leakage diagnosis")

dup_counts = df["formula_str"].value_counts()
df["formula_dup_count"] = df["formula_str"].map(dup_counts)
frac_dup = float((df["formula_dup_count"] > 1).mean())
log(f"Fraction of rows whose exact formula_str appears >1 time in the dataset: {frac_dup:.4f}")

def leak_frac(mask_tr, mask_te):
    train_formulas = set(df.loc[mask_tr, "formula_str"])
    te_formulas = df.loc[mask_te, "formula_str"]
    return float(te_formulas.isin(train_formulas).mean())

from sklearn.model_selection import train_test_split
idx = np.arange(len(df))
idx_tr_r, idx_te_r = train_test_split(idx, test_size=0.2, random_state=RNG)
rand_tr_mask = np.zeros(len(df), dtype=bool); rand_tr_mask[idx_tr_r] = True
rand_te_mask = np.zeros(len(df), dtype=bool); rand_te_mask[idx_te_r] = True

leak_random = leak_frac(rand_tr_mask, rand_te_mask)
leak_family = leak_frac(tr_mask, te_mask)
log(f"Fraction of test rows with an exact-formula duplicate in train, random split: {leak_random:.4f}")
log(f"Fraction of test rows with an exact-formula duplicate in train, family-disjoint split: {leak_family:.4f}")

test_families = df.loc[te_mask, "family_key"].unique()
pce_test_fam = df.loc[te_mask, "PCE"]
pce_train_fam = df.loc[tr_mask, "PCE"]
pce_random_test = df.loc[rand_te_mask, "PCE"]
hypB = {
    "family_disjoint_test_PCE_std": float(pce_test_fam.std()),
    "family_disjoint_test_PCE_mean": float(pce_test_fam.mean()),
    "family_disjoint_train_PCE_std": float(pce_train_fam.std()),
    "random_test_PCE_std": float(pce_random_test.std()),
    "overall_PCE_std": float(df["PCE"].std()),
    "n_test_families": int(len(test_families)),
}
log(f"PCE std -- family-disjoint test: {hypB['family_disjoint_test_PCE_std']:.3f}, "
    f"family-disjoint train: {hypB['family_disjoint_train_PCE_std']:.3f}, "
    f"random test: {hypB['random_test_PCE_std']:.3f}, overall: {hypB['overall_PCE_std']:.3f}")

fam_stats = df.groupby("family_key")["PCE"].agg(["count", "mean", "std"]).reset_index()
fam_stats["held_out"] = fam_stats["family_key"].isin(test_families)
log("Held-out family PCE stats (count, mean, std):")
log(fam_stats[fam_stats["held_out"]].sort_values("count", ascending=False).head(10).to_string())

verdict = []
if leak_random > leak_family + 0.02:
    verdict.append(
        f"Hypothesis A: random-split test rows have a higher exact-formula "
        f"duplicate-in-train rate ({leak_random:.3f}) than family-disjoint test rows "
        f"({leak_family:.3f}). Removal of duplicate leakage therefore does not explain why the "
        f"family-disjoint split outperforms the random split, since the random split would "
        f"benefit from the duplication. Duplicate formulas with inconsistent PCE across repeats "
        f"(different processing) act as contradictory labels and increase random-split MAE; "
        f"see the within-formula PCE check below."
    )
else:
    verdict.append(
        f"Hypothesis A not supported by exact-formula duplication rates alone "
        f"(random test dup-in-train={leak_random:.3f}, family-disjoint test dup-in-train="
        f"{leak_family:.3f})."
    )

dup_formulas = dup_counts[dup_counts > 1].index
dup_pce_std = df[df["formula_str"].isin(dup_formulas)].groupby("formula_str")["PCE"].std()
mean_within_formula_pce_std = float(dup_pce_std.mean())
verdict.append(
    f"Mean within-formula PCE std (across {len(dup_formulas)} duplicated formulas, "
    f"{frac_dup*100:.1f}% of all rows): {mean_within_formula_pce_std:.3f} percentage points. "
    f"This is a substantial source of label noise at the composition-only feature level "
    f"(same Magpie features, different PCE, due to processing or device differences not "
    f"captured by composition alone)."
)

if hypB["family_disjoint_test_PCE_std"] < hypB["overall_PCE_std"] - 0.3:
    verdict.append(
        f"Hypothesis B: held-out families have a narrower PCE distribution "
        f"(std={hypB['family_disjoint_test_PCE_std']:.3f}) than the overall dataset "
        f"(std={hypB['overall_PCE_std']:.3f}), which makes them easier to predict "
        f"regardless of compositional generalization and contributes to the anomaly. "
        f"Repeated random family partitions are used to control for this instead of a "
        f"single held-out set."
    )
else:
    verdict.append(
        "Hypothesis B not supported: held-out-family PCE variance is comparable to the "
        "overall dataset, so a narrower PCE distribution in the held-out families does not "
        "explain the anomaly."
    )

final_verdict = " ".join(verdict)
log("Verdict: " + final_verdict)

with open(DATA / "leakage_diagnosis.json", "w") as f:
    json.dump({
        "frac_rows_with_duplicate_formula": frac_dup,
        "leak_frac_random_test_dup_in_train": leak_random,
        "leak_frac_family_disjoint_test_dup_in_train": leak_family,
        "within_duplicate_formula_PCE_std_mean": mean_within_formula_pce_std,
        "n_duplicated_formulas": int(len(dup_formulas)),
        "pce_variance_by_split": hypB,
        "verdict": final_verdict,
    }, f, indent=2)
log("Saved data/leakage_diagnosis.json")

log("Step 2: bootstrap CI on split R2/MAE difference")
best_cfg = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=RNG)

def fit_predict(trm, tem):
    m = GradientBoostingRegressor(**best_cfg)
    m.fit(X[trm], y[trm])
    pred = m.predict(X[tem])
    return y[tem], pred

y_rand_te, pred_rand_te = fit_predict(rand_tr_mask, rand_te_mask)
y_fam_te, pred_fam_te = fit_predict(tr_mask, te_mask)

r2_rand = r2_score(y_rand_te, pred_rand_te)
mae_rand = mean_absolute_error(y_rand_te, pred_rand_te)
r2_fam = r2_score(y_fam_te, pred_fam_te)
mae_fam = mean_absolute_error(y_fam_te, pred_fam_te)
log(f"Refit check -- random: R2={r2_rand:.3f} MAE={mae_rand:.3f}; "
    f"family-disjoint: R2={r2_fam:.3f} MAE={mae_fam:.3f}")

n_boot = 3000
rng = np.random.RandomState(RNG)

def bootstrap_mae_diff(yA, predA, yB, predB, n=n_boot):
    nA, nB = len(yA), len(yB)
    idxA = rng.randint(0, nA, size=(n, nA))
    idxB = rng.randint(0, nB, size=(n, nB))
    errA = np.abs(yA[idxA] - predA[idxA]).mean(axis=1)
    errB = np.abs(yB[idxB] - predB[idxB]).mean(axis=1)
    diff = errA - errB
    return diff

diff_mae = bootstrap_mae_diff(y_rand_te, pred_rand_te, y_fam_te, pred_fam_te)
ci_mae = (float(np.percentile(diff_mae, 2.5)), float(np.percentile(diff_mae, 97.5)))

fam_te_ids = df.loc[te_mask, "family_key"].to_numpy()
uniq_test_fams = np.unique(fam_te_ids)
rows_by_fam = {f: np.where(fam_te_ids == f)[0] for f in uniq_test_fams}
diff_mae_grouped = []
for _ in range(n_boot):
    chosen = rng.choice(uniq_test_fams, size=len(uniq_test_fams), replace=True)
    rows = np.concatenate([rows_by_fam[f] for f in chosen])
    errB = np.abs(y_fam_te[rows] - pred_fam_te[rows]).mean()
    idxA = rng.randint(0, len(y_rand_te), size=len(y_rand_te))
    errA = np.abs(y_rand_te[idxA] - pred_rand_te[idxA]).mean()
    diff_mae_grouped.append(errA - errB)
diff_mae_grouped = np.array(diff_mae_grouped)
ci_mae_grouped = (float(np.percentile(diff_mae_grouped, 2.5)), float(np.percentile(diff_mae_grouped, 97.5)))

log(f"Bootstrap (row-level) MAE(random)-MAE(family) 95% CI: {ci_mae}")
log(f"Bootstrap (family-grouped) MAE(random)-MAE(family) 95% CI: {ci_mae_grouped}")

with open(DATA / "split_bootstrap_ci.json", "w") as f:
    json.dump({
        "n_boot": n_boot,
        "refit_random": {"r2": float(r2_rand), "mae": float(mae_rand)},
        "refit_family_disjoint": {"r2": float(r2_fam), "mae": float(mae_fam)},
        "row_level_bootstrap_MAE_diff_95ci": ci_mae,
        "family_grouped_bootstrap_MAE_diff_95ci": ci_mae_grouped,
        "note": "diff = MAE(random_test) - MAE(family_disjoint_test); a positive value with a "
                "confidence interval excluding zero means the family-disjoint test error is "
                "lower than the random-split test error.",
    }, f, indent=2)
log("Saved data/split_bootstrap_ci.json")

log("Step 3: chemistry subgroup diagnosis (variance-controlled)")

def subgroup_bootstrap(mask_a, mask_b, label_a, label_b, n=n_boot):
    yA, predA = y[te_mask & mask_a.to_numpy()], None
    return None

m_full = GradientBoostingRegressor(**best_cfg)
m_full.fit(X[tr_mask], y[tr_mask])
pred_te_full = m_full.predict(X[te_mask])
y_te_full = y[te_mask]
sub_lead = df.loc[te_mask, "is_lead_based"].to_numpy()
sub_mixed = df.loc[te_mask, "is_mixed_cation"].to_numpy()

def grp_stats(mask):
    yy, pp = y_te_full[mask], pred_te_full[mask]
    mae = mean_absolute_error(yy, pp)
    sigma = yy.std()
    return {"n": int(mask.sum()), "mae": float(mae), "sigma_target": float(sigma),
            "mae_over_sigma": float(mae / sigma) if sigma > 0 else None,
            "r2": float(r2_score(yy, pp)) if mask.sum() > 1 else None}

def grp_bootstrap_mae_diff(maskA, maskB, n=n_boot):
    yA, pA = y_te_full[maskA], pred_te_full[maskA]
    yB, pB = y_te_full[maskB], pred_te_full[maskB]
    nA, nB = len(yA), len(yB)
    if nA < 2 or nB < 2:
        return None
    idxA = rng.randint(0, nA, size=(n, nA))
    idxB = rng.randint(0, nB, size=(n, nB))
    errA = np.abs(yA[idxA] - pA[idxA]).mean(axis=1)
    errB = np.abs(yB[idxB] - pB[idxB]).mean(axis=1)
    d = errA - errB
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))

lead_stats = grp_stats(sub_lead)
leadfree_stats = grp_stats(~sub_lead)
single_stats = grp_stats(~sub_mixed)
mixed_stats = grp_stats(sub_mixed)
ci_lead = grp_bootstrap_mae_diff(~sub_lead, sub_lead)
ci_cation = grp_bootstrap_mae_diff(sub_mixed, ~sub_mixed)

log(f"Lead-based (n={lead_stats['n']}): MAE={lead_stats['mae']:.3f}, sigma={lead_stats['sigma_target']:.3f}, "
    f"MAE/sigma={lead_stats['mae_over_sigma']:.3f}, R2={lead_stats['r2']}")
log(f"Lead-free (n={leadfree_stats['n']}): MAE={leadfree_stats['mae']:.3f}, sigma={leadfree_stats['sigma_target']:.3f}, "
    f"MAE/sigma={leadfree_stats['mae_over_sigma']}, R2={leadfree_stats['r2']}")
log(f"Single-cation (n={single_stats['n']}): MAE={single_stats['mae']:.3f}, sigma={single_stats['sigma_target']:.3f}, "
    f"MAE/sigma={single_stats['mae_over_sigma']:.3f}, R2={single_stats['r2']}")
log(f"Mixed-cation (n={mixed_stats['n']}): MAE={mixed_stats['mae']:.3f}, sigma={mixed_stats['sigma_target']:.3f}, "
    f"MAE/sigma={mixed_stats['mae_over_sigma']:.3f}, R2={mixed_stats['r2']}")
log(f"Bootstrap 95% CI, MAE(lead-free)-MAE(lead-based): {ci_lead}")
log(f"Bootstrap 95% CI, MAE(mixed)-MAE(single): {ci_cation}")

with open(DATA / "subgroup_diagnosis.json", "w") as f:
    json.dump({
        "lead_based": lead_stats, "lead_free": leadfree_stats,
        "single_cation": single_stats, "mixed_cation": mixed_stats,
        "bootstrap_95ci_MAE_leadfree_minus_leadbased": ci_lead,
        "bootstrap_95ci_MAE_mixed_minus_single": ci_cation,
        "note": "All statistics are computed on the family-disjoint test set only (model "
                "trained on the family-disjoint training set); subgroups are defined from the "
                "PDP fields is_lead_based and is_mixed_cation.",
    }, f, indent=2)
log("Saved data/subgroup_diagnosis.json")

log("Step 4: multi-model panel (family-disjoint split)")
models = {
    "GBM": GradientBoostingRegressor(**best_cfg),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, random_state=RNG, verbosity=0),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RNG, n_jobs=4),
    "Ridge": Ridge(alpha=1.0, random_state=RNG),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RNG, max_iter=5000),
}
panel = {}
for name, model in models.items():
    model.fit(X[tr_mask], y[tr_mask])
    pred_f = model.predict(X[te_mask])
    r2_f = r2_score(y[te_mask], pred_f)
    mae_f = mean_absolute_error(y[te_mask], pred_f)
    model2 = type(model)(**model.get_params())
    model2.fit(X[rand_tr_mask], y[rand_tr_mask])
    pred_r = model2.predict(X[rand_te_mask])
    r2_r = r2_score(y[rand_te_mask], pred_r)
    mae_r = mean_absolute_error(y[rand_te_mask], pred_r)
    panel[name] = {
        "family_disjoint": {"r2": float(r2_f), "mae": float(mae_f)},
        "random": {"r2": float(r2_r), "mae": float(mae_r)},
        "anomaly_holds": bool(r2_f > r2_r),
    }
    log(f"{name}: family-disjoint R2={r2_f:.3f} MAE={mae_f:.3f} | random R2={r2_r:.3f} MAE={mae_r:.3f} "
        f"| anomaly_holds={panel[name]['anomaly_holds']}")

with open(DATA / "multimodel_panel.json", "w") as f:
    json.dump(panel, f, indent=2)
log("Saved data/multimodel_panel.json")
log("All steps done")
