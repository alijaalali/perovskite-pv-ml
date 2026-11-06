import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score

PROJ = Path(__file__).resolve().parent
DATA = PROJ / "data"
RNG = 0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


ION_FORMULA = {
    **{el: {el: 1} for el in [
        "Ag", "Al", "Au", "Ba", "Bi", "Ca", "Co", "Cr", "Cu", "Eu", "Fe", "Ge",
        "Hg", "In", "K", "La", "Li", "Mg", "Mn", "Na", "Nb", "Ni", "Pb", "Rb",
        "Sb", "Sm", "Sn", "Sr", "Tb", "Te", "Ti", "Y", "Zn", "Cs",
        "Br", "Cl", "F", "I", "O", "S",
    ]},
    "MA": {"C": 1, "H": 6, "N": 1},
    "FA": {"C": 1, "H": 5, "N": 2},
    "GU": {"C": 1, "H": 6, "N": 3},
    "IM": {"C": 3, "H": 5, "N": 2},
    "BA": {"C": 4, "H": 12, "N": 1},
    "EA": {"C": 2, "H": 8, "N": 1},
    "HA": {"C": 6, "H": 16, "N": 1},
    "PA": {"C": 3, "H": 10, "N": 1},
    "PEA": {"C": 8, "H": 12, "N": 1},
    "(PEA)": {"C": 8, "H": 12, "N": 1},
    "DMA": {"C": 2, "H": 8, "N": 1},
    "TMA": {"C": 4, "H": 12, "N": 1},
    "(TMA)": {"C": 4, "H": 12, "N": 1},
    "(TBA)": {"C": 16, "H": 36, "N": 1},
    "EDA": {"C": 2, "H": 10, "N": 2},
    "(EDA)": {"C": 2, "H": 10, "N": 2},
    "(AVA)": {"C": 5, "H": 12, "N": 1, "O": 2},
    "(5-AVA)": {"C": 5, "H": 12, "N": 1, "O": 2},
    "(5-AVAI)": {"C": 5, "H": 12, "N": 1, "O": 2},
    "(BDA)": {"C": 4, "H": 14, "N": 2},
    "(PDA)": {"C": 3, "H": 12, "N": 2},
    "PDA": {"C": 3, "H": 12, "N": 2},
    "(6-ACA)": {"C": 6, "H": 14, "N": 1, "O": 2},
    "(BF4)": {"B": 1, "F": 4},
    "(SCN)": {"S": 1, "C": 1, "N": 1},
    "SCN": {"S": 1, "C": 1, "N": 1},
}


def parse_site(ion_str, coef_str):
    if pd.isna(ion_str) or pd.isna(coef_str):
        return None
    ions = [t.strip() for t in str(ion_str).split(";")]
    coefs_raw = [t.strip() for t in str(coef_str).split(";")]
    if len(ions) != len(coefs_raw):
        return None
    try:
        coefs = [float(c) for c in coefs_raw]
    except ValueError:
        return None
    out = {}
    for ion, coef in zip(ions, coefs):
        if "|" in ion:
            return None
        formula = ION_FORMULA.get(ion)
        if formula is None:
            return None
        for el, amt in formula.items():
            out[el] = out.get(el, 0.0) + amt * coef
    return out


def build_formula_dict(row):
    a = parse_site(row["Perovskite_composition_a_ions"], row["Perovskite_composition_a_ions_coefficients"])
    b = parse_site(row["Perovskite_composition_b_ions"], row["Perovskite_composition_b_ions_coefficients"])
    c = parse_site(row["Perovskite_composition_c_ions"], row["Perovskite_composition_c_ions_coefficients"])
    if a is None or b is None or c is None:
        return None
    out = {}
    for d in (a, b, c):
        for el, amt in d.items():
            out[el] = out.get(el, 0.0) + amt
    return out, b


def main():
    log("Loading raw PDP data...")
    df = pd.read_csv(DATA / "perovskite_raw.csv", low_memory=False)
    n0 = len(df)
    log(f"Raw rows: {n0}")

    single = df[df["Perovskite_dimension_list_of_layers"].astype(str).str.strip() == "3.0"].copy()
    n1 = len(single)
    log(f"Single-layer (3D, non-tandem) rows: {n1}")

    formulas, b_sites = [], []
    for _, row in single.iterrows():
        res = build_formula_dict(row)
        if res is None:
            formulas.append(None)
            b_sites.append(None)
        else:
            formulas.append(res[0])
            b_sites.append(res[1])
    single["formula_dict"] = formulas
    single["b_site_dict"] = b_sites
    n2 = int(single["formula_dict"].notna().sum())
    log(f"Rows with fully-parsed composition: {n2} ({n0-n2} dropped to unmapped/ambiguous ion tokens "
        f"or pipe-containing multi-layer entries; ion dictionary: ION_FORMULA)")

    parsed = single[single["formula_dict"].notna()].copy()

    def to_formula_string(d):
        return "".join(f"{el}{amt:.6f}" for el, amt in d.items())

    parsed["formula_str"] = parsed["formula_dict"].apply(to_formula_string)

    def safe_composition(s):
        try:
            return Composition(s)
        except Exception:
            return None

    parsed["composition_obj"] = parsed["formula_str"].apply(safe_composition)
    n3 = int(parsed["composition_obj"].notna().sum())
    parsed = parsed[parsed["composition_obj"].notna()].copy()
    log(f"Rows with a valid pymatgen Composition: {n3}")

    parsed["PCE"] = pd.to_numeric(parsed["JV_default_PCE"], errors="coerce")
    parsed = parsed.dropna(subset=["PCE"])
    n4 = len(parsed)
    log(f"Rows with a non-null PCE: {n4}")

    parsed = parsed[(parsed["PCE"] >= 0) & (parsed["PCE"] <= 35)]
    n5 = len(parsed)
    log(f"Rows with physically plausible PCE (0-35%): {n5}")

    parsed = parsed.reset_index(drop=True)

    def b_site_class(d):
        if d is None:
            return "unknown"
        elements = set(d.keys())
        if "Pb" in elements and len(elements) == 1:
            return "lead_only"
        if "Pb" in elements:
            return "lead_mixed"
        return "lead_free"

    parsed["b_site_class"] = parsed["b_site_dict"].apply(b_site_class)
    parsed["is_lead_based"] = parsed["b_site_class"].isin(["lead_only", "lead_mixed"])

    def a_site_cation_count(row):
        ions = str(row["Perovskite_composition_a_ions"]).split(";")
        return len([t for t in ions if t.strip()])

    parsed["n_a_cations"] = parsed.apply(a_site_cation_count, axis=1)
    parsed["is_mixed_cation"] = parsed["n_a_cations"] > 1

    log(f"Lead-based: {int(parsed['is_lead_based'].sum())}, lead-free: {int((~parsed['is_lead_based']).sum())}")
    log(f"Mixed-cation A-site: {int(parsed['is_mixed_cation'].sum())}, "
        f"single-cation: {int((~parsed['is_mixed_cation']).sum())}")

    def c_site_elements(d, b):
        return None

    parsed["family_key"] = (
        parsed["b_site_class"] + "|" +
        parsed["Perovskite_composition_c_ions"].astype(str).apply(
            lambda s: "-".join(sorted(t.strip() for t in s.split(";")))
        )
    )
    n_families = parsed["family_key"].nunique()
    log(f"Number of distinct perovskite families (B-site class + C-site anion set): {n_families}")

    rng = np.random.RandomState(RNG)
    families = parsed["family_key"].unique()
    rng.shuffle(families)
    test_frac = 0.2
    cum, test_families = 0, []
    fam_sizes = parsed["family_key"].value_counts()
    target_n = int(test_frac * len(parsed))
    for fam in families:
        if cum >= target_n:
            break
        test_families.append(fam)
        cum += fam_sizes[fam]
    parsed["split_family_disjoint"] = np.where(parsed["family_key"].isin(test_families), "test", "train")
    n_test_fam = int((parsed["split_family_disjoint"] == "test").sum())
    log(f"Family-disjoint split: {len(parsed)-n_test_fam} train / {n_test_fam} test "
        f"({len(test_families)} families held out of {n_families})")

    log("Magpie featurization (132 descriptors)...")
    ep = ElementProperty.from_preset("magpie")
    feat_df = ep.featurize_dataframe(
        pd.DataFrame({"composition": parsed["composition_obj"].values}),
        "composition", ignore_errors=True,
    )
    feat_df = feat_df.drop(columns=["composition"]).reset_index(drop=True)
    parsed = parsed.reset_index(drop=True)
    full = pd.concat([parsed.reset_index(drop=True), feat_df], axis=1)
    feat_cols = list(feat_df.columns)
    n_nan_feat = int(full[feat_cols].isna().any(axis=1).sum())
    log(f"Rows with any NaN Magpie feature: {n_nan_feat}")
    full = full.dropna(subset=feat_cols).reset_index(drop=True)
    n6 = len(full)
    log(f"Final usable N after featurization: {n6}")

    keep_cols = [
        "formula_str", "PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF",
        "Perovskite_band_gap", "b_site_class", "is_lead_based", "is_mixed_cation",
        "family_key", "split_family_disjoint", "n_a_cations",
    ] + feat_cols
    out = full[keep_cols].copy()
    out.to_parquet(DATA / "pdp_featurized.parquet")
    log(f"Saved data/pdp_featurized.parquet, shape={out.shape}")

    X = out[feat_cols].to_numpy(dtype=float)
    y = out["PCE"].to_numpy(dtype=float)

    log("Hyperparameter sweep...")
    configs = [
        dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8),
        dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8),
        dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8),
        dict(n_estimators=500, max_depth=2, learning_rate=0.05, subsample=0.8),
    ]
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    sweep_results = []
    for cfg in configs:
        r2s = []
        for tr_idx, te_idx in kf.split(X):
            m = GradientBoostingRegressor(random_state=RNG, **cfg)
            m.fit(X[tr_idx], y[tr_idx])
            r2s.append(r2_score(y[te_idx], m.predict(X[te_idx])))
        sweep_results.append({"config": cfg, "cv_r2_mean": float(np.mean(r2s)), "cv_r2_std": float(np.std(r2s))})
        log(f"  {cfg}: CV R2={np.mean(r2s):.3f}+-{np.std(r2s):.3f}")
    best_cfg = max(sweep_results, key=lambda r: r["cv_r2_mean"])["config"]
    json.dump(sweep_results, open(DATA / "hyperparam_sweep.json", "w"), indent=2)
    log(f"Best config: {best_cfg}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RNG)
    m_rand = GradientBoostingRegressor(random_state=RNG, **best_cfg)
    m_rand.fit(Xtr, ytr)
    pred_rand = m_rand.predict(Xte)
    r2_rand = r2_score(yte, pred_rand)
    mae_rand = mean_absolute_error(yte, pred_rand)
    log(f"Random split: R2={r2_rand:.3f} MAE={mae_rand:.3f}")

    tr_mask = out["split_family_disjoint"].values == "train"
    te_mask = out["split_family_disjoint"].values == "test"
    m_fam = GradientBoostingRegressor(random_state=RNG, **best_cfg)
    m_fam.fit(X[tr_mask], y[tr_mask])
    pred_fam = m_fam.predict(X[te_mask])
    r2_fam = r2_score(y[te_mask], pred_fam)
    mae_fam = mean_absolute_error(y[te_mask], pred_fam)
    log(f"Family-disjoint split: R2={r2_fam:.3f} MAE={mae_fam:.3f}")

    sanity = {
        "n_final": int(n6),
        "best_config": best_cfg,
        "random_split": {"n_train": int(len(ytr)), "n_test": int(len(yte)), "r2": float(r2_rand), "mae": float(mae_rand)},
        "family_disjoint_split": {
            "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
            "n_families_total": int(n_families), "n_families_held_out": len(test_families),
            "r2": float(r2_fam), "mae": float(mae_fam),
        },
        "funnel": {
            "raw_rows": n0, "single_layer": n1, "composition_parsed": n2,
            "valid_pymatgen_composition": n3, "pce_nonnull": n4,
            "pce_plausible_range": n5, "final_after_featurization": n6,
        },
        "subgroups": {
            "lead_based": int(out["is_lead_based"].sum()),
            "lead_free": int((~out["is_lead_based"]).sum()),
            "mixed_cation": int(out["is_mixed_cation"].sum()),
            "single_cation": int((~out["is_mixed_cation"]).sum()),
        },
    }
    json.dump(sanity, open(DATA / "phase1_sanity_check.json", "w"), indent=2)
    log("Saved data/phase1_sanity_check.json")
    log("DONE")


if __name__ == "__main__":
    main()
