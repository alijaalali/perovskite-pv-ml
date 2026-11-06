import json
import time
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

PROJ = Path(__file__).resolve().parent
DATA = PROJ / "data"
FIG = PROJ / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA / "pdp_deduplicated.parquet")
magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
X = df[magpie_cols].values
y = df["PCE"].values
log(f"Deduplicated data: {X.shape[0]} rows, {X.shape[1]} features")

GBM_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)
model = GradientBoostingRegressor(**GBM_PARAMS)
model.fit(X, y)
log("Fit full-data GBM (same hyperparameters as the main pipeline) for interpretability")

fi = pd.Series(model.feature_importances_, index=magpie_cols).sort_values(ascending=False)
log("Top 15 by GBM feature_importances_:")
for name, val in fi.head(15).items():
    log(f"  {name}: {val:.4f}")

n_shap = min(1473, X.shape[0])
idx = np.random.RandomState(0).choice(X.shape[0], n_shap, replace=False)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X[idx])
mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=magpie_cols).sort_values(ascending=False)
log("Top 15 by mean |SHAP value|:")
for name, val in mean_abs_shap.head(15).items():
    log(f"  {name}: {val:.4f}")

PHYSICS_MAP = {
    "CovalentRadius": "Ionic/covalent radius statistics are the direct inputs to the "
        "Goldschmidt tolerance factor and octahedral factor, the classic stability/"
        "distortion descriptors for ABX3 perovskites (Goldschmidt 1926; Bartel et al. 2019).",
    "Electronegativity": "Electronegativity mismatch between A/B/X-site elements governs "
        "bond ionicity and has been linked to defect tolerance and band-edge character "
        "in halide perovskites.",
    "MeltingT": "Melting temperature correlates with lattice/bond energy and is an "
        "established proxy for thermal and phase stability in perovskite composition design.",
    "GSbandgap": "Ground-state bandgap of the constituent elements is a direct physical "
        "precursor to the compound bandgap, which sets the Shockley-Queisser PCE ceiling.",
    "GSvolume_pa": "Ground-state atomic volume relates to ionic size mismatch and lattice "
        "strain, both tied to the tolerance-factor stability picture.",
    "NValence": "Valence electron count/configuration statistics relate to orbital "
        "character at the band edges and hence to bandgap trends.",
    "NsValence": "s-valence electron count is specifically relevant to halide-perovskite "
        "band-edge states, which are known to have strong s-orbital (Pb/Sn 6s, halide p) "
        "character.",
    "NpValence": "p-valence electron count relates to halide/anion p-orbital contributions "
        "to the valence band edge in halide perovskites.",
    "NfValence": "f-valence electron count is relevant for lanthanide/heavy-element A- or "
        "B-site substituents and their orbital contribution near the band edges.",
    "NUnfilled": "Unfilled valence-orbital count is the direct complement of valence electron "
        "count (same underlying electronic-structure quantity, same band-edge-character "
        "rationale as the NValence family above).",
    "NsUnfilled": "Unfilled s-orbital count, complement of s-valence count; same band-edge "
        "rationale as NsValence.",
    "NpUnfilled": "Unfilled p-orbital count, complement of p-valence count; same halide/anion "
        "band-edge rationale as NpValence.",
    "NdUnfilled": "Unfilled d-orbital count, complement of d-valence count; relevant mainly "
        "for transition-metal B-site substituents' band contributions.",
    "NfUnfilled": "Unfilled f-orbital count, complement of f-valence count; relevant mainly "
        "for lanthanide-type substituents.",
    "AtomicWeight": "Atomic weight (mass) has no direct, well-established photophysical "
        "mechanism for PCE in the perovskite literature; likely acting here as a loose "
        "proxy/correlate for element identity (e.g., heavy-metal vs. light-element "
        "substitution) rather than a mechanistic driver.",
    "MendeleevNumber": "Mendeleev number is a periodic-table-ordering descriptor with no "
        "direct physical mechanism of its own; it mainly serves ML models as a compact "
        "proxy for \"which group of elements\" without a specific photophysical interpretation.",
    "Column": "Periodic-table column (group) correlates with valence-electron count, so "
        "shares some of the valence/bandgap rationale above, but is a cruder proxy.",
    "Row": "Periodic-table row (period) correlates with atomic/ionic radius and principal "
        "quantum number, so loosely connects to the tolerance-factor size picture.",
    "SpaceGroupNumber": "This is the ground-state elemental space-group statistic from "
        "Magpie's underlying elemental reference data, not the perovskite's own space "
        "group; it has no direct, established connection to ABX3 photovoltaic physics.",
    "GSmagmom": "Ground-state magnetic moment of constituent elements has no established "
        "mechanistic connection to PCE in non-magnetic halide/oxide perovskites; likely a "
        "weak, physically unmotivated correlate if it appears among top features.",
}

def classify_feature(name):
    for key, note in PHYSICS_MAP.items():
        if key in name:
            return key, note
    return None, "No documented physical-interpretation mapping available for this specific "\
                 "Magpie statistic in the context of ABX3 perovskite PV physics; treated as "\
                 "uninterpreted."

top_features = mean_abs_shap.head(15)
physics_table = []
for name, shap_val in top_features.items():
    key, note = classify_feature(name)
    physics_table.append(dict(
        feature=name, mean_abs_shap=float(shap_val),
        gbm_importance=float(fi.get(name, np.nan)),
        physics_category=key, physics_note=note,
        has_defensible_physical_interpretation=bool(key is not None and key not in
            ("AtomicWeight", "MendeleevNumber", "SpaceGroupNumber", "GSmagmom")),
    ))

n_interpretable = sum(1 for r in physics_table if r["has_defensible_physical_interpretation"])
log(f"{n_interpretable} / {len(physics_table)} top-15 features have a defensible, "
    f"documented physical interpretation tied to known ABX3 perovskite PV physics.")

try:
    plt.figure(figsize=(7, 5))
    top_features.iloc[::-1].plot(kind="barh")
    plt.xlabel("Mean |SHAP value| (PCE, percentage points)")
    plt.title("Top 15 features, GBM on deduplicated PDP (composition-only)")
    plt.tight_layout()
    plt.savefig(FIG / "figs1_shap_importance.pdf")
    plt.savefig(FIG / "figs1_shap_importance.png", dpi=150)
    log("Saved figures/figs1_shap_importance.pdf/.png")
except Exception as e:
    log(f"Figure generation failed (non-fatal): {e}")

out = dict(
    n_rows_fit=int(X.shape[0]),
    model_r2_in_sample_note="model fit on the full deduplicated data for interpretability only, "
        "not a held-out generalization estimate; generalization results are in "
        "data/dedup_benchmark_comparison.json",
    top15_shap=physics_table,
    n_top15_with_defensible_physical_interpretation=n_interpretable,
    verdict=(
        f"{n_interpretable} of the top 15 SHAP-ranked composition features have a documented "
        "connection to known ABX3 perovskite photovoltaic physics (ionic-radius/"
        "tolerance-factor proxies, electronegativity mismatch, valence-electron/bandgap "
        "proxies). The remainder (atomic weight, Mendeleev number, elemental ground-state "
        "space group/magnetic moment) have no established mechanistic interpretation in this "
        "context and most likely act as coarse element-identity proxies in the model."
    ),
)
with open(DATA / "shap_feature_physics.json", "w") as f:
    json.dump(out, f, indent=2)
log("Saved data/shap_feature_physics.json")
log("DONE")
