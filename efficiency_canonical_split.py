import json
import numpy as np, pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
D = Path(__file__).resolve().parent / "data"
d = pd.read_parquet(D / "pdp_featurized.parquet"); feat = [c for c in d.columns if c.startswith("MagpieData") or c.startswith("Magpie")] or [c for c in d.columns if d[c].dtype != object][8:]
feat = [c for c in d.columns if c not in ("formula_str", "PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF", "Perovskite_band_gap", "b_site_class", "is_lead_based", "is_mixed_cation", "family_key", "split_family_disjoint", "n_a_cations")]
X = d[feat].to_numpy(float); y = d.PCE.to_numpy(float); tr = (d.split_family_disjoint == "train").values; te = ~tr
cfg = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8); out = []
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
for rep in range(2):
    g = GradientBoostingRegressor(random_state=0, **cfg).fit(X[tr], y[tr]); gr = GradientBoostingRegressor(random_state=0, **cfg).fit(Xtr, ytr)
    x_ = xgb.XGBRegressor(random_state=0, n_jobs=8, tree_method="hist", **cfg).fit(X[tr], y[tr]); xr = xgb.XGBRegressor(random_state=0, n_jobs=8, tree_method="hist", **cfg).fit(Xtr, ytr)
    out.append(dict(repeat=rep, n_test_family=int(te.sum()), n_test_random=len(yte), gbm_family=float(r2_score(y[te], g.predict(X[te]))), gbm_random=float(r2_score(yte, gr.predict(Xte))), xgb_family=float(r2_score(y[te], x_.predict(X[te]))), xgb_random=float(r2_score(yte, xr.predict(Xte)))))
    print(out[-1], flush=True)
json.dump(out, open(D / "efficiency_canonical_split.json", "w"), indent=1)
