import json, re
from pathlib import Path
import numpy as np, pandas as pd
PROJ = Path(__file__).resolve().parent; DATA = PROJ / "data"
t = pd.read_parquet(DATA / "stability_table.parquet", columns=["raw_idx"])
raw = pd.read_csv(DATA / "perovskite_raw.csv", low_memory=False)
r = raw.loc[t.raw_idx.values]; n = len(r); rows = []
for c in raw.columns:
    v = r[c]; cov = float(v.notna().mean())
    if cov < 0.05: continue
    nun = int(v.nunique(dropna=True)); top = v.value_counts(dropna=True).head(3)
    rows.append(dict(column=c, coverage=cov, n_unique=nun, numeric_share=float(pd.to_numeric(v, errors="coerce").notna().mean()), top=" | ".join(f"{str(k)[:28]}:{int(x)}" for k, x in top.items())))
a = pd.DataFrame(rows).sort_values("coverage", ascending=False); a.to_csv(DATA / "stability_column_audit.csv", index=False)
print("cells", n, "columns with >=5% coverage:", len(a))
pref = ["Perovskite_additives", "Perovskite_deposition", "Perovskite_composition", "Perovskite_band", "Perovskite_thickness", "Perovskite_pl", "Perovskite_single", "Perovskite_dimension", "ETL_", "HTL_", "Backcontact", "Substrate", "Add_lay", "Encapsulation", "Cell_", "Stability_", "JV_", "EQE", "Outdoor"]
for p in pref:
    sub = a[a.column.str.startswith(p)]
    if len(sub): print("\n##", p, f"({len(sub)} cols)"); print(sub.head(14)[["column", "coverage", "n_unique", "top"]].to_string(index=False, max_colwidth=70))
