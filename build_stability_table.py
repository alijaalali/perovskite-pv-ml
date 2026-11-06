import importlib.util, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

PROJ = Path(__file__).resolve().parent
DATA = PROJ / "data"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

spec = importlib.util.spec_from_file_location("phase1", PROJ / "scripts" / "phase1_pipeline.py")
p1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p1)

raw = pd.read_csv(DATA / "perovskite_raw.csv", low_memory=False)
FLOW = {"raw_cells": len(raw)}
s = raw[raw["Stability_protocol"].notna()].copy(); FLOW["with_stability_protocol"] = len(s)
s = s[s["Perovskite_dimension_list_of_layers"].astype(str).str.strip() == "3.0"].copy(); FLOW["single_layer_3D"] = len(s)

num = lambda x: pd.to_numeric(x, errors="coerce")
fd = []
for _, r in s.iterrows():
    res = p1.build_formula_dict(r); fd.append(None if res is None else res[0])
s["fd"] = fd; s = s[s.fd.notna()].copy(); FLOW["composition_parsed"] = len(s)
s["formula_str"] = s.fd.apply(lambda d: "".join(f"{el}{amt:.6f}" for el, amt in d.items()))
def comp(x):
    try: return Composition(x)
    except Exception: return None
s["comp_obj"] = s.formula_str.map(comp); s = s[s.comp_obj.notna()].copy(); FLOW["valid_composition"] = len(s)

s["raw_idx"] = s.index.values
s["paper_id"] = s["Ref_DOI_number"].fillna("NA_" + s["Ref_ID"].astype(str))
s["year"] = pd.to_datetime(s["Ref_publication_date"], errors="coerce").dt.year
b_sites = [p1.parse_site(r["Perovskite_composition_b_ions"], r["Perovskite_composition_b_ions_coefficients"]) for _, r in s.iterrows()]
def b_class(d):
    if d is None: return "unknown"
    e = set(d); return "lead_only" if e == {"Pb"} else ("lead_mixed" if "Pb" in e else "lead_free")
s["b_site_class"] = [b_class(d) for d in b_sites]
s["family_key"] = s["b_site_class"] + "|" + s["Perovskite_composition_c_ions"].astype(str).apply(lambda x: "-".join(sorted(t.strip() for t in x.split(";"))))
s["a_site"] = s["Perovskite_composition_a_ions"].astype(str)
s["n_a_cations"] = s["a_site"].apply(lambda x: len([t for t in x.split(";") if t.strip()]))

s["pce_initial"] = num(s["Stability_PCE_initial_value"])
s["retention_end"] = num(s["Stability_PCE_end_of_experiment"]) / 100.0
s["t_exp_h"] = num(s["Stability_time_total_exposure"])
s["t80_h"] = num(s["Stability_PCE_T80"]); s["t95_h"] = num(s["Stability_PCE_T95"])
ok = s.t_exp_h.gt(0) & s.retention_end.between(0, 1.5)
FLOW["valid_time_and_retention"] = int(ok.sum()); s = s[ok].copy()
def status(r):
    if pd.notna(r.t80_h) and r.t80_h > 0: return "event"
    if r.retention_end < 0.8: return "interval"
    return "right"
s["status"] = s.apply(status, axis=1)
s["time_obs_h"] = np.where(s.status == "event", s.t80_h, s.t_exp_h)
s["event_flag"] = (s.status == "event").astype(int)
s["contradiction"] = ((s.status == "event") & (s.retention_end >= 0.95) & (s.t80_h < s.t_exp_h * 0.5)).astype(int)
FLOW["status_counts"] = s.status.value_counts().to_dict()

def rng_mean(x):
    v = [float(t) for t in re.findall(r"-?\d+\.?\d*", str(x))]; return np.mean(v) if v else np.nan
s["stab_temp_C"] = s["Stability_temperature_range"].map(rng_mean)
s["stab_RH"] = num(s["Stability_relative_humidity_average_value"])
s["stab_light_intensity"] = num(s["Stability_light_intensity"])
s["cell_area_cm2"] = num(s["Cell_area_measured"])
s["band_gap"] = num(s["Perovskite_band_gap"])
s["pce_fresh"] = num(s["JV_default_PCE"])
s["isos"] = s["Stability_protocol"].astype(str).str.strip()
cats = ["Cell_architecture", "ETL_stack_sequence", "HTL_stack_sequence", "Backcontact_stack_sequence", "Perovskite_deposition_procedure", "Encapsulation", "Stability_atmosphere", "Stability_light_load_condition", "Stability_potential_bias_load_condition", "Stability_temperature_load_condition", "b_site_class"]
for c in cats:
    if c in s: s[c] = s[c].astype(str).replace({"nan": "NA"}).str.strip()
cols_keep = ["raw_idx", "paper_id", "year", "family_key", "formula_str", "a_site", "n_a_cations", "b_site_class", "pce_initial", "pce_fresh", "band_gap", "cell_area_cm2", "stab_temp_C", "stab_RH", "stab_light_intensity", "isos"] + [c for c in cats if c in s and c != "b_site_class"] + ["retention_end", "t_exp_h", "t80_h", "t95_h", "status", "time_obs_h", "event_flag", "contradiction"]
out = s[cols_keep].reset_index(drop=True)

log("Magpie featurization...")
ep = ElementProperty.from_preset("magpie")
comps = [Composition(f) for f in out.formula_str]
feat = pd.DataFrame([ep.featurize(c) for c in comps], columns=ep.feature_labels())
feat = feat.loc[:, feat.notna().all()]
out = pd.concat([out, feat], axis=1)
out["comp_group"] = out.formula_str.map(lambda f: Composition(f).reduced_formula)
out.to_parquet(DATA / "stability_table.parquet")
FLOW.update(final_cells=len(out), papers=int(out.paper_id.nunique()), unique_compositions=int(out.comp_group.nunique()), families=int(out.family_key.nunique()),
            years=[int(out.year.min()), int(out.year.max())], magpie_features=int(feat.shape[1]),
            cells_per_paper_median=float(out.groupby("paper_id").size().median()), frac_event=float(out.event_flag.mean()),
            retention_median=float(out.retention_end.median()), t_exp_median_h=float(out.t_exp_h.median()),
            contradictions=int(out.contradiction.sum()), isos_counts=out.isos.value_counts().head(8).to_dict())
json.dump(FLOW, open(DATA / "stability_table_flow.json", "w"), indent=1, default=str)
log(json.dumps(FLOW, indent=1, default=str))
