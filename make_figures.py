# Figures, LaTeX table fragments and number macros for the manuscript, generated from the consolidated result files.
# Figures: figures/figN_*.pdf/.png (the manuscript's figures folder links here); fragments: data/manuscript_fragments/.
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROJ = Path(__file__).resolve().parent; D = PROJ / "data"; FIG = PROJ / "figures"; FIG.mkdir(exist_ok=True); FR = D / "manuscript_fragments"; FR.mkdir(exist_ok=True)
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00", grey="#777777", sky="#56B4E9", purple="#CC79A7")
from matplotlib.text import Text
MIN_FS = 8
def save(fig, name):
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_FS]
    assert not small, (name, small[:5])
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(FIG / f"{name}.png", dpi=300); plt.close(fig)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.titlesize": 9, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "figure.constrained_layout.use": True, "mathtext.fontset": "dejavusans"})
MK = ["o", "s", "^", "D", "v", "P"]
def plabel(ax, lab, txt=""): ax.set_title(r"$\mathbf{" + lab + "}$" + (" " + txt if txt else ""), loc="left", fontsize=9)
W = {}  # macro dict
DIG = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))
def put(k, v): W["".join(DIG.get(ch, ch) for ch in k)] = v   # macro names cannot contain digits
def esc(x): return str(x).replace("&", "\\&").replace("_", " ")
def sym(x): return str(x).replace("_", " ").replace(">=", "$\\geq$").replace("<=", "$\\leq$").replace(">", "$>$")
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; pc = lambda x: f"{100*x:.0f}"
# ---------------- load ----------------
A = pd.read_csv(D / "stability_benchmark_all_models.csv"); C = pd.DataFrame(json.load(open(D / "stability_benchmark_more_models.json"))); E = pd.DataFrame(json.load(open(D / "stability_enriched_features.json")))
CE = json.load(open(D / "stability_noise_ceiling.json")); MV = pd.read_csv(D / "sensitivity_multiverse.csv"); SM = json.load(open(D / "sensitivity_multiverse_summary.json"))
CT = pd.read_csv(D / "within_paper_contrasts_window01.csv"); PR = pd.read_csv(D / "prospective_contrasts.csv"); FL = json.load(open(D / "stability_table_flow.json"))
# ---------------- flow / data macros ----------------
put("sRawCells", f"{FL['raw_cells']:,}"); put("sWithProtocol", f"{FL['with_stability_protocol']:,}"); put("sNEffCells", f"{len(pd.read_parquet(D / 'pdp_featurized.parquet', columns=['PCE'])):,}"); put("sFinalCells", f"{FL['final_cells']:,}"); put("sPapers", f"{FL['papers']:,}"); put("sComps", f"{FL['unique_compositions']}")
put("sFamilies", FL["families"]); put("sEvent", f"{FL['status_counts']['event']:,}"); put("sInterval", f"{FL['status_counts']['interval']:,}"); put("sRight", f"{FL['status_counts']['right']:,}")
put("sFracEvent", pc(FL["frac_event"])); put("sRetMedian", f"{100*FL['retention_median']:.0f}"); put("sTexpMedian", f"{FL['t_exp_median_h']:.0f}"); put("sCellsPerPaper", f"{FL['cells_per_paper_median']:.0f}")
# ---------------- benchmark: base-feature R2, 4 common protocols ----------------
PRO4 = ["random", "paper", "composition", "temporal"]
a = A[(A.task == "retention") & (A.features == "all")][["model", "protocol", "r2"]]
c = C[(C.task == "retention") & (C.features == "base")][["model", "protocol", "r2"]]
allr = pd.concat([a, c]); H = allr.pivot_table(index="model", columns="protocol", values="r2")[PRO4]
H["mean"] = H.mean(axis=1); H = H.sort_values("mean", ascending=False)
put("sNModelsReg", int(len(H) - 1)); best = H.drop(index="null", errors="ignore").iloc[0]
Hn = H.drop(index="null", errors="ignore"); put("sBestBaseModel", Hn["paper"].idxmax()); put("sBestBasePaper", f2(Hn["paper"].max())); put("sBestBaseRandom", f2(Hn["random"].max()))
fig, ax = plt.subplots(figsize=(5.5, 0.2 * len(H) + 1.5)); im = ax.imshow(H[PRO4].values, cmap="RdBu", vmin=-0.25, vmax=0.25, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels(["random 5-fold", "paper-held-out", "composition-held-out", "temporal (train ≤2018, test 2019+)"], rotation=30, ha="right", rotation_mode="anchor"); ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index)
for sp in ax.spines.values(): sp.set_visible(False)
for i in range(len(H)):
    for j in range(4):
        v_ = H[PRO4].values[i, j]; ax.text(j, i, f"{v_:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v_) > 0.17 else "black")
fig.colorbar(im, ax=ax, orientation="horizontal", location="top", shrink=0.6, aspect=30, label="retention $R^2$"); save(fig, "fig1_stability_benchmark_heatmap")
H.round(3).to_csv(D / "stability_benchmark_matrix.csv")
# classification and survival bests (base features, 'all')
cl = pd.concat([A[(A.task == "retained80") & (A.features == "all")][["model", "protocol", "auc"]], C[(C.task == "retained80") & (C.features == "base")][["model", "protocol", "auc"]]]).pivot_table(index="model", columns="protocol", values="auc")
put("sNModelsClf", int(len(cl) - 1)); put("sBestAucBase", f2(cl["paper"].drop(index="null", errors="ignore").max())); put("sBestAucBaseModel", cl["paper"].drop(index="null", errors="ignore").idxmax())
sv = pd.concat([A[(A.task == "time_to_80") & (A.features == "all")][["model", "protocol", "cindex_matched"]], C[(C.task == "time_to_80") & (C.features == "base")][["model", "protocol", "cindex_matched"]]]).pivot_table(index="model", columns="protocol", values="cindex_matched")
ref = sv.loc["exposure-time-only (reference)"]; put("sRefC", f2(ref[["paper", "temporal", "random", "composition"]].mean())); put("sRefCpaper", f2(ref["paper"]))
svm = sv.drop(index=["null", "exposure-time-only (reference)"], errors="ignore"); put("sNSurv", int(len(svm))); put("sBestCBase", f2(svm["paper"].max())); put("sBestCBaseModel", svm["paper"].idxmax())
# ---------------- enrichment (phase 14) ----------------
e = E[E.task == "retention"]; ep = e[e.protocol == "paper"].pivot(index="model", columns="features", values="r2"); et = e[e.protocol == "temporal"].pivot(index="model", columns="features", values="r2")
bm = ep["+all enriched"].idxmax(); put("sBestEnrModel", bm); put("sBestEnrPaper", f2(ep.loc[bm, "+all enriched"])); put("sBestEnrBasePaper", f2(ep.loc[bm, "base"])); put("sBestEnrTemporal", f2(et.loc[bm, "+all enriched"])); put("sBestEnrBaseTemporal", f2(et.loc[bm, "base"]))
dl = E[(E.task == "delta_retention") & (E.protocol == "paper")]; dall = dl[dl.features == "+all enriched"]
put("sDeltaAllMin", f2(dall.delta_r2.min())); put("sDeltaAllMax", f2(dall.delta_r2.max())); put("sDeltaProc", f"{dl[dl.features=='+process'].delta_r2.median():.2f}"); put("sDeltaFresh", f"{dl[dl.features=='+fresh PCE'].delta_r2.median():.2f}"); put("sDeltaStack", f"{dl[dl.features=='+stack tokens'].delta_r2.median():.3f}")
cs = E[E.task == "clf_surv"]; pcv = lambda fs, pr, col: float(cs[(cs.features == fs) & (cs.protocol == pr)][col].iloc[0])
put("sAucBase", f2(pcv("base", "paper", "auc"))); put("sAucEnr", f2(pcv("+all enriched", "paper", "auc"))); put("sAucBaseT", f2(pcv("base", "temporal", "auc"))); put("sAucEnrT", f2(pcv("+all enriched", "temporal", "auc")))
put("sCbase", f2(pcv("base", "paper", "cindex_matched"))); put("sCenr", f2(pcv("+all enriched", "paper", "cindex_matched")))
# Fig: enrichment gains (paired deltas) + multiverse distributions
fig, axs = plt.subplots(1, 3, figsize=(6.5, 3.4), gridspec_kw={"width_ratios": [1.5, 1, 1]})
mods = ["Ridge", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "Stacking"]; cols = {"+process": OI["blue"], "+fresh PCE": OI["orange"], "+stack tokens": OI["grey"], "+all enriched": OI["verm"]}
for k, (fs, col) in enumerate(cols.items()):
    for i, mname in enumerate(mods):
        r = dl[(dl.model == mname) & (dl.features == fs)]
        if len(r): axs[0].errorbar(r.delta_r2.iloc[0], i + (k - 1.5) * 0.15, xerr=[[r.delta_r2.iloc[0] - r.delta_ci.iloc[0][0]], [r.delta_ci.iloc[0][1] - r.delta_r2.iloc[0]]], fmt=MK[k], ms=3.5, color=col, lw=1, label=fs if i == 0 else None)
axs[0].axvline(0, color="k", lw=.7); axs[0].set_yticks(range(len(mods))); axs[0].set_yticklabels(mods); axs[0].set_xlabel("Paired gain in $R^2$\n(paper-held-out)"); plabel(axs[0], "(a)")
rg = MV[MV.task == "reg"]; cf = MV[MV.task == "clf"]
for ax, df, lab, pl in [(axs[1], rg, "gain in $R^2$", "(b)"), (axs[2], cf, "gain in AUC", "(c)")]:
    ax.hist(df.delta_enriched, bins=24, color=OI["verm"], alpha=.75, label="with reporting flags"); ax.hist(df.delta_enriched_noflags, bins=24, color=OI["blue"], alpha=.55, hatch="//", label="flags removed"); ax.axvline(0, color="k", lw=.7); ax.set_xlabel(f"Multiverse {lab}"); ax.set_ylabel("analysis paths"); plabel(ax, pl)
h_, l_ = axs[1].get_legend_handles_labels(); fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="outside lower left", ncol=2); fig.legend(h_, l_, frameon=False, loc="outside lower right", ncol=1); save(fig, "fig3_enrichment_multiverse")
rgs, cfs = SM["reg"], SM["clf"]
put("sMvPaths", int(len(MV))); put("sMvReg", rgs["paths"]); put("sMvClf", cfs["paths"]); put("sMvRegMed", f3(rgs["delta_median"])); put("sMvRegLo", f3(rgs["delta_iqr"][0])); put("sMvRegHi", f3(rgs["delta_iqr"][1])); put("sMvRegPos", pc(rgs["frac_positive"]))
put("sMvClfMed", f3(cfs["delta_median"])); put("sMvClfPos", f"{100*cfs['frac_positive']:.1f}"); put("sMvNoflagReg", f3(rgs["noflags_delta_median"])); put("sMvNoflagClf", f3(cfs["noflags_delta_median"]))
put("sMvKeepReg", pc(rgs["noflags_delta_median"] / rgs["delta_median"])); put("sMvKeepClf", pc(cfs["noflags_delta_median"] / cfs["delta_median"])); put("sMvNonpos", int((MV.delta_enriched <= 0).sum())); put("sMvNoflagNonpos", int((MV.delta_enriched_noflags <= 0).sum()))
byd = {r["dup"]: r for r in SM["by_dup"]}; put("sMvDupNone", f3(byd["none"]["delta_median"])); put("sMvDupCollapse", f3(byd["collapse_replicates"]["delta_median"]))
# ---------------- noise ceiling ladder ----------------
lv = ["L0 composition", "L1 composition+ISOS+time", "L3 L1+device stack", "L2 L1 within one paper", "L4 L3 within one paper"]
lab = {"L0 composition": "Composition only", "L1 composition+ISOS+time": "+ protocol and exposure time", "L3 L1+device stack": "+ device stack", "L2 L1 within one paper": "protocol/time, same paper", "L4 L3 within one paper": "+ device stack, same paper"}
fig, ax = plt.subplots(figsize=(5.0, 3.6)); y = np.arange(len(lv))[::-1]
for yi, k in zip(y, lv):
    r = CE["retention"][k]; ax.barh(yi, r["ceiling_r2"], color=OI["blue"], alpha=.85, xerr=[[r["ceiling_r2"] - r["ceiling_r2_ci"][0]], [r["ceiling_r2_ci"][1] - r["ceiling_r2"]]], error_kw=dict(lw=.8)); ax.text(max(r["ceiling_r2"], r["ceiling_r2_ci"][1]) + .015, yi, f"{r['ceiling_r2']:.2f}\n({100*r['share_of_cells']:.0f}% of cells)", va="center")
yp = len(lv) + 0.4; pce = CE["pce_L0"]; ax.barh(yp, pce["ceiling_r2"], color=OI["orange"], alpha=.9, hatch="//"); ax.text(pce["ceiling_r2"] + .015, yp, f"{pce['ceiling_r2']:.2f} (PCE,\ncomposition only)", va="center")
ax.axvline(float(ep.loc[bm, "+all enriched"]), color=OI["verm"], ls="--", lw=1); ax.text(float(ep.loc[bm, "+all enriched"]) + .01, 4.62, "best stability model\n(paper-held-out $R^2$)", color=OI["verm"], va="center")
ax.set_yticks(list(y) + [yp]); ax.set_yticklabels([lab[k].replace("protocol/time, same paper", "protocol/time,\nsame paper").replace("+ device stack, same paper", "+ device stack,\nsame paper").replace("+ protocol and exposure time", "+ protocol and\nexposure time") for k in lv] + ["Efficiency (PCE)"]); ax.set_xlim(0, 0.62); ax.set_ylim(-0.9, yp + 0.6); ax.set_xlabel("Ceiling on $R^2$")
save(fig, "fig2_noise_ceiling_ladder")
for k, nm in [("L0 composition", "L0"), ("L1 composition+ISOS+time", "L1"), ("L2 L1 within one paper", "L2"), ("L3 L1+device stack", "L3"), ("L4 L3 within one paper", "L4")]:
    r = CE["retention"][k]; put(f"sCeil{nm}", f2(r["ceiling_r2"])); put(f"sCeil{nm}lo", f2(r["ceiling_r2_ci"][0])); put(f"sCeil{nm}hi", f2(r["ceiling_r2_ci"][1])); put(f"sCeil{nm}cells", pc(r["share_of_cells"])); put(f"sOracle{nm}", f2(r["oracle_r2"]))
put("sCeilPCE", f2(pce["ceiling_r2"])); put("sCeilPCEcells", pc(pce["share_of_cells"])); icc = CE["between_paper_icc"]; put("sIccRet", f2(icc["y"]["icc"])); put("sIccRetRes", f2(icc["y_res"]["icc"])); put("sIccClf", f2(icc["yc"]["icc"]))
# ---------------- within-paper contrasts + prospective ----------------
ct = CT[CT.papers >= 4].sort_values("perm_p"); nm = {"A_site": "A-site", "B_site": "B-site", "HTL": "HTL", "encapsulation": "Encapsulation", "deposition": "Deposition", "architecture": "Architecture", "backcontact": "Back contact"}
lv_nm = {"lead_mixed": "doped/alloyed Pb", "FA; MA": "FA+MA", "Spin-coating >> Spin-coating": "two-step spin-coating", "True": "encapsulated"}
rows = []
for r in ct.itertuples():
    rows.append(f"{nm.get(r.factor, r.factor)}: {lv_nm.get(r.level, r.level)} vs {r.reference.replace('lead_only', 'Pb only').replace('False', 'none')} & {r.papers} & {r.delta_retention_pp:+.1f} ({r.ci_lo:+.1f} to {r.ci_hi:+.1f}) & {100*r.frac_papers_positive:.0f} & {r.bh_q:.3f} \\\\")
(FR / "table_contrasts.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
sig = CT[(CT.bh_q < 0.05) & (CT.papers >= 4)]; put("sNSigContrasts", int(len(sig))); put("sNContrasts", int(len(CT)))
for key, r_ in [("Enc", ("encapsulation", "True")), ("Fam", ("A_site", "FA; MA")), ("Bsite", ("B_site", "lead_mixed")), ("Dep", ("deposition", "Spin-coating >> Spin-coating"))]:
    x = CT[(CT.factor == r_[0]) & (CT.level == r_[1])].iloc[0]; put(f"sCt{key}", f"{x.delta_retention_pp:+.0f}"); put(f"sCt{key}lo", f"{x.ci_lo:+.0f}"); put(f"sCt{key}hi", f"{x.ci_hi:+.0f}"); put(f"sCt{key}n", int(x.papers)); put(f"sCt{key}q", f"{x.bh_q:.3f}")
pr = PR.dropna(subset=["test_pp"]); put("sPrN", int(len(pr))); put("sPrAgree", int((np.sign(pr.train_pp) == np.sign(pr.test_pp)).sum())); put("sPrTrainPapers", int(PR.train_papers.sum()))
rows = []
for r in PR.sort_values("train_papers", ascending=False).itertuples():
    tst = "--" if pd.isna(getattr(r, "test_pp", np.nan)) else f"{r.test_pp:+.1f}" + ("" if pd.isna(getattr(r, "test_lo", np.nan)) else f" ({r.test_lo:+.1f} to {r.test_hi:+.1f})")
    rows.append(f"{nm.get(r.factor, r.factor)}: {lv_nm.get(r.level, r.level)} vs {r.reference.replace('lead_only', 'Pb only').replace('False', 'none')} & {r.train_papers} & {r.train_pp:+.1f} & {r.test_papers} & {tst} \\\\")
(FR / "table_prospective.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
fig, ax = plt.subplots(figsize=(5.0, 0.45 * len(pr) + 1.4)); p2 = pr.sort_values("train_pp").reset_index(drop=True)
for i, r in p2.iterrows():
    ax.plot([r.train_pp, r.test_pp], [i, i], color=OI["grey"], lw=.8); ax.plot(r.train_pp, i, "o", color=OI["blue"], ms=5); ax.plot(r.test_pp, i, "s", color=OI["verm"], ms=5)
    if not pd.isna(r.get("test_lo", np.nan)): ax.plot([r.test_lo, r.test_hi], [i, i], color=OI["verm"], lw=1.2, alpha=.6)
ax.axvline(0, color="k", lw=.7); ax.set_yticks(range(len(p2))); ax.set_yticklabels([f"{nm.get(r.factor, r.factor)}: {lv_nm.get(r.level, r.level)}\n({r.train_papers}/{r.test_papers} papers)" for r in p2.itertuples()])
ax.plot([], [], "o", color=OI["blue"], label="papers ≤2018"); ax.plot([], [], "s", color=OI["verm"], label="papers 2019–2020"); fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="outside upper center", ncol=2); ax.set_xlabel("Within-paper change in retention (pp)")
save(fig, "fig5_prospective_contrasts")
# ---------------- main benchmark table (selected models) ----------------
sel = [("Stacking", "E"), ("CatBoost", "E"), ("XGBoost", "E"), ("RandomForest", "E"), ("LightGBM", "E"), ("Ridge", "E"), ("Blend (XGB+CatBoost+RF)", "C"), ("Hurdle (clf x two regressors)", "C"), ("Lasso", "C"), ("Spline-additive ridge", "C"), ("Deep ensemble (5 ResNets)", "C")]
rows = []
er = e[e.protocol == "random"].pivot(index="model", columns="features", values="r2")
for mname, src in sel:
    if src == "E": b_p, e_p, b_t, e_t = ep.loc[mname, "base"], ep.loc[mname, "+all enriched"], et.loc[mname, "base"], et.loc[mname, "+all enriched"]; b_r, e_r = er.loc[mname, "base"], er.loc[mname, "+all enriched"]
    else:
        g = C[(C.task == "retention") & (C.model == mname)]; q = lambda fs, pr_: float(g[(g.features == fs) & (g.protocol == pr_)].r2.iloc[0]); b_p, e_p, b_t, e_t = q("base", "paper"), q("+all enriched", "paper"), q("base", "temporal"), q("+all enriched", "temporal"); b_r, e_r = q("base", "random"), q("+all enriched", "random")
    rows.append(f"{mname.replace('(clf x two regressors)', '(two-part)').replace('Deep ensemble (5 ResNets)', 'Deep ensemble (5 residual nets)')} & {b_r:.3f} & {e_r:.3f} & {b_p:.3f} & {e_p:.3f} & {b_t:.3f} & {e_t:.3f} \\\\")
(FR / "table_models.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
# ---------------- SI tables: full benchmark matrices ----------------
def si_rows(df, cols, fmt="{:.3f}"):
    return "\n".join(esc(i) + " & " + " & ".join("--" if pd.isna(v) else fmt.format(v) for v in df.loc[i, cols]) + " \\\\" for i in df.index) + "\n\\bottomrule\n"
PRO6 = ["random", "paper", "composition", "family", "temporal", "protocol"]
Hall = allr.pivot_table(index="model", columns="protocol", values="r2"); Hall = Hall.reindex(columns=PRO6); Hall = Hall.loc[H.index]
(FR / "table_si_models_r2.tex").write_text(si_rows(Hall, PRO6))
cl6 = cl.reindex(columns=PRO6).loc[cl.drop(index="null", errors="ignore").mean(axis=1).sort_values(ascending=False).index.tolist() + (["null"] if "null" in cl.index else [])]
(FR / "table_si_models_auc.tex").write_text(si_rows(cl6, PRO6))
sv6 = sv.reindex(columns=PRO6); sv6 = sv6.loc[sv6.mean(axis=1).sort_values(ascending=False).index]
(FR / "table_si_models_surv.tex").write_text(si_rows(sv6, PRO6))
# multiverse tables by factor
rows = []
for fac, nm_ in [("by_outcome", "Outcome"), ("by_filter", "Cell filter"), ("by_dup", "Duplicate rule"), ("by_model", "Model"), ("by_protocol", "Protocol")]:
    for r in SM[fac]:
        lvl = list(r.values())[0]; rows.append(f"{nm_}: {sym(lvl)} & {r['paths']} & {r['delta_median']:+.3f} & {100*r['frac_positive']:.0f} & {r['delta_noflags_median']:+.3f} \\\\")
(FR / "table_si_multiverse.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
# enrichment table (SI): all models x sets paper & temporal
rows = []
for mname in mods:
    rows.append(f"{mname} & " + " & ".join(("--" if pd.isna(ep.loc[mname].get(fs, np.nan)) else f"{ep.loc[mname, fs]:.3f}") for fs in ["base", "+process", "+stack tokens", "+fresh PCE", "+all enriched"]) + " \\\\")
(FR / "table_si_enrichment_paper.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
rows = []
for mname in mods:
    rows.append(f"{mname} & " + " & ".join(("--" if pd.isna(et.loc[mname].get(fs, np.nan)) else f"{et.loc[mname, fs]:.3f}") for fs in ["base", "+process", "+stack tokens", "+fresh PCE", "+all enriched"]) + " \\\\")
(FR / "table_si_enrichment_temporal.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
# ablation XGBoost base groups
ab = A[(A.task == "ablation")]; rows = []
for fs in ["time", "time+comp", "time+device", "time+protocol", "all"]:
    x = ab[ab.features == fs]; rows.append(f"{fs.replace('time', 'time').replace('+', ' + ')} & {x[x.protocol=='paper'].r2.iloc[0]:.3f} & {x[x.protocol=='temporal'].r2.iloc[0]:.3f} \\\\")
(FR / "table_ablation.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
a_ = ab.set_index(["protocol", "features"]).r2; put("sAbTimeP", f3(a_["paper", "time"])); put("sAbCompP", f3(a_["paper", "time+comp"])); put("sAbDevP", f3(a_["paper", "time+device"])); put("sAbProtP", f3(a_["paper", "time+protocol"])); put("sAbAllP", f3(a_["paper", "all"]))
mx = max(E[(E.task == "retention") & (E.protocol == "paper") & (E.features == "+all enriched")].r2.max(), C[(C.task == "retention") & (C.protocol == "paper") & (C.features == "+all enriched")].r2.max())
mt = max(E[(E.task == "retention") & (E.protocol == "temporal") & (E.features == "+all enriched")].r2.max(), C[(C.task == "retention") & (C.protocol == "temporal") & (C.features == "+all enriched")].r2.max())
mr = max(E[(E.task == "retention") & (E.protocol == "random") & (E.features == "+all enriched")].r2.max(), C[(C.task == "retention") & (C.protocol == "random") & (C.features == "+all enriched")].r2.max())
put("sMaxEnrPaper", f2(mx)); put("sMaxEnrTemporal", f2(mt)); put("sMaxEnrRandom", f2(mr)); put("sMvPos", int((MV.delta_enriched > 0).sum()))
# figure 4: matched within-paper contrasts (forest plot)
m_ = CT[CT.papers >= 4].sort_values("delta_retention_pp").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(5.0, 0.42 * len(m_) + 1.2))
for i, r in m_.iterrows():
    col = OI["verm"] if r.bh_q < 0.10 else OI["blue"]; ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=1.4); ax.plot(r.delta_retention_pp, i, "o" if r.bh_q < 0.10 else "s", color=col, ms=4)
ax.axvline(0, color="k", lw=0.7); ax.set_yticks(range(len(m_))); ax.set_yticklabels([nm.get(r.factor, r.factor) + ": " + lv_nm.get(r.level, r.level) + "\nvs " + str(r.reference).replace("lead_only", "Pb only").replace("False", "none") + " (" + str(r.papers) + " papers)" for r in m_.itertuples()])
ax.set_xlabel("Within-paper change in retention (pp)"); save(fig, "fig4_within_paper_contrasts")
# ---------------- within-paper ranking, selection sensitivity, untested-contrast map ----------------
PW = pd.read_csv(D / "pairwise_learning_nofresh.csv"); PWF = pd.read_csv(D / "pairwise_learning.csv"); BL = json.load(open(D / "pairwise_trivial_baselines.json"))
SEL = pd.read_csv(D / "design_effects_selection_model.csv"); SP = json.load(open(D / "design_effects_selection_pooled.json")); GM = pd.read_csv(D / "untested_contrast_map.csv"); GC = pd.read_csv(D / "untested_contrast_map_check.csv")
def pw(proto, model, subset): return PW[(PW.protocol == proto) & (PW.model == model) & (PW.subset == subset)].iloc[0]
PH, TP = "paper-held-out", "temporal (train <=2018, test 2019+)"
put("sPwPairs", f"{int(pw(PH, 'pooled Ridge', 'all pairs').n_pairs):,}"); put("sPwPapers", f"{int(pw(PH, 'pooled Ridge', 'all pairs').n_papers):,}"); put("sPwGapPairs", f"{int(pw(PH, 'pooled Ridge', 'gap >= 0.10').n_pairs):,}"); put("sPwTempPairs", f"{int(pw(TP, 'pooled Ridge', 'all pairs').n_pairs):,}")
POOLED = [m for m in PW.model.unique() if m.startswith("pooled")]; PAIRED = [m for m in PW.model.unique() if not m.startswith("pooled") and not m.startswith("demeaned")]
for tag, proto, sub in [("Held", PH, "all pairs"), ("HeldGap", PH, "gap >= 0.10"), ("Temp", TP, "all pairs"), ("TempGap", TP, "gap >= 0.10"), ("One", PH, "one coded factor differs"), ("None", PH, "no coded factor differs")]:
    x = PW[(PW.protocol == proto) & (PW.subset == sub)]; bp = x[x.model.isin(POOLED)].sort_values("accuracy").iloc[-1]; bq = x[x.model.isin(PAIRED)].sort_values("accuracy").iloc[-1]
    put(f"sPw{tag}PooledBest", f"{100*bp.accuracy:.0f}"); put(f"sPw{tag}PairedBest", f"{100*bq.accuracy:.0f}"); put(f"sPw{tag}PooledRidge", f"{100*x[x.model == 'pooled Ridge'].iloc[0].accuracy:.0f}")
    put(f"sPw{tag}PairedLo", f"{100*min(x.accuracy):.0f}"); put(f"sPw{tag}PairedHi", f"{100*max(x.accuracy):.0f}")
put("sPwFreshHeld", f"{100*PWF[(PWF.protocol == PH) & (PWF.model == 'pooled Ridge') & (PWF.subset == 'all pairs')].iloc[0].accuracy:.0f}"); put("sPwFreshRule", f"{100*BL['fresh_pce_acc']:.0f}")
# Table: ranking accuracy, selected models
sel_models = ["pooled Ridge", "pooled XGBoost", "pooled CatBoost", "demeaned Ridge", "pairwise logistic (Bradley-Terry)", "pairwise XGBoost (differences)", "RankNet (neural, pairwise loss)"]
nm_m = {"pooled Ridge": "Pooled ridge", "pooled XGBoost": "Pooled XGBoost", "pooled CatBoost": "Pooled CatBoost", "demeaned Ridge": "Ridge on paper-demeaned target", "pairwise logistic (Bradley-Terry)": "Pairwise logistic (Bradley--Terry)", "pairwise XGBoost (differences)": "Pairwise XGBoost on differences", "RankNet (neural, pairwise loss)": "RankNet (pairwise loss)"}
rows = []
for m in sel_models:
    cells = []
    for proto, sub in [(PH, "all pairs"), (PH, "gap >= 0.10"), (PH, "one coded factor differs"), (TP, "all pairs"), (TP, "gap >= 0.10")]:
        r = pw(proto, m, sub); lo, hi = eval(r.acc_ci); cells.append(f"{100*r.accuracy:.1f} ({100*lo:.0f}--{100*hi:.0f})")
    rows.append(nm_m[m] + " & " + " & ".join(cells) + " \\\\")
(FR / "table_ranking.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
# Selection sensitivity
put("sSelOmega", f"{SP['omega_hat']:.2f}"); put("sSelOmegaLo", f"{SP['omega_ci_grid'][0]:.2f}"); put("sSelOmegaHi", f"{SP['omega_ci_grid'][1]:.2f}"); put("sSelLR", f"{-2*SP['pooled_loglik_rel'][0]:.1f}"); put("sSelNContrasts", SP["n_contrasts"])
def selrow(f, lv): return SEL[(SEL.factor == f) & (SEL.level == lv)].iloc[0]
for key, f_, lv_ in [("Pb", "B_site", "lead_mixed"), ("Enc", "encapsulation", "True"), ("Fam", "A_site", "FA; MA"), ("Dep", "deposition", "Spin-coating >> Spin-coating"), ("Tri", "A_site", "Cs; FA; MA")]:
    x = selrow(f_, lv_)
    put(f"sSel{key}Naive", f"{x.naive_mean:+.0f}"); put(f"sSel{key}Half", f"{x['mu_omega0.5']:+.0f}"); put(f"sSel{key}HalfLo", f"{x['mu_omega0.5_lo']:+.0f}"); put(f"sSel{key}HalfHi", f"{x['mu_omega0.5_hi']:+.0f}"); put(f"sSel{key}Quarter", f"{x['mu_omega0.25']:+.0f}"); put(f"sSel{key}Tenth", f"{x['mu_omega0.1']:+.0f}")
    put(f"sSel{key}PredLo", f"{x.naive_pred_lo:+.0f}"); put(f"sSel{key}PredHi", f"{x.naive_pred_hi:+.0f}")
put("sSelPredMin", f"{min(SEL.naive_pred_hi - SEL.naive_pred_lo)/2:.0f}"); put("sSelPredMax", f"{max(SEL.naive_pred_hi - SEL.naive_pred_lo)/2:.0f}")
fig, axs = plt.subplots(1, 2, figsize=(6.5, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
ax = axs[0]; OMs = [1, .5, .25, .1]; cols_ = [OI["verm"], OI["blue"], OI["green"], OI["orange"], OI["purple"], OI["sky"]]
lab = {"B_site": "B-site", "A_site": "A-site", "HTL": "HTL", "encapsulation": "Encapsulation", "deposition": "Deposition"}; lv2 = {"lead_mixed": "doped/alloyed Pb", "FA; MA": "FA+MA", "Cs; FA; MA": "Cs+FA+MA", "True": "encapsulated", "Spin-coating >> Spin-coating": "two-step spin-coating", "P3HT": "P3HT"}
for i, (_, r) in enumerate(SEL.sort_values("naive_mean", ascending=False).iterrows()):
    ys = [r.naive_mean, r['mu_omega0.5'], r['mu_omega0.25'], r['mu_omega0.1']]; x_ = np.arange(4) + (i - 2.5) * 0.04
    ax.plot(x_, ys, "-o", color=cols_[i], ms=3.5, lw=1.2, label=f"{lab.get(r.factor, r.factor)}: {lv2.get(r.level, r.level)} ({r.papers} papers)")
    ax.plot([x_[1]] * 2, [r['mu_omega0.5_lo'], r['mu_omega0.5_hi']], color=cols_[i], lw=.8, alpha=.7)
ax.axhline(0, color="k", lw=.7); ax.set_xticks(range(4)); ax.set_xticklabels(["1\n(none)", "0.5", "0.25", "0.1\n(strong)"]); ax.set_xlabel("Relative probability ω that a harmful-direction contrast is reported"); ax.set_ylabel("Effect on retention (pp)"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1); plabel(ax, "(a)", "Contrast effects under selection")
ax = axs[1]; om = np.array(SP["omega_grid"]); ll_ = np.array(SP["pooled_loglik_rel"]); o_ = np.argsort(om); ax.plot(om[o_], ll_[o_], "-o", color=OI["blue"], ms=3.5); ax.axhline(-1.92, color=OI["grey"], ls="--", lw=.8); ax.text(0.97, -1.7, "95% limit", ha="right", color=OI["grey"])
ax.set_xlabel("ω"); ax.set_ylabel("Pooled log-likelihood (relative)"); plabel(ax, "(b)", "Profile over ω"); save(fig, "fig6_selection_sensitivity")
# Untested-contrast map
put("sMapPairs", len(GM)); put("sMapDirect", int((GM.status == ">=3 papers").sum())); put("sMapFew", int((GM.status == "1-2 papers").sum())); put("sMapEntangled", int((GM.status == "entangled with another factor").sum())); put("sMapNever", int((GM.status == "never co-occur in one paper").sum()))
put("sMapSpearman", f2(GC.pred_effect_pp.corr(GC.direct_mean_pp, method="spearman"))); put("sMapSign", f"{100*(np.sign(GC.pred_effect_pp) == np.sign(GC.direct_mean_pp)).mean():.0f}"); put("sMapChk", len(GC))
def gp(f, a, b_): return GM[(GM.factor == f) & (GM.level_a == a) & (GM.level_b == b_)].iloc[0]
x1 = gp("ETL", "PCBM-60", "SnO2-c"); put("sMapPcbmSno", f"{x1.pred_effect_pp:+.0f}"); put("sMapPcbmSnoLo", f"{x1.ci_lo:+.0f}"); put("sMapPcbmSnoHi", f"{x1.ci_hi:+.0f}")
x2 = gp("ETL", "PCBM-60", "SnO2-np"); put("sMapPcbmSnoNp", f"{x2.pred_effect_pp:+.0f}")
G2 = GM[GM.status != ">=3 papers"].copy(); G2["ord"] = G2.status.map({"never co-occur in one paper": 0, "entangled with another factor": 1, "1-2 papers": 2}); G2 = G2.sort_values(["ord", "pred_effect_pp"]).reset_index(drop=True)
scol = {"never co-occur in one paper": OI["verm"], "entangled with another factor": OI["orange"], "1-2 papers": OI["blue"]}
fig, ax = plt.subplots(figsize=(6.5, 0.15 * len(G2) + 2.0))
for i, r in G2.iterrows():
    ax.plot([r.ci_lo, r.ci_hi], [i, i], color=scol[r.status], lw=1.3, alpha=.8); ax.plot(r.pred_effect_pp, i, {"never co-occur in one paper": "o", "entangled with another factor": "^", "1-2 papers": "s"}[r.status], color=scol[r.status], ms=3.5)
ax.axvline(0, color="k", lw=.7); ax.set_yticks(range(len(G2))); ax.set_yticklabels([(f"{r.factor}: {r.level_b} vs {r.level_a}").replace("_", " ").replace("backcontact", "back contact") for r in G2.itertuples()]); ax.invert_yaxis()
for s_, c_ in scol.items(): ax.plot([], [], {"never co-occur in one paper": "o", "entangled with another factor": "^", "1-2 papers": "s"}[s_], color=c_, label=s_.replace("never co-occur in one paper", "never in the same paper").replace("1-2 papers", "compared in 1-2 papers"))
fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="outside upper center", ncol=1); ax.set_xlabel("Predicted effect of level B vs A on retention (pp)\n(paper-cluster bootstrap 95% interval)"); save(fig, "fig7_untested_contrast_map")
with open(FR / "numbers.tex", "w") as o:
    o.write("% Written by make_figures.py\n")
    for k, v in W.items(): o.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(W), "macros;", "figures:", sorted(p.name for p in FIG.glob("fig*.png")))
