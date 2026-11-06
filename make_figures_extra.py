# Figures and table fragments for the author-held-out validation, all-model enrichment, nested tuning, IPCW/landmark and efficiency-ladder analyses.
# Reads data/*.json written by the corresponding stability_* / efficiency_* scripts; writes figures/figR*.pdf/.png and data/manuscript_fragments/table_rev_*.tex.
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROJ = Path(__file__).resolve().parent; D = PROJ / "data"; FIG = PROJ / "figures"; FR = D / "manuscript_fragments"; FR.mkdir(exist_ok=True)
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
W = {}
def esc(x): return str(x).replace("&", "\\&").replace("_", " ")

# ---- 1. author-held-out validation: paired gains from added features, with paper-cluster intervals, per model ----
AHall = pd.DataFrame(json.load(open(D / "stability_author_heldout.json"))); AH = AHall[(AHall.get("task").isna()) & (~AHall.noflags.astype(bool))]
TREE = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
prot = ["paper", "last_author", "first_author", "purged_any_author"]; lab = ["paper", "last author", "first author", "any author\n(purged)"]
DL = AHall[(AHall.task == "delta") & (~AHall.noflags.astype(bool))]
mcol = dict(zip(TREE, [OI["blue"], OI["orange"], OI["green"], OI["purple"]]))
fig, axs = plt.subplots(2, 1, figsize=(4.2, 5.4), sharex=True)
for ax, fs, ttl, pl in zip(axs, ["+process+stack (no fresh PCE)", "+all enriched"], ["process and stack details", "all enriched features\n(with fresh-device efficiency)"], ["(a)", "(b)"]):
    for k, m in enumerate(TREE):
        for xi, pr_ in enumerate(prot):
            r = DL[(DL.protocol == pr_) & (DL.model == m) & (DL.features == fs)]
            if len(r): ax.errorbar(xi + (k - 1.5) * 0.15, r.delta_r2.iloc[0], yerr=[[r.delta_r2.iloc[0] - r.delta_ci.iloc[0][0]], [r.delta_ci.iloc[0][1] - r.delta_r2.iloc[0]]], fmt=MK[k], color=mcol[m], ms=3.5, lw=1, capsize=1.5, label=m if xi == 0 else None)
    ax.axhline(0, color="k", lw=.7); ax.set_xticks(range(4)); ax.set_xticklabels(lab); plabel(ax, pl, ttl); ax.set_ylabel("Paired gain in retention $R^2$\nover base features")
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="outside upper center", ncol=2)
save(fig, "figR1_author_heldout")
rows = []
for p_, l in zip(prot, lab):
    for m in ["Ridge"] + TREE:
        g = lambda fs: AH[(AH.protocol == p_) & (AH.model == m) & (AH.features == fs)].r2
        rows.append(f"{l.replace(chr(10), ' ')} & {m} & " + " & ".join(f"{g(fs).iloc[0]:.3f}" for fs in ["base", "+process+stack (no fresh PCE)", "+all enriched"]) + " \\\\")
(FR / "table_rev_author_heldout.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
gd = DL[(DL.features == "+process+stack (no fresh PCE)") & (DL.model.isin(TREE))]
for pr_, tag in [("paper", "Paper"), ("last_author", "Last"), ("purged_any_author", "Purge")]:
    x = gd[gd.protocol == pr_]; W[f"rGainPs{tag}Lo"] = f"{x.delta_r2.min():.2f}"; W[f"rGainPs{tag}Hi"] = f"{x.delta_r2.max():.2f}"; W[f"rGainPs{tag}NSig"] = int(sum(1 for c in x.delta_ci if c[0] > 0)); W[f"rGainPs{tag}N"] = len(x)

# ---- 2. nested tuning vs default ----
NT = pd.DataFrame(json.load(open(D / "stability_tuned_headline_paper.json")) + json.load(open(D / "stability_tuned_headline_temporal.json"))); NT = NT[NT.get("task").isna()]
AM = pd.DataFrame(json.load(open(D / "stability_all_models_enriched.json"))); AMr = AM[AM.task == "retention"]
rows = []
for p in ["paper", "temporal"]:
    for m in TREE:
        t = lambda fs: NT[(NT.protocol == p) & (NT.model == m) & (NT.features == fs)].r2.iloc[0]
        d = lambda fs, f2: AMr[(AMr.protocol == p) & (AMr.model == m) & (AMr.features == f2)].r2.iloc[0]
        rows.append(f"{p} & {m} & {d('', 'base'):.3f} & {t('base'):.3f} & {d('', '+process+stack'):.3f} & {t('+process+stack'):.3f} \\\\")
(FR / "table_rev_nested_tuning.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
W["rNestedBaseMin"] = f"{NT[NT.features=='base'].r2.min():.2f}"; W["rNestedBaseMax"] = f"{NT[NT.features=='base'].r2.max():.2f}"
W["rNestedEnrMin"] = f"{NT[NT.features=='+process+stack'].r2.min():.2f}"; W["rNestedEnrMax"] = f"{NT[NT.features=='+process+stack'].r2.max():.2f}"

# ---- 3. all-model scatter, base vs +all enriched (paper-held-out) ----
P = AMr[AMr.protocol == "paper"].pivot(index="model", columns="features", values="r2").dropna(subset=["base", "+all enriched"]).sort_values("+all enriched")
fig, ax = plt.subplots(figsize=(4.2, 0.2 * len(P) + 1.3)); y = np.arange(len(P))
ax.hlines(y, P["base"], P["+all enriched"], color=OI["grey"], lw=1); ax.plot(P["base"], y, "o", color=OI["grey"], ms=3.5, label="base"); ax.plot(P["+all enriched"], y, "s", color=OI["verm"], ms=3.5, label="+ all enriched")
ax.axvline(0, color="k", lw=.7); ax.set_yticks(y); ax.set_yticklabels(P.index); ax.set_xlabel("Retention $R^2$ (paper-held-out)"); fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="outside upper center", ncol=2)
save(fig, "figR2_all_models_enrichment")
W["rNAllModels"] = len(P); W["rAllBestEnr"] = f"{P['+all enriched'].max():.2f}"; W["rAllBestEnrModel"] = P["+all enriched"].idxmax(); W["rAllBestBase"] = f"{P['base'].max():.2f}"
W["rAllNNeg"] = int((P["+all enriched"] <= 0).sum())

# ---- 4. horizon AUC by cohort and weighting ----
IP = pd.DataFrame(json.load(open(D / "stability_ipcw_horizons.json")))
DCf = pd.DataFrame(json.load(open(D / "stability_ipcw_decomposition.json"))["decomposition"]); fsd = "+all enriched (no time)"
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.8), sharey=True)
combos = [("IPCW-KM", "known cohort, KM weights", "KM-weighted cohort", OI["blue"], "o"), ("IPCW-covariate", "known cohort, covariate weights", "covariate-weighted cohort", OI["sky"], "s"), ("IPCW-KM", "known cohort, unweighted", "unweighted cohort", OI["green"], "^"), ("IPCW-KM", "landmark subset", "landmark subset (weighted-trained)", OI["verm"], "D"), ("landmark", "landmark subset", "landmark subset (landmark-trained)", OI["orange"], "v")]
for ax, pr_ in zip(axs, ["paper", "temporal"]):
    for tr_, sc_, lab_, col, mk in combos:
        v = [DCf[(DCf.horizon == h) & (DCf.protocol == pr_) & (DCf.features == fsd) & (DCf.trained_on == tr_) & (DCf.scored_on == sc_)].auc.iloc[0] for h in (500, 1000)]
        ax.plot([500, 1000], v, mk, color=col, ms=4, label=lab_, ls=["-", "--", ":", "-.", (0, (5, 1))][[c[2] for c in combos].index(lab_)])
    ax.axhline(.5, color="k", lw=.7); ax.set_xticks([500, 1000]); ax.set_xlabel("horizon (h)"); plabel(ax, "(a)" if pr_ == "paper" else "(b)", f"{pr_}-held-out")
axs[0].set_ylabel("AUC, retain 80% to horizon"); fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="outside lower center", ncol=2)
save(fig, "figR3_ipcw_vs_landmark")
for h in (500, 1000):
    for pr_ in ("paper", "temporal"):
        for an in ("ipcw", "landmark"):
            x = IP[(IP.analysis == an) & (IP.protocol == pr_) & (IP.horizon == h) & (IP.model == "XGBoost") & (IP.features == "+all enriched (no time)")]
            if len(x): W[f"rAuc{an.capitalize()}{h}{pr_.capitalize()}"] = f"{x.auc.iloc[0]:.2f}"
rows = []
for h in sorted(IP.horizon.unique()):
    for pr_ in ["paper", "temporal"]:
        r = []
        for an in ("ipcw", "landmark"):
            x = IP[(IP.analysis == an) & (IP.protocol == pr_) & (IP.horizon == h) & (IP.model == "XGBoost") & (IP.features == "+all enriched (no time)")]
            r.append("--" if x.empty else f"{x.auc.iloc[0]:.3f} [{x.auc_ci.iloc[0][0]:.2f}, {x.auc_ci.iloc[0][1]:.2f}]")
        rows.append(f"{h} & {pr_} & {r[0]} & {r[1]} \\\\")
(FR / "table_rev_ipcw_landmark.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

# ---- 5. efficiency ladder ----
ED = pd.DataFrame(json.load(open(D / "efficiency_device_features.json"))); EC = json.load(open(D / "efficiency_ceilings.json"))
fs_e = ["composition (Magpie)", "+ device stack", "+ stack + process"]; pr_e = ["random", "paper", "last_author", "temporal", "family"]
fig, ax = plt.subplots(figsize=(4.2, 3.6)); w = 0.26
for k, (fs, col) in enumerate(zip(fs_e, [OI["grey"], OI["blue"], OI["verm"]])):
    v = [ED[(ED.protocol == p) & (ED.features == fs) & (ED.model == "XGBoost")].r2 for p in pr_e]
    ax.bar(np.arange(len(pr_e)) + (k - 1) * w, [x.iloc[0] if len(x) else np.nan for x in v], w, color=col, label=fs, hatch=["", "//", ".."][k], edgecolor="white" if k == 0 else "k", lw=0.5)
for (nm_, v) in zip(["L0", "L1", "L2"], [EC["nested_on_L3"][k]["ceiling"] for k in ["L0 composition", "L1 + device stack", "L2 + stack + process"]]):
    pass
ax.set_xticks(range(len(pr_e))); ax.set_xticklabels([p.replace("_", "\n") for p in pr_e]); ax.set_ylabel("PCE $R^2$ (XGBoost)"); fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="outside upper center", ncol=1)
save(fig, "figR4_efficiency_feature_ladder")
rows = []
for k in EC["nested_on_L3"]:
    r = EC["nested_on_L3"][k]; rows.append(f"{k} & {r['cells']:,} & {r['ceiling']:.3f} \\\\")
(FR / "table_rev_efficiency_ceilings.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
rows = []
for p in pr_e:
    for m in ["XGBoost", "LightGBM", "Ridge"]:
        g = lambda fs: ED[(ED.protocol == p) & (ED.model == m) & (ED.features == fs)]
        if all(len(g(fs)) for fs in fs_e): rows.append(f"{p.replace('_', ' ')} & {m} & " + " & ".join(f"{g(fs).r2.iloc[0]:.3f}" for fs in fs_e) + " \\\\")
(FR / "table_rev_efficiency_ladder.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

# ---- number macros ----
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
def rng(v): return f2(min(v)), f2(max(v))
for tag, pr_, fs in [("BasePaper", "paper", "base"), ("BaseLast", "last_author", "base"), ("BaseFirst", "first_author", "base"), ("BasePurge", "purged_any_author", "base"),
                     ("EnrPaper", "paper", "+all enriched"), ("EnrLast", "last_author", "+all enriched"), ("EnrPurge", "purged_any_author", "+all enriched"), ("ProcPaper", "paper", "+process+stack (no fresh PCE)"), ("ProcPurge", "purged_any_author", "+process+stack (no fresh PCE)")]:
    v = AH[(AH.protocol == pr_) & (AH.features == fs) & AH.model.isin(TREE)].r2; lo, hi = rng(v); W[f"rAh{tag}Lo"] = lo; W[f"rAh{tag}Hi"] = hi
W["rAhPurgeTrain"] = "32--38"
MA = pd.read_csv(D / "contrast_meta_analysis.csv").set_index(["factor", "level"])
for key, ix in [("Enc", ("encapsulation", "True")), ("Fam", ("A_site", "FA; MA")), ("Dep", ("deposition", "Spin-coating >> Spin-coating")), ("Pb", ("B_site", "lead_mixed")), ("Pthree", ("HTL", "P3HT")), ("Csfam", ("A_site", "Cs; FA; MA"))]:
    r = MA.loc[ix]; W[f"rMa{key}"] = f"{r.re_mean:+.1f}"; W[f"rMa{key}Lo"] = f"{r.hk_lo:+.1f}"; W[f"rMa{key}Hi"] = f"{r.hk_hi:+.1f}"; W[f"rMa{key}K"] = int(r.papers)
    if not pd.isna(r.egger_p): W[f"rMa{key}Egger"] = f"{r.egger_p:.2f}"; W[f"rMa{key}Pet"] = f"{r.pet_intercept:+.1f}"
NC = json.load(open(D / "noise_ceiling_validation.json"))["nested"]
for lev, nm_ in [("L0 composition", "Lzero"), ("L1 composition+ISOS+time", "Lone"), ("L3 L1+device stack", "Lthree"), ("L2 L1 within one paper", "Ltwo"), ("L4 L3 within one paper", "Lfour")]:
    v = [NC[sub]["levels"][lev]["ceiling"] for sub in NC if lev in NC[sub]["levels"]]; W[f"rNc{nm_}Lo"], W[f"rNc{nm_}Hi"] = rng(v)
W["rNcCells"] = f"{NC['L4 L3 within one paper']['cells']:,}"
EC2 = EC["nested_on_L3"]; W["rEcZero"] = f3(EC2["L0 composition"]["ceiling"]); W["rEcOne"] = f3(EC2["L1 + device stack"]["ceiling"]); W["rEcTwo"] = f3(EC2["L2 + stack + process"]["ceiling"]); W["rEcThree"] = f3(EC2["L3 L2 within one paper"]["ceiling"]); W["rEcCells"] = f"{EC2['L0 composition']['cells']:,}"
for tag, pr_ in [("Rand", "random"), ("Paper", "paper"), ("Last", "last_author"), ("Temp", "temporal"), ("Fam", "family")]:
    for t2, fs in [("C", fs_e[0]), ("D", fs_e[1]), ("P", fs_e[2])]:
        x = ED[(ED.protocol == pr_) & (ED.features == fs) & (ED.model == "XGBoost")]; W[f"rEff{tag}{t2}"] = f2(x.r2.iloc[0])
EB = json.load(open(D / "efficiency_partition_blocks.json")); W["rEffShareLargest"] = f"{100*EB['share_largest_family']:.0f}"; W["rEffPartDelta"] = f"{EB['overall_mean_delta']:+.3f}"; W["rEffPartLo"] = f"{EB['overall_ci_partition_bootstrap'][0]:+.3f}"; W["rEffPartHi"] = f"{EB['overall_ci_partition_bootstrap'][1]:+.3f}"
W["rEffPartCorr"] = f2(EB["corr_delta_testfrac"])
SV = json.load(open(D / "simulation_validation.json"))["matched_concordance"]
sv_ = [r for r in SV if r["b_informative"] == 1.0 and r["rho"] == 0.6][0]; sv0 = [r for r in SV if r["b_informative"] == 1.0 and r["rho"] == 0.0][0]
W["rSimExpAll"] = f2(sv0["exposure only | all comparable pairs"]); W["rSimExpMatched"] = f2(sv0["exposure only | matched 0.15"]); W["rSimTruth"] = f2(sv_["truth_all_pairs"]); W["rSimModelMatched"] = f2(sv_["model score | matched 0.15"])
PWd = pd.read_csv(D / "pairwise_learning_nofresh.csv")
for tag, proto in [("Held", "paper-held-out"), ("Temp", "temporal (train <=2018, test 2019+)")]:
    x = PWd[(PWd.protocol == proto) & (PWd.subset == "all pairs") & (PWd.model != "pooled Ridge")]; W[f"rPwDiff{tag}Max"] = f"{100*x.acc_diff_vs_pooled_ridge.max():+.1f}"; W[f"rPwDiff{tag}Min"] = f"{100*x.acc_diff_vs_pooled_ridge.min():+.1f}"
    W[f"rPwDiff{tag}NSig"] = int(sum(1 for c in x.acc_diff_ci if eval(c)[0] > 0))

# ---- tuned headline gains, high-R2 mechanisms, lead-free and strict contrasts ----
import os
if (D / "stability_tuned_headline_paper.json").exists() and (D / "stability_tuned_headline_temporal.json").exists():
    TH = pd.DataFrame(json.load(open(D / "stability_tuned_headline_paper.json")) + json.load(open(D / "stability_tuned_headline_temporal.json"))); TG_ = TH[TH.task == "gain"] if "task" in TH else TH.iloc[0:0]; TR = TH[TH.get("task").isna()]
    rows = []
    for pr_ in ["paper", "temporal"]:
        for m in ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "Blend (tuned)"]:
            b = TR[(TR.protocol == pr_) & (TR.model == m) & (TR.features == "base")]; e1 = TR[(TR.protocol == pr_) & (TR.model == m) & (TR.features == "+process+stack")]; e2 = TR[(TR.protocol == pr_) & (TR.model == m) & (TR.features == "+all enriched")]
            g1 = TG_[(TG_.protocol == pr_) & (TG_.model == m) & (TG_.features == "+process+stack")]; g2 = TG_[(TG_.protocol == pr_) & (TG_.model == m) & (TG_.features == "+all enriched")]
            if b.empty: continue
            rows.append(f"{pr_} & {m.replace(' (tuned)', '')} & {b.r2.iloc[0]:.3f} & {e1.r2.iloc[0]:.3f} & {g1.gain.iloc[0]:+.3f} [{g1.gain_ci.iloc[0][0]:+.3f}, {g1.gain_ci.iloc[0][1]:+.3f}] & {e2.r2.iloc[0]:.3f} & {g2.gain.iloc[0]:+.3f} [{g2.gain_ci.iloc[0][0]:+.3f}, {g2.gain_ci.iloc[0][1]:+.3f}] \\\\")
    (FR / "table_rev_tuned.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    rows = []
    for r in TR.itertuples():
        pass
    for pr_ in ["paper", "temporal"]:
        for m in ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "Blend (tuned)"]:
            vals = [TR[(TR.protocol == pr_) & (TR.model == m) & (TR.features == fs)] for fs in ["base", "+process+stack", "+all enriched"]]
            if any(v.empty for v in vals): continue
            rows.append(f"{pr_} & {m.replace(' (tuned)', '')} & " + " & ".join(f"{v.r2.iloc[0]:.3f} / {v.r2_paperweighted.iloc[0]:.3f}" for v in vals) + " \\\\")
    (FR / "table_rev_paperweighted.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    bl = lambda pr_, fs, col: TR[(TR.protocol == pr_) & (TR.model == "Blend (tuned)") & (TR.features == fs)][col].iloc[0]
    for pr_, tag in [("paper", "Paper"), ("temporal", "Temp")]:
        W[f"rTun{tag}BlendBase"] = f2(bl(pr_, "base", "r2")); W[f"rTun{tag}BlendPs"] = f2(bl(pr_, "+process+stack", "r2")); W[f"rTun{tag}BlendAll"] = f2(bl(pr_, "+all enriched", "r2"))
        g = TG_[(TG_.protocol == pr_) & (TG_.model == "Blend (tuned)") & (TG_.features == "+process+stack")].iloc[0]; W[f"rTun{tag}BlendGain"] = f2(g.gain); W[f"rTun{tag}BlendGainLo"] = f2(g.gain_ci[0]); W[f"rTun{tag}BlendGainHi"] = f2(g.gain_ci[1])
        g = TG_[(TG_.protocol == pr_) & (TG_.model == "Blend (tuned)") & (TG_.features == "+all enriched")].iloc[0]; W[f"rTun{tag}BlendFreshGain"] = f2(g.gain)
        gg = TG_[(TG_.protocol == pr_) & (TG_.features == "+process+stack")]; W[f"rTun{tag}GainLo"] = f2(gg.gain.min()); W[f"rTun{tag}GainHi"] = f2(gg.gain.max()); W[f"rTun{tag}GainNSig"] = int(sum(1 for c in gg.gain_ci if c[0] > 0)); W[f"rTun{tag}GainN"] = len(gg)
        bb = TR[(TR.protocol == pr_) & (TR.features == "base") & (TR.model != "Blend (tuned)")]; W[f"rTun{tag}BaseLo"] = f3(bb.r2.min()); W[f"rTun{tag}BaseHi"] = f3(bb.r2.max())
        W[f"rTun{tag}PwBlendGain"] = f2(TG_[(TG_.protocol == pr_) & (TG_.model == "Blend (tuned)") & (TG_.features == "+process+stack")].gain_paperweighted.iloc[0])
if (D / "stability_ipcw_decomposition.json").exists():
    DC = json.load(open(D / "stability_ipcw_decomposition.json")); DD = pd.DataFrame(DC["decomposition"]); fs_ = "+all enriched (no time)"
    rows = []
    for h in [500, 1000]:
        for pr_ in ["paper", "temporal"]:
            def g_(tr_, sc_):
                x = DD[(DD.horizon == h) & (DD.protocol == pr_) & (DD.features == fs_) & (DD.trained_on == tr_) & (DD.scored_on == sc_)]
                return "--" if x.empty else f"{x.auc.iloc[0]:.3f} [{x.auc_ci.iloc[0][0]:.2f}, {x.auc_ci.iloc[0][1]:.2f}]"
            rows.append(f"{h} & {pr_} & {g_('IPCW-KM', 'known cohort, KM weights')} & {g_('IPCW-covariate', 'known cohort, covariate weights')} & {g_('IPCW-KM', 'known cohort, unweighted')} & {g_('IPCW-KM', 'landmark subset')} & {g_('landmark', 'landmark subset')} \\\\")
            for tag, tr_, sc_ in [("Km", "IPCW-KM", "known cohort, KM weights"), ("Cov", "IPCW-covariate", "known cohort, covariate weights"), ("Unw", "IPCW-KM", "known cohort, unweighted"), ("LmOnIpcw", "IPCW-KM", "landmark subset"), ("LmOnLm", "landmark", "landmark subset")]:
                x = DD[(DD.horizon == h) & (DD.protocol == pr_) & (DD.features == fs_) & (DD.trained_on == tr_) & (DD.scored_on == sc_)]
                if len(x): W[f"rDec{tag}{h}{pr_.capitalize()}"] = f2(x.auc.iloc[0])
    (FR / "table_rev_ipcw_decomposition.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    SMs = DC["simulation"]; rows = [f"{esc(r['scenario'])} & {r['true']:.3f} & {r['km']:.3f} & {r['cov']:.3f} & {r['unweighted']:.3f} & {r['landmark']:.3f} & {100*r['known_share']:.0f} / {100*r['landmark_share']:.0f} \\\\" for r in SMs]
    (FR / "table_rev_ipcw_sim.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    r0 = SMs[0]; W["rIpcwSimTrue"] = f2(r0["true"]); W["rIpcwSimKm"] = f2(r0["km"]); W["rIpcwSimCov"] = f2(r0["cov"]); W["rIpcwSimLm"] = f2(r0["landmark"])
HM = pd.DataFrame(json.load(open(D / "high_r2_mechanisms.json")))
W["rHighMaxFinal"] = f2(HM[HM.target == "final efficiency"].r2.max()); W["rHighMaxT"] = f2(HM[HM.target.str.startswith("log10")].r2.max())
W["rHighFinalNoFresh"] = f2(HM[(HM.target == "final efficiency") & (HM.protocol == "random") & (HM.features == "base (no time)")].r2.iloc[0]); W["rHighFinalFresh"] = f2(HM[(HM.target == "final efficiency") & (HM.protocol == "random") & (HM.features == "base + fresh efficiency")].r2.iloc[0])
W["rHighFinalPaperBase"] = f2(HM[(HM.target == "final efficiency") & (HM.protocol == "paper") & (HM.features == "base (no time)")].r2.iloc[0]); W["rHighFinalPaperFresh"] = f2(HM[(HM.target == "final efficiency") & (HM.protocol == "paper") & (HM.features == "base + fresh efficiency")].r2.iloc[0])
W["rHighTNoTime"] = f2(HM[(HM.target.str.startswith("log10")) & (HM.protocol == "random") & (HM.features == "base (no time)")].r2.iloc[0]); W["rHighTTime"] = f2(HM[(HM.target.str.startswith("log10")) & (HM.protocol == "random") & (HM.features == "base + exposure time")].r2.iloc[0])
rows = [f"{r.target} & {r.protocol.replace('paper', 'paper-held-out').replace('random', 'random')} & {r.features} & {r.n:,} & {r.r2:.3f} \\\\" for r in HM.itertuples()]
(FR / "table_rev_high_r2.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
ST = pd.read_csv(D / "within_paper_contrasts_strict_window01.csv").set_index(["factor", "level"]); LO = pd.read_csv(D / "within_paper_contrasts_window01.csv").set_index(["factor", "level"])
rows = []
for key, ix in [("Enc", ("encapsulation", "True")), ("Fam", ("A_site", "FA; MA")), ("Pb", ("B_site", "lead_mixed"))]:
    a, b_ = LO.loc[ix], ST.loc[ix]; W[f"rStrict{key}"] = f"{b_.delta_retention_pp:+.0f}"; W[f"rStrict{key}Lo"] = f"{b_.ci_lo:+.0f}"; W[f"rStrict{key}Hi"] = f"{b_.ci_hi:+.0f}"; W[f"rStrict{key}K"] = int(b_.papers); W[f"rLoose{key}"] = f"{a.delta_retention_pp:+.0f}"
NMF = {"A_site": "A-site", "B_site": "B-site", "HTL": "HTL", "encapsulation": "Encapsulation", "deposition": "Deposition", "backcontact": "Back contact", "architecture": "Architecture"}
LVN = {"FA; MA": "FA+MA vs MA", "Cs; FA; MA": "Cs+FA+MA vs MA", "Cs": "Cs vs MA", "lead_mixed": "doped/alloyed Pb vs Pb only", "True": "encapsulated vs none", "Spin-coating >> Spin-coating": "two-step vs one-step spin-coating", "P3HT": "P3HT vs Spiro-MeOTAD", "PTAA": "PTAA vs Spiro-MeOTAD", "Carbon": "carbon vs Au", "pin": "p-i-n vs n-i-p"}
for ix in LO.index:
    if LO.loc[ix].papers < 3: continue
    a = LO.loc[ix]; nm_ = f"{NMF.get(ix[0], ix[0])}: {LVN.get(ix[1], ix[1])}"
    if ix in ST.index: b_ = ST.loc[ix]; rows.append(f"{nm_} & {int(a.papers)} & {a.delta_retention_pp:+.1f} ({a.ci_lo:+.0f} to {a.ci_hi:+.0f}) & {int(b_.papers)} & {b_.delta_retention_pp:+.1f} ({b_.ci_lo:+.0f} to {b_.ci_hi:+.0f}) \\\\")
    else: rows.append(f"{nm_} & {int(a.papers)} & {a.delta_retention_pp:+.1f} ({a.ci_lo:+.0f} to {a.ci_hi:+.0f}) & fewer than 3 & -- \\\\")
(FR / "table_rev_strict.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
LF = pd.DataFrame(json.load(open(D / "lead_free_contrast.json"))); W["rLfPapers"] = int(LF[LF.label == "all lead-free"].papers_contributing.max()); W["rLfCells"] = int(LF[LF.label == "all lead-free"].cells_free.max()); W["rLfSnPapers"] = int(LF[LF.label == "Sn-based only"].papers_contributing.max())
stab = pd.read_parquet(D / "stability_table.parquet"); yy = 100 * stab.retention_end.clip(0, 1.2); varY = float(yy.var()); W["rRetSd"] = f"{np.sqrt(varY):.0f}"
for key, mask, delta in [("Enc", stab.Encapsulation.astype(str) == "True", float(MA.loc[("encapsulation", "True"), "re_mean"])), ("Fam", stab.a_site.astype(str) == "FA; MA", float(MA.loc[("A_site", "FA; MA"), "re_mean"])), ("Pb", stab.b_site_class == "lead_mixed", float(MA.loc[("B_site", "lead_mixed"), "re_mean"]))]:
    f_ = float(mask.mean()); W[f"rVar{key}Share"] = f"{100*f_:.0f}"; W[f"rVar{key}Expl"] = f"{100*f_*(1-f_)*delta**2/varY:.1f}"

# ---- tuned author-aware, seeds, composition ablation, strata, ranking baselines ----
import glob
def load_r3(pat): return [r for f in sorted(glob.glob(str(D / pat))) if "_partial" not in f for r in json.load(open(f))]
RA = pd.DataFrame(load_r3("stability_tuned_robustness_author_*.json"))
if len(RA):
    TRr = RA[RA.get("task").isna()]; TGr = RA[RA.task == "gain"]; rows = []
    for pr_, lb in [("last_author", "last author"), ("purged_any_author", "purged any author")]:
        for m in ["XGBoost", "LightGBM", "RandomForest", "Blend (XGB+LGBM+RF)"]:
            b = TRr[(TRr.protocol == pr_) & (TRr.model == m) & (TRr.features == "base")]; e = TRr[(TRr.protocol == pr_) & (TRr.model == m) & (TRr.features == "+process+stack")]; g = TGr[(TGr.protocol == pr_) & (TGr.model == m)]
            if b.empty or e.empty or g.empty: continue
            rows.append(f"{lb} & {m} & {b.r2.iloc[0]:.3f} & {e.r2.iloc[0]:.3f} & {g.gain.iloc[0]:+.3f} [{g.gain_ci.iloc[0][0]:+.3f}, {g.gain_ci.iloc[0][1]:+.3f}] \\\\")
        tag = "Last" if pr_ == "last_author" else "Purge"
        bb = TRr[(TRr.protocol == pr_) & (TRr.features == "base") & (TRr.model != "Blend (XGB+LGBM+RF)")]; W[f"rAut{tag}BaseLo"] = f3(bb.r2.min()); W[f"rAut{tag}BaseHi"] = f3(bb.r2.max())
        gg = TGr[(TGr.protocol == pr_) & (TGr.model != "Blend (XGB+LGBM+RF)")]; W[f"rAut{tag}GainLo"] = f3(gg.gain.min()); W[f"rAut{tag}GainHi"] = f3(gg.gain.max()); W[f"rAut{tag}NSig"] = int(sum(1 for c in gg.gain_ci if c[0] > 0)); W[f"rAut{tag}N"] = len(gg)
        bl = TGr[(TGr.protocol == pr_) & (TGr.model == "Blend (XGB+LGBM+RF)")]
        if len(bl): W[f"rAut{tag}BlendGain"] = f3(bl.gain.iloc[0]); W[f"rAut{tag}BlendGainLo"] = f3(bl.gain_ci.iloc[0][0]); W[f"rAut{tag}BlendGainHi"] = f3(bl.gain_ci.iloc[0][1])
    (FR / "table_rev_tuned_authors.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
RS = pd.DataFrame(load_r3("stability_tuned_robustness_seed_*.json"))
if len(RS) and (D / "stability_tuned_headline_paper.json").exists():
    TH0 = pd.DataFrame(json.load(open(D / "stability_tuned_headline_paper.json"))); rows = []; gains = []; bases = []
    b0 = TH0[(TH0.get("task").isna()) & (TH0.model == "XGBoost") & (TH0.features == "base")].r2.iloc[0]; e0 = TH0[(TH0.get("task").isna()) & (TH0.model == "XGBoost") & (TH0.features == "+process+stack")].r2.iloc[0]
    rows.append(f"0 (grouped folds) & {b0:.3f} & {e0:.3f} & {e0-b0:+.3f} \\\\"); gains.append(e0 - b0); bases.append(b0)
    for k in sorted(RS.protocol.unique()):
        x = RS[(RS.protocol == k) & (RS.model == "XGBoost")]
        if x[x.get("task").isna()].empty: continue
        b = x[(x.get("task").isna()) & (x.features == "base")].r2.iloc[0]; e = x[(x.get("task").isna()) & (x.features == "+process+stack")].r2.iloc[0]; rows.append(f"{k.replace('paper_seed', '')} (random folds) & {b:.3f} & {e:.3f} & {e-b:+.3f} \\\\"); gains.append(e - b); bases.append(b)
    (FR / "table_rev_seeds.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    W["rSeedN"] = len(gains); W["rSeedBaseLo"] = f3(min(bases)); W["rSeedBaseHi"] = f3(max(bases)); W["rSeedGainLo"] = f3(min(gains)); W["rSeedGainHi"] = f3(max(gains)); W["rSeedGainMean"] = f3(np.mean(gains)); W["rSeedGainSd"] = f3(np.std(gains, ddof=1))
RC = pd.DataFrame(load_r3("stability_tuned_robustness_comp_*.json"))
if len(RC):
    rows = []; TRc = RC[RC.get("task").isna()]; TGc = RC[RC.task == "gain"]
    for pr_ in ["paper", "temporal"]:
        for m in ["XGBoost", "LightGBM", "RandomForest"]:
            b = TRc[(TRc.protocol == pr_) & (TRc.model == m) & (TRc.features == "base")]; e = TRc[(TRc.protocol == pr_) & (TRc.model == m) & (TRc.features == "base without composition")]; g = TGc[(TGc.protocol == pr_) & (TGc.model == m)]
            if b.empty or e.empty or g.empty: continue
            rows.append(f"{pr_} & {m} & {b.r2.iloc[0]:.3f} & {e.r2.iloc[0]:.3f} & {-g.gain.iloc[0]:+.3f} [{-g.gain_ci.iloc[0][1]:+.3f}, {-g.gain_ci.iloc[0][0]:+.3f}] \\\\")
    (FR / "table_rev_composition_tuned.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    for pr_, tag in [("paper", "Paper"), ("temporal", "Temp")]:
        gg = TGc[(TGc.protocol == pr_) & (TGc.model != "Blend (XGB+LGBM+RF)")]; W[f"rCompTun{tag}Lo"] = f3(-gg.gain.max()); W[f"rCompTun{tag}Hi"] = f3(-gg.gain.min()); W[f"rCompTun{tag}NSig"] = int(sum(1 for c in gg.gain_ci if -c[1] > 0))
        W[f"rCompTun{tag}N"] = len(gg)
SB = json.load(open(D / "pairwise_prior_baselines.json"))
for key, nm_ in [("prior", "Prior"), ("more_treated", "More")]:
    for sub, tg in [("all pairs", "All"), ("one coded factor differs", "One"), ("no coded factor differs", "None")]:
        r = SB[f"{key} | {sub}"]; W[f"rBase{nm_}{tg}"] = f"{100*r['acc']:.0f}"; W[f"rBase{nm_}{tg}Lo"] = f"{100*r['lo']:.0f}"; W[f"rBase{nm_}{tg}Hi"] = f"{100*r['hi']:.0f}"
if (D / "stability_ipcw_strata.json").exists():
    ST_ = pd.DataFrame(json.load(open(D / "stability_ipcw_strata.json"))); rows = []
    for h in [500, 1000]:
        for st_ in ST_[(ST_.horizon == h) & (ST_.protocol == "paper")].stratum.unique():
            a = ST_[(ST_.horizon == h) & (ST_.protocol == "paper") & (ST_.stratum == st_) & (ST_.cohort == "known cohort, KM-weighted")]; b = ST_[(ST_.horizon == h) & (ST_.protocol == "paper") & (ST_.stratum == st_) & (ST_.cohort == "landmark subset")]
            rows.append(f"{h} & {st_} & " + (f"{a.auc.iloc[0]:.3f} ({a.n.iloc[0]})" if len(a) else "--") + " & " + (f"{b.auc.iloc[0]:.3f} ({b.n.iloc[0]})" if len(b) else "--") + " \\\\")
    (FR / "table_rev_ipcw_strata.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

MVT = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(D / "sensitivity_multiverse_tuned_c*.csv")))])
for tk, tg in [("reg", "Reg"), ("clf", "Clf")]:
    x = MVT[MVT.task == tk]; W[f"rMvt{tg}N"] = len(x); W[f"rMvt{tg}Med"] = f3(x.delta_enriched.median()); W[f"rMvt{tg}Lo"] = f3(x.delta_enriched.quantile(.25)); W[f"rMvt{tg}Hi"] = f3(x.delta_enriched.quantile(.75)); W[f"rMvt{tg}Pos"] = f"{100*(x.delta_enriched>0).mean():.0f}"
    W[f"rMvt{tg}NfMed"] = f3(x.delta_enriched_noflags.median()); W[f"rMvt{tg}NfPos"] = f"{100*(x.delta_enriched_noflags>0).mean():.0f}"
W["rMvtN"] = len(MVT); W["rMvtCollapse"] = f3(MVT[(MVT.task == "reg") & (MVT.dup == "collapse_replicates")].delta_enriched.median())

# ---- further macros ----
if len(RC):
    for pr_, tag in [("paper", "Paper"), ("temporal", "Temp")]:
        gg = TGc[(TGc.protocol == pr_) & (TGc.model != "Blend (XGB+LGBM+RF)")]; W[f"rCompTun{tag}CiHi"] = f3(max(-c[0] for c in gg.gain_ci)); W[f"rCompTun{tag}PtHi"] = f3(-gg.gain.min())
MAp = pd.read_csv(D / "contrast_meta_analysis.csv"); hw = (MAp.pred_hi - MAp.pred_lo) / 2; W["rMaPredHwMin"] = f"{hw.min():.0f}"; W["rMaPredHwMax"] = f"{hw.max():.0f}"
ee = MAp[(MAp.factor == "encapsulation") & (MAp.level == "True")].iloc[0]; W["rMaEncPredLo"] = f"{ee.pred_lo:+.0f}"; W["rMaEncPredHi"] = f"{ee.pred_hi:+.0f}"
if (D / "stability_ipcw_strata.json").exists():
    ST2 = pd.DataFrame(json.load(open(D / "stability_ipcw_strata.json"))); kn = ST2[(ST2.protocol == "paper") & (ST2.cohort == "known cohort, KM-weighted") & (ST2.stratum != "all")]
    for h in (500, 1000):
        x = kn[kn.horizon == h]; W[f"rStr{h}Lo".replace("500", "fivezerozero").replace("1000", "onezerozerozero")] = f2(x.auc.min()); W[f"rStr{h}Hi".replace("500", "fivezerozero").replace("1000", "onezerozerozero")] = f2(x.auc.max())
        t_ = x[x.stratum.str.startswith("exposure")]; W[f"rStrT{h}Lo".replace("500", "fivezerozero").replace("1000", "onezerozerozero")] = f2(t_.auc.min()); W[f"rStrT{h}Hi".replace("500", "fivezerozero").replace("1000", "onezerozerozero")] = f2(t_.auc.max())
# reduced tuned multiverse table
rows = []
for fac, nm_ in [("outcome", "Outcome"), ("filter", "Cell filter"), ("dup", "Duplicate rule"), ("model", "Model"), ("protocol", "Protocol")]:
    for lv, g in MVT.groupby(fac):
        rg = g[g.task == "reg"]; cf = g[g.task == "clf"]
        rows.append(f"{nm_}: {esc(lv)} & {len(g)} & " + (f"{rg.delta_enriched.median():+.3f}" if len(rg) else "--") + " & " + (f"{100*(rg.delta_enriched>0).mean():.0f}" if len(rg) else "--") + " & " + (f"{rg.delta_enriched_noflags.median():+.3f}" if len(rg) else "--") + " & " + (f"{cf.delta_enriched.median():+.3f}" if len(cf) else "--") + " & " + (f"{100*(cf.delta_enriched>0).mean():.0f}" if len(cf) else "--") + " \\\\")
(FR / "table_rev_mvt.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

if (D / "efficiency_canonical_split.json").exists():
    EC3 = json.load(open(D / "efficiency_canonical_split.json")); W["rEcsGbmFam"] = f3(EC3[0]["gbm_family"]); W["rEcsGbmRand"] = f3(EC3[0]["gbm_random"]); W["rEcsXgbFam"] = f3(EC3[0]["xgb_family"]); W["rEcsXgbRand"] = f3(EC3[0]["xgb_random"])
    W["rEcsReproduced"] = "yes" if len(EC3) > 1 and all(abs(EC3[0][k] - EC3[1][k]) < 1e-9 for k in ("gbm_family", "xgb_family")) else "no"

RZ = pd.DataFrame(load_r3("stability_tuned_robustness_size_*.json"))
if len(RZ):
    bz = RZ[RZ.get("task").isna()]; gz = RZ[RZ.task == "gain"]; rows = []
    for m in ["XGBoost", "LightGBM", "RandomForest", "Blend (XGB+LGBM+RF)"]:
        b_ = bz[(bz.model == m) & (bz.features == "base")].r2; e_ = bz[(bz.model == m) & (bz.features == "+process+stack")].r2; g_ = gz[gz.model == m].gain
        pu = TRr[(TRr.protocol == "purged_any_author") & (TRr.model == m)]; pb = pu[pu.features == "base"].r2.iloc[0]; pe = pu[pu.features == "+process+stack"].r2.iloc[0]
        rows.append(f"{m} & {b_.min():.3f}--{b_.max():.3f} & {pb:.3f} & {e_.min():.3f}--{e_.max():.3f} & {pe:.3f} & {g_.min():+.3f} to {g_.max():+.3f} & {pe-pb:+.3f} \\\\")
    (FR / "table_rev_size.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    m3 = bz[(bz.features == "base") & (bz.model != "Blend (XGB+LGBM+RF)")].r2; W["rSizeBaseLo"] = f3(m3.min()); W["rSizeBaseHi"] = f3(m3.max())
    g3 = gz[gz.model != "Blend (XGB+LGBM+RF)"].gain; W["rSizeGainLo"] = f3(g3.min()); W["rSizeGainHi"] = f3(g3.max())

CF = pd.DataFrame(json.load(open(D / "stability_conformal.json"))); rows = []
for r in CF.itertuples():
    rows.append(f"{r.protocol} & {r.features} & {r.calibration} & {r.coverage:.3f} [{r.coverage_ci[0]:.3f}, {r.coverage_ci[1]:.3f}] & {r.mean_width:.2f} & {r.paper_mean_coverage:.3f} & " + " / ".join(f"{v:.2f}" for v in r.coverage_by_exposure_tertile) + " \\\\")
(FR / "table_rev_conformal.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
pg = CF[CF.calibration == "paper-grouped"]; W["rConfCovLo"] = f"{pg.coverage.min():.2f}"; W["rConfCovHi"] = f"{pg.coverage.max():.2f}"; W["rConfWidthLo"] = f"{pg.mean_width.min():.2f}"; W["rConfWidthHi"] = f"{pg.mean_width.max():.2f}"
W["rConfSpread"] = f"{float(stab.retention_end.clip(0, 1.2).quantile(.95) - stab.retention_end.clip(0, 1.2).quantile(.05)):.2f}"
cl = CF[CF.calibration == "cell-level"]; W["rConfCellCovLo"] = f"{cl.coverage.min():.2f}"; W["rConfCellCovHi"] = f"{cl.coverage.max():.2f}"

with open(FR / "numbers_rev.tex", "w") as o:
    o.write("% Written by make_figures_extra.py\n")
    for k, v in W.items(): o.write(f"\\newcommand{{\\{''.join({'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}.get(c, c) for c in k)}}}{{{v}}}\n")
print(W)
