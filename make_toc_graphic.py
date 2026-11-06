# Table-of-contents graphic: headline results of the manuscript.
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.text import Text
FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
OI = dict(blue="#0072B2", orange="#E69F00", verm="#D55E00", grey="#777777", sky="#56B4E9")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "figure.constrained_layout.use": True, "mathtext.fontset": "dejavusans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(4.5, 2.43))
fig.suptitle("Perovskite solar cell stability:\nwhat the literature data can support", fontsize=9, fontweight="bold")
# (a) R2 on unseen papers
x = [0, 1, 2]; v = [0.04, 0.12, 0.03]; cols = [OI["grey"], OI["blue"], OI["sky"]]
for xi, vi, c, h in zip(x, v, cols, ["", "", "//"]): a.bar(xi, vi, 0.7, color=c, hatch=h, edgecolor="k", lw=0.5)
for xi, t in zip(x, ["0.04", "0.12", "+0.03"]): a.text(xi, v[xi] + 0.004, t, ha="center", va="bottom")
a.set_xticks(x); a.set_xticklabels(["Compo-\nsition only\n(ceiling)", "Tuned\nmodels", "Process,\nstack\ndetails"]); a.set_ylim(0, 0.16); a.set_yticks([0, 0.05, 0.1, 0.15]); a.set_yticklabels(["0", "0.05", "0.10", "0.15"]); a.set_ylabel("$R^2$ on unseen papers")
a.set_title(r"$\mathbf{(a)}$", loc="left", fontsize=9)
# (b) ranking of pairs within one paper
b.bar(0, 65, 0.7, color=OI["blue"], edgecolor="k", lw=0.5); b.bar(0, 4, 0.7, bottom=65, color=OI["sky"], edgecolor="k", lw=0.5)
b.bar(1, 82, 0.7, color=OI["blue"], edgecolor="k", lw=0.5); b.bar(2, 56, 0.7, color="white", hatch="//", edgecolor=OI["grey"], lw=0.5)
b.axhline(50, color=OI["verm"], ls="--", lw=1)
b.text(0, 71, "65–69%", ha="center", va="bottom"); b.text(1, 84, "82%", ha="center", va="bottom"); b.text(2, 58, "56–57%\n(near chance)", ha="center", va="bottom")
b.set_xticks([0, 1, 2]); b.set_xticklabels(["All\npairs", "One\nfactor\ndiffers", "No\nfactor\ndiffers"]); b.set_ylim(0, 112); b.set_yticks([0, 50, 100]); b.set_ylabel("Pairs ranked correctly (%)")
b.set_title(r"$\mathbf{(b)}$", loc="left", fontsize=9)
small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t.get_fontsize() < 8]
assert not small, small
fig.savefig(FIG / "toc_graphic.pdf"); fig.savefig(FIG / "toc_graphic.png", dpi=300)
