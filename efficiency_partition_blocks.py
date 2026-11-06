import json
from pathlib import Path
import numpy as np, pandas as pd
D = Path(__file__).resolve().parent / "data"
f = pd.read_parquet(D / "pdp_featurized.parquet", columns=["family_key"]); sizes = f.family_key.value_counts(); big = list(sizes.index[:3])
mp = json.load(open(D / "multipartition_robustness.json")); pr = pd.DataFrame(mp["partitions"]); pr["delta"] = pr.family_disjoint_r2 - pr.random_r2
pr["largest_held_out"] = pr.held_out_families.apply(lambda l: big[0] in l); pr["top3_held_out"] = pr.held_out_families.apply(lambda l: sum(b in l for b in big))
rng = np.random.default_rng(0); out = {"family_sizes_top5": {k: int(v) for k, v in sizes.head(5).items()}, "share_largest_family": float(sizes.iloc[0] / sizes.sum()), "largest_family": big[0], "n_partitions": len(pr)}
def boot(x): x = np.asarray(x); b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
for key, g in pr.groupby("largest_held_out"):
    out["largest_held_out" if key else "largest_kept_in_training"] = dict(partitions=int(len(g)), mean_delta=float(g.delta.mean()), ci=boot(g.delta), share_positive=float((g.delta > 0).mean()), mean_test_frac=float(g.test_frac.mean()), mean_dominant_share=float(g.dominant_family_share_of_test.mean()))
out["range_dominant_share"] = [float(pr.dominant_family_share_of_test.min()), float(pr.dominant_family_share_of_test.max())]
out["corr_delta_testfrac"] = float(np.corrcoef(pr.delta, pr.test_frac)[0, 1]); out["overall_mean_delta"] = float(pr.delta.mean()); out["overall_ci_partition_bootstrap"] = boot(pr.delta)
json.dump(out, open(D / "efficiency_partition_blocks.json", "w"), indent=1); print(json.dumps(out, indent=1))
