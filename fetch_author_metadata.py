import json, time, sys, subprocess, urllib.parse
from pathlib import Path
import pandas as pd
P = Path(__file__).resolve().parent; D = P / "data"
raw = pd.read_csv(D / "perovskite_raw.csv", usecols=["Ref_DOI_number"], low_memory=False)
dois = sorted({str(x).strip().lower() for x in raw.Ref_DOI_number.dropna() if str(x).startswith("10.")})
out_path = D / "paper_authors_openalex.json"
res = json.load(open(out_path)) if out_path.exists() else {}
todo = [x for x in dois if x not in res]
print(len(dois), "dois;", len(todo), "to fetch", flush=True)
for i in range(0, len(todo), 40):
    batch = todo[i:i + 40]
    url = "https://api.openalex.org/works?filter=doi:" + "|".join(urllib.parse.quote(x, safe="") for x in batch) + "&per-page=50&select=doi,authorships"
    for attempt in range(4):
        try:
            o = subprocess.run(["curl", "-s", "-m", "60", "-A", "perovskite-ml-metadata-client", url], capture_output=True, text=True).stdout
            data = json.loads(o); break
        except Exception as e:
            print("retry", attempt, e, flush=True); time.sleep(5 * (attempt + 1)); data = None
    if data is None: continue
    got = set()
    for w in data.get("results", []):
        doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
        res[doi] = [{"aid": a["author"].get("id"), "name": a["author"].get("display_name"), "inst": [x.get("id") for x in a.get("institutions", [])], "pos": a.get("author_position")} for a in w.get("authorships", [])]
        got.add(doi)
    for x in batch:
        if x not in got and x not in res: res[x] = None
    if (i // 40) % 10 == 0:
        json.dump(res, open(out_path, "w")); print(i, len(res), flush=True)
    time.sleep(0.2)
json.dump(res, open(out_path, "w")); print("done", len(res), sum(v is not None for v in res.values()))
