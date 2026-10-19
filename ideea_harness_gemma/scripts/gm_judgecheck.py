"""Judge batch-invariance check: re-judge 3 saved generation files with batch size 1 and compare with the stored (batch 16) labels."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from tasks.truthfulqa import TruthfulQA
recs, ref = [], []
for n in ["base", "eucref_0.4", "ng_0.4", "iti_5"]:
    d = json.load(open(f"results_gm/test_grid/{n}.json")); j = json.load(open(f"results_gm/test_grid/{n}.judged.json"))["judgements"]
    for r in d["records"][:150]:
        recs.append({"key": len(recs), "meta": r["meta"], "response": r["response"]}); ref.append(j[str(r["key"])])
res = TruthfulQA().judge(recs, 1)["judgements"]
dt = sum(res[str(i)]["truth"] != ref[i]["truth"] for i in range(len(recs))); di = sum(res[str(i)]["info"] != ref[i]["info"] for i in range(len(recs)))
print(json.dumps({"n": len(recs), "truth_flips": dt, "info_flips": di}))
json.dump({"n": len(recs), "truth_flips": dt, "info_flips": di}, open("results_gm/judgecheck.json", "w"))
