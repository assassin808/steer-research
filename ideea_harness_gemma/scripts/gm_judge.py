"""[harness addition] Bulk official-judge scoring of saved generation files (results_gm/<tag>/*.json), judges loaded
one at a time (IDEEA prompts and rule, see tasks/truthfulqa.py). Writes <file>.judged.json with per-item truth/info.
usage: gm_judge.py FILE [FILE ...]   (skips files whose .judged.json exists)
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from tasks.truthfulqa import TruthfulQA  # noqa

files = [f for f in sys.argv[1:] if not os.path.exists(f.replace(".json", ".judged.json")) and not f.endswith(".judged.json")]
print(len(files), "files to judge", flush=True)
if files:
    recs, owner = [], []
    data = {}
    for f in files:
        d = json.load(open(f)); data[f] = d
        for r in d["records"]:
            owner.append((f, str(r["key"])))
            recs.append({"key": len(recs), "meta": r["meta"], "response": r["response"]})
    # deduplicate identical (question, answer) pairs to save judge calls
    uniq, idx = {}, []
    for r in recs:
        k = (r["meta"]["question"], r["response"])
        if k not in uniq:
            uniq[k] = {"key": len(uniq), "meta": r["meta"], "response": r["response"]}
        idx.append(uniq[k]["key"])
    print(len(recs), "items,", len(uniq), "unique", flush=True)
    res = TruthfulQA().judge(list(uniq.values()), 16)["judgements"]
    per = {f: {} for f in files}
    for (f, k), u in zip(owner, idx):
        j = res[str(u)]
        per[f][k] = {"truth": j["truth"], "info": j["info"], "truth_pyes_top": j["truth_pyes_top"], "info_pyes_top": j["info_pyes_top"]}
    for f in files:
        v = per[f]
        t = [x["truth"] for x in v.values()]; i = [x["info"] for x in v.values()]
        out = {"generations_path": f, "n": len(v), "truth_rate": sum(t) / len(t), "info_rate": sum(i) / len(i),
               "truth_and_info_rate": sum(a and b for a, b in zip(t, i)) / len(t), "judgements": v}
        json.dump(out, open(f.replace(".json", ".judged.json"), "w"))
        print(os.path.basename(f), out["truth_rate"], out["info_rate"], out["truth_and_info_rate"], flush=True)
