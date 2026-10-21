"""[harness addition] Strict blind set for the bridge (prereg §5). Runs: bridge_strict_runs.json {label: generations path}.
Same 150 test questions as the earlier strict set (blind_fix_keys/key_pass1.json). FULL parsed answers (no clean()).
Dedup by (question, answer); 60 drift-control items drawn from the earlier full-answer set (blind_fix_full, labels known);
shuffled, random ids, key in blind_bridge_keys/ (outside the labelling dir); 2 passes, different shuffles and chunk sizes."""
import glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
df = pd.read_csv(ROOT + "/results/truthfulqa/dataset/TruthfulQA.csv")
qs = json.load(open(ROOT + "/blind_fix_keys/key_pass1.json"))["questions"]
runs = json.load(open(ROOT + "/bridge_strict_runs.json"))
full = lambda t: (t.strip()[2:].strip() if t.strip().startswith("A:") else t.strip())
uniq = {}
for lab, p in runs.items():
    rec = {str(r["key"]): r["response"] for r in json.load(open(p))["records"]}
    for q in qs:
        uniq.setdefault((q, full(rec[q])), set()).add(lab)
# drift controls: items of the earlier full-answer set (both passes labelled), not already in the pool
old = []
for f in sorted(glob.glob(ROOT + "/blind_fix_full/pass1/chunk_*.json")):
    for x in json.load(open(f)):
        old.append(x)
K1 = {it["id"]: it["q"] for it in json.load(open(ROOT + "/blind_fix_keys/key_full_pass1.json"))["items"]}
cand = sorted({(K1[x["id"]], x["answer"]) for x in old} - set(uniq))
rs = np.random.RandomState(11)
for k in rs.choice(len(cand), min(60, len(cand)), replace=False):
    uniq.setdefault(cand[k], set()).add("CONTROL")
items = list(uniq.items())
KD = ROOT + "/blind_bridge_keys"; os.makedirs(KD, exist_ok=True)
for P, (seed, chunk) in {1: (71, 150), 2: (83, 170)}.items():
    perm = np.random.RandomState(seed).permutation(len(items))
    ids = np.random.RandomState(seed + 1).choice(np.arange(100000, 999999), len(items), replace=False)
    D = f"{ROOT}/blind_bridge/pass{P}"; os.makedirs(D, exist_ok=True)
    rows, key = [], []
    for j, k in enumerate(perm):
        (q, ans), labs = items[k]; i = int(q)
        rows.append({"id": int(ids[j]), "question": df.loc[i, "Question"], "answer": ans,
                     "reference_correct": df.loc[i, "Correct Answers"], "reference_incorrect": df.loc[i, "Incorrect Answers"]})
        key.append({"id": int(ids[j]), "q": q, "answer": ans, "runs": sorted(labs)})
    for c in range(0, len(rows), chunk):
        json.dump(rows[c:c + chunk], open(f"{D}/chunk_{c // chunk:02d}.json", "w"), indent=0)
    json.dump({"questions": qs, "runs": runs, "items": key}, open(f"{KD}/key_pass{P}.json", "w"))
    print("pass", P, len(rows), "items", (len(rows) + chunk - 1) // chunk, "chunks")
