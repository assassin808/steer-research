"""[harness addition] Build the blind strict-label set (prereg section 5): 150 random test questions (seed 0) x runs,
deduplicated by (question, answer) within a pass, shuffled and renumbered with a pass-specific seed, chunked.
Labelling dir: blind_fix/pass{P}/chunk_{k}.json (no run names, no keys). Key: blind_fix_keys/ (outside the labelling dir).
usage: gm_blind_build.py LABEL=path/to/run.json [...]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from gm_fix_select import clean  # noqa
import pandas as pd

df = pd.read_csv(ROOT + "/results/truthfulqa/dataset/TruthfulQA.csv")
runs = dict(a.split("=", 1) for a in sys.argv[1:])
recs = {lab: {str(r["key"]): clean(r["response"]) for r in json.load(open(p))["records"]} for lab, p in runs.items()}
keys = sorted(next(iter(recs.values())), key=int)
assert all(sorted(v, key=int) == keys for v in recs.values())
qs = sorted(np.random.RandomState(0).choice(keys, 150, replace=False).tolist(), key=int)
KD = ROOT + "/blind_fix_keys"; os.makedirs(KD, exist_ok=True)
for P, (seed, chunk) in {1: (11, 120), 2: (22, 135)}.items():
    uniq = {}
    for lab in runs:
        for q in qs:
            uniq.setdefault((q, recs[lab][q]), []).append(lab)
    items = list(uniq.items())
    perm = np.random.RandomState(seed).permutation(len(items))
    ids = np.random.RandomState(seed + 1).choice(np.arange(10000, 99999), len(items), replace=False)
    D = f"{ROOT}/blind_fix/pass{P}"; os.makedirs(D, exist_ok=True)
    key, rows = [], []
    for j, k in enumerate(perm):
        (q, ans), labs = items[k]
        i = int(q)
        rows.append({"id": int(ids[j]), "question": df.loc[i, "Question"], "answer": ans,
                     "reference_correct": df.loc[i, "Correct Answers"], "reference_incorrect": df.loc[i, "Incorrect Answers"]})
        key.append({"id": int(ids[j]), "q": q, "runs": labs})
    for c in range(0, len(rows), chunk):
        json.dump(rows[c:c + chunk], open(f"{D}/chunk_{c // chunk:02d}.json", "w"), indent=0)
    json.dump({"questions": qs, "runs": runs, "items": key}, open(f"{KD}/key_pass{P}.json", "w"))
    print("pass", P, len(rows), "unique items,", (len(rows) + chunk - 1) // chunk, "chunks")
