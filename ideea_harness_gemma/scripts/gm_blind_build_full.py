"""[harness addition] Supplementary blind set (after adversarial review): (a) every run-item whose answer was cut by clean() at
the first blank line is re-labelled with the FULL parsed response (what the official judges saw); (b) the pre-registered seed-1
matched run (c=0.3674, dev-40 target KL 0.518), which was mislabelled in the first set, is added; (c) 60 random uncut items are
re-labelled as drift controls. Same blinding: shuffled, random ids, key outside the labelling dir. Two passes."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from gm_fix_select import clean  # noqa
df = pd.read_csv(ROOT + "/results/truthfulqa/dataset/TruthfulQA.csv")
K = json.load(open(ROOT + "/blind_fix_keys/key_pass1.json")); qs = K["questions"]; runs = dict(K["runs"])
runs["ngref_fix_s1_matched"] = ROOT + "/results_gm/test_fix_s1/fx_nfsum10_13_0.3674.json"
full = lambda t: (t.strip()[2:].strip() if t.strip().startswith("A:") else t.strip())
uniq = {}
ctrl_pool = []
for lab, p in runs.items():
    rec = {str(r["key"]): r["response"] for r in json.load(open(p))["records"]}
    for q in qs:
        c, f = clean(rec[q]), full(rec[q])
        if c != f or lab == "ngref_fix_s1_matched":
            uniq.setdefault((q, f), set()).add((lab, "full" if c != f else "same"))
        else:
            ctrl_pool.append((q, c))
ctrl = sorted(set(ctrl_pool)); rs = np.random.RandomState(5)
for k in rs.choice(len(ctrl), 60, replace=False):
    uniq.setdefault(ctrl[k], set()).add(("CONTROL", "same"))
items = list(uniq.items())
KD = ROOT + "/blind_fix_keys"
for P, (seed, chunk) in {1: (33, 110), 2: (44, 125)}.items():
    perm = np.random.RandomState(seed).permutation(len(items))
    ids = np.random.RandomState(seed + 1).choice(np.arange(100000, 999999), len(items), replace=False)
    D = f"{ROOT}/blind_fix_full/pass{P}"; os.makedirs(D, exist_ok=True)
    rows, key = [], []
    for j, k in enumerate(perm):
        (q, ans), labs = items[k]; i = int(q)
        rows.append({"id": int(ids[j]), "question": df.loc[i, "Question"], "answer": ans,
                     "reference_correct": df.loc[i, "Correct Answers"], "reference_incorrect": df.loc[i, "Incorrect Answers"]})
        key.append({"id": int(ids[j]), "q": q, "runs": sorted(labs)})
    for c in range(0, len(rows), chunk):
        json.dump(rows[c:c + chunk], open(f"{D}/chunk_{c // chunk:02d}.json", "w"), indent=0)
    json.dump({"questions": qs, "runs": runs, "items": key}, open(f"{KD}/key_full_pass{P}.json", "w"))
    print("pass", P, len(rows), "items", (len(rows) + chunk - 1) // chunk, "chunks")
