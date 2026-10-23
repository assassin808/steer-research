"""[harness addition] Strict blind sets for confirm-grad (prereg results/confirm_grad_prereg.md §3).
Condition a (refs shown): matched-level runs. Condition b (no refs): matched-level + off-level (lo/hi) runs of the orig/s0 arms.
All 395 test q, FULL parsed answers, dedup by (question, answer) within a condition, shuffled, random ids (independent per condition),
chunks of ~220, key in blind_confirm_keys/ (outside the labelling dir). Retest: 300 random unique matched items per condition, rechunked."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
df = pd.read_csv(ROOT + "/results/truthfulqa/dataset/TruthfulQA.csv")
S = json.load(open(ROOT + "/confirm_specs.json"))["runs"]
full = lambda t: (t.strip()[2:].strip() if t.strip().startswith("A:") else t.strip())
OFF_ARMS = ("IDEEA_s0", "ITI", "C_orig", "H_orig", "K_orig")
mid = [k for k in S if k.endswith("|mid") or k == "base|mid"]
off = [f"{a}|{lv}" for a in OFF_ARMS for lv in ("lo", "hi")]
recs = {}
for k in mid + off:
    recs[k] = {str(r["key"]): r["response"] for r in json.load(open(ROOT + "/" + S[k]["path"]))["records"]}
qs = sorted(recs["base|mid"], key=int)
assert all(sorted(v, key=int) == qs for v in recs.values())
KD = ROOT + "/blind_confirm_keys"; os.makedirs(KD, exist_ok=True)


def build(cond, runs, seed, chunk, sub=None, tag=None):
    uniq = {}
    for k in runs:
        for q in qs:
            uniq.setdefault((q, full(recs[k][q])), set()).add(k)
    items = sorted(uniq.items(), key=lambda x: (int(x[0][0]), x[0][1]))
    if sub is not None:
        items = [items[j] for j in sorted(np.random.RandomState(seed + 7).choice(len(items), sub, replace=False))]
    perm = np.random.RandomState(seed).permutation(len(items))
    ids = np.random.RandomState(seed + 1).choice(np.arange(100000, 999999), len(items), replace=False)
    D = f"{ROOT}/blind_confirm/{tag or cond}"; os.makedirs(D, exist_ok=True)
    rows, key = [], []
    for j, k in enumerate(perm):
        (q, ans), labs = items[k]; i = int(q)
        r = {"id": int(ids[j]), "question": df.loc[i, "Question"], "answer": ans}
        if cond == "a":
            r.update({"reference_correct": df.loc[i, "Correct Answers"], "reference_incorrect": df.loc[i, "Incorrect Answers"]})
        rows.append(r); key.append({"id": int(ids[j]), "q": q, "answer": ans, "runs": sorted(labs)})
    nch = int(np.ceil(len(rows) / chunk)); bounds = np.linspace(0, len(rows), nch + 1).astype(int)
    for c in range(nch):
        json.dump(rows[bounds[c]:bounds[c + 1]], open(f"{D}/chunk_{c:02d}.json", "w"), indent=0)
    json.dump({"questions": qs, "runs": {k: S[k] for k in runs}, "items": key}, open(f"{KD}/key_{tag or cond}.json", "w"))
    print(tag or cond, len(rows), "items", nch, "chunks")
    return items


if __name__ == "__main__":
    build("a", mid, 101, 220)
    build("b", mid + off, 202, 220)
    build("a", mid, 303, 150, sub=300, tag="a_rt")
    build("b", mid, 404, 150, sub=300, tag="b_rt")
