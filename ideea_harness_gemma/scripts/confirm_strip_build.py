"""[harness addition, POST HOC exploratory] IDEEA s0 matched answers that start with 'I have no comment' -> prefix removed, relabelled blind
under condition (b) together with an equal number of already-labelled condition-(b) control items (drift check). Key in blind_confirm_keys/."""
import json, os, re
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(ROOT + "/results/truthfulqa/dataset/TruthfulQA.csv")
S = json.load(open(ROOT + "/confirm_specs.json"))["runs"]
full = lambda t: (t.strip()[2:].strip() if t.strip().startswith("A:") else t.strip())
PRE = re.compile(r"^\s*i have no comment\s*[.,;:!]?\s*", re.I)
rec = {str(r["key"]): r["response"] for r in json.load(open(ROOT + "/" + S["IDEEA_s0|mid"]["path"]))["records"]}
items = []
for q, t in rec.items():
    a = full(t)
    if PRE.match(a):
        s = PRE.sub("", a, count=1).strip()
        if s:
            items.append({"q": q, "answer": s, "kind": "stripped", "orig": a})
Kb = json.load(open(ROOT + "/blind_confirm_keys/key_b.json"))["items"]
rs = np.random.RandomState(505)
ctrl = [Kb[j] for j in rs.choice(len(Kb), len(items), replace=False)]
items += [{"q": c["q"], "answer": c["answer"], "kind": "control", "orig_id": c["id"]} for c in ctrl]
perm = np.random.RandomState(506).permutation(len(items)); ids = np.random.RandomState(507).choice(np.arange(100000, 999999), len(items), replace=False)
D = ROOT + "/blind_confirm/strip"; os.makedirs(D, exist_ok=True)
rows, key = [], []
for j, k in enumerate(perm):
    it = items[k]; rows.append({"id": int(ids[j]), "question": df.loc[int(it["q"]), "Question"], "answer": it["answer"]}); key.append({"id": int(ids[j]), **it})
nch = int(np.ceil(len(rows) / 200)); b = np.linspace(0, len(rows), nch + 1).astype(int)
for c in range(nch):
    json.dump(rows[b[c]:b[c + 1]], open(f"{D}/chunk_{c:02d}.json", "w"), indent=0)
json.dump({"items": key}, open(ROOT + "/blind_confirm_keys/key_strip.json", "w"))
print(len(items), "items", nch, "chunks", sum(i["kind"] == "stripped" for i in items), "stripped")
