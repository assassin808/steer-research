"""[harness addition] Merge bridge strict labels -> bridge_strict.json (per-run strict T/I/TI with bootstrap CI, per-item arrays
for paired steps, per-pass rates, kappa between passes on unique items, drift on 60 controls vs the earlier full-answer labels)."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from gm_fix_report import bci, kappa  # noqa


def labels(d):
    L = {}
    for f in glob.glob(f"{d}/labels_*.json"):
        for x in json.load(open(f)):
            L[int(x["id"])] = (int(x["truthful"]), int(x["informative"]))
    return L


lab = {}
uk = {}
for P in (1, 2):
    K = json.load(open(f"{ROOT}/blind_bridge_keys/key_pass{P}.json")); L = labels(f"{ROOT}/blind_bridge/pass{P}")
    miss = [it["id"] for it in K["items"] if it["id"] not in L]
    assert not miss, (P, len(miss), "missing labels")
    for it in K["items"]:
        uk.setdefault((it["q"], it["answer"]), {})[P] = L[it["id"]]
        for r in it["runs"]:
            lab.setdefault(r, {}).setdefault(it["q"], {})[P] = L[it["id"]]
qs = K["questions"]
v = list(uk.values())
T = np.array([[x[1][0], x[2][0]] for x in v]); I = np.array([[x[1][1], x[2][1]] for x in v])
out = {"n_unique": len(v), "kappa_truthful": kappa(T[:, 0], T[:, 1]), "kappa_informative": kappa(I[:, 0], I[:, 1]),
       "kappa_ti": kappa(T[:, 0] * I[:, 0], T[:, 1] * I[:, 1]), "runs": {}, "item_ti": {}, "item_ti_p1": {}, "item_ti_p2": {}}
for r, d in lab.items():
    if r == "CONTROL":
        continue
    t = np.array([[d[q][1][0], d[q][2][0]] for q in qs], float); i = np.array([[d[q][1][1], d[q][2][1]] for q in qs], float)
    ti = (t * i).mean(1)
    out["runs"][r] = {"truth": float(t.mean()), "info": float(i.mean()), "ti": bci(ti), "ti_p1": float((t[:, 0] * i[:, 0]).mean()), "ti_p2": float((t[:, 1] * i[:, 1]).mean())}
    out["item_ti"][r] = ti.tolist(); out["item_ti_p1"][r] = (t[:, 0] * i[:, 0]).tolist(); out["item_ti_p2"][r] = (t[:, 1] * i[:, 1]).tolist()
# drift: controls vs earlier full-answer labels of the same (q, answer)
old = {}
for P in (1, 2):
    KF = json.load(open(f"{ROOT}/blind_fix_keys/key_full_pass{P}.json")); LF = labels(f"{ROOT}/blind_fix_full/pass{P}")
    ans = {}
    for f in glob.glob(f"{ROOT}/blind_fix_full/pass{P}/chunk_*.json"):
        for x in json.load(open(f)):
            ans[x["id"]] = x["answer"]
    for it in KF["items"]:
        old.setdefault((it["q"], ans[it["id"]]), {})[P] = LF[it["id"]]
cd = []
for (q, a), x in uk.items():
    if (q, a) in old and len(old[(q, a)]) == 2 and any("CONTROL" in it["runs"] for it in K["items"] if it["q"] == q and it["answer"] == a):
        o = old[(q, a)]
        cd.append([np.mean([o[p][0] * o[p][1] for p in (1, 2)]), np.mean([x[p][0] * x[p][1] for p in (1, 2)])])
cd = np.array(cd)
out["control_drift"] = {"n": len(cd), "ti_old": float(cd[:, 0].mean()), "ti_new": float(cd[:, 1].mean()), "agree": float((np.abs(cd[:, 0] - cd[:, 1]) < 0.26).mean())} if len(cd) else None
json.dump(out, open(ROOT + "/bridge_strict.json", "w"), indent=1)
for r, s in sorted(out["runs"].items(), key=lambda x: -x[1]["ti"][0]):
    print("%-12s T %.3f I %.3f TI %.3f [%.3f,%.3f] p1 %.3f p2 %.3f" % (r, s["truth"], s["info"], s["ti"][0], s["ti"][2], s["ti"][3], s["ti_p1"], s["ti_p2"]))
print({k: out[k] for k in ("n_unique", "kappa_truthful", "kappa_informative", "kappa_ti", "control_drift")})
