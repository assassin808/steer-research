"""[harness addition] Merge strict labels: first blind set (clean()-truncated answers) + supplementary full-answer set.
For every (run, q) whose answer was truncated, the full-answer labels replace the truncated ones; seed-1 matched run comes
from the supplementary set only. Reports both 'truncated' (original) and 'full' (corrected, primary after review) strict TI,
per pass, kappa on unique items, drift on 60 control items, paired contrasts with bootstrap CI. -> strict_merged.json"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from gm_fix_report import bci, kappa  # noqa


def labels(d):
    L = {}
    for f in glob.glob(f"{ROOT}/{d}/labels_*.json"):
        for x in json.load(open(f)):
            L[int(x["id"])] = (int(x["truthful"]), int(x["informative"]))
    return L


orig, full = {}, {}
uniq_k = {"orig": [[], [], []], "full": [[], [], []]}
for P in (1, 2):
    K = json.load(open(f"{ROOT}/blind_fix_keys/key_pass{P}.json")); L = labels(f"blind_fix/pass{P}")
    for it in K["items"]:
        for r in it["runs"]:
            orig.setdefault(r, {}).setdefault(it["q"], {})[P] = L[it["id"]]
    KF = json.load(open(f"{ROOT}/blind_fix_keys/key_full_pass{P}.json")); LF = labels(f"blind_fix_full/pass{P}")
    assert len(LF) == len(KF["items"]), ("missing full labels", P, len(LF), len(KF["items"]))
    for it in KF["items"]:
        for r, kind in it["runs"]:
            full.setdefault(r, {}).setdefault(it["q"], {})[P] = LF[it["id"]]
qs = K["questions"]
# kappa on unique items (each pass labels the same unique set)
def uk(keyf, d):
    a = {}
    for P in (1, 2):
        K_ = json.load(open(f"{ROOT}/blind_fix_keys/{keyf.format(P)}")); L_ = labels(d.format(P))
        for it in K_["items"]:
            a.setdefault((it["q"], json.dumps(it["runs"])), {})[P] = L_[it["id"]]
    v = [x for x in a.values() if 1 in x and 2 in x]
    T = np.array([[x[1][0], x[2][0]] for x in v]); I = np.array([[x[1][1], x[2][1]] for x in v])
    return {"n_unique": len(v), "kappa_truthful": kappa(T[:, 0], T[:, 1]), "kappa_informative": kappa(I[:, 0], I[:, 1]), "kappa_ti": kappa(T[:, 0] * I[:, 0], T[:, 1] * I[:, 1])}
out = {"kappa_unique_first_set": uk("key_pass{}.json", "blind_fix/pass{}"), "kappa_unique_full_set": uk("key_full_pass{}.json", "blind_fix_full/pass{}")}
# drift: control items (uncut answers) labelled in both sets
ctl = full.get("CONTROL", {}); dr = []
key1 = {(it["q"]): it for it in json.load(open(f"{ROOT}/blind_fix_keys/key_full_pass1.json"))["items"] if ["CONTROL", "same"] in it["runs"]}
out["n_control"] = len(ctl)
runs = [r for r in set(orig) | set(full) if r != "CONTROL"]


def score(r, mode):
    T, I = [], []
    for q in qs:
        src = None
        if r in full and q in full[r] and (mode == "full" or r not in orig):
            src = full[r][q]
        elif r in orig and q in orig[r]:
            src = orig[r][q]
        T.append([src[1][0], src[2][0]]); I.append([src[1][1], src[2][1]])
    T, I = np.array(T, float), np.array(I, float)
    return {"truth": T.mean(1), "info": I.mean(1), "ti": (T * I).mean(1), "ti_p1": T[:, 0] * I[:, 0], "ti_p2": T[:, 1] * I[:, 1]}


for mode in ("truncated", "full"):
    S = {r: score(r, mode) for r in runs}
    out[mode] = {r: {"truth": float(s["truth"].mean()), "info": float(s["info"].mean()), "ti": bci(s["ti"]), "ti_p1": float(s["ti_p1"].mean()), "ti_p2": float(s["ti_p2"].mean()),
                     "n_items_relabelled_full": sum(1 for q in qs if r in full and q in full[r])} for r, s in S.items()}
    out[mode + "_contrasts"] = {f"{a}-{b}": {"d": bci(S[a]["ti"] - S[b]["ti"]), "p1": float((S[a]["ti_p1"] - S[b]["ti_p1"]).mean()), "p2": float((S[a]["ti_p2"] - S[b]["ti_p2"]).mean())}
                                for a in runs for b in ("base", "ideea_paper", "iti_sel") if a != b}
# control drift: compare control items' new labels against the first-set labels of the same (q, answer)
first = {}
for P in (1, 2):
    K = json.load(open(f"{ROOT}/blind_fix_keys/key_pass{P}.json")); L = labels(f"blind_fix/pass{P}")
    items = {}
    for f in glob.glob(f"{ROOT}/blind_fix/pass{P}/chunk_*.json"):
        for x in json.load(open(f)):
            items[x["id"]] = x["answer"]
    for it in K["items"]:
        first.setdefault((it["q"], items[it["id"]]), {})[P] = L[it["id"]]
cd = []
for P in (1, 2):
    KF = json.load(open(f"{ROOT}/blind_fix_keys/key_full_pass{P}.json")); LF = labels(f"blind_fix_full/pass{P}")
    items = {}
    for f in glob.glob(f"{ROOT}/blind_fix_full/pass{P}/chunk_*.json"):
        for x in json.load(open(f)):
            items[x["id"]] = x["answer"]
    for it in KF["items"]:
        if ["CONTROL", "same"] in it["runs"] and (it["q"], items[it["id"]]) in first:
            o = first[(it["q"], items[it["id"]])][P]; n = LF[it["id"]]
            cd.append((o[0] * o[1], n[0] * n[1], o[1], n[1]))
cd = np.array(cd, float)
out["control_drift"] = {"n": len(cd), "ti_old": float(cd[:, 0].mean()), "ti_new": float(cd[:, 1].mean()), "agree_ti": float((cd[:, 0] == cd[:, 1]).mean()), "agree_info": float((cd[:, 2] == cd[:, 3]).mean())}
json.dump(out, open(ROOT + "/strict_merged.json", "w"), indent=1)
for mode in ("truncated", "full"):
    print("==", mode)
    for r, v in sorted(out[mode].items(), key=lambda x: -x[1]["ti"][0]):
        print("  %-22s T %.3f I %.3f TI %.3f [%.3f,%.3f] p1 %.3f p2 %.3f relabelled %d" % (r, v["truth"], v["info"], v["ti"][0], v["ti"][2], v["ti"][3], v["ti_p1"], v["ti_p2"], v["n_items_relabelled_full"]))
    for k, v in out[mode + "_contrasts"].items():
        if k.split("-")[0] in ("ngref_fix", "ngref_fix_s1", "ngref_fix_s1_matched", "eucref_fix", "abst_posthoc", "ng_supervised", "ideea_paper", "ngref_old"):
            print("  %-34s %+.3f [%+.3f,%+.3f] p1 %+.3f p2 %+.3f" % (k, v["d"][0], v["d"][2], v["d"][3], v["p1"], v["p2"]))
print({k: out[k] for k in ("kappa_unique_first_set", "kappa_unique_full_set", "control_drift")})
