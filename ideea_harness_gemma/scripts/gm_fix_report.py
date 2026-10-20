"""[harness addition] Report for the ngref fix + fair comparison (prereg results/ngref_fix_prereg.md).
Inputs: results_gm/test_grid (old runs), results_gm/test_fix{,_s1,_s2} (new), results_gm/mc*, fix_selection.json,
blind_fix/pass*/labels_*.json + blind_fix_keys/. Output: fix_report.json (+ prints)."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from gm_fix_select import clean, is_ref  # noqa
rng = np.random.RandomState(0)
NEG = re.compile(r"\b(not|no|never|none|nothing|cannot)\b|n't\b", re.I)


def degen(t):
    sents = [s for s in re.split(r"[.!?;]", t) if s.strip()]
    dn = any(len(NEG.findall(s)) >= 2 for s in sents)
    w = t.lower().split(); tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return dn, len(tri) != len(set(tri))


def load(D, sfx=""):
    runs = {}
    for f in sorted(glob.glob(D + "/*.json")):
        if f.endswith(".judged.json") or not os.path.exists(f.replace(".json", ".judged.json")):
            continue
        g, jd = json.load(open(f)), json.load(open(f.replace(".json", ".judged.json")))["judgements"]
        keys = [str(r["key"]) for r in g["records"]]
        resp = [clean(r["response"]) for r in g["records"]]
        t = np.array([jd[k]["truth"] for k in keys], float); i = np.array([jd[k]["info"] for k in keys], float)
        dg = np.array([degen(x) for x in resp], float)
        runs[g["summary"]["spec"] + sfx] = {"path": f, "keys": keys, "truth": t, "info": i, "ti": t * i, "refusal": np.array([is_ref(x) for x in resp], float),
            "nwords": np.array([len(x.split()) for x in resp], float), "dneg": dg[:, 0], "rep3": dg[:, 1],
            "hitlim": np.array([r["ntok"] >= 50 for r in g["records"]], float), "kl": float(np.mean([r["kl_tok"] for r in g["records"]]))}
    return runs


def bci(x, B=2000):
    n = len(x); m = x[rng.randint(0, n, (B, n))].mean(1)
    return float(x.mean()), float(m.std()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def interp(runs, specs, T, metric="ti"):
    pts = sorted([(runs[s]["kl"], s) for s in specs], key=lambda x: x[0])
    for (k0, s0), (k1, s1) in zip(pts[:-1], pts[1:]):
        if k0 <= T <= k1 and k1 > k0:
            w = (np.log(T + 1e-3) - np.log(k0 + 1e-3)) / (np.log(k1 + 1e-3) - np.log(k0 + 1e-3))
            return (1 - w) * runs[s0][metric] + w * runs[s1][metric]
    return None


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def strict(labels_runs):
    """-> per run: dict q -> (t,i) per pass; agreement stats."""
    per = {}; agree = {"truthful": [[], []], "informative": [[], []], "ti": [[], []]}
    item_lab = {}
    for P in (1, 2):
        key = json.load(open(f"{ROOT}/blind_fix_keys/key_pass{P}.json"))
        lab = {}
        for f in sorted(glob.glob(f"{ROOT}/blind_fix/pass{P}/labels_*.json")):
            for x in json.load(open(f)):
                lab[int(x["id"])] = (int(x["truthful"]), int(x["informative"]))
        miss = [it["id"] for it in key["items"] if it["id"] not in lab]
        if miss:
            print(f"pass {P}: {len(miss)} items missing labels", file=sys.stderr)
        for it in key["items"]:
            if it["id"] in lab:
                for r in it["runs"]:
                    per.setdefault(r, {}).setdefault(it["q"], {})[P] = lab[it["id"]]
    qs = key["questions"]
    out = {}
    for r, d in per.items():
        both = [q for q in qs if q in d and 1 in d[q] and 2 in d[q]]
        T = np.array([[d[q][1][0], d[q][2][0]] for q in both], float); I = np.array([[d[q][1][1], d[q][2][1]] for q in both], float)
        out[r] = {"qs": both, "truth": T.mean(1), "info": I.mean(1), "ti": (T * I).mean(1), "ti_p1": T[:, 0] * I[:, 0], "ti_p2": T[:, 1] * I[:, 1],
                  "truth_p": T.mean(0).tolist(), "info_p": I.mean(0).tolist()}
        for k, M in (("truthful", T), ("informative", I), ("ti", T * I)):
            agree[k][0] += M[:, 0].tolist(); agree[k][1] += M[:, 1].tolist()
    ag = {k: {"kappa": kappa(*v), "agreement": float(np.mean(np.array(v[0]) == np.array(v[1]))), "n_run_items": len(v[0])} for k, v in agree.items()}
    return out, ag


if __name__ == "__main__":
    sel = json.load(open(ROOT + "/fix_selection.json"))
    runs = load(ROOT + "/results_gm/test_grid")
    for sfx in ("", "_s1", "_s2"):
        runs.update(load(ROOT + "/results_gm/test_fix" + sfx, "@" + sfx[1:] if sfx else ""))
    base = runs["base"]; R = {"native": {}, "matched": {}, "mc": {}, "strict": {}, "strict_contrasts": {}}
    for s, r in runs.items():
        d = r["ti"] - base["ti"]
        R["native"][s] = {**{k: float(r[k].mean()) for k in ("truth", "info", "ti", "refusal", "nwords", "dneg", "rep3", "hitlim")}, "kl": r["kl"],
                          "d_ti": bci(d)}
    # equal-KL curves (official TI): family -> specs
    fam = {"ideea_default": [s for s in runs if re.fullmatch(r"ideea:[0-9.]+", s)], "iti_default": [s for s in runs if re.fullmatch(r"iti:[0-9.]+", s)],
           "caa13": [s for s in runs if s.startswith("caa:13:")], "ngref_old": [s for s in runs if s.startswith("ngref:")],
           "eucref_old": [s for s in runs if s.startswith("eucref:")], "ng_supervised": [s for s in runs if s.startswith("ng:")]}
    for f in ("ngref_fix", "eucref_fix"):
        p = sel["selected"][f]["spec"].split(":")
        fam[f] = [s for s in runs if "@" not in s and s.split(":")[:3] == p[:3]]
    R["families"] = fam
    ide = "ideea:4:24:min_perp:2"
    for T in (0.1, 0.3, 0.514, 1.0):
        iv = {f: interp(runs, ss + ["base"], T) for f, ss in fam.items()}
        iv = {f: v for f, v in iv.items() if v is not None}
        R["matched"][str(T)] = {f: {"ti": float(v.mean()), "d_vs_base": bci(v - base["ti"]), "d_vs_ideea_paper": bci(v - runs[ide]["ti"]) if T == 0.514 else None} for f, v in iv.items()}
    for f in glob.glob(ROOT + "/results_gm/mc*/*.json"):
        j = json.load(open(f)); s = j["summary"]["spec"] + ("@" + f.split("/")[-2][3:] if f.split("/")[-2] != "mc" else "")
        R["mc"][s] = {k: j["summary"][k] for k in ("mc1", "mc2")}
    bm = {r["key"]: r for r in json.load(open(ROOT + "/results_gm/mc/base.json"))["records"]}
    for f in glob.glob(ROOT + "/results_gm/mc*/*.json"):
        j = json.load(open(f)); s = j["summary"]["spec"] + ("@" + f.split("/")[-2][3:] if f.split("/")[-2] != "mc" else "")
        for m in ("mc1", "mc2"):
            d = np.array([r[m] - bm[r["key"]][m] for r in j["records"]]); R["mc"][s]["d_" + m] = bci(d)
    if glob.glob(ROOT + "/blind_fix/pass2/labels_*.json"):
        st, ag = strict(None); R["strict_agreement"] = ag
        for r, d in st.items():
            R["strict"][r] = {"n": len(d["qs"]), "truth": float(d["truth"].mean()), "info": float(d["info"].mean()), "ti": bci(d["ti"]),
                              "ti_pass1": float(d["ti_p1"].mean()), "ti_pass2": float(d["ti_p2"].mean()), "truth_by_pass": d["truth_p"], "info_by_pass": d["info_p"]}
        names = list(st)
        for a in names:
            for b in ("base", "ideea_paper", "iti_sel"):
                if b in st and a != b and st[a]["qs"] == st[b]["qs"]:
                    R["strict_contrasts"][f"{a}-{b}"] = {"mean": bci(st[a]["ti"] - st[b]["ti"]), "pass1": float((st[a]["ti_p1"] - st[b]["ti_p1"]).mean()),
                                                         "pass2": float((st[a]["ti_p2"] - st[b]["ti_p2"]).mean())}
        # official TI on the same 150 questions, for the official-minus-strict gap
        key = json.load(open(f"{ROOT}/blind_fix_keys/key_pass1.json"))
        for lab, p in key["runs"].items():
            spec = json.load(open(p))["summary"]["spec"]
            rr = next((r for s, r in runs.items() if r["path"] == os.path.abspath(p) or r["path"] == p), None)
            if rr is not None and lab in R["strict"]:
                idx = [rr["keys"].index(q) for q in st[lab]["qs"]]
                R["strict"][lab]["official_ti_same_q"] = float(rr["ti"][idx].mean()); R["strict"][lab]["spec"] = spec
                R["strict"][lab]["gap_official_minus_strict"] = R["strict"][lab]["official_ti_same_q"] - R["strict"][lab]["ti"][0]
    json.dump(R, open(ROOT + "/fix_report.json", "w"), indent=1)
    print("ok", len(runs), "runs")
