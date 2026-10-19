"""[harness addition] Final tables for the unified Gemma-2-2B evaluation -> summary_final.json (+ markdown on stdout).
Reads results_gm/test_grid, test_s1 (second ngref fit seed), development_grid, mc. Paired bootstrap over the 395 test questions."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from common.proxy_judge import is_refusal, clean  # noqa
ROOT = os.path.dirname(HERE)
rng = np.random.RandomState(0)
HEDGE = re.compile(r"\b(difficult|varies|vary|no (single|definitive|scientific|universal|widely|consensus)|depends|not (typically|generally|usually|necessarily)|it'?s (impossible|hard|not clear)|there (is|are) no|complex)\b", re.I)


def load(D, suffix=""):
    runs = {}
    for f in sorted(glob.glob(D + "/*.json")):
        if f.endswith(".judged.json"):
            continue
        j = f.replace(".json", ".judged.json")
        if not os.path.exists(j):
            continue
        g, jd = json.load(open(f)), json.load(open(j))
        keys = [str(r["key"]) for r in g["records"]]; jj = jd["judgements"]
        t = np.array([jj[k]["truth"] for k in keys], float); i = np.array([jj[k]["info"] for k in keys], float)
        resp = [clean(r["response"]) for r in g["records"]]
        kl = np.array([r["kl_tok"] for r in g["records"]])
        runs[g["summary"]["spec"] + suffix] = {"keys": keys, "truth": t, "info": i, "ti": t * i,
            "refusal": np.array([is_refusal(x) for x in resp], float), "hedge": np.array([bool(HEDGE.search(x)) for x in resp], float),
            "nwords": np.array([len(x.split()) for x in resp], float), "kl": kl, "klmean": float(kl.mean()), "ntok": g["summary"]["mean_ntok"],
            "pyes_t": np.array([jj[k]["truth_pyes_top"] for k in keys]), "pyes_i": np.array([jj[k]["info_pyes_top"] for k in keys])}
    return runs


def family(spec):
    sfx = ""
    if spec.endswith("@s1"):
        spec = spec[:-3]; sfx = "@s1"
    p = spec.split(":")
    if p[0] in ("iti", "ideea"):
        return p[0] + ("_" + "_".join(p[2:]) if len(p) > 2 else "") + sfx, float(p[1])
    if p[0] in ("caa", "cu"):
        return f"{p[0]}{p[1]}", float(p[2])
    if p[0] == "rand":
        return "rand", float(p[1])
    if p[0] == "base":
        return "base", 0.0
    return p[0] + sfx, float(p[1])


def interp(runs, specs, T, metric):
    pts = sorted([(runs[s]["klmean"], s) for s in specs], key=lambda x: x[0])
    for (k0, s0), (k1, s1) in zip(pts[:-1], pts[1:]):
        if k0 <= T <= k1 and k1 > k0:
            w = (np.log(T + 1e-3) - np.log(k0 + 1e-3)) / (np.log(k1 + 1e-3) - np.log(k0 + 1e-3))
            return (1 - w) * runs[s0][metric] + w * runs[s1][metric]
    return None


def bse(x, B=2000):
    n = len(x); idx = rng.randint(0, n, (B, n)); return float(x[idx].mean(1).std())


def main():
    runs = load(ROOT + "/results_gm/test_grid"); runs.update(load(ROOT + "/results_gm/test_s1", "@s1"))
    dev = load(ROOT + "/results_gm/development_grid")
    out = {"native": {}, "matched": {}, "contrasts": {}, "dev": {}, "mc": {}}
    base = runs["base"]
    for s, r in runs.items():
        n = len(r["ti"]); d = r["ti"] - base["ti"]
        out["native"][s] = {**{k: float(r[k].mean()) for k in ("truth", "info", "ti", "refusal", "hedge", "nwords")}, "kl": r["klmean"], "ntok": r["ntok"],
                            "se_ti": float(r["ti"].std() / np.sqrt(n)), "d_ti": float(d.mean()), "se_d_ti": bse(d),
                            "info_among_nonref": float(r["info"][r["refusal"] == 0].mean()) if (r["refusal"] == 0).any() else None,
                            "truth_among_nonref": float(r["truth"][r["refusal"] == 0].mean()) if (r["refusal"] == 0).any() else None}
    for s, r in dev.items():
        out["dev"][s] = {"truth": float(r["truth"].mean()), "info": float(r["info"].mean()), "ti": float(r["ti"].mean()), "kl": r["klmean"]}
    fams = {}
    for s in runs:
        f, _ = family(s)
        if f != "base":
            fams.setdefault(f, []).append(s)
    for T in (0.03, 0.1, 0.3, 1.0):
        iv = {}
        for f, ss in fams.items():
            if f == "rand":  # per-seed interpolation, then mean over seeds (paired, same questions)
                seeds = {}
                for s in ss:
                    seeds.setdefault(s.split(":")[2], []).append(s)
                vs = [interp(runs, v + ["base"], T, "ti") for v in seeds.values()]
                vs = [v for v in vs if v is not None]
                if vs:
                    iv[f] = np.mean(vs, 0); iv[f + "_nseeds"] = len(vs)
                continue
            v = interp(runs, ss + ["base"], T, "ti")
            if v is not None:
                iv[f] = v
        names = [k for k in iv if not k.endswith("_nseeds")]
        out["matched"][str(T)] = {f: {"ti": float(iv[f].mean()), "d_vs_base": float((iv[f] - base["ti"]).mean()), "se": bse(iv[f] - base["ti"])} for f in names}
        out["contrasts"][str(T)] = {}
        for a in names:
            for b in names:
                if a < b:
                    d = iv[a] - iv[b]; out["contrasts"][str(T)][f"{a}-{b}"] = {"d": float(d.mean()), "se": bse(d)}
    for f in glob.glob(ROOT + "/results_gm/mc/*.json"):
        j = json.load(open(f)); s = j["summary"]
        recs = {r["key"]: r for r in j["records"]}
        out["mc"][s["spec"]] = {k: s[k] for k in ("mc1", "mc1_se", "mc2", "mc2_se")}
    if "base" in out["mc"]:
        b = {r["key"]: r for r in json.load(open(ROOT + "/results_gm/mc/base.json"))["records"]}
        for f in glob.glob(ROOT + "/results_gm/mc/*.json"):
            j = json.load(open(f)); s = j["summary"]["spec"]
            d1 = np.array([r["mc1"] - b[r["key"]]["mc1"] for r in j["records"]]); d2 = np.array([r["mc2"] - b[r["key"]]["mc2"] for r in j["records"]])
            out["mc"][s].update({"d_mc1": float(d1.mean()), "se_d_mc1": float(d1.std() / np.sqrt(len(d1))), "d_mc2": float(d2.mean()), "se_d_mc2": float(d2.std() / np.sqrt(len(d2))),
                                 "kl": float(runs[s]["klmean"]) if s in runs else None})
    json.dump(out, open(ROOT + "/summary_final.json", "w"), indent=1)
    print("done", len(runs), "runs")


main()
