"""[harness addition] Summaries for the unified Gemma-2-2B evaluation: native grids, matched-KL contrasts (paired
bootstrap over questions), MC1/MC2. Reads results_gm/test_grid/*.json + *.judged.json and results_gm/mc/*.json."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from common.proxy_judge import is_refusal, clean  # noqa
ROOT = os.path.dirname(HERE)
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
D = os.path.join(ROOT, "results_gm", f"{SPLIT}_grid")
rng = np.random.RandomState(0)


def load():
    runs = {}
    for f in sorted(glob.glob(D + "/*.json")):
        if f.endswith(".judged.json"):
            continue
        j = f.replace(".json", ".judged.json")
        if not os.path.exists(j):
            continue
        g, jd = json.load(open(f)), json.load(open(j))
        keys = [str(r["key"]) for r in g["records"]]
        jj = jd["judgements"]
        t = np.array([jj[k]["truth"] for k in keys], float); i = np.array([jj[k]["info"] for k in keys], float)
        ref = np.array([is_refusal(clean(r["response"])) for r in g["records"]], float)
        kl = np.array([r["kl_tok"] for r in g["records"]])
        runs[g["summary"]["spec"]] = {"keys": keys, "truth": t, "info": i, "ti": t * i, "refusal": ref, "kl": kl,
                                      "klmean": float(kl.mean()), "ntok": g["summary"]["mean_ntok"], "info_s": g["summary"]}
    return runs


def family(spec):
    p = spec.split(":")
    if p[0] in ("iti", "ideea"):
        return p[0] + ("_" + "_".join(p[2:]) if len(p) > 2 else ""), float(p[1])
    if p[0] == "caa":
        return f"caa{p[1]}", float(p[2])
    if p[0] == "cu":
        return f"cu{p[1]}", float(p[2])
    if p[0] == "rand":
        return f"rand{p[2]}", float(p[1])
    if p[0] == "base":
        return "base", 0.0
    return p[0], float(p[1])


def interp(runs, fam_specs, target, metric="ti"):
    """per-item outcome vector at mean per-token KL == target, linear in log KL between bracketing runs
    (same questions -> paired). None if the family does not bracket the target."""
    pts = sorted([(runs[s]["klmean"], s) for s in fam_specs], key=lambda x: x[0])
    for (k0, s0), (k1, s1) in zip(pts[:-1], pts[1:]):
        if k0 <= target <= k1 and k1 > k0:
            w = (np.log(target + 1e-6) - np.log(k0 + 1e-6)) / (np.log(k1 + 1e-6) - np.log(k0 + 1e-6))
            return (1 - w) * runs[s0][metric] + w * runs[s1][metric], (s0, s1, w)
    return None, None


def boot(x, B=2000):
    n = len(x); idx = rng.randint(0, n, (B, n))
    return x[idx].mean(1).std()


def main():
    runs = load()
    out = {"split": SPLIT, "n_runs": len(runs), "native": {}, "matched": {}, "contrasts": {}}
    fams = {}
    for s in runs:
        f, c = family(s)
        fams.setdefault(f, []).append(s)
    base = runs["base"]
    print(f"{'spec':16s} {'KL':>6s} {'truth':>6s} {'info':>6s} {'T*I':>6s} {'refus':>6s} {'ntok':>5s}")
    for s in sorted(runs, key=lambda s: (family(s)[0], family(s)[1])):
        r = runs[s]
        n = len(r["ti"])
        row = {k: float(r[k].mean()) for k in ("truth", "info", "ti", "refusal")}
        row.update({"kl": r["klmean"], "ntok": r["ntok"], "se_ti": float(r["ti"].std() / np.sqrt(n)),
                    "d_ti_vs_base": float((r["ti"] - base["ti"]).mean()), "se_d_ti": float((r["ti"] - base["ti"]).std() / np.sqrt(n))})
        out["native"][s] = row
        print(f"{s:16s} {row['kl']:6.3f} {row['truth']:6.3f} {row['info']:6.3f} {row['ti']:6.3f} {row['refusal']:6.3f} {row['ntok']:5.1f}  d_ti {row['d_ti_vs_base']:+.3f}±{row['se_d_ti']:.3f}")
    targets = (0.01, 0.03, 0.1, 0.3)
    methods = [f for f in fams if f != "base"]
    for T in targets:
        iv = {}
        for f in methods:
            v, info = interp(runs, fams[f] + ["base"], T)
            if v is not None:
                iv[f] = v
        out["matched"][str(T)] = {f: {"ti": float(v.mean()), "d_ti_vs_base": float((v - base["ti"]).mean()), "se": float((v - base["ti"]).std() / np.sqrt(len(v)))} for f, v in iv.items()}
        out["contrasts"][str(T)] = {}
        for a in iv:
            for b in iv:
                if a < b:
                    d = iv[a] - iv[b]
                    out["contrasts"][str(T)][f"{a}-{b}"] = {"d": float(d.mean()), "se": float(d.std() / np.sqrt(len(d)))}
        print(f"\nmatched KL {T}: " + ", ".join(f"{f} {out['matched'][str(T)][f]['d_ti_vs_base']:+.3f}±{out['matched'][str(T)][f]['se']:.3f}" for f in iv))
    mc = {}
    for f in glob.glob(os.path.join(ROOT, "results_gm", "mc", "*.json")):
        s = json.load(open(f))["summary"]
        mc[s["spec"]] = {k: s[k] for k in ("mc1", "mc1_se", "mc2", "mc2_se")}
    out["mc"] = mc
    json.dump(out, open(os.path.join(ROOT, f"summary_{SPLIT}.json"), "w"), indent=1)


main()
