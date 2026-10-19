"""[harness addition] Summarise Track 1 runs: per-run table, matched-KL interpolation per method, paired bootstrap SEs."""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = sys.argv[1:] or ["grid"]


def load():
    runs = {}
    for tag in TAGS:
        for f in sorted(glob.glob(f"{ROOT}/results_t1/{tag}/*.json")):
            z = json.load(open(f))
            runs[z["summary"]["spec"]] = z
    return runs


def arr(z, key):
    recs = sorted(z["records"], key=lambda r: r["key"])
    if key == "ti_emb":
        return np.array([r["truth_emb"] and r["info"] for r in recs], float)
    if key == "ti_qwen":
        return np.array([r["truth_qwen"] and r["info"] for r in recs], float)
    if key == "info_q":
        return np.array([r.get("p_info_qwen", np.nan) > 0.5 for r in recs], float)
    if key == "ti_qq":  # truth (qwen) AND info (qwen info judge)
        return np.array([r["truth_qwen"] and r["info"] and r.get("p_info_qwen", 0) > 0.5 for r in recs], float)
    return np.array([r[key] for r in recs], float)


def family(spec):
    p = spec.split(":")
    return p[0] + (":" + p[2] if len(p) > 2 else "")


def main():
    runs = load()
    base = runs.get("base")
    rows = []
    B = 2000
    rng = np.random.RandomState(0)
    n = len(base["records"])
    boot = rng.randint(0, n, size=(B, n))
    for spec, z in runs.items():
        s = z["summary"]
        row = {k: s.get(k) for k in ("spec", "truth_emb", "truth_qwen", "info", "ti_emb", "ti_qwen", "kl_tok_mean", "kl_first_mean", "mean_ntok")}
        for key in ("ti_emb", "ti_qwen", "info", "truth_emb", "truth_qwen", "info_q", "ti_qq"):
            a = arr(z, key)
            row[key] = float(a.mean())
            row[f"{key}_se"] = float(a.std() / np.sqrt(len(a)))
            if base is not None and spec != "base":
                dlt = a - arr(base, key)
                row[f"d_{key}"] = float(dlt.mean())
                row[f"d_{key}_se"] = float(dlt[boot].mean(1).std())
        rows.append(row)
    # matched KL: per family, interpolate in log KL (monotone sort), at targets
    targets = [0.02, 0.05, 0.1, 0.2, 0.4, 1.0]
    fams = {}
    for spec, z in runs.items():
        if spec == "base":
            continue
        fams.setdefault(family(spec), []).append(spec)
    matched, matched_first = {}, {}
    for klkey, M in (("kl_tok_mean", matched), ("kl_first_mean", matched_first)):
      for fam, specs0 in fams.items():
        specs = sorted(specs0, key=lambda sp: runs[sp]["summary"][klkey])
        kls = np.array([runs[sp]["summary"][klkey] for sp in specs])
        M[fam] = {}
        for t in targets:
            if len(kls) < 2 or t < kls.min() or t > kls.max():
                continue
            if not np.all(np.diff(kls) > 0):
                pass
            j = int(np.searchsorted(kls, t))
            j = min(max(j, 1), len(kls) - 1)
            lo, hi = specs[j - 1], specs[j]
            w = (np.log(t) - np.log(kls[j - 1])) / (np.log(kls[j]) - np.log(kls[j - 1]) + 1e-12)
            out = {"between": [lo, hi], "w": float(w)}
            for key in ("ti_emb", "ti_qwen", "info", "truth_emb", "truth_qwen", "info_q", "ti_qq"):
                a = (1 - w) * arr(runs[lo], key) + w * arr(runs[hi], key)
                dlt = a - arr(base, key)
                out[key] = float(a.mean())
                out[f"d_{key}"] = float(dlt.mean())
                out[f"d_{key}_se"] = float(dlt[boot].mean(1).std())
            M[fam][str(t)] = out
    res = {"runs": sorted(rows, key=lambda r: (family(r["spec"]), r["kl_tok_mean"] or 0)), "matched_kl": matched,
           "matched_first_token_kl": matched_first,
           "kl_targets": targets, "n_questions": n}
    out = os.path.join(ROOT, "results_t1", "summary.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"{'spec':18s} {'KLtok':>7s} {'KL1':>6s} {'truthE':>6s} {'truthQ':>6s} {'info':>5s} {'infoQ':>5s} {'TIemb':>6s} {'TIqwen':>6s} {'TIqq':>6s}  dTIemb(se) dTIqwen(se)")
    for r in res["runs"]:
        print(f"{r['spec']:18s} {r['kl_tok_mean']:7.3f} {r['kl_first_mean']:6.3f} {r['truth_emb']:6.3f} {r['truth_qwen']:6.3f} {r['info']:5.3f} {r['info_q']:5.3f} {r['ti_emb']:6.3f} {r['ti_qwen']:6.3f} {r['ti_qq']:6.3f}  "
              f"{r.get('d_ti_emb', 0):+.3f}({r.get('d_ti_emb_se', 0):.3f}) {r.get('d_ti_qwen', 0):+.3f}({r.get('d_ti_qwen_se', 0):.3f})")
    print("\nmatched KL (per-token):")
    for fam, d in matched.items():
        print(fam, {t: f"TIe {v['ti_emb']:.3f} ({v['d_ti_emb']:+.3f}±{v['d_ti_emb_se']:.3f}) TIq {v['ti_qwen']:.3f} ({v['d_ti_qwen']:+.3f}±{v['d_ti_qwen_se']:.3f}) TIqq {v['ti_qq']:.3f}" for t, v in d.items()})


if __name__ == "__main__":
    main()
