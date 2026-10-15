"""[harness addition] Paired contrasts between methods at matched realized KL (per-token and first-token), bootstrap over questions."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_summarize import load, arr, family  # noqa

runs = load()
fams = {}
for s in runs:
    if s != "base":
        fams.setdefault(family(s), []).append(s)
n = len(runs["base"]["records"])
boot = np.random.RandomState(1).randint(0, n, size=(2000, n))


def interp(fam, t, klkey, key):
    specs = sorted(fams[fam], key=lambda sp: runs[sp]["summary"][klkey])
    kls = np.array([runs[sp]["summary"][klkey] for sp in specs])
    if t < kls.min() or t > kls.max():
        return None
    j = min(max(int(np.searchsorted(kls, t)), 1), len(kls) - 1)
    w = (np.log(t) - np.log(kls[j - 1])) / (np.log(kls[j]) - np.log(kls[j - 1]) + 1e-12)
    return (1 - w) * arr(runs[specs[j - 1]], key) + w * arr(runs[specs[j]], key)


pairs = [("ng", "euc"), ("ng", "iti"), ("ng", "ideea"), ("ng", "rand:0"), ("ng", "caa"), ("ngkl", "ng"), ("ngx", "ng"), ("ngx", "ngkl"), ("iti", "ideea"), ("ideea", "rand:0")]
out = {}
for klkey in ("kl_tok_mean", "kl_first_mean"):
    for key in ("ti_qwen", "ti_emb"):
        for t in (0.02, 0.05, 0.1, 0.2, 0.4):
            for a, b in pairs:
                x, y = interp(a, t, klkey, key), interp(b, t, klkey, key)
                if x is None or y is None:
                    continue
                d = x - y
                out[f"{klkey}|{key}|{t}|{a}-{b}"] = {"diff": float(d.mean()), "se": float(d[boot].mean(1).std())}
json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_t1", "contrasts.json"), "w"), indent=1)
for k, v in out.items():
    if "|ti_qwen|" in k and ("|0.05|" in k or "|0.2|" in k):
        print(f"{k:45s} {v['diff']:+.3f} ± {v['se']:.3f}")
