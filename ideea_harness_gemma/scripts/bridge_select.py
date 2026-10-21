"""[harness addition] Bridge dev-only selection (results/bridge_prereg.md §4, rule of ngref_fix_prereg §3).
Reads results_gm/development_bridge (br: runs), results_gm/development_fix (K family = fx:{efsum,nfsum*}:13),
results_gm/development_grid (base). Writes bridge_selection.json: per family the selected config, the best finite-lam config,
and the matched (dev KL 0.518) strength of every config, interpolated from its 4-point dev KL sweep."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from gm_fix_select import load, matched_c  # noqa


def config_of(spec):
    p = spec.split(":")
    if p[0] == "br":
        return p[1]
    if p[0] == "fx" and p[2] == "13" and (p[1] == "efsum" or p[1].startswith("nfsum")):
        return {"efsum": "K_inf", "nfsum0.1": "K_0.1", "nfsum1": "K_1", "nfsum10": "K_10"}.get(p[1])
    return None


def family_of(cfg):
    f = cfg.split("_")[0]
    return f  # A B C D E F G H I J K O1 O2 O3 O4


def is_plain(cfg):
    f = family_of(cfg)
    if f == "F":
        return cfg == "F_inf_inf"
    if f.startswith("O"):
        return False
    return cfg.endswith("_inf")


def strength(spec):
    return float(spec.split(":")[-1])


def matched(runs, specs, T=0.518):
    pts = sorted((strength(s), runs[s]["kl"]) for s in specs)
    for (c0, k0), (c1, k1) in zip(pts[:-1], pts[1:]):
        if k0 <= T <= k1 and k1 > k0:
            return round(float(np.exp(np.log(c0) + (np.log(T) - np.log(k0)) / (np.log(k1) - np.log(k0)) * (np.log(c1) - np.log(c0)))), 4), pts
    (c0, k0), (c1, k1) = (pts[0], pts[1]) if T < pts[0][1] else (pts[-2], pts[-1])
    return round(float(c1 * np.exp((np.log(T) - np.log(k1)) / ((np.log(k1) - np.log(k0)) / (np.log(c1) - np.log(c0))))), 4), pts


def pick(cands, base):
    ok = [(s, r) for s, r in cands if r["kl"] <= 1.2 and r["refusal"] <= base["refusal"]]
    if not ok:
        return None, 0
    top = max(r["ti"] for _, r in ok)
    s, r = min([(s, r) for s, r in ok if r["ti"] >= top - 0.005], key=lambda x: x[1]["kl"])
    return {"spec": s, "config": config_of(s), **{k: r[k] for k in ("ti", "truth", "info", "refusal", "kl")}}, len(ok)


if __name__ == "__main__":
    br = load(ROOT + "/results_gm/development_bridge"); fx = load(ROOT + "/results_gm/development_fix")
    base = load(ROOT + "/results_gm/development_grid")["base"]
    runs = {s: r for s, r in {**br, **fx}.items() if config_of(s)}
    cfg = {}
    for s in runs:
        cfg.setdefault(config_of(s), []).append(s)
    fams = sorted(set(family_of(c) for c in cfg))
    out = {"base_dev": base, "families": {}}
    for f in fams:
        cs = [c for c in cfg if family_of(c) == f]
        allc = [(s, runs[s]) for c in cs for s in cfg[c]]
        sel, nok = pick(allc, base)
        fin, _ = pick([(s, runs[s]) for c in cs if not is_plain(c) for s in cfg[c]], base)
        out["families"][f] = {"n_dev_configs": len(allc), "n_pass_constraints": nok, "selected": sel, "best_nonplain": fin,
                              "matched": {c: matched(runs, cfg[c])[0] for c in cs if len(cfg[c]) >= 2},
                              "dev_sweeps": {c: sorted([(strength(s), round(runs[s]["kl"], 4), round(runs[s]["ti"], 4), round(runs[s]["refusal"], 3)) for s in cfg[c]]) for c in cs}}
    json.dump(out, open(ROOT + "/bridge_selection.json", "w"), indent=1)
    for f, v in out["families"].items():
        print(f, v["n_dev_configs"], "sel", v["selected"] and (v["selected"]["spec"], round(v["selected"]["ti"], 3), round(v["selected"]["kl"], 3)),
              "| nonplain", v["best_nonplain"] and (v["best_nonplain"]["spec"], round(v["best_nonplain"]["ti"], 3)))
