"""[harness addition] Final report numbers: matched-KL TI and MC (dense sweep families only), native single configs,
paired contrasts, blind-label audit, seed noise. Writes report_numbers.json and prints markdown tables."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from common.proxy_judge import is_refusal, clean  # noqa
ROOT = os.path.dirname(HERE)
rng = np.random.RandomState(0)
import importlib.util
spec = importlib.util.spec_from_file_location("gf", HERE + "/gm_final.py")
src = open(HERE + "/gm_final.py").read().replace("\nmain()\n", "\n")
gf = {"__file__": HERE + "/gm_final.py"}; exec(compile(src, "gm_final", "exec"), gf)
runs = gf["load"](ROOT + "/results_gm/test_grid"); runs.update(gf["load"](ROOT + "/results_gm/test_s1", "@s1"))
base = runs["base"]
mc = {}
for f in glob.glob(ROOT + "/results_gm/mc/*.json"):
    j = json.load(open(f)); mc[j["summary"]["spec"]] = {r["key"]: r for r in j["records"]}
keys = sorted(mc["base"])
for s in mc:
    mc[s] = {m: np.array([mc[s][k][m] for k in keys]) for m in ("mc1", "mc2")}
bse = gf["bse"]
DENSE = {"iti": "iti", "ideea": "ideea", "caa13": "caa13", "cu13": "cu13", "ngref": "ngref", "eucref": "eucref", "ng": "ng", "euc": "euc", "rand": "rand"}


def fam_specs(fam, source):
    out = []
    for s in source:
        if s == "base" or s.endswith("@s1"):
            continue
        f, _ = gf["family"](s)
        if f == fam:
            out.append(s)
    return out


def interp_vec(source_kl, source_vec, specs, T):
    pts = sorted([(source_kl(s), s) for s in specs] + [(0.0, "base")], key=lambda x: x[0])
    for (k0, s0), (k1, s1) in zip(pts[:-1], pts[1:]):
        if k0 <= T <= k1 and k1 > k0:
            w = (np.log(T + 1e-3) - np.log(k0 + 1e-3)) / (np.log(k1 + 1e-3) - np.log(k0 + 1e-3))
            return (1 - w) * source_vec(s0) + w * source_vec(s1), (s0, s1)
    return None, None


out = {"matched_ti": {}, "matched_mc1": {}, "matched_mc2": {}, "contr_ti": {}, "contr_mc1": {}, "contr_mc2": {}}
TS = (0.03, 0.1, 0.3, 1.0)
for metric, key in (("ti", "ti"), ("mc1", "mc1"), ("mc2", "mc2")):
    for T in TS:
        iv = {}
        for fam in DENSE:
            if metric == "ti":
                ss = fam_specs(fam, runs)
                if fam == "rand":
                    seeds = {}
                    for s in ss:
                        seeds.setdefault(s.split(":")[2], []).append(s)
                    vs = [interp_vec(lambda s: runs[s]["klmean"], lambda s: runs[s]["ti"], v, T)[0] for v in seeds.values()]
                    vs = [v for v in vs if v is not None]
                    if len(vs) >= 2:
                        iv[fam] = np.mean(vs, 0)
                    continue
                v, _ = interp_vec(lambda s: runs[s]["klmean"], lambda s: (base["ti"] if s == "base" else runs[s]["ti"]), ss, T) if len(ss) >= 3 else (None, None)
                ref = base["ti"]
            else:
                ss = [s for s in fam_specs(fam, runs) if s in mc]
                if fam == "rand":
                    seeds = {}
                    for s in ss:
                        seeds.setdefault(s.split(":")[2], []).append(s)
                    vs = [interp_vec(lambda s: runs[s]["klmean"], lambda s: mc[s][key], v, T)[0] if len(v) >= 1 else None for v in seeds.values()]
                    vs = [v for v in vs if v is not None]
                    if vs:
                        iv[fam] = np.mean(vs, 0)
                    continue
                v, _ = interp_vec(lambda s: runs[s]["klmean"], lambda s: mc["base"][key] if s == "base" else mc[s][key], ss, T) if len(ss) >= 3 else (None, None)
                ref = mc["base"][key]
            if v is not None:
                iv[fam] = v
        out["matched_" + metric][str(T)] = {f: {"mean": float(v.mean()), "d": float((v - ref).mean()), "se": bse(v - ref)} for f, v in iv.items()}
        out["contr_" + metric][str(T)] = {f"{a}-{b}": {"d": float((iv[a] - iv[b]).mean()), "se": bse(iv[a] - iv[b])} for a in iv for b in iv if a < b}
# rand baseline cannot be MC-interpolated well (only 0.2 x3, 0.8 x1): its MC delta is reported natively
# ---- blind audit
key = json.load(open(ROOT + "/blind/key.json")); L = []
for i in range(3):
    L += json.load(open(ROOT + f"/blind/labels_{i}.json"))
lab = {x["id"]: x for x in L}
aud = {}
for k in key:
    aud.setdefault(k["run"], []).append((k["key"], lab[k["id"]]))
out["blind"] = {}
for r, items in aud.items():
    spec = None
    jj = json.load(open(ROOT + f"/results_gm/test_grid/{r}.judged.json"))["judgements"]
    o = np.array([[jj[str(q)]["truth"], jj[str(q)]["info"]] for q, _ in items], float)
    c = np.array([[l["truthful"], l["informative"]] for _, l in items], float)
    out["blind"][r] = {"n": len(items), "off_truth": o[:, 0].mean(), "off_info": o[:, 1].mean(), "off_ti": (o[:, 0] * o[:, 1]).mean(),
                       "cl_truth": c[:, 0].mean(), "cl_info": c[:, 1].mean(), "cl_ti": (c[:, 0] * c[:, 1]).mean(),
                       "se_cl_ti": float((c[:, 0] * c[:, 1]).std() / np.sqrt(len(items)))}
json.dump(out, open(ROOT + "/report_numbers.json", "w"), indent=1, default=float)
for m in ("ti", "mc1", "mc2"):
    print("== matched", m)
    for T in TS:
        print(T, {f: f"{v['d']:+.3f}±{v['se']:.3f}" for f, v in out["matched_" + m][str(T)].items()})
for m in ("ti", "mc1"):
    print("== contrasts", m)
    for T in (0.03, 0.1, 0.3):
        print(T, {k: f"{v['d']:+.3f}±{v['se']:.3f}" for k, v in out["contr_" + m][str(T)].items() if k.split("-")[0] in ("ngref", "eucref", "ng") or k.split("-")[1] in ("ngref", "eucref", "ng")})
