"""[harness addition] Bridge report: official metrics of the test runs (results_gm/test_bridge + anchors), ladder steps
(paired bootstrap over the 395 test q), strict labels (if bridge_strict.json exists), cosines/gains from fit_bridge.json.
-> bridge_report.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from gm_fix_report import load, bci  # noqa

SEL = json.load(open(ROOT + "/bridge_selection.json"))
T = {}
T.update(load(ROOT + "/results_gm/test_bridge"))
G = load(ROOT + "/results_gm/test_grid"); F = load(ROOT + "/results_gm/test_fix")
T["base"] = G["base"]; T["L0"] = G["ideea:4:24:min_perp:2"]
ANCH = {"K_inf": "fx:efsum:13:0.2673", "K_10": "fx:nfsum10:13:0.4014"}
for k, s in ANCH.items():
    T[k] = F[s]


def cfg(spec):
    return spec.split(":")[1] if spec.startswith("br:") else spec


runs = {}
for s, r in T.items():
    runs[cfg(s)] = r
# K_0.1 / K_1 matched runs live in test_bridge as fx specs
for s, r in list(runs.items()):
    if s.startswith("fx:nfsum0.1:13"):
        runs["K_0.1"] = r
    if s.startswith("fx:nfsum1:13"):
        runs["K_1"] = r
order = sorted(runs)
keys = runs["base"]["keys"]
for k in order:
    assert runs[k]["keys"] == keys, k
row = lambda r: {"kl": r["kl"], "truth": float(r["truth"].mean()), "info": float(r["info"].mean()), "ti": bci(r["ti"]), "refusal": float(r["refusal"].mean()),
                 "words": float(r["nwords"].mean()), "dneg": float(r["dneg"].mean()), "rep3": float(r["rep3"].mean()), "hitlim": float(r["hitlim"].mean())}
out = {"official": {k: row(runs[k]) for k in order if k in runs}}
sel = {f: (v["selected"] or {}).get("config") for f, v in SEL["families"].items()}
nonplain = {f: (v["best_nonplain"] or {}).get("config") for f, v in SEL["families"].items()}
msel = {f: (nonplain[f] if (sel[f] or "").endswith("_inf") or sel[f] == "F_inf_inf" else sel[f]) for f in sel}
for f_ in ("G", "H", "I"):  # declared deviation: no finite-lam config passes the refusal cap -> max-dev-TI finite lam (KL<=1.2)
    if not msel.get(f_) or msel[f_] == f_ + "_inf" or msel[f_] is None:
        msel[f_] = f_ + "_10"
out["sel"], out["nonplain"], out["metric_sel"] = sel, nonplain, msel
S = lambda f: msel.get(f)
STEPS = [("S1 data", "L0", "A_inf"), ("S2 clustering", "A_inf", "B_inf"), ("S3 context", "B_inf", "E_inf"), ("S4 md->grad (ctx)", "E_inf", "C_inf"),
         ("S4b md->grad (bare)", "B_inf", "C_inf"), ("S5 norm ph->gl (grad)", "C_inf", "H_inf"), ("S5b norm ph->gl (md)", "B_inf", "G_inf"),
         ("S6 metric head gl grad", "H_inf", S("H")), ("S6 metric head ph grad", "C_inf", S("C")), ("S6 metric head ph md", "B_inf", S("B")),
         ("S6 metric head gl md", "G_inf", S("G")), ("S6 metric IDEEA clustered", "A_inf", S("A")),
         ("S7 site grad (head gl -> L13)", "H_inf", "K_inf"), ("S7b site md (head gl -> L13)", "G_inf", "I_inf"),
         ("S8 metric resid grad (eucref->ngref)", "K_inf", "K_10"), ("S8b metric resid md", "I_inf", S("I")), ("S8c resid ctx: md bare -> ctx", "I_inf", "J_inf"),
         ("S8d resid md->grad (ctx)", "J_inf", "K_inf"), ("S9 clustering on grad", "C_inf", "D_inf"), ("S9b clustered grad metric", "D_inf", S("D")),
         ("F kernel vs IDEEA-train", "A_inf", S("F")), ("F kernel vs plain mass-mean", "B_inf", S("F")),
         ("rev: ngref site (L13 -> head)", "K_10", S("H")), ("rev: ngref objective (grad -> md)", "K_10", S("I"))]
for o in ("O1", "O2", "O3", "O4"):
    if sel.get(o):
        STEPS += [(f"{o} vs ngref-fix", "K_10", sel[o]), (f"{o} vs eucref-fix", "K_inf", sel[o]), (f"{o} vs IDEEA", "L0", sel[o])]
out["steps_official"] = {}
for name, a, b in STEPS:
    if a in runs and b in runs:
        out["steps_official"][name] = {"from": a, "to": b, "d_ti": bci(runs[b]["ti"] - runs[a]["ti"]), "kl": (runs[a]["kl"], runs[b]["kl"])}
out["steps_def"] = STEPS
if os.path.exists(ROOT + "/bridge_strict.json"):
    st = json.load(open(ROOT + "/bridge_strict.json"))
    out["strict"] = st["runs"]
    out["steps_strict"] = {}
    for name, a, b in STEPS:
        if a in st["item_ti"] and b in st["item_ti"]:
            x = np.array(st["item_ti"][b]) - np.array(st["item_ti"][a])
            p1 = np.array(st["item_ti_p1"][b]) - np.array(st["item_ti_p1"][a]); p2 = np.array(st["item_ti_p2"][b]) - np.array(st["item_ti_p2"][a])
            out["steps_strict"][name] = {"from": a, "to": b, "d_ti": bci(x), "p1": float(p1.mean()), "p2": float(p2.mean())}
json.dump(out, open(ROOT + "/bridge_report.json", "w"), indent=1)
print("%-14s %6s %6s %6s %6s %6s %5s %5s" % ("run", "KL", "T", "I", "TI", "ref", "words", "dneg"))
for k, v in out["official"].items():
    print("%-14s %6.3f %6.3f %6.3f %6.3f %6.3f %5.1f %5.2f" % (k, v["kl"], v["truth"], v["info"], v["ti"][0], v["refusal"], v["words"], v["dneg"]))
for k, v in out["steps_official"].items():
    s_ = out.get("steps_strict", {}).get(k)
    print("%-40s %-12s->%-14s off %+.3f [%+.3f,%+.3f]" % (k, v["from"], v["to"], v["d_ti"][0], v["d_ti"][2], v["d_ti"][3]),
          "" if not s_ else "strict %+.3f [%+.3f,%+.3f] p1 %+.3f p2 %+.3f" % (s_["d_ti"][0], s_["d_ti"][2], s_["d_ti"][3], s_["p1"], s_["p2"]))
