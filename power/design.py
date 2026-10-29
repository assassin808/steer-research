"""n per level needed for 80% oracle power at the persona SNR (q mode), per statistic. Writes ~/scratch/power/design.json"""
import json, numpy as np, core as C
A = json.load(open(f"{C.R}/aggA.json")); PL = json.load(open(f"{C.R}/place.json"))
NS = [20, 50, 100, 300, 1000]; S = C.SNRS
def need_n(bg, L, mode, st, sh, snr):
    pw = np.array([np.interp(np.log10(snr), np.log10(S), A[f"{bg}_rand_L{L}_n{n}_{mode}"][st][C.SHAPES.index(sh)]) for n in NS])
    if pw[0] >= .8: return "≤20"
    for i in range(1, len(NS)):
        if pw[i] >= .8:
            x = np.interp(.8, [pw[i - 1], pw[i]], np.log10([NS[i - 1], NS[i]])); return f"{10 ** x:.0f}"
    return ">1000"
out = {}
for bg, tr in [("syc", "sycophantic"), ("opt", "optimistic")]:
    s2 = PL[tr]["layers"]["14"]["snr"]
    for L in [5, 7]:
        vt = (L + 1) / (12 * (L - 1)); s = 4 * vt * s2
        for mode in ["q", "iid"]:
            for sh in ["arc60", "arc90", "bend90"]:
                k = f"{bg} L{L} {mode} {sh} SNR_L={s:.4f}"
                out[k] = {st: need_n(bg, L, mode, st, sh, s) for st in ["quad", "offline", "turn"]}
                out[k]["ord1(arc0)"] = need_n(bg, L, mode, "ord1", "arc0", s)
                print(k, out[k])
# TQA: graded TQA would need: tqa q mode at SNR .0098 (if a graded ladder had the binary SNR)
for L in [5, 7]:
    s = 4 * (L + 1) / (12 * (L - 1)) * PL["tqa_top24"]["snr_med"]
    for sh in ["arc60", "arc90"]:
        print(f"tqa13_5 L{L} q {sh} SNR_L={s:.4f}", {st: need_n("tqa13_5", L, "q", st, sh, s) for st in ["quad", "offline", "turn"]})
json.dump(out, open(f"{C.R}/design.json", "w"), indent=1)
