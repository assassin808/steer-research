"""Prereg predictions P2-P6 from the oracle grids."""
import json, numpy as np, core as C
A = json.load(open(f"{C.R}/aggA.json")); P = json.load(open(f"{C.R}/aggP.json"))
S = list(C.SNRS); SH = C.SHAPES
def pw(bg, L, n, mode, st, sh, plane="rand"): return np.array(A[f"{bg}_{plane}_L{L}_n{n}_{mode}"][st][SH.index(sh)])
def interp_n(bg, L, n, mode, st, sh, snr):
    ns = [20, 50, 100, 300, 1000]; out = []
    for nn in ns: out.append(np.interp(np.log10(snr), np.log10(S), pw(bg, L, nn, mode, st, sh)))
    return float(np.interp(np.log10(n), np.log10(ns), out))
print("P2 weekday consistency (arc309, L7):")
for st in ["turn", "offline", "quad", "f1c", "ord1"]:
    print(f"  {st:8s} wk last tok n78 SNR .0138: {interp_n('wk', 7, 78, 'iid', st, 'arc309', .0138):.2f} | wkz n54 SNR 1.86: {interp_n('wkz', 7, 54, 'iid', st, 'arc309', 1.86):.2f}")
print("P4 centroid stats on clust (max oracle power over all cells, rand plane):")
for st in ["turn", "offline", "quad"]:
    m = max(max(A[k][st][SH.index(s)]) for k in A if "_rand_" in k for s in ["clust60", "clust90", "clust180"])
    print(f"  {st}: max {m:.2f}")
print("FPR on sat (oracle, arc0 threshold): median / max over rand cells")
for st in ["turn", "offline", "quad"]:
    v = np.array([A[k][st][SH.index("sat")] for k in A if "_rand_" in k]).ravel()
    print(f"  {st}: median {np.nanmedian(v):.3f} max {np.nanmax(v):.2f}")
print("ord1 / f1c FPR at SNR 0: ", np.round([A[k]["ord1_fpr0"] for k in A][:8], 3), "mean", np.mean([A[k]["ord1_fpr0"] for k in A]).round(3), np.mean([A[k]["f1c_fpr0"] for k in A]).round(3))
print("top plane vs rand (quad arc90 L7 n100 iid min power diff):")
for bg in ["tqa13_5", "syc", "wk"]:
    print("  ", bg, "rand", pw(bg, 7, 100, "iid", "quad", "arc90").round(2), "top", pw(bg, 7, 100, "iid", "quad", "arc90", "top").round(2))
print("bend90 L7 n100 (tqa iid):", {st: pw("tqa13_5", 7, 100, "iid", st, "bend90").round(2).tolist() for st in ["turn", "offline", "quad"]})
print("L=7 vs 5 arc360 quad tqa n100:", pw("tqa13_5", 7, 100, "iid", "quad", "arc360").round(2), pw("tqa13_5", 5, 100, "iid", "quad", "arc360").round(2))
