"""Aggregate grids -> oracle power tables (prereg power_curve_prereg.md). Writes ~/scratch/power/aggA.json, aggP.json."""
import os, glob, json, re, numpy as np
import core as C
R = C.R


def oracle_power(stat_cur, stat_null, alpha=.05):
    """stat_* [sims]; threshold = 95% quantile of null (arc0) sims; nan -> not detected"""
    nul = stat_null[np.isfinite(stat_null)]
    if len(nul) < 10: return np.nan
    thr = np.quantile(nul, 1 - alpha)
    return float(np.mean(np.nan_to_num(stat_cur, nan=-np.inf) > thr))


def aggA():
    out = {}
    for fn in sorted(glob.glob(f"{R}/partA/*.npz")):
        key = os.path.basename(fn)[:-4]; z = np.load(fn)
        d = {}
        for st in ["turn", "offline", "quad"]:
            X = z[st]   # [sims, nsh, nsnr]
            d[st] = [[oracle_power(X[:, j, k], X[:, 0, k]) for k in range(len(C.SNRS))] for j in range(len(C.SHAPES))]
        for st in ["ord1", "f1c"]:
            P = z[st + "_p"]
            d[st] = [[float(np.mean(P[:, j, k] <= .05)) for k in range(len(C.SNRS))] for j in range(len(C.SHAPES))]
            d[st + "_fpr0"] = float(np.mean(z[st + "_p_snr0"] <= .05))
        # null-vs-null sanity: sat vs arc0 threshold (oracle FPR of a straight non-uniform line)
        out[key] = d
    json.dump(out, open(f"{R}/aggA.json", "w"))
    return out


def aggP():
    out = {}
    for fn in sorted(glob.glob(f"{R}/partP/*.npz")):
        key = os.path.basename(fn)[:-4]; z = np.load(fn); res = z["res"]; sh = list(z["shapes"])
        d = {"shapes": sh}
        for si, st in enumerate(["frot", "fincoh", "dist"]):
            d[st] = [[oracle_power(res[:, j, k, si], res[:, 0, k, si]) for k in range(res.shape[2])] for j in range(len(sh))]
        out[key] = d
    json.dump(out, open(f"{R}/aggP.json", "w"))
    return out


if __name__ == "__main__":
    A = aggA(); P = aggP()
    sh = C.SHAPES
    for key in ["tqa13_5_rand_L7_n100_iid", "syc_rand_L7_n300_q", "syc_rand_L5_n100_q", "wk_rand_L7_n100_iid"]:
        print(key)
        for st in ["turn", "offline", "quad", "ord1"]:
            for s in ["arc0", "sat", "arc60", "arc90", "arc180", "clust90"]:
                print(f"  {st:8s} {s:8s}", np.round(A[key][st][sh.index(s)], 2))
    for key in ["tqa13_5_L7_n100_iid", "syc_L7_n100_qraw", "syc_L7_n100_qdm", "tqa13_5_L2_n300_iid"]:
        print(key, P[key]["shapes"])
        for st in ["frot", "fincoh", "dist"]:
            print("  ", st, np.round(np.array(P[key][st], float), 2).tolist())
