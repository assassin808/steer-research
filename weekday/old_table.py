"""Summarise old_stats outputs: per layer, detections over the 14 class pairs."""
import json, sys, os
import numpy as np


def summ(fn):
    d = json.load(open(fn)); out = {}
    for l, P in d.items():
        o1a = sum(p["turn1"] > max(v[0] for v in p["turn1_null"]["rshift"]) and p["turn1"] > max(v[0] for v in p["turn1_null"]["gshared"]) for p in P)
        o1c = sum(p["turn1_cv"] > max(v[1] for v in p["turn1_null"]["rshift"]) and p["turn1_cv"] > max(v[1] for v in p["turn1_null"]["gshared"]) for p in P)
        gap = [np.mean(p["paths"]["straight"]["min_mmd_mid"]) - np.mean(p["paths"]["polyline"]["min_mmd_mid"]) for p in P]
        rel = [g / np.mean(p["paths"]["straight"]["min_mmd_mid"]) for g, p in zip(gap, P)]
        ot = sum(p["ot_r2s"] < min(p["ot_r2s_rshift"]) for p in P)
        out[l] = dict(n=len(P), O1_alpha100=f"{o1a}/{len(P)}", O1_cv=f"{o1c}/{len(P)}", turn1_cv_med=float(np.median([p["turn1_cv"] for p in P])),
                      O2_dstar_med=float(np.median([p["paths"]["straight"]["dstar"] for p in P])),
                      O2_mid_gap_pos=f"{sum(g > 0 for g in gap)}/{len(P)}", O2_mid_gap_rel_med=float(np.median(rel)),
                      O3_geo_two_med=float(np.median([p["geo_two"] for p in P])), O3_geo_all_med=float(np.median([p["geo_all"] for p in P])),
                      OT_r2s_med=float(np.median([p["ot_r2s"] for p in P])), OT_curved=f"{ot}/{len(P)}")
    return out


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        print("==", os.path.basename(fn))
        for l, r in summ(fn).items():
            print(f"  L{l}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
