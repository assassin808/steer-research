"""Class SNR of TruthfulQA top-24 IDEEA heads, same definition as fourier.py (centroid power / C / mean sq. within-class distance)."""
import json, os, numpy as np
R = os.path.expanduser("~/scratch/ot_geom")
m = np.load(f"{R}/meta.npz"); lab = m["label"]; tr = m["is_train"]
H = json.load(open(os.path.expanduser("~/steer_math/ot_geometry/heads.json")))["top24"]
for sp in ["raw", "z"]:
    v = []
    for l, h in H:
        X = np.load(f"{R}/heads/L{l}H{h}.npy").astype(float)
        Z = X if sp == "raw" else (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-8)
        y = (lab == 1).astype(int); c = np.stack([Z[y == 0].mean(0), Z[y == 1].mean(0)])
        W = np.mean(((Z - c[y]) ** 2).sum(1)); v.append(((c - c.mean(0)) ** 2).sum() / 2 / W)
    print(sp, "median %.4f range %.4f-%.4f" % (np.median(v), min(v), max(v)))
