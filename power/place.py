"""Real-data placement: class SNR (fourier.py / tqa_snr.py definition: centroid power / C / mean sq. within-class distance),
plus the question share of within-class variance. Writes ~/scratch/power/place.json"""
import os, json, numpy as np
H = os.path.expanduser("~")


def snr(X, y):
    cl = np.unique(y); c = np.stack([X[y == k].mean(0) for k in cl]); yi = np.searchsorted(cl, y)
    W = np.mean(((X - c[yi]) ** 2).sum(1))
    return float(((c - c.mean(0)) ** 2).sum() / len(cl) / W)


def qshare(X, y, g):
    R = X.copy()
    for k in np.unique(y): R[y == k] -= R[y == k].mean(0)
    tot = (R ** 2).sum(); gm = 0
    for u in np.unique(g):
        m = g == u; gm += m.sum() * (R[m].mean(0) ** 2).sum()
    return float(gm / tot)


out = {}
m = np.load(f"{H}/scratch/ot_geom/meta.npz"); lab, q, tr = m["label"], m["q"], m["is_train"]
heads = json.load(open(f"{H}/steer_math/ot_geometry/heads.json"))["top24"]
v, qs = [], []
for l, h in heads:
    X = np.load(f"{H}/scratch/ot_geom/heads/L{l}H{h}.npy").astype(float); Z = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-8)
    v.append(snr(Z[tr], lab[tr])); qs.append(qshare(Z[tr], lab[tr], q[tr]))
out["tqa_top24"] = dict(L=2, n_per_class=[int((lab[tr] == 1).sum()), int((lab[tr] == -1).sum())], snr_med=float(np.median(v)), snr_min=float(min(v)), snr_max=float(max(v)), qshare_med=float(np.median(qs)))
for tag, nm in [("gemma", "wk_last"), ("gemmaztok", "wk_ztok")]:
    z = np.load(f"{H}/scratch/weekday/{tag}_weekday.npz")
    keep = np.sort(np.unique(z["H"][:, z["H"].shape[1] // 2].round(4), axis=0, return_index=True)[1])
    Hh, zz = z["H"][keep], z["z"][keep]
    s = [snr(Hh[:, l].astype(float), zz) for l in range(6, 19)]
    out[nm] = dict(L=7, n_per_class=int(len(keep) / 7), snr_mid_med=float(np.median(s)), snr_L12=float(s[6]))
for tr_ in ["sycophantic", "optimistic"]:
    A = np.load(f"{H}/steer_poc/v31/out/{tr_}/w23_acts.npz"); nq, npar = A["pos"].shape[:2]
    rows = {}
    for li, Lc in enumerate([3, 7, 10, 14, 17, 20, 24]):
        X = np.concatenate([A["pos"][:, :, li].reshape(nq * npar, -1), A["neg"][:, :, li].reshape(nq * npar, -1)]).astype(float)
        y = np.r_[np.ones(nq * npar), np.zeros(nq * npar)]; g = np.tile(np.repeat(np.arange(nq), npar), 2)
        rows[Lc] = dict(snr=snr(X, y), qshare=qshare(X, y, g))
    out[tr_] = dict(L=2, n_per_class=nq * npar, nq=nq, layers=rows)
json.dump(out, open(f"{H}/scratch/power/place.json", "w"), indent=1)
print(json.dumps(out, indent=1))
