"""Build background clouds (prereg results/power_curve_prereg.md). Real class means removed; rows expressed in full-rank PCA
coordinates (an isometry of the centred row span, so every Euclidean / inner-product statistic is unchanged).
Writes ~/scratch/power/bg_<id>.npz: X [N, r] float64, g [N] group ids, plus W (trace cov), PR, question share of W."""
import os, json, numpy as np
R = os.path.expanduser("~/scratch/power"); os.makedirs(R, exist_ok=True)
OG = os.path.expanduser("~/scratch/ot_geom"); WK = os.path.expanduser("~/scratch/weekday")
V31 = os.path.expanduser("~/steer_poc/v31/out")


def demean_by(X, lab):
    X = X.copy()
    for c in np.unique(lab):
        m = lab == c; X[m] -= X[m].mean(0)
    return X


def save(name, X, g, info):
    X = X - X.mean(0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    keep = s > s[0] * 1e-7
    Z = U[:, keep] * s[keep]
    ev = s[keep] ** 2 / len(X)
    W = float(ev.sum()); PR = float(ev.sum() ** 2 / (ev ** 2).sum())
    gm = np.stack([Z[g == u].mean(0) for u in np.unique(g)]); cnt = np.array([(g == u).sum() for u in np.unique(g)])
    qshare = float((cnt[:, None] * gm ** 2).sum() / (Z ** 2).sum())
    info.update(N=len(Z), rank=int(keep.sum()), W=W, PR=PR, qshare=qshare, ngroups=int(len(cnt)))
    np.savez(f"{R}/bg_{name}.npz", X=Z, g=g, info=json.dumps(info))
    print(name, info, flush=True)


def main():
    m = np.load(f"{OG}/meta.npz"); lab, q, tr = m["label"], m["q"], m["is_train"]
    for l, h in [(13, 5), (21, 6), (15, 2)]:
        X = np.load(f"{OG}/heads/L{l}H{h}.npy").astype(np.float64)
        Z = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-8)          # ot_geometry z space
        save(f"tqa{l}_{h}", demean_by(Z, lab), q, dict(src=f"TQA L{l}H{h} z, POS/NEG means removed"))
    for tag, nm in [("gemma", "wk"), ("gemmaztok", "wkz")]:
        z = np.load(f"{WK}/{tag}_weekday.npz"); H = z["H"][:, 12].astype(np.float64)
        keep = np.sort(np.unique(z["H"][:, z["H"].shape[1] // 2].round(4), axis=0, return_index=True)[1])
        H, zz, k, t = H[keep], z["z"][keep], z["k"][keep], z["t"][keep]
        save(nm, demean_by(H, zz), (k * 10 + t).astype(int), dict(src=f"weekday {tag} L12, z-class means removed"))
    for tr_ in ["sycophantic", "optimistic"]:
        A = np.load(f"{V31}/{tr_}/w23_acts.npz"); li = [3, 7, 10, 14, 17, 20, 24].index(14)
        P, Nn = A["pos"][:, :, li].reshape(-1, A["pos"].shape[-1]), A["neg"][:, :, li].reshape(-1, A["pos"].shape[-1])
        nq, npar = A["pos"].shape[:2]
        X = np.concatenate([P, Nn]).astype(np.float64); lab2 = np.r_[np.ones(len(P)), np.zeros(len(Nn))]
        g = np.tile(np.repeat(np.arange(nq), npar), 2)
        save(tr_[:3], demean_by(X, lab2), g, dict(src=f"PoC v3.1 {tr_} w23 pos/neg L14, class means removed"))


if __name__ == "__main__":
    main()
