"""POST HOC S8: circular-Fourier share of the (split-half, noise-free) centroid Gram.
f1 = power of the first circular harmonic / total reliable centroid power. Perfect circle 1; regular simplex (pure identity)
2/(C-1); a straight equally spaced line .66 (7) / .63 (12) under the cyclic index - so f1 alone does not exclude a line:
we also report 'lin' = share explained by the linear index (ordinal line). Reference rows: circle f1 1 / lin .33(7);
line .66 / 1.0; simplex .33 / .17 (7). Nulls: label shuffle (class signal exists) and, after review, ORDER relabel
(permute class identities of the reliable Gram, M[p][:, p]: keeps the geometry, destroys the cyclic order) -> p_order.
usage: python fourier.py TAG DOM LABEL  (TAG = gemma | gemmaztok)
"""
import sys, os, json, itertools
import numpy as np

R = os.path.expanduser("~/scratch/weekday")


def gram_split(X, y, tm, C):
    T = np.unique(tm); M = 0; n = 0
    for c in itertools.combinations(T, len(T) // 2):
        if T[0] not in c:
            continue
        A = np.isin(tm, c)
        cA = np.stack([X[A & (y == k)].mean(0) for k in range(C)]); cB = np.stack([X[~A & (y == k)].mean(0) for k in range(C)])
        a, b = cA - cA.mean(0), cB - cB.mean(0); M = M + 0.5 * (a @ b.T + b @ a.T); n += 1
    return M / n


def shares(M, C):
    i = np.arange(C); tot = np.trace(M)
    out = {}
    for f in range(1, C // 2 + 1):
        u = np.cos(2 * np.pi * f * i / C); u /= np.linalg.norm(u)
        p = u @ M @ u
        if not (C % 2 == 0 and f == C // 2):
            v = np.sin(2 * np.pi * f * i / C); v /= np.linalg.norm(v); p += v @ M @ v
        out[f] = p / tot
    t = i - i.mean(); t /= np.linalg.norm(t)
    return out, float(t @ M @ t / tot), float(tot)


def main():
    tag, dom, lab = sys.argv[1:4]
    z = np.load(f"{R}/{tag}_{dom}.npz"); H, y, tm = z["H"], z[lab], z["t"]
    _, keep = np.unique(H[:, H.shape[1] // 2].round(4), axis=0, return_index=True); keep = np.sort(keep)
    H, y, tm = H[keep], y[keep], tm[keep]
    C = 7 if dom == "weekday" else 12
    rng = np.random.default_rng(0); res = []
    for l in range(1, H.shape[1]):
        X = H[:, l].astype(np.float64); X = X - X.mean(0)
        sh, lin, tot = shares(gram_split(X, y, tm, C), C)
        nul = []
        for r in range(20):
            yp = y.copy()
            for t in np.unique(tm):  # shuffle class labels consistently within template (keeps the split structure)
                m = tm == t; yp[m] = rng.permutation(y[m])
            s2, _, t2 = shares(gram_split(X, yp, tm, C), C); nul.append(s2[1] * t2)  # unnormalised f1 power
        M = gram_split(X, y, tm, C); f1d = sh[1]; cnt = 0
        for r in range(300):
            p = rng.permutation(C); s3, _, _ = shares(M[np.ix_(p, p)], C); cnt += s3[1] >= f1d - 1e-12
        p_order = (cnt + 1) / 301
        # reliability: reliable centroid power / total within-class variance
        W = np.mean(((X - np.stack([X[y == k].mean(0) for k in range(C)])[y]) ** 2).sum(1))
        res.append(dict(layer=l, f1=float(sh[1]), f2=float(sh[2]), lin=lin, p1=float(sh[1] * tot), p1_null_max=float(np.max(nul)), p1_null_med=float(np.median(nul)), p1_ratio=float(sh[1] * tot / np.max(nul)), p_order=float(p_order),
                        snr=float(tot / C / W)))
        print(f"L{l} f1 {sh[1]:.2f} f2 {sh[2]:.2f} lin {lin:.2f} p1/null-max {sh[1] * tot / np.max(nul):.1f} p_order {p_order:.3f} snr {tot / C / W:.4f}", flush=True)
    json.dump(res, open(f"{R}/fourier_{tag}_{dom}_{lab}.json", "w"))


if __name__ == "__main__":
    main()
