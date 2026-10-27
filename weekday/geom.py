"""Coordinate-free geometry toolkit + nulls (prereg: results/weekday_calib_prereg.md)."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def pca(X, d):
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    d = min(d, len(s))
    return Xc @ Vt[:d].T


# ---------------------------------------------------------------- point-level
def twonn(X, discard=0.1):
    D, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    r1, r2 = D[:, 1], D[:, 2]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu); keep = int(np.floor(n * (1 - discard)))
    F = np.arange(1, n + 1) / n
    x, y = np.log(mu[:keep]), -np.log(1 - F[:keep])
    return float((x @ y) / (x @ x))


def geodesic(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = nn.kneighbors(X)
    n = len(X)
    G = csr_matrix((dist[:, 1:].ravel(), (np.repeat(np.arange(n), k), ind[:, 1:].ravel())), shape=(n, n))
    G = G.maximum(G.T)
    ncomp, _ = connected_components(G, directed=False)
    return dijkstra(G, directed=False), ncomp


def eucl(X):
    s = (X * X).sum(1)
    return np.sqrt(np.maximum(s[:, None] + s[None] - 2 * X @ X.T, 0))


def distortion(DG, DE):
    iu = np.triu_indices(len(DG), 1)
    g, e = DG[iu], DE[iu]; ok = np.isfinite(g)
    return float(1 - spearmanr(g[ok], e[ok])[0]), float(ok.mean())


# ---------------------------------------------------------------- class-level
def circ_lin(C):
    i = np.arange(C)
    lin = np.abs(i[:, None] - i[None]); circ = np.minimum(lin, C - lin)
    return circ, lin


def cyclicity(Dc):
    C = len(Dc); circ, lin = circ_lin(C); iu = np.triu_indices(C, 1)
    d = Dc[iu]
    if not np.all(np.isfinite(d)):
        return np.nan, np.nan
    rc, rl = spearmanr(d, circ[iu])[0], spearmanr(d, lin[iu])[0]
    return float(rc - rl), float(rc)


def class_D(DG, DE_c, y, C):
    DGc = np.zeros((C, C))
    for a in range(C):
        for b in range(C):
            if a != b:
                blk = DG[np.ix_(y == a, y == b)]
                DGc[a, b] = np.mean(blk[np.isfinite(blk)]) if np.isfinite(blk).any() else np.inf
    return DGc


def centroids(X, y, C):
    return np.stack([X[y == c].mean(0) for c in range(C)])


def ang(u, v):
    return np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300), -1, 1)))


def _splits(y, C, rng, nsplit, tmpl):
    """split halves. With template ids: A = 3 of the 6 templates (all 10 distinct partitions), so both halves contain
    the same (z, k) combinations; random within-class splits make the halves' nuisance composition complementary
    (anti-correlated centroid noise; amendment A3)."""
    if tmpl is not None:
        from itertools import combinations
        T = np.unique(tmpl); parts = [c for c in combinations(T, len(T) // 2) if T[0] in c]
        for c in parts:
            yield np.isin(tmpl, c)
        return
    for s in range(nsplit):
        A = np.zeros(len(y), bool)
        for k in range(C):
            idx = np.where(y == k)[0]; p = rng.permutation(idx); A[p[:len(p) // 2]] = True
        yield A


def curve_stats(X, y, C, rng, nsplit=50, tmpl=None):
    c = centroids(X, y, C)
    Dl = np.roll(c, -1, 0) - c                       # Δ_i = c_{i+1} - c_i (closed)
    naive = np.array([ang(Dl[i], Dl[(i + 1) % C]) for i in range(C)])
    chord2 = np.array([np.linalg.norm(c[(i + 2) % C] - c[i]) / (np.linalg.norm(Dl[i]) + np.linalg.norm(Dl[(i + 1) % C])) for i in range(C)])
    chord3 = np.array([np.linalg.norm(c[(i + 3) % C] - c[i]) / sum(np.linalg.norm(Dl[(i + j) % C]) for j in range(3)) for i in range(C)])
    # split-half corrected turning + cross-covariance spectrum
    cosc, sh1, sh2, CL = [], [], [], []
    for A in _splits(y, C, rng, nsplit, tmpl):
        if min(np.bincount(y[A], minlength=C).min(), np.bincount(y[~A], minlength=C).min()) == 0:
            continue
        cA, cB = centroids(X[A], y[A], C), centroids(X[~A], y[~A], C); CL.append((cA, cB))
        dA, dB = np.roll(cA, -1, 0) - cA, np.roll(cB, -1, 0) - cB
        num = 0.5 * ((dA * np.roll(dB, -1, 0)).sum(1) + (dB * np.roll(dA, -1, 0)).sum(1))
        ii = (dA * dB).sum(1); den = ii * np.roll(ii, -1)
        cosc.append(np.where(den > 0, num / np.sqrt(np.abs(den)), np.nan))
        a, b = cA - cA.mean(0), cB - cB.mean(0)
        M = 0.5 * (a @ b.T + b @ a.T)
        ev = np.sort(np.linalg.eigvalsh(M))[::-1]; ev = np.maximum(ev, 0)
        tot = ev.sum()
        sh1.append(ev[0] / tot if tot > 0 else np.nan); sh2.append(ev[:2].sum() / tot if tot > 0 else np.nan)
    cosc = np.nanmean(np.array(cosc), 0)
    corr = np.degrees(np.arccos(np.clip(cosc, -1, 1)))
    adj, wind = circ_order(CL, C)
    return dict(adj=adj, wind=wind, turn_naive_med=float(np.median(naive)), turn_corr_med=float(np.nanmedian(corr)), turn_corr_max=float(np.nanmax(corr)) if np.isfinite(corr).any() else np.nan,
                chord2_med=float(np.median(chord2)),
                turn_naive=float(np.mean(naive)), turn_corr=float(np.nanmean(corr)), turn_corr_nan=float(np.isnan(cosc).mean()),
                turn_sum_naive=float(np.sum(naive)), chord2=float(np.mean(chord2)), chord3=float(np.mean(chord3)),
                top1=float(np.nanmean(sh1)) if np.isfinite(sh1).any() else np.nan, top2=float(np.nanmean(sh2)) if np.isfinite(sh2).any() else np.nan,
                rel=float(np.mean(np.isfinite(sh1))))


def circ_order(c_list, C):
    """S7 (amendment A4): cyclic order of classes in the reliable top-2 subspace.
    c_list = [(cA, cB), ...] split-half centroids. Embedding = top-2 eigvecs of the averaged symmetrised cross-Gram.
    adj = fraction of true neighbours (i, i+1) that are neighbours in angular order; wind = |signed sum of angle steps
    i->i+1| / 360 (1 for a circle traversed once in order)."""
    M = 0
    for cA, cB in c_list:
        a, b = cA - cA.mean(0), cB - cB.mean(0); M = M + 0.5 * (a @ b.T + b @ a.T)
    w, V = np.linalg.eigh(M / len(c_list)); o = np.argsort(w)[::-1][:2]
    P = V[:, o] * np.sqrt(np.maximum(w[o], 0))
    th = np.arctan2(P[:, 1], P[:, 0])
    order = np.argsort(th); pos = np.empty(C, int); pos[order] = np.arange(C)
    adj = np.mean([min((pos[i] - pos[(i + 1) % C]) % C, (pos[(i + 1) % C] - pos[i]) % C) == 1 for i in range(C)])
    st = np.angle(np.exp(1j * (np.roll(th, -1) - th)))
    return float(adj), float(abs(st.sum()) / (2 * np.pi))


def local_field(X, y, C, m=10):
    """v_i(x) for x in class i: mean of m nearest class-(i+1) pts - mean of m nearest class-i pts (excl. self)."""
    means, coh = [], []
    for i in range(C):
        j = (i + 1) % C
        Xi, Xj = X[y == i], X[y == j]
        mi, mj = min(m, len(Xi) - 1), min(m, len(Xj))
        _, ii = NearestNeighbors(n_neighbors=mi + 1).fit(Xi).kneighbors(Xi)
        _, jj = NearestNeighbors(n_neighbors=mj).fit(Xj).kneighbors(Xi)
        V = Xj[jj].mean(1) - Xi[ii[:, 1:]].mean(1)
        mv = V.mean(0); means.append(mv)
        coh.append(np.mean(V @ mv / (np.linalg.norm(V, axis=1) * np.linalg.norm(mv) + 1e-300)))
    rot = [ang(means[i], means[(i + 1) % C]) for i in range(C)]
    return dict(field_rot_med=float(np.median(rot)), field_rot=float(np.mean(rot)), field_coh=float(np.mean(coh)))


def all_stats(X, y, C, k, rng, point=True, tmpl=None):
    out = {}
    DE = eucl(X)
    DG, ncomp = geodesic(X, k)
    if point:
        out["id"] = twonn(X)
        out["distortion"], out["connected"] = distortion(DG, DE)
        out["ncomp"] = int(ncomp)
    c = centroids(X, y, C)
    DEc = eucl(c); DGc = class_D(DG, DEc, y, C)
    out["cyc_E"], out["rcirc_E"] = cyclicity(DEc)
    out["cyc_G"], out["rcirc_G"] = cyclicity(DGc)
    out.update(curve_stats(X, y, C, rng, nsplit=20, tmpl=tmpl))
    out.update(local_field(X, y, C))
    return out


# ---------------------------------------------------------------- nulls
def null_flatG(X, y, rng):
    S = np.cov(X.T) + 1e-9 * np.eye(X.shape[1])
    return rng.multivariate_normal(X.mean(0), S, size=len(X), method="cholesky"), y


def null_line(X, y, C, rng):
    c = centroids(X, y, C); m = c.mean(0)
    _, _, Vt = np.linalg.svd(c - m, full_matrices=False); u = Vt[0]
    cl = m + np.outer((c - m) @ u, u)
    R = X - c[y]
    return cl[y] + R[rng.permutation(len(X))], y


def null_line2(X, y, C, rng):
    """A5 (post hoc): SNR-matched unrolled line. Class i centroid = m + (i - mean_i) * s * u, u = top PC of centroids,
    s chosen so the total centroid spread equals the data's; residuals pooled and permuted. This is the canonical
    'flat ordinal concept' (one straight direction, equally spaced) with the data's own between/within ratio."""
    c = centroids(X, y, C); m = c.mean(0)
    _, _, Vt = np.linalg.svd(c - m, full_matrices=False); u = Vt[0]
    t = np.arange(C) - (C - 1) / 2; s = np.sqrt(((c - m) ** 2).sum() / (t ** 2).sum())
    cl = m + np.outer(t * s, u)
    R = X - c[y]
    return cl[y] + R[rng.permutation(len(X))], y


def null_blob(X, y, C, rng):
    c = centroids(X, y, C); R = X - c[y]
    S = np.cov(R.T) + 1e-9 * np.eye(X.shape[1])
    return c[y] + rng.multivariate_normal(np.zeros(X.shape[1]), S, size=len(X), method="cholesky"), y


def null_shuffle(X, y, rng):
    return X, rng.permutation(y)


def null_resample(X, y, C, rng):
    idx = np.concatenate([rng.choice(np.where(y == c)[0], (y == c).sum()) for c in range(C)])
    return X[idx] + 1e-6 * rng.standard_normal((len(idx), X.shape[1])), y[idx]
