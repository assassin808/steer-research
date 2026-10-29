"""Planted-curvature benchmark core (prereg results/power_curve_prereg.md).
Backgrounds, shapes, sampling, and the centroid statistics C1-C4 computed from split-half Grams, so that one background draw serves
every (shape, SNR) by common random numbers."""
import os, json, itertools
import numpy as np

R = os.path.expanduser("~/scratch/power")
SNRS = np.array([1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0])
SHAPES = ["arc0", "arc30", "arc60", "arc90", "arc180", "arc309", "arc360", "sat", "bend60", "bend90", "clust60", "clust90", "clust180"]
REP = 5          # q mode: replicates per (question, level)


# ------------------------------------------------------------------ backgrounds
class BG:
    def __init__(self, name):
        z = np.load(f"{R}/bg_{name}.npz")
        self.name, self.X, self.g = name, z["X"], z["g"]
        self.info = json.loads(str(z["info"]))
        self.N, self.r = self.X.shape
        self.W = float((self.X ** 2).sum(1).mean())          # X is centred
        ug = np.unique(self.g); self.ug = ug
        self.mu = np.stack([self.X[self.g == u].mean(0) for u in ug])
        gi = np.searchsorted(ug, self.g)
        self.eps = self.X - self.mu[gi]                      # within-group residual pool
        from sklearn.cluster import KMeans
        km = KMeans(2, random_state=0, n_init=10).fit(self.X)
        c0, c1 = km.cluster_centers_
        self.cw = c1 - c0; self.cb = -0.5 * (c1 + c0) @ self.cw   # member = x.cw + cb > 0
        self.clu = (self.X @ self.cw + self.cb > 0).astype(float)

    def plane(self, kind, rng):
        if kind == "top":
            P = np.zeros((self.r, 2)); P[0, 0] = P[1, 1] = 1; return P
        A = rng.standard_normal((self.r, 2)); return np.linalg.qr(A)[0]

    def sample(self, L, n, mode, rng):
        """rows [L, n, r], cluster membership [L, n], question index per row [L, n] (iid: -1), replicate id [L, n]"""
        if mode == "iid":
            tot = L * n
            idx = rng.permutation(self.N)[:tot] if tot <= self.N else rng.integers(0, self.N, tot)
            idx = idx.reshape(L, n)
            return self.X[idx], self.clu[idx], np.full((L, n), -1), np.tile(np.arange(n), (L, 1))
        Q = n // REP
        qs = rng.permutation(len(self.ug))[:Q] if Q <= len(self.ug) else rng.integers(0, len(self.ug), Q)
        e = rng.integers(0, self.N, (L, Q, REP))
        Xr = self.mu[qs][None, :, None, :] + self.eps[e]                   # [L, Q, REP, r]
        Xr = Xr.reshape(L, Q * REP, self.r)
        clu = (Xr @ self.cw + self.cb > 0).astype(float)
        qid = np.tile(np.repeat(np.arange(Q), REP), (L, 1)); rep = np.tile(np.tile(np.arange(REP), Q), (L, 1))
        return Xr, clu, qid, rep


# ------------------------------------------------------------------ shapes (unit-scale curves in plane coords)
def curve(shape, t):
    t = np.asarray(t, float)
    if shape.startswith("arc"):
        th = np.radians(float(shape[3:]) if shape != "arc309" else 360 * 6 / 7)
        if th == 0:
            return np.stack([t, 0 * t], -1)
        return np.stack([np.sin(th * t) / th, (1 - np.cos(th * t)) / th], -1)
    if shape == "sat":
        return np.stack([(1 - np.exp(-3 * t)) / (1 - np.exp(-3)), 0 * t], -1)
    if shape.startswith("bend"):
        th = np.radians(float(shape[4:])); u2 = np.array([np.cos(th), np.sin(th)])
        a = np.minimum(t, .5)[:, None] * np.array([1, 0]) + np.maximum(t - .5, 0)[:, None] * u2
        return a
    raise ValueError(shape)


def clust_dirs(shape):
    th = np.radians(float(shape[5:]))
    return np.array([1.0, 0.0]), np.array([np.cos(th), np.sin(th)])


def amplitude(shape, L, snr, W):
    """a such that (1/L) sum ||a g_l - mean||^2 / W = snr; clust uses the straight-line amplitude (per-point energy matched)."""
    t = np.linspace(0, 1, L)
    g = curve("arc0" if shape.startswith("clust") else shape, t)
    s = ((g - g.mean(0)) ** 2).sum() / L
    return np.sqrt(snr * W / s) if s > 0 else 0.0


def shift_means(shape, L, frac1):
    """mean planted shift per level for a set of rows with cluster-1 fraction frac1 [L] -> [L, 2] (unit amplitude)."""
    t = np.linspace(0, 1, L)
    if shape.startswith("clust"):
        u0, u1 = clust_dirs(shape)
        return t[:, None] * ((1 - frac1)[:, None] * u0 + frac1[:, None] * u1)
    return curve(shape, t)


def row_shift(shape, L, clu):
    """per-row planted shift [L, n, 2] (unit amplitude)"""
    t = np.linspace(0, 1, L)
    if shape.startswith("clust"):
        u0, u1 = clust_dirs(shape)
        return t[:, None, None] * ((1 - clu)[..., None] * u0 + clu[..., None] * u1)
    return np.repeat(curve(shape, t)[:, None, :], clu.shape[1], 1)


# ------------------------------------------------------------------ splits
def splits(L, n, qid, rep, rng, ns):
    """boolean masks [ns, L, n] for half A. iid: random half within level. q: per question, 2 of its 5 replicates
    (the same replicate ids at every level, so question composition is identical across levels)."""
    M = np.zeros((ns, L, n), bool)
    for s in range(ns):
        if qid[0, 0] < 0:
            for l in range(L):
                M[s, l, rng.permutation(n)[: n // 2]] = True
        else:
            Q = qid.max() + 1
            pick = np.argsort(rng.random((Q, REP)), 1)[:, :2]               # replicate ids in half A, per question
            sel = np.zeros((Q, REP), bool); np.put_along_axis(sel, pick, True, 1)
            M[s] = sel[qid, rep]
    return M


# ------------------------------------------------------------------ centroid statistics from Grams
def poly_contrasts(L):
    t = np.linspace(0, 1, L); V = np.stack([np.ones(L), t, t ** 2], 1)
    Qm, _ = np.linalg.qr(V)
    p1, p2 = Qm[:, 1], Qm[:, 2]
    if p1 @ t < 0: p1 = -p1
    return p1, p2


_PERM = {}


def perms(L):
    if L not in _PERM:
        _PERM[L] = np.array(list(itertools.permutations(range(L))))
    return _PERM[L]


def stats_from_gram(G, L, wA, ns_axis=0):
    """G [..., ns, 2L, 2L] Gram of stacked half centroids [cA_0..cA_{L-1}, cB_0..cB_{L-1}]; wA = weight of half A in the full centroid.
    returns dict of arrays [...] (averaged over splits as in the weekday kit)."""
    GAB = G[..., :L, L:]; GBA = G[..., L:, :L]; GAA = G[..., :L, :L]; GBB = G[..., L:, L:]
    H = np.eye(L) - 1.0 / L
    MAB = H @ GAB @ H; M = 0.5 * (MAB + np.swapaxes(MAB, -1, -2))          # reliable centred cross Gram
    out = {}
    # C1 turning (open polyline): kit formula
    D = np.diff(np.eye(L), axis=0)                                           # [L-1, L]
    SAB = D @ GAB @ D.T; SBA = D @ GBA @ D.T
    ii = np.diagonal(SAB, axis1=-2, axis2=-1)
    num = 0.5 * (np.diagonal(SAB, 1, -2, -1) + np.diagonal(SBA, 1, -2, -1))
    den = ii[..., :-1] * ii[..., 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        cs = np.where(den > 0, num / np.sqrt(np.abs(den)), np.nan)
        csm = np.nanmean(cs, axis=-2)                                        # mean over splits
        ang = np.degrees(np.arccos(np.clip(csm, -1, 1)))
        out["turn"] = np.nanmedian(ang, -1)        # kit: nan vertices ignored; all-nan -> nan (counted as 'not detected')
    # C2 offline share
    ev = np.linalg.eigvalsh(M); ev = np.maximum(ev, 0); tot = ev.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        top1 = np.where(tot > 0, ev[..., -1] / tot, np.nan)
    out["offline"] = 1 - np.nanmean(top1, -1)
    # C3 quad-perp
    p1, p2 = poly_contrasts(L)
    z = np.zeros(L)
    vA2, vB2 = np.r_[p2, z], np.r_[z, p2]; vA1, vB1 = np.r_[p1, z], np.r_[z, p1]
    v1 = np.r_[wA * p1, (1 - wA) * p1]
    q = lambda a, b: np.einsum("i,...ij,j->...", a, G, b)
    b11 = q(v1, v1)
    numq = q(vA2, vB2) - q(vA2, v1) * q(v1, vB2) / b11
    out["quad"] = numq.mean(-1) / b11.mean(-1)
    # C4 order shares on the split-averaged reliable Gram, with the exact order-relabel null
    Mb = M.mean(-3)
    trM = np.trace(Mb, axis1=-2, axis2=-1)
    Pm = perms(L)
    i = np.arange(L)
    u, v = np.cos(2 * np.pi * i / L), np.sin(2 * np.pi * i / L); u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    for nm, vecs in [("ord1", [p1]), ("f1c", [u, v])]:
        def share(Mx):
            return sum(np.einsum("i,...ij,j->...", w, Mx, w) for w in vecs)
        s0 = share(Mb)
        # permuted: s(pi) = sum_w w_pi^T M w_pi with (w_pi)_k = w[pi^-1(k)]
        cnt = 0
        for w in vecs:
            Wp = np.zeros((len(Pm), L)); Wp[np.arange(len(Pm))[:, None], Pm] = w[None, :]   # Wp[p, Pm[p,k]] = w[k]
            sp = np.einsum("pi,...ij,pj->...p", Wp, Mb, Wp)
            cnt = cnt + sp
        with np.errstate(invalid="ignore", divide="ignore"):
            out[nm] = s0 / trM
            out[nm + "_p"] = (cnt >= s0[..., None] - 1e-12).mean(-1)
    return out


def half_centroids(Xr, M):
    """Xr [L, n, r], M [ns, L, n] -> stacked [ns, 2L, r] (A then B), and wA"""
    MA = M.astype(float); MB = 1 - MA
    cA = np.einsum("sln,lnr->slr", MA, Xr) / MA.sum(-1, keepdims=True)
    cB = np.einsum("sln,lnr->slr", MB, Xr) / MB.sum(-1, keepdims=True)
    wA = float(MA[0].sum() / MA[0].size)
    return np.concatenate([cA, cB], 1), wA
