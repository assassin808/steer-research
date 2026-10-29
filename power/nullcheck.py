"""Practical-null check (prereg 'Practical tests'): C1-C3 (and P1 frot) against the data-driven LINE null:
centroids projected on the top PC of the observed centred centroids (projected positions kept, so unequal spacing is kept),
residual rows permuted across levels (within question in q mode). p = (1 + #null >= obs) / (1 + reps).
usage: python nullcheck.py [--procs 12]"""
import os, sys, argparse, time
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[k] = "1"
import numpy as np, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import core as C

OUT = f"{C.R}/nullcheck"
NULLF = [None]   # set in main: line_null (prereg) or perp_null (amendment A1)
SH = ["arc0", "sat", "arc90", "clust90"]
SN = [0.03, 0.3, 3.0]


def cstats(X, L, n, qid, rep, rng, ns=10):
    M = C.splits(L, n, qid, rep, rng, ns)
    E, wA = C.half_centroids(X, M)
    o = C.stats_from_gram(E @ np.swapaxes(E, 1, 2), L, wA)
    return np.array([o["turn"], o["offline"], o["quad"]], float)


def line_null(X, qid, rng):
    L, n, r = X.shape
    c = X.mean(1); m = c.mean(0)
    u = np.linalg.svd(c - m, full_matrices=False)[2][0]
    cl = m + np.outer((c - m) @ u, u)
    R = X - c[:, None]
    Rf = R.reshape(L * n, r); q = qid.reshape(-1)
    if q[0] < 0:
        perm = rng.permutation(L * n)
    else:
        perm = np.arange(L * n)
        for qq in np.unique(q):
            idx = np.where(q == qq)[0]; perm[idx] = rng.permutation(idx)
    return cl[:, None] + Rf[perm].reshape(L, n, r)


def run(args):
    bgname, L, n, mode, sims, reps, seed, do_point = args
    fn = f"{OUT}/{bgname}_L{L}_n{n}_{mode}{'_P' if do_point else ''}.npz"
    if os.path.exists(fn): return fn
    t0 = time.time(); bg = C.BG(bgname); rng = np.random.default_rng(seed)
    sn = SN if not do_point else [0.3, 3.0]
    pv = np.full((sims, len(SH), len(sn), 4 if do_point else 3), np.nan)
    if do_point:
        import partP
    for it in range(sims):
        Xr, clu, qid, rep = bg.sample(L, n, mode, rng); P = bg.plane("rand", rng)
        for j, sh in enumerate(SH):
            U = C.row_shift(sh, L, clu)
            for k, s in enumerate(sn):
                X = Xr + C.amplitude(sh, L, s, bg.W) * (U @ P.T)
                if not do_point:
                    obs = cstats(X, L, n, qid, rep, rng)
                    nul = np.array([cstats(NULLF[0](X, qid, rng), L, n, qid, rep, rng) for _ in range(reps)])
                    pv[it, j, k] = (1 + np.sum(np.nan_to_num(nul, nan=-np.inf) >= obs[None], 0)) / (1 + reps)
                    pv[it, j, k][~np.isfinite(obs)] = 1.0
                else:
                    obs = partP.field(partP.pca64(X.reshape(L * n, -1)), L, n)[0]
                    nul = np.array([partP.field(partP.pca64(line_null(X, qid, rng).reshape(L * n, -1)), L, n)[0] for _ in range(reps)])
                    pv[it, j, k, 3] = (1 + np.sum(nul >= obs)) / (1 + reps)
        if it % 10 == 9: np.savez(fn + ".part.npz", pv=pv[:it + 1])
    np.savez(fn, pv=pv, shapes=np.array(SH), snrs=np.array(sn), secs=time.time() - t0)
    print(fn, f"{time.time() - t0:.0f}s", flush=True)
    return fn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--null", default="line", choices=["line", "perp"]); ap.add_argument("--ns", default="50,300"); a = ap.parse_args()
    global OUT
    if a.null == "perp": OUT = f"{C.R}/nullcheck_perp"
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for b, mode in [("tqa13_5", "iid"), ("syc", "q")]:
        for n in map(int, a.ns.split(",")):
            for ch in range(4):   # 4 chunks x 25 sims = 100
                jobs.append((b, 7, n, mode + f"c{ch}", 25, 99, hash((b, n, ch)) % 2 ** 31, False))
    if a.null == "line": jobs.append(("tqa13_5", 7, 50, "iid", 40, 39, 7, True))
    jobs = [(j[0], j[1], j[2], j[3], j[4], j[5], j[6], j[7]) for j in jobs]
    with Pool(a.procs, initializer=_init, initargs=(a.null, OUT)) as p:
        for _ in p.imap_unordered(run2, jobs): pass


def _init(nl, out):
    global OUT
    NULLF[0] = line_null if nl == "line" else perp_null; OUT = out


def run2(j):
    b, L, n, mode, sims, reps, seed, dp = j
    m = mode[:-2] if mode[-2] == "c" else mode
    fnmode = mode
    global OUT
    # mode tag carries the chunk id; sampling uses the real mode
    a = (b, L, n, m, sims, reps, seed, dp)
    old = OUT; OUT = f"{old}/{fnmode}"; os.makedirs(OUT, exist_ok=True)
    try:
        return run(a)
    finally:
        OUT = old


def perp_null(X, qid, rng):
    """TRIED, NOT ADOPTED (post hoc; line null FPR was .07-.10 at n = 50; this perp null does not fix it, see power_curve.md):
    keep every row's component along u (top PC of the observed centred centroids) exactly, permute only the orthogonal
    component x_perp across levels (within question in q mode). The Gram splits exactly into M_par + M_perp, so the along-line
    spacing and its split-half noise are kept; under a straight line, x_perp is exchangeable across levels."""
    L, n, r = X.shape
    c = X.mean(1); m = c.mean(0)
    u = np.linalg.svd(c - m, full_matrices=False)[2][0]
    Xf = X.reshape(L * n, r) - m
    par = np.outer(Xf @ u, u); perp = Xf - par
    q = qid.reshape(-1)
    if q[0] < 0:
        perm = rng.permutation(L * n)
    else:
        perm = np.arange(L * n)
        for qq in np.unique(q):
            idx = np.where(q == qq)[0]; perm[idx] = rng.permutation(idx)
    return (m + par + perp[perm]).reshape(L, n, r)


if __name__ == "__main__":
    main()
