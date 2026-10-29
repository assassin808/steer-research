"""POST HOC (after adversarial review, objections 5/7/8): simulate the exact proposed persona design.
Q = 35 questions x r = 5 replicates x L = 5 levels on the syc/opt backgrounds.
- Planted plane: random, or containing the real pos-neg direction (dmu).
- A question x level cell effect of relative size kappa (E||cell||^2 = kappa * W_eps / r).
- Practical tests: line null with rows permuted within question ('row', prereg-style), and with whole cells permuted across levels
  within question ('cell', exchangeable under cell effects). p from 39 reps; rejection at p <= .02 and .05.
usage: python designcheck.py [--procs 4]"""
import os, sys, argparse, time, zlib
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[k] = "1"
import numpy as np, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import core as C
from nullcheck import cstats

OUT = f"{C.R}/designcheck"
V31 = os.path.expanduser("~/steer_poc/v31/out")
Q, REP, L = int(os.environ.get("DC_Q", 35)), int(os.environ.get("DC_R", 5)), 5
C.REP = REP


def make_bg(trait):
    A = np.load(f"{V31}/{trait}/w23_acts.npz"); li = 3                      # layer 14
    P, Nn = A["pos"][:, :, li].reshape(-1, A["pos"].shape[-1]), A["neg"][:, :, li].reshape(-1, A["pos"].shape[-1])
    dmu = P.mean(0) - Nn.mean(0)
    X = np.concatenate([P - P.mean(0), Nn - Nn.mean(0)]).astype(np.float64)
    nq, npar = A["pos"].shape[:2]; g = np.tile(np.repeat(np.arange(nq), npar), 2)
    m = X.mean(0); X = X - m
    U, s, Vt = np.linalg.svd(X, full_matrices=False); keep = s > s[0] * 1e-7
    Z = U[:, keep] * s[keep]; d = Vt[keep] @ dmu; d /= np.linalg.norm(d)
    bg = object.__new__(C.BG)
    bg.name, bg.X, bg.g = trait, Z, g; bg.N, bg.r = Z.shape; bg.W = float((Z ** 2).sum(1).mean())
    bg.ug = np.unique(g); bg.mu = np.stack([Z[g == u].mean(0) for u in bg.ug]); bg.eps = Z - bg.mu[np.searchsorted(bg.ug, g)]
    bg.cw = np.zeros(bg.r); bg.cb = -1.0; bg.clu = np.zeros(bg.N)
    return bg, d


def null_data(X, qid, rng, kind):
    """X [L, n, r]. line null: centroids projected on the top PC of the observed centroids; residuals permuted
    (row: rows within question; cell: whole (question, level) cells across levels within question)."""
    Lx, n, r = X.shape
    c = X.mean(1); m = c.mean(0); u = np.linalg.svd(c - m, full_matrices=False)[2][0]
    cl = m + np.outer((c - m) @ u, u); R = X - c[:, None]
    if kind == "row":
        Rf = R.reshape(Lx * n, r); q = qid.reshape(-1); perm = np.arange(Lx * n)
        for qq in np.unique(q):
            idx = np.where(q == qq)[0]; perm[idx] = rng.permutation(idx)
        return cl[:, None] + Rf[perm].reshape(Lx, n, r)
    R4 = R.reshape(Lx, Q, REP, r); out = np.empty_like(R4)
    for qq in range(Q):
        out[:, qq] = R4[rng.permutation(Lx), qq]
    return cl[:, None] + out.reshape(Lx, n, r)


def run(args):
    trait, plane, kappa, snrs, sims, reps, seed = args
    fn = f"{OUT}/{trait}_{plane}_k{kappa}_Q{Q}r{REP}_{seed}.npz"
    if os.path.exists(fn): return fn
    t0 = time.time(); bg, d = make_bg(trait); rng = np.random.default_rng(seed)
    n = Q * REP; shapes = ["arc0", "arc60", "bend90"]
    pv = np.full((sims, len(shapes), len(snrs), 2, 3), np.nan)              # [.., null kind, stat]
    for it in range(sims):
        Xr, clu, qid, rep = bg.sample(L, n, "q", rng)
        if kappa > 0:
            cell = np.sqrt(kappa / 5) * bg.eps[rng.integers(0, bg.N, (L, Q))]     # [L, Q, r]; absolute size: kappa x (5-sample cell-mean noise)
            Xr = Xr + np.repeat(cell, REP, axis=1)
        if plane == "dmu":
            v = rng.standard_normal(bg.r); v -= (v @ d) * d; v /= np.linalg.norm(v); P = np.stack([d, v], 1)
        else:
            P = bg.plane("rand", rng)
        for j, sh in enumerate(shapes):
            U = C.row_shift(sh, L, clu)
            for k, s in enumerate(snrs):
                X = Xr + C.amplitude(sh, L, s, bg.W) * (U @ P.T)
                obs = cstats(X, L, n, qid, rep, rng)
                for ki, kind in enumerate(["row", "cell"]):
                    nul = np.array([cstats(null_data(X, qid, rng, kind), L, n, qid, rep, rng) for _ in range(reps)])
                    pv[it, j, k, ki] = (1 + np.sum(np.nan_to_num(nul, nan=-np.inf) >= obs[None], 0)) / (1 + reps)
        np.savez(fn + ".part.npz", pv=pv[:it + 1])
    np.savez(fn, pv=pv, shapes=np.array(shapes), snrs=np.array(snrs), secs=time.time() - t0)
    print(fn, f"{time.time() - t0:.0f}s", flush=True)
    return fn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    snr = {"sycophantic": [0.009, 0.018], "optimistic": [0.0113, 0.0226]}     # x0.5 and x1 of 4 var(t) SNR_2 (L = 5)
    jobs = []
    for tr in (["sycophantic", "optimistic"] if os.environ.get("DC_SET") != "b" else ["sycophantic"]):
        for plane, kappa in ([("rand", 0.0), ("rand", 0.55), ("dmu", 0.55)] if os.environ.get("DC_SET") != "b" else [("rand", 0.55), ("dmu", 0.55)]):
                for ch in range(2):
                    jobs.append((tr, plane, kappa, snr[tr], 30, 39, zlib.crc32(f"{tr}{plane}{kappa}{ch}".encode())))
    with Pool(a.procs) as p:
        for _ in p.imap_unordered(run, jobs): pass


if __name__ == "__main__":
    main()
