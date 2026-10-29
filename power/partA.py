"""Centroid grid (C1-C4). usage: python partA.py [--sims 400 --procs 16 --bgs ...]"""
import os, sys, argparse, time
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[k] = "1"
import numpy as np, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import core as C

OUT = f"{C.R}/partA"
STATS = ["turn", "offline", "quad", "ord1", "ord1_p", "f1c", "f1c_p"]


def run_cell(args):
    bgname, plane, L, n, mode, sims, seed = args
    fn = f"{OUT}/{bgname}_{plane}_L{L}_n{n}_{mode}.npz"
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    bg = C.BG(bgname); rng = np.random.default_rng(seed)
    K = len(C.SHAPES) * len(C.SNRS)
    amp = np.array([[C.amplitude(sh, L, s, bg.W) for s in C.SNRS] for sh in C.SHAPES])   # [nsh, nsnr]
    res = {k: np.zeros((sims, len(C.SHAPES), len(C.SNRS))) for k in STATS}
    res0 = {k: np.zeros(sims) for k in STATS}
    for it in range(sims):
        Xr, clu, qid, rep = bg.sample(L, n, mode, rng)
        M = C.splits(L, n, qid, rep, rng, 10)
        E, wA = C.half_centroids(Xr, M)                      # [ns, 2L, r]
        P = bg.plane(plane, rng)
        G0 = E @ np.swapaxes(E, 1, 2); Pr = E @ P            # [ns,2L,2L], [ns,2L,2]
        MA = M.astype(float)
        fA = (MA * clu).sum(-1) / MA.sum(-1); fB = ((1 - MA) * clu).sum(-1) / (1 - MA).sum(-1)    # [ns, L]
        U = np.zeros((len(C.SHAPES), 10, 2 * L, 2))
        for j, sh in enumerate(C.SHAPES):
            for s in range(10):
                U[j, s, :L] = C.shift_means(sh, L, fA[s]); U[j, s, L:] = C.shift_means(sh, L, fB[s])
        S = amp[:, :, None, None, None] * U[:, None]          # [nsh, nsnr, ns, 2L, 2]
        G = G0 + Pr @ np.swapaxes(S, -1, -2) + S @ np.swapaxes(Pr, -1, -2) + S @ np.swapaxes(S, -1, -2)
        o = C.stats_from_gram(G, L, wA); o0 = C.stats_from_gram(G0, L, wA)
        for k in STATS:
            res[k][it] = o[k]; res0[k][it] = o0[k]
    np.savez(fn, **res, **{k + "_snr0": v for k, v in res0.items()}, amp=amp, W=bg.W, secs=time.time() - t0)
    print(f"{fn} {time.time() - t0:.0f}s", flush=True)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=400); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--bgs", default="tqa13_5,syc,opt,wk,wkz,tqa21_6,tqa15_2")
    ap.add_argument("--ns", default="20,50,100,300,1000"); ap.add_argument("--Ls", default="5,7")
    ap.add_argument("--planes", default="rand,top"); ap.add_argument("--modes", default="iid,q")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for b in a.bgs.split(","):
        for pl in a.planes.split(","):
            for L in map(int, a.Ls.split(",")):
                for n in map(int, a.ns.split(",")):
                    for m in a.modes.split(","):
                        jobs.append((b, pl, L, n, m, a.sims, hash((b, pl, L, n, m)) % 2 ** 31))
    jobs.sort(key=lambda j: -j[3])   # big n first
    with Pool(a.procs) as p:
        for _ in p.imap_unordered(run_cell, jobs):
            pass


if __name__ == "__main__":
    main()
