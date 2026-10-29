"""Binary grid (B1 O1 flow-line turning, B2 1 - OT R2_s50), L = 2: level 0 = background rows, level 1 = background + planted shift.
Same code as weekday/old_stats.py (ot_geometry trait_flow / ot_geom). usage: python partB.py [--sims 40 --procs 12]"""
import os, sys, argparse, time
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[k] = "1"
import numpy as np, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
sys.path.insert(0, os.path.expanduser("~/steer_math/ot_geometry"))
import torch
torch.set_num_threads(1)
import trait_flow as T
from ot_geom import sinkhorn, flat_stats, null_sample
from sklearn.neighbors import NearestNeighbors
import core as C

OUT = f"{C.R}/partB"
SNRS = np.array([0.01, 0.1])          # budget cut (login node, ~4 cores per user): prereg grid .003-1
SHAPES = ["arc0", "clust90", "clust180"]   # budget cut: clust60 dropped
ALPHA = 100.0


def turn_at1(Zf, yf, Xs, Zpos_ho, HN, alpha=ALPHA):
    S = T.MLPScore(T.fit_mlp(Zf, yf, alpha))
    nnpos = NearestNeighbors(n_neighbors=10).fit(Zf[yf == 1])
    band = np.quantile(nnpos.kneighbors(Zpos_ho)[0][:, -1], 0.95)
    h = T.H_FRAC * HN; n = int(round(1.0 / T.H_FRAC))
    T.HN = HN
    res, _ = T.integrate(Xs, S, None, nnpos, band, h, n)
    return res["turn"]["1.0"]


def binary_stats(F0, F1, H0, H1, space, rng=None):
    """F*: fit rows per class, H*: held-out rows. returns (O1 turning, 1 - OT R2_s50)"""
    A = np.concatenate([F0, F1, H0, H1])
    if space == "pca64z":
        Ac = A - A.mean(0); w, V = np.linalg.eigh(Ac.T @ Ac) if Ac.shape[1] <= Ac.shape[0] else (None, None)
        if w is None:
            w2, U = np.linalg.eigh(Ac @ Ac.T); o = np.argsort(w2)[::-1][:64]; A = U[:, o] * np.sqrt(np.maximum(w2[o], 0))
        else:
            o = np.argsort(w)[::-1][:64]; A = Ac @ V[:, o]
    nf = len(F0) + len(F1)
    mu, sd = A[:nf].mean(0), A[:nf].std(0) + 1e-8
    A = (A - mu) / sd
    n0, n1, h0 = len(F0), len(F1), len(H0)
    Zf = A[:nf]; yf = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    Xs, Zp = A[nf:nf + h0], A[nf + h0:]
    dmu = Zf[yf == 1].mean(0) - Zf[yf == 0].mean(0); HN = float(np.linalg.norm(dmu))
    o1 = turn_at1(Zf, yf, Xs, Zp, HN)
    ot = sinkhorn(Zf[yf == 0], Zf[yf == 1], 0.01)
    r2s = flat_stats(Zf[yf == 0], ot["d"], dmu, k=min(50, n0 - 1))["r2s"]
    return o1, 1 - r2s


def draw(bg, n, nho, rng):
    tot = 2 * n + 2 * nho
    idx = rng.permutation(bg.N)[:tot] if tot <= bg.N else rng.integers(0, bg.N, tot)
    X = bg.X[idx]; clu = bg.clu[idx]
    if tot > bg.N:   # duplicates: tiny jitter so kNN ties break (flagged limitation)
        X = X + 1e-3 * np.sqrt(bg.W / bg.r) * rng.standard_normal(X.shape)
    s = np.cumsum([0, n, n, nho, nho])
    return [X[s[i]:s[i + 1]] for i in range(4)], [clu[s[i]:s[i + 1]] for i in range(4)]


def run_cell(args):
    bgname, n, sims, seed = args
    fn = f"{OUT}/{bgname}_n{n}.npz"
    if os.path.exists(fn):
        return fn
    t0 = time.time(); bg = C.BG(bgname); rng = np.random.default_rng(seed)
    space = "z" if bgname.startswith("tqa") else "pca64z"
    nho = 150 if bgname.startswith("tqa") else 75
    res = np.full((sims, len(SHAPES), len(SNRS), 2), np.nan)
    for it in range(sims):
        (F0, F1, H0, H1), (c0, c1, d0, d1) = draw(bg, n, nho, rng)
        P = bg.plane("rand", rng)
        for j, sh in enumerate(SHAPES):
            for k, s in enumerate(SNRS):
                a = C.amplitude(sh, 2, s, bg.W)
                U1 = C.row_shift(sh, 2, np.stack([c1, c1]))[1]          # level-1 per-row shift [n, 2]
                Uh = C.row_shift(sh, 2, np.stack([d1, d1]))[1]
                res[it, j, k] = binary_stats(F0, F1 + a * U1 @ P.T, H0, H1 + a * Uh @ P.T, space)
        np.savez(fn + ".part.npz", res=res[:it + 1])
        print(f"{fn} sim {it} {time.time() - t0:.0f}s", res[it, :, :, 0].round(1).tolist(), flush=True)
    np.savez(fn, res=res, shapes=np.array(SHAPES), snrs=SNRS, secs=time.time() - t0)
    print(f"{fn} {time.time() - t0:.0f}s", flush=True)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=40); ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--cells", default="tqa13_5:1200")
    ap.add_argument("--chunks", type=int, default=6, help="split each cell's sims into chunks for parallelism")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for c in a.cells.split(","):
        b, n = c.split(":")
        for ch in range(a.chunks):
            jobs.append((f"{b}", int(n), a.sims // a.chunks, hash((b, n, ch)) % 2 ** 31, ch))
    jobs.sort(key=lambda j: -j[1])
    with Pool(a.procs) as p:
        for _ in p.imap_unordered(run_chunk, jobs):
            pass


def run_chunk(j):
    b, n, sims, seed, ch = j
    global OUT
    out0 = OUT; OUT = f"{out0}/ch{ch}"; os.makedirs(OUT, exist_ok=True)
    try:
        return run_cell((b, n, sims, seed))
    finally:
        OUT = out0


if __name__ == "__main__":
    main()
