"""Point-level grid (P1 frot, P2 fincoh, P3 dist). usage: python partP.py [--sims 60 --procs 16]"""
import os, sys, argparse, time
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[k] = "1"
import numpy as np, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.expanduser("~/steer_math/weekday"))
import geom as Gk
import core as C

OUT = f"{C.R}/partP"
SNRS = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
SH_L = {2: ["arc0", "arc90", "clust90", "clust180"],
        5: ["arc0", "arc60", "arc90", "arc180", "bend90", "sat", "clust90", "clust180"],
        7: ["arc0", "arc60", "arc90", "arc180", "bend90", "sat", "clust90", "clust180"]}
DIST_SH = {"arc0", "arc90", "clust90"}
DIST_NMAX = 50            # all-pairs Dijkstra + Spearman over N^2 pairs: budget cut (prereg said n*L <= 2100; reported)


def ang(u, v):
    return np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300), -1, 1)))


def pca64(X):
    Xc = X - X.mean(0)
    if Xc.shape[0] < Xc.shape[1]:
        w, V = np.linalg.eigh(Xc @ Xc.T); o = np.argsort(w)[::-1][:64]
        return V[:, o] * np.sqrt(np.maximum(w[o], 0))
    w, V = np.linalg.eigh(Xc.T @ Xc); o = np.argsort(w)[::-1][:64]
    return Xc @ V[:, o]


def field(Z, L, n, m=10):
    """open local field (kit S5): v(x) for x at level l = mean of m nearest level-(l+1) pts - mean of m nearest level-l pts (excl. self)"""
    Zl = Z.reshape(L, n, -1); means, coh = [], []
    for l in range(L - 1):
        Xi, Xj = Zl[l], Zl[l + 1]
        mi = min(m, n - 1)
        _, ii = NearestNeighbors(n_neighbors=mi + 1).fit(Xi).kneighbors(Xi)
        _, jj = NearestNeighbors(n_neighbors=min(m, n)).fit(Xj).kneighbors(Xi)
        V = Xj[jj].mean(1) - Xi[ii[:, 1:]].mean(1)
        mv = V.mean(0); means.append(mv)
        coh.append(np.mean(V @ mv / (np.linalg.norm(V, axis=1) * np.linalg.norm(mv) + 1e-300)))
    rot = [ang(means[i], means[i + 1]) for i in range(L - 2)]
    return (float(np.median(rot)) if rot else np.nan), float(1 - np.mean(coh))


def point_stats(X, L, n, do_dist):
    Z = pca64(X)
    fr, fc = field(Z, L, n)
    d = np.nan
    if do_dist:
        DG, _ = Gk.geodesic(Z, 10); d = Gk.distortion(DG, Gk.eucl(Z))[0]
    return fr, fc, d


def run_cell(args):
    bgname, L, n, mode, sims, seed = args
    fn = f"{OUT}/{bgname}_L{L}_n{n}_{mode}.npz"
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    bg = C.BG(bgname); rng = np.random.default_rng(seed)
    shapes = SH_L[L]
    res = np.full((sims, len(shapes), len(SNRS), 3), np.nan)
    smode = "iid" if mode == "iid" else "q"
    for it in range(sims):
        Xr, clu, qid, rep = bg.sample(L, n, smode, rng)
        P = bg.plane("rand", rng)
        for j, sh in enumerate(shapes):
            U = C.row_shift(sh, L, clu)                                   # [L, n, 2]
            for k, s in enumerate(SNRS):
                X = Xr + C.amplitude(sh, L, s, bg.W) * (U @ P.T)
                if mode == "qdm":
                    for qq in range(qid.max() + 1):
                        msk = qid == qq; X[msk] -= X[msk].mean(0)
                res[it, j, k] = point_stats(X.reshape(L * n, -1), L, n, n <= DIST_NMAX and sh in DIST_SH)
    np.savez(fn, res=res, shapes=np.array(shapes), snrs=SNRS, secs=time.time() - t0)
    print(f"{fn} {time.time() - t0:.0f}s", flush=True)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=40); ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--bgs", default="tqa13_5,syc,wk"); ap.add_argument("--ns", default="20,50,100,300"); ap.add_argument("--Ls", default="2,5,7")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for b in a.bgs.split(","):
        for L in map(int, a.Ls.split(",")):
            for n in map(int, a.ns.split(",")):
                for m in (["iid", "qraw", "qdm"] if b != "wk" else ["iid"]):
                    jobs.append((b, L, n, m, a.sims, hash((b, L, n, m, "P")) % 2 ** 31))
    jobs.sort(key=lambda j: -j[2] * j[1])
    with Pool(a.procs) as p:
        for _ in p.imap_unordered(run_cell, jobs):
            pass


if __name__ == "__main__":
    main()
