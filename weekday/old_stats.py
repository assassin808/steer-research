"""Apply the statistics of results/ot_geometry.md + klflow.md to weekday/month class pairs.
O1 flow-line turning (MLP alpha=100, unprojected, angle after arc |dmu|) vs G-shared / R-shift nulls
O2 straight-path MMD diagnostics (target d*, overshoot) + MMD/nearest-class to the intermediate class, straight vs polyline
O3 geo_ratio (two-class graph as in ot_geom; and all-class graph), OT R2_s50 vs R-shift
usage: python old_stats.py --dom month --layers 6,10,14 [--reps 10]
"""
import argparse, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import torch
torch.set_num_threads(1)
from multiprocessing import Pool
sys.path.insert(0, os.path.expanduser("~/steer_math/ot_geometry"))
import trait_flow as T
from ot_geom import null_sample, geo_ratio, sinkhorn, flat_stats
from sklearn.neighbors import NearestNeighbors
import geom as G

R = os.path.expanduser("~/scratch/weekday")
ALPHA = 100.0
SGRID = np.round(np.arange(0, 1.51, 0.05), 3)


def turn_at1(Zf, yf, Xs, Zpos_ho, HN, alpha=ALPHA):
    S = T.MLPScore(T.fit_mlp(Zf, yf, alpha))
    nnpos = NearestNeighbors(n_neighbors=10).fit(Zf[yf == 1])
    band = np.quantile(nnpos.kneighbors(Zpos_ho)[0][:, -1], 0.95)
    h = T.H_FRAC * HN; n = int(round(1.0 / T.H_FRAC))
    T.HN = HN
    res, _ = T.integrate(Xs, S, None, nnpos, band, h, n)
    return res["turn"]["1.0"]


def mmd(X, P, bw):
    g = 1 / (2 * bw ** 2)
    def K(A, B):
        s = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T
        return np.exp(-g * np.maximum(s, 0))
    n, m = len(X), len(P)
    kxx, kyy, kxy = K(X, X), K(P, P), K(X, P)
    return float((kxx.sum() - n) / (n * (n - 1)) + (kyy.sum() - m) / (m * (m - 1)) - 2 * kxy.mean())


def pair(args):
    Z, y, tmpl, C, i, j, reps, seed = args
    rng = np.random.default_rng(seed)
    fit = tmpl <= 3; ho = ~fit
    mu = Z[fit].mean(0); sd = Z[fit].std(0) + 1e-8
    Zs = (Z - mu) / sd                                   # z-space as in ot_geometry
    sel_f = fit & np.isin(y, [i, j]); sel_h = ho & np.isin(y, [i, j])
    Zf, yf = Zs[sel_f], (y[sel_f] == j).astype(int)
    Xs, Zpos_ho = Zs[sel_h & (y == i)], Zs[sel_h & (y == j)]
    dmu = Zf[yf == 1].mean(0) - Zf[yf == 0].mean(0); HN = float(np.linalg.norm(dmu))
    out = dict(i=i, j=j, dmu_norm=HN)
    # O1 (alpha: ot_geometry's post-hoc 100, and CV-selected over a grid with template groups)
    gf = tmpl[sel_f]
    acv, _ = T.cv_select(Zf, yf, gf, [0.01, 0.1, 1.0, 10.0, 100.0], T.fit_mlp)
    out["alpha_cv"] = acv
    out["turn1"] = turn_at1(Zf, yf, Xs, Zpos_ho, HN)
    out["turn1_cv"] = turn_at1(Zf, yf, Xs, Zpos_ho, HN, acv)
    nl = {}
    for kind in ["gshared", "rshift"]:
        v = []
        for r in range(reps):
            a_all, b_all = null_sample(kind, np.concatenate([Zf[yf == 0], Xs]), np.concatenate([Zf[yf == 1], Zpos_ho]), rng)
            na = int((yf == 0).sum()); nb = int((yf == 1).sum())
            Zn = np.concatenate([a_all[:na], b_all[:nb]]); yn = np.r_[np.zeros(na, int), np.ones(nb, int)]
            v.append([turn_at1(Zn, yn, a_all[na:], b_all[nb:], HN), turn_at1(Zn, yn, a_all[na:], b_all[nb:], HN, acv)])
        nl[kind] = v
    out["turn1_null"] = nl
    # O2: paths from held-out class-i points
    cf = np.stack([Zs[fit & (y == c)].mean(0) for c in range(C)])
    span = (j - i) % C; seq = [(i + s) % C for s in range(span + 1)]
    def poly(s):
        u = s * span; a = int(min(np.floor(u), span - 1)) if s <= 1 else span - 1
        return cf[seq[a]] + (u - a) * (cf[seq[a + 1]] - cf[seq[a]])
    pool = np.concatenate([Xs, Zs[fit & (y == j)]])
    D = G.eucl(pool); bw = float(np.median(D[np.triu_indices(len(pool), 1)]))
    tgt = Zs[fit & (y == j)]; mids = [Zs[fit & (y == m)] for m in seq[1:-1]]
    nn_all = NearestNeighbors(n_neighbors=10).fit(Zs[fit]); band = np.quantile(nn_all.kneighbors(Zs[ho])[0][:, -1], .95)
    rec = {}
    for pn in ["straight", "polyline"]:
        mt, mm, nc, ins = [], [], [], []
        for s in SGRID:
            X = Xs + (s * (cf[j] - cf[i]) if pn == "straight" else poly(min(s, 1.0)) + max(s - 1, 0) * (cf[j] - cf[seq[-2]]) - cf[i])
            mt.append(mmd(X, tgt, bw)); mm.append([mmd(X, M, bw) for M in mids])
            dc = G.eucl(np.concatenate([X, cf]))[:len(X), len(X):]
            nc.append(np.bincount(dc.argmin(1), minlength=C).tolist())
            ins.append(float((nn_all.kneighbors(X)[0][:, -1] <= band).mean()))
        mt = np.array(mt); k = int(np.argmin(mt))
        rec[pn] = dict(mmd_tgt=mt.tolist(), dstar=float(SGRID[k]), rise=float((mt[-1] - mt[k]) / max(mt[0] - mt[k], 1e-12)),
                       mmd_mid=np.array(mm).tolist(), min_mmd_mid=np.array(mm).min(0).tolist(), nearest=nc, in_support=ins)
    out["paths"] = rec
    out["mmd_floor_tgt"] = mmd(Zs[ho & (y == j)], tgt, bw)
    # O3
    ot = sinkhorn(Zf[yf == 0], Zf[yf == 1], 0.01)
    out["geo_two"] = geo_ratio(Zf[yf == 0], Zf[yf == 1], ot["jstar"])[0]
    DGall, _ = G.geodesic(Zs[fit], 10)
    idx_f = np.where(fit)[0]; pos_in = {v: q for q, v in enumerate(idx_f)}
    ni = [pos_in[v] for v in np.where(sel_f & (y == i))[0]]; nj = [pos_in[v] for v in np.where(sel_f & (y == j))[0]]
    geo = DGall[ni][:, nj][np.arange(len(ni)), ot["jstar"]]
    euc = np.linalg.norm(Zf[yf == 0] - Zf[yf == 1][ot["jstar"]], axis=1)
    out["geo_all"] = float(np.median(geo[np.isfinite(geo)] / euc[np.isfinite(geo)]))
    fs = flat_stats(Zf[yf == 0], ot["d"], dmu)
    out["ot_r2s"] = fs["r2s"]
    rs = []
    for r in range(reps):
        a, b = null_sample("rshift", Zf[yf == 0], Zf[yf == 1], rng)
        o2 = sinkhorn(a, b, 0.01); rs.append(flat_stats(a, o2["d"], b.mean(0) - a.mean(0))["r2s"])
    out["ot_r2s_rshift"] = rs
    print(f"pair {i}->{j} turn1 {out['turn1']:.1f} cv(a={acv}) {out['turn1_cv']:.1f} (G {np.max(nl['gshared'], 0)} R {np.max(nl['rshift'], 0)}) "
          f"d* {rec['straight']['dstar']} minMMDmid str {rec['straight']['min_mmd_mid']} poly {rec['polyline']['min_mmd_mid']} "
          f"geo2 {out['geo_two']:.3f} geoall {out['geo_all']:.3f} r2s {out['ot_r2s']:.2f} (R {np.min(rs):.2f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma"); ap.add_argument("--dom", required=True)
    ap.add_argument("--layers", required=True); ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--label", default="y", choices=["y", "z"])
    a = ap.parse_args()
    z = np.load(f"{R}/{a.model}_{a.dom}.npz"); H, y, tm = z["H"], z[a.label], z["t"]
    _, keep = np.unique(H[:, H.shape[1] // 2].round(4), axis=0, return_index=True); keep = np.sort(keep)
    H, y, tm = H[keep], y[keep], tm[keep]
    C = 7 if a.dom == "weekday" else 12
    out = {}
    for l in [int(x) for x in a.layers.split(",")]:
        Z = G.pca(H[:, l].astype(np.float64), a.d)
        jobs = [(Z, y, tm, C, i, (i + s) % C, a.reps, 77 + 100 * l + 10 * i + s) for s in (2, 3) for i in range(C)]
        with Pool(a.procs) as p:
            out[str(l)] = p.map(pair, jobs)
        print("layer", l, "done", flush=True)
        json.dump(out, open(f"{R}/old_{a.model}_{a.dom}_{a.label}.json", "w"))


if __name__ == "__main__":
    main()
