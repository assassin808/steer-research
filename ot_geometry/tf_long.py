"""POST-HOC (deviation D2): long-horizon boundary along trait-flow lines + alpha-grid sensitivity for turning.
usage: python tf_long.py --heads L:H --space z --out DIR [--agrid ext]"""
import argparse, json, os, time
import numpy as np
from sklearn.neighbors import NearestNeighbors
import trait_flow as T
from ot_geom import META, HEADS, ITI, make_space, null_sample

LONG_H, LONG_L = 0.25, 30.0
OUTGRID = [0, .5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]


def integrate2(X0, S, tan, nnpos, band, hs, HN, straight=None):
    X = X0.copy(); n = len(X); ells = np.r_[0, np.cumsum(hs)] / HN
    K = len(hs) + 1
    sv = np.zeros((n, K)); fn = np.zeros((n, K)); dk = np.zeros((n, K)); V0 = None; turn = {}
    for k in range(K):
        sv[:, k] = S.s(X); dk[:, k] = nnpos.kneighbors(X)[0][:, -1]
        if straight is not None:
            V = np.tile(straight, (n, 1)); fn[:, k] = np.nan
        else:
            G = S.grad(X)
            if tan is not None:
                G, _ = tan.project(X, G)
            nrm = np.linalg.norm(G, axis=1, keepdims=True); fn[:, k] = nrm[:, 0]; V = G / np.maximum(nrm, 1e-12)
        if V0 is None:
            V0 = V.copy()
        for c in [1, 3, 10, 30]:
            if c not in turn and ells[k] >= c - 1e-9:
                turn[c] = float(np.median(np.degrees(np.arccos(np.clip(np.sum(V0 * V, 1), -1, 1)))))
        if k < K - 1:
            X = X + hs[k] * V
    inside = dk <= band
    def first_exit(r):
        ins = np.where(r)[0]
        if not len(ins): return np.nan
        after = np.where(~r[ins[0]:])[0]
        return ells[ins[0] + after[0]] if len(after) else np.inf
    ex = np.array([first_exit(r) for r in inside])
    ds = np.diff(sv, axis=1) / np.array(hs)
    def first_below(arr, frac):
        out = np.full(len(arr), np.inf)
        for i, a in enumerate(arr):
            w = np.where(a < frac * a[0])[0]
            if len(w): out[i] = ells[w[0]]
        return out
    gi = [int(np.argmin(np.abs(ells - g))) for g in OUTGRID]
    finite = ex[~np.isnan(ex)]
    r = dict(turn=turn, exit_med=float(np.median(finite)) if len(finite) else None,
             exit_q=[float(np.quantile(finite, p)) for p in (.25, .75)] if len(finite) else None,
             frac_never_inside=float(np.isnan(ex).mean()), frac_never_exit=float(np.isinf(ex).mean()),
             sat_med=float(np.median(first_below(ds, 0.1))),
             frac_inside=inside.mean(0)[gi].tolist(), dk_over_band=(np.median(dk, 0) / band)[gi].tolist(),
             s=np.median(sv, 0)[gi].tolist(), exits=ex.tolist())
    if straight is None:
        r["vanish_med"] = float(np.median(first_below(fn, 0.1)))
        r["fnorm"] = (np.median(fn, 0) / np.median(fn[:, 0]))[gi].tolist()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True); ap.add_argument("--space", default="z")
    ap.add_argument("--out", required=True); ap.add_argument("--agrid", default="prereg")
    a = ap.parse_args()
    z = np.load(META); lab, is_tr, q = z["label"], z["is_train"], z["q"]
    naive_std = json.load(open(ITI))["naive_std"]; os.makedirs(a.out, exist_ok=True)
    for hs_ in a.heads.split(","):
        l, h = map(int, hs_.split(":")); fn = f"{a.out}/L{l}H{h}_{a.space}_{a.agrid}.json"
        if os.path.exists(fn): continue
        t0 = time.time(); rng = np.random.default_rng(11 + 1000 * l + h)
        Xr = np.load(f"{HEADS}/L{l}H{h}.npy").astype(np.float64)
        mu, A = make_space(a.space, Xr[is_tr]); Z = (Xr - mu) @ A
        y = (lab == 1).astype(int); ho = ~is_tr
        Zf, yf, gf = Z[is_tr], y[is_tr], q[is_tr]
        dmu = Zf[yf == 1].mean(0) - Zf[yf == 0].mean(0); HN = float(np.linalg.norm(dmu)); T.HN = HN
        grid = T.A_GRID if a.agrid == "prereg" else [10.0, 30.0, 100.0, 300.0]
        alpha, cv = T.cv_select(Zf, yf, gf, grid, T.fit_mlp)
        C, _ = T.cv_select(Zf, yf, gf, T.C_GRID, T.fit_lr)
        S = T.MLPScore(T.fit_mlp(Zf, yf, alpha)); lr = T.fit_lr(Zf, yf, C)
        Zh, yh = Z[ho], y[ho]; Xs = Zh[yh == 0]
        nnpos = NearestNeighbors(n_neighbors=10).fit(Zf[yf == 1])
        band = np.quantile(nnpos.kneighbors(Zh[yh == 1])[0][:, -1], 0.95)
        hs = [T.H_FRAC * HN] * 150 + [LONG_H * HN] * int(round((LONG_L - 3) / LONG_H))
        out = dict(layer=l, head=h, space=a.space, agrid=a.agrid, alpha=alpha, cv=cv, dmu_norm=HN, band=float(band), grid=OUTGRID)
        dmu_raw = Xr[is_tr & (lab == 1)].mean(0) - Xr[is_tr & (lab == -1)].mean(0)
        out["t_equiv"] = {str(al): float(al * naive_std[l][h] / np.linalg.norm(dmu_raw)) for al in [1, 2.5, 4, 6, 15]}
        class LinS:
            def s(s_, X): return X @ lr.coef_[0] + lr.intercept_[0]
            def grad(s_, X): return np.tile(lr.coef_[0], (len(X), 1))
        tan = T.Tangent(Zf)
        if a.agrid == "prereg":
            out["unproj"] = integrate2(Xs, S, None, nnpos, band, hs, HN)
            out["proj"] = integrate2(Xs, S, tan, nnpos, band, hs, HN)
            out["straight"] = integrate2(Xs, S, None, nnpos, band, hs, HN, straight=dmu / HN)
            out["straight_lrw"] = integrate2(Xs, S, None, nnpos, band, hs, HN, straight=lr.coef_[0] / np.linalg.norm(lr.coef_[0]))
            out["linear_proj"] = integrate2(Xs, LinS(), tan, nnpos, band, hs, HN)
            print(f"L{l}H{h} {a.space} exit straight {out['straight']['exit_med']} unproj {out['unproj']['exit_med']} proj {out['proj']['exit_med']} "
                  f"linproj {out['linear_proj']['exit_med']} | never-exit {out['straight']['frac_never_exit']:.2f}/{out['proj']['frac_never_exit']:.2f} "
                  f"| t(a=4) {out['t_equiv']['4']:.1f} t(a=15) {out['t_equiv']['15']:.1f} | sat {out['unproj']['sat_med']} vanish {out['unproj']['vanish_med']} {time.time()-t0:.0f}s", flush=True)
        else:
            hs1 = hs[:150]
            out["unproj"] = integrate2(Xs, S, None, nnpos, band, hs1, HN)
            out["proj"] = integrate2(Xs, S, tan, nnpos, band, hs1, HN)
            Zn_f, Zp_f = Zf[yf == 0], Zf[yf == 1]; Zpos_ho = Zh[yh == 1]
            nl = []
            for r in range(5):
                a_all, b_all = null_sample("gshared", np.concatenate([Zn_f, Xs]), np.concatenate([Zp_f, Zpos_ho]), rng)
                an, ho_n = a_all[:len(Zn_f)], a_all[len(Zn_f):]; bn = b_all[:len(Zp_f)]
                Zn_ = np.concatenate([an, bn]); yn_ = np.r_[np.zeros(len(an), int), np.ones(len(bn), int)]
                Sn = T.MLPScore(T.fit_mlp(Zn_, yn_, alpha)); nnp = NearestNeighbors(n_neighbors=10).fit(bn)
                nl.append(dict(unproj=integrate2(ho_n, Sn, None, nnp, band, hs1, HN)["turn"],
                               proj=integrate2(ho_n, Sn, T.Tangent(Zn_), nnp, band, hs1, HN)["turn"]))
            out["null_gshared"] = nl
            print(f"L{l}H{h} {a.space} ext alpha={alpha} turn@1 unproj {out['unproj']['turn'][1]:.1f} null {[round(x['unproj'][1],1) for x in nl]} "
                  f"proj {out['proj']['turn'][1]:.1f} null {[round(x['proj'][1],1) for x in nl]} {time.time()-t0:.0f}s", flush=True)
        json.dump(out, open(fn, "w"))


if __name__ == "__main__":
    main()
