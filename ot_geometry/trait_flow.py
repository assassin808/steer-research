"""Trait flow V = P_T(x) grad s(x) (unit speed) from unpaired POS/NEG head activations.
Prereg: results/ot_geometry_prereg.md, amendment A2.
usage: python trait_flow.py --heads 13:5 --space z --out DIR
"""
import argparse, json, os, time, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
from ot_geom import META, HEADS, ITI, make_space, sinkhorn, ideea_fit, ideea_npn, null_sample, cosrows

warnings.filterwarnings("ignore", category=ConvergenceWarning)
C_GRID = [0.001, 0.01, 0.1, 1.0]
A_GRID = [0.01, 0.1, 1.0, 10.0]
KT, EV, H_FRAC, L_MAX = 64, 0.90, 0.02, 3.0
CHECK = [0.5, 1.0, 2.0]


# ------------------------------------------------ trait score
def logloss(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_lr(X, y, C):
    return LogisticRegression(C=C, max_iter=3000).fit(X, y)


def fit_mlp(X, y, a):
    return MLPClassifier(hidden_layer_sizes=(64,), activation="tanh", alpha=a, solver="lbfgs", max_iter=500,
                         random_state=0).fit(X, y)


def cv_select(X, y, g, grid, fitter):
    gkf = GroupKFold(n_splits=3)
    best, bl = None, np.inf
    scores = {}
    for v in grid:
        ls = []
        for a, b in gkf.split(X, y, g):
            m = fitter(X[a], y[a], v)
            ls.append(logloss(m.predict_proba(X[b])[:, 1], y[b]).mean())
        scores[str(v)] = float(np.mean(ls))
        if np.mean(ls) < bl:
            best, bl = v, np.mean(ls)
    return best, scores


class MLPScore:
    def __init__(self, m):
        self.W1, self.W2 = m.coefs_; self.b1, self.b2 = m.intercepts_

    def s(self, X):
        return (np.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2)[:, 0]

    def grad(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        return ((1 - h ** 2) * self.W2[:, 0]) @ self.W1.T


# ------------------------------------------------ tangent projection
class Tangent:
    def __init__(self, Zfit):
        self.Z = Zfit; self.nn = NearestNeighbors(n_neighbors=KT).fit(Zfit)

    def project(self, X, G):
        _, ind = self.nn.kneighbors(X)
        out = np.empty_like(G); ranks = np.empty(len(X), int)
        for i in range(len(X)):
            N = self.Z[ind[i]]; N = N - N.mean(0)
            _, sv, Vt = np.linalg.svd(N, full_matrices=False)
            ev = np.cumsum(sv ** 2) / np.sum(sv ** 2)
            r = int(np.searchsorted(ev, EV) + 1); ranks[i] = r
            U = Vt[:r]
            out[i] = U.T @ (U @ G[i])
        return out, ranks


# ------------------------------------------------ flow integration
def integrate(X0, S, tan, nnpos, band, h, nsteps, straight=None):
    """returns dict of per-point trajectories summaries. straight: unit vector for straight-line baseline."""
    X = X0.copy(); n = len(X)
    sv = np.zeros((n, nsteps + 1)); fn = np.zeros((n, nsteps + 1)); dk = np.zeros((n, nsteps + 1))
    V0 = None; Vprev = None; curv = np.zeros(n); turn = {}
    ranks = []
    for k in range(nsteps + 1):
        sv[:, k] = S.s(X)
        dk[:, k] = nnpos.kneighbors(X)[0][:, -1]
        if straight is not None:
            V = np.tile(straight, (n, 1)); fn[:, k] = np.nan
        else:
            G = S.grad(X)
            if tan is not None:
                G, r = tan.project(X, G); ranks.append(np.median(r))
            nrm = np.linalg.norm(G, axis=1, keepdims=True)
            fn[:, k] = nrm[:, 0]
            V = G / np.maximum(nrm, 1e-12)
        if V0 is None:
            V0 = V.copy()
        else:
            curv += np.linalg.norm(V - Vprev, axis=1) / h
        Vprev = V
        ell = k * h
        for c in CHECK:
            if c not in turn and ell >= c * HN - 1e-9:
                turn[c] = np.degrees(np.arccos(np.clip(np.sum(V0 * V, 1), -1, 1)))
        if k < nsteps:
            X = X + h * V
    inside = dk <= band
    ells = np.arange(nsteps + 1) * h / HN  # in units of ||dmu||
    def first_exit(r):
        ins = np.where(r)[0]
        if len(ins) == 0:
            return np.nan
        after = np.where(~r[ins[0]:])[0]
        return ells[ins[0] + after[0]] if len(after) else np.inf
    ex = np.array([first_exit(r) for r in inside])
    ds = np.diff(sv, axis=1) / h
    def first_below(arr, frac):
        out = np.full(len(arr), np.inf)
        for i, a in enumerate(arr):
            w = np.where(a < frac * a[0])[0]
            if len(w):
                out[i] = ells[w[0]]
        return out
    res = dict(
        turn={str(c): float(np.median(v)) for c, v in turn.items()},
        curv=float(np.median(curv / nsteps)),
        exit_med=float(np.median(ex[np.isfinite(ex) | np.isinf(ex)])) if np.any(~np.isnan(ex)) else None,
        frac_never_inside=float(np.isnan(ex).mean()), frac_never_exit=float(np.isinf(ex).mean()),
        frac_inside_t0=float(inside[:, 0].mean()), frac_inside_curve=inside.mean(0)[::5].tolist(),
        sat_med=float(np.median(first_below(ds, 0.1))), s_start=float(np.median(sv[:, 0])), s_end=float(np.median(sv[:, -1])),
        s_curve=np.median(sv, 0)[::5].tolist(), dk_curve=(np.median(dk, 0) / band)[::5].tolist(),
    )
    if straight is None:
        res["vanish_med"] = float(np.median(first_below(fn, 0.1)))
        res["s_at_exit"] = float(np.nanmedian([sv[i, int(round(e * HN / h))] for i, e in enumerate(ex) if np.isfinite(e)])) if np.any(np.isfinite(ex)) else None
    if ranks:
        res["tangent_rank_med"] = float(np.median(ranks))
    return res, V0


HN = None  # ||dmu||, set per head


def analyse(Zfit, yfit, gfit, Xstart, Zpos_ho, C, A, nsteps_h, do_proj=True, lr_w=None):
    """fit s, integrate flows. returns stats + V at start."""
    m = fit_mlp(Zfit, yfit, A); S = MLPScore(m)
    nnpos = NearestNeighbors(n_neighbors=10).fit(Zfit[yfit == 1])
    band = np.quantile(nnpos.kneighbors(Zpos_ho)[0][:, -1], 0.95)
    h, n = nsteps_h
    out = {}
    out["unproj"], V0u = integrate(Xstart, S, None, nnpos, band, h, n)
    if do_proj:
        out["proj"], V0p = integrate(Xstart, S, Tangent(Zfit), nnpos, band, h, n)
    else:
        V0p = None
    return out, V0u, V0p, m, band


def run_head(lab, is_tr, q, l, h, space, reps, naive_std, rng):
    global HN
    t0 = time.time()
    Xr = np.load(f"{HEADS}/L{l}H{h}.npy").astype(np.float64)
    mu, Amat = make_space(space, Xr[is_tr]); Z = (Xr - mu) @ Amat
    y = (lab == 1).astype(int); ho = ~is_tr
    Zf, yf, gf = Z[is_tr], y[is_tr], q[is_tr]
    dmu = Zf[yf == 1].mean(0) - Zf[yf == 0].mean(0); HN = float(np.linalg.norm(dmu))
    hstep = H_FRAC * HN; nsteps = int(round(L_MAX / H_FRAC))
    out = dict(layer=l, head=h, space=space, dmu_norm=HN, h=hstep, nsteps=nsteps)

    # 1. trait score: LR vs MLP
    C, cvC = cv_select(Zf, yf, gf, C_GRID, fit_lr)
    A, cvA = cv_select(Zf, yf, gf, A_GRID, fit_mlp)
    lr = fit_lr(Zf, yf, C); mlp = fit_mlp(Zf, yf, A)
    Zh, yh, gh = Z[ho], y[ho], q[ho]
    pl, pm = lr.predict_proba(Zh)[:, 1], mlp.predict_proba(Zh)[:, 1]
    dl = logloss(pl, yh) - logloss(pm, yh)
    uq = np.unique(gh); idx = {u: np.where(gh == u)[0] for u in uq}
    boots = []
    for b in range(1000):
        sel = np.concatenate([idx[u] for u in rng.choice(uq, len(uq))])
        boots.append(dl[sel].mean())
    out["score"] = dict(C=C, alpha=A, cvC=cvC, cvA=cvA, ll_lr=float(logloss(pl, yh).mean()), ll_mlp=float(logloss(pm, yh).mean()),
                        auc_lr=float(roc_auc_score(yh, pl)), auc_mlp=float(roc_auc_score(yh, pm)),
                        dll=float(dl.mean()), dll_ci=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))])
    print(f"L{l}H{h} {space} C={C} a={A} AUC lr {out['score']['auc_lr']:.3f} mlp {out['score']['auc_mlp']:.3f} "
          f"dLL {dl.mean():+.4f} [{out['score']['dll_ci'][0]:+.4f},{out['score']['dll_ci'][1]:+.4f}]", flush=True)

    # 2+4. flows on data
    Xs = Zh[yh == 0]; Zpos_ho = Zh[yh == 1]
    fl, V0u, V0p, _, band = analyse(Zf, yf, gf, Xs, Zpos_ho, C, A, (hstep, nsteps))
    nnpos = NearestNeighbors(n_neighbors=10).fit(Zf[yf == 1])
    S = MLPScore(mlp)
    fl["straight"], _ = integrate(Xs, S, None, nnpos, band, hstep, nsteps, straight=dmu / HN)
    # linear-s flows (sanity: unprojected = constant; projected turns only via tangent)
    class LinS:
        def __init__(s_, m): s_.w = m.coef_[0]; s_.b = m.intercept_[0]
        def s(s_, X): return X @ s_.w + s_.b
        def grad(s_, X): return np.tile(s_.w, (len(X), 1))
    fl["linear_proj"], _ = integrate(Xs, LinS(lr), Tangent(Zf), nnpos, band, hstep, nsteps)
    fl["linear_unproj"], _ = integrate(Xs, LinS(lr), None, nnpos, band, hstep, nsteps)
    out["flow"] = fl; out["band_q95"] = float(band)
    print(f"  turn@1 unproj {fl['unproj']['turn']['1.0']:.1f} proj {fl['proj']['turn']['1.0']:.1f} linproj {fl['linear_proj']['turn']['1.0']:.1f} "
          f"| exit unproj {fl['unproj']['exit_med']} proj {fl['proj']['exit_med']} straight {fl['straight']['exit_med']} "
          f"| sat unproj {fl['unproj']['sat_med']} r {fl['proj'].get('tangent_rank_med')} {time.time()-t0:.0f}s", flush=True)

    # 3. estimator agreement at held-out NEG starts
    Xr_h_neg = Xr[ho][yh == 0]
    kp, kn = ideea_fit(Xr[is_tr & (lab == 1)], Xr[is_tr & (lab == -1)], 2)
    ide = ideea_npn(Xr_h_neg, kp, kn) @ Amat
    ot = sinkhorn(Z[lab == -1], Z[lab == 1], 0.01)
    ho_neg_mask = ho[lab == -1]
    otd = ot["d"][ho_neg_mask]
    w = lr.coef_[0]
    est = {"ideea_npn_nc2": ide, "meandiff": np.tile(dmu, (len(Xs), 1)), "ot": otd, "lr_w": np.tile(w, (len(Xs), 1))}
    ag = {}
    for nm, V in [("V_unproj", V0u), ("V_proj", V0p)]:
        ag[nm] = {k: float(cosrows(V, e).mean()) for k, e in est.items()}
        ag[nm]["self_mean"] = float(cosrows(V, np.tile(V.mean(0), (len(V), 1))).mean())
    ag["ot_vs_meandiff"] = float(cosrows(otd, est["meandiff"]).mean())
    ag["ideea_vs_meandiff"] = float(cosrows(ide, est["meandiff"]).mean())
    ag["lrw_vs_meandiff"] = float(cosrows(est["lr_w"][:1], dmu[None])[0])
    out["agree"] = ag
    print("  agree V_unproj", {k: round(v, 3) for k, v in ag["V_unproj"].items()}, "V_proj", {k: round(v, 3) for k, v in ag["V_proj"].items()}, flush=True)

    # nulls
    nulls = {}
    Zn_f, Zp_f = Zf[yf == 0], Zf[yf == 1]
    nho_neg, nho_pos = int((yh == 0).sum()), int((yh == 1).sum())
    for kind, R in [("gshared", reps), ("perm", max(reps // 2, 2)), ("rshift", max(reps // 2, 2))]:
        L = []
        for r in range(R):
            if kind == "perm":
                yp = rng.permutation(yf)
                a, b = Zf[yp == 0], Zf[yp == 1]
                ho_n, ho_p = Xs, Zpos_ho  # start from real held-out points
            else:
                # sample fit + held-out jointly from the null, then split
                a_all, b_all = null_sample(kind, np.concatenate([Zn_f, Xs]), np.concatenate([Zp_f, Zpos_ho]), rng)
                a, ho_n = a_all[:len(Zn_f)], a_all[len(Zn_f):]
                b, ho_p = b_all[:len(Zp_f)], b_all[len(Zp_f):]
            Zn_ = np.concatenate([a, b]); yn_ = np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
            gn_ = np.r_[gf[yf == 0], gf[yf == 1]] if kind != "perm" else np.r_[gf[yp == 0], gf[yp == 1]]
            res, V0u_n, V0p_n, m_n, _ = analyse(Zn_, yn_, gn_, ho_n, ho_p, C, A, (hstep, nsteps), do_proj=(kind == "gshared"))
            lrn = fit_lr(Zn_, yn_, C)
            dmun = b.mean(0) - a.mean(0)
            res["agree_V_unproj"] = dict(lr_w=float(cosrows(V0u_n, np.tile(lrn.coef_[0], (len(V0u_n), 1))).mean()),
                                         meandiff=float(cosrows(V0u_n, np.tile(dmun, (len(V0u_n), 1))).mean()),
                                         self_mean=float(cosrows(V0u_n, np.tile(V0u_n.mean(0), (len(V0u_n), 1))).mean()))
            L.append(res)
        nulls[kind] = L
        print(f"  null {kind}: turn@1 unproj {np.median([x['unproj']['turn']['1.0'] for x in L]):.1f} "
              + (f"proj {np.median([x['proj']['turn']['1.0'] for x in L]):.1f} " if 'proj' in L[0] else "")
              + f"self {np.median([x['agree_V_unproj']['self_mean'] for x in L]):.3f} {time.time()-t0:.0f}s", flush=True)
    out["nulls"] = nulls
    out["sec"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True); ap.add_argument("--space", default="z")
    ap.add_argument("--reps", type=int, default=10); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(META); lab, is_tr, q = z["label"], z["is_train"], z["q"]
    naive_std = json.load(open(ITI))["naive_std"]
    os.makedirs(a.out, exist_ok=True)
    for hs in a.heads.split(","):
        l, h = map(int, hs.split(":"))
        fn = f"{a.out}/L{l}H{h}_{a.space}.json"
        if os.path.exists(fn):
            continue
        r = run_head(lab, is_tr, q, l, h, a.space, a.reps, naive_std, np.random.default_rng(7 + 1000 * l + h))
        json.dump(r, open(fn, "w"))
        print(f"done L{l}H{h} {a.space} {r['sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
