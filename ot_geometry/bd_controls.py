"""POST HOC D4: support-exit controls. Exit = first time (after being inside) the 10-NN distance to fit POS exceeds q95
of held-out POS; reported in units of |dmu| as Euclidean displacement from the start."""
import argparse, json, os, time
import numpy as np
from sklearn.neighbors import NearestNeighbors
import trait_flow as T
from ot_geom import META, HEADS, make_space
ap = argparse.ArgumentParser(); ap.add_argument("--heads", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
z = np.load(META); lab, tr, q = z["label"], z["is_train"], z["q"]; y = (lab == 1).astype(int); os.makedirs(a.out, exist_ok=True)
for hs_ in a.heads.split(","):
    l, h = map(int, hs_.split(":")); t0 = time.time(); rng = np.random.default_rng(31 + 1000 * l + h)
    X = np.load(f"{HEADS}/L{l}H{h}.npy").astype(float); m, A = make_space("z", X[tr]); Z = (X - m) @ A
    Zf, yf, gf = Z[tr], y[tr], q[tr]; Zh, yh = Z[~tr], y[~tr]; Xs = Zh[yh == 0]
    dmu = Zf[yf == 1].mean(0) - Zf[yf == 0].mean(0); HN = float(np.linalg.norm(dmu)); T.HN = HN
    nnp = NearestNeighbors(n_neighbors=10).fit(Zf[yf == 1]); band = np.quantile(nnp.kneighbors(Zh[yh == 1])[0][:, -1], 0.95)
    nnall = NearestNeighbors(n_neighbors=10).fit(Zf); band_all = np.quantile(nnall.kneighbors(Zh)[0][:, -1], 0.95)
    D = np.linspace(0, 30, 121)  # displacement grid in |dmu|
    def exit_line(u, nn, bd):
        dk = np.stack([nn.kneighbors(Xs + d * HN * u)[0][:, -1] for d in D], 1); ins = dk <= bd
        ex = []
        for r in ins:
            i = np.where(r)[0]
            if not len(i): ex.append(np.nan); continue
            o = np.where(~r[i[0]:])[0]; ex.append(D[i[0] + o[0]] if len(o) else np.inf)
        return float(np.nanmedian(ex))
    out = dict(layer=l, head=h, dmu_norm=HN, band=float(band), band_all=float(band_all))
    U = rng.standard_normal((5, Z.shape[1])); U /= np.linalg.norm(U, axis=1, keepdims=True)
    w, V = np.linalg.eigh(np.cov(Zf.T)); P10 = V[:, np.argsort(w)[::-1][:10]]
    U10 = rng.standard_normal((5, 10)) @ P10.T; U10 /= np.linalg.norm(U10, axis=1, keepdims=True)
    for nm, nn, bd in [("pos", nnp, band), ("all", nnall, band_all)]:
        out[nm] = dict(plus_dmu=exit_line(dmu / HN, nn, bd), minus_dmu=exit_line(-dmu / HN, nn, bd),
                       random=[exit_line(u, nn, bd) for u in U], random_pc10=[exit_line(u, nn, bd) for u in U10])
    # projected flows, Euclidean displacement
    C, _ = T.cv_select(Zf, yf, gf, T.C_GRID, T.fit_lr); lr = T.fit_lr(Zf, yf, C)
    S = T.MLPScore(T.fit_mlp(Zf, yf, 10.0))
    class LinS:
        def s(s_, X): return X @ lr.coef_[0]
        def grad(s_, X): return np.tile(lr.coef_[0], (len(X), 1))
    tan = T.Tangent(Zf)
    for nm, Sc in [("flow_proj", S), ("linear_proj", LinS()), ("flow_unproj", S)]:
        x = Xs.copy(); hstep = 0.1 * HN; ex = np.full(len(x), np.nan); entered = np.zeros(len(x), bool); exd = np.full(len(x), np.nan)
        for k in range(301):
            disp = np.linalg.norm(x - Xs, axis=1) / HN
            ins = nnp.kneighbors(x)[0][:, -1] <= band
            newexit = entered & ~ins & np.isnan(exd); exd[newexit] = disp[newexit]; ex[newexit] = k * 0.1
            entered |= ins
            G = Sc.grad(x)
            if nm != "flow_unproj": G, _ = tan.project(x, G)
            x = x + hstep * G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
        out[nm] = dict(exit_disp=float(np.nanmedian(exd)), exit_arc=float(np.nanmedian(ex)), never_exit=float(np.mean(entered & np.isnan(exd))),
                       disp_per_arc_at_exit=float(np.nanmedian(exd / ex)))
    print(f"L{l}H{h} +dmu {out['pos']['plus_dmu']:.2f} -dmu {out['pos']['minus_dmu']:.2f} rand {np.median(out['pos']['random']):.2f} pc10 {np.median(out['pos']['random_pc10']):.2f} | "
          f"proj disp {out['flow_proj']['exit_disp']:.2f} (arc {out['flow_proj']['exit_arc']:.2f}) linproj disp {out['linear_proj']['exit_disp']:.2f} unproj {out['flow_unproj']['exit_disp']:.2f} {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(f"{a.out}/L{l}H{h}.json", "w"))
