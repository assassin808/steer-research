"""OT geometry of unpaired POS/NEG head activations (Gemma-2-2B TruthfulQA, IDEEA dev split).
Prereg: ~/steer_math/results/ot_geometry_prereg.md.

usage: python ot_geom.py --heads 13:5,13:0 --space z --out DIR [--reps 20] [--quick]
"""
import argparse, json, os, time
import numpy as np, torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

META = "/home/yang0531/scratch/ot_geom/meta.npz"
HEADS = "/home/yang0531/scratch/ot_geom/heads"
ITI = "/home/yang0531/scratch/ideea_gemma/results/truthfulqa/gemma2_2b/directions/iti/directions.json"
EPS_C = [0.003, 0.01, 0.03, 0.1]
PRIMARY = 0.01
KS = 50
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- OT
def sinkhorn(Xn, Xp, c, tol=1e-4, max_iter=3000):
    """Uniform-marginal entropic OT, eps = c * median(C). Returns barycentric map, plan stats."""
    A = torch.as_tensor(Xn); B = torch.as_tensor(Xp)
    C = torch.cdist(A, B) ** 2
    eps = c * C.median().item()
    n, m = C.shape
    la, lb = -np.log(n), -np.log(m)
    f = torch.zeros(n); g = torch.zeros(m)
    it, err = 0, np.inf
    for it in range(max_iter):
        f = eps * la - eps * torch.logsumexp((g[None, :] - C) / eps, dim=1)
        g = eps * lb - eps * torch.logsumexp((f[:, None] - C) / eps, dim=0)
        if it % 10 == 0:
            logP = (f[:, None] + g[None, :] - C) / eps
            err = (torch.exp(torch.logsumexp(logP, 1) - la) - 1).abs().max().item()
            if err < tol:
                break
    logP = (f[:, None] + g[None, :] - C) / eps
    P = torch.exp(logP)
    rows = P.sum(1, keepdim=True)
    T = (P @ B) / rows
    Pr = P / rows
    ent = -(Pr * torch.log(Pr.clamp_min(1e-300))).sum(1)
    return dict(T=T.numpy(), d=(T - A).numpy(), jstar=P.argmax(1).numpy(),
                neff=float(torch.exp(ent).median()), iters=it, err=err, eps=eps)


def _r2cos(d, dmu):
    md = d.mean(0)
    r2 = float((md @ md) / np.mean(np.sum(d * d, 1)))
    cs = d @ dmu / (np.linalg.norm(d, axis=1) * np.linalg.norm(dmu) + 1e-12)
    return r2, float(cs.mean())


def flat_stats(X, d, dmu, k=KS):
    """returns dict: r2/cos of raw field, r2s/coss of kNN-smoothed field (k NEG neighbours, self excluded)."""
    r2, cs = _r2cos(d, dmu)
    _, ind = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    r2s, css = _r2cos(d[ind[:, 1:]].mean(1), dmu)
    return dict(r2=r2, cos=cs, r2s=r2s, coss=css)


# ---------------------------------------------------------------- nulls
def null_sample(kind, Xn, Xp, rng):
    n, m = len(Xn), len(Xp)
    mn, mp = Xn.mean(0), Xp.mean(0)
    if kind == "perm":
        Z = np.concatenate([Xn, Xp]); idx = rng.permutation(n + m)
        return Z[idx[:n]], Z[idx[n:]]
    if kind == "rshift":
        R = np.concatenate([Xn - mn, Xp - mp]); idx = rng.permutation(n + m)
        return R[idx[:n]] + mn, R[idx[n:]] + mp
    D = Xn.shape[1]
    if kind == "gshared":
        R = np.concatenate([Xn - mn, Xp - mp])
        S = np.cov(R.T) + 1e-6 * np.eye(D)
        L = np.linalg.cholesky(S)
        return mn + rng.standard_normal((n, D)) @ L.T, mp + rng.standard_normal((m, D)) @ L.T
    if kind == "gsep":
        Ln = np.linalg.cholesky(np.cov(Xn.T) + 1e-6 * np.eye(D))
        Lp = np.linalg.cholesky(np.cov(Xp.T) + 1e-6 * np.eye(D))
        return mn + rng.standard_normal((n, D)) @ Ln.T, mp + rng.standard_normal((m, D)) @ Lp.T
    raise ValueError(kind)


# ---------------------------------------------------------------- geodesic
def geo_ratio(Xn, Xp, jstar, k=10):
    Z = np.concatenate([Xn, Xp]); n = len(Xn)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, ind = nn.kneighbors(Z)
    rows = np.repeat(np.arange(len(Z)), k); cols = ind[:, 1:].ravel(); w = dist[:, 1:].ravel()
    G = csr_matrix((w, (rows, cols)), shape=(len(Z), len(Z)))
    G = G.maximum(G.T)
    D = dijkstra(G, directed=False, indices=np.arange(n))
    tgt = n + jstar
    geo = D[np.arange(n), tgt]
    euc = np.linalg.norm(Xn - Xp[jstar], axis=1)
    ok = np.isfinite(geo) & (euc > 0)
    rho = geo[ok] / euc[ok]
    return float(np.median(rho)), int((~ok).sum())


# ---------------------------------------------------------------- IDEEA predictors (raw space, as in ideea.py)
def ideea_fit(Pr, Nr, nc, seed=0):
    kp = KMeans(n_clusters=nc, random_state=seed, n_init=10).fit(Pr).cluster_centers_
    kn = KMeans(n_clusters=nc, random_state=seed, n_init=10).fit(Nr).cluster_centers_
    return kp, kn


def ideea_npn(x, kp, kn):
    near = lambda c: c[np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), 1)]
    v = near(kp) - near(kn)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def ideea_min_perp(x, kp, kn):
    """fit_fixed_nc QAP (brute force, nc small) + argmax |x.v|; as in ideea.py."""
    from itertools import permutations
    nc = len(kp)
    dm = np.zeros((nc, nc, kp.shape[1]))
    for i in range(nc):
        for j in range(nc):
            v = kp[j] - kn[i]; dm[i, j] = v / max(np.linalg.norm(v), 1e-12)
    best, bp = -np.inf, None
    for p in permutations(range(nc)):
        s = sum(dm[a, p[a]] @ dm[b, p[b]] for a in range(nc) for b in range(a + 1, nc))
        if s > best:
            best, bp = s, p
    dirs = np.stack([dm[i, bp[i]] for i in range(nc)])
    return dirs[np.argmax(np.abs(dirs @ x.T), 0)]


def cosrows(a, b):
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


# ---------------------------------------------------------------- main per head
def make_space(space, Xfit_raw):
    m = Xfit_raw.mean(0)
    if space == "raw":
        A = np.eye(Xfit_raw.shape[1])
    elif space == "z":
        A = np.diag(1.0 / (Xfit_raw.std(0) + 1e-8))
    elif space.startswith("pw"):
        w, V = np.linalg.eigh(np.cov((Xfit_raw - m).T))
        o = np.argsort(w)[::-1][:int(space[2:])]
        A = V[:, o] / np.sqrt(w[o])
    return m, A


def run_head(lab, is_tr, l, h, space, reps, naive_std, rng):
    out = dict(layer=l, head=h, space=space)
    Xr = np.load(f"{HEADS}/L{l}H{h}.npy").astype(np.float64)
    pos, neg = lab == 1, lab == -1
    m, A = make_space(space, Xr[is_tr])
    Z = (Xr - m) @ A
    Zn, Zp = Z[is_tr & neg], Z[is_tr & pos]
    dmu = Zp.mean(0) - Zn.mean(0)
    out.update(n_neg=len(Zn), n_pos=len(Zp), dim=Z.shape[1])

    # (a)+(b)
    res = {}
    for c in EPS_C:
        t0 = time.time()
        ot = sinkhorn(Zn, Zp, c)
        e = flat_stats(Zn, ot["d"], dmu)
        e.update(neff=ot["neff"], iters=ot["iters"], err=ot["err"],
                 dnorm=float(np.linalg.norm(ot["d"], axis=1).mean()), dmu_norm=float(np.linalg.norm(dmu)))
        nulls = {}
        for kind, R in [("gshared", reps), ("rshift", reps), ("gsep", reps // 2), ("perm", reps // 2)]:
            vals = []
            for r in range(R):
                a, b = null_sample(kind, Zn, Zp, rng)
                o2 = sinkhorn(a, b, c)
                vals.append(flat_stats(a, o2["d"], dmu))
            nulls[kind] = {s_: [v[s_] for v in vals] for s_ in vals[0]}
        e["nulls"] = nulls
        e["sec"] = time.time() - t0
        if c == PRIMARY:
            ot_primary = ot
        res[str(c)] = e
        print(f"  L{l}H{h} {space} c={c}: r2s={e['r2s']:.3f} r2={e['r2']:.3f} cos={e['cos']:.3f} gsh={np.mean(nulls['gshared']['r2s']):.3f} "
              f"rsh={np.mean(nulls['rshift']['r2s']):.3f} gsep={np.mean(nulls['gsep']['r2s']):.3f} "
              f"perm={np.mean(nulls['perm']['r2s']):.3f} neff={ot['neff']:.1f} it={ot['iters']} {e['sec']:.0f}s", flush=True)
    out["flat"] = res

    # (c) geodesic, primary eps
    g, nd = geo_ratio(Zn, Zp, ot_primary["jstar"])
    gn = {}
    for kind in ["gshared", "rshift"]:
        vals = []
        for r in range(max(reps // 2, 2)):
            a, b = null_sample(kind, Zn, Zp, rng)
            o2 = sinkhorn(a, b, PRIMARY)
            vals.append(geo_ratio(a, b, o2["jstar"])[0])
        gn[kind] = vals
    out["geo"] = dict(median_ratio=g, disconnected=nd, nulls=gn)
    print(f"  geo {g:.3f} gsh {np.mean(gn['gshared']):.3f} rsh {np.mean(gn['rshift']):.3f}", flush=True)

    # (d) predictors on held-out NEG; target = OT on full dev
    ho = ~is_tr
    Zn_all, Zp_all = Z[neg], Z[pos]
    ho_in_neg = ho[neg]
    full = sinkhorn(Zn_all, Zp_all, PRIMARY)
    D = full["d"]; tgt = D[ho_in_neg]; xh = Zn_all[ho_in_neg]
    fitneg = Zn_all[~ho_in_neg]; Dfit = D[~ho_in_neg]
    xh_raw = Xr[neg][ho_in_neg]
    pred = {}
    for nc in [2, 4]:
        kp, kn = ideea_fit(Xr[is_tr & pos], Xr[is_tr & neg], nc)
        pred[f"ideea_npn_nc{nc}"] = ideea_npn(xh_raw, kp, kn) @ A
        if nc == 2:
            pred["ideea_minperp_nc2"] = ideea_min_perp(xh_raw, kp, kn) @ A
    pred["meandiff"] = np.tile(dmu, (len(xh), 1))
    nn = NearestNeighbors(n_neighbors=20).fit(fitneg)
    _, ind = nn.kneighbors(xh)
    pred["knn20"] = Dfit[ind].mean(1)
    # pos-centroid-of-question oracle is NOT used (would need pairing)
    out["pred"] = {k: dict(cos=float(cosrows(v, tgt).mean())) for k, v in pred.items()}
    out["pred"]["n_heldout_neg"] = int(len(xh))
    # ceiling under R-shift null (truth constant)
    ceil = {"meandiff": [], "knn20": []}
    for r in range(3):
        a, b = null_sample("rshift", Zn_all, Zp_all, rng)
        o2 = sinkhorn(a, b, PRIMARY); D2 = o2["d"]
        fit_mask = ~ho_in_neg
        mu2 = b[is_tr[pos]].mean(0) - a[fit_mask].mean(0)
        ceil["meandiff"].append(float(cosrows(np.tile(mu2, (ho_in_neg.sum(), 1)), D2[ho_in_neg]).mean()))
        nn2 = NearestNeighbors(n_neighbors=20).fit(a[fit_mask])
        _, i2 = nn2.kneighbors(a[ho_in_neg])
        ceil["knn20"].append(float(cosrows(D2[fit_mask][i2].mean(1), D2[ho_in_neg]).mean()))
    out["pred_ceiling_rshift"] = ceil
    print("  pred", {k: round(v["cos"], 3) for k, v in out["pred"].items() if isinstance(v, dict)},
          "ceil", {k: round(np.mean(v), 3) for k, v in ceil.items()}, flush=True)

    # (e) boundary
    Pfit = Z[is_tr & pos]
    nnp = NearestNeighbors(n_neighbors=10).fit(Pfit)
    band = np.quantile(nnp.kneighbors(Z[ho & pos])[0][:, -1], 0.95)
    band_neg_side = np.quantile(NearestNeighbors(n_neighbors=10).fit(Z[is_tr & neg]).kneighbors(Z[ho & neg])[0][:, -1], 0.95)
    ts = np.round(np.arange(0, 3.01, 0.1), 2)
    bd = {}
    for name, dd in [("ot", tgt), ("meandiff", np.tile(dmu, (len(xh), 1)))]:
        dk = np.stack([nnp.kneighbors(xh + t * dd)[0][:, -1] for t in ts], 1)
        inside = dk <= band
        frac = inside.mean(0)
        texit = np.array([ts[np.where(r)[0].max()] if r.any() else np.nan for r in inside])
        bd[name] = dict(frac_inside=frac.tolist(), median_texit=float(np.nanmedian(texit)) if np.isfinite(texit).any() else None,
                        frac_never_inside=float(np.isnan(texit).mean()),
                        median_dk_over_band=(np.median(dk, 0) / band).tolist())
    dmu_raw = Xr[is_tr & pos].mean(0) - Xr[is_tr & neg].mean(0)
    ns = naive_std[l][h]
    bd["t_equiv"] = {str(a): float(a * ns / np.linalg.norm(dmu_raw)) for a in [1, 2.5, 4, 6, 15]}
    bd["band_q95"] = float(band); bd["band_neg_q95"] = float(band_neg_side); bd["ts"] = ts.tolist()
    out["boundary"] = bd
    print(f"  boundary texit ot={bd['ot']['median_texit']} md={bd['meandiff']['median_texit']} "
          f"t(a=4)={bd['t_equiv']['4']:.2f} t(a=15)={bd['t_equiv']['15']:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True)
    ap.add_argument("--space", default="z")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    z = np.load(META)
    lab, is_tr = z["label"], z["is_train"]
    naive_std = json.load(open(ITI))["naive_std"]
    os.makedirs(a.out, exist_ok=True)
    for hs in a.heads.split(","):
        l, h = map(int, hs.split(":"))
        fn = f"{a.out}/L{l}H{h}_{a.space}.json"
        if os.path.exists(fn):
            continue
        rng = np.random.default_rng(1000 * l + h)
        t0 = time.time()
        r = run_head(lab, is_tr, l, h, a.space, a.reps, naive_std, rng)
        r["sec"] = time.time() - t0
        json.dump(r, open(fn, "w"))
        print(f"done L{l}H{h} {a.space} {r['sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
