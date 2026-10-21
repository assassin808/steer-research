"""P2: frozen velocity vs KL gradient flow vs guidance flow. Prereg: results/klflow_prereg.md.
usage: python klflow.py --head 13:5 --space z --out DIR [--nulls 20]
"""
import argparse, json, os, sys, time
import numpy as np, torch
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from scipy.special import logsumexp
sys.path.insert(0, "/home/yang0531/scratch/steer_live/ot_geometry")
import trait_flow as T
from ot_geom import META, HEADS, make_space, null_sample

ALPHA = 100.0
H_GRID = [0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
KS = [0.25, 1.0, 4.0]
NSTEP, STEP = 150, 0.1
BW = [0.5, 1.0, 2.0]


# ---------------------------------------------------------------- scores
def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T, 0)


def kde_score(X, Y, h):
    L = -sqd(X, Y) / (2 * h * h)
    L -= L.max(1, keepdims=True); W = np.exp(L); W /= W.sum(1, keepdims=True)
    return (W @ Y - X) / (h * h)


def kde_ll(X, Y, h):
    d = Y.shape[1]
    return logsumexp(-sqd(X, Y) / (2 * h * h), axis=1) - np.log(len(Y)) - 0.5 * d * np.log(2 * np.pi * h * h)


def kde_cv_h(Y, g):
    base = np.median(NearestNeighbors(n_neighbors=2).fit(Y).kneighbors(Y)[0][:, 1])
    best, bl, sc = None, -np.inf, {}
    for c in H_GRID:
        h = c * base; ll = []
        for a, b in GroupKFold(3).split(Y, groups=g):
            ll.append(kde_ll(Y[b], Y[a], h).mean())
        sc[str(c)] = float(np.mean(ll))
        if np.mean(ll) > bl:
            best, bl = h, np.mean(ll)
    return best, base, sc


class DSM(torch.nn.Module):
    def __init__(s, d):
        super().__init__()
        s.f = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.SiLU(), torch.nn.Linear(512, 512), torch.nn.SiLU(),
                                  torch.nn.Linear(512, d))

    def forward(s, x):
        return s.f(x)


def fit_dsm(Y, sigma, steps=4000, seed=0):
    torch.manual_seed(seed)
    net = DSM(Y.shape[1]).double(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    Yt = torch.as_tensor(Y)
    for it in range(steps):
        x = Yt[torch.randint(len(Yt), (256,))]; e = torch.randn_like(x)
        loss = ((net(x + sigma * e) * sigma + e) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    def score(X):
        with torch.no_grad():
            return net(torch.as_tensor(X)).numpy()
    return score, float(loss.item())


# ---------------------------------------------------------------- metrics
class Metrics:
    def __init__(s, Pho, Zfit_all, Zfit_pos, Zho_all, Zho_pos, S, HN, X0):
        s.P = Pho; s.S = S; s.HN = HN; s.X0 = X0
        pool = np.concatenate([X0, Pho]); D = np.sqrt(sqd(pool, pool)); s.med = np.median(D[np.triu_indices(len(pool), 1)])
        s.Dyy = np.sqrt(sqd(Pho, Pho))
        s.nn_all = NearestNeighbors(n_neighbors=10).fit(Zfit_all); s.b_all = np.quantile(s.nn_all.kneighbors(Zho_all)[0][:, -1], .95)
        s.nn_pos = NearestNeighbors(n_neighbors=10).fit(Zfit_pos); s.b_pos = np.quantile(s.nn_pos.kneighbors(Zho_pos)[0][:, -1], .95)

    def mmd(s, X):
        Dxx = sqd(X, X); Dxy = sqd(X, s.P); Dyy = s.Dyy ** 2
        n, m = len(X), len(s.P); out = []
        for b in BW:
            g = 1 / (2 * (b * s.med) ** 2)
            kxx = np.exp(-g * Dxx); kyy = np.exp(-g * Dyy); kxy = np.exp(-g * Dxy)
            out.append(float((kxx.sum() - n) / (n * (n - 1)) + (kyy.sum() - m) / (m * (m - 1)) - 2 * kxy.mean()))
        return out

    def energy(s, X):
        return float(2 * np.sqrt(sqd(X, s.P)).mean() - np.sqrt(sqd(X, X)).mean() - s.Dyy.mean())

    def __call__(s, X):
        return dict(mmd=s.mmd(X), en=s.energy(X), s=float(np.median(s.S.s(X))),
                    in_all=float((s.nn_all.kneighbors(X)[0][:, -1] <= s.b_all).mean()),
                    in_pos=float((s.nn_pos.kneighbors(X)[0][:, -1] <= s.b_pos).mean()),
                    disp=float(np.median(np.linalg.norm(X - s.X0, axis=1)) / s.HN))


def run_path(X0, vel, M, unit=False, HN=1.0):
    X = X0.copy(); v0 = vel(X)
    dt = STEP * HN if unit else STEP * HN / np.median(np.linalg.norm(v0, axis=1))
    rec = [M(X)]
    for k in range(NSTEP):
        v = vel(X)
        if unit:
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        X = X + dt * v; rec.append(M(X))
    return rec


def summarise(rec):
    out = {}
    for bi, b in enumerate(BW):
        m = np.array([r["mmd"][bi] for r in rec]); i = int(np.argmin(m))
        rise = (m[-1] - m[i]) / max(m[0] - m[i], 1e-12)
        out[str(b)] = dict(start=float(m[0]), min=float(m[i]), final=float(m[-1]), dstar=rec[i]["disp"], istar=i, rise=float(rise),
                           plateau=bool(rise <= 0.1), overshoot=bool(rise >= 0.5))
    e = np.array([r["en"] for r in rec]); i = int(np.argmin(e))
    out["energy"] = dict(start=float(e[0]), min=float(e[i]), final=float(e[-1]), dstar=rec[i]["disp"], rise=float((e[-1] - e[i]) / max(e[0] - e[i], 1e-12)))
    return out


def paths_for(Zf, yf, gf, Xs, Pho, Zho_all, h, HN, dmu, which, C=None, dsm=None, keep_curves=True):
    S = T.MLPScore(T.fit_mlp(Zf, yf, ALPHA))
    Yp = Zf[yf == 1]
    M = Metrics(Pho, Zf, Yp, Zho_all, Pho, S, HN, Xs)
    res = {}
    sc0 = kde_score(Xs, Yp, h); gs0 = S.grad(Xs)
    gam = {k: k * np.median(np.linalg.norm(gs0, axis=1)) / np.median(np.linalg.norm(sc0, axis=1)) for k in KS}
    P = {}
    if "dmu" in which: P["straight_dmu"] = (lambda X: np.tile(dmu, (len(X), 1)), True)
    if "w" in which:
        lr = T.fit_lr(Zf, yf, C); w = lr.coef_[0]; P["straight_w"] = (lambda X: np.tile(w, (len(X), 1)), True)
    if "frozen" in which: P["frozen_grad_s"] = (S.grad, True)
    for k in KS:
        if f"guidK{k}" in which: P[f"guid_kde_k{k}"] = ((lambda g: lambda X: S.grad(X) + g * kde_score(X, Yp, h))(gam[k]), False)
    if dsm is not None:
        sc0d = dsm(Xs); gamd = {k: k * np.median(np.linalg.norm(gs0, axis=1)) / np.median(np.linalg.norm(sc0d, axis=1)) for k in KS}
        for k in KS:
            P[f"guid_dsm_k{k}"] = ((lambda g: lambda X: S.grad(X) + g * dsm(X))(gamd[k]), False)
    if "particle" in which: P["particle_kl"] = (lambda X: kde_score(X, Yp, h) - kde_score(X, X, h), False)
    for nm, (vel, unit) in P.items():
        rec = run_path(Xs, vel, M, unit=unit, HN=HN)
        res[nm] = dict(summary=summarise(rec))
        if keep_curves:
            res[nm]["curve"] = {k: [r[k] for r in rec] for k in ["en", "s", "in_all", "in_pos", "disp"]}
            res[nm]["curve"]["mmd"] = [r["mmd"] for r in rec]
    return res, dict(gamma={str(k): float(v) for k, v in gam.items()}, med_bw=float(M.med))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True); ap.add_argument("--space", default="z")
    ap.add_argument("--out", required=True); ap.add_argument("--nulls", type=int, default=20)
    a = ap.parse_args(); torch.set_num_threads(4)
    l, hh = map(int, a.head.split(":")); t0 = time.time(); rng = np.random.default_rng(41 + 1000 * l + hh)
    z = np.load(META); lab, tr, q = z["label"], z["is_train"], z["q"]; y = (lab == 1).astype(int)
    X = np.load(f"{HEADS}/L{l}H{hh}.npy").astype(float); m, A = make_space(a.space, X[tr]); Z = (X - m) @ A
    Zf, yf, gf = Z[tr], y[tr], q[tr]; Zh, yh = Z[~tr], y[~tr]; Xs, Pho = Zh[yh == 0], Zh[yh == 1]
    dmu = Zf[yf == 1].mean(0) - Zf[yf == 0].mean(0); HN = float(np.linalg.norm(dmu))
    C, _ = T.cv_select(Zf, yf, gf, T.C_GRID, T.fit_lr)
    h, base, hsc = kde_cv_h(Zf[yf == 1], gf[yf == 1])
    dsm, dsm_loss = fit_dsm(Zf[yf == 1], h)
    out = dict(layer=l, head=hh, space=a.space, dmu_norm=HN, h=float(h), h_base=float(base), h_cv=hsc, dsm_loss=dsm_loss, C=C)
    # floor
    Mf = Metrics(Pho, Zf, Zf[yf == 1], Zh, Pho, T.MLPScore(T.fit_mlp(Zf, yf, ALPHA)), HN, Xs)
    fl = [Mf.mmd(Zf[yf == 1][rng.choice((yf == 1).sum(), len(Xs), replace=False)]) for _ in range(5)]
    out["floor_mmd"] = np.mean(fl, 0).tolist()
    which = ["dmu", "w", "frozen", "guidK0.25", "guidK1.0", "guidK4.0", "particle"]
    res, info = paths_for(Zf, yf, gf, Xs, Pho, Zh, h, HN, dmu, which, C=C, dsm=dsm)
    out["paths"] = res; out.update(info)
    s1 = {k: (round(v["summary"]["1.0"]["min"], 5), round(v["summary"]["1.0"]["dstar"], 2), round(v["summary"]["1.0"]["rise"], 2)) for k, v in res.items()}
    print(f"L{l}H{hh} {a.space} h={h:.2f} (c={h/base:.2f}) floor {out['floor_mmd'][1]:.5f} start {res['straight_dmu']['summary']['1.0']['start']:.5f} | (min,d*,rise) {s1} {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(f"{a.out}/L{l}H{hh}_{a.space}.json", "w"))
    if a.space == "z" and a.nulls > 0:
        nulls = {"rshift": [], "gshared": []}
        Zn_f, Zp_f = Zf[yf == 0], Zf[yf == 1]
        for kind in nulls:
            for r in range(a.nulls):
                aa, bb = null_sample(kind, np.concatenate([Zn_f, Xs]), np.concatenate([Zp_f, Pho]), rng)
                an, hn = aa[:len(Zn_f)], aa[len(Zn_f):]; bp, hp = bb[:len(Zp_f)], bb[len(Zp_f):]
                Zf_ = np.concatenate([an, bp]); yf_ = np.r_[np.zeros(len(an), int), np.ones(len(bp), int)]
                gf_ = np.r_[gf[yf == 0], gf[yf == 1]]
                dmu_ = bp.mean(0) - an.mean(0)
                rr, _ = paths_for(Zf_, yf_, gf_, hn, hp, np.concatenate([hn, hp]), h, HN, dmu_, ["dmu", "frozen", "guidK1.0", "particle"], keep_curves=False)
                nulls[kind].append({k: v["summary"] for k, v in rr.items()})
            print(f"  null {kind}: gap(dmu - min(guid,part)) {[round(x['straight_dmu']['1.0']['min'] - min(x['guid_kde_k1.0']['1.0']['min'], x['particle_kl']['1.0']['min']), 5) for x in nulls[kind][:5]]} {time.time()-t0:.0f}s", flush=True)
            out["nulls"] = nulls
            json.dump(out, open(f"{a.out}/L{l}H{hh}_{a.space}.json", "w"))


if __name__ == "__main__":
    main()
