"""POST HOC D3: tilted-empirical null (linear log-ratio, real data shape) + R-shift null at the CV-selected ext alpha (z)."""
import argparse, json, os, time
import numpy as np
from sklearn.neighbors import NearestNeighbors
import trait_flow as T
from tf_long import integrate2
from ot_geom import META, HEADS, make_space, null_sample
ap = argparse.ArgumentParser(); ap.add_argument("--heads", required=True); ap.add_argument("--out", required=True); ap.add_argument("--jit", type=float, default=0.1); a = ap.parse_args()
z = np.load(META); lab, tr, q = z["label"], z["is_train"], z["q"]; y = (lab == 1).astype(int)
os.makedirs(a.out, exist_ok=True)
for hs_ in a.heads.split(","):
    l, h = map(int, hs_.split(":")); t0 = time.time(); rng = np.random.default_rng(23 + 1000 * l + h)
    ext = json.load(open(f"/home/yang0531/scratch/ot_geom/tflong/L{l}H{h}_z_ext.json")); alpha = ext["alpha"]
    X = np.load(f"{HEADS}/L{l}H{h}.npy").astype(float); m, A = make_space("z", X[tr]); Z = (X - m) @ A
    Zf, yf, gf = Z[tr], y[tr], q[tr]; Zh, yh = Z[~tr], y[~tr]; Xs = Zh[yh == 0]; Zpos_ho = Zh[yh == 1]
    dmu = Zf[yf == 1].mean(0) - Zf[yf == 0].mean(0); HN = float(np.linalg.norm(dmu)); T.HN = HN
    C, _ = T.cv_select(Zf, yf, gf, T.C_GRID, T.fit_lr); lr = T.fit_lr(Zf, yf, C)
    p = lr.predict_proba(Zf)[:, 1]
    jit = a.jit * np.median(NearestNeighbors(n_neighbors=11).fit(Zf).kneighbors(Zf)[0][:, -1])
    hs1 = [T.H_FRAC * HN] * 150
    nN, nP = int((yf == 0).sum()), int((yf == 1).sum())
    res = dict(layer=l, head=h, alpha=alpha, jitter=float(jit), data_turn=ext["unproj"]["turn"], tilt=[], rshift=[])
    def run(a_, b_, starts, starts_pos):
        Zn_ = np.concatenate([a_, b_]); yn_ = np.r_[np.zeros(len(a_), int), np.ones(len(b_), int)]
        S = T.MLPScore(T.fit_mlp(Zn_, yn_, alpha)); nnp = NearestNeighbors(n_neighbors=10).fit(b_)
        band = np.quantile(nnp.kneighbors(starts_pos)[0][:, -1], 0.95)
        lrn = T.fit_lr(Zn_, yn_, C)
        r = integrate2(starts, S, None, nnp, band, hs1, HN)
        return dict(turn=r["turn"])
    for r in range(5):
        iN = rng.choice(len(Zf), nN, p=(1 - p) / (1 - p).sum()); iP = rng.choice(len(Zf), nP, p=p / p.sum())
        a_ = Zf[iN] + jit * rng.standard_normal((nN, Zf.shape[1])); b_ = Zf[iP] + jit * rng.standard_normal((nP, Zf.shape[1]))
        res["tilt"].append(run(a_, b_, Xs, Zpos_ho))
    for r in range(3):
        a_all, b_all = null_sample("rshift", np.concatenate([Zf[yf == 0], Xs]), np.concatenate([Zf[yf == 1], Zpos_ho]), rng)
        res["rshift"].append(run(a_all[:nN], b_all[:nP], a_all[nN:], b_all[nP:]))
    print(f"L{l}H{h} a={alpha} data turn@1 {ext['unproj']['turn']['1']:.1f} @3 {ext['unproj']['turn']['3']:.1f} | tilt @1 {[round(x['turn'][1],1) for x in res['tilt']]} "
          f"@3 {[round(x['turn'][3],1) for x in res['tilt']]} | rshift @1 {[round(x['turn'][1],1) for x in res['rshift']]} {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(f"{a.out}/L{l}H{h}.json", "w"))
