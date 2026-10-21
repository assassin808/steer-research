"""POST-HOC (D2): held-out gate for the alpha-extended MLP: LR vs MLP(alpha_ext) log-loss/AUC, bootstrap over held-out questions."""
import json, glob, numpy as np
import trait_flow as T
from ot_geom import META, HEADS, make_space
from sklearn.metrics import roc_auc_score
z = np.load(META); lab, tr, q = z["label"], z["is_train"], z["q"]; y = (lab == 1).astype(int)
hj = json.load(open("heads.json")); out = {}
for grp, hs in [("top", hj["top24"]), ("rnd", hj["random8"])]:
    for l, h in hs:
        import os
        fn_ = f"/home/yang0531/scratch/ot_geom/tflong/L{l}H{h}_z_ext.json"
        if not os.path.exists(fn_):
            print("missing", fn_); continue
        r = json.load(open(fn_))
        X = np.load(f"{HEADS}/L{l}H{h}.npy").astype(float); m, A = make_space("z", X[tr]); Z = (X - m) @ A
        C, _ = T.cv_select(Z[tr], y[tr], q[tr], T.C_GRID, T.fit_lr)
        lr = T.fit_lr(Z[tr], y[tr], C); mlp = T.fit_mlp(Z[tr], y[tr], r["alpha"])
        pl, pm = lr.predict_proba(Z[~tr])[:, 1], mlp.predict_proba(Z[~tr])[:, 1]; yh, gh = y[~tr], q[~tr]
        dl = T.logloss(pl, yh) - T.logloss(pm, yh); uq = np.unique(gh); idx = {u: np.where(gh == u)[0] for u in uq}
        rng = np.random.default_rng(0)
        bo = [dl[np.concatenate([idx[u] for u in rng.choice(uq, len(uq))])].mean() for _ in range(1000)]
        bauc = []
        for _ in range(300):
            s = np.concatenate([idx[u] for u in rng.choice(uq, len(uq))])
            bauc.append(roc_auc_score(yh[s], pm[s]) - roc_auc_score(yh[s], pl[s]))
        out[f"L{l}H{h}"] = dict(group=grp, alpha=r["alpha"], C=C, ll_lr=float(T.logloss(pl, yh).mean()), ll_mlp=float(T.logloss(pm, yh).mean()),
                               dll=float(dl.mean()), dll_ci=[float(np.quantile(bo, .025)), float(np.quantile(bo, .975))],
                               auc_lr=float(roc_auc_score(yh, pl)), auc_mlp=float(roc_auc_score(yh, pm)),
                               dauc_ci=[float(np.quantile(bauc, .025)), float(np.quantile(bauc, .975))])
        o = out[f"L{l}H{h}"]; print(grp, f"L{l}H{h}", o["alpha"], f"dLL {o['dll']:+.4f} [{o['dll_ci'][0]:+.4f},{o['dll_ci'][1]:+.4f}] AUC {o['auc_lr']:.3f}->{o['auc_mlp']:.3f} dAUC CI [{o['dauc_ci'][0]:+.3f},{o['dauc_ci'][1]:+.3f}]", flush=True)
json.dump(out, open("/home/yang0531/scratch/ot_geom/gate_ext.json", "w"), indent=1)
