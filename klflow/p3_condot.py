"""P3: conditional (within-question) OT vs unconditional OT vs grad s. Prereg results/klflow_prereg.md."""
import json, sys, numpy as np
from scipy.optimize import linprog
sys.path.insert(0, "/home/yang0531/scratch/steer_live/ot_geometry")
import trait_flow as T
from ot_geom import META, HEADS, make_space, sinkhorn, cosrows

def exact_ot(Xn, Xp):
    m, n = len(Xn), len(Xp); C = ((Xn[:, None] - Xp[None]) ** 2).sum(-1).ravel()
    Aeq = np.zeros((m + n, m * n))
    for i in range(m): Aeq[i, i * n:(i + 1) * n] = 1
    for j in range(n): Aeq[m + j, j::n] = 1
    beq = np.r_[np.full(m, 1 / m), np.full(n, 1 / n)]
    P = linprog(C, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs").x.reshape(m, n)
    return (P @ Xp) / P.sum(1, keepdims=True) - Xn

z = np.load(META); lab, tr, q = z["label"], z["is_train"], z["q"]; y = (lab == 1).astype(int)
hj = json.load(open("/home/yang0531/scratch/steer_live/ot_geometry/heads.json")); out = {}
for grp, hs in [("top", hj["top24"]), ("rnd", hj["random8"])]:
    for l, h in hs:
        rng = np.random.default_rng(5 + 1000 * l + h)
        X = np.load(f"{HEADS}/L{l}H{h}.npy").astype(float); m, A = make_space("z", X[tr]); Z = (X - m) @ A
        Zf, yf, gf = Z[tr], y[tr], q[tr]
        S = T.MLPScore(T.fit_mlp(Zf, yf, 100.0)); C, _ = T.cv_select(Zf, yf, gf, T.C_GRID, T.fit_lr); w = T.fit_lr(Zf, yf, C).coef_[0]
        dmu = Zf[yf == 1].mean(0) - Zf[yf == 0].mean(0)
        hq = [u for u in np.unique(q[~tr]) if ((q == u) & (y == 1)).sum() >= 1 and ((q == u) & (y == 0)).sum() >= 1]
        Xn_l, D_l, G_l = [], [], []
        for u in hq:
            Xn = Z[(q == u) & (y == 0)]; Xp = Z[(q == u) & (y == 1)]
            Xn_l.append(Xn); D_l.append(exact_ot(Xn, Xp))
        Xn_all = np.concatenate(Xn_l); Dc = np.concatenate(D_l); G = S.grad(Xn_all)
        # unconditional entropic OT (all dev NEG -> all dev POS), take held-out NEG rows in the same order
        ot = sinkhorn(Z[y == 0], Z[y == 1], 0.01); idx_neg = np.where(y == 0)[0]
        pos_in = {i: k for k, i in enumerate(idx_neg)}
        rows = np.concatenate([np.where((q == u) & (y == 0))[0] for u in hq]); Du = ot["d"][[pos_in[i] for i in rows]]
        r = dict(group=grp, n_q=len(hq), n_neg=len(Xn_all),
                 cond_grad=float(cosrows(Dc, G).mean()), uncond_grad=float(cosrows(Du, G).mean()),
                 cond_w=float(cosrows(Dc, np.tile(w, (len(Dc), 1))).mean()), uncond_w=float(cosrows(Du, np.tile(w, (len(Du), 1))).mean()),
                 cond_dmu=float(cosrows(Dc, np.tile(dmu, (len(Dc), 1))).mean()), uncond_dmu=float(cosrows(Du, np.tile(dmu, (len(Du), 1))).mean()),
                 dmu_grad=float(cosrows(np.tile(dmu, (len(G), 1)), G).mean()), w_grad=float(cosrows(np.tile(w, (len(G), 1)), G).mean()))
        sh = []
        for rep in range(20):
            perm = rng.permutation(len(hq)); Ds = []
            for k, u in enumerate(hq):
                v = hq[perm[k]] if perm[k] != k else hq[(k + 1) % len(hq)]
                Ds.append(exact_ot(Z[(q == u) & (y == 0)], Z[(q == v) & (y == 1)]))
            sh.append(float(cosrows(np.concatenate(Ds), G).mean()))
        r["shuffle_grad"] = sh
        sp = []
        for u in hq:
            ni = rng.permutation(np.where((q == u) & (y == 0))[0]); pi = rng.permutation(np.where((q == u) & (y == 1))[0])
            if len(ni) >= 2 and len(pi) >= 2:
                dA = exact_ot(Z[ni[: len(ni) // 2]], Z[pi[: len(pi) // 2]]).mean(0); dB = exact_ot(Z[ni[len(ni) // 2:]], Z[pi[len(pi) // 2:]]).mean(0)
                sp.append(float(dA @ dB / (np.linalg.norm(dA) * np.linalg.norm(dB) + 1e-12)))
        r["splithalf_cos"] = float(np.mean(sp)); r["splithalf_nq"] = len(sp)
        out[f"L{l}H{h}"] = r
        print(grp, f"L{l}H{h}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "shuffle_grad"},
              "shuffle", round(float(np.median(sh)), 3), round(float(np.quantile(sh, .975)), 3), flush=True)
json.dump(out, open("/home/yang0531/scratch/klflow_out/p3.json", "w"), indent=1)
