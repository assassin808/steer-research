"""Synthetic power check at the real fit-set sizes (n-=1300, n+=1130 approx).
NEG: 2-mode mixture (modes at +-4 e0), unit isotropic noise. POS = NEG-like mixture shifted by a field that is
a constant shift s/sqrt2 (e1+e2) (flat, frac=0) or mode-dependent e1 / e2 shifts (curved).
'frac' mixes them: shift_mode0 = s*[(1-f)/sqrt2 (e1+e2) + f e1], mode1 similarly with e2. True field R2 decreases with f."""
import numpy as np, torch, json, sys
from ot_geom import sinkhorn, flat_stats, null_sample
torch.set_num_threads(4)
rng = np.random.default_rng(0)
n, m = 1300, 1130
res = []
for D in [8, 32, 256]:
    for s in [1.0, 3.0]:
        for f in [0.0, 0.25, 0.5, 1.0]:
            sn = rng.integers(0, 2, n); sp = rng.integers(0, 2, m)
            Xn = rng.standard_normal((n, D)); Xn[:, 0] += 4 * (2 * sn - 1)
            Xp = rng.standard_normal((m, D)); Xp[:, 0] += 4 * (2 * sp - 1)
            c0 = np.zeros(D); c0[1] = c0[2] = (1 - f) / np.sqrt(2)
            a0 = c0.copy(); a0[1] += f; a1 = c0.copy(); a1[2] += f
            Xp[sp == 0] += s * a0; Xp[sp == 1] += s * a1
            sh = np.stack([a0, a1]) * s; tr = (sh.mean(0) @ sh.mean(0)) / np.mean((sh ** 2).sum(1))
            st = lambda a, b: flat_stats(a, sinkhorn(a, b, 0.01)["d"], Xp.mean(0) - Xn.mean(0))["r2s"]
            d = st(Xn, Xp); nl = [st(*null_sample("rshift", Xn, Xp, rng)) for _ in range(6)]
            r = dict(D=D, s=s, f=f, true_r2=tr, r2s=d, null=nl, detect=bool(d < min(nl)))
            print(r, flush=True); res.append(r)
json.dump(res, open("/home/yang0531/scratch/ot_geom/power.json", "w"))
