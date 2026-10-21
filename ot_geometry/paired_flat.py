"""POST-HOC exploratory: flatness of the per-question displacement delta_q = mean POS_q - mean NEG_q (z space, fit stats).
Noise-unbiased: split each question's POS and NEG rows into halves A/B; signal second moment = mean_q dA_q.dB_q,
constant part = mean(dA).mean(dB). R2_pair = const / signal (=1 for a constant field, up to sampling error).
Also cos between delta_q (full) and the global mean diff, and the across-question 'within' share."""
import json, numpy as np
H = "/home/yang0531/scratch/ot_geom/heads"; m = np.load("/home/yang0531/scratch/ot_geom/meta.npz")
lab, q, tr = m["label"], m["q"], m["is_train"]
hj = json.load(open("heads.json")); heads = [("top", *h) for h in hj["top24"]] + [("rnd", *h) for h in hj["random8"]]
rng = np.random.default_rng(0); out = {}
for g, l, h in heads:
    X = np.load(f"{H}/L{l}H{h}.npy").astype(np.float64)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8; Z = (X - mu) / sd
    qs = [u for u in np.unique(q) if ((q == u) & (lab == 1)).sum() >= 2 and ((q == u) & (lab == -1)).sum() >= 2]
    r2s, sig, con = [], [], []
    for rep in range(20):
        dA, dB = [], []
        for u in qs:
            P = rng.permutation(np.where((q == u) & (lab == 1))[0]); N = rng.permutation(np.where((q == u) & (lab == -1))[0])
            dA.append(Z[P[: len(P) // 2]].mean(0) - Z[N[: len(N) // 2]].mean(0))
            dB.append(Z[P[len(P) // 2:]].mean(0) - Z[N[len(N) // 2:]].mean(0))
        dA, dB = np.array(dA), np.array(dB)
        s = np.mean(np.sum(dA * dB, 1)); c = dA.mean(0) @ dB.mean(0)
        r2s.append(c / s); sig.append(s); con.append(c)
    dmu = Z[lab == 1].mean(0) - Z[lab == -1].mean(0)
    full = np.array([Z[(q == u) & (lab == 1)].mean(0) - Z[(q == u) & (lab == -1)].mean(0) for u in qs])
    cs = full @ dmu / (np.linalg.norm(full, axis=1) * np.linalg.norm(dmu))
    out[f"L{l}H{h}"] = dict(group=g, n_q=len(qs), r2_pair=float(np.median(r2s)), r2_pair_iqr=[float(np.quantile(r2s, .25)), float(np.quantile(r2s, .75))],
                           signal=float(np.median(sig)), const=float(np.median(con)), cos_dq_dmu=float(cs.mean()))
    print(f"{g} L{l}H{h} nq={len(qs)} R2_pair={np.median(r2s):.3f} [{np.quantile(r2s,.25):.3f},{np.quantile(r2s,.75):.3f}] cos(dq,dmu)={cs.mean():.3f}", flush=True)
json.dump(out, open("/home/yang0531/scratch/ot_geom/paired_flat.json", "w"), indent=1)
