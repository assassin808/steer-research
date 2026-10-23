"""[harness addition, 2026-10-23] Confirmatory replication of C_inf vs IDEEA (prereg ~/steer_math/results/confirm_grad_prereg.md).

stages
  dirs       (CPU) bootstrap seeds b1-b3 of the Euclidean r_fix gradient at the head site (C per-head / H global, bridge file format)
             and at L13 (K = eucref-fix, runs_ng_confirm/directions_fix_L13.npz); IDEEA nc2 KMeans refits random_state 0/1/2
             (24 used heads; seed 0 must reproduce the paper nc_2 file). Sanity checks -> runs_ng/confirm/dirs.json
  devspecs   print the dev calibration specs (label-free KL sweeps)
  testspecs  strengths for target dev KLs {0.30, 0.518, 0.80} per arm -> confirm_specs.json; prints the test specs still to run
Generation: bridge.py eval (br: specs; fx: specs with NG_RUNS=runs_ng_confirm), tags development_confirm / test_confirm.
"""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(HERE)
import bridge as B  # noqa: E402

CONF = os.path.join(ROOT, "runs_ng", "confirm"); os.makedirs(CONF, exist_ok=True)
KRUNS = os.path.join(ROOT, "runs_ng_confirm"); os.makedirs(KRUNS, exist_ok=True)
SEEDS = (1, 2, 3)
LEVELS = {"lo": 0.30, "mid": 0.518, "hi": 0.80}
C_GRID = [0.6219, 1.1789, 2.2954, 3.8222]
H_GRID = [0.4023, 0.8799, 1.784, 3.7908]
K_GRID = [0.1917, 0.2804, 0.3293, 0.3731]
A_GRID = [2, 4, 6]


def draws(R, b):
    """row-index lists (correct, incorrect) per drawn question; b=0 -> identity (all usable train questions, all kept refs)."""
    sel = R["is_train"] & ~R["noninfo"]
    q, c = R["qidx"], R["correct"]
    per = []
    for i in np.unique(q[sel]):
        ic, ii = np.where(sel & (q == i) & c)[0], np.where(sel & (q == i) & ~c)[0]
        if len(ic) and len(ii):
            per.append((ic, ii))
    if b == 0:
        return per
    rs = np.random.RandomState(20261023 + b)
    out = []
    for k in rs.randint(0, len(per), len(per)):
        ic, ii = per[k]
        out.append((rs.choice(ic, len(ic)), rs.choice(ii, len(ii))))
    return out


def grad(U, D):
    return np.mean([U[ic].astype(np.float64).mean(0) - U[ii].astype(np.float64).mean(0) for ic, ii in D], 0)


def stage_dirs(a):
    R = dict(np.load(f"{B.OUT}/refs.npz")); Rr = dict(np.load(f"{ROOT}/runs_ng/refU_L13.npz"))
    for k in ("qidx", "correct", "noninfo", "is_train", "ntok"):
        assert (R[k] == Rr[k]).all(), k
    res = {"sanity": {}, "cos_vs_orig": {}, "n_q": {}}
    g0H, g0R = grad(R["U"], draws(R, 0)), grad(Rr["U"], draws(R, 0))
    # identity resample reproduces the bridge files
    cj = np.array(json.load(open(f"{B.DIRS}/C_inf.json"))["normalized_dir"])
    res["sanity"]["C_inf_min_perhead_cos"] = float(min(B.cos(B.unit(b), cj[l, h]) for b, (l, h) in zip(B.blocks(g0H), B.HEADS)))
    hj = np.array(json.load(open(f"{B.DIRS}/H_inf.json"))["normalized_dir"])
    hv = np.concatenate([hj[l, h] * s for (l, h), s in zip(B.HEADS, B.STD)])
    res["sanity"]["H_inf_cos"] = B.cos(hv, g0H)
    RZ = dict(np.load(f"{ROOT}/runs_ng/directions_fix_L13.npz"))
    res["sanity"]["efsum_cos"] = B.cos(g0R, RZ["efsum"])
    assert res["sanity"]["C_inf_min_perhead_cos"] > 1 - 1e-6 and res["sanity"]["H_inf_cos"] > 1 - 1e-6 and res["sanity"]["efsum_cos"] > 1 - 1e-9, res
    K = {"efsum": RZ["efsum"]}
    for b in SEEDS:
        D = draws(R, b); gH, gR = grad(R["U"], D), grad(Rr["U"], D)
        B.save_iti_format(f"C_inf_b{b}", [B.unit(x) for x in B.blocks(gH)])
        B.save_iti_format(f"H_inf_b{b}", B.gl_encode(gH))
        K[f"efsum_b{b}"] = gR
        res["n_q"][b] = len(D)
        res["cos_vs_orig"][f"b{b}"] = {"head_global": B.cos(gH, g0H), "head_perhead_eff": B.cos(B.ph_effective(gH), B.ph_effective(g0H)), "L13": B.cos(gR, g0R)}
    np.savez(f"{KRUNS}/directions_fix_L13.npz", **K)
    # IDEEA nc2 KMeans refits (dev rows = train+val, all refs, IDEEA's own data path), only the 24 steered heads are needed
    from sklearn.cluster import KMeans
    from methods.ideea import solve_qap_bruteforce
    from common.config import NEGATIVE, POSITIVE, resolve_dims
    from common.io import to_per_head
    from tasks.truthfulqa import TruthfulQA
    import pandas as pd
    df = pd.read_pickle(B.HEAD_PKL)
    dev_df, _, _ = TruthfulQA().probe_indices(df)
    ds = to_per_head(dev_df, resolve_dims("gemma2_2b"))
    paper = json.load(open(B.IDEEA_NC2))
    for s in (0, 1, 2):
        dirs, pc_, nc__, mt = {}, {}, {}, {}
        for (l, h) in B.HEADS:
            act = ds[l][h]; P_, N_ = act["data"][act["label"] == POSITIVE], act["data"][act["label"] == NEGATIVE]
            kp = KMeans(n_clusters=2, random_state=s, n_init=10).fit(P_); kn = KMeans(n_clusters=2, random_state=s, n_init=10).fit(N_)
            pc, nc = kp.cluster_centers_, kn.cluster_centers_
            dm = np.zeros((2, 2, P_.shape[1]))
            for i in range(2):
                for j in range(2):
                    d = (pc[j] - nc[i]).astype(np.float32); n = np.linalg.norm(d)
                    dm[i][j] = d / n if n > 0 else d
            perm, _ = solve_qap_bruteforce(dm, 2)
            dirs[(l, h)] = np.array([dm[i][perm[i]] for i in range(2)]); pc_[(l, h)], nc__[(l, h)], mt[(l, h)] = pc, nc, perm
        B.save_ideea_format(f"A_kms{s}", 2, dirs, pc_, nc__, mt)
        dd = max(float(np.abs(np.array(paper["cluster_dirs"][l][h]) - dirs[(l, h)]).max()) for l, h in B.HEADS)
        cc = float(np.mean([max(abs(B.cos(np.array(paper["cluster_dirs"][l][h][i]), dirs[(l, h)][k])) for k in range(2)) for l, h in B.HEADS for i in range(2)]))
        same_match = all(list(map(int, paper["matching"][l][h])) == list(map(int, mt[(l, h)])) for l, h in B.HEADS)
        res["sanity"][f"A_kms{s}_vs_paper"] = {"max_abs_dir_diff": dd, "mean_best_cos": cc, "same_matching": same_match}
        print(s, res["sanity"][f"A_kms{s}_vs_paper"], flush=True)
    assert res["sanity"]["A_kms0_vs_paper"]["max_abs_dir_diff"] < 1e-5, res["sanity"]
    json.dump(res, open(f"{CONF}/dirs.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def dev_specs():
    sp = []
    for b in SEEDS:
        sp += [f"br:C_inf_b{b}:{c:g}" for c in C_GRID] + [f"br:H_inf_b{b}:{c:g}" for c in H_GRID] + [f"fx:efsum_b{b}:13:{c:g}" for c in K_GRID]
    for s in (1, 2):
        sp += [f"br:A_kms{s}:{c:g}" for c in A_GRID]
    sp += ["br:A_kms0:4", "iti:8:16"]
    return sp


def kl_of(path):
    return json.load(open(path))["summary"]["kl_tok_mean"]


def fname(spec):
    return spec.replace(":", "_") + ".json"


def sweep(arm):
    """-> list of (strength, dev KL) and a function strength -> test spec"""
    G = ROOT + "/results_gm/"
    if arm == "IDEEA_s0":
        return [(c, kl_of(G + f"development_grid/ideea_{c}_24_min_perp_2.json")) for c in A_GRID], lambda c: f"ideea:{c:g}:24:min_perp:2"
    if arm in ("IDEEA_s1", "IDEEA_s2"):
        s = arm[-1]
        return [(c, kl_of(G + f"development_confirm/br_A_kms{s}_{c:g}.json")) for c in A_GRID], lambda c: f"br:A_kms{s}:{c:g}"
    if arm == "ITI":
        return [(c, kl_of(G + f"development_grid/iti_{c}_16.json" if c != 8 else G + "development_confirm/iti_8_16.json")) for c in (2, 4, 6, 8)], lambda c: f"iti:{c:g}:16"
    fam, seed = arm.split("_")
    grid = {"C": C_GRID, "H": H_GRID, "K": K_GRID}[fam]
    if seed == "orig":
        if fam == "K":
            return [(c, kl_of(G + f"development_fix/fx_efsum_13_{c:g}.json")) for c in grid], lambda c: f"fx:efsum:13:{c:g}"
        return [(c, kl_of(G + f"development_bridge/br_{fam}_inf_{c:g}.json")) for c in grid], lambda c: f"br:{fam}_inf:{c:g}"
    if fam == "K":
        return [(c, kl_of(G + f"development_confirm/fx_efsum_{seed}_13_{c:g}.json")) for c in grid], lambda c: f"fx:efsum_{seed}:13:{c:g}"
    return [(c, kl_of(G + f"development_confirm/br_{fam}_inf_{seed}_{c:g}.json")) for c in grid], lambda c: f"br:{fam}_inf_{seed}:{c:g}"


def interp(pts, T):
    pts = sorted(pts)
    for (c0, k0), (c1, k1) in zip(pts[:-1], pts[1:]):
        if k0 <= T <= k1 and k1 > k0:
            return round(float(np.exp(np.log(c0) + (np.log(T) - np.log(k0)) / (np.log(k1) - np.log(k0)) * (np.log(c1) - np.log(c0)))), 4)
    (c0, k0), (c1, k1) = (pts[0], pts[1]) if T < pts[0][1] else (pts[-2], pts[-1])
    return round(float(c1 * np.exp((np.log(T) - np.log(k1)) / ((np.log(k1) - np.log(k0)) / (np.log(c1) - np.log(c0))))), 4)


ARMS = ["IDEEA_s0", "IDEEA_s1", "IDEEA_s2", "ITI"] + [f"{f}_{s}" for f in "CHK" for s in ["orig"] + [f"b{b}" for b in SEEDS]]
REUSE = {("base", "mid"): "results_gm/test_grid/base.json", ("IDEEA_s0", "mid"): "results_gm/test_grid/ideea_4_24_min_perp_2.json",
         ("C_orig", "mid"): "results_gm/test_bridge/br_C_inf_2.0749.json", ("H_orig", "mid"): "results_gm/test_bridge/br_H_inf_1.6322.json",
         ("K_orig", "mid"): "results_gm/test_fix/fx_efsum_13_0.2673.json"}


def test_specs():
    out = {"levels": LEVELS, "runs": {}, "sweeps": {}}
    todo = []
    out["runs"]["base|mid"] = {"spec": "base", "path": REUSE[("base", "mid")]}
    for arm in ARMS:
        pts, mk = sweep(arm)
        out["sweeps"][arm] = pts
        for lv, T in LEVELS.items():
            if (arm, lv) in REUSE:
                p = REUSE[(arm, lv)]; out["runs"][f"{arm}|{lv}"] = {"spec": json.load(open(ROOT + "/" + p))["summary"]["spec"], "path": p, "reused": True}
                continue
            c = 4 if (arm == "IDEEA_s0" and lv == "mid") else interp(pts, T)
            s = mk(c); out["runs"][f"{arm}|{lv}"] = {"spec": s, "path": "results_gm/test_confirm/" + fname(s)}
            todo.append(s)
    json.dump(out, open(ROOT + "/confirm_specs.json", "w"), indent=1)
    return todo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["dirs", "devspecs", "testspecs"])
    a = ap.parse_args()
    if a.stage == "dirs":
        stage_dirs(a)
    elif a.stage == "devspecs":
        print(" ".join(dev_specs()))
    else:
        print(" ".join(test_specs()))
