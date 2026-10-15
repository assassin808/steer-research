"""H1 v2 follow-up (CPU, from saved score features): better-behaved steerability estimators than the floored
cross-fitted R^2, and per-question first-order predictions along FIXED directions.

  (a) pred_eff_v (already stored): Cov(s, u.v)/sqrt(Var(u.v)) on the fit samples = first-order dT/ds along a fixed
      direction v (rand, caa, ngG are not fitted on these samples, so no overfitting). Compared with realized
      efficiency gain/sqrt(2 KL) on independent eval samples.
  (b) split-sample quadratic forms (unbiased up to the metric estimate): eta_split = g_1^T M g_2 / Var, with g_1, g_2
      from disjoint halves of the fit samples and M = (G_glob + lam I)^-1 (global train metric, fixed) or the
      k-dim sketch metric. Reliability: the same estimator on the two independent 64-sample halves (quarters inside).
"""
import glob, json, os
import numpy as np
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.expanduser("~/steer_math/ideea_harness/runs_ng/directions_L14_qwen.npz"))
G, evecs, evals = D["G"], D["evecs"], D["evals"]
lam = 3 * np.trace(G) / G.shape[0]
Minv = np.linalg.inv(G + lam * np.eye(G.shape[0]))


def cov_su(U, s):
    return (U - U.mean(0)).T @ (s - s.mean()) / len(s)


def eta_split(U, s, M, seed=0):
    idx = np.random.RandomState(seed).permutation(len(s)); a, b = idx[: len(s) // 2], idx[len(s) // 2:]
    return float(cov_su(U[a], s[a]) @ M @ cov_su(U[b], s[b]))


def sketch_M(U, k):
    B = evecs[:, :k]
    Z = U @ B
    C = np.cov(Z.T) + 1e-2 * np.trace(np.cov(Z.T)) / k * np.eye(k)
    return B, np.linalg.inv(C)


rows = []
for f in sorted(glob.glob(f"{HERE}/runs_h1v2/q0*.json")):
    r = json.load(open(f)); z = np.load(f.replace(".json", ".npz"))
    U, s = z["U"].astype(np.float64), z["s_fit"].astype(np.float64)
    var = s.var()
    row = {"q": r["q"], "var": var}
    for d in ("pq", "rand", "caa", "ngG"):
        for e in ("0.05", "0.2"):
            st = r["steer"][f"{d}@{e}"]
            row[f"eff_{d}@{e}"] = st["gain"] / np.sqrt(2 * max(st["kl_rb"], 1e-6))
        row[f"pred_{d}"] = r[f"pred_eff_{d}"]
    # global-metric split-sample eta (full d), averaged over 5 random splits; and its two-halves reliability
    row["eta_glob"] = np.mean([eta_split(U, s, Minv, sd) for sd in range(5)])
    row["eta_glob_A"] = np.mean([eta_split(U[:64], s[:64], Minv, sd) for sd in range(5)])
    row["eta_glob_B"] = np.mean([eta_split(U[64:], s[64:], Minv, sd) for sd in range(5)])
    for k in (4, 8, 16, 32):
        B, Mk = sketch_M(U, k)
        Z = U @ B
        row[f"eta_k{k}"] = np.mean([eta_split(Z, s, Mk, sd) for sd in range(5)])
        BA, MA = sketch_M(U[:64], k); BB, MB = sketch_M(U[64:], k)
        row[f"eta_k{k}_A"] = np.mean([eta_split(U[:64] @ BA, s[:64], MA, sd) for sd in range(5)])
        row[f"eta_k{k}_B"] = np.mean([eta_split(U[64:] @ BB, s[64:], MB, sd) for sd in range(5)])
    rows.append(row)

out = {"n": len(rows)}
g = lambda k: np.array([r[k] for r in rows])  # noqa
# (a) fixed-direction first-order predictions vs realized efficiency
out["fixed_direction_prediction"] = {}
for d in ("rand", "caa", "ngG", "pq"):
    for e in ("0.05", "0.2"):
        x, y = g(f"pred_{d}"), g(f"eff_{d}@{e}")
        out["fixed_direction_prediction"][f"{d}@{e}"] = {"spearman": float(spearmanr(x, y)[0]), "pearson": float(pearsonr(x, y)[0]),
                                                          "mean_pred": float(x.mean()), "mean_realized": float(y.mean()),
                                                          "slope": float(np.polyfit(x, y, 1)[0])}
# (b) split-sample eta: reliability and prediction
out["eta_split"] = {}
for k in ["glob"] + [f"k{k}" for k in (4, 8, 16, 32)]:
    a, b = g(f"eta_{k}_A"), g(f"eta_{k}_B")
    e_all = g(f"eta_{k}")
    ok = g("var") > 0
    rec = {"reliability_pearson": float(pearsonr(a[ok], b[ok])[0]), "reliability_spearman": float(spearmanr(a[ok], b[ok])[0]),
           "share_negative": float((e_all[ok] < 0).mean()), "mean_eta": float(e_all[ok].mean()), "mean_var": float(g("var")[ok].mean())}
    for d in ("pq", "ngG"):
        for e in ("0.05", "0.2"):
            y = g(f"eff_{d}@{e}")
            rec[f"rho_eta_eff_{d}@{e}"] = float(spearmanr(e_all, y)[0])
            rec[f"rho_sqrt_eta_plus_eff_{d}@{e}"] = float(spearmanr(np.sqrt(np.maximum(e_all, 0)), y)[0])
    out["eta_split"][k] = rec
# Var alone for reference
out["rho_var_eff"] = {f"{d}@{e}": float(spearmanr(g("var"), g(f"eff_{d}@{e}"))[0]) for d in ("pq", "ngG", "rand", "caa") for e in ("0.05", "0.2")}
json.dump(out, open(os.path.expanduser("~/steer_math/results/h1v2_eta2.json"), "w"), indent=1)
print(json.dumps(out, indent=1))


# --- partial correlations given Var, bound check, eta_hat = eta/Var
from scipy.stats import rankdata


def partial_rho(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(pearsonr(ex, ey)[0])


ok = g("var") > 0
extra = {}
for k in ("k8", "k16", "k32"):
    e = g(f"eta_{k}")[ok]; v = g("var")[ok]
    rec = {"share_eta_gt_var": float((e > v).mean()), "median_eta_hat": float(np.median(e / v)), "mean_eta_hat": float(np.mean(e / v))}
    for d in ("pq", "ngG"):
        for ee in ("0.05", "0.2"):
            y = g(f"eff_{d}@{ee}")[ok]
            rec[f"partial_rho_eta_given_var_{d}@{ee}"] = partial_rho(e, y, v)
            rec[f"rho_eta_hat_{d}@{ee}"] = float(spearmanr(e / v, y)[0])
    extra[k] = rec
for d in ("ngG", "rand", "caa"):
    for ee in ("0.05", "0.2"):
        y = g(f"eff_{d}@{ee}")[ok]
        extra[f"partial_rho_pred_{d}_given_var@{ee}"] = partial_rho(g(f"pred_{d}")[ok], y, g("var")[ok])
out["partial_and_bound"] = extra
json.dump(out, open(os.path.expanduser("~/steer_math/results/h1v2_eta2.json"), "w"), indent=1)
print(json.dumps(extra, indent=1))


# --- bootstrap CIs for headline correlations
rng = np.random.RandomState(0)
V = g("var")[ok]
heads = {"spearman_pred_ngG_eff@0.2": (g("pred_ngG")[ok], g("eff_ngG@0.2")[ok], None),
         "partial_pred_ngG_given_var@0.2": (g("pred_ngG")[ok], g("eff_ngG@0.2")[ok], V),
         "spearman_pred_ngG_eff@0.05": (g("pred_ngG")[ok], g("eff_ngG@0.05")[ok], None),
         "partial_eta_k8_given_var_pq@0.2": (g("eta_k8")[ok], g("eff_pq@0.2")[ok], V),
         "partial_eta_k32_given_var_pq@0.2": (g("eta_k32")[ok], g("eff_pq@0.2")[ok], V),
         "partial_eta_k8_given_var_ngG@0.2": (g("eta_k8")[ok], g("eff_ngG@0.2")[ok], V),
         "spearman_var_pq@0.2": (V, g("eff_pq@0.2")[ok], None)}
ci = {}
for name, (x, y, z) in heads.items():
    vals = []
    for _ in range(2000):
        i = rng.randint(0, len(x), len(x))
        vals.append(partial_rho(x[i], y[i], z[i]) if z is not None else float(spearmanr(x[i], y[i])[0]))
    est = partial_rho(x, y, z) if z is not None else float(spearmanr(x, y)[0])
    ci[name] = [est, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
out["bootstrap_ci"] = ci
json.dump(out, open(os.path.expanduser("~/steer_math/results/h1v2_eta2.json"), "w"), indent=1)
print(json.dumps(ci, indent=1))
