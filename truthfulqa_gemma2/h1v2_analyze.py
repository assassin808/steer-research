"""Analyse H1 v2 (runs_h1v2/q0*.json) -> ~/steer_math/results/h1v2.json + printed tables."""
import glob
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser("~/steer_math/results")
EPS = ("0.05", "0.2")
DIRS = ("pq", "rand", "caa", "ngG")


def boot_ci(x, y, fn, B=2000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(x)
    vals = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        if np.std(x[i]) == 0 or np.std(y[i]) == 0:
            continue
        vals.append(fn(x[i], y[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def rho(x, y):
    return float(spearmanr(x, y)[0])


def loo_r2(X, y, lam=1e-3):
    """Leave-one-out R^2 of ridge on standardised X (intercept unpenalised)."""
    n = len(y)
    pred = np.zeros(n)
    for i in range(n):
        m = np.arange(n) != i
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-12
        Xm = (X[m] - mu) / sd
        w = np.linalg.solve(Xm.T @ Xm + lam * n * np.eye(X.shape[1]), Xm.T @ (y[m] - y[m].mean()))
        pred[i] = y[m].mean() + ((X[i] - mu) / sd) @ w
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{HERE}/runs_h1v2/q0*.json"))]
    n = len(R)
    res = {"n_questions": n}
    g = lambda k: np.array([r[k] for r in R], float)  # noqa: E731
    # --- reliability of eta_hat (two independent halves of 64 fit samples)
    rel = {}
    for k in ("eta_k8", "eta_k16", "eta_k", "eta_k64", "eta_k128", "eta_k256"):
        a, b = g(f"{k}_A"), g(f"{k}_B")
        ok = ~(np.isnan(a) | np.isnan(b))
        name = "eta_k32" if k == "eta_k" else k
        rel[name] = {"pearson": float(pearsonr(a[ok], b[ok])[0]) if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0 else None,
                     "spearman": rho(a[ok], b[ok]) if ok.sum() > 2 else None, "n": int(ok.sum()),
                     "mean_A": float(np.nanmean(a)), "mean_all": float(np.nanmean(g(f"{k}_all"))),
                     "share_zero_all": float(np.nanmean(g(f"{k}_all") == 0))}
    a, b = g("eta_k_A_raw"), g("eta_k_B_raw")
    ok = ~(np.isnan(a) | np.isnan(b))
    rel["eta_k32_unfloored"] = {"pearson": float(pearsonr(a[ok], b[ok])[0]), "n": int(ok.sum())}
    res["eta_reliability"] = rel
    # --- realized KL matching and gains
    tab = {}
    for e in EPS:
        for d in DIRS:
            key = f"{d}@{e}"
            gain = np.array([r["steer"][key]["gain"] for r in R])
            kl = np.array([r["steer"][key]["kl_rb"] for r in R])
            klmc = np.array([r["steer"][key]["kl_mc"] for r in R])
            ga, gb = np.array([r["steer"][key]["gain_A"] for r in R]), np.array([r["steer"][key]["gain_B"] for r in R])
            eff = gain / np.sqrt(2 * np.maximum(kl, 1e-6))
            rr = float(pearsonr(ga, gb)[0]) if ga.std() > 0 and gb.std() > 0 else float("nan")
            tab[key] = {"mean_gain": float(gain.mean()), "se_gain": float(gain.std(ddof=1) / np.sqrt(n)),
                        "share_improved": float((gain > 0).mean()), "mean_kl_rb": float(kl.mean()), "median_kl_rb": float(np.median(kl)),
                        "sd_kl_rb": float(kl.std()), "mean_kl_mc": float(klmc.mean()),
                        "mean_eff": float(eff.mean()), "se_eff": float(eff.std(ddof=1) / np.sqrt(n)),
                        "gain_splithalf_r": rr, "gain_reliability_SB": float(2 * rr / (1 + rr)) if rr == rr else None,
                        "mean_within_q_gain_se": float(np.mean([r["steer"][key]["gain_se"] for r in R])),
                        "sd_gain_across_q": float(gain.std(ddof=1))}
            # paired difference vs random direction at the same budget
            if d != "rand":
                gr = np.array([r["steer"][f"rand@{e}"]["gain"] for r in R])
                dd = gain - gr
                tab[key]["minus_rand"] = float(dd.mean()); tab[key]["minus_rand_se"] = float(dd.std(ddof=1) / np.sqrt(n))
            if d != "caa":
                gc = np.array([r["steer"][f"caa@{e}"]["gain"] for r in R])
                dd = gain - gc
                tab[key]["minus_caa"] = float(dd.mean()); tab[key]["minus_caa_se"] = float(dd.std(ddof=1) / np.sqrt(n))
    res["steering"] = tab
    # --- bound: gain <= sqrt(2 KL Var_p(s)) to first order (any direction)
    bound = {}
    var = g("var_fit")
    for e in EPS:
        for d in DIRS:
            key = f"{d}@{e}"
            gain = np.array([r["steer"][key]["gain"] for r in R]); se = np.array([r["steer"][key]["gain_se"] for r in R])
            kl = np.array([r["steer"][key]["kl_rb"] for r in R])
            bd = np.sqrt(2 * kl * var)
            bound[key] = {"mean_gain": float(gain.mean()), "mean_bound": float(bd.mean()),
                          "share_gain_gt_bound": float((gain > bd).mean()),
                          "share_gain_gt_bound_2se": float((gain - bd > 2 * se).mean()),
                          "mean_ratio_gain_to_bound": float(np.mean(gain / np.maximum(bd, 1e-6)))}
    res["bound"] = bound
    # --- prediction of per-question efficiency
    eta_hat = np.nan_to_num(g("eta_k_all"))
    eta = var * eta_hat
    feats = {"var": var, "eta_hat": eta_hat, "sqrt_eta": np.sqrt(eta), "p": g("p_fit"),
             "h_norm": g("h_norm"), "cos_h_caa": g("cos_h_caa"), "ent_first": g("ent_first"), "ent_tok": g("ent_tok")}
    pred = {}
    euc = ["h_norm", "cos_h_caa", "ent_first", "ent_tok"]
    for e in EPS:
        for d in DIRS:
            key = f"{d}@{e}"
            gain = np.array([r["steer"][key]["gain"] for r in R])
            kl = np.array([r["steer"][key]["kl_rb"] for r in R])
            eff = gain / np.sqrt(2 * np.maximum(kl, 1e-6))
            P = {"spearman": {}, "spearman_ci": {}}
            for fn, x in feats.items():
                P["spearman"][fn] = rho(x, eff)
                P["spearman_ci"][fn] = boot_ci(x, eff, rho)
            Xe = np.stack([feats[f] for f in euc], 1)
            Xv = np.stack([feats[f] for f in euc + ["var"]], 1)
            Xa = np.stack([feats[f] for f in euc + ["var", "eta_hat", "sqrt_eta"]], 1)
            P["loo_r2_euclid"] = loo_r2(Xe, eff); P["loo_r2_euclid_var"] = loo_r2(Xv, eff); P["loo_r2_euclid_var_eta"] = loo_r2(Xa, eff)
            P["loo_r2_var_eta_only"] = loo_r2(np.stack([feats["var"], feats["eta_hat"], feats["sqrt_eta"]], 1), eff)
            # bootstrap the incremental LOO R^2 of (eta_hat, sqrt_eta) over (euclid + var)
            rng = np.random.RandomState(1)
            inc = []
            for _ in range(300):
                ii = rng.randint(0, len(eff), len(eff))
                if eff[ii].std() == 0:
                    continue
                inc.append(loo_r2(Xa[ii], eff[ii]) - loo_r2(Xv[ii], eff[ii]))
            P["delta_loo_r2_eta_over_euclid_var"] = float(P["loo_r2_euclid_var_eta"] - P["loo_r2_euclid_var"])
            P["delta_loo_r2_boot_ci"] = [float(np.percentile(inc, 2.5)), float(np.percentile(inc, 97.5))]
            # partial Spearman of eta_hat with efficiency controlling for Var (rank residuals)
            from scipy.stats import rankdata
            rv, re_, rh = rankdata(var), rankdata(eff), rankdata(eta_hat)
            res_e = re_ - np.polyval(np.polyfit(rv, re_, 1), rv); res_h = rh - np.polyval(np.polyfit(rv, rh, 1), rv)
            P["partial_spearman_eta_hat_given_var"] = float(pearsonr(res_e, res_h)[0])
            nz = eta_hat > 0
            P["n_eta_nonzero"] = int(nz.sum())
            P["mean_eff_eta_zero"] = float(eff[~nz].mean()) if (~nz).any() else None
            P["mean_eff_eta_nonzero"] = float(eff[nz].mean()) if nz.any() else None
            P["pearson_sqrt_eta_eff"] = float(pearsonr(np.sqrt(eta), eff)[0])
            P["mean_sqrt_eta"] = float(np.sqrt(eta).mean())
            P["mean_eff"] = float(eff.mean())
            pred[key] = P
    res["prediction_of_efficiency"] = pred
    # first-order theory check for pq: predicted dT/ds = sqrt(eta) vs realized efficiency
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/h1v2.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_questions", "eta_reliability")}, indent=1))
    print(f"\n{'dir@eps':12s} {'gain':>7s} {'se':>6s} {'KLrb':>6s} {'eff':>6s} {'-rand':>7s} {'relSB':>6s} {'sdQ':>5s} {'seQ':>5s}")
    for k, v in tab.items():
        print(f"{k:12s} {v['mean_gain']:+7.3f} {v['se_gain']:6.3f} {v['mean_kl_rb']:6.3f} {v['mean_eff']:6.3f} "
              f"{v.get('minus_rand', 0):+7.3f} {v['gain_reliability_SB'] or 0:6.2f} {v['sd_gain_across_q']:5.3f} {v['mean_within_q_gain_se']:5.3f}")
    print("\nbound:", json.dumps(bound, indent=0))
    for k, P in pred.items():
        print(k, {f: round(x, 2) for f, x in P["spearman"].items()},
              "LOO R2 euc/+var/+eta:", round(P["loo_r2_euclid"], 3), round(P["loo_r2_euclid_var"], 3), round(P["loo_r2_euclid_var_eta"], 3),
              "dR2 CI", [round(x, 3) for x in P["delta_loo_r2_boot_ci"]], "partial rho", round(P["partial_spearman_eta_hat_given_var"], 2),
              "eff eta=0/>0", P["mean_eff_eta_zero"], P["mean_eff_eta_nonzero"], "n>0", P["n_eta_nonzero"])


if __name__ == "__main__":
    main()
