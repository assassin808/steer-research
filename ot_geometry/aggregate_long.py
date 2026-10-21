"""Aggregate POST-HOC long-horizon boundary + alpha-sensitivity (deviation D2)."""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/yang0531/scratch/ot_geom/tflong"; RES = "/home/yang0531/steer_math/results"
HJ = json.load(open("/home/yang0531/steer_math/ot_geometry/heads.json"))
TOP = [tuple(h) for h in HJ["top24"]]; RND = [tuple(h) for h in HJ["random8"]]
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", violet="#6250d6", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
rows = {}
for f in glob.glob(f"{OUT}/*.json"):
    r = json.load(open(f)); rows[(r["layer"], r["head"], r["space"], r["agrid"])] = r


def fin(v):
    v = [x for x in v if x is not None]
    return v


S = {}
for space in ["z", "pw32"]:
    for grp, heads in [("top24", TOP), ("random8", RND)]:
        R = [rows[(l, h, space, "prereg")] for l, h in heads if (l, h, space, "prereg") in rows]
        if not R:
            continue
        g = dict(n=len(R))
        for k in ["straight", "straight_lrw", "unproj", "proj", "linear_proj"]:
            ex = [r[k]["exit_med"] if r[k]["exit_med"] is not None else np.nan for r in R]
            g[k] = dict(exit_med=float(np.nanmedian(ex)) if np.isfinite(np.nanmedian(ex)) or np.isinf(np.nanmedian(ex)) else None,
                        never_exit=float(np.median([r[k]["frac_never_exit"] for r in R])),
                        never_inside=float(np.median([r[k]["frac_never_inside"] for r in R])),
                        sat_med=float(np.median([r[k]["sat_med"] for r in R])),
                        s_curve=np.median([r[k]["s"] for r in R], 0).tolist(),
                        frac_inside_curve=np.median([r[k]["frac_inside"] for r in R], 0).tolist())
            if k in ("unproj", "proj"):
                g[k]["vanish_med"] = float(np.median([r[k]["vanish_med"] for r in R]))
                g[k]["fnorm_curve"] = np.median([r[k]["fnorm"] for r in R], 0).tolist()
                g[k]["turn"] = {c: float(np.median([r[k]["turn"][c] for r in R])) for c in R[0][k]["turn"]}
        ex = lambda r, k: r[k]["exit_med"] if r[k]["exit_med"] is not None else np.nan
        g["frac_straight_before_proj"] = float(np.mean([ex(r, "straight") < ex(r, "proj") for r in R]))
        g["frac_straight_before_unproj"] = float(np.mean([ex(r, "straight") < ex(r, "unproj") for r in R]))
        g["frac_straight_before_linproj"] = float(np.mean([ex(r, "straight") < ex(r, "linear_proj") for r in R]))
        # paired per-point comparison: straight vs projected, pooled over heads
        d = []
        for r in R:
            a, b = np.array(r["straight"]["exits"], float), np.array(r["proj"]["exits"], float)
            ok = ~np.isnan(a) & ~np.isnan(b)
            d.append(np.mean(a[ok] < b[ok]) if ok.any() else np.nan)
        g["perpoint_frac_straight_before_proj_med"] = float(np.nanmedian(d))
        g["t_equiv"] = {a: float(np.median([r["t_equiv"][a] for r in R])) for a in R[0]["t_equiv"]}
        g["t_equiv_range"] = {a: [float(np.min([r["t_equiv"][a] for r in R])), float(np.max([r["t_equiv"][a] for r in R]))] for a in R[0]["t_equiv"]}
        g["frac_heads_ta4_before_straight_exit"] = float(np.mean([r["t_equiv"]["4"] < (r["straight"]["exit_med"] or np.inf) for r in R]))
        g["frac_heads_ta15_after_straight_exit"] = float(np.mean([r["t_equiv"]["15"] > (r["straight"]["exit_med"] or np.inf) for r in R]))
        g["frac_heads_sat_before_exit_unproj"] = float(np.mean([r["unproj"]["sat_med"] < (r["unproj"]["exit_med"] or np.inf) for r in R]))
        g["per_head"] = {f"L{r['layer']}H{r['head']}": {k: r[k]["exit_med"] for k in ["straight", "unproj", "proj", "linear_proj"]} | {"t_a4": r["t_equiv"]["4"], "t_a15": r["t_equiv"]["15"]} for r in R}
        S[f"{space}/{grp}"] = g
    for grp, heads in [("top24", TOP), ("random8", RND)]:
        R = [rows[(l, h, space, "ext")] for l, h in heads if (l, h, space, "ext") in rows]
        if not R:
            continue
        cls = {}
        for fl in ["unproj", "proj"]:
            cl = [r[fl]["turn"]["1"] > np.quantile([x[fl]["1"] for x in r["null_gshared"]], .975) for r in R]
            cls[fl] = dict(frac_curved=float(np.mean(cl)), data=float(np.median([r[fl]["turn"]["1"] for r in R])),
                           null=float(np.median([np.median([x[fl]["1"] for x in r["null_gshared"]]) for r in R])),
                           null_q975_med=float(np.median([np.quantile([x[fl]["1"] for x in r["null_gshared"]], .975) for r in R])),
                           data3=float(np.median([r[fl]["turn"]["3"] for r in R])),
                           null3=float(np.median([np.median([x[fl]["3"] for x in r["null_gshared"]]) for r in R])),
                           frac_curved3=float(np.mean([r[fl]["turn"]["3"] > np.quantile([x[fl]["3"] for x in r["null_gshared"]], .975) for r in R])))
        S[f"{space}/{grp}/alpha_ext"] = dict(n=len(R), alpha=[r["alpha"] for r in R], **cls)
json.dump(S, open(f"{RES}/trait_flow_long_summary.json", "w"), indent=1, default=float)

# figure: boundary along long paths
for space in ["z", "pw32"]:
    R = [rows[(l, h, space, "prereg")] for l, h in TOP if (l, h, space, "prereg") in rows]
    if not R:
        continue
    grid = R[0]["grid"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for k, col, lab in [("straight", C["ink2"], "straight along Δμ"), ("straight_lrw", C["violet"], "straight along probe w"),
                        ("unproj", C["orange"], "trait flow, unprojected"), ("proj", C["blue"], "trait flow, tangent-projected")]:
        F = np.array([r[k]["frac_inside"] for r in R]); D = np.array([r[k]["dk_over_band"] for r in R]); Sv = np.array([r[k]["s"] for r in R])
        axs[0].plot(grid, np.median(F, 0), color=col, lw=2, label=lab)
        axs[1].plot(grid, np.median(D, 0), color=col, lw=2, label=lab)
        axs[2].plot(grid, np.median(Sv, 0), color=col, lw=2, label=lab)
    te4 = np.median([r["t_equiv"]["4"] for r in R]); te15 = np.median([r["t_equiv"]["15"] for r in R])
    for ax in axs:
        for v, t in [(te4, "IDEEA α=4"), (te15, "α=15")]:
            ax.axvline(v, color=C["yellow"], lw=1, ls="--")
        ax.set_xlabel("arc length / |Δμ|"); ax.grid(color=C["grid"], lw=.5)
    axs[0].text(te4, 0.03, " α=4", fontsize=7, color=C["ink2"]); axs[0].text(te15, 0.03, " α=15", fontsize=7, color=C["ink2"])
    axs[0].set_ylabel("fraction inside POS 95% kNN band"); axs[1].set_ylabel("median 10-NN dist to POS / band")
    axs[1].axhline(1, color=C["ink2"], lw=.8); axs[2].set_ylabel("median trait score s (MLP logit)")
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Leaving the POS support: flow lines vs straight extrapolation (top-24 heads, {space}; post hoc horizon)", fontsize=9, x=.01, ha="left")
    fig.tight_layout(); fig.savefig(f"{RES}/trait_flow_boundary_long_{space}.png"); plt.close(fig)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("per_head",)} for k, v in S.items()}, indent=1, default=float)[:14000])
print("rows", len(rows))
