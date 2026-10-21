"""Aggregate trait-flow JSONs -> results/trait_flow_summary.json + figures."""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/yang0531/scratch/ot_geom/tf"; RES = "/home/yang0531/steer_math/results"
HJ = json.load(open("/home/yang0531/steer_math/ot_geometry/heads.json"))
TOP = [tuple(h) for h in HJ["top24"]]; RND = [tuple(h) for h in HJ["random8"]]
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", violet="#6250d6",
         ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", null="#b9b8b2")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.edgecolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"]})
rows = {}
for f in glob.glob(f"{OUT}/*.json"):
    r = json.load(open(f)); rows[(r["layer"], r["head"], r["space"])] = r
q = lambda v, p: float(np.quantile(v, p))
med = lambda v: float(np.median([x for x in v if x is not None and np.isfinite(x)])) if any(x is not None and np.isfinite(x) for x in v) else None

S = {}
for space in ["z", "pw32"]:
    S[space] = {}
    for grp, heads in [("top24", TOP), ("random8", RND)]:
        hs = [h for h in heads if (h[0], h[1], space) in rows]
        if not hs:
            continue
        R = [rows[(l, h, space)] for l, h in hs]
        g = dict(n=len(hs))
        sc = [r["score"] for r in R]
        g["score"] = dict(auc_lr=med([s["auc_lr"] for s in sc]), auc_mlp=med([s["auc_mlp"] for s in sc]),
                          dll_med=med([s["dll"] for s in sc]),
                          frac_mlp_better_ci=float(np.mean([s["dll_ci"][0] > 0 for s in sc])),
                          frac_mlp_worse_ci=float(np.mean([s["dll_ci"][1] < 0 for s in sc])),
                          alpha=[s["alpha"] for s in sc], C=[s["C"] for s in sc])
        for fl in ["unproj", "proj"]:
            cl = []
            for r in R:
                t = r["flow"][fl]["turn"]["1.0"]
                nul = [x[fl]["turn"]["1.0"] for x in r["nulls"]["gshared"]]
                cl.append(t > q(nul, .975))
            g[f"turn_{fl}"] = dict(
                data={c: med([r["flow"][fl]["turn"][c] for r in R]) for c in ["0.5", "1.0", "2.0"]},
                gshared={c: med([np.median([x[fl]["turn"][c] for x in r["nulls"]["gshared"]]) for r in R]) for c in ["0.5", "1.0", "2.0"]},
                perm={c: med([np.median([x["unproj"]["turn"][c] for x in r["nulls"]["perm"]]) for r in R]) for c in ["0.5", "1.0", "2.0"]} if fl == "unproj" else None,
                rshift={c: med([np.median([x["unproj"]["turn"][c] for x in r["nulls"]["rshift"]]) for r in R]) for c in ["0.5", "1.0", "2.0"]} if fl == "unproj" else None,
                curv=med([r["flow"][fl]["curv"] for r in R]),
                curv_gshared=med([np.median([x[fl]["curv"] for x in r["nulls"]["gshared"]]) for r in R]),
                frac_curved=float(np.mean(cl)), classes=dict(zip([f"L{l}H{h}" for l, h in hs], ["curved" if c else "flat" for c in cl])))
        g["turn_linear_proj"] = med([r["flow"]["linear_proj"]["turn"]["1.0"] for r in R])
        g["turn_linear_unproj"] = med([r["flow"]["linear_unproj"]["turn"]["1.0"] for r in R])
        g["agree"] = {v: {k: med([r["agree"][v][k] for r in R]) for k in R[0]["agree"][v]} for v in ["V_unproj", "V_proj"]}
        g["agree"]["ot_vs_meandiff"] = med([r["agree"]["ot_vs_meandiff"] for r in R])
        g["agree"]["ideea_vs_meandiff"] = med([r["agree"]["ideea_vs_meandiff"] for r in R])
        g["agree"]["lrw_vs_meandiff"] = med([r["agree"]["lrw_vs_meandiff"] for r in R])
        g["agree"]["gshared_V_unproj"] = {k: med([np.median([x["agree_V_unproj"][k] for x in r["nulls"]["gshared"]]) for r in R]) for k in ["lr_w", "meandiff", "self_mean"]}
        g["agree"]["perm_V_unproj_self"] = med([np.median([x["agree_V_unproj"]["self_mean"] for x in r["nulls"]["perm"]]) for r in R])
        b = {}
        for fl in ["unproj", "proj", "straight", "linear_unproj", "linear_proj"]:
            b[fl] = dict(exit=med([r["flow"][fl]["exit_med"] for r in R]), sat=med([r["flow"][fl]["sat_med"] for r in R]),
                         never_inside=med([r["flow"][fl]["frac_never_inside"] for r in R]),
                         never_exit=med([r["flow"][fl]["frac_never_exit"] for r in R]),
                         inside_t0=med([r["flow"][fl]["frac_inside_t0"] for r in R]),
                         s_start=med([r["flow"][fl]["s_start"] for r in R]), s_end=med([r["flow"][fl]["s_end"] for r in R]))
            if fl in ("unproj", "proj"):
                b[fl]["vanish"] = med([r["flow"][fl]["vanish_med"] for r in R])
                b[fl]["s_at_exit"] = med([r["flow"][fl].get("s_at_exit") for r in R])
        ex = lambda r, k: r["flow"][k]["exit_med"] if r["flow"][k]["exit_med"] is not None else np.nan
        b["frac_straight_exits_before_proj"] = float(np.mean([ex(r, "straight") < ex(r, "proj") for r in R]))
        b["frac_straight_exits_before_unproj"] = float(np.mean([ex(r, "straight") < ex(r, "unproj") for r in R]))
        b["per_head_exit"] = {f"L{r['layer']}H{r['head']}": {k: r["flow"][k]["exit_med"] for k in ["straight", "unproj", "proj"]} for r in R}
        g["boundary"] = b
        S[space][grp] = g
json.dump(S, open(f"{RES}/trait_flow_summary.json", "w"), indent=1, default=float)


def fig_turn(space):
    hs = [h for h in TOP + RND if (h[0], h[1], space) in rows]
    if not hs:
        return
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    for ax, fl, ttl in [(axs[0], "unproj", "unprojected flow ∇s/|∇s|"), (axs[1], "proj", "tangent-projected flow P_T∇s/|P_T∇s|")]:
        for i, (l, h) in enumerate(hs):
            r = rows[(l, h, space)]
            g = [x[fl]["turn"]["1.0"] for x in r["nulls"]["gshared"]]
            ax.plot([i - .15] * 2, [q(g, .025), q(g, .975)], color=C["null"], lw=2.4, solid_capstyle="round", label="null: Gaussian equal cov (field constant)" if i == 0 else None)
            if fl == "unproj":
                p = [x["unproj"]["turn"]["1.0"] for x in r["nulls"]["perm"]]
                ax.plot([i + .15] * 2, [q(p, .025), q(p, .975)], color=C["magenta"], lw=2.4, solid_capstyle="round", label="null: label permutation" if i == 0 else None)
                rs = [x["unproj"]["turn"]["1.0"] for x in r["nulls"]["rshift"]]
                ax.plot([i] * 2, [q(rs, .025), q(rs, .975)], color=C["yellow"], lw=2.4, solid_capstyle="round", label="null: data-shape translation" if i == 0 else None)
            ax.plot(i, r["flow"][fl]["turn"]["1.0"], "o", color=C["blue"], ms=6, mec="white", mew=1, zorder=5, label="data (MLP s)" if i == 0 else None)
            if fl == "proj":
                ax.plot(i, r["flow"]["linear_proj"]["turn"]["1.0"], "D", color=C["ink2"], ms=4, zorder=5, label="data, linear s (tangent turning only)" if i == 0 else None)
        ax.axvline(len([h for h in hs if h in TOP]) - .5, color=C["grid"], lw=1)
        ax.set_ylabel("turning angle (deg)\nafter arc length |Δμ|"); ax.set_title(ttl, fontsize=9, loc="left")
        ax.grid(axis="y", color=C["grid"], lw=.6); ax.legend(fontsize=7, frameon=False, ncol=2)
    axs[1].set_xticks(range(len(hs))); axs[1].set_xticklabels([f"L{l}H{h}" for l, h in hs], rotation=90, fontsize=7)
    fig.suptitle(f"Do trait-flow lines turn?  space={space}  (left: top-24 IDEEA heads | right: 8 random)", fontsize=9, x=.01, ha="left")
    fig.tight_layout(); fig.savefig(f"{RES}/trait_flow_turning_{space}.png"); plt.close(fig)


def fig_agree(space):
    hs = [h for h in TOP + RND if (h[0], h[1], space) in rows]
    if not hs:
        return
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for k, col, lab in [("lr_w", C["violet"], "linear-probe weight"), ("meandiff", C["ink2"], "mean difference"),
                        ("ideea_npn_nc2", C["orange"], "IDEEA nearest_pos_neg nc=2"), ("ot", C["aqua"], "OT displacement")]:
        ax.plot(range(len(hs)), [rows[(l, h, space)]["agree"]["V_unproj"][k] for l, h in hs], "o-", ms=4, lw=1.2, color=col, label=lab)
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([f"L{l}H{h}" for l, h in hs], rotation=90, fontsize=7)
    ax.axvline(len([h for h in hs if h in TOP]) - .5, color=C["grid"], lw=1)
    ax.set_ylabel("mean cos with V(x)\n(held-out NEG)"); ax.grid(axis="y", color=C["grid"], lw=.6)
    ax.set_title(f"Which estimator points along the local trait flow?  space={space}, unprojected V", fontsize=9, loc="left")
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.38))
    fig.tight_layout(); fig.savefig(f"{RES}/trait_flow_agreement_{space}.png"); plt.close(fig)


def fig_boundary(space):
    hs = [h for h in TOP if (h[0], h[1], space) in rows]
    if not hs:
        return
    r0 = rows[(hs[0][0], hs[0][1], space)]
    ell = np.arange(r0["nsteps"] + 1)[::5] * r0["h"] / r0["dmu_norm"]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.3))
    for fl, col, lab in [("straight", C["ink2"], "straight line along Δμ"), ("unproj", C["orange"], "trait flow, unprojected"), ("proj", C["blue"], "trait flow, tangent-projected")]:
        F = np.array([rows[(l, h, space)]["flow"][fl]["frac_inside_curve"] for l, h in hs])
        D = np.array([rows[(l, h, space)]["flow"][fl]["dk_curve"] for l, h in hs])
        axs[0].plot(ell, np.median(F, 0), color=col, lw=2, label=lab)
        axs[0].fill_between(ell, np.quantile(F, .25, 0), np.quantile(F, .75, 0), color=col, alpha=.12, lw=0)
        axs[1].plot(ell, np.median(D, 0), color=col, lw=2, label=lab)
    axs[0].set_xlabel("arc length / |Δμ|"); axs[0].set_ylabel("fraction of held-out NEG starts\ninside POS 95% kNN band")
    axs[1].set_xlabel("arc length / |Δμ|"); axs[1].set_ylabel("median 10-NN dist to POS / band"); axs[1].axhline(1, color=C["grid"])
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Support boundary along trait-flow lines vs straight extrapolation, top-24 heads, space={space}", fontsize=9, x=.01, ha="left")
    fig.tight_layout(); fig.savefig(f"{RES}/trait_flow_boundary_{space}.png"); plt.close(fig)


for sp in ["z", "pw32"]:
    fig_turn(sp); fig_agree(sp); fig_boundary(sp)
print(json.dumps({sp: {g: {k: v for k, v in d.items() if k != "boundary" or True} for g, d in S[sp].items()} for sp in S}, indent=1, default=float)[:15000])
print("rows", len(rows))
