"""Aggregate per-head OT-geometry JSONs -> results/ot_geometry_summary.json + figures."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/yang0531/scratch/ot_geom/out"
RES = "/home/yang0531/steer_math/results"
HJ = json.load(open("/home/yang0531/steer_math/ot_geometry/heads.json"))
TOP = [tuple(h) for h in HJ["top24"]]; RND = [tuple(h) for h in HJ["random8"]]
EPS = ["0.003", "0.01", "0.03", "0.1"]; PRIM = "0.01"
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", violet="#6250d6",
         ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", null="#b9b8b2")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
                     "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

rows = {}
for f in glob.glob(f"{OUT}/*.json"):
    r = json.load(open(f)); rows[(r["layer"], r["head"], r["space"])] = r


def q(v, p): return float(np.quantile(v, p))


def classify(e, stat="r2s"):
    lo = lambda k: q(e["nulls"][k][stat], 0.025)
    x = e[stat]
    flat = x >= lo("gshared") and x >= lo("rshift")
    curved = x < lo("gshared") and x < lo("rshift")
    affine = curved and x >= lo("gsep")
    return "flat" if flat else ("affine" if affine else ("curved" if curved else "mixed"))


S = {}
for space in ["z", "raw", "pw8", "pw32"]:
    sp = {}
    for grp, heads in [("top24", TOP), ("random8", RND)]:
        hs = [h for h in heads if (h[0], h[1], space) in rows]
        if not hs:
            continue
        g = {"n_heads": len(hs)}
        for eps in EPS:
            cls = [classify(rows[(l, h, space)]["flat"][eps]) for l, h in hs]
            cls_raw = [classify(rows[(l, h, space)]["flat"][eps], "r2") for l, h in hs]
            g[eps] = dict(
                frac_flat=np.mean([c == "flat" for c in cls]),
                frac_curved_any=np.mean([c in ("curved", "affine") for c in cls]),
                frac_curved_beyond_affine=np.mean([c == "curved" for c in cls]),
                frac_affine=np.mean([c == "affine" for c in cls]),
                frac_flat_rawR2=np.mean([c == "flat" for c in cls_raw]),
                med_r2s=float(np.median([rows[(l, h, space)]["flat"][eps]["r2s"] for l, h in hs])),
                med_r2s_rshift=float(np.median([np.median(rows[(l, h, space)]["flat"][eps]["nulls"]["rshift"]["r2s"]) for l, h in hs])),
                med_r2s_gshared=float(np.median([np.median(rows[(l, h, space)]["flat"][eps]["nulls"]["gshared"]["r2s"]) for l, h in hs])),
                med_r2s_gsep=float(np.median([np.median(rows[(l, h, space)]["flat"][eps]["nulls"]["gsep"]["r2s"]) for l, h in hs])),
                med_r2s_perm=float(np.median([np.median(rows[(l, h, space)]["flat"][eps]["nulls"]["perm"]["r2s"]) for l, h in hs])),
                med_r2=float(np.median([rows[(l, h, space)]["flat"][eps]["r2"] for l, h in hs])),
                med_r2_rshift=float(np.median([np.median(rows[(l, h, space)]["flat"][eps]["nulls"]["rshift"]["r2"]) for l, h in hs])),
                med_coss=float(np.median([rows[(l, h, space)]["flat"][eps]["coss"] for l, h in hs])),
                med_coss_rshift=float(np.median([np.median(rows[(l, h, space)]["flat"][eps]["nulls"]["rshift"]["coss"]) for l, h in hs])),
                med_neff=float(np.median([rows[(l, h, space)]["flat"][eps]["neff"] for l, h in hs])),
                classes=dict(zip([f"L{l}H{h}" for l, h in hs], cls)))
        geo = [rows[(l, h, space)]["geo"] for l, h in hs]
        g["geo"] = dict(med_ratio=float(np.median([x["median_ratio"] for x in geo])),
                        med_null_gshared=float(np.median([np.median(x["nulls"]["gshared"]) for x in geo])),
                        med_null_rshift=float(np.median([np.median(x["nulls"]["rshift"]) for x in geo])),
                        frac_curved=float(np.mean([x["median_ratio"] > max(q(x["nulls"]["gshared"], .975), q(x["nulls"]["rshift"], .975)) for x in geo])),
                        disconnected=int(sum(x["disconnected"] for x in geo)))
        pk = ["ideea_npn_nc2", "ideea_npn_nc4", "ideea_minperp_nc2", "meandiff", "knn20"]
        P = {k: [rows[(l, h, space)]["pred"][k]["cos"] for l, h in hs] for k in pk}
        g["pred"] = {k: dict(mean=float(np.mean(v)), median=float(np.median(v))) for k, v in P.items()}
        g["pred"]["frac_knn_beats_md_by_.05"] = float(np.mean(np.array(P["knn20"]) - np.array(P["meandiff"]) >= 0.05))
        g["pred"]["frac_ideea_npn2_beats_md_by_.05"] = float(np.mean(np.array(P["ideea_npn_nc2"]) - np.array(P["meandiff"]) >= 0.05))
        g["pred"]["frac_ideea_minperp2_beats_md_by_.05"] = float(np.mean(np.array(P["ideea_minperp_nc2"]) - np.array(P["meandiff"]) >= 0.05))
        g["pred"]["ceiling_rshift"] = {k: float(np.mean([np.mean(rows[(l, h, space)]["pred_ceiling_rshift"][k]) for l, h in hs])) for k in ["meandiff", "knn20"]}
        bds = [rows[(l, h, space)]["boundary"] for l, h in hs]
        g["boundary"] = dict(
            med_texit_ot=float(np.nanmedian([b["ot"]["median_texit"] if b["ot"]["median_texit"] is not None else np.nan for b in bds])),
            med_texit_md=float(np.nanmedian([b["meandiff"]["median_texit"] if b["meandiff"]["median_texit"] is not None else np.nan for b in bds])),
            frac_inside_t0_md=float(np.median([b["meandiff"]["frac_inside"][0] for b in bds])),
            frac_inside_t1_md=float(np.median([b["meandiff"]["frac_inside"][10] for b in bds])),
            frac_inside_t1_ot=float(np.median([b["ot"]["frac_inside"][10] for b in bds])),
            t_equiv={a: float(np.median([b["t_equiv"][a] for b in bds])) for a in bds[0]["t_equiv"]},
            t_equiv_range={a: [float(np.min([b["t_equiv"][a] for b in bds])), float(np.max([b["t_equiv"][a] for b in bds]))] for a in bds[0]["t_equiv"]})
        sp[grp] = g
    S[space] = sp
json.dump(S, open(f"{RES}/ot_geometry_summary.json", "w"), indent=1, default=float)

# ------------------------------------------------ figures
def fig_flat(space, eps=PRIM, stat="r2s"):
    hs = [h for h in TOP + RND if (h[0], h[1], space) in rows]
    if not hs:
        return
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, (l, h) in enumerate(hs):
        e = rows[(l, h, space)]["flat"][eps]
        for k, col, off in [("gshared", C["null"], -0.27), ("rshift", C["ink2"], -0.09), ("gsep", C["yellow"], 0.09), ("perm", C["magenta"], 0.27)]:
            v = e["nulls"][k][stat]
            ax.plot([i + off] * 2, [q(v, .025), q(v, .975)], color=col, lw=2.2, solid_capstyle="round",
                    label={"gshared": "null: Gaussian, shared cov", "rshift": "null: residual shift (data shape)",
                           "gsep": "null: Gaussian, separate cov (affine)", "perm": "null: label permutation"}[k] if i == 0 else None)
        ax.plot(i, e[stat], "o", ms=6, color=C["blue"], mec="white", mew=1, zorder=5, label="data" if i == 0 else None)
    ax.axvline(len([h for h in hs if h in TOP]) - 0.5, color=C["grid"], lw=1)
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([f"L{l}H{h}" for l, h in hs], rotation=90, fontsize=7)
    ax.set_ylabel({"r2s": "flatness R²(kNN-smoothed field)", "r2": "flatness R² (raw field)"}[stat])
    ax.set_title(f"Is the NEG→POS transport a constant shift?  space={space}, ε={eps}·median cost  (left: top-24 IDEEA heads | right: 8 random)", fontsize=9, loc="left")
    ax.legend(fontsize=7, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    fig.tight_layout(); fig.savefig(f"{RES}/ot_geometry_flatness_{space}_{stat}.png"); plt.close(fig)


def fig_pred(space):
    hs = [h for h in TOP + RND if (h[0], h[1], space) in rows]
    if not hs:
        return
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for k, col, lab in [("meandiff", C["ink2"], "global mean difference"), ("ideea_npn_nc2", C["orange"], "IDEEA nearest_pos_neg nc=2"),
                        ("ideea_minperp_nc2", C["yellow"], "IDEEA min_perp nc=2"), ("knn20", C["blue"], "kNN-local average of d (k=20)")]:
        ax.plot(range(len(hs)), [rows[(l, h, space)]["pred"][k]["cos"] for l, h in hs], "o-", ms=4, lw=1.2, color=col, label=lab)
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([f"L{l}H{h}" for l, h in hs], rotation=90, fontsize=7)
    ax.axvline(len([h for h in hs if h in TOP]) - 0.5, color=C["grid"], lw=1)
    ax.set_ylabel("mean cos(pred, OT displacement)\non held-out NEG")
    ax.set_title(f"Which predictor recovers the OT displacement on held-out questions?  space={space}", fontsize=9, loc="left")
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.38)); ax.grid(axis="y", color=C["grid"], lw=0.6)
    fig.tight_layout(); fig.savefig(f"{RES}/ot_geometry_predictors_{space}.png"); plt.close(fig)


def fig_boundary(space):
    hs = [h for h in TOP if (h[0], h[1], space) in rows]
    if not hs:
        return
    ts = rows[(hs[0][0], hs[0][1], space)]["boundary"]["ts"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, col, lab in [("meandiff", C["ink2"], "step along μ+−μ−"), ("ot", C["blue"], "step along own OT displacement")]:
        F = np.array([rows[(l, h, space)]["boundary"][name]["frac_inside"] for l, h in hs])
        R = np.array([rows[(l, h, space)]["boundary"][name]["median_dk_over_band"] for l, h in hs])
        axs[0].plot(ts, np.median(F, 0), color=col, lw=2, label=lab)
        axs[0].fill_between(ts, np.quantile(F, .25, 0), np.quantile(F, .75, 0), color=col, alpha=.15, lw=0)
        axs[1].plot(ts, np.median(R, 0), color=col, lw=2, label=lab)
    te = {a: np.median([rows[(l, h, space)]["boundary"]["t_equiv"][a] for l, h in hs]) for a in ["1", "4"]}
    for a, v in te.items():
        if v <= 3:
            for ax in axs:
                ax.axvline(v, color=C["orange"], lw=1, ls="--")
            axs[0].text(v, 0.02, f" IDEEA α={a}", color=C["ink2"], fontsize=7)
    axs[0].set_xlabel("t  (x + t·d)"); axs[0].set_ylabel("fraction of held-out NEG\ninside POS 95% kNN band")
    axs[1].set_xlabel("t"); axs[1].set_ylabel("median 10-NN distance to POS / band q95")
    axs[1].axhline(1, color=C["grid"], lw=1)
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Support boundary along the steering path, top-24 heads, space={space} (band = median, IQR shaded)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(f"{RES}/ot_geometry_boundary_{space}.png"); plt.close(fig)


for space in ["z", "raw", "pw8", "pw32"]:
    fig_flat(space); fig_flat(space, stat="r2"); fig_pred(space); fig_boundary(space)
print(json.dumps({sp: {g: {k: v for k, v in d.items() if k in (PRIM, "geo", "pred", "boundary")} for g, d in S[sp].items()} for sp in S}, indent=1, default=float)[:12000])
print("n rows", len(rows))
