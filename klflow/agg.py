"""Aggregate P2 (and P3) -> results/klflow_summary.json + figures."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/yang0531/scratch/klflow_out"; RES = "/home/yang0531/scratch/steer_live/results"
HJ = json.load(open("/home/yang0531/scratch/steer_live/ot_geometry/heads.json"))
TOP = [tuple(h) for h in HJ["top24"]]; RND = [tuple(h) for h in HJ["random8"]]
COL = {"straight_dmu": "#52514e", "straight_w": "#6250d6", "frozen_grad_s": "#eb6834"}
LAB = {"straight_dmu": "straight Δμ", "straight_w": "straight probe w", "frozen_grad_s": "frozen ∇s"}
for hv, shade in [("cv", 0), ("sm", 1)]:
    for k, ck in zip(["0.25", "1.0", "4.0"], [("#86b6ef", "#3987e5"), ("#2a78d6", "#1c5cab"), ("#104281", "#0d366b")]):
        COL[f"guid_kde_k{k}_{hv}"] = ck[shade]; LAB[f"guid_kde_k{k}_{hv}"] = f"guidance KDE k={k} h_{hv}"
    for k, ck in zip(["0.25", "1.0", "4.0"], [("#7fd6b3", "#5cc79d"), ("#1baf7a", "#199e70"), ("#0b6b4a", "#08563b")]):
        COL[f"guid_dsm_k{k}_{hv}"] = ck[shade]; LAB[f"guid_dsm_k{k}_{hv}"] = f"guidance DSM k={k} σ_{hv}"
    COL[f"particle_kl_{hv}"] = ["#e87ba4", "#d55181"][shade]; LAB[f"particle_kl_{hv}"] = f"particle KL flow h_{hv}"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
rows = {}
for f in glob.glob(f"{OUT}/p2/*.json"):
    r = json.load(open(f)); rows[(r["layer"], r["head"], r["space"])] = r
GUID = [k for k in LAB if k.startswith("guid") or k == "particle_kl"]
S = {}
for space in ["z", "pw32"]:
    for grp, heads in [("top24", TOP), ("random8", RND)]:
        R = [rows[(l, h, space)] for l, h in heads if (l, h, space) in rows]
        if not R:
            continue
        g = dict(n=len(R), n_with_nulls=sum("nulls" in r and len(r["nulls"].get("gshared", [])) > 0 for r in R),
                 h_c=[round(r["h"] / r["h_base"], 2) for r in R])
        for b in ["0.5", "1.0", "2.0"]:
            pb = {}
            for k in LAB:
                ss = [r["paths"][k]["summary"][b] for r in R if k in r["paths"]]
                pb[k] = dict(min_over_start=float(np.median([s["min"] / s["start"] for s in ss])),
                             min_over_floor=float(np.median([s["min"] / r["floor_mmd"][["0.5", "1.0", "2.0"].index(b)] for s, r in zip(ss, R)])),
                             dstar=float(np.median([s["dstar"] for s in ss])), rise=float(np.median([s["rise"] for s in ss])),
                             frac_plateau=float(np.mean([s["plateau"] for s in ss])), frac_overshoot=float(np.mean([s["overshoot"] for s in ss])))
            bi = ["0.5", "1.0", "2.0"].index(b)
            gap = [r["paths"]["straight_dmu"]["summary"][b]["min"] - min(r["paths"][k]["summary"][b]["min"] for k in GUID) for r in R]
            best = [min(GUID, key=lambda k: r["paths"][k]["summary"][b]["min"]) for r in R]
            MATCH = {hv: [f"guid_kde_k1.0_{hv}", f"particle_kl_{hv}"] for hv in ["cv", "sm"]}
            gap_hv = {hv: [r["paths"]["straight_dmu"]["summary"][b]["min"] - min(r["paths"][k]["summary"][b]["min"] for k in GUID if k.endswith(hv)) for r in R] for hv in ["cv", "sm"]}
            gap_m = {hv: [r["paths"]["straight_dmu"]["summary"][b]["min"] - min(r["paths"][k]["summary"][b]["min"] for k in MATCH[hv]) for r in R] for hv in ["cv", "sm"]}
            pb["P2b"] = dict(frac_guid_below_dmu=float(np.mean([x > 0 for x in gap])),
                             frac_guid_below_dmu_by_h={hv: float(np.mean([x > 0 for x in gap_hv[hv]])) for hv in gap_hv},
                             frac_matched_below_dmu={hv: float(np.mean([x > 0 for x in gap_m[hv]])) for hv in gap_m},
                             gap_med={hv: float(np.median(gap_hv[hv])) for hv in gap_hv},
                             best_method_counts={k: best.count(k) for k in set(best)})
            sig = {}
            for kind in ["rshift", "gshared"]:
                sig[kind] = {}
                for hv in ["cv", "sm"]:
                    ok = []
                    for r, gm, gg in zip(R, gap_m[hv], gap_hv[hv]):
                        nl = r.get("nulls", {}).get(kind, [])
                        if not nl:
                            continue
                        ng = [x["straight_dmu"][b]["min"] - min(x[f"guid_kde_k1.0_{hv}"][b]["min"], x[f"particle_kl_{hv}"][b]["min"]) for x in nl]
                        q = np.quantile(ng, .975); ok.append((gg > q, gm > q, len(nl), float(np.median(ng)), gm, gm > 0 and gm > q))
                    if ok:
                        sig[kind][hv] = dict(frac_gap_sig=float(np.mean([o[0] for o in ok])), frac_matched_gap_sig=float(np.mean([o[1] for o in ok])),
                                             n=len(ok), reps=int(np.median([o[2] for o in ok])), null_gap_med=float(np.median([o[3] for o in ok])),
                                             data_matched_gap_med=float(np.median([o[4] for o in ok])))
                if any(r.get("nulls", {}).get(kind) for r in R):
                    sh = {}
                    for k in ["straight_dmu", "frozen_grad_s", "guid_kde_k1.0_cv", "particle_kl_cv", "guid_kde_k1.0_sm", "particle_kl_sm"]:
                        sh[k] = dict(frac_plateau=float(np.mean([np.mean([x[k][b]["plateau"] for x in r["nulls"][kind]]) for r in R if r.get("nulls", {}).get(kind)])),
                                     frac_overshoot=float(np.mean([np.mean([x[k][b]["overshoot"] for x in r["nulls"][kind]]) for r in R if r.get("nulls", {}).get(kind)])),
                                     min_over_start=float(np.median([np.median([x[k][b]["min"] / x[k][b]["start"] for x in r["nulls"][kind]]) for r in R if r.get("nulls", {}).get(kind)])))
                    sig[kind]["shape"] = sh
            pb["P2b"]["nulls"] = sig
            g[b] = pb
        g["s_final"] = {k: float(np.median([r["paths"][k]["curve"]["s"][-1] for r in R])) for k in LAB}
        g["in_all_final"] = {k: float(np.median([r["paths"][k]["curve"]["in_all"][-1] for r in R])) for k in LAB}
        g["disp_final"] = {k: float(np.median([r["paths"][k]["curve"]["disp"][-1] for r in R])) for k in LAB}
        S[f"{space}/{grp}"] = g
if os.path.exists(f"{OUT}/p3.json"):
    P3 = json.load(open(f"{OUT}/p3.json")); S["P3"] = {}
    for grp in ["top", "rnd"]:
        v = [x for x in P3.values() if x["group"] == grp]
        keys = ["cond_grad", "uncond_grad", "cond_w", "uncond_w", "cond_dmu", "uncond_dmu", "dmu_grad", "w_grad", "splithalf_cos"]
        d = {k: float(np.median([x[k] for x in v])) for k in keys}
        d["shuffle_grad_med"] = float(np.median([np.median(x["shuffle_grad"]) for x in v]))
        d["frac_cond_ge_uncond_plus_.1"] = float(np.mean([x["cond_grad"] >= x["uncond_grad"] + 0.1 for x in v]))
        d["frac_cond_gt_shuffle975"] = float(np.mean([x["cond_grad"] > np.quantile(x["shuffle_grad"], .975) for x in v]))
        d["n"] = len(v)
        S["P3"][grp] = d
json.dump(S, open(f"{RES}/klflow_summary.json", "w"), indent=1)

# figure: MMD / s / in-support vs displacement, median over top heads
for space in ["z", "pw32"]:
    R = [rows[(l, h, space)] for l, h in TOP if (l, h, space) in rows]
    if not R:
        continue
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for k in LAB:
        D = np.array([r["paths"][k]["curve"]["disp"] for r in R]); Mm = np.array([[x[1] / r["paths"][k]["curve"]["mmd"][0][1] for x in r["paths"][k]["curve"]["mmd"]] for r in R])
        Sv = np.array([r["paths"][k]["curve"]["s"] for r in R]); Ia = np.array([r["paths"][k]["curve"]["in_all"] for r in R])
        if not all(k in r["paths"] for r in R):
            continue
        if not (k.startswith(("straight", "frozen")) or k.endswith("_sm") or k.startswith("particle")):
            continue
        ls = "-" if k.startswith(("straight", "frozen", "particle")) or "k1.0" in k else ":"
        lw = 2 if ls == "-" else 1.2
        x = np.median(D, 0)
        axs[0].plot(x, np.median(Mm, 0), ls, color=COL[k], lw=lw, label=LAB[k])
        axs[1].plot(x, np.median(Sv, 0), ls, color=COL[k], lw=lw)
        axs[2].plot(x, np.median(Ia, 0), ls, color=COL[k], lw=lw)
    fl = np.median([r["floor_mmd"][1] / r["paths"]["straight_dmu"]["curve"]["mmd"][0][1] for r in R])
    axs[0].axhline(fl, color="#b9b8b2", ls="--", lw=1); axs[0].text(0.2, fl, " floor (fit POS vs held-out POS)", fontsize=7, va="bottom", color="#52514e")
    axs[0].set_ylabel("MMD² to held-out POS / start"); axs[1].set_ylabel("median trait score s (MLP logit)"); axs[2].set_ylabel("fraction inside data band")
    for ax in axs:
        ax.set_xlabel("median displacement / |Δμ|"); ax.set_xlim(0, 15); ax.grid(color="#e4e3df", lw=.5)
    axs[0].set_yscale("log"); axs[0].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.suptitle(f"Frozen velocity vs KL / guidance flows from held-out NEG (top-24 heads, {space}, median)", fontsize=9, x=.01, ha="left")
    fig.tight_layout(); fig.savefig(f"{RES}/klflow_paths_{space}.png"); plt.close(fig)
print(json.dumps(S, indent=1, default=float)[:20000]); print("rows", len(rows))
