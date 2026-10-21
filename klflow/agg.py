"""Aggregate P2 (and P3) -> results/klflow_summary.json + figures."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/yang0531/scratch/klflow_out"; RES = "/home/yang0531/scratch/steer_live/results"
HJ = json.load(open("/home/yang0531/scratch/steer_live/ot_geometry/heads.json"))
TOP = [tuple(h) for h in HJ["top24"]]; RND = [tuple(h) for h in HJ["random8"]]
COL = {"straight_dmu": "#52514e", "straight_w": "#6250d6", "frozen_grad_s": "#eb6834", "guid_kde_k0.25": "#86b6ef",
       "guid_kde_k1.0": "#2a78d6", "guid_kde_k4.0": "#104281", "guid_dsm_k0.25": "#7fd6b3", "guid_dsm_k1.0": "#1baf7a",
       "guid_dsm_k4.0": "#0b6b4a", "particle_kl": "#e87ba4"}
LAB = {"straight_dmu": "straight Δμ", "straight_w": "straight probe w", "frozen_grad_s": "frozen ∇s",
       "guid_kde_k0.25": "guidance KDE k=.25", "guid_kde_k1.0": "guidance KDE k=1", "guid_kde_k4.0": "guidance KDE k=4",
       "guid_dsm_k0.25": "guidance DSM k=.25", "guid_dsm_k1.0": "guidance DSM k=1", "guid_dsm_k4.0": "guidance DSM k=4",
       "particle_kl": "particle KL flow"}
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
            gap_m = [r["paths"]["straight_dmu"]["summary"][b]["min"] - min(r["paths"][k]["summary"][b]["min"] for k in ["guid_kde_k1.0", "particle_kl"]) for r in R]
            pb["P2b"] = dict(frac_guid_below_dmu=float(np.mean([x > 0 for x in gap])), frac_matched_below_dmu=float(np.mean([x > 0 for x in gap_m])),
                             best_method_counts={k: best.count(k) for k in set(best)})
            sig = {}
            for kind in ["rshift", "gshared"]:
                ok = [];
                for r, gm, gg in zip(R, gap_m, gap):
                    nl = r.get("nulls", {}).get(kind, [])
                    if not nl:
                        continue
                    ng = [x["straight_dmu"][b]["min"] - min(x["guid_kde_k1.0"][b]["min"], x["particle_kl"][b]["min"]) for x in nl]
                    q = np.quantile(ng, .975); ok.append((gg > q, gm > q, len(nl), float(np.median(ng)), gm))
                if ok:
                    sig[kind] = dict(frac_gap_sig=float(np.mean([o[0] for o in ok])), frac_matched_gap_sig=float(np.mean([o[1] for o in ok])),
                                     n=len(ok), reps=int(np.median([o[2] for o in ok])), null_gap_med=float(np.median([o[3] for o in ok])),
                                     data_matched_gap_med=float(np.median([o[4] for o in ok])))
                    # null shape: overshoot/plateau of dmu & guidance under null
                    sh = {}
                    for k in ["straight_dmu", "frozen_grad_s", "guid_kde_k1.0", "particle_kl"]:
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
        ls = "-" if k.startswith(("straight", "frozen", "particle")) or k.endswith("1.0") else ":"
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
