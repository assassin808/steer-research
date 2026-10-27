"""Aggregate geom / old / steer outputs -> results/weekday_calib_summary.json + figures."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.expanduser("~/scratch/weekday"); OUT = os.path.expanduser("~/steer_math/results")
MID = range(6, 19)
RULES = {  # stat: (null, direction, extra threshold)
    "id": [("flatG", "lt"), ("blob", "lt")],
    "distortion": [("flatG", "gt"), ("line", "gt"), ("line2", "gt"), ("blob", "gt")],
    "cyc_G": [("shuffle", "gt"), ("line", "gt"), ("line2", "gt")],
    "cyc_E": [("shuffle", "gt"), ("line", "gt"), ("line2", "gt")],
    "turn_corr_med": [("line", "gt"), ("line2", "gt"), ("shuffle", "lt")],
    "turn_naive_med": [("line", "gt"), ("line2", "gt"), ("shuffle", "lt")],
    "turn_corr": [("line", "gt"), ("line2", "gt")],
    "chord2_med": [("line", "lt"), ("line2", "lt")],
    "top1": [("line", "lt"), ("line2", "lt")],
    "field_rot_med": [("line", "gt"), ("line2", "gt"), ("shuffle", "lt")],
    "adj": [("shuffle", "gt"), ("line", "gt"), ("line2", "gt")],
    "wind": [("shuffle", "gt"), ("line", "gt"), ("line2", "gt")],
}


def beats(v, null, d):
    null = np.array([x for x in null if x is not None and np.isfinite(x)])
    if len(null) == 0 or v is None or not np.isfinite(v):
        return None
    return bool(v < null.min()) if d == "lt" else bool(v > null.max())


def geom_summary(fn):
    g = json.load(open(fn)); rows = []
    for L in g["layers"]:
        r = {"layer": L["layer"]}
        for s, rules in RULES.items():
            r[s] = L["data"][s]
            for nm, d in rules:
                nv = L["nulls"][nm].get(s)
                r[f"{s}|{nm}"] = beats(L["data"][s], nv, d) if nv is not None else None
                if nv is not None:
                    r[f"{s}|{nm}_med"] = float(np.nanmedian(np.array(nv, float)))
            for nm in ["blob", "resample"]:
                nv = L["nulls"][nm].get(s)
                if nv is not None:
                    r[f"{s}|{nm}_q"] = [float(np.nanquantile(np.array(nv, float), q)) for q in (.025, .5, .975)]
        rows.append(r)
    mid = [r for r in rows if r["layer"] in MID]
    det = {}
    for s, rules in RULES.items():
        for nm, _ in rules:
            v = [r[f"{s}|{nm}"] for r in mid if r[f"{s}|{nm}"] is not None]
            det[f"{s}|{nm}"] = f"{sum(v)}/{len(v)}"
        det[f"{s}|data_mid_median"] = float(np.median([r[s] for r in mid]))
    return dict(args=g["args"], n=g["n"], rows=rows, mid_detect=det)


def plot_geom(sm, tag):
    rows = sm["rows"]; Ls = [r["layer"] for r in rows]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    for a, (s, nulls, ttl) in zip(ax.ravel(), [
            ("id", ["flatG", "line"], "TwoNN ID"), ("distortion", ["flatG", "line"], "distortion 1-ρ(d_G,d_E)"),
            ("cyc_G", ["shuffle", "line"], "cyclicity of class d_G"), ("turn_corr_med", ["line", "shuffle"], "median corrected turning (°)"),
            ("top1", ["line"], "split-half top-1 share of centroids"), ("field_rot_med", ["line", "shuffle"], "local-field rotation (°)")]):
        a.plot(Ls, [r[s] for r in rows], "k-o", ms=3, label="data")
        for nm, c in zip(nulls, ["tab:blue", "tab:orange"]):
            k = f"{s}|{nm}_med"
            if k in rows[0]:
                a.plot(Ls, [r[k] for r in rows], "--", color=c, label=f"null {nm} (median)")
        if s == "cyc_G":
            a.plot(Ls, [r["cyc_E"] for r in rows], "g-", lw=1, label="data, class d_E")
        a.axvspan(min(MID), max(MID), color="0.93", zorder=-1); a.set_title(ttl); a.set_xlabel("layer"); a.legend(fontsize=7)
    fig.suptitle(tag); fig.tight_layout(); fig.savefig(f"{OUT}/weekday_calib_geom_{tag}.png", dpi=110); plt.close(fig)


def steer_summary(fn, C):
    s = json.load(open(fn)); out = {}
    for key, v in s["paths"].items():
        Lk, pr, pn = key.split("_"); i0, i1 = map(int, pr.split("-"))
        P = np.array(v["pc"]); oth = np.array(v["other"])
        mids = list(range(i0 + 1, i1))
        d = out.setdefault(f"{Lk}_{pr}", {})
        d[pn] = dict(bump=[float(P[:, m].max()) for m in mids], p_mid_at_half=[float(P[len(P) // 2, m]) for m in mids],
                     p_tgt_end=float(P[-1, i1]), p_src_start=float(P[0, i0]), other_max=float(oth.max()), other_mean=float(oth.mean()),
                     argmax_seq=[int(np.argmax(p)) for p in P])
    return dict(layers=s["layers"], sanity=s.get("sanity_t0_maxdiff"), n_prompts=s["n_prompts"], cells=out)


def plot_steer(fn, names, tag):
    s = json.load(open(fn)); t = np.array(s["t"]); Ls = s["layers"]
    pairs = sorted({k.split("_")[1] for k in s["paths"]})
    fig, ax = plt.subplots(len(pairs), len(Ls), figsize=(2.3 * len(Ls), 2.4 * len(pairs)), squeeze=False)
    for pi, pr in enumerate(pairs):
        i0, i1 = map(int, pr.split("-"))
        for li, l in enumerate(Ls):
            a = ax[pi, li]
            for pn, ls in [("straight", "-"), ("spline", ":")]:
                v = s["paths"][f"L{l}_{pr}_{pn}"]; P = np.array(v["pc"])
                for c, col in zip(range(i0, i1 + 1), ["k", "tab:red", "tab:green", "tab:blue"]):
                    a.plot(t, P[:, c], ls, color=col, lw=1.2, label=f"{names[c][:3]} {pn}" if li == 0 else None)
                a.plot(t, v["other"], ls, color="0.6", lw=1, label=f"off-class {pn}" if li == 0 else None)
            a.set_title(f"L{l} {names[i0][:3]}→{names[i1][:3]}", fontsize=8); a.set_ylim(0, 1)
        ax[pi, 0].legend(fontsize=5)
    fig.suptitle(f"{tag}: straight (solid) vs spline (dotted) steering"); fig.tight_layout()
    fig.savefig(f"{OUT}/weekday_calib_steer_{tag}.png", dpi=100); plt.close(fig)


if __name__ == "__main__":
    S = {}
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    for fn in sorted(os.listdir(R)):
        if fn.startswith("geom_") and fn.endswith(".json"):
            sm = geom_summary(f"{R}/{fn}"); tag = fn[5:-5]; S[f"geom_{tag}"] = {k: v for k, v in sm.items()}
            plot_geom(sm, tag)
        if fn.endswith("_steer.json"):
            dom = "weekday" if "weekday" in fn else "month"
            S[fn[:-5]] = steer_summary(f"{R}/{fn}", 7 if dom == "weekday" else 12)
            plot_steer(f"{R}/{fn}", DAYS if dom == "weekday" else MONTHS, fn[:-11])
        if fn.startswith("old_") and fn.endswith(".json"):
            S[fn[:-5]] = json.load(open(f"{R}/{fn}"))
    json.dump(S, open(f"{OUT}/weekday_calib_summary.json", "w"), indent=0)
    for k, v in S.items():
        if k.startswith("geom_"):
            print(k, json.dumps(v["mid_detect"]))
