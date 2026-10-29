"""Geometry analysis H0-H4 + trivial controls (prereg sections 3-5). CPU.
usage: RUN=main python analyze_geom.py [--reps 99] [--nfield 20] [--split train|test] [--layers primary|all] [--reps_list neu,txt,res,len,sys]
Representations: neu (acts_neu, PRIMARY), sys (acts_sys), txt (sentence-encoder embedding of the response text; surface control 1),
res (acts_neu minus a cross-fitted ridge prediction from the text embedding; surface control 2), len (acts_neu with log ntok
regressed out of every coordinate; length control C4).
Primary layer per trait = l*_t from <OUT>/layers.json (steering sweep); if absent, every layer in LPROFILE is 'primary'.
Output: <OUT>/geom_<rep>_<split>.json, <OUT>/prompt_profile.json (C1 + H0 + wording rule), <OUT>/length.json (C4).
"""
import os, argparse, numpy as np
from scipy.stats import spearmanr
from common import OUT, TRAITS, NEIGHBOUR_PAIRS, jload, jdump, log, cells
import geom as G
from paths import labels, fit_path, path_curvature

ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=99); ap.add_argument("--nfield", type=int, default=20)
ap.add_argument("--reps_list", default="neu,txt,res,len,sys"); ap.add_argument("--layers", default="primary"); ap.add_argument("--split", default="train")
args = ap.parse_known_args()[0]
NSIM_POWER = int(os.environ.get("NSIM_POWER", 100))
LPROFILE = list(range(2, 26, 2))
STATS = ("offline", "quadperp", "turn")


def load_meta():
    rows = jload(f"{OUT}/judged_gen_main.json")
    meta = {k: np.array([r[k] for r in rows]) for k in ["trait", "level", "sample", "split", "src", "qid", "ntok", "coh"]}
    qidx = {v: i for i, v in enumerate(sorted(set(meta["qid"])))}; meta["q"] = np.array([qidx[v] for v in meta["qid"]])
    for t in TRAITS: meta[f"P_{t}"] = np.array([r[f"P_{t}"] for r in rows])
    Dp = np.zeros(len(rows), bool); Dn = np.zeros(len(rows), bool)
    for t in TRAITS:
        a, b = labels(meta, t); Dp |= a; Dn |= b
    meta["Dpos"], meta["Dneg"] = Dp, Dn
    return rows, meta


def prompt_profile(meta):
    """C1: the graded prompt itself on the full judge profile; H0 manipulation check and the +1 wording rule on TRAIN."""
    out = {}
    for t, levels in [("none", [0]), ("placebo", [1, 2, 3, 4])] + [(t, [-3, 1, 2, 3, 4]) for t in TRAITS]:
        for l in levels:
            m = (meta["trait"] == t) & (meta["level"] == l)
            out[f"{t}|{l}"] = dict(n=int(m.sum()), coh=float(meta["coh"][m].mean()), ntok=float(meta["ntok"][m].mean()),
                                   **{u: float(meta[f"P_{u}"][m].mean()) for u in TRAITS})
    tr = meta["split"] == "train"; h0 = {}
    for t in TRAITS:
        P = lambda tt, l: float(meta[f"P_{t}"][tr & (meta["trait"] == tt) & (meta["level"] == l)].mean())
        p0, p1, p2, p3, p4 = P("placebo", 3), P(t, 1), P(t, 2), P(t, 3), P(t, 4)
        h0[t] = dict(placebo_strong=p0, slight=p1, moderate=p2, strong=p3, extreme=p4, gain=p3 - p0, H0=bool(p3 - p0 >= .30),
                     graded=bool(p0 < p1 < p3 and p0 < p2 < p3 and p4 > p2))
    out["_H0"] = h0; out["_wording_rule_pass"] = bool(sum(v["graded"] for v in h0.values()) >= 4)
    return out


def length_control(meta):
    out = {}
    for t in TRAITS:
        m = meta["trait"] == t
        out[t] = float(spearmanr(meta["level"][m], meta["ntok"][m])[0])
    m = meta["trait"] == "placebo"; out["placebo"] = float(spearmanr(meta["level"][m], meta["ntok"][m])[0])
    return out


def ridge_resid(X, E, q, nfold=5, lams=(1e-3, 1e-2, 1e-1, 1.0)):
    """X - E @ B, B by ridge on the other question folds; lambda (relative to trace(E'E)/dim) chosen by mean out-of-fold R2."""
    Qs = np.unique(q); f = {v: i % nfold for i, v in enumerate(np.random.default_rng(0).permutation(Qs))}
    fold = np.array([f[v] for v in q]); best = None
    for lr in lams:
        R = np.empty_like(X)
        for i in range(nfold):
            tr, te = fold != i, fold == i
            C = E[tr].T @ E[tr]; lam = lr * np.trace(C) / C.shape[0]
            R[te] = X[te] - E[te] @ np.linalg.solve(C + lam * np.eye(C.shape[0]), E[tr].T @ X[tr])
        r2 = 1 - (R ** 2).sum() / (X ** 2).sum()
        if best is None or r2 > best[0]: best = (r2, lr, R)
    log(f"ridge text->act out-of-fold R2 {best[0]:.3f} (lambda_rel {best[1]})")
    return best[2], float(best[0])


def len_resid(X, ntok, q):
    z = G.demean_by(np.log1p(ntok.astype(np.float64))[:, None], q)[:, 0]
    return X - np.outer(z, z @ X / (z @ z))


def pval(nulls, obs, key, greater=True):
    v = np.array([x[key] for x in nulls], float); v = v[np.isfinite(v)]
    if not np.isfinite(obs) or len(v) == 0: return np.nan
    return float((1 + np.sum(v >= obs if greater else v <= obs)) / (1 + len(v)))


def per_trait(X, meta, t, Z, PCAW, rng, reps, nfield, full=True, power=False):
    lad = G.ladder(meta, t); mL = {k: v[lad] for k, v in meta.items()}; Xl = X[lad]
    r = {}
    for side, two in (("one", False), ("two", True)):
        c = G.curve_stats(Xl, mL, t, two); r[f"curve_{side}"] = {k: v for k, v in c.items() if k != "_M"}
        if not full: continue
        r[f"curve_{side}"]["ord1_p"] = G.ord1_relabel_p(c)
        lc = G.ladder_cells(Xl, mL, t, two)
        r[f"curve_{side}_test"] = G.line_test(lc["D"], lc, rng, reps)        # PRIMARY (A1): max-p of cell permutation / cell sign-flip
        nl = [G.curve_stats(G.null_line(Xl, mL, t, rng, two), mL, t, two) for _ in range(min(reps, 99))]   # row-level, descriptive
        r[f"curve_{side}_null_line"] = {k: dict(mean=float(np.nanmean([x[k] for x in nl])), p95=float(np.nanpercentile([x[k] for x in nl], 95)),
                                                p=pval(nl, c[k], k)) for k in STATS}
        r[f"curve_{side}_null_line"]["straight"] = dict(mean=float(np.nanmean([x["straight"] for x in nl])),
                                                        p05=float(np.nanpercentile([x["straight"] for x in nl], 5)), p=pval(nl, c["straight"], "straight", False))
        nw = [G.curve_stats(G.null_line(Xl, mL, t, rng, two, mode="wild"), mL, t, two) for _ in range(min(reps, 99))]
        r[f"curve_{side}_null_line_wild"] = {k: dict(p=pval(nw, c[k], k)) for k in STATS}     # v1.1 null, secondary (liberal under cell effects)
        sx = [G.curve_stats(G.null_simplex(Xl, mL, t, rng, two), mL, t, two) for _ in range(10)]
        r[f"curve_{side}_ref_simplex"] = {k: float(np.nanmean([x[k] for x in sx])) for k in STATS + ("straight", "ord1")}
    if not full: return r
    r["kappa"] = G.cell_kappa(Xl, mL, t, C=len(cells()))
    if power:   # in-sample power for FLAT (powered), A1: planted arc60, primary test at the conservative Holm threshold .02/6
        lc = G.ladder_cells(Xl, mL, t)
        r["planted_arc60"] = {pl: G.planted_power(lc["D"], lc, rng, .02 / len(TRAITS), 60.0, pl, nsim=ns)
                              for pl, ns in (("noise", NSIM_POWER), ("random", max(NSIM_POWER // 3, 2)))}   # random plane: descriptive
    r["input_dep"] = G.input_dependence(Xl, mL, t, rng=rng, nboot=500)
    Dp, Dn = labels(mL, t); r["n_pos"], r["n_neg"] = int(Dp.sum()), int(Dn.sum())
    if Dp.sum() >= 20 and Dn.sum() >= 20:
        tt = mL["trait"] == t; Zt = Z[lad][tt]; qt, lt, Dpt, Dnt = mL["q"][tt], mL["level"][tt], Dp[tt], Dn[tt]
        r["field"] = G.local_field(Zt, qt, Dpt, Dnt, lt)
        nf, sh, gn, pn = [], [], [], []
        for _ in range(nfield):
            Xn = G.null_line(Xl, mL, t, rng, True); Zn = (Xn - Xl.mean(0)) @ PCAW.T
            nf.append(G.local_field(Zn[tt], qt, Dpt, Dnt, lt)); gn.append(G.geo_ratio(Zn[tt], lt)["geo_ratio"])
            try: pn.append(path_curvature(fit_path(Zn, mL["level"] * tt, Dp, Dn)["pi"]))
            except RuntimeError: pass
            lab = np.where(Dpt, 1, np.where(Dnt, -1, 0)); lab2 = lab.copy()
            for v in np.unique(qt):
                w = np.where((qt == v) & (lab != 0))[0]; lab2[w] = lab[rng.permutation(w)]
            sh.append(G.local_field(Zt, qt, lab2 == 1, lab2 == -1, lt))
        for k in ("field_coh", "field_rot"):
            for nm, src in (("null_line", nf), ("shuffle", sh)):
                v = [x[k] for x in src]; r["field"][f"{k}_{nm}"] = [float(np.mean(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))]
        r["geo"] = G.geo_ratio(Zt, lt); r["geo"]["null_line"] = [float(np.nanmean(gn)), float(np.nanpercentile(gn, 95))]
        # attribution rule (prereg 6.4): fitted-path straightness on data vs the same fit on line-null data
        try:
            r["path"] = dict(straight=path_curvature(fit_path(Z[lad], mL["level"] * tt, Dp, Dn)["pi"]),
                             null_mean=float(np.mean(pn)) if pn else np.nan, null_p05=float(np.percentile(pn, 5)) if pn else np.nan)
        except RuntimeError as e:
            r["path"] = dict(error=str(e))
        r["ideea_demeaned"] = G.ideea_connectivity(Zt[Dpt], mL["src"][tt][Dpt], mL["qid"][tt][Dpt], rng=rng)
    return r


def h1_verdicts(res, lay, h0):
    """CURVED: ord1 relabel p <= .025 (one-sided) and Holm-adjusted primary quadperp p <= .02 over H0-passing traits.
    FLAT (powered): not CURVED and planted-arc60 power (noise plane, alpha .02/6) has 90% lower bound >= .8. Else UNDETERMINED."""
    T = [t for t in TRAITS if h0[t]["H0"] and str(lay[t]) in res and "curve_one_test" in res[str(lay[t])][t]]
    p = {t: res[str(lay[t])][t]["curve_one_test"]["p_quadperp"] for t in T}; ph = G.holm(p) if p else {}
    out = {}
    for t in TRAITS:
        if t not in T: out[t] = dict(verdict="DROPPED (H0 failed or missing)"); continue
        r = res[str(lay[t])][t]; gate = r["curve_one"]["ord1_p"] <= .025
        pw = r.get("planted_arc60", {}).get("noise", {})
        if not gate: v = "NO GRADED ORDER (gate failed; shape not interpreted)"
        elif ph[t] <= .02: v = "CURVED"
        else: v = "FLAT (powered)" if pw.get("lo90", 0) >= .8 else "UNDETERMINED (underpowered)"
        out[t] = dict(verdict=v, gate=bool(gate), p=p[t], p_holm=ph[t], power=pw.get("power"), power_lo90=pw.get("lo90"), layer=lay[t])
    return out


if __name__ == "__main__":
    rows, meta0 = load_meta()
    jdump(prompt_profile(meta0), f"{OUT}/prompt_profile.json"); jdump(length_control(meta0), f"{OUT}/length.json")
    split = meta0["split"] == args.split; idx = np.where(split)[0]
    meta = {k: v[split] for k, v in meta0.items()}
    lay = jload(f"{OUT}/layers.json") if os.path.exists(f"{OUT}/layers.json") else None
    rng = np.random.default_rng(0)
    for rep in args.reps_list.split(","):
        res = {}
        if rep == "txt":
            src = {None: np.load(f"{OUT}/emb_gen_main.npy", mmap_mode="r")}
        else:
            A = np.load(f"{OUT}/acts_{'sys' if rep == 'sys' else 'neu'}.npy", mmap_mode="r")
            prim = sorted(set(lay.values())) if lay else LPROFILE
            src = {L: A for L in (sorted(set(LPROFILE) | set(prim)) if args.layers == "all" else prim)}
        for L, AA in src.items():
            Xraw = np.asarray(AA[idx] if L is None else AA[idx, L], np.float64)
            X = G.demean_by(Xraw, meta["q"]); extra = {}
            if rep == "res":
                E = G.demean_by(np.load(f"{OUT}/emb_gen_main.npy")[idx].astype(np.float64), meta["q"])
                X, extra["ridge_R2"] = ridge_resid(X, E, meta["q"])
            if rep == "len":
                X = len_resid(X, meta["ntok"], meta["q"])
            mu, PCAW = G.pca_fit(X, 64); Z = (X - mu) @ PCAW.T
            out = dict(extra)
            for t in TRAITS:
                full = (L is None) or (lay is None) or (lay[t] == L)
                out[t] = per_trait(X, meta, t, Z, PCAW, rng, args.reps, args.nfield, full=full, power=full and rep in ("neu", "res") and L is not None)
            for anchor in ("placebo", "dpm"):
                out[f"separation_{anchor}"] = G.separation(X, meta, TRAITS, anchor=anchor)
                out[f"separation_{anchor}_ci"] = G.separation_boot(X, meta, TRAITS, NEIGHBOUR_PAIRS, nboot=100, anchor=anchor)
            acc = {}
            for a, b in NEIGHBOUR_PAIRS:
                obs = G.knn_trait_acc(Z, meta, a, b); nul = []
                for s in range(20):
                    m2 = dict(meta); tr = meta["trait"].copy(); sel = np.where(meta["Dpos"] & np.isin(tr, [a, b]))[0]
                    for v in np.unique(meta["q"][sel]):
                        w = sel[meta["q"][sel] == v]; tr[w] = tr[rng.permutation(w)]
                    m2["trait"] = tr; nul.append(G.knn_trait_acc(Z, m2, a, b, rng=np.random.default_rng(s)))
                acc[f"{a}|{b}"] = dict(acc=obs, shuffle_mean=float(np.nanmean(nul)), shuffle_p95=float(np.nanpercentile(nul, 95)))
            out["knn_acc"] = acc
            if L is not None and rep in ("neu", "sys"):   # IDEEA's own setting: RAW (not demeaned) activations
                mr, Wr = G.pca_fit(Xraw, 64); Zr = (Xraw - mr) @ Wr.T; out["ideea_raw"] = {}
                for t in TRAITS:
                    Dp = labels(meta, t)[0]
                    if Dp.sum() >= 20: out["ideea_raw"][t] = G.ideea_connectivity(Zr[Dp], meta["src"][Dp], meta["qid"][Dp], rng=rng)
            res[str(L)] = out
            if lay is not None and rep != "txt":   # H1 verdicts at each trait's primary layer (prereg section 5, A1)
                res["H1"] = h1_verdicts(res, lay, jload(f"{OUT}/prompt_profile.json")["_H0"])
            log(rep, L, {t: round(out[t]["curve_one"]["offline"], 3) for t in TRAITS})
            jdump(res, f"{OUT}/geom_{rep}_{args.split}.json")
