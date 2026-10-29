"""Geometry analysis H1-H4 + trivial controls (prereg sections 4-5). CPU.
usage: RUN=main python analyze_geom.py [--reps 99] [--layers all|primary]
Representations: neu (acts_neu, PRIMARY), sys (acts_sys), txt (sentence-encoder embedding of the response text; surface control 1),
res (acts_neu minus a cross-fitted ridge prediction from the text embedding; surface control 2).
Primary layer per trait = l*_t from <OUT>/layers.json (steering sweep); if absent, every layer in LPROFILE is 'primary'.
Output: <OUT>/geom_<rep>.json and a judge-profile table <OUT>/prompt_profile.json.
"""
import os, sys, json, argparse, numpy as np
from common import OUT, TRAITS, NEIGHBOUR_PAIRS, LEVELS, jload, jdump, log
import geom as G
from paths import labels, fit_path, path_curvature

ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=99); ap.add_argument("--nfield", type=int, default=20)
ap.add_argument("--reps_list", default="neu,txt,res,sys"); ap.add_argument("--layers", default="primary"); ap.add_argument("--split", default="train")
args = ap.parse_args()
LPROFILE = list(range(2, 26, 2))


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


def ridge_resid(X, E, q, nfold=5, lam_rel=1e-2):
    """X - E @ B, B fitted by ridge on the other question folds (lambda = lam_rel * trace(E'E)/dim)."""
    Qs = np.unique(q); f = {v: i % nfold for i, v in enumerate(np.random.default_rng(0).permutation(Qs))}
    fold = np.array([f[v] for v in q]); R = np.empty_like(X); r2 = []
    for i in range(nfold):
        tr, te = fold != i, fold == i
        C = E[tr].T @ E[tr]; lam = lam_rel * np.trace(C) / C.shape[0]
        B = np.linalg.solve(C + lam * np.eye(C.shape[0]), E[tr].T @ X[tr])
        R[te] = X[te] - E[te] @ B; r2.append(1 - (R[te] ** 2).sum() / (X[te] ** 2).sum())
    log("ridge text->act held-out R2", np.round(r2, 3))
    return R


def prompt_profile(meta):
    """trivial control 1: the graded prompt itself, on the full judge profile (all splits)."""
    out = {}
    for t in ["none"] + TRAITS:
        for l in (LEVELS if t != "none" else [0]):
            m = (meta["trait"] == t) & (meta["level"] == l)
            if t != "none" and l == 0: continue
            out[f"{t}|{l}"] = dict(n=int(m.sum()), coh=float(meta["coh"][m].mean()), ntok=float(meta["ntok"][m].mean()),
                                   **{u: float(meta[f"P_{u}"][m].mean()) for u in TRAITS})
    return out


def per_trait(X, meta, t, Z, rng, reps, nfield, full=True):
    lad = G.ladder(meta, t); Xl = X[lad]; lev = meta["level"][lad]; smp = meta["sample"][lad]; q = meta["q"][lad]
    r = dict(curve=G.curve_stats(Xl, lev, smp))
    if not full: return r
    nl = [G.curve_stats(G.null_line(Xl, lev, q, rng), lev, smp) for _ in range(reps)]
    r["curve_null_line"] = {k: dict(mean=float(np.nanmean([x[k] for x in nl])), p95=float(np.nanpercentile([x[k] for x in nl], 95)),
                                    p=float((1 + np.sum([x[k] >= r["curve"][k] for x in nl])) / (1 + reps))) for k in ("offline", "quadperp", "turn")}
    r["curve_null_line"]["straight"] = dict(mean=float(np.nanmean([x["straight"] for x in nl])),
                                            p05=float(np.nanpercentile([x["straight"] for x in nl], 5)),
                                            p=float((1 + np.sum([x["straight"] <= r["curve"]["straight"] for x in nl])) / (1 + reps)))
    sx = [G.curve_stats(G.null_simplex(Xl, lev, q, rng), lev, smp) for _ in range(10)]
    r["curve_ref_simplex"] = {k: float(np.nanmean([x[k] for x in sx])) for k in ("offline", "quadperp", "turn", "straight", "ord1")}
    r["ord1_p"], _ = G.ord1_relabel_p(Xl, lev, smp)
    r["input_dep"] = G.input_dependence(Xl, lev, q, smp, rng=rng, nboot=500)
    Zl = Z[lad]; Dp, Dn = labels({k: v[lad] for k, v in meta.items()}, t)
    r["n_pos"], r["n_neg"] = int(Dp.sum()), int(Dn.sum())
    if Dp.sum() >= 20 and Dn.sum() >= 20:
        r["field"] = G.local_field(Zl, q, Dp, Dn, lev)
        nf = []
        for _ in range(nfield):
            Xn = G.null_line(Xl, lev, q, rng); Zn = (Xn - Xn.mean(0)) @ PCAW.T
            nf.append(G.local_field(Zn, q, Dp, Dn, lev))
        sh = []
        for _ in range(nfield):  # label shuffle within question among the labelled rows
            lab = np.where(Dp, 1, np.where(Dn, -1, 0)); lab2 = lab.copy()
            for v in np.unique(q):
                w = np.where((q == v) & (lab != 0))[0]; lab2[w] = lab[rng.permutation(w)]
            sh.append(G.local_field(Zl, q, lab2 == 1, lab2 == -1, lev))
        for k in ("field_coh", "field_rot"):
            r["field"][k + "_null_line"] = [float(np.mean([x[k] for x in nf])), float(np.percentile([x[k] for x in nf], 5)), float(np.percentile([x[k] for x in nf], 95))]
            r["field"][k + "_shuffle"] = [float(np.mean([x[k] for x in sh])), float(np.percentile([x[k] for x in sh], 5)), float(np.percentile([x[k] for x in sh], 95))]
        r["geo"] = G.geo_ratio(Zl, lev)
        gn = [G.geo_ratio((G.null_line(Xl, lev, q, rng) - Xl.mean(0)) @ PCAW.T, lev)["geo_ratio"] for _ in range(nfield)]
        r["geo"]["null_line"] = [float(np.nanmean(gn)), float(np.nanpercentile(gn, 95))]
        # attribution rule (prereg 6.4): fitted-path straightness on data vs the same fit on line-null data
        try:
            ps = path_curvature(fit_path(Zl, lev, Dp, Dn)["pi"])
            pn = [path_curvature(fit_path((G.null_line(Xl, lev, q, rng) - Xl.mean(0)) @ PCAW.T, lev, Dp, Dn)["pi"]) for _ in range(nfield)]
            r["path"] = dict(straight=float(ps), null_mean=float(np.mean(pn)), null_p05=float(np.percentile(pn, 5)))
        except RuntimeError as e:
            r["path"] = dict(error=str(e))
        mP = Dp
        r["ideea_demeaned"] = G.ideea_connectivity(Zl[mP], meta["src"][lad][mP], meta["qid"][lad][mP], rng=rng)
    return r


if __name__ == "__main__":
    rows, meta0 = load_meta()
    jdump(prompt_profile(meta0), f"{OUT}/prompt_profile.json")
    split = meta0["split"] == args.split
    meta = {k: v[split] for k, v in meta0.items()}
    lay = jload(f"{OUT}/layers.json") if os.path.exists(f"{OUT}/layers.json") else None
    rng = np.random.default_rng(0)
    for rep in args.reps_list.split(","):
        res = {}
        if rep == "txt":
            src = {None: np.load(f"{OUT}/emb_gen_main.npy")}
        else:
            A = np.load(f"{OUT}/acts_{'neu' if rep == 'res' else rep}.npy", mmap_mode="r")
            Ls = sorted(set(LPROFILE) | (set(lay.values()) if lay else set())) if args.layers == "all" else (sorted(set(lay.values())) if lay else LPROFILE)
            src = {L: A for L in Ls}
        for L, AA in src.items():
            idx = np.where(split)[0]
            Xraw = np.asarray(AA[idx] if L is None else AA[idx, L], np.float64)
            X = G.demean_by(Xraw, meta["q"])
            if rep == "res":   # surface control 2: remove what a cross-fitted ridge from the text embedding predicts
                X = ridge_resid(X, G.demean_by(np.load(f"{OUT}/emb_gen_main.npy")[idx].astype(np.float64), meta["q"]), meta["q"])
            mu, PCAW = G.pca_fit(X, 64); Z = (X - mu) @ PCAW.T   # PCAW is read by per_trait (module global)
            out = {}
            for t in TRAITS:
                full = (L is None) or (lay is None) or (lay[t] == L)
                out[t] = per_trait(X, meta, t, Z, rng, args.reps, args.nfield, full=full)
            out["separation"] = G.separation(X, meta, TRAITS)
            out["separation_ci"] = G.separation_boot(X, meta, TRAITS, NEIGHBOUR_PAIRS, nboot=100)
            acc = {}
            for a, b in NEIGHBOUR_PAIRS:
                obs = G.knn_trait_acc(Z, meta, a, b)
                nul = []
                for s in range(20):
                    m2 = dict(meta); tr = meta["trait"].copy(); sel = np.where(meta["Dpos"] & np.isin(tr, [a, b]))[0]
                    for v in np.unique(meta["q"][sel]):
                        w = sel[meta["q"][sel] == v]; tr[w] = tr[rng.permutation(w)]
                    m2["trait"] = tr; nul.append(G.knn_trait_acc(Z, m2, a, b, rng=np.random.default_rng(s)))
                acc[f"{a}|{b}"] = dict(acc=obs, shuffle_mean=float(np.nanmean(nul)), shuffle_p95=float(np.nanpercentile(nul, 95)))
            out["knn_acc"] = acc
            # raw (not demeaned) IDEEA clustering, as IDEEA does it
            if L is not None:
                Xr = Xraw; mr, Wr = G.pca_fit(Xr, 64); Zr = (Xr - mr) @ Wr.T
                out["ideea_raw"] = {t: G.ideea_connectivity(Zr[labels(meta, t)[0]], meta["src"][labels(meta, t)[0]], meta["qid"][labels(meta, t)[0]], rng=rng)
                                    for t in TRAITS if labels(meta, t)[0].sum() >= 20}
            res[str(L)] = out
            log(rep, L, {t: round(out[t]["curve"]["offline"], 3) for t in TRAITS})
            jdump(res, f"{OUT}/geom_{rep}_{args.split}.json")
