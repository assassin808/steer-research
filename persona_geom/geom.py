"""Geometry statistics for the persona study (prereg: results/persona_geom_prereg.md, sections 4-5).

Conventions. X: [n, d] rows (activations at one layer, or text embeddings), already demeaned per question.
meta (dict of arrays, length n): q (int, question; after a bootstrap, copies get new q but keep 'oq' = original question),
trait ('none' = bare neutral, 'placebo', or a trait), level (common.cells codes: trait 1/2/3/4 = slight/moderate/strong/extreme,
-3 = neg instruction at strong; placebo 1-4; none 0), sample (int).
LADDER of trait t (difference-in-differences against the placebo at the SAME intensity word, so the instruction form cancels):
  one-sided (PRIMARY):  c_0 = 0, c_k = mean(t, level k) - mean(placebo, level k), k = 1..4        -> L = 5 points
  two-sided (secondary): c_-3 = mean(t, -3) - mean(placebo, 3) prepended                                 -> L = 6 points
Split halves = all balanced bipartitions of the sample indices within every (question, cell); paraphrase is fixed per question,
so both halves contain every question and the same paraphrase (weekday A3, review finding 4).
Every statistic is oriented "larger = more of the structure named".
"""
import itertools, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, completeness_score

ONE, TWO = [0, 1, 2, 3, 4], [-3, 0, 1, 2, 3, 4]      # amendment A1: level 4 = extreme


# ------------------------------------------------------------------ basics
def demean_by(X, g):
    """subtract the per-group (question) mean. Balanced crossed design: every question has the same cells, so this removes
    question content without biasing any cell contrast."""
    X = X.astype(np.float64, copy=True)
    for v in np.unique(g):
        m = g == v; X[m] -= X[m].mean(0)
    return X


def pca_fit(X, d=64):
    mu = X.mean(0); _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:d]


def ladder(meta, t):
    return (meta["trait"] == t) | (meta["trait"] == "placebo") | (meta["trait"] == "none")


def half_masks(sample):
    S = int(sample.max()) + 1
    parts = [c for c in itertools.combinations(range(S), S // 2) if 0 in c] if S >= 2 else []
    return [np.isin(sample, c) for c in parts]


def contrast_centroids(X, meta, t, mask=None, two_sided=False):
    m = np.ones(len(X), bool) if mask is None else mask
    tr, pl, lv = meta["trait"], meta["trait"] == "placebo", meta["level"]
    mean = lambda sel: X[sel & m].mean(0)
    rows = [np.zeros(X.shape[1])] + [mean((tr == t) & (lv == k)) - mean(pl & (lv == k)) for k in (1, 2, 3, 4)]
    if two_sided:
        rows = [mean((tr == t) & (lv == -3)) - mean(pl & (lv == 3))] + rows
    return np.stack(rows)


def cross_gram(cA, cB):
    a, b = cA - cA.mean(0), cB - cB.mean(0)
    return 0.5 * (a @ b.T + b @ a.T)


def _poly(L):
    x = np.linspace(0, 1, L); V = np.vander(x, 3, increasing=True)
    Qm, _ = np.linalg.qr(V); Qm = Qm * np.sign(Qm[-1])
    return Qm[:, 1], Qm[:, 2]          # orthonormal linear and quadratic contrasts over equally spaced (ordinal) levels


def within_var(X, meta, t, two_sided=False):
    """mean squared deviation of trait-t and placebo rows from their (question-pooled) cell mean"""
    lv = TWO if two_sided else ONE; tot, n = 0.0, 0
    for tr in (t, "placebo"):
        for k in set(lv + [3]) - {0}:
            sel = (meta["trait"] == tr) & (meta["level"] == k)
            if sel.sum() > 1:
                tot += ((X[sel] - X[sel].mean(0)) ** 2).sum(); n += sel.sum()
    return tot / max(n, 1)


def curve_from_halves(CA, CB, c_all, W):
    """CA, CB: lists of half-centroid arrays [L, d] (one per split). Returns the split-half-corrected statistics:
    offline = 1 - top-1 eigen share of the cross-Gram (0 for any straight line, also with unequal steps)
    quadperp = <P_perp b2^A, P_perp b2^B> / <b1^A, b1^B>  (power-curve C3)
    turn = median corrected turning angle at interior vertices (deg); straight = cross-estimated chord / polyline
    ord1 = share of the cross-Gram along the linear level contrast; snr = class SNR (power-curve definition)."""
    L = len(c_all); p1, p2 = _poly(L)
    b1 = p1 @ c_all; u = b1 / (np.linalg.norm(b1) + 1e-300)
    res = {k: [] for k in ("offline", "quadperp", "turn", "straight", "ord1")}; Ms = []
    for cA, cB in zip(CA, CB):
        M = cross_gram(cA, cB); Ms.append(M); ev = np.sort(np.linalg.eigvalsh(M))[::-1]; tot = np.trace(M)
        res["offline"].append(1 - ev[0] / tot if tot > 0 else np.nan)
        res["ord1"].append(float(p1 @ M @ p1) / tot if tot > 0 else np.nan)
        Pp = lambda v: v - (v @ u) * u
        den = (p1 @ cA) @ (p1 @ cB)
        res["quadperp"].append(float(Pp(p2 @ cA) @ Pp(p2 @ cB)) / den if den > 0 else np.nan)
        dA, dB = np.diff(cA, axis=0), np.diff(cB, axis=0); ii = (dA * dB).sum(1); ang = []
        for k in range(L - 2):
            num = 0.5 * (dA[k] @ dB[k + 1] + dB[k] @ dA[k + 1]); dd = ii[k] * ii[k + 1]
            ang.append(np.degrees(np.arccos(np.clip(num / np.sqrt(dd), -1, 1))) if dd > 0 and ii[k] > 0 else np.nan)
        res["turn"].append(np.nanmedian(ang) if np.isfinite(ang).any() else np.nan)
        ch = (cA[-1] - cA[0]) @ (cB[-1] - cB[0])
        res["straight"].append(np.sqrt(max(ch, 0)) / max(np.sqrt(np.maximum(ii, 0)).sum(), 1e-300))
    out = {k: float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else np.nan for k, v in res.items()}
    out["snr"] = float(((c_all - c_all.mean(0)) ** 2).sum(1).mean() / W)
    out["_M"] = np.mean(Ms, 0) if Ms else None
    return out


def curve_stats(X, meta, t, two_sided=False):
    masks = half_masks(meta["sample"])
    CA = [contrast_centroids(X, meta, t, A, two_sided) for A in masks]
    CB = [contrast_centroids(X, meta, t, ~A, two_sided) for A in masks]
    return curve_from_halves(CA, CB, contrast_centroids(X, meta, t, None, two_sided), within_var(X, meta, t, two_sided))


def ord1_relabel_p(cs):
    """exact order-relabel null for ord1 on the averaged cross-Gram (all L! relabelings; reversal ties -> floor 2/L!)"""
    M = cs["_M"]; L = len(M); p1, _ = _poly(L); tot = np.trace(M)
    obs = p1 @ M @ p1 / tot
    vals = [p1[list(pm)] @ M @ p1[list(pm)] / tot for pm in itertools.permutations(range(L))]
    return float(np.mean(np.array(vals) >= obs - 1e-12))


def _null_shift(X, meta, t, cnew, rng, two_sided, mode="cell"):
    """Rebuild every trait-t ladder row i (question q, sample s, level k) as  X_pl(q, s, word(k)) + cnew_k + eps_i * R_i, with
    R_i = (X_i - X_pl(q, s, word(k))) - c_k the row's PAIRED residual (placebo row of the same question and sample) and eps_i a
    random sign (wild bootstrap). Placebo / neutral / other rows unchanged. Pairing keeps the half-split noise structure of the
    contrasts exactly as in the data (building rows around the pooled placebo mean biased the null low: test_geom.py).
    mode='cell' (PRIMARY, amendment A1, power_curve.md v2): residuals R = D - cnew_k (Freedman-Lane, null-model residuals); the
    residual of row (q, s, k) is taken from
    row (q, s, pi_q(k)), pi_q a random permutation of the non-zero ladder levels drawn per question -> whole (question, level)
    cells move together, so the null stays exchangeable when question x level cell effects exist (the wild / row nulls are
    anti-conservative there). mode='wild': the v1.1 paired sign-flip bootstrap (secondary)."""
    lv = TWO if two_sided else ONE; Xn = X.copy(); tr = meta["trait"]
    pl_idx = {(q, s, k): i for i, (q, s, k) in enumerate(zip(meta["q"], meta["sample"], meta["level"])) if tr[i] == "placebo"}
    sel = np.where((tr == t) & np.isin(meta["level"], [k for k in lv if k != 0]))[0]
    pj = np.array([pl_idx[(meta["q"][i], meta["sample"][i], 3 if meta["level"][i] == -3 else meta["level"][i])] for i in sel])
    Dd = X[sel] - X[pj]; lev = meta["level"][sel]
    c = {k: Dd[lev == k].mean(0) for k in lv if k != 0}
    if mode == "wild":
        R = Dd - np.stack([c[k] for k in lev])
    else:   # Freedman-Lane: residuals around the NULL-model (line) fit; full-model residuals biased the cell null low (test_geom)
        R = Dd - cnew[[lv.index(k) for k in lev]]
    if mode == "wild":
        R = R * rng.choice([-1.0, 1.0], len(sel))[:, None]
    else:
        nz = [k for k in lv if k != 0]; qs, ss = meta["q"][sel], meta["sample"][sel]
        pos = {(q, s, k): j for j, (q, s, k) in enumerate(zip(qs, ss, lev))}; src = np.arange(len(sel))
        for q in np.unique(qs):
            pm = dict(zip(nz, [nz[i] for i in rng.permutation(len(nz))]))
            for j in np.where(qs == q)[0]:
                src[j] = pos[(q, ss[j], pm[lev[j]])]
        R = R[src]
    for j, i in enumerate(sel):
        Xn[i] = X[pj[j]] + cnew[lv.index(lev[j])] + R[j]
    return Xn


def null_line(X, meta, t, rng, two_sided=False, mode="cell"):
    """practical flat null: ladder contrasts projected on their top PC (keeps the observed, possibly unequal, spacing), shifted so
    that level 0 stays at 0; paired residuals cell-permuted across levels within question (mode='cell', PRIMARY) or
    sign-flipped (mode='wild', secondary)."""
    c = contrast_centroids(X, meta, t, None, two_sided); m = c.mean(0)
    u = np.linalg.svd(c - m, full_matrices=False)[2][0]
    cl = m + np.outer((c - m) @ u, u); cl = cl - cl[(TWO if two_sided else ONE).index(0)]
    return _null_shift(X, meta, t, cl, rng, two_sided, mode)


def null_simplex(X, meta, t, rng, two_sided=False):
    """'clustered' reference: ladder points at mutually orthogonal random directions with the observed total spread"""
    c = contrast_centroids(X, meta, t, None, two_sided); L = len(c); s2 = ((c - c.mean(0)) ** 2).sum()
    E = np.linalg.qr(rng.standard_normal((X.shape[1], L)))[0].T
    E = E - E.mean(0); E *= np.sqrt(s2 / (E ** 2).sum()); E = E - E[(TWO if two_sided else ONE).index(0)]
    return _null_shift(X, meta, t, E, rng, two_sided)


# ------------------------------------------------------------------ A1: fast cell-level line tests, planted power, kappa
# All ladder statistics depend on the data only through the paired trait-minus-placebo differences D(q, k, s) (balanced design:
# contrast_centroids on a half = mean over questions of D on that half). Nulls and planted arcs are therefore built on D directly
# (D_null = cnew_k + residual cells), which is exact and ~100x faster than rebuilding rows (null_line / _null_shift, kept for the
# row-level uses: direction field, geodesic path checks).
def ladder_cells(X, meta, t, two_sided=False):
    """D [Q, K, S, d] for the non-zero ladder levels (order of ONE / TWO without 0), plus level list and the half masks"""
    lv = TWO if two_sided else ONE; nz = [k for k in lv if k != 0]; tr = meta["trait"]
    Qs = np.unique(meta["q"][tr == t]); S = int(meta["sample"].max()) + 1
    qi = {v: i for i, v in enumerate(Qs)}; D = np.zeros((len(Qs), len(nz), S, X.shape[1])); cnt = np.zeros((len(Qs), len(nz), S))
    pl = {(q, s, k): i for i, (q, s, k) in enumerate(zip(meta["q"], meta["sample"], meta["level"])) if tr[i] == "placebo"}
    for i in np.where((tr == t) & np.isin(meta["level"], nz))[0]:
        q, s, k = meta["q"][i], meta["sample"][i], meta["level"][i]
        j = pl[(q, s, 3 if k == -3 else k)]; a = (qi[q], nz.index(k), s); D[a] = X[i] - X[j]; cnt[a] += 1
    assert (cnt == 1).all(), "ladder_cells needs a balanced design (one row per question x level x sample)"
    masks = [np.array([s in c for s in range(S)]) for c in
             ([c for c in itertools.combinations(range(S), S // 2) if 0 in c] if S >= 2 else [])]
    return dict(D=D, lv=lv, nz=nz, i0=lv.index(0), masks=masks)


def _full(cz, lc):
    """insert the level-0 zero row"""
    return np.insert(cz, lc["i0"], 0.0, axis=0)


def stats_D(D, lc, W=1.0):
    CA = [_full(D[:, :, m].mean((0, 2)), lc) for m in lc["masks"]]; CB = [_full(D[:, :, ~m].mean((0, 2)), lc) for m in lc["masks"]]
    return curve_from_halves(CA, CB, _full(D.mean((0, 2)), lc), W)


def line_fit(D, lc):
    """null-model centroids: observed ladder projected on its top PC, level 0 kept at 0 -> [K, d] for the non-zero levels"""
    c = _full(D.mean((0, 2)), lc); m = c.mean(0); u = np.linalg.svd(c - m, full_matrices=False)[2][0]
    cl = m + np.outer((c - m) @ u, u); cl = cl - cl[lc["i0"]]
    return np.delete(cl, lc["i0"], axis=0), u


def null_D(D, cnew, rng, mode="cell"):
    """Freedman-Lane: residual cells R = D - cnew (null-model fit); 'cell' permutes whole (question, level) cells across levels
    within question; 'cellwild' flips the sign of whole cells (robust to level-dependent cell variance). Returns cnew + R*."""
    R = D - cnew[None, :, None, :]; Q, K = R.shape[:2]
    if mode == "cell":
        R = R[np.arange(Q)[:, None], np.argsort(rng.random((Q, K)), 1)]
    elif mode == "cellwild":
        R = R * rng.choice([-1.0, 1.0], (Q, K))[:, :, None, None]
    else:
        raise ValueError(mode)
    return cnew[None, :, None, :] + R


def line_test(D, lc, rng, reps=999, keys=("quadperp", "offline"), modes=("cell", "cellwild")):
    """PRIMARY H1 test (A1): p of each statistic against both cell-level line nulls; p_<key> = max over the two nulls"""
    obs = stats_D(D, lc); cnew, _ = line_fit(D, lc); out = {}
    for mode in modes:
        nl = [stats_D(null_D(D, cnew, rng, mode), lc) for _ in range(reps)]
        for k in keys:
            v = np.nan_to_num(np.array([x[k] for x in nl], float), nan=-np.inf)
            out[f"p_{k}_{mode}"] = float((1 + np.sum(v >= obs[k])) / (1 + reps)) if np.isfinite(obs[k]) else np.nan
    for k in keys:
        out[f"p_{k}"] = max(out[f"p_{k}_{m}"] for m in modes)
    out.update({k: float(obs[k]) for k in keys}); out["reps"] = reps
    return out


def holm(p):
    """Holm-adjusted p for a dict {name: p} (adjusted p_(j) = max_{i <= j} min(1, (m - i + 1) p_(i)))"""
    names = sorted(p, key=lambda n: p[n]); m = len(names); out, run = {}, 0.0
    for i, n in enumerate(names):
        run = max(run, min(1.0, (m - i) * p[n])); out[n] = run
    return out


def arc_D(D, lc, rng, deg=60.0, plane="noise", Qboot=None):
    """planted-arc data on the cell level: centroids replaced by a circular arc of total turning `deg` that keeps the ladder's
    positions along u (u = top eigenvector of the split-half cross-Gram, positions = mean of the two halves' projections, so
    sampling noise does not inflate the arc), plus cell-permuted residual cells of the real data. plane 'noise': v = top PC of
    the residual cells orthogonal to u (hardest plane); 'random': uniform v orthogonal to u. Qboot: resample questions with
    replacement to Qboot (pilot projection to the main design)."""
    if Qboot: D = D[rng.integers(0, len(D), Qboot)]
    M = np.mean([cross_gram(_full(D[:, :, m].mean((0, 2)), lc), _full(D[:, :, ~m].mean((0, 2)), lc)) for m in lc["masks"]], 0)
    c = _full(D.mean((0, 2)), lc); w, V = np.linalg.eigh(M); a = V[:, -1]          # top eigvector in level space
    u = (c - c.mean(0)).T @ a; u /= np.linalg.norm(u)
    sp = np.mean([(_full(D[:, :, m].mean((0, 2)), lc) - c[lc["i0"]]) @ u for m in lc["masks"]] +
                 [(_full(D[:, :, ~m].mean((0, 2)), lc) - c[lc["i0"]]) @ u for m in lc["masks"]], 0)
    sp = sp - sp[lc["i0"]]
    if sp[-1] < 0: u, sp = -u, -sp
    cl, _ = line_fit(D, lc); R = (D - cl[None, :, None, :]).mean(2).reshape(-1, D.shape[-1])
    if plane == "noise":
        R = R - np.outer(R @ u, u); v = np.linalg.svd(R, full_matrices=False)[2][0]
    else:
        v = rng.standard_normal(D.shape[-1]); v -= (v @ u) * u
    v /= np.linalg.norm(v)
    th = np.radians(deg) * sp / sp[-1]; Rr = sp[-1] / np.radians(deg)
    arc = np.delete(Rr * np.outer(np.sin(th), u) + Rr * np.outer(1 - np.cos(th), v), lc["i0"], axis=0)
    return null_D(D - cl[None, :, None, :] + arc[None, :, None, :], arc, rng, "cell")


def planted_power(D, lc, rng, alpha, deg=60.0, plane="noise", nsim=100, reps=None, Qboot=None, stat="quadperp"):
    """share of planted-arc datasets in which the PRIMARY test (max-p of both cell nulls) rejects at `alpha` (use the trait's Holm
    threshold, conservatively .02/m); reps >= 1/alpha. Reports a one-sided 90% Clopper-Pearson lower bound."""
    from scipy.stats import beta
    reps = reps or int(np.ceil(1 / alpha)) * 2 - 1; hits = 0
    for _ in range(nsim):
        hits += line_test(arc_D(D, lc, rng, deg, plane, Qboot), lc, rng, reps, keys=(stat,))[f"p_{stat}"] <= alpha + 1e-12
    lo = float(beta.ppf(0.10, hits, nsim - hits + 1)) if hits > 0 else 0.0
    return dict(power=hits / nsim, lo90=lo, nsim=nsim, reps=reps, alpha=alpha, deg=deg, plane=plane, Qboot=Qboot)


def cell_kappa(X, meta, t, C=None, levels=(1, 2, 3, 4)):
    """question x level cell-effect size of trait-t and placebo cells in the power/designcheck.py unit
    kappa = E||cell effect||^2 / (W_eps / 5), W_eps = within-cell (between-sample) row variance. ||cell||^2 is estimated by the
    cross-sample product <x_s - m_k, x_s' - m_k> (s != s'). If X was demeaned per question over C cells x S samples (pass C = cells
    per question of the FULL design), E[cross] = B (1 - 1/C)(1 - 1/Q) - W/(C S) - W/(Q S) and B is solved from it (without the
    correction the estimate is biased low by ~5/(C S) in kappa units: review of A1, finding 4). Also returned per level."""
    per, Wl = {}, []
    for k in levels:
        B = []; 
        for tr in (t, "placebo"):
            sel = (meta["trait"] == tr) & (meta["level"] == k)
            if not sel.any(): continue
            mk = X[sel].mean(0)
            for q in np.unique(meta["q"][sel]):
                r = X[sel & (meta["q"] == q)] - mk; S = len(r)
                if S < 2: continue
                G2 = r @ r.T; B.append((G2.sum() - np.trace(G2)) / (S * (S - 1)))
                Wl.append(((r - r.mean(0)) ** 2).sum() / (S - 1))
        per[k] = (float(np.mean(B)), len(np.unique(meta["q"][sel]))) if B else (np.nan, 1)
    W = float(np.mean(Wl)); kap = {}
    for k, (b, Qn) in per.items():
        if C: b = (b + W / (C * S) + W / (Qn * S)) / ((1 - 1 / C) * (1 - 1 / Qn))
        else: b = (b + W / (Qn * S)) / (1 - 1 / Qn)
        kap[k] = b / (W / 5)
    return dict(kappa=float(np.nanmean(list(kap.values()))), kappa_by_level={str(k): float(v) for k, v in kap.items()}, W_eps=W)


# ------------------------------------------------------------------ H2: input dependence of the intensity direction
def input_dependence(X, meta, t, nboot=1000, rng=None):
    """per-question direction d_q = mean(t, strong) - mean(placebo, strong), split halves by sample.
    D1 = sum_q <d_q^A,u_-q><d_q^B,u_-q> / sum_q <d_q^A,d_q^B>, u_-q = unit global direction from questions with a DIFFERENT
    original id (bootstrap copies excluded; review finding 5). 1 = one input-independent direction (magnitude-free).
    G = mean_{q != q'} <d_q^A, d_q'^B> / mean_q <d_q^A, d_q^B> (pairs of copies of the same original excluded)."""
    rng = rng or np.random.default_rng(0)
    q, oq = meta["q"], meta.get("oq", meta["q"]); masks = half_masks(meta["sample"])
    hi = (meta["trait"] == t) & (meta["level"] == 3); pl = (meta["trait"] == "placebo") & (meta["level"] == 3)
    Q = np.unique(q); dA, dB, O = [], [], []
    for v in Q:
        mq = q == v
        dA.append([X[mq & hi & A].mean(0) - X[mq & pl & A].mean(0) for A in masks])
        dB.append([X[mq & hi & ~A].mean(0) - X[mq & pl & ~A].mean(0) for A in masks]); O.append(oq[mq][0])
    dA, dB, O = np.array(dA), np.array(dB), np.array(O)

    def stat(ix):
        a, b, o = dA[ix], dB[ix], O[ix]; num = den = 0.0; within = []; cr = []
        for j in range(len(ix)):
            other = o != o[j]
            u = (a[other] + b[other]).sum(0); u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-300)
            num += ((a[j] * u).sum(-1) * (b[j] * u).sum(-1)).mean(); w = (a[j] * b[j]).sum(-1).mean(); den += w; within.append(w)
            cr.append(((a[j][None] * b[other]).sum(-1)).mean())
        return num / den, np.mean(cr) / np.mean(within)
    D1, G = stat(np.arange(len(Q)))
    bs = np.array([stat(rng.integers(0, len(Q), len(Q))) for _ in range(nboot)])
    return dict(D1=float(D1), D1_ci=[float(x) for x in np.percentile(bs[:, 0], [2.5, 97.5])],
                G=float(G), G_ci=[float(x) for x in np.percentile(bs[:, 1], [2.5, 97.5])], nq=int(len(Q)))


# ------------------------------------------------------------------ H3: trait separation
def trait_dirs_split(X, meta, traits, qmaskA, anchor="placebo"):
    """anchor='placebo' (PRIMARY): w_t = mean(t, strong) - mean(placebo, strong)  (same instruction form; no judge selection);
    anchor='dpm': w_t = mean(D+_t) - mean(D-_t). On the questions in qmaskA (row mask) and on the rest."""
    out = {}
    tr, lv = meta["trait"], meta["level"]
    for half, M in (("A", qmaskA), ("B", ~qmaskA)):
        if anchor == "placebo":
            pl = X[M & (tr == "placebo") & (lv == 3)].mean(0)
            out[half] = {t: X[M & (tr == t) & (lv == 3)].mean(0) - pl for t in traits}
        else:
            out[half] = {t: X[M & meta["Dpos"] & (tr == t)].mean(0) - X[M & meta["Dneg"] & (tr == t)].mean(0) for t in traits}
    return out


def separation(X, meta, traits, nsplit=50, rng=None, anchor="placebo"):
    """disattenuated cosine between trait directions, from question-half splits (split by ORIGINAL question 'oq', so bootstrap
    copies never fall in both halves; review finding 5). r(t,t') = mean_splits 0.5(<wA_t,wB_t'> + <wB_t,wA_t'>) /
    sqrt(<wA_t,wB_t><wA_t',wB_t'>). 1 = same direction up to noise. All 15 pairs are returned (non-neighbours = baseline)."""
    rng = rng or np.random.default_rng(0)
    oq = meta.get("oq", meta["q"]); Q = np.unique(oq); R = {}
    num = {}; dia = {t: [] for t in traits}
    for s in range(nsplit):
        qa = rng.permutation(Q)[: len(Q) // 2]; M = np.isin(oq, qa)
        w = trait_dirs_split(X, meta, traits, M, anchor)
        for t in traits: dia[t].append(w["A"][t] @ w["B"][t])
        for t, u in itertools.combinations(traits, 2):
            num.setdefault((t, u), []).append(0.5 * (w["A"][t] @ w["B"][u] + w["B"][t] @ w["A"][u]))
    for (t, u), v in num.items():
        den = np.sqrt(max(np.mean(dia[t]), 1e-300) * max(np.mean(dia[u]), 1e-300))
        R[f"{t}|{u}"] = float(np.mean(v) / den)
    return R


def separation_boot(X, meta, traits, pairs, nboot=200, nsplit=10, rng=None, anchor="placebo"):
    rng = rng or np.random.default_rng(1)
    Q = np.unique(meta["q"]); bs = []
    for b in range(nboot):
        pick = rng.integers(0, len(Q), len(Q))
        rows = np.concatenate([np.where(meta["q"] == Q[i])[0] for i in pick])
        newq = np.repeat(np.arange(len(Q)), [int((meta["q"] == Q[i]).sum()) for i in pick])
        m2 = {k: v[rows] for k, v in meta.items()}; m2["q"] = newq; m2["oq"] = meta["q"][rows]
        r = separation(X[rows], m2, traits, nsplit=nsplit, rng=rng, anchor=anchor)
        bs.append([r[f"{a}|{b_}"] for a, b_ in pairs])
    bs = np.array(bs)
    return {f"{a}|{b_}": [float(x) for x in np.percentile(bs[:, i], [2.5, 97.5])] for i, (a, b_) in enumerate(pairs)}


def knn_trait_acc(Z, meta, t, u, k=10, nfold=5, rng=None):
    """leave-question-group-out kNN accuracy of D+_t vs D+_u (balanced by subsampling the larger class)"""
    rng = rng or np.random.default_rng(0)
    m = meta["Dpos"] & np.isin(meta["trait"], [t, u])
    idx = np.where(m)[0]; y = (meta["trait"][idx] == t).astype(int)
    n = min((y == 0).sum(), (y == 1).sum())
    if n < 10: return np.nan
    idx = np.concatenate([rng.choice(idx[y == c], n, replace=False) for c in (0, 1)]); y = (meta["trait"][idx] == t).astype(int)
    Qs = np.unique(meta["q"][idx]); fold = {v: i % nfold for i, v in enumerate(rng.permutation(Qs))}
    f = np.array([fold[v] for v in meta["q"][idx]]); acc = []
    for i in range(nfold):
        tr, te = f != i, f == i
        if te.sum() == 0 or tr.sum() <= k: continue
        nn = NearestNeighbors(n_neighbors=k).fit(Z[idx[tr]]); _, nb = nn.kneighbors(Z[idx[te]])
        acc.append(((y[tr][nb].mean(1) > 0.5).astype(int) == y[te]).mean())
    return float(np.mean(acc))


# ------------------------------------------------------------------ local direction field (proposal)
def local_field(Z, q, Dpos, Dneg, lev, k=10):
    """v(a) = mean of the k nearest D+ rows minus mean of the k nearest D- rows, neighbours from OTHER questions.
    coh = mean_a cos(v(a), vbar); rot = angle between mean v over rows with level <= -1 and level >= +1 (deg)."""
    P, N = np.where(Dpos)[0], np.where(Dneg)[0]
    kk = min(k + 40, len(P), len(N))
    _, ip = NearestNeighbors(n_neighbors=kk).fit(Z[P]).kneighbors(Z)
    _, iN = NearestNeighbors(n_neighbors=kk).fit(Z[N]).kneighbors(Z)
    V = np.zeros_like(Z)
    ok = np.ones(len(Z), bool)
    for a in range(len(Z)):
        pp = [j for j in P[ip[a]] if q[j] != q[a]][:k]; nn_ = [j for j in N[iN[a]] if q[j] != q[a]][:k]
        if not pp or not nn_: ok[a] = False; continue
        V[a] = Z[pp].mean(0) - Z[nn_].mean(0)
    V, lev = V[ok], lev[ok]
    vb = V.mean(0)
    coh = float(np.mean(V @ vb / (np.linalg.norm(V, axis=1) * np.linalg.norm(vb) + 1e-300)))
    lo, hi = V[lev <= -1].mean(0), V[lev >= 1].mean(0)
    rot = float(np.degrees(np.arccos(np.clip(lo @ hi / (np.linalg.norm(lo) * np.linalg.norm(hi) + 1e-300), -1, 1))))
    return dict(field_coh=coh, field_rot=rot)


# ------------------------------------------------------------------ kNN graph / Isomap
def knn_graph(Z, k=10, power=1.0):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z); dist, ind = nn.kneighbors(Z)
    n = len(Z)
    G = csr_matrix((dist[:, 1:].ravel() ** power, (np.repeat(np.arange(n), k), ind[:, 1:].ravel())), shape=(n, n))
    return G.maximum(G.T)


def geo_ratio(Z, lev, k=10):
    """Isomap geodesic / Euclidean distance between the nodes nearest the level -3 (neg) and +3 (strong) centroids (1 = straight).
    Z, lev: trait-t rows only (placebo / neutral rows excluded by the caller)."""
    G = knn_graph(Z, k); nc, _ = connected_components(G, directed=False)
    a = np.argmin(((Z - Z[lev == -3].mean(0)) ** 2).sum(1)); b = np.argmin(((Z - Z[lev == lev.max()].mean(0)) ** 2).sum(1))
    d = dijkstra(G, directed=False, indices=[a])[0, b]
    return dict(geo_ratio=float(d / np.linalg.norm(Z[a] - Z[b])) if np.isfinite(d) else np.nan, ncomp=int(nc))


# ------------------------------------------------------------------ H4: IDEEA cluster connectivity
def ideea_connectivity(Zp, qsrc, qid, K=4, k=10, nnull=20, rng=None):
    """KMeans(K) on D+ rows (IDEEA's own clustering, no pairing). Content tracking: AMI(cluster, question source trait) and
    completeness(qid -> cluster) = how far each question's rows fall in one cluster (AMI with ~120 qids and K = 4 is capped
    near .45, review finding 6), with a cluster-label shuffle baseline; and the between-cluster share of kNN edges relative to a flat Gaussian null with D+'s
    mean and covariance (same KMeans re-run): edge_ratio = obs / mean(null); < 1 = clusters less connected than cuts of one blob."""
    rng = rng or np.random.default_rng(0)
    lab = KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(Zp)

    def edge_share(Z, l):
        G = knn_graph(Z, k).tocoo(); return float(np.mean(l[G.row] != l[G.col]))
    obs = edge_share(Zp, lab)
    mu, C = Zp.mean(0), np.cov(Zp.T) + 1e-9 * np.eye(Zp.shape[1])
    nul = []
    for _ in range(nnull):
        Y = rng.multivariate_normal(mu, C, size=len(Zp), method="cholesky")
        nul.append(edge_share(Y, KMeans(n_clusters=K, random_state=0, n_init=4).fit_predict(Y)))
    comp_sh = [completeness_score(qid, rng.permutation(lab)) for _ in range(20)]
    return dict(ami_src=float(adjusted_mutual_info_score(qsrc, lab)), comp_qid=float(completeness_score(qid, lab)),
                comp_qid_shuffle=[float(np.mean(comp_sh)), float(np.percentile(comp_sh, 95))],
                edge_share=obs, edge_null=float(np.mean(nul)), edge_null_p05=float(np.percentile(nul, 5)),
                edge_ratio=float(obs / np.mean(nul)), sizes=np.bincount(lab, minlength=K).tolist())
