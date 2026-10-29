"""Fit the steering arms for one trait at one layer from TRAIN-question rows (prereg section 6).

Space: acts_neu (response read in the neutral context), demeaned per question, PCA-64 fitted on all TRAIN rows of all traits.
Labels: D+_t = rows of trait t at level >= 1 (slight/moderate/strong) with judge P_t >= POS_THR; D-_t = rows of trait t at level -3
(neg instruction) with P_t <= NEG_THR. Graph rows = the trait's ladder rows (trait t + placebo + bare neutral).
Arms (all are displacements added to the residual at layer L; one factor changes between neighbouring arms):
  pv       persona vector, full-d: u = mean(D+_t) - mean(D-_t) (demeaned rows), delta = alpha * u.
  chord    PCA-64 straight segment from pi(s_neu) to pi(1):    delta(s) = lam * s * W^T (pi(1) - pi(s_neu))
  geo      PCA-64 kNN-graph geodesic, spline-smoothed:        delta(s) = lam * W^T (pi(s_neu + s (1 - s_neu)) - pi(s_neu))
           s in [0,1] = fraction of the way from the neutral point to the high end (chord length for chord, arc length for geo).
           chord vs geo: SAME endpoints and gain, only the route differs (single factor = path shape).
  geo_null CONTROL: the same path construction fitted on line-null data (bends = fit noise only), same lam.
  ideea1   last-position rule (IDEEA site), K = 1: mean RAW per-token residual of D+_t tokens minus that of D-_t tokens.
  ideea    IDEEA-CAA residual nearest_pos_neg, KMeans(K=4) separately on RAW per-token residuals of D+_t and D-_t (no question
           pairing or content removal, as IDEEA), direction selected from the current state at the last position of every
           forward call. ideea1 -> ideea changes one factor (K = 1 -> 4, i.e. state dependence); pv -> ideea1 changes the position
           rule and the activation source (stated, two factors).
pi(s): s in [0,1] = arc-length fraction from the lowest-intensity centroid (mean of D-_t rows, level -3) to the highest
(mean of D+_t rows at level 3, strong). Graph: kNN (k=10) over the trait's ladder rows in PCA-64, edge weight = euclid^POWER
(POWER=2 primary: the density-favouring approximation of the density metric) built on DENOISED rows (each row replaced by
the mean of its 20 nearest neighbours, twice; amendment-free design choice fixed before data, see test_geom.py); endpoints = nearest rows to the two centroids;
the node sequence (with the centroids prepended/appended) is smoothed coordinate-wise by a least-squares cubic in cumulative
chord length (a cubic spline without interior knots; GCV make_smoothing_spline left 0.77 straightness on a planted straight line
in test_geom.py, i.e. noise wander; 'gcv' kept as a secondary option) and resampled at 201 points uniform in arc length. s_neu = arc position nearest the neutral centroid.
"""
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from geom import demean_by, pca_fit, knn_graph, ladder, null_line

POS_THR, NEG_THR = 0.75, 0.25


def labels(meta, t):
    P = meta[f"P_{t}"]; tr = meta["trait"] == t
    return tr & (meta["level"] >= 1) & (P >= POS_THR), tr & (meta["level"] <= -1) & (P <= NEG_THR)


def _arc(Y):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Y, axis=0), axis=1))]
    return d / d[-1] if d[-1] > 0 else d


def denoise(Z, m=20, rounds=2):
    """local-mean shrinkage toward the data's principal manifold: each point -> mean of its m nearest neighbours, repeated.
    Without it the graph path hops between single noisy rows and its bends are noise (test_geom.py: straightness .6 on a
    straight line with 3 path nodes)."""
    for _ in range(rounds):
        _, ind = NearestNeighbors(n_neighbors=min(m, len(Z))).fit(Z).kneighbors(Z)
        Z = Z[ind].mean(1)
    return Z


def fit_path(Z, lev, Dp, Dn, k=10, power=2.0, M=201, m_denoise=20, smooth="cubic"):
    lo = Z[Dn & (lev == -3)].mean(0) if (Dn & (lev == -3)).any() else Z[Dn].mean(0)
    hi = Z[Dp & (lev == 3)].mean(0) if (Dp & (lev == 3)).any() else Z[Dp].mean(0)
    Z = denoise(Z, m_denoise) if m_denoise else Z
    a, b = np.argmin(((Z - lo) ** 2).sum(1)), np.argmin(((Z - hi) ** 2).sum(1))
    for kk in (k, 2 * k, 4 * k, 8 * k, len(Z) - 1):        # grow k until the endpoints are connected (k_used is reported)
        G = knn_graph(Z, min(kk, len(Z) - 1), power)
        dist, pred = dijkstra(G, directed=False, indices=[a], return_predecessors=True)
        if np.isfinite(dist[0, b]): break
    if not np.isfinite(dist[0, b]):
        raise RuntimeError("endpoints disconnected in the kNN graph")
    node, seq = b, []
    while node != a and node >= 0:
        seq.append(node); node = pred[0, node]
    seq = [a] + seq[::-1]
    Y = np.vstack([lo, Z[seq], hi])
    s = _arc(Y); keep = np.r_[True, np.diff(s) > 1e-9]; Y, s = Y[keep], s[keep]
    if smooth == "cubic" and len(s) >= 5:     # cubic spline with no interior knot = least-squares cubic in arc length
        V = np.vander(s, 4, increasing=True); Ys = V @ np.linalg.lstsq(V, Y, rcond=None)[0]
    elif smooth == "gcv" and len(s) >= 5:
        Ys = np.stack([make_smoothing_spline(s, Y[:, j])(s) for j in range(Y.shape[1])], 1)
    else:
        Ys = Y
    s2 = _arc(Ys); grid = np.linspace(0, 1, M)
    P = np.stack([np.interp(grid, s2, Ys[:, j]) for j in range(Ys.shape[1])], 1)
    P = np.stack([np.interp(grid, _arc(P), P[:, j]) for j in range(P.shape[1])], 1)   # re-uniformise after smoothing
    return dict(pi=P, grid=grid, nodes=np.array(seq), lo=lo, hi=hi, k_used=int(min(kk, len(Z) - 1)), raw_len=float(np.linalg.norm(np.diff(Y, axis=0), axis=1).sum()))


def path_curvature(P):
    """straightness of the fitted path = chord / length (1 = straight)"""
    L = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    return float(np.linalg.norm(P[-1] - P[0]) / L) if L > 0 else np.nan


def _neutral_index(P, Zneu):
    return int(np.argmin(((P - Zneu) ** 2).sum(1)))


def fit_arms(Xraw, meta, t, train_rows, k=10, power=2.0, K_ideea=4, seed=0, tok_pos=None, tok_neg=None):
    """Xraw: [N, H] response-mean activations at layer L (all rows). tok_pos / tok_neg: RAW per-token response residuals at L of
    D+_t / D-_t TRAIN rows (steer.py collects them), used for the IDEEA arms (IDEEA clusters the same kind of state it later
    selects on; review finding 3). Returns arm definitions (numpy) + diagnostics."""
    tr = train_rows
    Xd = demean_by(Xraw[tr], meta["q"][tr])
    mu, W = pca_fit(Xd, 64)
    m = {kk: v[tr] for kk, v in meta.items()}
    Dp, Dn = labels(m, t)
    lad = ladder(m, t)
    Z = (Xd - mu) @ W.T
    u = Xd[Dp].mean(0) - Xd[Dn].mean(0)
    f = fit_path(Z[lad], m["level"][lad] * (m["trait"][lad] == t), Dp[lad], Dn[lad], k=k, power=power)
    neu = Z[lad & (m["trait"] == "none")].mean(0); i_neu = _neutral_index(f["pi"], neu)
    # geo_null control (review finding 9): the SAME path construction on line-null data (ladder contrasts made collinear,
    # residuals permuted within question) -> a path whose bends are pure fit noise, with the same noise level.
    Xn = null_line(Xd, m, t, np.random.default_rng(seed), two_sided=True); Zn = (Xn - mu) @ W.T
    fn = fit_path(Zn[lad], m["level"][lad] * (m["trait"][lad] == t), Dp[lad], Dn[lad], k=k, power=power)
    i_neu_n = _neutral_index(fn["pi"], Zn[lad & (m["trait"] == "none")].mean(0))
    out = dict(trait=t, W=W.astype(np.float32), pi=f["pi"].astype(np.float32), grid=f["grid"], s_neu=float(f["grid"][i_neu]), i_neu=i_neu,
               pi_null=fn["pi"].astype(np.float32), i_neu_null=i_neu_n, pv=u.astype(np.float32),
               n_pos=int(Dp.sum()), n_neg=int(Dn.sum()), path_straightness=path_curvature(f["pi"][i_neu:]),
               path_straightness_full=path_curvature(f["pi"]), path_straightness_null=path_curvature(fn["pi"][i_neu_n:]),
               path_nodes=int(len(f["nodes"])))
    if tok_pos is not None:
        kmp = KMeans(n_clusters=K_ideea, random_state=seed, n_init=10).fit(tok_pos)
        kmn = KMeans(n_clusters=K_ideea, random_state=seed, n_init=10).fit(tok_neg)
        out.update(ideea_pos=kmp.cluster_centers_, ideea_neg=kmn.cluster_centers_,
                   ideea1_pos=tok_pos.mean(0, keepdims=True), ideea1_neg=tok_neg.mean(0, keepdims=True))
    return out


def delta(arm, A, s=None, lam=1.0, alpha=None):
    """displacement in model space (float32 [H]) for arm 'pv' (alpha), 'chord' / 'geo' / 'geo_null' (s in [0,1] = fraction of the
    way from the neutral point to the high end, gain lam). geo and chord coincide at s = 1."""
    if arm == "pv":
        return (alpha * A["pv"]).astype(np.float32)
    if arm == "geo_null":
        P, i0 = A["pi_null"], int(A["i_neu_null"])
    else:
        P, i0 = A["pi"], int(A["i_neu"])
    if arm in ("geo", "geo_null"):
        j = int(round(i0 + s * (len(P) - 1 - i0)))
        d = P[j] - P[i0]
    elif arm == "chord":
        d = s * (P[-1] - P[i0])
    else:
        raise ValueError(arm)
    return (lam * d @ A["W"]).astype(np.float32)
