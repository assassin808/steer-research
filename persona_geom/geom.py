"""Geometry statistics for the persona study (prereg: results/persona_geom_prereg.md, section 5).

Conventions. X: [n, d] rows (activations at one layer, or text embeddings). meta arrays (length n):
  q (int question id), trait (str, 'none' for the neutral cell), level (int -2..2; neutral = 0), sample (int), P (judge score of the
  trait under study, in [0,1]). For trait t the 'ladder' rows are trait==t rows plus the shared neutral rows (level 0).
Every statistic is oriented "larger = more of the structure named". Split halves are within (question, cell) by sample index,
so both halves contain every question and every paraphrase pattern (weekday amendment A3: never split by composition).
"""
import itertools, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

LV = np.array([-2, -1, 0, 1, 2])


# ------------------------------------------------------------------ basics
def demean_by(X, g):
    """subtract the per-group (question) mean. In the balanced crossed design every question has the same cell composition,
    so this removes question content without biasing any cell contrast."""
    X = X.astype(np.float64, copy=True)
    for v in np.unique(g):
        m = g == v; X[m] -= X[m].mean(0)
    return X


def pca_fit(X, d=64):
    mu = X.mean(0); _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:d]


def ladder(meta, t):
    return (meta["trait"] == t) | (meta["trait"] == "none")


def half_masks(sample):
    """all balanced bipartitions of the sample indices {0..S-1} (S even), A containing sample 0"""
    S = int(sample.max()) + 1
    parts = [c for c in itertools.combinations(range(S), S // 2) if 0 in c] if S >= 2 else []
    return [np.isin(sample, c) for c in parts]


def level_centroids(X, lev):
    return np.stack([X[lev == l].mean(0) for l in LV])


def cross_gram(cA, cB):
    a, b = cA - cA.mean(0), cB - cB.mean(0)
    return 0.5 * (a @ b.T + b @ a.T)


# ------------------------------------------------------------------ H1: shape of the intensity curve
POLY = None


def _poly(L=5):
    t = np.linspace(0, 1, L); V = np.vander(t, 3, increasing=True)
    Qm, _ = np.linalg.qr(V); Qm = Qm * np.sign(Qm[-1])
    return Qm[:, 1], Qm[:, 2]          # orthonormal linear and quadratic contrasts


def curve_stats(X, lev, sample):
    """X: ladder rows (demeaned). Returns split-half-corrected centroid statistics (power-curve prereg C1-C4 + straightness).
    offline  = 1 - top-1 eigen share of the cross-Gram of centred level centroids (0 for any straight line, also unequal steps)
    quadperp = <P_perp b2^A, P_perp b2^B> / <b1^A, b1^B>  (quadratic-in-level power orthogonal to the line direction)
    turn_med = median split-half corrected turning angle at the 3 interior vertices (deg)
    straight = sqrt(chord^2) / sum_k sqrt(step_k^2), cross-estimated (1 = straight and monotone)
    ord1     = share of the cross-Gram trace along the linear level contrast (graded order; straight line -> high)"""
    p1, p2 = _poly(len(LV))
    c = level_centroids(X, lev)
    b1 = p1 @ c; u = b1 / (np.linalg.norm(b1) + 1e-300)
    res = dict(offline=[], quadperp=[], turn=[], straight=[], ord1=[], snr_between=[])
    for A in half_masks(sample):
        cA, cB = level_centroids(X[A], lev[A]), level_centroids(X[~A], lev[~A])
        M = cross_gram(cA, cB); ev = np.sort(np.linalg.eigvalsh(M))[::-1]; tot = np.trace(M)
        res["offline"].append(1 - ev[0] / tot if tot > 0 else np.nan)
        res["ord1"].append(float(p1 @ M @ p1) / tot if tot > 0 else np.nan)
        b1A, b1B, b2A, b2B = p1 @ cA, p1 @ cB, p2 @ cA, p2 @ cB
        Pp = lambda v: v - (v @ u) * u
        den = b1A @ b1B
        res["quadperp"].append(float(Pp(b2A) @ Pp(b2B)) / den if den > 0 else np.nan)
        dA, dB = np.diff(cA, axis=0), np.diff(cB, axis=0)
        ii = (dA * dB).sum(1)
        ang = []
        for k in range(len(LV) - 2):
            num = 0.5 * (dA[k] @ dB[k + 1] + dB[k] @ dA[k + 1]); dd = ii[k] * ii[k + 1]
            ang.append(np.degrees(np.arccos(np.clip(num / np.sqrt(dd), -1, 1))) if dd > 0 and ii[k] > 0 else np.nan)
        res["turn"].append(np.nanmedian(ang) if np.isfinite(ang).any() else np.nan)
        ch = (cA[-1] - cA[0]) @ (cB[-1] - cB[0])
        res["straight"].append(np.sqrt(max(ch, 0)) / max(np.sqrt(np.maximum(ii, 0)).sum(), 1e-300))
    out = {k: float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else np.nan for k, v in res.items() if k != "snr_between"}
    W = ((X - c[np.searchsorted(LV, lev)]) ** 2).sum(1).mean()
    out["snr"] = float(((c - c.mean(0)) ** 2).sum(1).mean() / W)       # class SNR, same definition as power_curve_prereg
    return out


def ord1_relabel_p(X, lev, sample):
    """exact order-relabel null for ord1 (all 5! level relabelings)"""
    obs = curve_stats(X, lev, sample)["ord1"]; vals = []
    for perm in itertools.permutations(range(len(LV))):
        lv2 = LV[np.array(perm)][np.searchsorted(LV, lev)]
        vals.append(curve_stats(X, lv2, sample)["ord1"])
    vals = np.array(vals)
    return float(np.mean(vals >= obs - 1e-12)), obs


def null_line(X, lev, q, rng):
    """practical flat null (power-curve prereg): level centroids projected on the top PC of the observed centroids
    (keeps the observed, possibly unequal, spacing along the line); residuals permuted across rows WITHIN question."""
    c = level_centroids(X, lev); m = c.mean(0)
    _, _, Vt = np.linalg.svd(c - m, full_matrices=False); u = Vt[0]
    cl = m + np.outer((c - m) @ u, u)
    idx = np.searchsorted(LV, lev); R = X - c[idx]
    perm = np.arange(len(X))
    for v in np.unique(q):
        w = np.where(q == v)[0]; perm[w] = rng.permutation(w)
    return cl[idx] + R[perm]


def null_simplex(X, lev, q, rng):
    """'clustered' reference: level centroids at mutually orthogonal random directions with the observed total spread"""
    c = level_centroids(X, lev); m = c.mean(0); s2 = ((c - m) ** 2).sum()
    E = np.linalg.qr(rng.standard_normal((X.shape[1], len(LV))))[0].T
    E = E - E.mean(0); E *= np.sqrt(s2 / (E ** 2).sum())
    idx = np.searchsorted(LV, lev); R = X - c[idx]
    perm = np.arange(len(X))
    for v in np.unique(q):
        w = np.where(q == v)[0]; perm[w] = rng.permutation(w)
    return m + E[idx] + R[perm]


# ------------------------------------------------------------------ H2: input dependence of the intensity direction
def input_dependence(X, lev, q, sample, lo=-2, hi=2, nboot=1000, rng=None):
    """per-question intensity direction d_q = c_{q,hi} - c_{q,lo}, split halves by sample.
    D1   = sum_q <d_q^A, u_-q><d_q^B, u_-q> / sum_q <d_q^A, d_q^B>, u_-q = unit leave-one-out global direction
           (share of reliable per-question direction energy on ONE global direction; magnitude-free; 1 = input-independent)
    G    = mean_{q != q'} <d_q^A, d_q'^B> / mean_q <d_q^A, d_q^B> (global-vector share; < 1 also from magnitude spread)
    Bootstrap CI over questions."""
    rng = rng or np.random.default_rng(0)
    Q = np.unique(q); dA, dB = [], []
    masks = half_masks(sample)
    for v in Q:
        mq = q == v; a_l, b_l = [], []
        for A in masks:
            f = lambda M, l: X[mq & M & (lev == l)].mean(0)
            a_l.append(f(A, hi) - f(A, lo)); b_l.append(f(~A, hi) - f(~A, lo))
        dA.append(a_l); dB.append(b_l)
    dA, dB = np.array(dA), np.array(dB)            # [Q, nsplit, d]

    def stat(ix):
        a, b = dA[ix], dB[ix]; tot = (a + b).sum(0)  # [nsplit, d]
        num = den = 0.0; within = []; cross = []
        for j in range(len(ix)):
            u = tot - a[j] - b[j]; u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-300)
            num += ((a[j] * u).sum(-1) * (b[j] * u).sum(-1)).mean(); w = (a[j] * b[j]).sum(-1).mean(); den += w; within.append(w)
        A_, B_ = a.mean(0), b.mean(0)                # cross term via sums: sum_{q!=q'} <a_q, b_q'> = <sum a, sum b> - sum <a_q, b_q>
        n = len(ix); cr = (((a.sum(0) * b.sum(0)).sum(-1) - (a * b).sum(-1).sum(0)) / (n * (n - 1))).mean()
        return num / den, cr / np.mean(within)
    D1, G = stat(np.arange(len(Q)))
    bs = np.array([stat(rng.integers(0, len(Q), len(Q))) for _ in range(nboot)])
    return dict(D1=float(D1), D1_ci=[float(x) for x in np.percentile(bs[:, 0], [2.5, 97.5])],
                G=float(G), G_ci=[float(x) for x in np.percentile(bs[:, 1], [2.5, 97.5])], nq=int(len(Q)))


# ------------------------------------------------------------------ H3: trait separation
def trait_dirs_split(X, meta, traits, qmaskA):
    """w_t = mean(D+_t) - mean(neutral rows), on the questions in qmaskA (row mask) and on the rest."""
    out = {}
    for half, M in (("A", qmaskA), ("B", ~qmaskA)):
        neu = X[M & (meta["trait"] == "none")].mean(0)
        out[half] = {t: X[M & meta["Dpos"] & (meta["trait"] == t)].mean(0) - neu for t in traits}
    return out


def separation(X, meta, traits, nsplit=50, rng=None):
    """disattenuated cosine between neutral-anchored trait directions, from question-half splits.
    r(t,t') = mean_splits 0.5(<wA_t,wB_t'> + <wB_t,wA_t'>) / sqrt(<wA_t,wB_t><wA_t',wB_t'>). 1 = same direction up to noise."""
    rng = rng or np.random.default_rng(0)
    Q = np.unique(meta["q"]); R = {}
    num = {}; dia = {t: [] for t in traits}
    for s in range(nsplit):
        qa = rng.permutation(Q)[: len(Q) // 2]; M = np.isin(meta["q"], qa)
        w = trait_dirs_split(X, meta, traits, M)
        for t in traits: dia[t].append(w["A"][t] @ w["B"][t])
        for t, u in itertools.combinations(traits, 2):
            num.setdefault((t, u), []).append(0.5 * (w["A"][t] @ w["B"][u] + w["B"][t] @ w["A"][u]))
    for (t, u), v in num.items():
        den = np.sqrt(max(np.mean(dia[t]), 1e-300) * max(np.mean(dia[u]), 1e-300))
        R[f"{t}|{u}"] = float(np.mean(v) / den)
    return R


def separation_boot(X, meta, traits, pairs, nboot=200, nsplit=10, rng=None):
    rng = rng or np.random.default_rng(1)
    Q = np.unique(meta["q"]); bs = []
    for b in range(nboot):
        pick = rng.integers(0, len(Q), len(Q))
        rows = np.concatenate([np.where(meta["q"] == Q[i])[0] for i in pick])
        newq = np.repeat(np.arange(len(Q)), [int((meta["q"] == Q[i]).sum()) for i in pick])
        m2 = {k: v[rows] for k, v in meta.items()}; m2["q"] = newq
        r = separation(X[rows], m2, traits, nsplit=nsplit, rng=rng)
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
    for a in range(len(Z)):
        pp = [j for j in P[ip[a]] if q[j] != q[a]][:k]; nn_ = [j for j in N[iN[a]] if q[j] != q[a]][:k]
        V[a] = Z[pp].mean(0) - Z[nn_].mean(0)
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
    """Isomap geodesic / Euclidean distance between the nodes nearest the level -2 and +2 centroids (1 = straight)."""
    G = knn_graph(Z, k); nc, _ = connected_components(G, directed=False)
    a = np.argmin(((Z - Z[lev == -2].mean(0)) ** 2).sum(1)); b = np.argmin(((Z - Z[lev == 2].mean(0)) ** 2).sum(1))
    d = dijkstra(G, directed=False, indices=[a])[0, b]
    return dict(geo_ratio=float(d / np.linalg.norm(Z[a] - Z[b])) if np.isfinite(d) else np.nan, ncomp=int(nc))


# ------------------------------------------------------------------ H4: IDEEA cluster connectivity
def ideea_connectivity(Zp, qsrc, qid, K=4, k=10, nnull=20, rng=None):
    """KMeans(K) on D+ rows (IDEEA's own clustering, no pairing). Reports AMI of clusters with question source trait and with
    question id (content tracking), and the between-cluster share of kNN edges relative to a flat Gaussian null with D+'s
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
    return dict(ami_src=float(adjusted_mutual_info_score(qsrc, lab)), ami_qid=float(adjusted_mutual_info_score(qid, lab)),
                edge_share=obs, edge_null=float(np.mean(nul)), edge_null_p05=float(np.percentile(nul, 5)),
                edge_ratio=float(obs / np.mean(nul)), sizes=np.bincount(lab, minlength=K).tolist())
