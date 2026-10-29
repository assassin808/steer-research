"""Synthetic sanity checks of geom.py / paths.py on the real crossed design shape (CPU, ~minutes), plus the IDEEA
equivalence check against ~/ext/IDEEA (read-only import). Not a power study (that is power/ + results/power_curve.md).
usage: python test_geom.py
"""
import sys, os, numpy as np, torch
import geom as G
from paths import fit_path, path_curvature

rng = np.random.default_rng(0)
Q, S, d, TR = 40, 4, 128, ["a", "b"]


def make(shape="line", snr=1.0, hetero=0.0, same_dir=False):
    """rows: Q questions x (neutral + 2 traits x 4 levels) x S samples. question means ~ N(0, 4); noise N(0,1) per coord.
    trait ladder positions gamma(level) in a random plane, amplitude set by between-level 'snr' (class SNR definition)."""
    B = np.linalg.qr(rng.standard_normal((d, 6)))[0].T
    X, meta = [], dict(q=[], trait=[], level=[], sample=[])
    mu_q = 2 * rng.standard_normal((Q, d))
    rot = {}
    for qi in range(Q):
        th = hetero * rng.standard_normal()
        rot[qi] = th
    amp = np.sqrt(snr * d / np.mean(np.linspace(-1, 1, 5) ** 2))
    def gam(t_i, l, qi):
        u, v = (B[0], B[1]) if (same_dir or t_i == 0) else (B[2], B[3])
        x = (l + 2) / 4.0
        if shape == "line":
            p = (x - .5) * u
        else:  # 90-degree arc of the same length
            th = x * np.pi / 2; R = 1 / (np.pi / 2)
            p = R * (np.sin(th) * u + (1 - np.cos(th)) * v) - R * (np.sin(np.pi / 4) * u + (1 - np.cos(np.pi / 4)) * v)
        w = B[4]; th = rot[qi]                       # per-question rotation of the direction toward w (input dependence)
        pu = p @ u
        p = p + pu * ((np.cos(th) - 1) * u + np.sin(th) * w)
        return amp * p
    for qi in range(Q):
        cells = [("none", 0)] + [(t, l) for t in TR for l in (-2, -1, 1, 2)]
        for t, l in cells:
            for s in range(S):
                g = np.zeros(d) if t == "none" else gam(TR.index(t), l, qi)
                X.append(mu_q[qi] + g + rng.standard_normal(d))
                meta["q"].append(qi); meta["trait"].append(t); meta["level"].append(l); meta["sample"].append(s)
    meta = {k: np.array(v) for k, v in meta.items()}
    return np.array(X), meta


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    return cond


ok = True
for shape in ("line", "arc"):
    X, m = make(shape, snr=0.5)
    Xd = G.demean_by(X, m["q"]); lad = G.ladder(m, "a")
    c = G.curve_stats(Xd[lad], m["level"][lad], m["sample"][lad])
    nl = [G.curve_stats(G.null_line(Xd[lad], m["level"][lad], m["q"][lad], rng), m["level"][lad], m["sample"][lad]) for _ in range(30)]
    p = np.mean([x["offline"] >= c["offline"] for x in nl])
    print(shape, {k: round(v, 3) for k, v in c.items()}, "null offline p95", round(np.percentile([x["offline"] for x in nl], 95), 3), "p", p)
    if shape == "line":
        ok &= check("line: offline ~ 0 (<.05) and not significant", c["offline"] < .05 and p > .05)
        ok &= check("line: straightness > .9", c["straight"] > .9)
    else:
        ok &= check("arc90: offline > 3x line-null p95 and p < .05", c["offline"] > 3 * np.percentile([x["offline"] for x in nl], 95) and p < .05)
        ok &= check("arc90: straightness < line", c["straight"] < .97)
    pr, o1 = G.ord1_relabel_p(Xd[lad], m["level"][lad], m["sample"][lad])
    ok &= check(f"{shape}: graded order detected (ord1 p={pr:.3f})", pr < .05)
    # path fit on PCA-64 of the ladder
    Dp = (m["trait"] == "a") & (m["level"] >= 1); Dn = (m["trait"] == "a") & (m["level"] <= -1)
    mu, W = G.pca_fit(Xd, 64); Z = (Xd - mu) @ W.T
    f = fit_path(Z[lad], m["level"][lad], Dp[lad], Dn[lad])
    print(shape, "geodesic path straightness", round(path_curvature(f["pi"]), 3), "nodes", len(f["nodes"]))
    if shape == "arc":
        ok &= check("arc: geodesic path straightness < line-case straightness - .02", path_curvature(f["pi"]) < line_str - .02)
    else:
        line_str = path_curvature(f["pi"])
        ok &= check("line: geodesic path straightness >= .85 (denoised graph; noise floor)", line_str >= .85)

# input dependence
X, m = make("line", snr=0.5, hetero=0.0); Xd = G.demean_by(X, m["q"]); lad = G.ladder(m, "a")
r0 = G.input_dependence(Xd[lad], m["level"][lad], m["q"][lad], m["sample"][lad], nboot=100)
X, m = make("line", snr=0.5, hetero=1.0); Xd = G.demean_by(X, m["q"]); lad = G.ladder(m, "a")
r1 = G.input_dependence(Xd[lad], m["level"][lad], m["q"][lad], m["sample"][lad], nboot=100)
print("D1 global", r0, "\nD1 hetero", r1)
ok &= check("D1 ~ 1 for one global direction (CI covers .95)", r0["D1_ci"][1] > .95)
ok &= check("D1 < .8 for per-question rotations of sd 1 rad", r1["D1_ci"][1] < .8)

# separation
for same in (True, False):
    X, m = make("line", snr=0.5, same_dir=same); Xd = G.demean_by(X, m["q"])
    m["Dpos"] = (m["trait"] != "none") & (m["level"] >= 1)
    r = G.separation(Xd, m, TR, nsplit=20)["a|b"]
    print("separation same_dir" if same else "separation orthogonal", round(r, 3))
    ok &= check("separation r ~ 1 if same direction" if same else "separation r ~ 0 if orthogonal", (r > .9) if same else (abs(r) < .15))

# IDEEA equivalence (nearest_pos_neg) against the original implementation
try:
    from lm import _npn
    for k in [k for k in sys.modules if k == "common" or k.startswith("common.")]: del sys.modules[k]   # IDEEA has its own 'common'
    sys.path.insert(0, os.path.expanduser("~/ext/IDEEA"))
    from methods.ideea_caa import _select_nearest_pos_neg
    x = torch.randn(7, 32, dtype=torch.float64); pc = torch.randn(4, 32, dtype=torch.float64); nc = torch.randn(4, 32, dtype=torch.float64)
    a = _select_nearest_pos_neg(x, None, None, pc, nc, None); b = _npn(x, pc, nc)
    ok &= check("IDEEA nearest_pos_neg identical to ~/ext/IDEEA", bool(torch.allclose(a, b)))
except Exception as e:
    print("IDEEA import failed:", repr(e)); ok = False
print("ALL PASS" if ok else "SOME FAILED")
