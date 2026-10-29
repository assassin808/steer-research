"""Synthetic sanity checks of geom.py / paths.py on the real crossed-design shape (CPU, ~minutes), plus the IDEEA equivalence check
against ~/ext/IDEEA (read-only import). Not a power study (that is power/ + results/power_curve.md).
Design: Q questions x (bare, placebo x 4 words, traits a,b x {-3, 1, 2, 3, 4}) x S samples, question means N(0, 4), noise N(0, 1).
Every INSTRUCTED row also gets a word-dependent frame offset (the 'instruction present' confound of review finding 2); the
placebo-referenced ladder must cancel it.
usage: python test_geom.py
"""
import sys, os, numpy as np, torch
import geom as G
from paths import fit_path, path_curvature

rng = np.random.default_rng(0)
Q, S, d, TR = 40, 4, 128, ["a", "b"]
POS = {-3: -1.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}       # ladder position of each trait level (placebo-referenced level 0 = 0)


def make(shape="line", snr=0.5, hetero=0.0, same_dir=False, frame=3.0, kappa=0.0, S=S, Q=Q, kslope=0.0):
    """kappa: question x cell random effect, E||cell||^2 = kappa x W_eps / 5 (power/designcheck.py unit); kslope: cell variance
    multiplied by (1 + kslope * |level|) (level-dependent cell effects)"""
    B = np.linalg.qr(rng.standard_normal((d, 8)))[0].T
    off = {w: frame * (w / 3) ** 0.5 * B[6] + frame * B[7] for w in (1, 2, 3, 4)}   # frame offsets: big, non-linear in word
    amp = np.sqrt(snr * d / 0.35)
    rot = hetero * rng.standard_normal(Q)
    mu_q = 2 * rng.standard_normal((Q, d))

    def gam(ti, x, qi):
        u, v = (B[0], B[1]) if (same_dir or ti == 0) else (B[2], B[3])
        if shape == "line":
            p = x * u
        else:  # 90-degree arc over x in [0, 1]; negative side continues the arc backwards
            th = x * np.pi / 2; R = 2 / np.pi
            p = R * (np.sin(th) * u + (1 - np.cos(th)) * v)
        pu = p @ u; th = rot[qi]
        return amp * (p + pu * ((np.cos(th) - 1) * u + np.sin(th) * B[4]))
    X, meta = [], dict(q=[], trait=[], level=[], sample=[])
    cells = [("none", 0)] + [("placebo", w) for w in (1, 2, 3, 4)] + [(t, l) for t in TR for l in (-3, 1, 2, 3, 4)]
    for qi in range(Q):
        for t, l in cells:
            ce = np.sqrt(kappa / 5 * (1 + kslope * abs(l))) * rng.standard_normal(d)
            for s in range(S):
                x = mu_q[qi] + ce + rng.standard_normal(d)
                if t != "none": x = x + off[abs(l)]
                if t in TR: x = x + gam(TR.index(t), POS[l], qi)
                X.append(x); meta["q"].append(qi); meta["trait"].append(t); meta["level"].append(l); meta["sample"].append(s)
    return np.array(X), {k: np.array(v) for k, v in meta.items()}


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    return cond


ok = True
line_str = None
for shape in ("line", "arc"):
    X, m = make(shape)
    Xd = G.demean_by(X, m["q"]); lad = G.ladder(m, "a"); mL = {k: v[lad] for k, v in m.items()}; Xl = Xd[lad]
    c = G.curve_stats(Xl, mL, "a")
    nl = [G.curve_stats(G.null_line(Xl, mL, "a", rng), mL, "a") for _ in range(30)]
    p95 = np.percentile([x["offline"] for x in nl], 95); p = np.mean([x["offline"] >= c["offline"] for x in nl])
    print(shape, {k: round(v, 3) for k, v in c.items() if k != "_M"}, "null offline p95", round(p95, 4), "p", p)
    if shape == "line":
        ok &= check("line (+ frame confound): offline ~ 0 (<.02) and not significant", c["offline"] < .02 and p > .05)
        ok &= check("line: straightness > .9", c["straight"] > .9)
        # the uncorrected ladder (bare neutral as level 0) must look curved -> shows the confound is real in this synthetic
        raw = np.stack([Xl[mL["trait"] == "none"].mean(0)] + [Xl[(mL["trait"] == "a") & (mL["level"] == k)].mean(0) for k in (1, 2, 3, 4)])
        M = G.cross_gram(raw, raw); ev = np.sort(np.linalg.eigvalsh(M))[::-1]
        ok &= check(f"confound demo: bare-anchored ladder offline {1 - ev[0] / np.trace(M):.3f} > .02 (vs placebo-referenced ~0)", 1 - ev[0] / np.trace(M) > .02)
    else:
        ok &= check("arc90: offline > 2x line-null p95 and p < .05 (FL cell null is wider under curvature)", c["offline"] > 2 * p95 and p < .05)
        ok &= check("arc90: straightness < .97", c["straight"] < .97)
    ok &= check(f"{shape}: graded order (ord1 relabel p={G.ord1_relabel_p(c):.3f} <= .1)", G.ord1_relabel_p(c) <= .1)
    c2 = G.curve_stats(Xl, mL, "a", two_sided=True)
    print(shape, "two-sided", {k: round(v, 3) for k, v in c2.items() if k != "_M"})
    Dp = (mL["trait"] == "a") & (mL["level"] >= 1); Dn = (mL["trait"] == "a") & (mL["level"] == -3)
    mu, W = G.pca_fit(Xd, 64); Z = (Xl - mu) @ W.T
    f = fit_path(Z, mL["level"] * (mL["trait"] == "a"), Dp, Dn); ps = path_curvature(f["pi"])
    print(shape, "geodesic path straightness", round(ps, 3), "nodes", len(f["nodes"]))
    if shape == "line":
        line_str = ps; ok &= check("line: geodesic path straightness >= .85 (noise floor)", ps >= .85)
    else:
        ok &= check("arc: geodesic path straightness < line case - .02", ps < line_str - .02)

# ---- amendment A1: cell-level tests. (a) fast D-level statistics equal the row-level ones
X, m = make("arc", kappa=0.55); Xd = G.demean_by(X, m["q"]); lad = G.ladder(m, "a"); mL = {k: v[lad] for k, v in m.items()}
for two in (False, True):
    lc = G.ladder_cells(Xd[lad], mL, "a", two); a1 = G.stats_D(lc["D"], lc); a2 = G.curve_stats(Xd[lad], mL, "a", two)
    ok &= check(f"D-level stats == row-level stats (two_sided={two})", all(abs(a1[k] - a2[k]) < 1e-9 for k in ("offline", "quadperp", "ord1")))
# (b) calibration of the primary test (max-p of cell permutation and cell sign-flip, Freedman-Lane) on straight lines
def fpr(nsim=200, reps=99, **kw):
    ps = []
    for _ in range(nsim):
        X, m = make("line", frame=0.0, **kw); Xd = G.demean_by(X, m["q"]); lad = G.ladder(m, "a"); mL = {k: v[lad] for k, v in m.items()}
        lc = G.ladder_cells(Xd[lad], mL, "a"); ps.append(G.line_test(lc["D"], lc, rng, reps, keys=("quadperp",)))
    P = lambda key: np.array([p[key] for p in ps])
    return {k: (float(np.mean(P(k) <= .02)), float(np.mean(P(k) <= .05))) for k in ("p_quadperp", "p_quadperp_cell", "p_quadperp_cellwild")}
for kw in (dict(kappa=0.0), dict(kappa=0.55, S=2, Q=60), dict(kappa=1.0, S=2, Q=60), dict(kappa=1.0, kslope=1.0, S=2, Q=60)):
    f = fpr(**kw); print("line FPR (p<=.02, p<=.05)", kw, f)
    ok &= check(f"primary test FPR at .05 <= .09 and at .02 <= .045 ({kw}; 200 sims)", f["p_quadperp"][1] <= .09 and f["p_quadperp"][0] <= .045)
# (c) power: arc90 with cell effects, S = 2
X, m = make("arc", kappa=1.0, frame=0.0, S=2, Q=60); Xd = G.demean_by(X, m["q"]); lad = G.ladder(m, "a"); mL = {k: v[lad] for k, v in m.items()}
lc = G.ladder_cells(Xd[lad], mL, "a"); lt = G.line_test(lc["D"], lc, rng, 999); print("arc90 + kappa 1, S 2:", lt)
ok &= check("arc90 + cell effects detected at p <= .02/6 by the primary test", lt["p_quadperp"] <= .02 / 6)
# (d) Holm
h = G.holm(dict(a=.001, b=.004, c=.03, d=.2, e=.5, f=.01))
ok &= check(f"Holm adjusted {h}", abs(h["a"] - .006) < 1e-12 and abs(h["b"] - .02) < 1e-12 and abs(h["f"] - .04) < 1e-12 and abs(h["c"] - .09) < 1e-12)
# (e) kappa estimator, designcheck unit, S = 2 and 4
for SS, QQ in ((2, 120), (4, 40)):
    X, m = make("line", kappa=0.55, frame=0.0, S=SS, Q=QQ); Xd = G.demean_by(X, m["q"]); lad = G.ladder(m, "a"); mL = {k: v[lad] for k, v in m.items()}
    kh = G.cell_kappa(Xd[lad], mL, "a", C=15); print("kappa .55 planted, S", SS, "Q", QQ, "->", round(kh["kappa"], 3))
    ok &= check(f"cell_kappa recovers .55 within +-.15 (S={SS})", abs(kh["kappa"] - .55) < .15)
# (f) planted power runs and is high at the synthetic SNR (6 sims only; a smoke check, not a power study)
X, m = make("line", kappa=1.0, frame=0.0, S=2, Q=60); Xd = G.demean_by(X, m["q"]); lad = G.ladder(m, "a"); mL = {k: v[lad] for k, v in m.items()}
lc = G.ladder_cells(Xd[lad], mL, "a")
pp = G.planted_power(lc["D"], lc, rng, alpha=.02 / 6, nsim=6, plane="noise"); print("planted arc60 power (noise plane)", pp)
ok &= check("planted arc60 power >= .8 at synthetic snr .5, kappa 1, S 2", pp["power"] >= .8)

X, m = make("line", hetero=0.0); Xd = G.demean_by(X, m["q"]); lad = G.ladder(m, "a"); mL = {k: v[lad] for k, v in m.items()}
r0 = G.input_dependence(Xd[lad], mL, "a", nboot=100)
X, m = make("line", hetero=1.0); Xd = G.demean_by(X, m["q"]); lad = G.ladder(m, "a"); mL = {k: v[lad] for k, v in m.items()}
r1 = G.input_dependence(Xd[lad], mL, "a", nboot=100)
print("D1 global", r0, "\nD1 hetero", r1)
ok &= check("D1 ~ 1 for one global direction (CI upper > .95)", r0["D1_ci"][1] > .95)
ok &= check("D1 < .8 for per-question rotations of sd 1 rad", r1["D1_ci"][1] < .8)

for same in (True, False):
    X, m = make("line", same_dir=same); Xd = G.demean_by(X, m["q"])
    m["Dpos"] = np.isin(m["trait"], TR) & (m["level"] >= 1); m["Dneg"] = np.isin(m["trait"], TR) & (m["level"] == -3)
    r = G.separation(Xd, m, TR, nsplit=20)["a|b"]
    print("separation (placebo anchor)", "same_dir" if same else "orthogonal", round(r, 3))
    ok &= check("separation r ~ 1 if same direction" if same else "separation r ~ 0 if orthogonal (frame offset cancelled)",
                (r > .9) if same else (abs(r) < .15))

try:
    from lm import _npn
    for k in [k for k in sys.modules if k == "common" or k.startswith("common.")]: del sys.modules[k]   # IDEEA has its own 'common'
    sys.path.insert(0, os.path.expanduser("~/ext/IDEEA"))
    from methods.ideea_caa import _select_nearest_pos_neg
    x = torch.randn(7, 32, dtype=torch.float64); pc = torch.randn(4, 32, dtype=torch.float64); nc = torch.randn(4, 32, dtype=torch.float64)
    ok &= check("IDEEA nearest_pos_neg identical to ~/ext/IDEEA", bool(torch.allclose(_select_nearest_pos_neg(x, None, None, pc, nc, None), _npn(x, pc, nc))))
except Exception as e:
    print("IDEEA import failed:", repr(e)); ok = False
print("ALL PASS" if ok else "SOME FAILED")
