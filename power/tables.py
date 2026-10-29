"""Min-SNR-for-80%-power tables + heatmaps. Reads ~/scratch/power/aggA.json, aggP.json, place.json, nullcheck results."""
import os, json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import core as C
R = C.R; RES = os.path.expanduser("~/steer_math/results")
A = json.load(open(f"{R}/aggA.json")); P = json.load(open(f"{R}/aggP.json")); PL = json.load(open(f"{R}/place.json"))
NS = [20, 50, 100, 300, 1000]; SNP = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def minsnr(pw, snrs, target=.8):
    pw = np.array(pw, float)
    for k in range(len(snrs)):
        if np.isfinite(pw[k]) and pw[k] >= target and all(np.nan_to_num(pw[k:], nan=0) >= target - .1):
            if k == 0: return f"≤{snrs[0]:g}"
            a, b = np.log10(snrs[k - 1]), np.log10(snrs[k]); p0, p1 = pw[k - 1], pw[k]
            x = a + (target - p0) / max(p1 - p0, 1e-9) * (b - a)
            return f"{10 ** x:.2g}"
    return f">{snrs[-1]:g}"


rows = []
for bg, mode in [("tqa13_5", "iid"), ("tqa13_5", "q"), ("syc", "q"), ("opt", "q"), ("syc", "iid"), ("wk", "iid"), ("tqa21_6", "iid")]:
    for L in [5, 7]:
        for sh in ["arc60", "arc90"]:
            r = [bg, mode, L, sh]
            for st in ["turn", "offline", "quad"]:
                r.append(" / ".join(minsnr(A[f"{bg}_rand_L{L}_n{n}_{mode}"][st][C.SHAPES.index(sh)], C.SNRS) for n in NS))
            r.append(" / ".join(minsnr(A[f"{bg}_rand_L{L}_n{n}_{mode}"]["ord1"][C.SHAPES.index("arc0")], C.SNRS) for n in NS))
            rows.append(r)
with open(f"{R}/table_A.md", "w") as f:
    f.write("| bg | mode | L | shape | turn | offline | quad⊥ | ord1 (arc0, order) |\n|---|---|---|---|---|---|---|---|\n")
    for r in rows: f.write("| " + " | ".join(map(str, r)) + " |\n")
rowsP = []
for bg, modes in [("tqa13_5", ["iid", "qraw", "qdm"]), ("syc", ["iid", "qraw", "qdm"]), ("wk", ["iid"])]:
    for mode in modes:
        for L in [5, 7]:
            for sh in ["arc60", "arc90"]:
                cells = [P[f"{bg}_L{L}_n{n}_{mode}"] for n in [20, 50, 100, 300]]
                j = list(cells[0]["shapes"]).index(sh)
                rowsP.append([bg, mode, L, sh, " / ".join(minsnr(c["frot"][j], SNP) for c in cells)])
with open(f"{R}/table_P.md", "w") as f:
    f.write("| bg | mode | L | shape | frot (n=20/50/100/300) |\n|---|---|---|---|---|\n")
    for r in rowsP: f.write("| " + " | ".join(map(str, r)) + " |\n")

# heatmaps: power over (n, SNR) for arc90 / arc60 L7, per statistic, two backgrounds
marks = {"tqa13_5": [("TQA (L=2!)", 1200, PL["tqa_top24"]["snr_med"])],
         "wk": [("weekday last tok", 78, PL["wk_last"]["snr_L12"])],
         "syc": [("persona L=6 ladder\n(30q×20)", 600, 0.47 * PL["sycophantic"]["layers"]["14"]["snr"])]}
for sh in ["arc60", "arc90"]:
    fig, axs = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    for i, (bg, mode) in enumerate([("tqa13_5", "iid"), ("wk", "iid"), ("syc", "q")]):
        for j, (st, s2) in enumerate([("turn", sh), ("offline", sh), ("quad", sh), ("ord1", "arc0")]):
            M = np.array([A[f"{bg}_rand_L7_n{n}_{mode}"][st][C.SHAPES.index(s2)] for n in NS])
            ax = axs[i, j]; im = ax.imshow(M, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for a in range(M.shape[0]):
                for b in range(M.shape[1]):
                    ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[a, b] > .6 else "#333")
            ax.contour(M, levels=[.8], colors="#c0392b", linewidths=1.5)
            ax.set_xticks(range(len(C.SNRS))); ax.set_xticklabels([f"{s:g}" for s in C.SNRS], fontsize=7)
            ax.set_yticks(range(len(NS))); ax.set_yticklabels(NS, fontsize=8)
            ax.set_title(f"{st} ({'order, straight plant' if st == 'ord1' else sh}) | bg {bg} {mode}", fontsize=9)
            if j == 0: ax.set_ylabel("n per level")
            if i == 2: ax.set_xlabel("class SNR")
            for lab, n, s in marks.get(bg, []):
                x = np.interp(np.log10(s), np.log10(C.SNRS), np.arange(len(C.SNRS))); y = np.interp(np.log10(n), np.log10(NS), np.arange(len(NS)))
                ax.plot(x, y, "o", ms=9, mfc="none", mec="#e67e22", mew=2); ax.annotate(lab, (x, y), fontsize=6, color="#e67e22", xytext=(4, 4), textcoords="offset points")
    fig.suptitle(f"Oracle power, L = 7, random plane. Planted shape {sh} (ord1: straight line, exact order-relabel test). Red = 80% contour.", fontsize=11)
    fig.colorbar(im, ax=axs, shrink=.6, label="power")
    fig.savefig(f"{RES}/power_curve_heat_{sh}.png", dpi=110); plt.close(fig)
print(open(f"{R}/table_A.md").read()); print(open(f"{R}/table_P.md").read())
