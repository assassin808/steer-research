"""Figure: reliable cyclic structure (Fourier f1 share, p1 vs shuffle) and class SNR per layer, vs the TruthfulQA head SNR."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.expanduser("~/scratch/weekday"); OUT = os.path.expanduser("~/steer_math/results")
runs = [("gemma_weekday_z", "weekday, last tok, z", "tab:blue", "-"), ("gemma_weekday_y", "weekday, last tok, answer", "tab:blue", ":"),
        ("gemma_month_z", "month, last tok, z", "tab:orange", "-"), ("gemma_month_y", "month, last tok, answer", "tab:orange", ":"),
        ("gemmaztok_weekday_z", "weekday, z token", "tab:green", "-"), ("gemmaztok_month_z", "month, z token", "tab:red", "-")]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for tag, lab, c, ls in runs:
    fn = f"{R}/fourier_{tag}_{tag.split('_')[-1]}.json".replace(f"_{tag.split('_')[-1]}_{tag.split('_')[-1]}", f"_{tag.split('_')[-1]}")
    fn = f"{R}/fourier_{tag}.json"
    if not os.path.exists(fn):
        continue
    d = json.load(open(fn)); L = [r["layer"] for r in d]
    ax[0].semilogy(L, [max(r["snr"], 1e-5) for r in d], ls, color=c, label=lab)
    ax[1].plot(L, [r["f1"] for r in d], ls, color=c, label=lab)
    ax[2].semilogy(L, [max(r["p1_ratio"], 1e-2) for r in d], ls, color=c, label=lab)
ax[0].axhspan(0.0038, 0.0182, color="0.85", label="TruthfulQA top-24 heads (range)"); ax[0].set_title("class SNR = centroid power / C / within-class var")
ax[1].axhline(2 / 6, color="tab:blue", lw=.6, ls="--"); ax[1].axhline(2 / 11, color="tab:orange", lw=.6, ls="--")
ax[1].set_title("first circular harmonic share of reliable centroid Gram\n(dashed: pure simplex 2/(C-1); circle = 1)")
ax[2].axhline(1, color="k", lw=.6); ax[2].set_title("f1 power / max of 20 label-shuffle nulls")
for a in ax:
    a.set_xlabel("layer"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/weekday_calib_fourier.png", dpi=110)
